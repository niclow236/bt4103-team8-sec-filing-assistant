"""What the app keeps between one rerun and the next (#36).

Streamlit runs a page's script again on every widget change, so whatever is
worth keeping has to be held outside it. Everything a page keeps is here, so
no page holds a cache of its own:

- for the process, with ``st.cache_resource``: the indexes and the embedding
  model (``_indexes``), each built configuration (``load_stack``), and the
  local passages shown by the Browse page (``corpus_passages``), and with
  ``st.cache_data`` the runs under ``results/`` (``measured``,
  ``result_runs``) and one configuration's saved questions
  (``question_outcomes``);
- from ``.env``, read on every rerun: the providers a page can offer, the one
  it opens on, and what any of them lacks (``answer_models``);
- for the process, per configuration: the answer a repeated question was
  given (``Remembered``);
- for one browser session: the answer a page is showing and the request it
  answers (``Request``, ``keep``, ``kept``, ``has_kept``);
- the stopwatch behind the line that says how long an answer took.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from threading import Event, Lock
from time import perf_counter
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple
from urllib.parse import urlsplit

import streamlit as st

from src.pipeline.chunk import iter_chunks
from src.rag.constants import DEFAULT_PROVIDER, PROVIDERS
from src.rag.generate import ProviderUnavailable, check_provider, config_from_env
from src.stack import DEFAULT_STACK, RESULTS_ROOT, build_stack, measured_runs

if TYPE_CHECKING:
    from collections.abc import Callable

    from src.rag.records import Answer
    from src.retrieval.records import Query, RetrievedPassage
    from src.stack import Stack

# What a caller passes with a question that does not change its answer:
# ``parsed`` is read from the question; the callbacks only show evidence
# and the answer arriving.
NOT_PART_OF_THE_ANSWER = ("parsed", "on_token")

# How many answers one configuration keeps. Each holds the passages it was
# written from, sixteen of them, so the memory of a process left running
# would otherwise grow with every new question. Past this many the answer
# asked for longest ago is dropped, and asking it again asks the model again.
ANSWERS_KEPT = 128

from src.evaluation.benchmark import DEFAULT_GENERATED_QUESTIONS_PATH, DEFAULT_QUESTIONS_PATH
from src.evaluation.metrics import first_relevant_rank, rank_from_reciprocal
from src.evaluation.report import (benchmark_kind, question_rows, read_result_rows, read_runs,
                                   saved_score)

# Where a question's text is looked up when the rows a run saved for it kept
# none, as an embedding ablation's do: the team's questions and, where it has
# been generated on this machine, the XBRL benchmark.
BENCHMARK_FILES = (DEFAULT_QUESTIONS_PATH, DEFAULT_GENERATED_QUESTIONS_PATH)


@st.cache_data(show_spinner=False)
def result_runs(results_root: Path = RESULTS_ROOT) -> list[dict[str, Any]]:
    """Read saved evaluation runs, including older files, without rerunning them."""
    return read_runs(results_root)


class QuestionOutcome(NamedTuple):
    """How one configuration of a saved run did on one question (#169)."""

    question: str                     # its text, where the saved row kept it
    benchmark: str                    # as ``benchmark_kind`` classifies the row
    scored: bool = False              # whether it was scored on retrieval at all
    rank: int | None = None           # of its first right passage, None if none was retrieved
    correctness: float | None = None  # what its answer was scored, where it was
    # Whether its text was looked up in the benchmark as it stands now, which
    # may word it differently from what the run asked.
    current_text: bool = False
    # Its benchmark labels, as its retrieval row saved them, for the breakdown (#170).
    question_type: str | None = None
    difficulty: str | None = None


class WorkedExample(NamedTuple):
    """One correct answer the Results page can replay from a saved run (#167)."""

    question_id: str
    config_id: str
    question: str
    answer: str
    expected_answer: str
    passage: dict[str, Any]
    citation_marker: int | None
    other_citation_markers: tuple[int, ...]
    fact_checks: tuple[dict[str, Any], ...]
    correctness_reason: str


def _edgar_url(value: Any) -> bool:
    """Whether ``value`` is an HTTPS page served by the SEC."""
    if not isinstance(value, str):
        return False
    try:
        parsed = urlsplit(value)
    except ValueError:
        return False
    host = (parsed.hostname or "").lower()
    return parsed.scheme == "https" and (host == "sec.gov" or host.endswith(".sec.gov"))


def _example_passage(
        answer: dict[str, Any]) -> tuple[dict[str, Any], int | None, tuple[int, ...]] | None:
    """A saved EDGAR passage, its marker, and the answer's other resolved markers."""
    passages = answer.get("passages")
    if not isinstance(passages, list):
        return None
    usable = [passage for passage in passages
              if isinstance(passage, dict) and isinstance(passage.get("text"), str)
              and passage["text"].strip() and _edgar_url(passage.get("url"))]
    if not usable:
        return None
    by_chunk = {passage.get("chunk_id"): passage for passage in usable
                if isinstance(passage.get("chunk_id"), str)}
    citations = answer.get("citations")
    resolved: list[tuple[int, str]] = []
    if isinstance(citations, list):
        for citation in citations:
            if not isinstance(citation, dict) or citation.get("resolved") is not True:
                continue
            marker, chunk_id = citation.get("marker"), citation.get("chunk_id")
            if (isinstance(marker, int) and not isinstance(marker, bool) and marker > 0
                    and isinstance(chunk_id, str)):
                resolved.append((marker, chunk_id))
    for marker, chunk_id in resolved:
        if chunk_id in by_chunk:
            others = tuple(dict.fromkeys(other for other, _ in resolved if other != marker))
            return by_chunk[chunk_id], marker, others
    return usable[0], None, ()


def _examples_from(rows: list[dict[str, Any]], config_id: str) -> tuple[WorkedExample, ...]:
    """Displayable worked examples in saved ``rows``, ordered by verification value."""
    examples: list[WorkedExample] = []
    for row in rows:
        quality, answer = row.get("quality"), row.get("answer")
        if not isinstance(quality, dict) or not isinstance(answer, dict):
            continue
        if saved_score(quality.get("correctness")) != 1.0 or answer.get("abstained") is not False:
            continue
        question_id = row.get("question_id")
        question, text = answer.get("question"), answer.get("text")
        expected = row.get("expected_answer")
        selected = _example_passage(answer)
        if (not isinstance(question_id, str) or not question_id
                or not isinstance(question, str) or not question.strip()
                or not isinstance(text, str) or not text.strip()
                or not isinstance(expected, str) or not expected.strip()
                or selected is None):
            continue
        passage, citation_marker, other_citation_markers = selected
        verification = answer.get("verification")
        checks = verification.get("checks", []) if isinstance(verification, dict) else []
        fact_checks = tuple(dict(check) for check in checks
                            if isinstance(check, dict) and check.get("kind") == "fact")
        reason = quality.get("correctness_reason")
        matrix_config = row.get("matrix_config")
        saved_config = matrix_config.get("id") if isinstance(matrix_config, dict) else None
        examples.append(WorkedExample(
            question_id, str(saved_config or config_id), question, text, expected,
            dict(passage), citation_marker, other_citation_markers, fact_checks,
            reason if isinstance(reason, str) else ""))
    return tuple(sorted(examples, key=lambda example: (
        not any(check.get("status") == "supported" for check in example.fact_checks),
        not bool(example.fact_checks),
    )))


@st.cache_data(show_spinner=False)
def worked_examples(run_dir: str, config_id: str = DEFAULT_STACK) -> tuple[WorkedExample, ...]:
    """Correct, displayable examples from a run's best saved answer rows.

    Nothing is reconstructed from the current corpus or benchmark: every word,
    passage, score and verification check comes from the selected run. A usable
    run-local example wins; otherwise C6 supplies verification when available and
    the default C4 rows are the fallback.
    """
    root = Path(run_dir)
    curated = _examples_from(read_result_rows(root / "worked-example.jsonl"), config_id)
    if curated:
        return curated
    for saved_config in dict.fromkeys(("C6", config_id)):
        folder = root / saved_config
        rows = (read_result_rows(folder / "answers.jsonl")
                or read_result_rows(folder / "report.json"))
        examples = _examples_from(rows, saved_config)
        if examples:
            return examples
    return ()


def _label(row: dict[str, Any], key: str) -> str | None:
    """A saved row's ``question_type`` or ``difficulty``, or None if it kept no text there."""
    value = row.get(key)
    return value if isinstance(value, str) and value else None


def _chunk_ids(value: Any) -> list[str] | None:
    """``value`` if it is a saved list of chunk ids, and None if it is anything else."""
    if isinstance(value, list) and all(isinstance(chunk_id, str) for chunk_id in value):
        return value
    return None


def _first_right_rank(row: dict[str, Any]) -> tuple[bool, int | None]:
    """Whether ``row`` was scored on retrieval, and where its first right passage ranked.

    The rank is None when no right passage was retrieved at all, and a row
    whose retrieved passages cannot be read is left unscored. A row that
    does not name its right passages, as ``src.evaluation.run`` saves one, is
    ranked by its reciprocal rank, which is None for an unscored question and
    0 when no right passage made the cutoff it was scored at.
    """
    supporting = _chunk_ids(row.get("scoring_supporting_chunk_ids"))
    if supporting is None:
        reciprocal = row.get("mrr")
        if not isinstance(reciprocal, (int, float)):
            return False, None
        return True, rank_from_reciprocal(reciprocal)
    retrieved = _chunk_ids(row.get("retrieved_chunk_ids"))
    if not supporting or retrieved is None:
        return False, None
    return True, first_relevant_rank(retrieved, supporting)


def _question_text(row: dict[str, Any]) -> str:
    """The question a saved row asked: a retrieval row keeps it in its query, an answer row in its answer."""
    for field, key in (("query", "text"), ("answer", "question")):
        held = row.get(field)
        if isinstance(held, dict):
            return str(held.get(key, ""))
    return ""


def _benchmark_texts(paths: tuple[Path, ...]) -> dict[str, str]:
    """Each question's text, by id, as the benchmark files at ``paths`` word it."""
    return {str(row.get("question_id")): str(row.get("question") or "")
            for path in paths for row in read_result_rows(path)}


@st.cache_data(show_spinner=False)
def question_outcomes(run_dir: str, config_id: str) -> dict[str, QuestionOutcome]:
    """How one configuration of a saved run did on each question, by question id (#169).

    Read on demand for the two configurations a comparison names, rather than
    held by ``result_runs`` for every configuration of every run.
    """
    outcomes: dict[str, QuestionOutcome] = {}
    folder = Path(run_dir) / config_id
    for row in question_rows(folder):
        question_id = row.get("question_id")
        if not question_id:
            continue
        scored, rank = _first_right_rank(row)
        outcomes[str(question_id)] = QuestionOutcome(
            _question_text(row), benchmark_kind(row), scored, rank,
            question_type=_label(row, "question_type"), difficulty=_label(row, "difficulty"))
    answers = (read_result_rows(folder / "answers.jsonl")
               or read_result_rows(folder / "report.json"))
    for row in answers:
        question_id = row.get("question_id")
        quality = row.get("quality")
        if not question_id or not isinstance(quality, dict):
            continue
        outcome = outcomes.get(str(question_id)) or QuestionOutcome(
            _question_text(row), benchmark_kind(row))
        correctness = quality.get("correctness")
        # Only what the Answer quality table counts is a score: text there is a
        # malformed row, and a NaN (which json reads) would count as unchanged
        # against any score.
        outcomes[str(question_id)] = outcome._replace(correctness=saved_score(correctness))
    # The benchmark the run froze, where it froze one, is what it asked.
    frozen = _benchmark_texts((Path(run_dir) / "benchmark.jsonl",))
    current = _benchmark_texts(BENCHMARK_FILES)
    for question_id, outcome in outcomes.items():
        if outcome.question:
            continue
        if frozen.get(question_id):
            outcomes[question_id] = outcome._replace(question=frozen[question_id])
        elif current.get(question_id):
            outcomes[question_id] = outcome._replace(question=current[question_id],
                                                     current_text=True)
    return outcomes


@st.cache_resource(show_spinner="Reading the local filing corpus…", validate=bool)
def corpus_passages() -> tuple[dict[str, Any], ...]:
    """Read the processed passages once for the process (#40).

    The Browse page needs the passage text and filing metadata, not a search
    index and not a model. ``iter_chunks`` is the pipeline's common seam for
    joining those two records, so the page sees exactly what retrieval can
    see. The corpus is large and read-only while the app is running; a
    resource cache shares this tuple between reruns and browser sessions
    instead of copying all passage text into each session. Restart the app (or
    clear Streamlit's cache) after rebuilding ``data/processed/``.

    Callers must treat the dictionaries as read-only.
    """
    return tuple(iter_chunks())


@st.cache_resource
def _indexes() -> dict:
    """The BM25 and dense retrievers every configuration shares, once per process.

    Without it, switching from C4 to C3 -- the same hybrid retriever -- read
    both indexes again and loaded a second copy of the embedding model.
    """
    return {}


@st.cache_resource(show_spinner="Checking the local filing indexes…")
def load_stack(config_id: str, provider: str | None = None) -> Remembered:
    """Assemble a named configuration once per Streamlit process (#43).

    The same call the evaluation command makes, so what the demo shows is the
    system the numbers in ``results/`` describe, rather than a fourth stack
    assembled here. Cached on the id because building one reads the indexes,
    and because the one built keeps the embedding model that the dense and
    hybrid rows load on their first search.

    ``provider`` is what writes the answers, and None means the one ``.env``
    names. It is cached on too, so a page that offers both builds each once.
    They share the indexes, and each remembers its own answers, so an answer
    one provider wrote is never shown as the other's.

    The chat model is not built here. ``generate`` builds it when an answer
    first needs one, because not every answer does: built here, a figure the
    facts store could look up was refused for the key of a provider that
    would never have been asked. What a provider lacks is said before a
    question is asked instead, by ``answer_models``.
    """
    # Remembered, so a repeated demo question is answered once (#36).
    return Remembered(
        build_stack(config_id, provider=provider, parts=_indexes(), defer_llm=True))


@dataclass(frozen=True)
class AnswerModels:
    """What can write an answer, for a page to offer the choice."""

    # Provider to the model it would answer with, in the order they are offered.
    models: dict[str, str]
    default: str                   # the provider ``.env`` names, which a page opens on
    # Provider to why it could not write an answer as things stand, in a few
    # words: a missing key, a setting it cannot read. Absent where it could.
    unready: dict[str, str]
    problem: str | None = None     # why ``.env``'s own choice could not be used, if it could not


def answer_models() -> AnswerModels:
    """The providers a page can offer, read from ``.env`` as ``build_stack`` reads it.

    The provider used to be ``.env``'s alone to choose: the app opened on
    ``LLM_PROVIDER``, which unset means the local model, and nothing on the
    page could change it. A laptop with a Mistral key in ``.env`` and no
    ``LLM_PROVIDER`` line waited minutes for the local model with nothing on
    the page to say why.

    A mistyped ``LLM_PROVIDER`` is not an error here. A page can still be
    asked with the provider picked by hand, so it opens on the default one and
    ``problem`` says what was wrong with the setting.

    Nor is a ``.env`` that cannot be read at all. PowerShell 5's ``>`` and
    ``Out-File`` save a file as UTF-16, which python-dotenv refuses, and this
    is the first place a page reads ``.env``: raised from here, the refusal
    was a traceback where the sidebar should be, on every load. The page
    opens on each provider's default model and ``problem`` says the file
    could not be read. An Ask still fails on it, as an error on the page.

    ``unready`` is what each provider's own check refuses, asked of its
    settings alone (``check_provider``): no client is built and nothing is
    sent. It holds what is wrong and not the check's advice, which is written
    for a command: a page says for itself what to do. A provider that is not
    ready can still be picked, since a question the facts store answers asks
    no model, on a row with the metadata filter (#156).
    """
    models, unready = {}, {}
    try:
        for provider in PROVIDERS:
            config = config_from_env(provider=provider)
            models[provider] = config.model
            try:
                check_provider(config)
            except ProviderUnavailable as error:
                unready[provider] = error.reason
            except ValueError as error:
                unready[provider] = str(error)
    except (ValueError, OSError) as error:
        # Only reading .env raises here: a provider named in the code is not
        # refused, and what its settings lack was caught above. UTF-16 with
        # its byte order mark is a UnicodeDecodeError, and without one an
        # OSError on Windows, where a variable cannot hold the null bytes.
        models = {provider: config_from_env(provider=provider, environ={}).model
                  for provider in PROVIDERS}
        return AnswerModels(models, DEFAULT_PROVIDER, {}, (
            f"Could not read .env: {error}. If PowerShell 5's `>` or `Out-File` saved it, "
            "it is UTF-16: save it as UTF-8."))
    try:
        return AnswerModels(models, config_from_env().provider, unready)
    except ValueError as error:
        return AnswerModels(models, DEFAULT_PROVIDER, unready, str(error))


@st.cache_data(show_spinner=False)
def measured() -> list[tuple[str, str]]:
    """Each configuration measured under ``results/``, as (id, label).

    Read from the result files rather than listed here, so a configuration is
    offered because a run measured it. Several runs can measure the same row;
    the most recent run's label is the one shown.
    """
    seen: dict[str, str] = {}
    for run in measured_runs():
        seen.setdefault(run.config_id, run.label)
    return list(seen.items())


class Request(NamedTuple):
    """What an answer was asked with: the question, its filters, and what answered it.

    Everything that would change the answer, so that one kept for a request
    is never shown for another. A page used to build this as a bare tuple, in
    an order it chose itself. A field put in the wrong place there made a
    request that never matched, with no error to show for it, and each page
    to come builds one of its own. ``of`` takes the fields from where they
    already are.

    A tuple all the same, and compared as one. An answer kept in the session
    outlives this module when Streamlit loads it again after an edit, and a
    dataclass loaded again is a new class, equal to nothing kept before it.
    """

    question: str
    tickers: tuple[str, ...]
    fiscal_years: tuple[int, ...]
    items: tuple[str, ...]
    config_id: str
    provider: str

    @classmethod
    def of(cls, query: Query, config_id: str, provider: str) -> Request:
        """``query`` as the sidebar left it, asked of one configuration and one provider."""
        return cls(query.text, query.tickers, query.fiscal_years, query.items,
                   config_id, provider)


@dataclass(frozen=True)
class Kept:
    """The answer a page is showing: what was asked, the answer, and how long it took."""

    request: Request
    answer: Answer
    seconds: float


def _key(page: str) -> str:
    """Where the session holds what ``page`` is showing.

    One entry per page, under the page's own name, so the Ask page and a page
    that asks two configurations do not show each other's answers.
    """
    return f"{page}:kept"


def keep(page: str, request: Request, answer: Answer, seconds: float) -> None:
    """Hold ``answer`` for this browser session as what ``page`` is showing."""
    st.session_state[_key(page)] = Kept(request, answer, seconds)


def kept(page: str, request: Request) -> Kept | None:
    """What ``page`` is showing, if it answers ``request``, and None if not.

    The request is the question with everything it was asked with, so an
    answer is never shown under a question or filters it was not given for.
    """
    held = st.session_state.get(_key(page))
    return held if held is not None and held.request == request else None


def has_kept(page: str) -> bool:
    """Whether ``page`` holds an answer at all, to whatever request."""
    return _key(page) in st.session_state


class ComparisonSide(NamedTuple):
    """One side of a comparison, including its independent failure (#39)."""

    answer: Answer | None
    seconds: float
    error: str | None = None


class KeptComparison(NamedTuple):
    """The saved pair: both requests and the side each answered."""

    requests: tuple[Request, Request]
    sides: tuple[ComparisonSide, ComparisonSide]


def keep_comparison(requests: tuple[Request, Request],
                    sides: tuple[ComparisonSide, ComparisonSide]) -> None:
    """Hold both sides for this browser session as what Compare is showing."""
    st.session_state[_key("compare")] = KeptComparison(requests, sides)


def kept_comparison(requests: tuple[Request, Request]) -> tuple[ComparisonSide, ComparisonSide] | None:
    """Keep both sides together; changing either request hides the whole pair."""
    held = st.session_state.get(_key("compare"))
    return held.sides if held is not None and held.requests == requests else None


class Remembered:
    """A built configuration that gives a repeated question the answer it gave before.

    The app asks through one of these in place of the ``Stack`` it wraps.
    ``load_stack`` makes one per configuration and provider, and
    ``st.cache_resource`` keeps it for the process, so the answers are shared
    by every browser session and last until the app restarts. An answer is
    kept by the question and by what it was asked with: the ``Query``, which
    carries the sidebar's companies, years and Items, and any setting the
    caller overrode.

    Only a finished answer is kept. A failure is not, so a question that met a
    busy provider is asked again, and neither is an answer the model left
    malformed or cut short (``parse_error``, ``truncated``): it is shown for
    what it is, and the next Ask gets another try and not the same broken
    text until the app restarts. At most ``limit`` answers are kept, the ones
    asked for most recently.

    The sessions run on threads of one process, so two can ask the same
    question at once. The second waits for the first and is given its answer:
    asked twice, a slow local model would do minutes of work twice and could
    give the two sessions different answers. Different questions do not wait
    for each other. If the first fails, or its answer is not kept, a session
    that was waiting asks in its turn.

    The answers are held in a dict and not by ``st.cache_data``, which #36
    names. A function cached that way replays the page elements drawn while
    it ran, and raises ``CacheReplayClosureError`` on its second call when
    they were drawn on a placeholder made outside it. That is how an answer
    is streamed to the page, through the ``on_token`` a caller passes here.
    """

    def __init__(self, stack: Stack, limit: int = ANSWERS_KEPT) -> None:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        self.stack = stack
        self.limit = limit
        self._answers: OrderedDict[tuple[Any, ...], Answer] = OrderedDict()
        # The questions being answered now, each with the event its asker sets
        # when it is done. Both dicts are only touched while holding the lock,
        # and the lock is never held while the stack answers.
        self._being_answered: dict[tuple[Any, ...], Event] = {}
        self._lock = Lock()

    def answer(self, question: str, *,
               on_retrieved: Callable[[tuple[RetrievedPassage, ...]], None] | None = None,
               **overrides: Any) -> Answer:
        """The answer to ``question``, asked of the stack only the first time.

        ``overrides`` go to ``Stack.answer`` with that first request. A
        remembered answer comes back at once, and so does one another session
        was already being given, so nothing is streamed to ``on_token`` for
        either. The caller's ``on_retrieved`` is still notified of cached
        evidence, outside the lock, without repeating the search.
        """
        asked_with = tuple(sorted(
            (name, value) for name, value in overrides.items()
            if name not in NOT_PART_OF_THE_ANSWER
        ))
        key = (question, asked_with)
        while True:
            remembered = None
            with self._lock:
                if key in self._answers:
                    self._answers.move_to_end(key)
                    remembered = self._answers[key]
                else:
                    done = self._being_answered.get(key)
                    if done is None:
                        done = self._being_answered[key] = Event()
                        break
            if remembered is not None:
                if on_retrieved is not None:
                    on_retrieved(remembered.passages)
                return remembered
            # Another session is asking this now. Wait for it, then look again:
            # its answer is there, or it failed and this one asks in its turn.
            done.wait()
        try:
            answer = self.stack.answer(question, on_retrieved=on_retrieved, **overrides)
            if _finished(answer):
                with self._lock:
                    self._answers[key] = answer
                    while len(self._answers) > self.limit:
                        self._answers.popitem(last=False)
            return answer
        finally:
            # Whatever happened, the question is no longer being answered and
            # the sessions waiting for it are woken.
            with self._lock:
                del self._being_answered[key]
            done.set()


def _finished(answer: Answer) -> bool:
    """Whether the model wrote the whole of a well-formed answer."""
    return answer.parse_error is None and not answer.truncated


class Stopwatch:
    """How long a ``with`` block took: ``seconds``, once the block has ended."""

    seconds: float | None = None

    def __enter__(self) -> Stopwatch:
        self._started = perf_counter()
        return self

    def __exit__(self, *error: object) -> None:
        self.seconds = perf_counter() - self._started
