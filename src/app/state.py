"""What the app keeps between one rerun and the next (#36).

Streamlit runs a page's script again on every widget change, so whatever is
worth keeping has to be held outside it. Everything a page keeps is here, so
no page holds a cache of its own:

- for the process, with ``st.cache_resource``: the indexes and the embedding
  model (``_indexes``) and each built configuration (``load_stack``), and with
  ``st.cache_data`` the runs under ``results/`` (``measured``);
- from ``.env``, read on every rerun: the providers a page can offer, the one
  it opens on, and what any of them lacks (``answer_models``);
- for the process, per configuration: the answer a repeated question was
  given (``Remembered``);
- for one browser session: the answer a page is showing and the request it
  answers (``keep``, ``kept``, ``has_kept``);
- the stopwatch behind the line that says how long an answer took.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from threading import Event, Lock
from time import perf_counter
from typing import TYPE_CHECKING, Any

import streamlit as st

from src.rag.constants import DEFAULT_PROVIDER, PROVIDERS
from src.rag.generate import ProviderUnavailable, check_provider, config_from_env
from src.stack import build_stack, measured_runs

if TYPE_CHECKING:
    from src.rag.records import Answer
    from src.stack import Stack

# What a caller passes with a question that does not change its answer:
# ``parsed`` is read from the question, and ``on_token`` only shows the
# answer arriving.
NOT_PART_OF_THE_ANSWER = ("parsed", "on_token")

# How many answers one configuration keeps. Each holds the passages it was
# written from, sixteen of them, so the memory of a process left running
# would otherwise grow with every new question. Past this many the answer
# asked for longest ago is dropped, and asking it again asks the model again.
ANSWERS_KEPT = 128


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
    assembled here. Cached on the id because building one loads the indexes
    and, for the dense and hybrid rows, the embedding model.

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
    no model.
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


@dataclass(frozen=True)
class Kept:
    """The answer a page is showing: what was asked, the answer, and how long it took."""

    request: tuple[Any, ...]
    answer: Answer
    seconds: float


def keep(page: str, request: tuple[Any, ...], answer: Answer, seconds: float) -> None:
    """Hold ``answer`` for this browser session as what ``page`` is showing.

    One entry per page, under the page's own name, so the Ask page and a page
    that asks two configurations do not show each other's answers.
    """
    st.session_state[f"{page}:kept"] = Kept(request, answer, seconds)


def kept(page: str, request: tuple[Any, ...]) -> Kept | None:
    """What ``page`` is showing, if it answers ``request``, and None if not.

    The request is the question with everything it was asked with, so an
    answer is never shown under a question or filters it was not given for.
    """
    held = st.session_state.get(f"{page}:kept")
    return held if held is not None and held.request == request else None


def has_kept(page: str) -> bool:
    """Whether ``page`` holds an answer at all, to whatever request."""
    return f"{page}:kept" in st.session_state


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

    def answer(self, question: str, **overrides: Any) -> Answer:
        """The answer to ``question``, asked of the stack only the first time.

        ``overrides`` go to ``Stack.answer`` with that first request. A
        remembered answer comes back at once, and so does one another session
        was already being given, so nothing is streamed to ``on_token`` for
        either.
        """
        asked_with = tuple(sorted(
            (name, value) for name, value in overrides.items()
            if name not in NOT_PART_OF_THE_ANSWER
        ))
        key = (question, asked_with)
        while True:
            with self._lock:
                if key in self._answers:
                    self._answers.move_to_end(key)
                    return self._answers[key]
                done = self._being_answered.get(key)
                if done is None:
                    done = self._being_answered[key] = Event()
                    break
            # Another session is asking this now. Wait for it, then look again:
            # its answer is there, or it failed and this one asks in its turn.
            done.wait()
        try:
            answer = self.stack.answer(question, **overrides)
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
