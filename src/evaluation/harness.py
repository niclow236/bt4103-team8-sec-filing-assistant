"""Run grounded QA with gold-answer quality and query resource metrics."""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from math import isfinite
from time import perf_counter, sleep
from typing import Any, TypeVar

from src.rag import (
    ProviderBusy,
    ProviderUnavailable,
    answer_question,
    config_from_env,
    parse_question,
    verify_answer,
    worst_check,
)
from src.rag.answer import REFUSALS
from src.rag.records import Generation, GenerationConfig
from src.retrieval.base import Retriever
from src.retrieval.constants import FINAL_K
from .records import BenchmarkQuestion, UNANSWERABLE
from .quality import (
    JudgeInvalid, LLMJudge, TokenPrices, citation_scores, quality_summary, query_cost, resource_summary,
)

logger = logging.getLogger(__name__)

# What a caller of answer_with_retries asks for: an Answer in the harness, and
# whatever a measurement script's own ask() returns.
Asked = TypeVar("Asked")

# How long to wait before asking a question again after a failure that asking
# again may fix (``ProviderBusy``): a rate limit, a server error, a dropped
# connection. One wait per retry, so a question is asked at most three times.
# The second is a full minute because the free plan's rate limits are per
# minute. A provider that asks for longer, with Retry-After, is given that.
# Anything else, and the third failure, stops the run.
RETRY_WAITS_S = (10, 60)
# The longest wait a run sits through before asking again. A provider that
# asks for more has hit a limit a run cannot wait out, so the run stops then,
# keeping the answers it has.
MAX_RETRY_WAIT_S = 300


class _TimedRetriever:
    """Time outer searches once, including decomposition and facts evidence.

    Forward optional index capabilities, timing candidate diagnostics too.
    Hybrid's inner searches are already contained in its outer search.
    """

    def __init__(self, retriever):
        self.retriever = retriever
        self.latency_ms = 0.0

    def _call(self, method, *args, **kwargs):
        started = perf_counter()
        try:
            return method(*args, **kwargs)
        finally:
            self.latency_ms += (perf_counter() - started) * 1000

    def search(self, query, k=None):
        return self._call(self.retriever.search, query, k=k)

    def __getattr__(self, name):
        value = getattr(self.retriever, name)
        if name == "has_candidates":
            return lambda *args, **kwargs: self._call(value, *args, **kwargs)
        return value


class _Stopped:
    """What a run that stopped part-way carries: ``report``, the report of the
    questions answered before it, in which ``stopped`` names the question it
    stopped at and why, and ``results`` holds every row before it."""

    def __init__(self, report: dict[str, Any]) -> None:
        stopped = report["stopped"]
        super().__init__(f"stopped at {stopped['question_id']}: {stopped['error']}")
        self.report = report

    def __reduce__(self) -> tuple[type, tuple[dict[str, Any]]]:
        # Rebuilt from its report, since the message alone cannot rebuild it:
        # a copy, or one unpickled across a process boundary.
        return type(self), (self.report,)


class RunStopped(_Stopped, ProviderUnavailable):
    """A run a provider failure stopped part-way, with the answers before it on
    ``report``. A ``ProviderUnavailable``, so a caller's handler for one still
    catches it."""


class RunInterrupted(_Stopped, KeyboardInterrupt):
    """A run stopped with Ctrl-C part-way, with the answers before it on
    ``report``. A ``KeyboardInterrupt``, so it still stops a script or a
    notebook cell as Ctrl-C does, and no handler for a provider failure
    mistakes it for one."""


def _rates(rows: list[dict]) -> dict[str, Any]:
    abstentions = [row for row in rows if row["answer"]["abstained"]]
    return {
        "total": len(rows),
        "abstained": len(abstentions),
        "abstention_rate": len(abstentions) / len(rows) if rows else None,
        "reasons": dict(sorted(Counter(
            row["answer"]["abstention_reason"] for row in abstentions
        ).items())),
    }


def _checks(rows: list[dict]) -> dict[str, int]:
    """Answered rows by the worst of the figures each states (``verify.worst_check``).

    A figure is a mismatch if a check contradicts it, supported if a passage
    or the facts store supports it, and unverified otherwise, and an answer is
    counted by its worst figure. So ``supported`` is an answer every figure of
    which is supported, and one supported claim does not hide another that
    could not be checked. ``unchecked`` is an answer to a question that asks for
    no figure and states none: one that was asked for a figure and gave none is
    ``unverified``, since the checker records that there was nothing to check.
    Abstentions state nothing and are not counted.
    """
    worst: Counter[str] = Counter()
    for row in rows:
        if row["answer"]["abstained"]:
            continue
        worst[worst_check(row["answer"]["verification"]["checks"])] += 1
    return dict(sorted(worst.items()))


def answer_with_retries(label: str, ask: Callable[[], Asked]) -> tuple[Asked, int]:
    """``ask()``, and how many times it was asked: again after each wait in
    ``RETRY_WAITS_S``, or the longer ``retry_after`` the error asks for, while
    it raises ``ProviderBusy``. A wait longer than ``MAX_RETRY_WAIT_S``, and the
    last attempt's error, are not waited out: the error is raised.

    ``label`` names the question in the line logged before each wait. The
    measurement scripts under ``notebooks/answers/`` ask through this too, so
    a busy provider is waited out on the same terms wherever a run is made.
    """
    for attempt, wait in enumerate(RETRY_WAITS_S, start=1):
        try:
            return ask(), attempt
        except ProviderBusy as error:
            wait = max(wait, error.retry_after or 0)
            if wait > MAX_RETRY_WAIT_S:
                raise
            logger.warning("%s: %s; asking again in %.0fs", label, error, wait)
            sleep(wait)
    return ask(), len(RETRY_WAITS_S) + 1


@dataclass
class _ModelCalls:
    """Calls, observed usage, active attempt time and separate retry waits."""

    calls: int = 0
    usages: list[dict[str, Any]] = field(default_factory=list)
    ms: float = 0.0
    waited_ms: float = 0.0
    successes: int = 0

    def ask(self, label: str, ask: Callable[[], Asked],
            usage: Callable[[Asked], dict[str, Any]]) -> tuple[Asked, int]:
        def call():
            self.calls += 1
            started = perf_counter()
            try:
                result = ask()
                self.usages.append(usage(result))
                self.successes += 1
                return result
            except Exception as error:
                if (spent := getattr(error, "usage", None)) is not None:
                    self.usages.append(spent)
                raise
            finally:
                self.ms += (perf_counter() - started) * 1000

        started = perf_counter()
        previous_ms = self.ms
        try:
            return answer_with_retries(label, call)
        finally:
            elapsed_ms = (perf_counter() - started) * 1000
            self.waited_ms += max(0.0, elapsed_ms - (self.ms - previous_ms))


class _JudgeStopped(Exception):
    """A completed answer whose judging stopped, with the original failure."""

    def __init__(self, row: dict[str, Any], cause: Exception | KeyboardInterrupt):
        super().__init__(str(cause))
        self.row, self.cause = row, cause


def evaluate(
    questions: Iterable[BenchmarkQuestion],
    retriever: Retriever,
    config: GenerationConfig | None = None,
    *,
    run_id: str,
    min_score: float | None = None,
    top_k: int = FINAL_K,
    llm: Any | None = None,
    use_facts: bool = True,
    use_decomposition: bool = True,
    use_refusal: bool = True,
    stack: Any | None = None,
    judge: LLMJudge | None = None,
    token_prices: TokenPrices | None = None,
    judge_token_prices: TokenPrices | None = None,
) -> dict[str, Any]:
    """One configuration per run; count every completed question exactly once.

    Benchmark ticker/year fields override parsed scope when provided. Errors
    stop the run instead of being counted as abstentions. A failure asking
    again may fix (``ProviderBusy``) is asked again first, after each wait in
    ``RETRY_WAITS_S``, and each row records how many times its question was
    asked, as ``attempts``. Only the model is asked again: the passages and
    prompt of the first attempt are reused, so a retry costs no search. When
    the provider still cannot answer, the run raises :class:`RunStopped`, a
    ``ProviderUnavailable``, carrying the report of every question before that
    one, with ``stopped`` naming it: on a hosted
    free plan a rate limit can come at question 40 of 48, and it should not
    throw away the 39 answers before it. Ctrl-C raises :class:`RunInterrupted`,
    a ``KeyboardInterrupt``, carrying the same report. A complete report has
    ``stopped`` None. Outside judging, other errors such as bugs come through
    as themselves. Judge errors retain the completed answer in a partial report
    with ``stopped.stage`` set to ``judge``; that row needs only judging on resume.
    Empty subsets have a
    null rate, with their denominators explicit. The unanswerable subset is
    reported separately so a high overall rate cannot masquerade as quality.
    Rows can be written as JSONL and opened by ``src.app.answers``.

    Every answer is put through ``verify_answer`` as the app puts it before
    showing it, so each row carries its checks, and ``checks`` counts the
    answered rows by the worst of the figures each states: how many answers a
    check contradicts, how many state a figure that could not be checked, and
    how many have every figure supported by a passage or the facts store.

    ``stack`` is the named configuration a caller assembled from, recorded in
    the report so a results file says which configuration to select in the app
    to see the same system (#43). It also decides whether each question's
    filters are applied: a row measured without the metadata filter is answered
    from a search of the whole corpus. The settings below decide the rest.

    ``use_facts`` is the with/without half of the numeric-routing ablation
    (#34): False sends every question to retrieval and generation. It matters
    for the generated XBRL benchmark in particular, whose questions are built
    from the same ``facts.parquet`` the route answers from, so a run with the
    route on measures the store against itself. Each row records which route
    answered it, and ``routes`` counts them, so a run that mixes the two is
    read as what it is rather than as one number.

    ``use_decomposition`` is the same for #35: False searches every question
    once, instead of once per filing for a question naming several. Each row
    records the sub-questions its evidence came from, and ``decomposed`` counts
    the rows that were split, so a run says how many questions the row it is
    measuring could even apply to.

    ``use_refusal`` is the same for the refusal of a question the parser reads
    as asking for advice or a prediction, or for a figure of a company outside
    the corpus: False searches those too and leaves the abstaining to a model.
    A refused row abstains with the reason that says so, and ``refused`` counts
    them.

    ``judge`` optionally scores semantic faithfulness and gold correctness;
    absent it, semantic scores are null on every row. A separate
    ``answerability_correct`` metric scores abstention decisions on every row.
    Numeric checks remain separate. Citation precision/recall,
    stage timings and token usage are always recorded. ``token_prices`` and
    ``judge_token_prices`` are explicit USD rates per million tokens. Unknown
    usage/prices produce null hosted cost. Judge resources are recorded
    separately. Invalid judge output leaves an answer unscored; other judge
    failures stop the run, retaining that completed answer and any observed
    usage. Model attempt time and retry waits are separate resource stages.
    """
    if not run_id.strip():
        raise ValueError("run_id must be non-empty")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if min_score is not None and not isfinite(min_score):
        raise ValueError("min_score must be finite or None")
    questions = list(questions)
    if len({q.question_id for q in questions}) != len(questions):
        raise ValueError("question_id must be unique within a run")
    config = config if config is not None else config_from_env()
    rows = []

    def report(stopped: dict[str, str] | None) -> dict[str, Any]:
        # The rows answered so far and the rates over them, with ``stopped``
        # naming the question the run stopped at and why, or None.
        return {
            "run_id": run_id,
            # Which named configuration this run answered with (#43), so a
            # report says what to select in the app to see the same system.
            # None for a caller that assembled a stack of its own.
            "stack": None if stack is None else stack.to_dict(),
            "retriever": retriever.name,
            "config": config.to_dict(),
            "judge": None if judge is None else judge.to_dict(),
            "token_prices": None if token_prices is None else token_prices.to_dict(),
            "judge_token_prices": (None if judge_token_prices is None
                                   else judge_token_prices.to_dict()),
            "min_score": min_score,
            "top_k": top_k,
            "use_facts": use_facts,
            "use_decomposition": use_decomposition,
            "use_refusal": use_refusal,
            "routes": dict(sorted(Counter(row["route"] for row in rows).items())),
            "decomposed": sum(1 for row in rows if row["sub_questions"]),
            "refused": sum(1 for row in rows
                           if row["answer"]["abstention_reason"] in REFUSALS.values()),
            "summary": _rates(rows),
            "checks": _checks(rows),
            "quality": quality_summary(rows),
            "resources": resource_summary(rows),
            "by_answerability": {
                "unanswerable": _rates([r for r in rows if r["question_type"] == UNANSWERABLE]),
                "answerable": _rates([r for r in rows if r["question_type"] != UNANSWERABLE]),
            },
            "stopped": stopped,
            "results": rows,
        }

    def row_for(question: BenchmarkQuestion) -> dict[str, Any]:
        # One question asked and answered, as its row in the report.
        started = perf_counter()
        timed = _TimedRetriever(retriever)
        generation_calls = _ModelCalls()
        parsed = parse_question(question.question)
        query = parsed.to_query(top_k=top_k)
        if question.ticker is not None:
            query = replace(query, tickers=(question.ticker,))
        if question.fiscal_year is not None:
            query = replace(query, fiscal_years=(question.fiscal_year,))
        # The company and year the question is about, which the answer is
        # checked against whatever the row searches.
        asked = parsed.scoped_to(query)
        # After the benchmark's own ticker and year, so a row measured without
        # the metadata filter drops those too and searches what it measured.
        if stack is not None:
            query = stack.scoped(query)
        # Only the model call is asked again (#108): the passages and prompt
        # it runs on are built once, so a busy provider does not repeat the
        # lookup, decomposition and search before it. One attempt where the
        # question never reached a model, as before.
        attempts = 1

        def ask_model(run: Callable[[], Generation]) -> Generation:
            nonlocal attempts
            generation, attempts = generation_calls.ask(
                question.question_id, run,
                lambda result: {"input_tokens": result.input_tokens,
                                "output_tokens": result.output_tokens},
            )
            return generation

        answer = answer_question(
            question.question, timed, config, query=query, min_score=min_score,
            llm=llm, use_facts=use_facts, use_decomposition=use_decomposition,
            use_refusal=use_refusal, parsed=parsed, ask_model=ask_model,
        )
        # Checked as the app checks an answer before showing it, against the
        # company and year the question is about, so a row carries what the
        # answer card would say of it and the viewer (src.app.answers) has
        # checks to show. A row measured without the metadata filter searches
        # every filing, and the app still checks its answer against the
        # sidebar's company and year, so that is the scope here too.
        answer = verify_answer(answer, parsed=asked)
        total_ms = (perf_counter() - started) * 1000
        answerable = question.question_type != UNANSWERABLE
        quality = citation_scores(question, answer) | {
            "faithfulness": None, "correctness": None, "method": "not_scored",
            "answerability_correct": float(answer.abstained != answerable),
            "faithfulness_reason": None, "correctness_reason": None,
            "error": None,
        }
        judge_calls = _ModelCalls()
        judge_stopped = None
        if answer.abstained:
            # No factual claims to ground; do not reward vacuous faithfulness.
            quality.update(method="abstention")
            if judge is not None:
                quality.update(correctness=float(not answerable),
                               correctness_reason="Benchmark answerability")
        elif judge is not None:
            try:
                result, _ = judge_calls.ask(
                    question.question_id + ":judge", lambda: judge(question, answer),
                    lambda result: {"input_tokens": result.get("input_tokens"),
                                    "output_tokens": result.get("output_tokens")},
                )
                # LLMJudge validates scores once, at the provider boundary.
                quality.update({key: result[key] for key in
                                ("faithfulness", "correctness", "faithfulness_reason",
                                 "correctness_reason")}, method="llm_judge")
                if not answerable:
                    quality.update(correctness=0.0,
                                   correctness_reason="Answered an unanswerable question")
            except JudgeInvalid as error:
                quality.update(method="judge_failed", error=str(error))
            except (Exception, KeyboardInterrupt) as error:
                judge_stopped = error
                quality.update(method="judge_failed",
                               error="interrupted" if isinstance(error, KeyboardInterrupt) else str(error))
        cost = query_cost(config.provider, generation_calls.usages,
                          attempts=generation_calls.calls, prices=token_prices,
                          successful_calls=generation_calls.successes)
        judge_provider = config.provider if judge is None else judge.config.provider
        judge_cost = query_cost(judge_provider, judge_calls.usages, attempts=judge_calls.calls,
                                prices=judge_token_prices, successful_calls=judge_calls.successes)
        row = {"question_id": question.question_id, "run_id": run_id,
                "expected_answer": question.expected_answer,
                "supporting_chunk_ids": list(question.supporting_chunk_ids),
                "quality": quality,
                # No reranker exists in the current stack; fusion is retrieval.
                "latency_ms": {"retrieve": timed.latency_ms, "rerank": 0.0,
                               "generate": generation_calls.ms, "total": total_ms,
                               "retry_wait": generation_calls.waited_ms,
                               "other": max(0.0, total_ms - timed.latency_ms - generation_calls.ms
                                            - generation_calls.waited_ms)},
                "rerank_enabled": False,
                "cost": cost,
                "judge_overhead": {"latency_ms": judge_calls.ms,
                                   "retry_wait_ms": judge_calls.waited_ms,
                                   "total_ms": judge_calls.ms + judge_calls.waited_ms,
                                   "cost": judge_cost},
                "question_type": question.question_type,
                # Which route answered it. The Answer records the provider
                # that produced it, and a looked-up answer says "facts"
                # rather than naming the model that was never asked.
                "route": answer.config.provider,
                # What it was split into, empty where it was not (#35).
                "sub_questions": list(answer.sub_questions),
                # How many times it was asked, more than once where the
                # provider was busy: how often a hosted free plan was.
                "attempts": attempts,
                "answer": answer.to_dict()}
        if judge_stopped is not None:
            # The answer and its paid model call completed before judging failed.
            # Keep them in the partial report before the outer handler stops it.
            raise _JudgeStopped(row, judge_stopped)
        return row

    for question in questions:
        try:
            rows.append(row_for(question))
        except _JudgeStopped as stop:
            rows.append(stop.row)
            stopped = _stopped(question, f"{type(stop.cause).__name__}: {stop.cause}") | {"stage": "judge"}
            error_type = RunInterrupted if isinstance(stop.cause, KeyboardInterrupt) else RunStopped
            raise error_type(report(stopped)) from stop.cause
        except ProviderUnavailable as error:
            stopped = _stopped(question, f"{type(error).__name__}: {error}")
            raise RunStopped(report(stopped)) from error
        except KeyboardInterrupt as error:
            # Most likely pressed during a retry's wait, which is when someone
            # watching a run is most tempted to.
            stopped = _stopped(question, "KeyboardInterrupt: interrupted")
            raise RunInterrupted(report(stopped)) from error
    return report(None)


def _stopped(question: BenchmarkQuestion, error: str) -> dict[str, str]:
    """The ``stopped`` entry of a report: the question a run stopped at, and why."""
    return {"question_id": question.question_id, "error": error}
