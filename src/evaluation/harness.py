"""Run grounded QA and report abstention rates over a validated benchmark."""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import replace
from functools import partial
from math import isfinite
from time import sleep
from typing import Any

from src.rag import ProviderBusy, answer_question, config_from_env, parse_question
from src.rag.records import Answer, GenerationConfig
from src.retrieval.base import Retriever
from src.retrieval.constants import FINAL_K
from .records import BenchmarkQuestion, UNANSWERABLE

logger = logging.getLogger(__name__)

# How long to wait before asking a question again after a failure that asking
# again may fix (``ProviderBusy``): a rate limit, a server error, a dropped
# connection. One wait per retry, so a question is asked at most three times.
# The second is a full minute because the free plan's rate limits are per
# minute. Anything else, and the third failure, stops the run.
RETRY_WAITS_S = (10, 60)


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


def _answer_with_retries(question: BenchmarkQuestion, ask: Callable[[], Answer]) -> Answer:
    """``ask()``, asked again after each wait in ``RETRY_WAITS_S`` while it
    raises ``ProviderBusy``. The last attempt's error is not caught."""
    for wait in RETRY_WAITS_S:
        try:
            return ask()
        except ProviderBusy as error:
            logger.warning("%s: %s; asking again in %ss", question.question_id, error, wait)
            sleep(wait)
    return ask()


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
) -> dict[str, Any]:
    """One configuration per run; count every completed question exactly once.

    Benchmark ticker/year fields override parsed scope when provided. Errors
    propagate instead of being counted as abstentions, except that a failure
    asking again may fix (``ProviderBusy``) is asked again, after each wait in
    ``RETRY_WAITS_S``, before it stops the run: on a hosted free plan a rate
    limit or a dropped connection can come at question 40 of 48, and one
    should not throw away the 39 answers before it. Empty subsets have a
    null rate, with their denominators explicit. The unanswerable subset is
    reported separately so a high overall rate cannot masquerade as quality.
    Rows can be written as JSONL and opened by ``src.app.answers``.

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
    for question in questions:
        parsed = parse_question(question.question)
        query = parsed.to_query(top_k=top_k)
        if question.ticker is not None:
            query = replace(query, tickers=(question.ticker,))
        if question.fiscal_year is not None:
            query = replace(query, fiscal_years=(question.fiscal_year,))
        answer = _answer_with_retries(question, partial(
            answer_question, question.question, retriever, config, query=query,
            min_score=min_score, llm=llm, use_facts=use_facts,
            use_decomposition=use_decomposition, parsed=parsed,
        ))
        rows.append({"question_id": question.question_id, "run_id": run_id,
                     "question_type": question.question_type,
                     # Which route answered it. The Answer records the provider
                     # that produced it, and a looked-up answer says "facts"
                     # rather than naming the model that was never asked.
                     "route": answer.config.provider,
                     # What it was split into, empty where it was not (#35).
                     "sub_questions": list(answer.sub_questions),
                     "answer": answer.to_dict()})
    return {
        "run_id": run_id,
        "retriever": retriever.name,
        "config": config.to_dict(),
        "min_score": min_score,
        "top_k": top_k,
        "use_facts": use_facts,
        "use_decomposition": use_decomposition,
        "routes": dict(sorted(Counter(row["route"] for row in rows).items())),
        "decomposed": sum(1 for row in rows if row["sub_questions"]),
        "summary": _rates(rows),
        "by_answerability": {
            "unanswerable": _rates([r for r in rows if r["question_type"] == UNANSWERABLE]),
            "answerable": _rates([r for r in rows if r["question_type"] != UNANSWERABLE]),
        },
        "results": rows,
    }
