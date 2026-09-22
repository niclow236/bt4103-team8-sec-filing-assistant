"""Run grounded QA and report abstention rates over a validated benchmark."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import replace
from math import isfinite
from typing import Any

from src.rag import answer_question, config_from_env, parse_question
from src.rag.records import GenerationConfig
from src.retrieval.base import Retriever
from src.retrieval.constants import FINAL_K
from .records import BenchmarkQuestion, UNANSWERABLE


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


def evaluate(
    questions: Iterable[BenchmarkQuestion],
    retriever: Retriever,
    config: GenerationConfig | None = None,
    *,
    run_id: str,
    min_score: float | None = None,
    top_k: int = FINAL_K,
    llm: Any | None = None,
) -> dict[str, Any]:
    """One configuration per run; count every completed question exactly once.

    Benchmark ticker/year fields override parsed scope when provided. Errors
    propagate instead of being counted as abstentions. Empty subsets have a
    null rate, with their denominators explicit. The unanswerable subset is
    reported separately so a high overall rate cannot masquerade as quality.
    Rows can be written as JSONL and opened by ``src.app.answers``.
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
        query = parse_question(question.question).to_query(top_k=top_k)
        if question.ticker is not None:
            query = replace(query, tickers=(question.ticker,))
        if question.fiscal_year is not None:
            query = replace(query, fiscal_years=(question.fiscal_year,))
        answer = answer_question(question.question, retriever, config, query=query,
                                 min_score=min_score, llm=llm)
        rows.append({"question_id": question.question_id, "run_id": run_id,
                     "question_type": question.question_type, "answer": answer.to_dict()})
    return {
        "run_id": run_id,
        "retriever": retriever.name,
        "config": config.to_dict(),
        "min_score": min_score,
        "top_k": top_k,
        "summary": _rates(rows),
        "by_answerability": {
            "unanswerable": _rates([r for r in rows if r["question_type"] == UNANSWERABLE]),
            "answerable": _rates([r for r in rows if r["question_type"] != UNANSWERABLE]),
        },
        "results": rows,
    }
