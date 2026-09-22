"""Ranking metrics for evaluating retrieval results against a benchmark question."""

from __future__ import annotations

from collections.abc import Iterable
from math import log2

from .records import BenchmarkQuestion, RunResult


def _as_set(chunk_ids: Iterable[str]) -> set[str]:
    return set(chunk_ids)


def recall_at_k(
    result: RunResult,
    relevant_chunk_ids: Iterable[str],
    *,
    k: int = 10,
) -> float | None:
    """Fraction of relevant chunks reachable at ``k`` that were retrieved."""
    relevant = _as_set(relevant_chunk_ids)
    if not relevant:
        return None
    if k <= 0:
        return 0.0
    seen = set(result.retrieved_chunk_ids[:k])
    return len(seen & relevant) / min(len(relevant), k)


def ndcg_at_k(
    result: RunResult,
    relevant_chunk_ids: Iterable[str],
    *,
    k: int = 10,
) -> float | None:
    """Discounted cumulative gain at ``k`` for binary relevance."""
    relevant = _as_set(relevant_chunk_ids)
    if not relevant:
        return None
    if k <= 0:
        return 0.0

    dcg = 0.0
    for rank, chunk_id in enumerate(result.retrieved_chunk_ids[:k], start=1):
        if chunk_id in relevant:
            dcg += 1.0 / log2(rank + 1)

    ideal_count = min(len(relevant), k)
    idcg = sum(1.0 / log2(rank + 1) for rank in range(1, ideal_count + 1))
    return dcg / idcg


def mrr(
    result: RunResult,
    relevant_chunk_ids: Iterable[str],
    *,
    k: int = 10,
) -> float | None:
    """Reciprocal rank of the first relevant chunk in the top ``k``."""
    relevant = _as_set(relevant_chunk_ids)
    if not relevant:
        return None
    if k <= 0:
        return 0.0
    for rank, chunk_id in enumerate(result.retrieved_chunk_ids[:k], start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


def hard_negative_accuracy(
    result: RunResult,
    hard_negative_chunk_ids: Iterable[str],
    *,
    k: int = 10,
) -> float | None:
    """Fraction of hard negatives that remain outside the top ``k`` results."""
    hard_negatives = _as_set(hard_negative_chunk_ids)
    if not hard_negatives:
        return None
    if k <= 0:
        return 1.0
    seen = set(result.retrieved_chunk_ids[:k])
    avoided = hard_negatives - seen
    return len(avoided) / len(hard_negatives)


def score_question(
    question: BenchmarkQuestion,
    result: RunResult,
    *,
    k: int = 10,
) -> dict[str, float | str | None]:
    """Compute the evaluation metrics for one benchmark question and one run."""
    if question.question_id != result.question_id:
        raise ValueError(
            f"question_id mismatch: {question.question_id!r} != {result.question_id!r}"
        )

    metrics: dict[str, float | str | None] = {
        "question_id": question.question_id,
        "question_type": question.question_type,
        "retriever": result.retriever,
        "k": k,
        "latency_ms": result.latency_ms,
        "recall": recall_at_k(result, question.supporting_chunk_ids, k=k),
        "ndcg": ndcg_at_k(result, question.supporting_chunk_ids, k=k),
        "mrr": mrr(result, question.supporting_chunk_ids, k=k),
        "hard_negative_accuracy": hard_negative_accuracy(
            result,
            question.hard_negative_chunk_ids,
            k=k,
        ),
    }
    return metrics


__all__ = [
    "hard_negative_accuracy",
    "mrr",
    "ndcg_at_k",
    "recall_at_k",
    "score_question",
]
