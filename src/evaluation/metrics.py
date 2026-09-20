"""Ranking metrics for evaluating retrieval results against a benchmark question."""

from __future__ import annotations

from math import log2
from typing import Iterable, Sequence

from .records import BenchmarkQuestion, RunResult


def _as_set(chunk_ids: Iterable[str] | Sequence[str] | None) -> set[str]:
    if chunk_ids is None:
        return set()
    return set(chunk_ids)


def recall_at_k(
    result: RunResult,
    relevant_chunk_ids: Iterable[str] | Sequence[str] | None,
    *,
    k: int = 10,
) -> float:
    """Fraction of relevant chunk IDs retrieved in the first ``k`` positions."""
    relevant = _as_set(relevant_chunk_ids)
    if not relevant:
        return 1.0
    if k <= 0:
        return 0.0
    seen = set(result.retrieved_chunk_ids[:k])
    return len(seen & relevant) / len(relevant)


def ndcg_at_k(
    result: RunResult,
    relevant_chunk_ids: Iterable[str] | Sequence[str] | None,
    *,
    k: int = 10,
) -> float:
    """Discounted cumulative gain at ``k`` for binary relevance."""
    relevant = _as_set(relevant_chunk_ids)
    if not relevant:
        return 1.0
    if k <= 0:
        return 0.0

    dcg = 0.0
    for rank, chunk_id in enumerate(result.retrieved_chunk_ids[:k], start=1):
        if chunk_id in relevant:
            dcg += 1.0 / log2(rank + 1)

    ideal_count = min(len(relevant), k)
    idcg = sum(1.0 / log2(rank + 1) for rank in range(1, ideal_count + 1))
    return 0.0 if idcg == 0 else dcg / idcg


def mrr(
    result: RunResult,
    relevant_chunk_ids: Iterable[str] | Sequence[str] | None,
) -> float:
    """Mean reciprocal rank of the first relevant chunk retrieved."""
    relevant = _as_set(relevant_chunk_ids)
    if not relevant:
        return 1.0
    for rank, chunk_id in enumerate(result.retrieved_chunk_ids, start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


def hard_negative_accuracy(
    result: RunResult,
    hard_negative_chunk_ids: Iterable[str] | Sequence[str] | None,
    *,
    k: int = 10,
) -> float:
    """Fraction of hard negatives that remain outside the top ``k`` results."""
    hard_negatives = _as_set(hard_negative_chunk_ids)
    if not hard_negatives:
        return 1.0
    if k <= 0:
        return 0.0
    seen = set(result.retrieved_chunk_ids[:k])
    avoided = hard_negatives - seen
    return len(avoided) / len(hard_negatives)


def score_question(
    question: BenchmarkQuestion,
    result: RunResult,
    *,
    k: int = 10,
) -> dict[str, float | str]:
    """Compute the evaluation metrics for one benchmark question and one run."""
    if question.question_id != result.question_id:
        raise ValueError(
            f"question_id mismatch: {question.question_id!r} != {result.question_id!r}"
        )

    metrics: dict[str, float | str] = {
        "question_id": question.question_id,
        "retriever": result.retriever,
        "recall@k": recall_at_k(result, question.supporting_chunk_ids, k=k),
        "ndcg@10": ndcg_at_k(result, question.supporting_chunk_ids, k=10),
        "mrr": mrr(result, question.supporting_chunk_ids),
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
