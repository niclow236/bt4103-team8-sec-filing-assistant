"""Ranking metrics for evaluating retrieval results against a benchmark question."""

from __future__ import annotations

from collections.abc import Iterable
from math import inf, log2
from statistics import median

from .records import BenchmarkQuestion, RunResult

RETRIEVAL_METRICS = ("recall", "ndcg", "mrr", "hard_negative_accuracy")
# Where a benchmark's median question ranked its first right passage (#164),
# as a summary saves it.
MEDIAN_RANK = "median_rank"


def _as_set(chunk_ids: Iterable[str]) -> set[str]:
    return set(chunk_ids)


def recall_at_k(
    result: RunResult,
    relevant_chunk_ids: Iterable[str],
    *,
    k: int = 10,
) -> float | None:
    """Recall at ``k``: the share of all the relevant chunks that are in the top ``k``.

    Every relevant chunk is in the divisor, found or not: with twenty relevant
    chunks and five of them in a top 5, this is 0.25. So a question with more
    relevant chunks than ``k`` cannot reach 1.0, which is what recall means, and
    is where it parts from nDCG, whose ideal ranking is cut at ``k``.

    Until #140 the divisor was ``min(relevant, k)``, which scored that case 1.0.
    No reported figure moved with the change: ``generate_xbrl_questions`` lists
    at most three supporting chunks a question and nothing is scored at a
    cutoff under 3, so the two divisors were the same number for every
    question measured.
    """
    relevant = _as_set(relevant_chunk_ids)
    if not relevant:
        return None
    if k <= 0:
        return 0.0
    seen = set(result.retrieved_chunk_ids[:k])
    return len(seen & relevant) / len(relevant)


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


def first_relevant_rank(
    retrieved_chunk_ids: Iterable[str],
    relevant_chunk_ids: Iterable[str],
) -> int | None:
    """The 1-based rank of the first relevant chunk retrieved, or None if none was."""
    relevant = _as_set(relevant_chunk_ids)
    return next((rank for rank, chunk_id in enumerate(retrieved_chunk_ids, start=1)
                 if chunk_id in relevant), None)


def rank_from_reciprocal(reciprocal: float) -> int | None:
    """The rank a saved reciprocal rank stands for, or None for 0: no relevant
    chunk within the cutoff it was scored at."""
    return round(1 / reciprocal) if reciprocal > 0 else None


def median_rank(reciprocal_ranks: Iterable[float]) -> float | None:
    """The median rank of the first relevant chunk, read off each question's
    saved reciprocal rank, or None over no questions.

    A reciprocal rank of 0 found none within the cutoff, so it ranks past
    every rank counted, and a median there is infinite.
    """
    ranks = [inf if (rank := rank_from_reciprocal(reciprocal)) is None else rank
             for reciprocal in reciprocal_ranks]
    return median(ranks) if ranks else None


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
    rank = first_relevant_rank(result.retrieved_chunk_ids[:k], relevant)
    return 0.0 if rank is None else 1.0 / rank


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
    "first_relevant_rank",
    "hard_negative_accuracy",
    "median_rank",
    "mrr",
    "ndcg_at_k",
    "rank_from_reciprocal",
    "recall_at_k",
    "score_question",
]
