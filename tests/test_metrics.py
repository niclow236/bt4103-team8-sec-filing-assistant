"""Tests for retriever-independent retrieval metrics."""

import pytest

from src.evaluation.metrics import (
    hard_negative_accuracy,
    mrr,
    ndcg_at_k,
    recall_at_k,
    score_question,
)
from src.evaluation.records import BenchmarkQuestion, RunResult


def _question(**overrides):
    question = {
        "question_id": "q001",
        "question": "What was revenue?",
        "expected_answer": "$100",
        "supporting_chunk_ids": ["AAPL-2024-1"],
        "hard_negative_chunk_ids": ["AAPL-2023-1"],
        "ticker": "aapl",
        "fiscal_year": 2024,
        "question_type": "numeric",
        "difficulty": "easy",
        "source": "handwritten",
    }
    question.update(overrides)
    return BenchmarkQuestion.from_mapping(question)


def _result(chunk_ids, **overrides):
    values = {
        "question_id": "q001",
        "retriever": "bm25",
        "retrieved_chunk_ids": tuple(chunk_ids),
        "retrieved_scores": tuple(float(len(chunk_ids) - i) for i in range(len(chunk_ids))),
    }
    values.update(overrides)
    return RunResult(**values)


def test_metrics_match_hand_computed_values_for_a_hit_at_rank_three():
    question = _question()
    result = _result(["AAPL-2023-1", "MSFT-2024-1", "AAPL-2024-1"])

    row = score_question(question, result, k=10)

    assert row["recall"] == 1.0
    assert row["ndcg"] == pytest.approx(0.5)
    assert row["mrr"] == pytest.approx(1 / 3)
    assert row["hard_negative_accuracy"] == 0.0


def test_metrics_below_the_cutoff_score_zero():
    question = _question()
    result = _result(["AAPL-2023-1", "MSFT-2024-1", "AAPL-2024-1"])

    row = score_question(question, result, k=2)

    assert (row["recall"], row["ndcg"], row["mrr"]) == (0.0, 0.0, 0.0)
    assert row["k"] == 2
    assert row["question_type"] == "numeric"
    assert row["latency_ms"] is None


def test_an_unanswerable_question_scores_only_hard_negative_accuracy():
    question = _question(
        question_type="unanswerable",
        supporting_chunk_ids=[],
        ticker=None,
        expected_answer="The filings do not answer this question.",
    )
    result = _result(["MSFT-2024-1"])

    row = score_question(question, result, k=10)

    assert row["recall"] is None
    assert row["ndcg"] is None
    assert row["mrr"] is None
    assert row["hard_negative_accuracy"] == 1.0


def test_recall_and_ndcg_cap_relevant_chunks_at_the_cutoff():
    supporting = [f"AAPL-2024-{i}" for i in range(1, 21)]
    question = _question(
        supporting_chunk_ids=supporting,
        hard_negative_chunk_ids=["MSFT-2024-1"],
    )
    result = _result(supporting[:10])

    assert recall_at_k(result, question.supporting_chunk_ids, k=10) == 1.0
    assert ndcg_at_k(result, question.supporting_chunk_ids, k=10) == pytest.approx(1.0)


def test_retrieving_nothing_avoids_every_hard_negative():
    question = _question()
    result = _result(["AAPL-2023-1"])

    assert hard_negative_accuracy(result, question.hard_negative_chunk_ids, k=0) == 1.0
    assert mrr(result, question.supporting_chunk_ids, k=10) == 0.0


def test_run_result_from_passages_orders_by_rank():
    from src.retrieval.records import RetrievedPassage

    def passage(chunk_id, rank):
        return RetrievedPassage(
            chunk_id=chunk_id,
            text=chunk_id,
            score=float(rank),
            rank=rank,
            retriever="bm25",
            ticker="AAPL",
            company="Apple",
            fiscal_year=2024,
            item="7",
            title="Management's Discussion and Analysis",
            url="https://example.com/filing",
        )

    result = RunResult.from_passages(
        "q001",
        [passage("second", 2), passage("first", 1)],
        retriever="bm25",
    )

    assert result.retrieved_chunk_ids == ("first", "second")
