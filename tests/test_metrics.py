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


def test_ndcg_cuts_its_ideal_at_the_cutoff_and_recall_counts_every_relevant_chunk():
    """Ten of twenty supporting chunks fill a top 10. No ranking of ten could do
    better, so nDCG is 1.0, and half of what supports the question was found."""
    supporting = [f"AAPL-2024-{i}" for i in range(1, 21)]
    question = _question(
        supporting_chunk_ids=supporting,
        hard_negative_chunk_ids=["MSFT-2024-1"],
    )
    result = _result(supporting[:10])

    assert recall_at_k(result, question.supporting_chunk_ids, k=10) == 10 / 20
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


# --- the arithmetic, worked out by hand (#49) -----------------------------------
#
# Three chunks support the question below and two of them are retrieved, at
# ranks 2 and 4. Every expected value is written as the sum it comes from, so
# a change to a discount, a denominator or a cutoff fails on the line that
# states it.

THREE = ["AAPL-2024-1", "AAPL-2024-2", "AAPL-2024-3"]
RANKED = ["MSFT-2024-1", "AAPL-2024-2", "MSFT-2024-2", "AAPL-2024-1", "MSFT-2024-3"]


def test_two_of_three_supporting_chunks_at_ranks_two_and_four():
    from math import log2

    question = _question(supporting_chunk_ids=THREE, hard_negative_chunk_ids=[])
    row = score_question(question, _result(RANKED), k=10)

    assert row["recall"] == pytest.approx(2 / 3)
    # The gain at rank r is 1 / log2(r + 1). The best order puts all three first.
    found = 1 / log2(3) + 1 / log2(5)
    ideal = 1 / log2(2) + 1 / log2(3) + 1 / log2(4)
    assert row["ndcg"] == pytest.approx(found / ideal)
    assert row["ndcg"] == pytest.approx(0.49819, abs=1e-5)
    # The reciprocal rank is the first supporting chunk's alone.
    assert row["mrr"] == pytest.approx(1 / 2)
    assert row["hard_negative_accuracy"] is None


@pytest.mark.parametrize("k, recall, ndcg, reciprocal_rank", [
    (1, 0.0, 0.0, 0.0),
    # One of the three, at rank 2. nDCG's ideal is the two a top 2 could hold:
    # 1/log2(3) over 1 + 1/log2(3).
    (2, 1 / 3, 0.38685, 1 / 2),
    # Both retrieved chunks are inside a top 4, of the three that support it.
    (4, 2 / 3, 0.49819, 1 / 2),
])
def test_the_cutoff_decides_what_is_counted_and_what_it_is_counted_out_of(
        k, recall, ndcg, reciprocal_rank):
    question = _question(supporting_chunk_ids=THREE, hard_negative_chunk_ids=[])
    row = score_question(question, _result(RANKED), k=k)

    assert row["recall"] == pytest.approx(recall)
    assert row["ndcg"] == pytest.approx(ndcg, abs=1e-5)
    assert row["mrr"] == pytest.approx(reciprocal_rank)
    assert row["k"] == k


def test_recall_is_out_of_everything_that_supports_not_what_the_cutoff_could_hold():
    """Twenty chunks support a question. Five of them in a top 5 is a quarter of
    them found, though a top 5 could hold no more: TP / (TP + FN), with the
    fifteen left out counted as missed. Divided by what the cutoff could hold,
    as it was until #140, these read 1.0 and 0.4."""
    supporting = [f"AAPL-2024-{i}" for i in range(20)]
    full = _result(supporting[:5])
    partial = _result([*supporting[:2], "MSFT-1", "MSFT-2", "MSFT-3"])

    assert recall_at_k(full, supporting, k=5) == pytest.approx(5 / 20)
    assert recall_at_k(partial, supporting, k=5) == pytest.approx(2 / 20)


@pytest.mark.parametrize("supporting, k", [
    (1, 3), (2, 3), (3, 3), (3, 7), (3, 10), (3, 16),
    # More supporting chunks than the cutoff holds: still out of all of them.
    (5, 3), (12, 10),
])
def test_recall_is_found_over_supporting_at_every_cutoff(supporting, k):
    """The first six are the generated benchmark's cases: at most three supporting
    chunks a question (test_a_generated_question_lists_at_most_three_supporting_chunks)
    at a cutoff of 3 or more, where the old divisor gave the same number."""
    relevant = [f"AAPL-2024-{i}" for i in range(supporting)]
    for found in range(min(supporting, k) + 1):
        ranked = [*relevant[:found], *(f"MSFT-{i}" for i in range(k - found))]
        assert recall_at_k(_result(ranked), relevant, k=k) == pytest.approx(found / supporting)


def test_the_best_and_the_worst_rankings_score_one_and_zero():
    question = _question(supporting_chunk_ids=THREE, hard_negative_chunk_ids=[])
    best = score_question(question, _result([*THREE, "MSFT-2024-1"]), k=10)
    worst = score_question(question, _result(["MSFT-2024-1", "MSFT-2024-2"]), k=10)

    assert (best["recall"], best["ndcg"], best["mrr"]) == (1.0, pytest.approx(1.0), 1.0)
    assert (worst["recall"], worst["ndcg"], worst["mrr"]) == (0.0, 0.0, 0.0)


def test_supporting_chunks_in_another_order_score_the_same():
    """A metric reads which chunks support a question, not the order they were listed in."""
    result = _result(RANKED)
    forwards = (ndcg_at_k(result, THREE, k=10), recall_at_k(result, THREE, k=10),
                mrr(result, THREE, k=10))
    backwards = (ndcg_at_k(result, THREE[::-1], k=10), recall_at_k(result, THREE[::-1], k=10),
                 mrr(result, THREE[::-1], k=10))
    assert forwards == backwards


def test_a_hard_negative_counts_against_a_ranking_only_inside_the_cutoff():
    negatives = ["MSFT-2024-1", "MSFT-2024-3", "MSFT-2024-9"]
    result = _result(RANKED)

    # Ranks 1 and 5 hold two of the three; the third was never retrieved.
    assert hard_negative_accuracy(result, negatives, k=10) == pytest.approx(1 / 3)
    # Cut at 3, only the one at rank 1 is inside.
    assert hard_negative_accuracy(result, negatives, k=3) == pytest.approx(2 / 3)
    assert hard_negative_accuracy(result, [], k=10) is None


@pytest.mark.parametrize("k", [0, -1])
def test_a_cutoff_of_nothing_finds_nothing_and_avoids_everything(k):
    result = _result(RANKED)

    assert recall_at_k(result, THREE, k=k) == 0.0
    assert ndcg_at_k(result, THREE, k=k) == 0.0
    assert mrr(result, THREE, k=k) == 0.0
    assert hard_negative_accuracy(result, ["MSFT-2024-1"], k=k) == 1.0


def test_a_question_with_nothing_to_find_is_left_unscored_whatever_the_cutoff():
    result = _result(RANKED)
    for k in (0, 10):
        assert recall_at_k(result, [], k=k) is None
        assert ndcg_at_k(result, [], k=k) is None
        assert mrr(result, [], k=k) is None


def test_a_result_is_never_scored_against_another_questions_answer():
    with pytest.raises(ValueError, match="question_id mismatch: 'q001' != 'q002'"):
        score_question(_question(), _result(RANKED, question_id="q002"), k=10)


def test_the_row_names_the_retriever_the_question_and_what_it_took():
    row = score_question(_question(), _result(RANKED, retriever="hybrid", latency_ms=12.5), k=5)

    assert (row["question_id"], row["retriever"], row["k"]) == ("q001", "hybrid", 5)
    assert row["question_type"] == "numeric" and row["latency_ms"] == 12.5


def test_a_run_averages_only_the_questions_a_metric_scored():
    """An unanswerable question has no recall. Counted as zero, it would lower the
    average of every run by the share of questions that had nothing to find."""
    from src.evaluation.run import _summary

    rows = [
        {"recall": 1.0, "ndcg": 0.5, "mrr": 1.0, "hard_negative_accuracy": None},
        {"recall": 0.0, "ndcg": 0.0, "mrr": 0.0, "hard_negative_accuracy": 0.5},
        {"recall": None, "ndcg": None, "mrr": None, "hard_negative_accuracy": 1.0},
    ]
    summary = _summary(rows)

    assert summary["questions"] == 3
    assert (summary["recall"], summary["ndcg"], summary["mrr"]) == (0.5, 0.25, 0.5)
    assert summary["hard_negative_accuracy"] == 0.75
    unscored = _summary([{"recall": None, "ndcg": None, "mrr": None,
                          "hard_negative_accuracy": None}])
    assert unscored["recall"] is None and unscored["questions"] == 1
