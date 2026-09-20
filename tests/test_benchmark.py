"""Tests for the benchmark contract and JSONL loader."""

import json

import pytest

from src.evaluation.benchmark import load_questions
from src.evaluation.metrics import (
    hard_negative_accuracy,
    mrr,
    ndcg_at_k,
    recall_at_k,
    score_question,
)
from src.evaluation.records import BenchmarkQuestion, BenchmarkValidationError, RunResult


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
    return question


def _write_question(path, question):
    path.write_text(json.dumps(question) + "\n", encoding="utf-8")


def test_loader_validates_and_normalises_question(tmp_path):
    path = tmp_path / "questions.jsonl"
    _write_question(path, _question())

    questions = load_questions(path, chunk_ids={"AAPL-2024-1", "AAPL-2023-1"})

    assert questions[0].ticker == "AAPL"
    assert questions[0].supporting_chunk_ids == ("AAPL-2024-1",)


def test_a_comparison_across_companies_and_years_has_no_single_ticker_or_year(tmp_path):
    path = tmp_path / "questions.jsonl"
    _write_question(path, _question(question_type="comparative", ticker=None, fiscal_year=None))

    [question] = load_questions(path, chunk_ids={"AAPL-2024-1", "AAPL-2023-1"})

    assert (question.ticker, question.fiscal_year) == (None, None)


def test_an_unanswerable_question_has_no_supporting_chunks(tmp_path):
    path = tmp_path / "questions.jsonl"
    _write_question(path, _question(
        question_type="unanswerable", supporting_chunk_ids=[], ticker=None,
        expected_answer="The filings do not answer this question.",
    ))

    [question] = load_questions(path, chunk_ids={"AAPL-2023-1"})

    assert question.supporting_chunk_ids == ()
    assert question.hard_negative_chunk_ids == ("AAPL-2023-1",)


def test_an_unanswerable_question_with_supporting_chunks_is_rejected(tmp_path):
    path = tmp_path / "questions.jsonl"
    _write_question(path, _question(question_type="unanswerable"))

    with pytest.raises(BenchmarkValidationError, match="unanswerable question has no supporting"):
        load_questions(path, chunk_ids={"AAPL-2024-1", "AAPL-2023-1"})


def test_an_answerable_question_needs_supporting_chunks(tmp_path):
    path = tmp_path / "questions.jsonl"
    _write_question(path, _question(supporting_chunk_ids=[]))

    with pytest.raises(BenchmarkValidationError, match="at least one chunk ID"):
        load_questions(path, chunk_ids={"AAPL-2023-1"})


@pytest.mark.parametrize("question_type", ["comparison", "narrative", "Numeric"])
def test_question_type_must_be_one_the_engine_assigns(tmp_path, question_type):
    path = tmp_path / "questions.jsonl"
    _write_question(path, _question(question_type=question_type))

    with pytest.raises(BenchmarkValidationError, match="question_type must be one of"):
        load_questions(path, chunk_ids={"AAPL-2024-1", "AAPL-2023-1"})


def test_loader_rejects_unknown_chunk_ids(tmp_path):
    path = tmp_path / "questions.jsonl"
    _write_question(path, _question(supporting_chunk_ids=["missing"]))

    with pytest.raises(BenchmarkValidationError, match="unknown chunk IDs: missing"):
        load_questions(path, chunk_ids={"AAPL-2023-1"})


def test_loader_rejects_duplicate_question_ids(tmp_path):
    path = tmp_path / "questions.jsonl"
    path.write_text(
        "\n".join(json.dumps(_question()) for _ in range(2)) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(BenchmarkValidationError, match="duplicate question_id"):
        load_questions(path, chunk_ids={"AAPL-2024-1", "AAPL-2023-1"})


def test_loader_reports_missing_corpus(tmp_path):
    path = tmp_path / "questions.jsonl"
    _write_question(path, _question())

    with pytest.raises(FileNotFoundError, match="python -m src.pipeline chunk"):
        load_questions(path, processed_dir=tmp_path / "processed")


def test_loader_keeps_unicode_line_separators_inside_a_record(tmp_path):
    path = tmp_path / "questions.jsonl"
    answer = "Net sales rose in FY2024overall"
    path.write_text(
        json.dumps(_question(expected_answer=answer), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    questions = load_questions(path, chunk_ids={"AAPL-2024-1", "AAPL-2023-1"})

    assert len(questions) == 1
    assert questions[0].expected_answer == answer


def test_loader_accepts_a_byte_order_mark(tmp_path):
    path = tmp_path / "questions.jsonl"
    path.write_text(json.dumps(_question()) + "\n", encoding="utf-8-sig")

    questions = load_questions(path, chunk_ids={"AAPL-2024-1", "AAPL-2023-1"})

    assert questions[0].question_id == "q001"


def test_loader_reports_line_numbers_with_windows_line_endings(tmp_path):
    path = tmp_path / "questions.jsonl"
    path.write_bytes(
        ("\r\n".join(json.dumps(_question()) for _ in range(2)) + "\r\n").encode("utf-8")
    )

    with pytest.raises(BenchmarkValidationError, match=r"questions\.jsonl:2: duplicate question_id"):
        load_questions(path, chunk_ids={"AAPL-2024-1", "AAPL-2023-1"})


def test_empty_run_result_keeps_retriever_name():
    result = RunResult.from_passages("q001", [], retriever="bm25")

    assert result.retriever == "bm25"


def test_run_result_is_hashable_and_copies_inputs():
    chunk_ids = ["chunk-1"]
    scores = [3.5]
    config = {"k": 10}
    result = RunResult(
        question_id="q001",
        retriever="bm25",
        retrieved_chunk_ids=chunk_ids,
        retrieved_scores=scores,
        config=config,
    )
    chunk_ids.append("chunk-2")
    scores.append(1.2)
    config["k"] = 20

    assert result.to_dict()["retrieved_chunk_ids"] == ["chunk-1"]
    assert result.to_dict()["retrieved_scores"] == [3.5]
    assert result.config == {"k": 10}
    assert isinstance(hash(result), int)