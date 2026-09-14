"""Tests for the benchmark contract and JSONL loader."""

import json

import pytest

from src.evaluation.benchmark import load_questions
from src.evaluation.records import BenchmarkValidationError, RunResult


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


def test_run_result_keeps_retrieval_order_and_scores():
    result = RunResult(
        question_id="q001",
        retriever="bm25",
        retrieved_chunk_ids=("chunk-1", "chunk-2"),
        retrieved_scores=(3.5, 1.2),
    )

    assert result.to_dict()["retrieved_chunk_ids"] == ["chunk-1", "chunk-2"]
    assert result.to_dict()["retrieved_scores"] == [3.5, 1.2]