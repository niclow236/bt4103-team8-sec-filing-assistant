"""Tests for the benchmark contract and JSONL loader."""

import json
from pathlib import Path

import pandas as pd
import pytest

from src.evaluation.benchmark import generate_xbrl_questions, load_questions
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


def test_generate_xbrl_questions_uses_real_chunk_ids_and_xbrl_source(tmp_path):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    (processed_dir / "AAPL").mkdir()
    (processed_dir / "AAPL" / "filing.json").write_text(
        json.dumps({
            "ticker": "AAPL",
            "company": "Apple",
            "cik": 1,
            "form": "10-K",
            "filing_date": "2025-02-01",
            "accession_no": "0000000001-25-000001",
            "url": "https://example.com/filing",
            "source_path": "raw/filing.html",
            "chunks": [
                {
                    "chunk_id": "0000000001-25-000001_part_ii_item_7_000",
                    "section_id": "part_ii_item_7",
                    "part": "II",
                    "item": "7",
                    "title": "Management's Discussion",
                    "heading": "Revenue",
                    "text": "Revenue was $100 million.",
                    "n_chars": 30,
                    "chunk_index": 0,
                    "is_key_section": True,
                    "incorporated_into": [],
                    "content_type": "prose",
                    "table_index": None,
                    "table_caption": "",
                }
            ],
            "period_of_report": "2024-12-31",
        }), encoding="utf-8",
    )

    facts_file = tmp_path / "facts.parquet"
    pd.DataFrame([
        {
            "ticker": "AAPL",
            "cik": 1,
            "company": "Apple",
            "accession": "0000000001-25-000001",
            "concept": "Revenue",
            "label": "Revenue",
            "value": 100.0,
            "raw_value": "100",
            "unit": "USD",
            "scale": None,
            "fiscal_year": 2024,
            "fiscal_period": "FY",
            "period_of_report": "2024-12-31",
            "period_start": "2024-01-01",
            "period_end": "2024-12-31",
            "period_type": "duration",
            "statement_type": "",
            "is_audited": True,
            "is_current_year": True,
        }
    ]).to_parquet(facts_file, index=False)

    output = tmp_path / "generated.jsonl"
    questions = generate_xbrl_questions(facts_file, processed_dir=processed_dir, output_path=output)

    assert len(questions) == 1
    question = questions[0]
    assert question.source == "xbrl"
    assert question.question_type == "numeric"
    assert question.expected_answer == "100 USD"
    assert question.supporting_chunk_ids == ("0000000001-25-000001_part_ii_item_7_000",)
    assert output.exists()
    assert json.loads(output.read_text(encoding="utf-8").strip())["source"] == "xbrl"