"""Tests for the benchmark contract and JSONL loader."""

import json

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


def _write_xbrl_corpus(processed_dir, texts):
    (processed_dir / "AAPL").mkdir(parents=True, exist_ok=True)
    (processed_dir / "AAPL" / "filing.json").write_text(
        json.dumps({
            "ticker": "AAPL", "company": "Apple", "cik": 1, "form": "10-K",
            "filing_date": "2025-02-01", "accession_no": "0000000001-25-000001",
            "url": "https://example.com/filing", "source_path": "raw/filing.html",
            "period_of_report": "2024-12-31",
            "chunks": [
                {
                    "chunk_id": f"0000000001-25-000001_part_ii_item_8_{i:03d}",
                    "section_id": "part_ii_item_8", "part": "II", "item": "8",
                    "title": "Financial Statements", "heading": "Balance Sheet",
                    "text": text, "n_chars": len(text), "chunk_index": i,
                    "is_key_section": True, "incorporated_into": [],
                    "content_type": "table", "table_index": None, "table_caption": "",
                }
                for i, text in enumerate(texts)
            ],
        }), encoding="utf-8",
    )


def _xbrl_fact(**overrides):
    row = {
        "ticker": "AAPL", "cik": 1, "company": "Apple",
        "accession": "0000000001-25-000001", "concept": "us-gaap:Revenues",
        "label": "Revenue", "value": 100.0, "raw_value": "100", "unit": "USD",
        "scale": None, "fiscal_year": 2024, "fiscal_period": "FY",
        "period_of_report": "2024-12-31", "period_start": "2024-01-01",
        "period_end": "2024-12-31", "period_type": "duration", "statement_type": "",
        "is_audited": True, "is_current_year": True,
    }
    row.update(overrides)
    return row


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


def test_blank_labels_fall_back_to_concepts_and_keep_ids_unique(tmp_path):
    processed_dir = tmp_path / "processed"
    _write_xbrl_corpus(processed_dir, ["Revenue was 100 and tax was 250."])
    facts_file = tmp_path / "facts.parquet"
    pd.DataFrame([
        _xbrl_fact(concept="us-gaap:FdiiAmount", label="", raw_value="100"),
        _xbrl_fact(concept="us-gaap:InterestExpenseNonoperating", label="", raw_value="250"),
    ]).to_parquet(facts_file, index=False)

    questions = generate_xbrl_questions(
        facts_file, processed_dir=processed_dir, output_path=tmp_path / "generated.jsonl"
    )

    assert len(questions) == 2
    assert not any("figure" in question.question for question in questions)
    assert len({question.question_id for question in questions}) == 2


def test_same_label_in_two_units_keeps_both_rows(tmp_path):
    processed_dir = tmp_path / "processed"
    _write_xbrl_corpus(processed_dir, ["Revenue was 100 on 200 shares."])
    facts_file = tmp_path / "facts.parquet"
    pd.DataFrame([
        _xbrl_fact(raw_value="100", unit="USD"),
        _xbrl_fact(raw_value="200", unit="shares"),
    ]).to_parquet(facts_file, index=False)

    questions = generate_xbrl_questions(
        facts_file, processed_dir=processed_dir, output_path=tmp_path / "generated.jsonl"
    )

    assert len({question.question_id for question in questions}) == 2
    assert {question.expected_answer for question in questions} == {"100 USD", "200 shares"}


def test_empty_generation_leaves_existing_output_untouched(tmp_path):
    processed_dir = tmp_path / "processed"
    _write_xbrl_corpus(processed_dir, ["No figures appear in this passage."])
    facts_file = tmp_path / "facts.parquet"
    pd.DataFrame([_xbrl_fact(raw_value="999999")]).to_parquet(facts_file, index=False)
    output = tmp_path / "generated.jsonl"
    output.write_text("previous benchmark\n", encoding="utf-8")

    with pytest.raises(ValueError, match="No benchmark questions generated"):
        generate_xbrl_questions(facts_file, processed_dir=processed_dir, output_path=output)

    assert output.read_text(encoding="utf-8") == "previous benchmark\n"


# --- what the contract refuses (#49) ---------------------------------------------
#
# A benchmark file is written by hand, by six people, so each way a line can be
# wrong is refused with a message that says which field and why. Skipping a bad
# line would shrink the benchmark without anybody noticing.


@pytest.mark.parametrize("change, said", [
    ({"question_id": "  "}, "question_id must be a non-empty string"),
    ({"question": 7}, "question must be a non-empty string"),
    ({"expected_answer": ""}, "expected_answer must be a non-empty string"),
    ({"difficulty": None}, "difficulty must be a non-empty string"),
    ({"source": ""}, "source must be a non-empty string"),
    ({"ticker": ""}, "ticker must be a non-empty string"),
    ({"supporting_chunk_ids": "AAPL-2024-1"}, "supporting_chunk_ids must be a list"),
    ({"supporting_chunk_ids": ["AAPL-2024-1", ""]}, "supporting_chunk_ids must be a list"),
    ({"supporting_chunk_ids": ["AAPL-2024-1", "AAPL-2024-1"]},
     "supporting_chunk_ids must not contain duplicate values"),
    ({"hard_negative_chunk_ids": [3]}, "hard_negative_chunk_ids must be a list"),
    ({"hard_negative_chunk_ids": ["AAPL-2024-1"]},
     "supporting and hard-negative IDs overlap: AAPL-2024-1"),
    ({"fiscal_year": "2024"}, "fiscal_year must be an integer year or null"),
    ({"fiscal_year": True}, "fiscal_year must be an integer year or null"),
    ({"fiscal_year": 24}, "fiscal_year must be an integer year or null"),
    ({"reviewer": "Daryl"}, "unknown fields: reviewer"),
])
def test_each_way_a_question_breaks_the_contract_is_named(tmp_path, change, said):
    path = tmp_path / "questions.jsonl"
    _write_question(path, _question(**change))
    with pytest.raises(BenchmarkValidationError, match=said) as refusal:
        load_questions(path, chunk_ids=["AAPL-2024-1", "AAPL-2023-1"])
    assert str(refusal.value).startswith(f"{path}:1: ")


def test_a_question_missing_a_field_is_refused_with_every_field_it_lacks(tmp_path):
    path = tmp_path / "questions.jsonl"
    partial = {key: value for key, value in _question().items()
               if key not in ("difficulty", "source")}
    _write_question(path, partial)
    with pytest.raises(BenchmarkValidationError,
                       match="missing required fields: difficulty, source"):
        load_questions(path, chunk_ids=["AAPL-2024-1", "AAPL-2023-1"])


@pytest.mark.parametrize("text, said", [
    ('{"question_id": "q001",\n', r"questions\.jsonl:1: invalid JSON"),
    ('["q001"]\n', r"questions\.jsonl:1: expected a JSON object"),
    ("\n\n   \n", "benchmark contains no questions"),
])
def test_a_file_that_is_not_one_question_a_line_is_refused(tmp_path, text, said):
    path = tmp_path / "questions.jsonl"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(BenchmarkValidationError, match=said):
        load_questions(path, chunk_ids=["AAPL-2024-1"])


def test_a_benchmark_that_is_not_there_is_not_an_empty_one(tmp_path):
    with pytest.raises(FileNotFoundError, match="Benchmark questions file not found"):
        load_questions(tmp_path / "absent.jsonl", chunk_ids=[])


def test_passages_may_be_given_as_rows_as_well_as_ids(tmp_path):
    path = tmp_path / "questions.jsonl"
    _write_question(path, _question())
    rows = [{"chunk_id": "AAPL-2024-1", "text": "a"}, {"chunk_id": "AAPL-2023-1", "text": "b"}]
    assert load_questions(path, chunk_ids=rows)[0].supporting_chunk_ids == ("AAPL-2024-1",)


def test_a_result_holds_one_score_for_each_passage_and_each_passage_once():
    with pytest.raises(ValueError, match="must have equal lengths"):
        RunResult("q001", "bm25", ("a", "b"), (1.0,))
    with pytest.raises(ValueError, match="must not contain duplicates"):
        RunResult("q001", "bm25", ("a", "a"), (2.0, 1.0))


def test_a_result_is_not_built_from_another_retrievers_passages():
    from src.retrieval.records import RetrievedPassage

    passage = RetrievedPassage(
        chunk_id="a", text="a", score=1.0, rank=1, retriever="dense", ticker="AAPL",
        company="Apple", fiscal_year=2024, item="7", title="MD&A", url="https://example.com",
    )
    with pytest.raises(ValueError, match="must come from retriever 'bm25'"):
        RunResult.from_passages("q001", [passage], retriever="bm25")
    built = RunResult.from_passages("q001", [passage], retriever="dense", latency_ms=3.5,
                                    config={"id": "C2"})
    assert built.to_dict() == {
        "question_id": "q001", "retriever": "dense", "retrieved_chunk_ids": ["a"],
        "retrieved_scores": [1.0], "latency_ms": 3.5, "config": {"id": "C2"},
    }


def test_generation_needs_facts_about_the_year_a_filing_reports_on(tmp_path):
    """A comparative printed in a later filing is not that filing's own figure."""
    processed_dir = tmp_path / "processed"
    _write_xbrl_corpus(processed_dir, ["Revenue was 100."])
    facts_file = tmp_path / "facts.parquet"
    pd.DataFrame([_xbrl_fact(period_end="2023-12-31", period_start="2023-01-01")]).to_parquet(
        facts_file, index=False)

    with pytest.raises(ValueError, match="No current-year facts found"):
        generate_xbrl_questions(facts_file, processed_dir=processed_dir,
                                output_path=tmp_path / "generated.jsonl")


def test_a_fact_with_no_filing_or_no_printed_figure_makes_no_question(tmp_path):
    processed_dir = tmp_path / "processed"
    _write_xbrl_corpus(processed_dir, ["Revenue was 100 and assets were 250."])
    facts_file = tmp_path / "facts.parquet"
    pd.DataFrame([
        _xbrl_fact(raw_value="100", label="Revenue"),
        _xbrl_fact(raw_value="250", label="Assets", accession=""),      # no filing to cite
        _xbrl_fact(raw_value="777", label="Liabilities"),               # printed nowhere
    ]).to_parquet(facts_file, index=False)

    questions = generate_xbrl_questions(facts_file, processed_dir=processed_dir,
                                        output_path=tmp_path / "generated.jsonl")

    assert [question.question for question in questions] == [
        "What was Revenue for AAPL in FY2024?"]


def test_the_expected_answer_is_the_figure_as_filed_with_its_unit(tmp_path):
    processed_dir = tmp_path / "processed"
    _write_xbrl_corpus(processed_dir, ["Shares outstanding were 15,550 and revenue was 100."])
    facts_file = tmp_path / "facts.parquet"
    pd.DataFrame([
        _xbrl_fact(raw_value="100", unit="USD", label="Revenue."),
        _xbrl_fact(raw_value="15550", unit="", label="  Shares   outstanding "),
    ]).to_parquet(facts_file, index=False)

    questions = generate_xbrl_questions(facts_file, processed_dir=processed_dir,
                                        output_path=tmp_path / "generated.jsonl")

    assert {question.expected_answer for question in questions} == {"100 USD", "15550"}
    # A label is tidied into a question a person could have typed.
    assert {question.question for question in questions} == {
        "What was Revenue for AAPL in FY2024?",
        "What was Shares outstanding for AAPL in FY2024?",
    }