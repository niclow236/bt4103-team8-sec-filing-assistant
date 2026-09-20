"""Issue #32: adversarial numeric checks, saved evaluation results and visible UI."""

import json
from dataclasses import replace

import pandas as pd
import pytest

from src.app.answers import main, render_answer, write_answer_page
from src.rag import record_verification, resolve_citations, verify_answer
from src.rag.records import CitedSentence, Generation, GenerationConfig, GroundedAnswer
from src.retrieval.records import RetrievedPassage


CONFIG = GenerationConfig("ollama", "test-model", "grounded_v1")
QUESTION = "What was Apple's revenue in FY2024?"
ACCESSION = "0000320193-24-000123"


def passage(text="Revenue was $5.2 billion.", **changes):
    values = dict(chunk_id=ACCESSION + "_item8_0", text=text, score=1.0, rank=1,
                  retriever="hybrid", ticker="AAPL", company="Apple Inc.",
                  fiscal_year=2024, item="8", title="Financial Statements",
                  url="https://example.test/filing")
    return RetrievedPassage(**(values | changes))


def answer(text="Revenue was $5.2 billion.", *, passages=None, sources=(1,),
           question=QUESTION, abstained=False):
    structured = GroundedAnswer(answerable=not abstained,
                                sentences=() if abstained else (CitedSentence(text=text, sources=sources),))
    generation = Generation(text=structured.render(), answer=structured,
                            raw=structured.model_dump_json(), config=CONFIG,
                            latency_ms=1.0, input_tokens=10, output_tokens=10, stop_reason="stop")
    return resolve_citations(question, generation, [passage()] if passages is None else passages)


def fact(**changes):
    return dict(ticker="AAPL", accession=ACCESSION, fiscal_year=2024,
                concept="RevenueFromContractWithCustomerExcludingAssessedTax", unit="USD",
                value=5_200_000_000, period_start="2023-10-01", period_end="2024-09-28",
                period_of_report="2024-09-28", scale=6, **changes)


@pytest.fixture
def store(tmp_path):
    path = tmp_path / "facts.parquet"
    pd.DataFrame([fact()]).to_parquet(path, index=False)
    return path


def checks(result, kind):
    return [c for c in result.verification.checks if c.kind == kind]


@pytest.mark.parametrize("claim,source", [
    ("Revenue was $5.2 billion.", "Revenue was $5,200 million."),
    ("Revenue was $5.20 billion.", "Revenue was $5,204 million."),
    ("Revenue was $5,200,000,000.", "Revenue was $5.2 billion."),
    ("Revenue was ($5.2 billion).", "Revenue was -$5.2 billion."),
    ("Revenue was ($5.2 billion).", "Revenue was - $5.2 billion."),
    ("Revenue was $5.2 billion.", "Revenue was $5.249 billion."),
])
def test_scaled_rounded_and_signed_passage_figures(claim, source):
    result = verify_answer(answer(claim, question="Describe Apple's FY2024 results.",
                                  passages=[passage(source)]))
    assert checks(result, "passage")[0].status == "supported"


@pytest.mark.parametrize("claim,source", [
    ("Revenue was $5.2 billion.", "Revenue was $5.3 billion."),
    ("Revenue was $5.2 billion.", "Revenue was $5.2 million."),
    ("Revenue was $5.2 billion.", "Revenue was €5.2 billion."),
    ("Revenue was $5.2 billion.", "Revenue was ($5.2 billion)."),
    ("Revenue was $5.2 billion.", "Revenue was $5.250 billion."),
    ("Revenue was $5.2 billion.", "Total assets were $5.2 billion."),
    ("Revenue was $5.2 billion.", "Revenue was $4 billion. Total assets were $5.2 billion."),
])
def test_wrong_value_scale_currency_sign_and_metric_are_not_support(claim, source, store):
    result = verify_answer(answer(claim, passages=[passage(source)]), facts_file=store)
    assert checks(result, "passage")[0].status == "mismatch"
    assert result.verification_warnings


@pytest.mark.parametrize("claim,source", [
    ("The rate was 5%.", "The rate was 5 percent."),
    ("The rate was 1%.", "The rate was 100 basis points."),
    ("The rate was .5%.", "The rate was 50 basis points."),
    ("There were 2,024 employees.", "There were 2,024 employees."),
    ("There were 2024 employees.", "There were 2024 employees."),
    ("There were 2,024 stores.", "There were 2,024 stores."),
])
def test_percent_basis_points_and_year_like_counts(claim, source):
    result = verify_answer(answer(claim, passages=[passage(source)],
                                  question="Describe Apple's FY2024 business."))
    assert len(checks(result, "passage")) == 1
    assert checks(result, "passage")[0].status == "supported"


def test_years_dates_items_forms_and_citation_markers_are_not_figures():
    text = "Apple's FY2024 Form 10-K, Item 7, filed 2024-11-01, describes risks."
    result = verify_answer(answer(text, question="Describe Apple's risks in FY2024."))
    assert checks(result, "passage") == []


def test_table_scale_and_year_column(store):
    table = "Dollars in millions\n| Metric | 2024 | 2023 |\n| --- | --- | --- |\n| Revenue | 5,200 | 4,000 |"
    result = verify_answer(answer(passages=[passage(table, content_type="table")]), facts_file=store)
    assert checks(result, "passage")[0].status == "supported"
    wrong = verify_answer(answer("Revenue was $4 billion.", passages=[passage(table, content_type="table")]), facts_file=store)
    assert checks(wrong, "passage")[0].status == "mismatch"


def test_per_share_table_values_are_not_scaled_to_millions():
    text = "In millions, except per share\n| Metric | 2024 |\n| --- | --- |\n| Diluted EPS | 5.20 |"
    result = verify_answer(answer("Diluted EPS was $5.20 per share.", passages=[passage(text, content_type="table")],
                                  question="Describe Apple's FY2024 earnings."))
    assert checks(result, "passage")[0].status == "supported"


def test_table_currency_header_is_respected(store):
    text = "In millions of euros\n| Metric | 2024 |\n| --- | --- |\n| Revenue | 5,200 |"
    result = verify_answer(answer(passages=[passage(text, content_type="table")]), facts_file=store)
    assert checks(result, "passage")[0].status == "mismatch"


def test_year_like_currency_amount_is_not_a_fiscal_year(store):
    text = "Revenue was $2023."
    result = verify_answer(answer(text, passages=[passage(text)]), facts_file=store)
    assert checks(result, "passage")[0].status == "supported"


def test_year_like_table_value_does_not_replace_the_year_header(store):
    text = "| Metric | 2024 |\n| --- | --- |\n| Revenue | 2023 |"
    result = verify_answer(answer("Revenue was $2023.", passages=[passage(text, content_type="table")]), facts_file=store)
    assert checks(result, "passage")[0].status == "supported"


def test_unresolved_company_cannot_borrow_scope_from_retrieved_passages(store):
    result = verify_answer(answer(question="What was Intel's revenue in FY2024?"), facts_file=store)
    assert checks(result, "passage")[0].status == "unverified"
    assert checks(result, "fact")[0].status == "unverified"


@pytest.mark.parametrize("changes", [dict(ticker="MSFT"), dict(fiscal_year=2023)])
def test_wrong_company_or_year_passage_cannot_support(changes, store):
    result = verify_answer(answer(passages=[passage(**changes)]), facts_file=store)
    assert checks(result, "passage")[0].status == "mismatch"


def test_uncited_retrieved_match_cannot_rescue_cited_mismatch(store):
    result = verify_answer(answer(passages=[passage("Revenue was $7 billion."),
                                           passage(chunk_id="other")]), facts_file=store)
    assert checks(result, "passage")[0].status == "mismatch"


@pytest.mark.parametrize("sources", [(), (99,)])
def test_missing_and_invalid_citations_remain_visible(sources, store):
    result = verify_answer(answer(sources=sources), facts_file=store)
    assert checks(result, "groundedness")[0].status == "unverified"
    assert checks(result, "passage")[0].status == "unverified"
    assert "Citation needs review" in render_answer(result)


def test_facts_agree_and_value_is_not_scaled_twice(store):
    original = answer()
    result = verify_answer(original, facts_file=store)
    assert checks(result, "fact")[0].status == "supported"
    assert result.verification.numeric_support_rate == 1
    assert original.verification is None
    assert result.text == original.text
    assert json.loads(json.dumps(result.to_dict()))["verification"]["counts"]["supported"] == 3


@pytest.mark.parametrize("changes", [
    {"ticker": "MSFT"}, {"fiscal_year": 2023}, {"concept": "Assets"},
    {"unit": "EUR"}, {"accession": "other"}, {"period_start": "2024-07-01"},
    {"period_start": "2022-10-01", "period_end": "2023-09-28"},
    {"value": float("nan")}, {"value": float("inf")},
])
def test_unrelated_quarterly_comparative_or_invalid_facts_cannot_support(changes, tmp_path):
    path = tmp_path / "facts.parquet"
    pd.DataFrame([fact() | changes]).to_parquet(path, index=False)
    result = verify_answer(answer(), facts_file=path)
    assert checks(result, "fact")[0].status == "unverified"


def test_fact_mismatch_is_attached_to_answer_and_visible(store):
    result = verify_answer(answer("Revenue was $6 billion.",
                                  passages=[passage("Revenue was $6 billion.")]), facts_file=store)
    check = checks(result, "fact")[0]
    assert check.status == "mismatch"
    assert ACCESSION in check.evidence[0]
    assert result.verification.numeric_support_rate == 0.5
    html = render_answer(result)
    assert 'role="alert"' in html
    assert "fact: mismatch" in html
    assert "$6 billion" in html


def test_conflicting_facts_do_not_cherry_pick_the_matching_row(tmp_path):
    path = tmp_path / "facts.parquet"
    pd.DataFrame([fact(), fact() | {"value": 6_000_000_000}]).to_parquet(path, index=False)
    result = verify_answer(answer(), facts_file=path)
    assert checks(result, "fact")[0].status == "unverified"


@pytest.mark.parametrize("text", [
    "iPhone revenue was $5.2 billion.", "Revenue increased by $5.2 billion.",
    "Quarterly revenue was $5.2 billion.",
])
def test_segment_derived_or_quarterly_claims_cannot_use_the_annual_total(text, store):
    result = verify_answer(answer(text), facts_file=store)
    assert checks(result, "fact")[0].status == "unverified"


@pytest.mark.parametrize("contents", [None, "not parquet"])
def test_missing_or_corrupt_store_is_visible_and_unscored_as_support(contents, tmp_path):
    path = tmp_path / "facts.parquet"
    if contents:
        path.write_text(contents)
    result = verify_answer(answer(), facts_file=path)
    assert checks(result, "fact")[0].status == "unverified"
    assert "Facts store unavailable" in render_answer(result)
    assert result.verification.numeric_support_rate == 0.5


def test_nonnumeric_question_checks_figures_without_loading_facts(monkeypatch):
    def forbidden(*args):
        pytest.fail("Facts must not be read for a nonnumeric question")
    monkeypatch.setattr("src.rag.verify.load_facts", forbidden)
    result = verify_answer(answer(question="Describe Apple's FY2024 results."))
    assert checks(result, "passage")
    assert not checks(result, "fact")


def test_paraphrase_is_not_claimed_to_be_semantically_verified(store):
    result = verify_answer(answer("Revenue increased to $5.2 billion."), facts_file=store)
    assert checks(result, "passage")[0].status == "supported"
    assert checks(result, "groundedness")[0].status == "unverified"


def test_negative_sign_is_preserved_for_extractive_grounding(store):
    result = verify_answer(answer(passages=[passage("Revenue was -$5.2 billion.")]), facts_file=store)
    assert checks(result, "groundedness")[0].status == "unverified"


def test_ambiguous_multi_company_scope_is_not_a_pass(store):
    result = verify_answer(answer("Apple and Microsoft had revenue of $5.2 billion.",
                                  question="What were Apple and Microsoft revenues in FY2024?"), facts_file=store)
    assert checks(result, "fact")[0].status == "unverified"
    assert checks(result, "passage")[0].status == "unverified"


def test_abstention_is_not_a_perfect_faithfulness_score(tmp_path):
    result = verify_answer(answer(abstained=True), facts_file=tmp_path / "absent")
    assert result.verification.numeric_support_rate is None
    assert result.verification_warnings == ()
    assert [c.status for c in result.verification.checks] == ["not_applicable"]


def test_malformed_output_and_absent_sentence_records_are_flagged(store):
    result = verify_answer(replace(answer(), sentences=(), parse_error="invalid JSON", truncated=True), facts_file=store)
    assert checks(result, "output")[0].status == "unverified"
    assert checks(result, "groundedness")[0].status == "unverified"
    assert "incomplete or malformed" in render_answer(result)


def test_numeric_question_without_extractable_figures_is_unverified(store):
    result = verify_answer(answer("Revenue was five billion dollars."), facts_file=store)
    assert checks(result, "fact")[0].status == "unverified"


def test_jsonl_records_preserve_question_config_evidence_and_denominators(store, tmp_path):
    path = tmp_path / "results" / "checks.jsonl"
    result = verify_answer(answer(), facts_file=store)
    for run in ("baseline", "proposed"):
        record_verification(result, path, question_id="q01", run_id=run)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [row["run_id"] for row in rows] == ["baseline", "proposed"]
    assert rows[0]["question_id"] == "q01"
    assert rows[0]["answer"]["config"]["model"] == "test-model"
    assert rows[0]["answer"]["verification"]["counts"]["supported"] == 3
    assert rows[0]["answer"]["verification"]["checks"][-1]["evidence"]
    output = tmp_path / "answers.html"
    main([str(path), "--output", str(output)])
    assert output.read_text(encoding="utf-8").count("<article>") == 2


def test_recording_requires_verification_and_question_identity(store, tmp_path):
    with pytest.raises(ValueError, match="Verify"):
        record_verification(answer(), tmp_path / "results", question_id="q", run_id="run")
    with pytest.raises(ValueError, match="non-empty"):
        record_verification(verify_answer(answer(), facts_file=store), tmp_path / "results",
                            question_id="", run_id="run")


def test_viewer_escapes_model_content_and_rejects_script_urls(tmp_path):
    result = answer('<script>alert("x")</script>',
                    passages=[passage(url="javascript:alert(1)")])
    html = render_answer(result)
    assert "has not been verified" in html
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert "javascript:" not in html
    path = write_answer_page([result], tmp_path / "answers.html")
    assert path.exists()


def test_viewer_rejects_invalid_records_with_line_number(tmp_path, capsys):
    path = tmp_path / "results.jsonl"
    path.write_text('{"wrong": "shape"}\n')
    with pytest.raises(SystemExit):
        main([str(path)])
    assert "line 1" in capsys.readouterr().err
