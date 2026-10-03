"""Issue #32: adversarial numeric checks, saved evaluation results and visible UI."""

import json
from dataclasses import asdict, replace

import pandas as pd
import pytest

from src.app.answers import main, render_answer, write_answer_page
from src.rag import resolve_citations, verify_answer, worst_check
from src.rag.records import (
    CitedSentence,
    Generation,
    GenerationConfig,
    GroundedAnswer,
    VerificationCheck,
)
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
    monkeypatch.setattr("src.rag.verify.cached_facts", forbidden)
    result = verify_answer(answer(question="Describe Apple's FY2024 results."))
    assert checks(result, "passage")
    assert not checks(result, "fact")


def test_the_store_is_read_once_for_a_run_of_answers(store, monkeypatch):
    # The evaluation harness checks every answer of a run. Read per answer,
    # the whole store was loaded once for each of them.
    from src.rag import numeric

    numeric.cached_facts.cache_clear()
    reads = []
    real = numeric.load_facts
    monkeypatch.setattr(numeric, "load_facts", lambda path: reads.append(path) or real(path))
    results = [verify_answer(answer(), facts_file=store) for _ in range(3)]
    assert [checks(result, "fact")[0].status for result in results] == ["supported"] * 3
    assert len(reads) == 1
    numeric.cached_facts.cache_clear()


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


def test_an_untrimmed_question_still_verifies(store):
    result = verify_answer(answer(question=QUESTION + " "), facts_file=store)
    assert result.verification is not None


def test_verification_uses_custom_labels_but_sentence_scope_does_not(store, monkeypatch):
    import src.rag.query as query_module

    pd.DataFrame([fact() | {"label": "Marketable Securities"}]).to_parquet(store)
    seen = []
    read = query_module._mentions_fact_label

    def tracked(text, path):
        seen.append(path)
        return read(text, path)

    monkeypatch.setattr(query_module, "_mentions_fact_label", tracked)
    question = "What was Apple's Marketable Securities in FY2024?"
    structured = GroundedAnswer(answerable=True, sentences=(
        CitedSentence(text="Apple held securities of $5.2 billion.", sources=(1,)),
        CitedSentence(text="Apple held securities of $5.2 billion in FY2024.", sources=(1,)),
    ))
    generation = Generation(text=structured.render(), answer=structured, raw=structured.model_dump_json(),
                            config=CONFIG, latency_ms=1.0, input_tokens=10, output_tokens=10,
                            stop_reason="stop")
    original = resolve_citations(question, generation, [passage()])
    result = verify_answer(original, facts_file=store)
    assert result.verification.question_type == "numeric"
    assert len(checks(result, "fact")) == 2
    assert seen == [store]


@pytest.mark.parametrize("question, label", [
    ("What did Apple say about risks to its assets?", "Assets"),
    ("What is Apple's commercial paper program?", "Commercial Paper"),
    ("What does Apple say about inventories management?", "Inventories"),
])
def test_label_topics_do_not_create_spurious_fact_checks(question, label, store):
    pd.DataFrame([fact() | {"label": label}]).to_parquet(store)
    claim = "The company describes its financing policies."
    result = verify_answer(answer(claim, question=question, passages=[passage(claim)]), facts_file=store)
    assert result.verification.question_type == "factual"
    assert checks(result, "fact") == []


@pytest.mark.parametrize("item", [
    "inventory purchase obligations", "inventories purchase obligations",
    "inventory reserves", "inventories reserves", "inventory write-downs", "inventories write-downs",
    "write-downs of inventories",
])
@pytest.mark.parametrize("value", [5_200_000_000, 30_000_000_000])
def test_qualified_inventory_figures_never_use_inventory_net(item, value, store):
    # A different balance must neither contradict nor validate this claim.
    pd.DataFrame([fact() | {"concept": "InventoryNet", "label": "Inventories", "value": value}]).to_parquet(store)
    claim = f"Apple's {item} were $30 billion."
    result = verify_answer(answer(claim, question="What was Apple's inventories value in FY2024?",
                                  passages=[passage(claim)]), facts_file=store)
    check, = checks(result, "fact")
    assert check.status == "unverified"
    assert "qualified line item" in check.reason


def test_unqualified_inventories_still_verify_against_inventory_net(store):
    pd.DataFrame([fact() | {"concept": "InventoryNet", "label": "Inventories"}]).to_parquet(store)
    claim = "Apple's inventories were $5.2 billion."
    result = verify_answer(answer(claim, question="What was Apple's inventories value in FY2024?",
                                  passages=[passage(claim)]), facts_file=store)
    assert checks(result, "fact")[0].status == "supported"


def test_a_table_without_a_declared_scale_is_unverified_not_mismatch(store):
    table = passage("| Total assets | $364,980 | $352,583 |", content_type="table")
    scaled = passage("In millions" + chr(10) * 2 + "| Total assets | $364,980 | $352,583 |",
                     content_type="table")
    claim = "Apple's total assets were $364.98 billion."
    unscaled_checks = checks(verify_answer(answer(claim, passages=[table]), facts_file=store), "passage")
    assert [c.status for c in unscaled_checks] == ["unverified"]
    assert "declares no scale" in unscaled_checks[0].reason
    assert [c.status for c in checks(
        verify_answer(answer(claim, passages=[scaled]), facts_file=store), "passage")] == ["supported"]


@pytest.mark.parametrize("found, word", [
    # The two checks of one figure are companions: the store confirming what a
    # table with no scale cannot is support, as for a facts-route answer.
    ([("passage", "unverified", 0, "$5.2 billion"), ("fact", "supported", 0, "$5.2 billion")],
     "supported"),
    # A supported figure does not carry one that could not be checked, in
    # another sentence or beside it in the same one.
    ([("fact", "supported", 0, "$5.2 billion"), ("passage", "unverified", 1, "$4.0 billion")],
     "unverified"),
    ([("fact", "supported", 0, "$6.08"), ("fact", "unverified", 0, "$94 billion")],
     "unverified"),
    # One mismatch marks the answer whatever else agrees with it.
    ([("fact", "supported", 0, "$5.2 billion"), ("passage", "mismatch", 0, "$5.2 billion"),
      ("fact", "unverified", 1, "$4.0 billion")], "mismatch"),
    # A numeric question answered with no figure at all gets one check, on no figure.
    ([("fact", "unverified", None, None)], "unverified"),
    # Checks that are not about a figure do not count, and neither does nothing.
    ([("groundedness", "unverified", 0, None), ("output", "unverified", None, None)],
     "unchecked"),
    ([], "unchecked"),
])
def test_an_answer_is_summed_up_by_its_worst_figure(found, word):
    # One definition for the evaluation harness, which counts a run's answers
    # by this word, and the measurement scripts, which record it per answer.
    records = [VerificationCheck(kind, status, sentence, "the claim", "the reason", figure)
               for kind, status, sentence, figure in found]
    assert worst_check(records) == word
    # As the dicts a saved row holds, which is what the harness counts from.
    assert worst_check(asdict(record) for record in records) == word


def test_a_supported_sentence_does_not_hide_an_unverified_one(store):
    # The first sentence is supported by the passage and the store. The second
    # names two years, so the checker cannot say which the figure belongs to.
    structured = GroundedAnswer(answerable=True, sentences=(
        CitedSentence(text="Revenue was $5.2 billion.", sources=(1,)),
        CitedSentence(text="Revenue was $4.0 billion in 2023 and 2024.", sources=(1,)),
    ))
    generation = Generation(text=structured.render(), answer=structured,
                            raw=structured.model_dump_json(), config=CONFIG, latency_ms=1.0,
                            input_tokens=10, output_tokens=10, stop_reason="stop")
    result = verify_answer(resolve_citations(QUESTION, generation, [passage()]), facts_file=store)
    assert {check.sentence_index: check.status for check in checks(result, "passage")} == {
        0: "supported", 1: "unverified"}
    assert worst_check(result.verification.checks) == "unverified"


# --- per-share amounts ----------------------------------------------------------

EPS_TABLE = ("(in millions, except per share data)\n\n"
             "| (in millions, except per share data) | 2024 | 2023 |\n| --- | --- | --- |\n"
             "| Net income | $13,746 | $10,135 |\n"
             "| Basic earnings per share | $4.67 | $3.16 |\n"
             "| Diluted earnings per share | $4.55 | $3.08 |")


def test_a_dollar_amount_in_a_per_share_row_is_dollars_a_share():
    # Oracle prints "$4.55" in the row. The sign names the currency; it does
    # not make the row whole dollars, which a claim of $4.55 a share is not.
    cited = [passage(EPS_TABLE, content_type="table")]
    result = verify_answer(answer("Diluted earnings per share were $4.55 per share.",
                                  passages=cited, question="Describe Apple's FY2024 earnings."))
    assert checks(result, "passage")[0].status == "supported"
    basic_for_diluted = verify_answer(answer("Diluted earnings per share were $4.67 per share.",
                                             passages=cited,
                                             question="Describe Apple's FY2024 earnings."))
    assert checks(basic_for_diluted, "passage")[0].status == "mismatch"


@pytest.mark.parametrize("row", ["| Diluted EPS | 5.20 |", "| Diluted EPS | $5.20 |"])
def test_a_per_share_claim_is_per_share_whether_or_not_it_says_so(row):
    text = "In millions, except per share\n| Metric | 2024 |\n| --- | --- |\n" + row
    result = verify_answer(answer("Diluted EPS was $5.20.",
                                  passages=[passage(text, content_type="table")],
                                  question="Describe Apple's FY2024 earnings."))
    assert checks(result, "passage")[0].status == "supported"


@pytest.mark.parametrize("claim, status", [
    ("Diluted EPS was $5.20.", "supported"),
    ("Diluted EPS was $5.20 per share.", "supported"),
    ("Diluted EPS was $6.20.", "mismatch"),
])
def test_a_per_share_claim_is_checked_against_the_per_share_fact(claim, status, tmp_path):
    path = tmp_path / "facts.parquet"
    pd.DataFrame([fact() | {"concept": "EarningsPerShareDiluted", "unit": "USD per share",
                            "value": 5.2}]).to_parquet(path, index=False)
    result = verify_answer(answer(claim, question="What was Apple's diluted EPS in FY2024?",
                                  passages=[passage(claim)]), facts_file=path)
    assert checks(result, "fact")[0].status == status


def test_a_per_share_row_in_the_filer_s_words_is_read_as_per_share():
    # Adobe: "Diluted net income per share", with the dollar sign in a cell of
    # its own. Read as a net income row, it was skipped for a per-share claim.
    text = ("(in millions, except per share data)\n\n"
            "| (in millions, except per share data) | 2024 | 2024 | 2023 | 2023 |\n"
            "| --- | --- | --- | --- | --- |\n"
            "| Net income | $ | 5,560 | $ | 5,428 |\n"
            "| Diluted net income per share | $ | 12.36 | $ | 11.82 |")
    cited = [passage(text, content_type="table")]
    result = verify_answer(answer("Diluted earnings per share were $12.36 per share.",
                                  passages=cited, question="Describe Apple's FY2024 earnings."))
    assert checks(result, "passage")[0].status == "supported"
    last_year = verify_answer(answer("Diluted earnings per share were $11.82 per share.",
                                     passages=cited,
                                     question="Describe Apple's FY2024 earnings."))
    assert checks(last_year, "passage")[0].status == "mismatch"


def test_a_dollar_total_beside_a_per_share_figure_is_not_dollars_a_share(tmp_path):
    # Every "$" figure in a per-share claim was read as dollars a share, so the
    # "$94 billion" here was checked against the store's earnings per share
    # and a true sentence was marked a mismatch.
    path = tmp_path / "facts.parquet"
    pd.DataFrame([fact() | {"concept": "EarningsPerShareDiluted", "unit": "USD per share",
                            "value": 6.08}]).to_parquet(path, index=False)
    claim = "Apple's diluted EPS was $6.08 in FY2024, and it returned $94 billion to shareholders."
    result = verify_answer(answer(claim, question="What was Apple's diluted EPS in FY2024?",
                                  passages=[passage(claim)]), facts_file=path)
    assert [(check.unit, check.status) for check in checks(result, "fact")] == [
        ("USD/shares", "supported"), ("USD", "unverified")]


ADOBE_TABLE = ("(in millions, except per share data)\n\n"
               "| (in millions, except per share data) | 2024 | 2024 | 2023 | 2023 |\n"
               "| --- | --- | --- | --- | --- |\n"
               "| Net income | $ | 5,560 | $ | 5,428 |\n"
               "| Diluted net income per share | $ | 12.36 | $ | 11.82 |")


@pytest.fixture
def adobe_store(tmp_path):
    path = tmp_path / "facts.parquet"
    pd.DataFrame([
        fact() | {"concept": "EarningsPerShareDiluted", "unit": "USD per share", "value": 12.36},
        fact() | {"concept": "NetIncomeLoss", "value": 5_560_000_000},
    ]).to_parquet(path, index=False)
    return path


@pytest.mark.parametrize("question, status", [
    # Asked for earnings per share, the claim's figure is that line item's.
    ("What was Apple's diluted EPS in FY2024?", "supported"),
    # Asked in the claim's own words, or for net income, nothing says which
    # per-share figure it is, so there is nothing to compare it with.
    ("What was Apple's net income per share in FY2024?", "unverified"),
    ("What was Apple's net income in FY2024?", "unverified"),
])
def test_net_income_per_share_is_not_checked_as_net_income(question, status, adobe_store):
    # Four filers call earnings per share "net income per share". Without
    # "basic" or "diluted" it names neither, and it does not name net income:
    # read as net income, a right answer in those words was checked against
    # the store's $5,560 million and marked a mismatch.
    claim = "Apple's net income per share was $12.36 in fiscal year 2024."
    result = verify_answer(answer(claim, question=question,
                                  passages=[passage(ADOBE_TABLE, content_type="table")]),
                           facts_file=adobe_store)
    assert [check.status for check in checks(result, "passage")] == [status]
    assert [check.status for check in checks(result, "fact")] == [status]


def test_net_income_stated_beside_its_per_share_figure_is_still_checked(adobe_store):
    claim = "Apple's net income was $5,560 million in fiscal year 2024."
    result = verify_answer(
        answer(claim, question="What was Apple's net income per share in FY2024?",
               passages=[passage(ADOBE_TABLE, content_type="table")]),
        facts_file=adobe_store)
    assert [check.status for check in checks(result, "fact")] == ["supported"]


def test_a_per_share_row_that_does_not_say_which_is_not_another_line_item_s(adobe_store):
    # ServiceNow, Broadcom and Palo Alto print "Net income per share - diluted".
    # Read as a net income row it was another line item's, so a right answer
    # the store confirms was a mismatch against the row that prints it. It
    # names no line item the checker knows, which leaves the claim unverified.
    text = ("(in millions, except per share data)\n\n| | 2024 | 2023 |\n| --- | --- | --- |\n"
            "| Net income | $5,560 | $5,428 |\n"
            "| Net income per share - diluted | $12.36 | $11.82 |")
    result = verify_answer(answer("Diluted earnings per share were $12.36 in fiscal year 2024.",
                                  question="What was Apple's diluted EPS in FY2024?",
                                  passages=[passage(text, content_type="table")]),
                           facts_file=adobe_store)
    check, = checks(result, "passage")
    assert check.status == "unverified"
    assert "does not name the line item" in check.reason


# --- printed in the passage, and not tied to the line item or the year -----------

# Texas Instruments' income statement runs over several passages, and the one
# that prints earnings per share begins after its heading: the rows say only
# "Basic" and "Diluted".
UNNAMED_ROWS = ("Consolidated Statements of Income (In millions, except per-share amounts)\n\n"
                "| For Years Ended December 31, | 2024 | 2023 |\n| --- | --- | --- |\n"
                "| Basic | $5.24 | $7.13 |\n| Diluted | $5.20 | $7.07 |\n"
                "| Net income | $4,799 | $6,510 |")
EPS_QUESTION = "What was Apple's diluted EPS in FY2024?"


@pytest.fixture
def eps_store(tmp_path):
    path = tmp_path / "eps.parquet"
    pd.DataFrame([fact() | {"concept": "EarningsPerShareDiluted", "unit": "USD per share",
                            "value": 5.2}]).to_parquet(path, index=False)
    return path


@pytest.mark.parametrize("claim, status", [
    # Printed in a row that does not say what it is, and the store says it is
    # diluted earnings per share: not support from the passage, and not a mismatch.
    ("Diluted earnings per share were $5.20 per share.", "unverified"),
    # Not printed anywhere, and not what the store holds.
    ("Diluted earnings per share were $6.20 per share.", "mismatch"),
    # Printed in the other year's column.
    ("Diluted earnings per share were $7.07 per share.", "mismatch"),
    # Printed only in a row that names another line item.
    ("Diluted earnings per share were $4,799 per share.", "mismatch"),
])
def test_a_confirmed_figure_in_a_row_that_names_no_line_item_is_not_a_mismatch(
        claim, status, eps_store):
    result = verify_answer(answer(claim, question=EPS_QUESTION,
                                  passages=[passage(UNNAMED_ROWS, content_type="table")]),
                           facts_file=eps_store)
    check, = checks(result, "passage")
    assert check.status == status
    if status == "unverified":
        assert "does not name the line item" in check.reason


def test_an_unnamed_row_is_still_a_mismatch_where_no_store_confirms_the_figure():
    # A question that is not numeric gets no fact check, so nothing says the
    # figure in the row labelled "Diluted" is earnings per share.
    result = verify_answer(answer("Diluted earnings per share were $5.20 per share.",
                                  question="Describe Apple's FY2024 earnings.",
                                  passages=[passage(UNNAMED_ROWS, content_type="table")]))
    assert checks(result, "passage")[0].status == "mismatch"


TWO_YEARS = "Revenue totaled $5.2 billion and $4.0 billion in 2024 and 2023, respectively."


def test_a_confirmed_figure_in_a_sentence_naming_two_years_is_not_a_mismatch(store):
    # The checker does not work out which figure belongs to which year, so the
    # sentence cannot support the claim. The store says $5.2 billion is FY2024's.
    check, = checks(verify_answer(answer(passages=[passage(TWO_YEARS)]), facts_file=store),
                    "passage")
    assert check.status == "unverified"
    assert "more than one year" in check.reason


@pytest.mark.parametrize("claim, changes, question", [
    # A figure the sentence does not hold.
    ("Revenue was $6 billion.", {}, QUESTION),
    # The other year's figure: the store holds $5.2 billion for FY2024.
    ("Revenue was $4.0 billion.", {}, QUESTION),
    # The sentence is in another year's filing.
    ("Revenue was $5.2 billion.", {"fiscal_year": 2023}, QUESTION),
    # No fact check, so nothing confirms which year the figure is.
    ("Revenue was $5.2 billion.", {}, "Describe Apple's FY2024 results."),
])
def test_a_two_year_sentence_is_still_a_mismatch_without_that_confirmation(
        claim, changes, question, store):
    result = verify_answer(answer(claim, question=question,
                                  passages=[passage(TWO_YEARS, **changes)]), facts_file=store)
    assert checks(result, "passage")[0].status == "mismatch"


def test_a_single_year_sentence_still_supports_beside_a_two_year_one(store):
    source = TWO_YEARS + " Revenue was $5.2 billion in 2024."
    result = verify_answer(answer(passages=[passage(source)]), facts_file=store)
    assert checks(result, "passage")[0].status == "supported"


# --- the day in a date ------------------------------------------------------------

@pytest.mark.parametrize("claim", [
    "Revenue was $5.2 billion as of September 28, 2024.",
    "Revenue was $5.2 billion for the year ended Sept. 28, 2024.",
    "Revenue was $5.2 billion at 28 September 2024.",
    "Revenue was $5.2 billion as of December 31.",
    # In any case: a model writes lower case, and a table heading capitals.
    "Revenue was $5.2 billion as of september 28, 2024.",
    "Revenue was $5.2 billion as of SEPTEMBER 28, 2024.",
    "Revenue was $5.2 billion for the year ended sept. 28, 2024.",
    "Revenue was $5.2 billion at 28 sep 2024.",
    "Revenue was $5.2 billion as of 28TH SEPTEMBER 2024.",
    "Revenue was $5.2 billion as of MAY 31.",
])
def test_the_day_of_a_written_date_is_not_a_figure(claim, store):
    # Read as one, the 28 was checked against FY2024 revenue, and an answer
    # the store had just confirmed was marked a mismatch with it.
    result = verify_answer(answer(claim, passages=[passage("Revenue was $5.2 billion.")]),
                           facts_file=store)
    assert [check.figure for check in checks(result, "fact")] == ["$5.2 billion"]
    assert [check.status for check in checks(result, "fact")] == ["supported"]
    assert [check.status for check in checks(result, "passage")] == ["supported"]


def test_may_in_lower_case_is_the_verb_and_not_a_month(store):
    # Matched in any case like the other months, "10 may" would be the tenth
    # of May, and a number in front of the verb would be hidden from the
    # checker. It is read as a figure, as it was before.
    result = verify_answer(answer("Revenue was $5.2 billion, of which up to 10 may be deferred.",
                                  passages=[passage("Revenue was $5.2 billion.")]),
                           facts_file=store)
    assert [check.figure for check in checks(result, "fact")] == ["$5.2 billion", "10"]


def test_a_date_does_not_hide_the_figures_around_it(store):
    result = verify_answer(answer("On September 28, 2024, revenue was $6 billion, up 5%.",
                                  passages=[passage("Revenue was $5.2 billion.")]),
                           facts_file=store)
    assert [check.figure for check in checks(result, "fact")] == ["$6 billion", "5%"]
    assert checks(result, "fact")[0].status == "mismatch"


def test_the_year_of_a_written_date_still_scopes_the_claim(store):
    # A claim dated in another fiscal year is outside the question's scope.
    result = verify_answer(answer("Revenue was $5.2 billion as of September 30, 2023.",
                                  passages=[passage("Revenue was $5.2 billion.")]),
                           facts_file=store)
    assert checks(result, "passage")[0].status == "unverified"
