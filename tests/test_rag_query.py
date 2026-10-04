"""Query understanding: what it extracts, what it degrades on, and how it classifies."""

import dataclasses

import pytest

from src.rag.query import ParsedQuestion, parse_question
from src.retrieval.constants import TABLE_BOOST
from src.retrieval.records import Query

# Keep classification independent of the local corpus and built facts store.
SCOPE = ("AAPL", "MSFT", "AVGO", "GOOGL", "META", "AMZN", "ORCL", "CRM", "ADBE",
         "CSCO", "TXN", "MU", "INTU", "NOW", "PANW")
YEARS = (2021, 2025)


def _parse(question: str, **overrides) -> ParsedQuestion:
    options = dict(known_tickers=SCOPE, fiscal_years=YEARS, facts_file=None)
    options.update(overrides)
    return parse_question(question, **options)


@pytest.mark.parametrize("question, ticker, year", [
    ("What was Apple's total value of Accounts Payable at the end of fiscal year 2022?",
     "AAPL", 2022),
    ("What was Apple's Inventories value at the end of fiscal year 2025?", "AAPL", 2025),
    ("What was Amazon's total net sales in fiscal year 2025?", "AMZN", 2025),
])
def test_issue_89_figure_questions(question, ticker, year, tmp_path):
    # Everyday metric aliases work even before a teammate builds the store.
    parsed = _parse(question, facts_file=tmp_path / "missing.parquet")
    assert parsed.question_type == "numeric"
    assert parsed.wants_figures is True
    assert parsed.tickers == (ticker,)
    assert parsed.fiscal_years == (year,)
    assert parsed.to_query().table_boost == TABLE_BOOST


@pytest.mark.parametrize("line_item", [
    "accounts receivable", "income tax expense", "provision for income taxes",
    "income before income taxes", "stock-based compensation", "share-based compensation",
    "sales and marketing expense", "property and equipment", "property, plant and equipment",
    "intangible assets", "capital expenditures", "purchases of property and equipment",
    "interest paid",
])
def test_a_plain_question_about_a_statement_line_asks_for_a_figure(line_item):
    # None of these was a cue, and no label in the facts store is worded this
    # way, so the question was read as prose and searched with no lean toward
    # tables. Labels are off here: these are read without the store.
    parsed = _parse(f"What was Microsoft's {line_item} in fiscal year 2024?")
    assert parsed.question_type == "numeric"
    assert parsed.wants_figures is True
    assert parsed.to_query().table_boost == TABLE_BOOST


@pytest.mark.parametrize("question", [
    "How does Microsoft manage accounts receivable risk?",
    "What does Apple say about its capital expenditures plans for data centers?",
    "How does Meta account for stock-based compensation?",
    "Why did Amazon's provision for income taxes matter to its strategy?",
    "How does Adobe amortize its intangible assets?",
    # Left out on purpose: the cash paid and the amount bought under the
    # programme are two figures, and the question does not say which.
    "What were Cisco's share repurchases in fiscal year 2024?",
])
def test_a_statement_line_named_in_a_prose_question_stays_prose(question):
    parsed = _parse(question)
    assert parsed.wants_figures is False
    assert parsed.to_query().table_boost == 1.0


def test_inventory_risk_stays_a_prose_question(tmp_path):
    # Q32 of the test questions names inventory without asking for a figure.
    question = ("What factors did Amazon identify as creating significant inventory risk "
                "in fiscal year 2022?")
    parsed = _parse(question, facts_file=tmp_path / "missing.parquet")
    assert parsed.question_type == "factual"


@pytest.fixture
def broad_label_store(tmp_path):
    import pandas as pd

    path = tmp_path / "broad.parquet"
    # Short labels observed by both reviewers in their real stores.
    pd.DataFrame({"label": ["Assets", "Liabilities", "Depreciation", "Goodwill",
                            "Commercial Paper", "Lease, Cost", "Marketing Expense",
                            "Investments", "Cash", "Inventory", "Inventories",
                            "Accounts Payable"]}).to_parquet(path)
    return path


@pytest.mark.parametrize("question", [
    "What did Apple say about risks to its assets?",
    "How does Microsoft account for depreciation?",
    "What is Apple's commercial paper program?",
    "Why did Apple's liabilities matter to its strategy?",
    "What does Microsoft say about its investments in AI in FY2024?",
    "How does Apple manage cash and liquidity in FY2024?",
    "What does Meta say about depreciation of its servers in FY2023?",
    "What factors did Amazon identify as creating significant inventory risk in fiscal year 2022?",
    "What does Apple say about inventory management?",
    "What does Apple say about inventories management?",
    "How does Apple manage accounts payable?",
    "What was the rationale behind Apple's total restructuring?",
])
def test_review_prose_questions_stay_factual_with_or_without_labels(question, broad_label_store):
    for path in (None, broad_label_store):
        parsed = _parse(question, facts_file=path)
        assert parsed.question_type == "factual"
        assert parsed.wants_figures is False
        assert parsed.to_query().table_boost == 1.0


@pytest.mark.parametrize("label", ["Assets", "Liabilities", "Depreciation", "Commercial Paper",
                                  "Lease Cost", "Marketing Expense", "Investments", "Cash"])
def test_short_labels_still_recognise_requests_for_balances(label, broad_label_store):
    parsed = _parse(f"What was Apple's {label} in FY2024?", facts_file=broad_label_store)
    assert parsed.question_type == "numeric"
    assert parsed.wants_figures is True


def test_total_frame_cannot_swallow_arbitrary_words():
    from src.rag.query import _ASKS_TOTAL, _figure_text

    for question in ("What was the rationale behind Apple's total restructuring?",
                     "What is the reason for Apple's total debt increase?"):
        assert not _ASKS_TOTAL.search(_figure_text(question, frozenset(SCOPE)))


def test_disabled_labels_never_touch_the_store(monkeypatch):
    monkeypatch.setattr("src.rag.query._mentions_fact_label",
                        lambda *a: pytest.fail("unexpected facts-store lookup"))
    assert _parse("Apple's Marketable Securities").question_type == "factual"


@pytest.mark.parametrize("owner", [
    "Apple's", "Amazon’s", "Meta Platforms'", "Texas Instruments’", "ZZZZ's", "the",
])
def test_total_with_a_company_between_the_verb_and_total(owner, tmp_path):
    parsed = _parse(f"What was {owner} total expenditure in FY2024?",
                    known_tickers=(*SCOPE, "ZZZZ"), facts_file=tmp_path / "missing.parquet")
    assert parsed.question_type == "numeric"
    assert parsed.wants_figures is True


@pytest.fixture
def label_store(tmp_path):
    import pandas as pd

    path = tmp_path / "facts.parquet"
    pd.DataFrame({"label": ["Prepaid Expense, Current", "Assets Held for Sale",
                            "Prepaid Expense, Current", None, ""]}).to_parquet(path)
    return path


@pytest.mark.parametrize("label", ["Prepaid Expense, Current", "PREPAID expense current",
                                  "Prepaid\nExpense — Current", "Assets Held for Sale"])
def test_stored_labels_extend_numeric_cues_without_a_handwritten_entry(label, label_store):
    parsed = _parse(f"What was Apple's {label} in FY2024?", facts_file=label_store)
    assert parsed.question_type == "numeric"
    assert parsed.wants_figures is True
    assert parsed.to_query().table_boost == TABLE_BOOST


@pytest.mark.parametrize("question", [
    "What did Apple say about its sale process?",
    "What was Apple's prepaid expense currently used for?",
    "What was Apple's totality of responses?",
])
def test_label_fragments_and_longer_words_do_not_become_cues(question, label_store):
    parsed = _parse(question, facts_file=label_store)
    assert parsed.question_type == "factual"
    assert parsed.wants_figures is False


@pytest.mark.parametrize("question, expected", [
    ("Compare Apple and Microsoft's Prepaid Expense, Current in FY2024", "comparative"),
    ("Apple's Prepaid Expense, Current in FY2023 and FY2024", "temporal"),
    ("What will Apple's Prepaid Expense, Current be next year?", "unanswerable"),
    ("What was Intel's Prepaid Expense, Current in FY2024?", "unanswerable"),
])
def test_stored_labels_preserve_classification_priority(question, expected, label_store):
    parsed = _parse(question, facts_file=label_store)
    assert parsed.question_type == expected
    assert parsed.wants_figures is True


def test_label_cache_reads_only_labels_and_refreshes_after_rebuild(label_store, monkeypatch):
    import os
    import pandas as pd
    from src.rag.query import _fact_label_pattern

    _fact_label_pattern.cache_clear()
    read = pd.read_parquet
    calls = []

    def tracked(*args, **kwargs):
        calls.append(kwargs)
        return read(*args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", tracked)
    question = "What was Apple's Prepaid Expense, Current?"
    assert _parse(question, facts_file=label_store).wants_figures
    assert _parse(question, facts_file=label_store).wants_figures
    assert calls == [{"columns": ["label"]}]
    stamp = label_store.stat().st_mtime_ns
    pd.DataFrame({"label": ["Marketable Securities"]}).to_parquet(label_store)
    os.utime(label_store, ns=(stamp + 1_000_000_000, stamp + 1_000_000_000))
    assert not _parse(question, facts_file=label_store).wants_figures
    assert _parse("Apple's Marketable Securities", facts_file=label_store).wants_figures
    assert len(calls) == 2


def test_failed_label_reads_are_retried_without_a_file_change(label_store, monkeypatch):
    import pandas as pd
    from src.rag.query import _fact_label_pattern

    _fact_label_pattern.cache_clear()
    read = pd.read_parquet
    calls = []

    def transient_failure(*args, **kwargs):
        calls.append(args)
        if len(calls) == 1:
            raise OSError("store being replaced")
        return read(*args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", transient_failure)
    question = "What was Apple's Prepaid Expense, Current?"
    assert not _parse(question, facts_file=label_store).wants_figures
    assert _parse(question, facts_file=label_store).wants_figures
    assert len(calls) == 2


@pytest.mark.parametrize("state", ["missing", "corrupt", "no_labels", "empty"])
def test_unavailable_labels_leave_builtin_classification_working(tmp_path, state):
    import pandas as pd

    path = tmp_path / "facts.parquet"
    if state == "corrupt":
        path.write_bytes(b"not parquet")
    elif state == "no_labels":
        pd.DataFrame({"concept": ["Assets"]}).to_parquet(path)
    elif state == "empty":
        pd.DataFrame({"label": [None, "", "   "]}).to_parquet(path)
    assert _parse("Apple's revenue", facts_file=path).question_type == "numeric"
    assert _parse("Apple's supply chain", facts_file=path).question_type == "factual"
    # A store built after an earlier miss must become available in this process.
    pd.DataFrame({"label": ["Marketable Securities"]}).to_parquet(path)
    assert _parse("Apple's Marketable Securities", facts_file=path).wants_figures

# --- tickers -----------------------------------------------------------------

@pytest.mark.parametrize("question, expected", [
    ("What was Apple's revenue?", ("AAPL",)),
    ("What did AAPL say about supply chain?", ("AAPL",)),
    ("How does Google describe its cloud segment?", ("GOOGL",)),
    ("Facebook's headcount", ("META",)),
    ("Meta Platforms' capital expenditure", ("META",)),
    ("AWS operating income", ("AMZN",)),
    ("Palo Alto Networks risk factors", ("PANW",)),
    ("What does ServiceNow say about AI?", ("NOW",)),
    ("Texas Instruments' fab investments", ("TXN",)),
])
def test_company_names_and_tickers_resolve(question, expected):
    assert _parse(question).tickers == expected


def test_tickers_keep_order_of_first_mention_without_duplicates():
    parsed = _parse("Compare Microsoft and Apple; how does MSFT differ from Apple?")
    assert parsed.tickers == ("MSFT", "AAPL")


def test_lower_case_ticker_words_do_not_resolve():
    # "now" is a word before it is ServiceNow's ticker; "meta" is matched by
    # the alias table, so only the bare-ticker path is case-sensitive.
    assert _parse("How is revenue recognised now?").tickers == ()
    assert _parse("What is Apple's approach to pricing now?").tickers == ("AAPL",)


def test_a_name_inside_a_longer_word_does_not_resolve():
    assert _parse("What does the filing say about intellectual property?").unresolved == ()
    assert _parse("pineapple").tickers == ()


def test_only_tickers_in_scope_resolve_and_the_rest_are_reported():
    parsed = _parse("Compare Apple and Microsoft in FY2024", known_tickers=("AAPL",))
    assert parsed.tickers == ("AAPL",)
    assert parsed.unresolved == ("Microsoft",)
    assert parsed.question_type != "unanswerable"


@pytest.mark.parametrize("question, mention", [
    ("Microsoft revenue FY2024", "Microsoft"),
    ("MSFT revenue FY2024", "MSFT"),
])
def test_a_company_dropped_from_scope_is_reported_not_skipped(question, mention):
    # The alias table still knows Microsoft; the scope does not. Searching the
    # other filings for it without a word is the failure the unresolved list
    # exists to prevent.
    parsed = _parse(question, known_tickers=("AAPL",))
    assert parsed.tickers == ()
    assert parsed.unresolved == (mention,)
    assert parsed.question_type == "unanswerable"
    assert parsed.to_query().filters == {"fiscal_year": [2024]}


def test_a_ticker_in_scope_without_aliases_is_found_by_ticker():
    parsed = _parse("What did ZZZZ report?", known_tickers=("AAPL", "ZZZZ"))
    assert parsed.tickers == ("ZZZZ",)


def test_default_scope_is_the_companies_file():
    # No known_tickers: resolves against config/companies.txt, which holds AAPL.
    assert parse_question("What was Apple's revenue in FY2024?", facts_file=None).tickers == ("AAPL",)


# --- fiscal years --------------------------------------------------------------

@pytest.mark.parametrize("question, expected", [
    ("revenue in FY2024", (2024,)),
    ("revenue in FY 2024", (2024,)),
    ("revenue in FY24", (2024,)),
    ("revenue in fiscal 2023", (2023,)),
    ("revenue in fiscal year 2023", (2023,)),
    ("revenue in '23", (2023,)),
    ("revenue in 2022", (2022,)),
    ("revenue in 2022 and 2024", (2022, 2024)),
    ("revenue from 2022 to 2024", (2022, 2023, 2024)),
    ("revenue between 2022 and 2024", (2022, 2023, 2024)),
    ("revenue 2022-2024", (2022, 2023, 2024)),
    ("revenue FY22-24", (2022, 2023, 2024)),
    ("revenue FY22 to FY24", (2022, 2023, 2024)),
    ("revenue through 2021 to 2025", (2021, 2022, 2023, 2024, 2025)),
])
def test_years_are_read_in_every_form_a_question_writes_them(question, expected):
    assert _parse(question).fiscal_years == expected


def test_a_bare_two_digit_number_is_not_a_year():
    parsed = _parse("Did margin exceed 24 percent?")
    assert parsed.fiscal_years == ()
    assert parsed.unresolved == ()


def test_a_number_that_is_not_a_year_is_ignored():
    parsed = _parse("Were there 1000 employees, and $2500 million of debt?")
    assert parsed.fiscal_years == ()
    assert parsed.unresolved == ()


@pytest.mark.parametrize("question", [
    "Did Adobe report more than $2024 million in debt?",
    "Did Adobe report more than $2000 million in debt?",
    "Did Adobe report more than $ 2024 million in debt?",
    "Did Adobe report USD 2024 million in debt?",
    "Did Adobe have 2000 employees?",
    "Did Adobe issue 2021 shares?",
    "Was debt between $2021 and $2024 million?",
])
def test_an_amount_or_a_count_is_not_a_year(question):
    parsed = _parse(question)
    assert parsed.fiscal_years == ()
    assert parsed.unresolved == ()
    assert parsed.question_type != "unanswerable"


def test_a_year_next_to_an_amount_is_still_a_year():
    parsed = _parse("Was debt above $2000 million in 2024?")
    assert parsed.fiscal_years == (2024,)
    assert parsed.unresolved == ()
    assert _parse("Was debt above $2000 million in 2015?").unresolved == ("2015",)


def test_years_outside_the_corpus_do_not_filter_and_are_reported():
    parsed = _parse("What was Apple's revenue in 2015?")
    assert parsed.fiscal_years == ()
    assert parsed.unresolved == ("2015",)
    assert parsed.tickers == ("AAPL",)


def test_a_range_that_straddles_the_corpus_keeps_the_years_inside_it():
    parsed = _parse("Apple's revenue from 2019 to 2023")
    assert parsed.fiscal_years == (2021, 2022, 2023)
    assert parsed.unresolved == ("2019",)


def test_an_unresolved_year_is_reported_as_written():
    assert _parse("revenue in FY15").unresolved == ("FY15",)


@pytest.mark.parametrize("question", [
    "What was Apple's revenue between FY2021 and FY2025?",
    "What was Apple's revenue between FY2025 and FY2021?",
    "What was Apple's revenue from 2025 to 2021?",
])
def test_a_range_expands_in_either_endpoint_order(question):
    assert _parse(question).fiscal_years == (2021, 2022, 2023, 2024, 2025)


def test_a_plain_pair_is_still_a_pair_in_either_order():
    assert _parse("revenue in 2025 and 2021").fiscal_years == (2021, 2025)


def test_a_range_too_wide_to_be_a_filter_is_not_expanded():
    parsed = _parse("revenue from 1999 to 2024")
    assert parsed.fiscal_years == (2024,)
    assert parsed.unresolved == ("1999",)


def test_a_short_range_after_a_four_digit_year_expands():
    assert _parse("revenue FY2022-24").fiscal_years == (2022, 2023, 2024)


def test_a_curly_apostrophe_year_is_a_year():
    assert _parse("revenue in ’23").fiscal_years == (2023,)


@pytest.mark.parametrize("question", [
    "How does Micron describe the CHIPS Act of 2022?",
    "What does Apple say about the Securities Exchange Act of 1934?",
    "How does Oracle describe Sarbanes-Oxley Act of 2002 compliance?",
])
def test_the_year_of_a_law_is_not_a_fiscal_year(question):
    parsed = _parse(question)
    assert parsed.fiscal_years == ()
    assert parsed.unresolved == ()
    assert parsed.question_type != "unanswerable"


@pytest.mark.parametrize("question, expected", [
    ("What was Meta's 2023 headcount?", (2023,)),
    ("What were Apple's 2024 shares outstanding?", (2024,)),
    ("Was Apple's revenue in 2024 more than 2023?", (2023, 2024)),
])
def test_a_year_that_dates_a_count_or_a_comparison_is_still_a_year(question, expected):
    assert _parse(question).fiscal_years == expected


@pytest.mark.parametrize("question", [
    "Did Apple have more than 2000 suppliers?",
    "What does the term '10-K' mean for Apple?",
])
def test_a_quantity_or_a_quoted_number_is_not_a_year(question):
    parsed = _parse(question)
    assert parsed.fiscal_years == ()
    assert parsed.unresolved == ()
    assert parsed.question_type != "unanswerable"


# --- unresolved companies -------------------------------------------------------

def test_a_dropped_peer_does_not_filter_and_is_reported():
    parsed = _parse("How does Apple describe competition with NVIDIA?")
    assert parsed.tickers == ("AAPL",)
    assert parsed.unresolved == ("NVIDIA",)
    assert parsed.question_type != "unanswerable"


def test_a_question_only_about_a_dropped_peer_is_unanswerable_but_still_searches():
    parsed = _parse("What was Intel's revenue in 2023?")
    assert parsed.tickers == ()
    assert parsed.fiscal_years == (2023,)
    assert parsed.unresolved == ("Intel",)
    assert parsed.question_type == "unanswerable"
    assert parsed.to_query().filters == {"fiscal_year": [2023]}


def test_nothing_ever_raises_on_an_unknown_entity():
    parsed = _parse("What did Zebra Technologies report?")
    assert parsed.tickers == ()
    assert parsed.unresolved == ()
    assert parsed.question_type == "factual"


def test_a_blank_question_is_refused():
    with pytest.raises(ValueError, match="blank"):
        _parse("   ")


# --- classification ---------------------------------------------------------

@pytest.mark.parametrize("question, expected", [
    ("What does Apple say about its supply chain?", "factual"),
    ("Summarise Microsoft's risk factors", "factual"),
    ("What was Apple's revenue in FY2024?", "numeric"),
    ("How many employees does Cisco have?", "numeric"),
    ("What was Oracle's gross margin?", "numeric"),
    ("Compare Apple and Microsoft's revenue in FY2024", "comparative"),
    ("Which company has the highest operating margin?", "comparative"),
    ("Apple vs. Microsoft on cloud", "comparative"),
    ("How did Apple's revenue change from 2022 to 2024?", "temporal"),
    ("What was Adobe's revenue in 2022 and 2024?", "temporal"),
    ("How has Meta's headcount grown?", "temporal"),
    ("Apple's revenue year over year", "temporal"),
    ("What will Apple's revenue be next year?", "unanswerable"),
    ("Should I buy Microsoft stock?", "unanswerable"),
    ("What is Apple's stock price?", "unanswerable"),
    ("What did NVIDIA report in 2023?", "unanswerable"),
    ("What was Apple's revenue in 2015?", "unanswerable"),
])
def test_question_types(question, expected):
    assert _parse(question).question_type == expected


def test_two_companies_outrank_a_temporal_reading():
    parsed = _parse("Compare Apple and Microsoft's revenue in 2023 and 2024")
    assert parsed.question_type == "comparative"
    assert parsed.fiscal_years == (2023, 2024)


def test_one_company_over_two_years_is_temporal_even_when_it_says_compare():
    assert _parse("Compare Apple's revenue in 2022 and 2024").question_type == "temporal"


def test_between_two_years_is_a_range_not_a_comparison():
    parsed = _parse("How did Apple's revenue move between 2022 and 2024?")
    assert parsed.question_type == "temporal"
    assert parsed.fiscal_years == (2022, 2023, 2024)


def test_will_inside_a_real_question_is_not_a_forecast():
    parsed = _parse("What risks did Apple say will affect its supply chain?")
    assert parsed.question_type == "factual"


@pytest.mark.parametrize("question", [
    # A topic the filing discusses, asked as what the filing said about it.
    "What risks did Apple disclose about its stock price in FY2024?",
    "What did Apple say about forecasting risk?",
    "What does Microsoft's 10-K say about its outlook for next year?",
    "What did Apple say it expects next year?",
    "What guidance did Oracle give going forward?",
    "According to its 10-K, how does Cisco describe share price volatility?",
])
def test_a_question_about_what_the_filing_says_is_answerable(question):
    assert _parse(question).question_type != "unanswerable"


@pytest.mark.parametrize("question", [
    # The thing itself, not what the filing said about it.
    "What is Apple's stock price?",
    "What will Apple's revenue be next year?",
    "Should I buy Microsoft stock?",
    "What is Apple's price target?",
    # A reporting verb does not excuse a request for a new prediction.
    "Based on what Apple disclosed, predict next quarter's revenue",
    "Can you forecast Microsoft's FY2026 revenue?",
    "Given what Oracle reported, estimate its revenue going forward",
    # Nor is "you expect" a reporting verb.
    "What revenue do you expect from Apple next year?",
    # Nor is a reporting verb in the future, or "expected" describing the thing asked for.
    "What will Apple report next year?",
    "What is Apple's expected stock price next year?",
])
def test_a_request_for_advice_a_prediction_or_a_price_is_unanswerable(question):
    assert _parse(question).question_type == "unanswerable"


@pytest.mark.parametrize("question, because", [
    ("Should I buy Microsoft stock?", "request"),
    ("Predict Apple's revenue next year", "request"),
    # Advice about a company outside the corpus is refused as advice.
    ("Should I buy Intel stock?", "request"),
    # An outside company's own figure: a possessive, or the subject after an
    # auxiliary.
    ("What was Intel's revenue in 2023?", "company"),
    ("How many employees did NVIDIA have in FY2024?", "company"),
    ("How much revenue did Tesla report in FY2023?", "company"),
    # The words that mark a price or the future are also in questions a filing
    # answers: the contractual obligations table, remaining performance
    # obligations, Item 5's repurchases and the cover page.
    ("What will Apple's revenue be next year?", "topic"),
    ("What is Apple's stock price?", "topic"),
    ("What were Microsoft's purchase obligations due next year in FY2024?", "topic"),
    ("How much of Oracle's remaining performance obligations will be recognized in the "
     "next fiscal year, as of FY2024?", "topic"),
    ("What average share price did Apple pay for repurchases in FY2024?", "topic"),
    ("What was the market capitalization of Apple's stock held by non-affiliates in FY2024?",
     "topic"),
    # A company outside the corpus that the filings inside it are asked about,
    # or whose own figure is not what is asked for.
    ("Which companies named NVIDIA as a competitor in FY2024?", "topic"),
    ("What did the filings disclose about supply agreements with Intel in FY2023?", "topic"),
    ("Which companies reported revenue from Intel as a customer in FY2024?", "topic"),
    ("Did any company mention Qualcomm's patents?", "topic"),
    ("What did NVIDIA report in 2023?", "topic"),
    ("What was Apple's revenue in 2015?", "year"),
    # A year after the corpus is outside it like one before, and is searched.
    ("What will Apple's revenue be in fiscal 2026?", "year"),
])
def test_an_unanswerable_question_says_why_it_is_one(question, because):
    parsed = _parse(question)
    assert parsed.question_type == "unanswerable"
    assert parsed.unanswerable_because == because


def test_an_answerable_question_has_no_reason_to_be_refused():
    assert _parse("What was Apple's revenue in FY2024?").unanswerable_because is None
    with pytest.raises(ValueError, match="only for an unanswerable question"):
        ParsedQuestion("q", "factual", (), (), (), False, unanswerable_because="request")
    with pytest.raises(ValueError, match="unanswerable_because must be one of"):
        ParsedQuestion("q", "unanswerable", (), (), (), False, unanswerable_because="mood")


def test_a_prediction_word_inside_a_line_item_s_name_is_not_a_request():
    # One of the 12,579 benchmark questions: the XBRL label holds "Estimate"
    # after a comma, which read as an instruction to estimate.
    parsed = _parse("What was Loss Contingency, Estimate of Possible Loss for INTU in FY2021?")
    assert parsed.question_type != "unanswerable"
    assert _parse("Given what Oracle reported, estimate its revenue going forward"
                  ).question_type == "unanswerable"


def test_a_prediction_verb_after_a_subject_asks_what_the_filing_predicts():
    assert _parse("What does Apple predict for its supply chain?").question_type == "factual"


def test_goodwill_is_not_a_forecast_cue():
    assert _parse("How much goodwill did Broadcom carry?").question_type == "numeric"


# --- figures and the table boost ---------------------------------------------

def test_a_numeric_question_turns_the_table_boost_on():
    query = _parse("What was Apple's revenue in FY2024?").to_query()
    assert query.table_boost == TABLE_BOOST


def test_a_comparison_of_figures_wants_tables_too():
    parsed = _parse("Compare Apple and Microsoft's net income")
    assert parsed.question_type == "comparative"
    assert parsed.wants_figures is True
    assert parsed.to_query().table_boost == TABLE_BOOST


def test_a_prose_question_leaves_the_table_boost_off():
    parsed = _parse("What does Apple say about its supply chain?")
    assert parsed.wants_figures is False
    assert parsed.to_query().table_boost == 1.0


# --- the Query and the exposed filters -------------------------------------------

def test_to_query_carries_the_filters_in_the_corpus_form():
    query = _parse("What was Apple's revenue in FY2024?").to_query(top_k=8)
    assert isinstance(query, Query)
    assert query.text == "What was Apple's revenue in FY2024?"
    assert query.tickers == ("AAPL",)
    assert query.fiscal_years == (2024,)
    assert query.top_k == 8
    assert query.filters == {"ticker": ["AAPL"], "fiscal_year": [2024]}


def test_to_query_leaves_top_k_at_the_query_default_when_not_given():
    assert _parse("Apple's revenue").to_query().top_k == Query("x").top_k


def test_filters_match_the_query_filters():
    parsed = _parse("Compare Apple and Microsoft in FY2023")
    assert parsed.filters == parsed.to_query().filters
    assert _parse("What is a 10-K?").filters == {}


def test_describe_shows_every_decision_back_to_the_user():
    lines = _parse("How did Apple compete with NVIDIA from 2019 to 2022?").describe()
    assert "Question type: temporal" in lines
    assert "Companies: AAPL" in lines
    assert "Fiscal years: FY2021, FY2022" in lines
    assert "Not in the corpus: NVIDIA, 2019" in lines


@pytest.mark.parametrize("question, kind", [
    # Searched, so the line sits under whatever answer a filing gave.
    ("What were Microsoft's purchase obligations due next year in FY2024?",
     "Question type: unanswerable, searched in case a filing answers it"),
    ("What was Apple's revenue in 2015?",
     "Question type: unanswerable, searched in case a filing answers it"),
    # Refused, so there is no answer for the line to contradict.
    ("Should I buy Microsoft stock?", "Question type: unanswerable"),
    ("What was Intel's revenue in 2023?", "Question type: unanswerable"),
    ("What was Apple's revenue in FY2024?", "Question type: numeric"),
])
def test_describe_says_when_an_unanswerable_reading_was_searched_anyway(question, kind):
    assert _parse(question).describe()[0] == kind


def test_describe_and_the_refusal_are_one_decision():
    # A company picked by hand makes an outside company's figure a search
    # (the app's sidebar), and the line under the answer has to say so:
    # decided in two places, it read "unanswerable" under an answer.
    reading = _parse("What was Intel's revenue in 2023?")
    assert reading.refused == "company"
    by_hand = reading.scoped_to(Query(reading.question, tickers=("AAPL",)))
    assert by_hand.refused is None
    assert by_hand.describe()[0] == (
        "Question type: unanswerable, searched in case a filing answers it")
    # Advice is refused whichever company is searched.
    advice = _parse("Should I buy Intel stock?")
    assert advice.scoped_to(Query(advice.question, tickers=("AAPL",))).refused == "request"
    # And every reason a reading can be refused for has its abstention.
    from src.rag.answer import REFUSALS
    refused_for = {_parse(q).refused for q in (
        "Should I buy Intel stock?", "What was Intel's revenue in 2023?",
        "What was Apple's revenue in 2015?", "What is Apple's current stock price?")}
    assert refused_for - {None} == set(REFUSALS)


def test_describe_says_when_nothing_is_filtered():
    assert "Filters: none, searching every company and year" in _parse("What is a 10-K?").describe()


# --- search text ----------------------------------------------------------------

@pytest.mark.parametrize("question, expected", [
    ("What was Meta's total assets at the end of fiscal year 2025?",
     "What was total assets at the end?"),
    ("What did AAPL say in fiscal 2023 about supply chain?",
     "What did say about supply chain?"),
    ("Meta Platforms' capital expenditure", "capital expenditure"),
    ("How did Microsoft's revenue change from FY2022 to FY2024?",
     "How did revenue change?"),
    ("How did Microsoft's revenue change in FY22-24?", "How did revenue change?"),
    ("How much did Amazon generate in net sales during fiscal year 2025?",
     "How much did generate in net sales?"),
])
def test_search_text_drops_the_companies_and_years_the_filters_apply(question, expected):
    assert _parse(question).search_text == expected


@pytest.mark.parametrize("question, expected", [
    ("What was Amazon's AWS segment operating income in fiscal year 2024?",
     "What was AWS segment operating income?"),
    ("How did Google Cloud revenue change in FY2024?", "How did Google Cloud revenue change?"),
    ("What was Microsoft 365 revenue in FY2024?", "What was Microsoft 365 revenue?"),
])
def test_search_text_keeps_an_alias_that_names_a_segment_or_product(question, expected):
    assert _parse(question).search_text == expected


def test_search_text_keeps_a_year_inside_a_date():
    parsed = _parse("How much did Meta hold in total assets as of December 31, 2025?")
    assert parsed.fiscal_years == (2025,)
    assert parsed.search_text == "How much did hold in total assets as of December 31, 2025?"


def test_search_text_keeps_what_did_not_resolve():
    parsed = _parse("How did Apple compete with NVIDIA in FY2019?")
    assert "NVIDIA" in parsed.search_text
    assert "FY2019" in parsed.search_text
    assert "Apple" not in parsed.search_text


def test_search_text_is_the_question_when_nothing_would_be_left():
    assert _parse("Apple 2024").search_text == "Apple 2024"
    assert _parse("What is a 10-K?").search_text == "What is a 10-K?"


def test_to_query_gives_keyword_search_the_trimmed_text_and_dense_the_question():
    query = _parse("What was Apple's revenue in FY2024?").to_query()
    assert query.text == "What was Apple's revenue in FY2024?"
    assert query.keyword_text == "What was revenue?"


def test_to_query_sets_no_keyword_text_when_nothing_was_trimmed():
    assert _parse("What is a 10-K?").to_query().keyword_text is None


def test_a_hand_built_parsed_question_searches_its_question():
    parsed = ParsedQuestion(
        question="q", question_type="factual", tickers=(), fiscal_years=(),
        unresolved=(), wants_figures=False,
    )
    assert parsed.search_text == "q"


# --- frozen -------------------------------------------------------------------

def test_parsed_question_is_frozen_and_normalised():
    parsed = _parse("Apple's revenue")
    with pytest.raises(dataclasses.FrozenInstanceError):
        parsed.tickers = ("MSFT",)
    rebuilt = ParsedQuestion(
        question="q", question_type="factual", tickers=["aapl"], fiscal_years=[2024],
        unresolved=[], wants_figures=False,
    )
    assert isinstance(rebuilt.tickers, tuple)
    assert isinstance(rebuilt.fiscal_years, tuple)


def test_parsed_question_refuses_an_unknown_type():
    with pytest.raises(ValueError, match="question_type"):
        ParsedQuestion(
            question="q", question_type="rhetorical", tickers=(), fiscal_years=(),
            unresolved=(), wants_figures=False,
        )
