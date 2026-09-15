"""Query understanding: what it extracts, what it degrades on, and how it classifies."""

import dataclasses

import pytest

from src.rag.query import ParsedQuestion, build_query, parse_question
from src.retrieval.constants import TABLE_BOOST
from src.retrieval.records import Query

# The real scope, so the tests do not depend on config/companies.txt while
# still exercising the alias table for every company in it.
SCOPE = ("AAPL", "MSFT", "AVGO", "GOOGL", "META", "AMZN", "ORCL", "CRM", "ADBE",
         "CSCO", "TXN", "MU", "INTU", "NOW", "PANW")
YEARS = (2021, 2025)


def _parse(question: str, **overrides) -> ParsedQuestion:
    options = dict(known_tickers=SCOPE, fiscal_years=YEARS)
    options.update(overrides)
    return parse_question(question, **options)


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
    assert parse_question("What was Apple's revenue in FY2024?").tickers == ("AAPL",)


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
])
def test_a_request_for_advice_a_prediction_or_a_price_is_unanswerable(question):
    assert _parse(question).question_type == "unanswerable"


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


def test_describe_says_when_nothing_is_filtered():
    assert "Filters: none, searching every company and year" in _parse("What is a 10-K?").describe()


def test_build_query_is_the_short_form():
    query = build_query("Apple's revenue in FY2024", top_k=5, known_tickers=SCOPE)
    assert query.tickers == ("AAPL",)
    assert query.fiscal_years == (2024,)
    assert query.top_k == 5


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
