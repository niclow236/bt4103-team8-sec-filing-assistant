"""Issue #34: a numeric question is looked up in the facts store, not asked of a model.

The store is a parquet written per test, and the retriever is a stub holding
passages, so nothing here reads the real corpus, builds an index or starts a
model. What is under test is the routing: which questions take the lookup, what
the lookup refuses, and that a refusal reaches the retrieval path unchanged.
"""

import dataclasses
from decimal import Decimal

import pandas as pd
import pytest

from src.rag.answer import answer_question
from src.rag.constants import FACTS_PROVIDER, FACTS_SOURCE, FACTS_TEMPLATE_ID
from src.rag.numeric import (
    Fact,
    answer_from_facts,
    find_metric,
    format_figure,
    lookup_fact,
    prints_figure,
    supporting_passage,
)
from src.rag.records import GenerationConfig
from src.retrieval.records import RetrievedPassage

ACCESSION = "0000320193-24-000123"
OTHER_ACCESSION = "0000320193-23-000106"
QUESTION = "What was Apple's total revenue in FY2024?"
REVENUE = 391_035_000_000

# The figure as Apple's income statement prints it: a table in millions.
TABLE = ("CONSOLIDATED STATEMENTS OF OPERATIONS (in millions)\n\n"
         "| | 2024 | 2023 |\n| Total net sales | 391,035 | 383,285 |")


def _fact_row(**changes):
    row = dict(
        ticker="AAPL", cik=320193, company="Apple Inc.", accession=ACCESSION,
        form="10-K", filing_date="2024-11-01", period_of_report="2024-09-28",
        concept="us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
        label="Total net sales", value=float(REVENUE), raw_value=str(REVENUE),
        unit="USD", scale=6, fiscal_year=2024, fiscal_period="FY",
        period_start="2023-10-01", period_end="2024-09-28", period_type="duration",
        is_current_year=True, statement_type="IncomeStatement", is_audited=True,
    )
    return row | changes


@pytest.fixture
def store(tmp_path):
    """A facts store holding Apple's FY2024 revenue."""
    def write(*rows):
        path = tmp_path / "facts.parquet"
        pd.DataFrame(list(rows) or [_fact_row()]).to_parquet(path, index=False)
        return path
    return write


def _passage(text=TABLE, chunk_id=f"{ACCESSION}_part_ii_item_8_3", **changes):
    values = dict(
        chunk_id=chunk_id, text=text, score=1.0, rank=1, retriever="hybrid",
        ticker="AAPL", company="Apple Inc.", fiscal_year=2024, item="8",
        title="Financial Statements", url="https://example.test/filing",
        content_type="table", cik=320193, form="10-K", part="II",
        filing_date="2024-11-01",
    )
    return RetrievedPassage(**(values | changes))


class StubRetriever:
    """Returns fixed passages, recording the queries it was asked."""

    def __init__(self, *passages):
        self.passages = list(passages)
        self.queries = []

    def search(self, query, k=None):
        self.queries.append(query)
        wanted = query.content_type
        return [p for p in self.passages if wanted is None or p.content_type == wanted]


# --- reading the question ------------------------------------------------------

@pytest.mark.parametrize("question, expected", [
    ("What was Apple's total revenue in FY2024?", "revenue"),
    ("What were Apple's net sales in FY2024?", "revenue"),
    ("How much net income did Microsoft report in FY2023?", "net_income"),
    ("What was Oracle's operating income?", "operating_income"),
    ("What were total assets?", "assets"),
    ("What was diluted EPS?", "diluted_eps"),
    ("What was cash from operations?", "operating_cash"),
])
def test_find_metric_reads_the_line_item_a_question_names(question, expected):
    assert find_metric(question) == expected


def test_a_question_naming_no_supported_metric_is_not_a_lookup():
    assert find_metric("What are Apple's main risk factors?") is None
    assert find_metric("How many employees does Cisco have?") is None


def test_a_question_naming_two_metrics_is_not_a_lookup():
    # Two figures and a sentence joining them is #35's job, not a lookup.
    assert find_metric("How did revenue and net income compare in FY2024?") is None


@pytest.mark.parametrize("question", [
    # An alias inside a line item the annual figure does not answer. Each of
    # these was answered with total annual net sales, cited.
    "What was Apple's cost of revenue in FY2024?",
    "What was Microsoft's deferred revenue in FY2024?",
    "What was iPhone revenue for Apple in FY2024?",
    "What was Apple's revenue in Q4 of FY2024?",
    "What percentage of revenue did Apple spend on R&D in FY2024?",
    "What was Apple's non-operating income in FY2024?",
    "What was Apple's net income per diluted share in FY2024?",
    # Scopes the store's annual figures cannot answer at all.
    "What was AWS operating income in FY2024?",
    "What was Apple's operating margin in FY2024?",
    "What was Apple's revenue by segment in FY2024?",
    "How much did Apple's revenue increase by in FY2024?",
])
def test_a_qualifier_on_the_line_item_is_not_a_lookup(question):
    # The guard verify.py marks an answer unverified by: a question this route
    # answers and that checker cannot check is what neither should allow.
    assert find_metric(question) is None


@pytest.mark.parametrize("question, expected", [
    ("What was Apple's diluted earnings per share in FY2024?", "diluted_eps"),
    ("What was Apple's basic EPS in FY2024?", "basic_eps"),
])
def test_the_scope_guard_does_not_block_the_per_share_metrics(question, expected):
    assert find_metric(question) == expected


@pytest.mark.parametrize("question", [
    # A segment, a product line, a geography: the question asks for a part of
    # the company, and the whole-company figure is not a narrower answer to it.
    "What were Apple's Services net sales in FY2024?",
    "What were Amazon's North America net sales in FY2024?",
    "What was Microsoft's Intelligent Cloud revenue in FY2024?",
    "What was Alphabet's advertising revenue in FY2024?",
    "What was Apple's revenue in Greater China in FY2024?",
    "What was Meta's net loss in Reality Labs in FY2023?",
    # A second line item, or a narrower one sharing the words.
    "What was Apple's net income per share in FY2024?",
    "What was Apple's net income attributable to noncontrolling interests in FY2024?",
    "What were Apple's total liabilities and shareholders' equity in FY2024?",
    # A question about prose that happens to name a line item.
    "How does Apple recognise revenue in FY2024?",
    "How does Apple recognize revenue in FY2024?",
    "What drove Apple's net sales in FY2024?",
    "Explain Meta's net loss in FY2023",
    # A part of the company named with a possessive, which reads like the
    # company's own.
    "What was LinkedIn's revenue at Microsoft in fiscal 2024?",
    "What was VMware's revenue for Broadcom in fiscal 2024?",
    "What was YouTube's revenue for Alphabet in FY2024?",
    "What were Europe's net sales for Apple in FY2024?",
    "What was Microsoft's LinkedIn's revenue in fiscal 2024?",
])
def test_a_question_asking_for_something_narrower_is_not_a_lookup(question):
    # A blocklist would have to grow one segment name at a time; the test is
    # positive instead -- what is left after the metric's own words, the
    # company and the year has to be nothing but question scaffolding.
    assert find_metric(question) is None


@pytest.mark.parametrize("question, expected", [
    ("What was Apple's total revenue in FY2024?", "revenue"),
    ("What were Apple's net sales in fiscal 2024?", "revenue"),
    ("How much net income did Microsoft report in FY2023?", "net_income"),
    ("What was Oracle's operating income?", "operating_income"),
    ("What were total assets?", "assets"),
    ("What was cash from operations for AAPL in FY2024?", "operating_cash"),
    ("In FY2024, what was Apple revenue?", "revenue"),
    ("MSFT total revenue FY2024", "revenue"),
    # A possessive the scaffolding owns, which the rule above must not refuse.
    ("What was Oracle Corporation's total revenue in fiscal 2024?", "revenue"),
    ("What was the company's total revenue in FY2024?", "revenue"),
    # "total" is the statement's own word for the whole-company row, and
    # "generate" is a verb of reporting: refusing either lost questions asking
    # for exactly the figure the store holds.
    ("What was Microsoft's total net income in fiscal year 2023?", "net_income"),
    ("What were Oracle's total revenues in fiscal year 2024?", "revenue"),
    ("How much did Amazon generate in net sales during fiscal year 2025?", "revenue"),
    ("What were Apple's total net sales in FY2024?", "revenue"),
])
def test_a_question_asking_only_for_the_line_item_is_a_lookup(question, expected):
    assert find_metric(question) == expected


def test_metrics_in_reads_a_claim_without_judging_what_it_asks_for():
    # verify.py reads an answer's claims with this: "revenue was $391,035
    # million" is a claim about revenue however it is phrased, so the
    # positive test find_metric applies to a question must not apply here.
    from src.rag.numeric import metrics_in

    assert metrics_in("Revenue was $391,035 million, up 2 percent.") == {"revenue"}
    assert metrics_in("Services net sales grew.") == {"revenue"}
    assert metrics_in("Apple sells phones.") == set()


# --- writing the figure ---------------------------------------------------------

@pytest.mark.parametrize("value, unit, expected", [
    (Decimal("391035000000"), "USD", "$391,035,000,000"),
    (Decimal("-1500000"), "USD", "-$1,500,000"),
    (Decimal("-2.35"), "USD/shares", "-$2.35 per share"),
    (Decimal("6.11"), "USD/shares", "$6.11 per share"),
    (Decimal("0.247"), "ratio", "0.247"),
    (Decimal("15408095000"), "shares", "15,408,095,000 shares"),
    (Decimal("391035000000.00"), "USD", "$391,035,000,000"),
])
def test_format_figure_writes_the_figure_exactly(value, unit, expected):
    assert format_figure(value, unit) == expected


def test_the_figure_is_never_rounded_to_a_friendlier_scale():
    # The reader checks this against the filing, so 391.0 billion will not do.
    assert "billion" not in format_figure(Decimal(REVENUE), "USD")


@pytest.mark.parametrize("value, expected", [
    # The store holds exponent-form raw_value strings, and str(float) goes
    # exponential from 1e16. Printed back as itself, neither a reader nor
    # verify.py's parser can read the figure.
    (Decimal("3.91035E+11"), "$391,035,000,000"),
    (Decimal("1E+16"), "$10,000,000,000,000,000"),
    (Decimal("1e-05"), "$0.00001"),
    (Decimal("5.00"), "$5"),
])
def test_format_figure_never_prints_scientific_notation(value, expected):
    assert format_figure(value, "USD") == expected


# --- the lookup -------------------------------------------------------------------

def test_lookup_finds_the_figure_for_one_company_and_year(store):
    fact = lookup_fact(QUESTION, ("AAPL",), (2024,), facts_file=store())
    assert isinstance(fact, Fact)
    assert fact.value == Decimal(REVENUE)
    assert (fact.metric, fact.unit) == ("revenue", "USD")
    # The metric's own words, not the store's label column: that holds the
    # taxonomy's "Revenue from Contract with Customer, Excluding Assessed Tax".
    assert fact.label == "total revenue"
    assert (fact.ticker, fact.company, fact.fiscal_year) == ("AAPL", "Apple Inc.", 2024)
    assert fact.accession == ACCESSION
    assert fact.figure == "$391,035,000,000"
    assert fact.period == "fiscal year 2024, ended 2024-09-28"


@pytest.mark.parametrize("tickers, years", [
    ((), (2024,)),                  # no company named
    (("AAPL", "MSFT"), (2024,)),    # two companies is a comparison
    (("AAPL",), ()),                # no year named
    (("AAPL",), (2023, 2024)),      # two years is a movement
])
def test_lookup_needs_exactly_one_company_and_one_year(store, tickers, years):
    assert lookup_fact(QUESTION, tickers, years, facts_file=store()) is None


def test_lookup_returns_nothing_for_a_year_the_store_does_not_hold(store):
    assert lookup_fact(QUESTION, ("AAPL",), (2021,), facts_file=store()) is None


def test_lookup_returns_nothing_for_another_company(store):
    assert lookup_fact(QUESTION, ("MSFT",), (2024,), facts_file=store()) is None


def test_lookup_ignores_a_comparative_figure_in_the_same_filing(store):
    # A FY2024 filing prints FY2023 revenue too, flagged as not the current
    # year. Answering FY2024 with it is the failure the flag exists to prevent.
    comparative = _fact_row(value=float(383_285_000_000), raw_value="383285000000",
                            period_start="2022-10-02", period_end="2023-09-30",
                            is_current_year=False)
    fact = lookup_fact(QUESTION, ("AAPL",), (2024,), facts_file=store(_fact_row(), comparative))
    assert fact.value == Decimal(REVENUE)


def test_lookup_refuses_when_two_concepts_disagree(store):
    other = _fact_row(concept="us-gaap:Revenues", value=390_000_000_000.0,
                      raw_value="390000000000")
    assert lookup_fact(QUESTION, ("AAPL",), (2024,), facts_file=store(_fact_row(), other)) is None


def test_lookup_accepts_two_concepts_that_agree(store):
    same = _fact_row(concept="us-gaap:Revenues")
    fact = lookup_fact(QUESTION, ("AAPL",), (2024,), facts_file=store(_fact_row(), same))
    assert fact.value == Decimal(REVENUE)


def test_lookup_ignores_a_fact_reported_in_another_unit(store):
    shares = _fact_row(unit="shares", value=15_408_095_000.0, raw_value="15408095000")
    assert lookup_fact(QUESTION, ("AAPL",), (2024,), facts_file=store(shares)) is None


def test_a_missing_store_is_a_miss_not_an_error(tmp_path):
    assert lookup_fact(QUESTION, ("AAPL",), (2024,), facts_file=tmp_path / "absent.parquet") is None


def test_a_corrupt_store_is_a_miss_not_an_error(tmp_path):
    # Whatever the parquet engine raises for a file that is not a parquet, the
    # question goes to retrieval rather than failing on a route it never asked for.
    import src.rag.numeric as numeric

    numeric._cached_facts.cache_clear()
    path = tmp_path / "facts.parquet"
    path.write_bytes(b"not a parquet file at all")
    assert lookup_fact(QUESTION, ("AAPL",), (2024,), facts_file=path) is None


def test_a_store_without_the_needed_columns_is_a_miss(tmp_path):
    path = tmp_path / "facts.parquet"
    pd.DataFrame([{"ticker": "AAPL", "fiscal_year": 2024}]).to_parquet(path, index=False)
    assert lookup_fact(QUESTION, ("AAPL",), (2024,), facts_file=path) is None


def test_lookup_reads_a_frame_already_in_memory(store, tmp_path):
    # The facts_file does not exist: a caller answering many questions reads
    # the store once and passes the frame, and must not be sent back to disk.
    frame = pd.read_parquet(store())
    fact = lookup_fact(QUESTION, ("AAPL",), (2024,),
                       facts_file=tmp_path / "absent.parquet", frame=frame)
    assert fact is not None and fact.value == Decimal(REVENUE)


# --- finding the passage that prints it ---------------------------------------------

# label is the metric's first alias, as lookup_fact builds it.
FACT = Fact(metric="revenue", concept="RevenueFromContractWithCustomerExcludingAssessedTax",
            label="total revenue", value=Decimal(REVENUE), unit="USD", ticker="AAPL",
            company="Apple Inc.", fiscal_year=2024, accession=ACCESSION,
            period_end="2024-09-28")


def test_the_supporting_passage_is_the_table_that_prints_the_figure():
    found = supporting_passage(FACT, StubRetriever(_passage()))
    assert found is not None and "391,035" in found.text


def test_one_search_covers_both_kinds_of_passage():
    # Searching tables and then everything looked through the same passages
    # twice, on top of the search answer_question runs when this finds nothing.
    retriever = StubRetriever(_passage())
    supporting_passage(FACT, retriever)
    query, = retriever.queries
    assert query.content_type is None
    assert (query.tickers, query.fiscal_years) == (("AAPL",), (2024,))
    # The line item alone for keyword search: the ticker and year are already
    # hard filters, and #87 measured that repeating them buries the tables.
    assert query.keyword_text == "total revenue"


def test_a_table_is_preferred_over_prose_that_also_prints_the_figure():
    prose = _passage(text="Total net sales were $391,035 million in 2024.",
                     chunk_id=f"{ACCESSION}_part_ii_item_7_0", content_type="prose")
    found = supporting_passage(FACT, StubRetriever(prose, _passage()))
    assert found.content_type == "table"


def test_prose_is_accepted_when_no_table_prints_the_figure():
    prose = _passage(text="Total net sales were $391,035 million in 2024.",
                     chunk_id=f"{ACCESSION}_part_ii_item_7_0", content_type="prose")
    found = supporting_passage(FACT, StubRetriever(prose))
    assert found is not None and found.content_type == "prose"


def test_a_passage_from_another_filing_is_not_cited():
    # The FY2023 filing prints FY2024 nothing, but a stale index might return
    # it; citing it would point the reader at the wrong filing.
    other = _passage(chunk_id=f"{OTHER_ACCESSION}_part_ii_item_8_3")
    assert supporting_passage(FACT, StubRetriever(other)) is None


def test_a_passage_that_does_not_print_the_figure_is_not_cited():
    assert supporting_passage(FACT, StubRetriever(_passage(text="Revenue grew."))) is None


def test_nothing_retrieved_is_no_passage():
    assert supporting_passage(FACT, StubRetriever()) is None


@pytest.mark.parametrize("text", [
    # The needle is inside a longer number, so the passage prints a different
    # figure: 391,035 lives in 1,391,035, and a scaled needle in 17,000.
    "STATEMENTS (in millions)\n\n| Total net sales | 1,391,035 |",
    "STATEMENTS (in millions)\n\n| Total net sales | 391,0351 |",
])
def test_digits_inside_a_longer_number_are_not_the_figure(text):
    assert prints_figure(text, {1_000_000: {"391,035"}}) is False


def test_a_scaled_figure_needs_the_passage_to_say_which_scale():
    by_scale = {1_000_000: {"391,035"}}
    # A bare 391,035 with no scale anywhere could be 391,035 dollars.
    assert prints_figure("| Total net sales | 391,035 |", by_scale) is False
    assert prints_figure("STATEMENTS (in millions)\n| net sales | 391,035 |", by_scale) is True
    # Prose names the scale beside the figure rather than in a heading.
    assert prints_figure("Net sales were $391,035 million.", by_scale) is True
    # The wrong scale is still the wrong figure.
    assert prints_figure("STATEMENTS (in thousands)\n| 391,035 |", by_scale) is False


def test_an_unscaled_figure_needs_no_heading():
    assert prints_figure("Diluted earnings per share were $6.08.", {1: {"6.08"}}) is True


def test_an_unscaled_figure_is_not_the_same_figure_with_a_scale_word_after_it():
    # A fact worth 5.2 is not the 5.2 in "$5.2 billion", which is 5,200,000,000.
    assert prints_figure("Revenue was $5.2 billion.", {1: {"5.2"}}) is False
    assert prints_figure("Revenue was $5.2 million.", {1: {"5.2"}}) is False
    assert prints_figure("Diluted EPS was $5.2 for the year.", {1: {"5.2"}}) is True


def test_a_per_share_figure_survives_a_table_headed_in_millions():
    # An income statement says "in millions, except per share amounts" and
    # prints its EPS row unscaled in the same table.
    text = ("CONSOLIDATED STATEMENTS OF OPERATIONS (in millions, except per share amounts)\n\n"
            "| Net sales | 391,035 |\n| Diluted (in dollars per share) | 6.08 |")
    assert prints_figure(text, {1: {"6.08"}}) is True


@pytest.mark.parametrize("text, negative", [
    ("| Operating loss | (1,500) |", True),        # the accounting form
    ("| Operating loss | $(1,500) |", True),
    ("Operating loss was -1,500.", True),
    ("Operating loss was -$1,500.", True),
    ("Operating loss was $-1,500.", True),
    ("| Operating income | 1,500 |", False),
])
def test_the_sign_printed_has_to_be_the_sign_of_the_fact(text, negative):
    assert prints_figure(text, {1: {"1,500"}}, negative=negative) is True
    # The same passage must not support the opposite sign: a positive 1,500 is
    # a different line item from a loss of 1,500.
    assert prints_figure(text, {1: {"1,500"}}, negative=not negative) is False


def test_a_negative_fact_is_cited_only_where_the_loss_is_printed():
    loss = dataclasses.replace(FACT, value=Decimal("-1500"), metric="operating_income",
                               label="operating income")
    positive = _passage(text="| Operating income | 1,500 |")
    assert supporting_passage(loss, StubRetriever(positive)) is None
    printed = _passage(text="| Operating income (loss) | (1,500) |")
    assert supporting_passage(loss, StubRetriever(printed)) is printed


# --- the same bar the retrieval path sets ------------------------------------------

def test_a_passage_below_the_callers_score_floor_is_not_cited():
    # answer_question(min_score=...) abstains rather than answer on weak
    # evidence; citing a passage it would have rejected answers where the same
    # question, asked the same way, abstains.
    weak = _passage(score=0.1)
    assert supporting_passage(FACT, StubRetriever(weak), min_score=0.5) is None
    assert supporting_passage(FACT, StubRetriever(weak), min_score=0.05) is weak


@pytest.mark.parametrize("passage", [
    _passage(score=float("nan")),
    _passage(score=float("inf")),
    _passage(text="   "),
])
def test_a_passage_with_no_usable_score_or_no_text_is_not_cited(passage):
    assert supporting_passage(FACT, StubRetriever(passage)) is None


def test_the_score_floor_reaches_the_route_from_answer_question(store, monkeypatch):
    monkeypatch.setattr("src.rag.answer.generate",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("generated")))
    # Above the floor: answered from the store.
    answered = answer_question(QUESTION, StubRetriever(_passage(score=0.9)),
                               facts_file=store(), min_score=0.5)
    assert answered.config.provider == FACTS_PROVIDER
    # Below it: the route declines, and the question then abstains on the
    # retrieval path exactly as it would have with no facts route at all.
    abstained = answer_question(QUESTION, StubRetriever(_passage(score=0.1)),
                                facts_file=store(), min_score=0.5)
    assert abstained.abstained is True
    assert abstained.abstention_reason == "below_threshold"
    assert abstained.config.provider != FACTS_PROVIDER


# --- the answer -----------------------------------------------------------------------

def test_the_answer_states_the_figure_and_cites_the_passage(store):
    answer = answer_from_facts(QUESTION, ("AAPL",), (2024,), StubRetriever(_passage()),
                               facts_file=store())
    assert answer is not None
    assert answer.text == (
        "Apple Inc. reported total revenue of $391,035,000,000 for fiscal year 2024, "
        "ended 2024-09-28. [1]"
    )
    assert answer.abstained is False
    citation, = answer.citations
    assert (citation.marker, citation.resolved) == (1, True)
    assert citation.chunk_id == f"{ACCESSION}_part_ii_item_8_3"
    assert answer.passages == (_passage(),)
    assert [s.text for s in answer.sentences] == [answer.text]


def test_the_answer_records_that_the_store_produced_it_not_a_model(store):
    answer = answer_from_facts(QUESTION, ("AAPL",), (2024,), StubRetriever(_passage()),
                               facts_file=store())
    assert answer.config == GenerationConfig(
        provider=FACTS_PROVIDER, model=FACTS_SOURCE, prompt_template_id=FACTS_TEMPLATE_ID,
    )
    assert answer.latency_ms is not None and answer.latency_ms >= 0


def test_no_fact_and_no_passage_each_give_no_answer(store):
    assert answer_from_facts("What are the risks?", ("AAPL",), (2024,),
                             StubRetriever(_passage()), facts_file=store()) is None
    assert answer_from_facts(QUESTION, ("AAPL",), (2024,),
                             StubRetriever(_passage(text="Revenue grew.")),
                             facts_file=store()) is None


# --- routing -------------------------------------------------------------------------

def _never_called(*args, **kwargs):
    raise AssertionError("the model must not be called for a looked-up answer")


def test_a_numeric_question_is_answered_from_the_store_without_a_model(store):
    answer = answer_question(QUESTION, StubRetriever(_passage()), llm=_never_called,
                             facts_file=store())
    assert answer.config.provider == FACTS_PROVIDER
    assert "391,035,000,000" in answer.text
    assert answer.citations[0].resolved is True


@pytest.mark.parametrize("question, ticker, year, concept, label", [
    ("What was Apple's total value of Accounts Payable at the end of fiscal year 2022?",
     "AAPL", 2022, "AccountsPayableCurrent", "Accounts Payable"),
    ("What was Apple's Inventories value at the end of fiscal year 2025?",
     "AAPL", 2025, "InventoryNet", "Inventories"),
    ("What was Amazon's total net sales in fiscal year 2025?",
     "AMZN", 2025, "RevenueFromContractWithCustomerExcludingAssessedTax", "Net sales"),
])
def test_issue_89_questions_receive_a_cited_fact_without_generation(
    store, monkeypatch, question, ticker, year, concept, label,
):
    monkeypatch.setattr("src.rag.answer.generate", lambda *a, **k: pytest.fail("generated"))
    # Synthetic figures: assert routing and exact reproduction, not live financial data.
    path = store(_fact_row(
        ticker=ticker, company=ticker, fiscal_year=year, concept=f"us-gaap:{concept}",
        label=label, period_of_report=f"{year}-12-31", period_end=f"{year}-12-31",
        period_start=None, period_type="instant", value=123_000_000.0, raw_value="123000000",
    ))
    passage = _passage(text=f"Statement (in millions)\n| | {year} |\n| {label} | 123 |",
                       ticker=ticker, company=ticker, fiscal_year=year)
    answer = answer_question(question, StubRetriever(passage), facts_file=path)
    assert answer.config.provider == FACTS_PROVIDER
    assert "$123,000,000" in answer.text
    assert f"fiscal year {year}" in answer.text
    assert answer.citations[0].resolved is True
    assert answer.passages == (passage,)


def test_answer_parser_uses_the_callers_label_store(store, monkeypatch):
    seen = []
    monkeypatch.setattr("src.rag.answer.answer_from_facts",
                        lambda *args, **kwargs: seen.append(kwargs))
    path = store(_fact_row(label="Marketable Securities"))
    answer = answer_question("What was Apple's Marketable Securities in FY2024?",
                             StubRetriever(), facts_file=path)
    assert seen[0]["facts_file"] == path
    assert answer.abstained is True


def test_net_sales_finds_the_statement_beyond_revenue_distractors(store, monkeypatch):
    from src.retrieval.bm25 import BM25Retriever

    question = "What was Amazon's total net sales in fiscal year 2025?"
    value = 716_924_000_000
    path = store(_fact_row(ticker="AMZN", company="Amazon", fiscal_year=2025,
                           value=float(value), raw_value=str(value),
                           period_of_report="2025-12-31", period_end="2025-12-31",
                           period_start="2025-01-01"))
    table = _passage("Statement (in millions)\n| | 2025 |\n| Total net sales | 716,924 |",
                     ticker="AMZN", company="Amazon", fiscal_year=2025)
    chunks = [dataclasses.asdict(table)]
    # More passages using the question's words than the route looks through,
    # FACT_PASSAGE_K, so the statement is out of reach of a search for them,
    # and enough others that those words still tell a passage apart.
    for index in range(200):
        text = ("Total revenue discussion about annual performance and future business prospects."
                if index < 60 else "Employees technology operations research strategy competition.")
        chunks.append(dataclasses.asdict(_passage(
            text=text, chunk_id=f"{ACCESSION}_item7_{index}", ticker="AMZN", company="Amazon",
            fiscal_year=2025, content_type="prose")))
    retriever = BM25Retriever(chunks)
    fact = lookup_fact(question, ("AMZN",), (2025,), facts_file=path)
    assert fact is not None
    # The old canonical search loses the statement below its cutoff.
    assert supporting_passage(fact, retriever) is None
    monkeypatch.setattr("src.rag.answer.generate", lambda *a, **k: pytest.fail("generated"))
    result = answer_question(question, retriever, facts_file=path)
    assert result.config.provider == FACTS_PROVIDER
    assert "$716,924,000,000" in result.text
    assert result.citations[0].chunk_id == table.chunk_id


def test_the_looked_up_answer_reaches_a_streaming_caller(store):
    seen = []
    answer = answer_question(QUESTION, StubRetriever(_passage()), llm=_never_called,
                             facts_file=store(), on_token=seen.append)
    assert "".join(seen) == answer.text


def test_a_question_with_no_matching_fact_falls_back_to_retrieval(store, monkeypatch):
    # The lookup finds nothing for FY2021, so the question takes the retrieval
    # path: a prompt is built and the model is asked, as if #34 were not here.
    asked = {}

    def fake_generate(prompt, config, **kwargs):
        asked["prompt"] = prompt
        raise RuntimeError("reached generation")

    monkeypatch.setattr("src.rag.answer.generate", fake_generate)
    with pytest.raises(RuntimeError, match="reached generation"):
        answer_question("What was Apple's total revenue in FY2021?",
                        StubRetriever(_passage(fiscal_year=2021)), facts_file=store())
    assert asked["prompt"].n_sources == 1


def test_a_non_numeric_question_never_reaches_the_store(store, monkeypatch):
    looked_up = []
    monkeypatch.setattr("src.rag.answer.answer_from_facts",
                        lambda *a, **k: looked_up.append(a) or None)
    monkeypatch.setattr("src.rag.answer.generate",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("generated")))
    with pytest.raises(RuntimeError, match="generated"):
        answer_question("What are Apple's risk factors?", StubRetriever(_passage()),
                        facts_file=store())
    assert looked_up == []


def test_use_facts_false_turns_the_route_off_for_the_ablation(store, monkeypatch):
    monkeypatch.setattr("src.rag.answer.generate",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("generated")))
    with pytest.raises(RuntimeError, match="generated"):
        answer_question(QUESTION, StubRetriever(_passage()), facts_file=store(),
                        use_facts=False)


def test_earnings_per_share_is_answered_with_the_strings_the_real_store_holds(store):
    # Two gaps this covers, both invisible to a synthetic frame: the store
    # writes the unit "USD per share", not "usd/shares"; and a per-share figure
    # is a round number at no scale, so it needs a decimal needle.
    eps = _fact_row(
        concept="us-gaap:EarningsPerShareDiluted",
        label="Earnings Per Share, Diluted", unit="USD per share",
        value=6.08, raw_value="6.08", scale=0,
    )
    passage = _passage(
        text="EARNINGS PER SHARE\n\n| | 2024 | 2023 |\n| Diluted (in dollars per share) | 6.08 | 6.13 |",
        chunk_id=f"{ACCESSION}_part_ii_item_8_5",
    )
    answer = answer_from_facts("What was Apple's diluted earnings per share in FY2024?",
                               ("AAPL",), (2024,), StubRetriever(passage),
                               facts_file=store(eps))
    assert answer is not None
    assert answer.text == (
        "Apple Inc. reported diluted earnings per share of $6.08 per share "
        "for fiscal year 2024, ended 2024-09-28. [1]"
    )
    assert answer.citations[0].chunk_id == passage.chunk_id


def test_the_store_is_read_once_however_many_questions_are_asked(store, monkeypatch):
    # Each question re-read the parquet and reran mark_current_year over every
    # row, including the questions that then fall through to retrieval.
    import src.rag.numeric as numeric

    path = store()
    numeric._cached_facts.cache_clear()
    reads = []
    real = numeric.load_facts
    monkeypatch.setattr(numeric, "load_facts", lambda p: reads.append(p) or real(p))
    for _ in range(3):
        assert lookup_fact(QUESTION, ("AAPL",), (2024,), facts_file=path) is not None
    assert len(reads) == 1


def test_a_rebuilt_store_is_not_served_stale(store):
    import os

    import src.rag.numeric as numeric

    numeric._cached_facts.cache_clear()
    path = store()
    assert lookup_fact(QUESTION, ("AAPL",), (2024,), facts_file=path).value == Decimal(REVENUE)
    # Rebuilt with a different figure. The cache is keyed on the modification
    # time as well as the path, so the new store is read rather than served
    # from the first call; the time is set explicitly because two writes in one
    # test can land inside the filesystem's timestamp resolution.
    pd.DataFrame([_fact_row(value=1.0, raw_value="1")]).to_parquet(path, index=False)
    os.utime(path, ns=(0, 0))
    assert lookup_fact(QUESTION, ("AAPL",), (2024,), facts_file=path).value == Decimal(1)


@pytest.mark.parametrize("question", [
    "What were Apple's net sales in fiscal 2024?",
    "What were Apple's cash and cash equivalents in FY2024?",
])
def test_a_metric_named_in_the_company_s_own_words_reaches_the_store(question, store):
    # Every FINANCIAL_METRICS alias is a numeric cue, so these are classified
    # numeric rather than factual and take the lookup. Read as prose questions
    # they never reached the store that holds the answer.
    from src.rag.query import parse_question

    assert parse_question(question).question_type == "numeric"


def test_a_supplied_query_decides_which_figure_is_looked_up(store):
    # The caller narrowed the search to FY2021; the store has no FY2021 revenue,
    # so the question must not be answered with FY2024's figure.
    from src.rag.query import parse_question

    query = parse_question(QUESTION).to_query(top_k=8)
    narrowed = type(query)(text=query.text, tickers=("AAPL",), fiscal_years=(2021,), top_k=8)
    answer = answer_from_facts(QUESTION, narrowed.tickers, narrowed.fiscal_years,
                               StubRetriever(_passage()), facts_file=store())
    assert answer is None


def test_a_scaled_figure_is_cited_from_a_row_that_names_the_metric():
    # Apple's FY2024 income statement as the corpus holds it: the "(In
    # millions)" caption is not in the passage, as for 73% of table passages.
    from src.rag.constants import FINANCIAL_METRICS
    from src.retrieval.facts import printed_forms_by_scale

    text = ("CONSOLIDATED STATEMENTS OF OPERATIONS (part 1 of 3)\n\n"
            "| Years ended | September 28, 2024 | September 30, 2023 |\n"
            "| --- | --- | --- |\n"
            "| Total net sales | 391,035 | 383,285 |\n"
            "| Research and development | 31,370 | 29,915 |\n")
    revenue = FINANCIAL_METRICS["revenue"].aliases
    by_scale = printed_forms_by_scale(Decimal("391035000000"))
    assert prints_figure(text, by_scale, labels=revenue)
    assert not prints_figure(text, by_scale)
    assert not prints_figure(text.replace("Total net sales", "Deferred items"), by_scale,
                             labels=revenue)
    assert not prints_figure("(In thousands)\n" + text, by_scale, labels=revenue)
    loss = "| Net income (loss) | $ | 21,331 | $ | 33,364 | $(2,722) |\n"
    assert prints_figure(loss, printed_forms_by_scale(Decimal("-2722000000")), negative=True,
                         labels=FINANCIAL_METRICS["net_income"].aliases)


def test_a_figure_outside_a_table_row_still_needs_its_scale_named():
    # The row check is a table rule: prose that merely mentions the line item
    # near the digits says nothing about what scale they are in.
    from src.rag.constants import FINANCIAL_METRICS
    from src.retrieval.facts import printed_forms_by_scale

    revenue = FINANCIAL_METRICS["revenue"].aliases
    by_scale = printed_forms_by_scale(Decimal("391035000000"))
    assert not prints_figure("Total net sales of 391,035 were reported.", by_scale, labels=revenue)
    assert prints_figure("Total net sales of $391,035 million were reported.", by_scale,
                         labels=revenue)


def test_a_figure_printed_to_a_decimal_of_a_million_is_cited_from_its_row():
    # Palo Alto Networks reports in millions to one decimal place, so its
    # total assets of 10,241,600,000 are never printed as a whole number.
    from src.rag.constants import FINANCIAL_METRICS
    from src.retrieval.facts import printed_forms_by_scale

    text = ("CONSOLIDATED BALANCE SHEETS (part 2 of 5)\n\n"
            "| July 31, | 2021 |  | 2020 |  |\n"
            "| --- | --- | --- | --- | --- |\n"
            "| Total assets | $ | 10,241.6 | $ | 9,065.4 |\n")
    assets = FINANCIAL_METRICS["assets"].aliases
    by_scale = printed_forms_by_scale(Decimal("10241600000"))
    assert prints_figure(text, by_scale, labels=assets)
    assert prints_figure("(in millions)\n" + text, by_scale, labels=assets)
    assert not prints_figure("(in thousands)\n" + text, by_scale, labels=assets)
    # A declared scale is not enough for a decimal, as it is for a whole
    # figure: a table in millions holds rates written the same way.
    other_row = "(in millions)\n\n| Deferred items | 10,241.6 |\n"
    assert not prints_figure(other_row, by_scale, labels=assets)
    in_thousands = "(in thousands)\n\n| Deferred items | 10,241,600 |\n"
    assert prints_figure(in_thousands, by_scale, labels=assets)
    # Prose carries its scale beside the figure.
    assert prints_figure("Total assets were $10,241.6 million.", by_scale)
    # A loss printed to a decimal keeps the sign rule.
    loss = "| Net loss | $ | -498.9 | $ | -267 |\n"
    net_income = FINANCIAL_METRICS["net_income"].aliases
    assert prints_figure(loss, printed_forms_by_scale(Decimal("-498900000")), negative=True,
                         labels=net_income)
    assert not prints_figure(loss, printed_forms_by_scale(Decimal("498900000")), labels=net_income)


def test_a_statement_ranked_below_thirty_passages_is_still_found():
    # A search for the line item's name ranks the prose that uses those words
    # above a statement that words the line differently. Thirty such passages
    # put the table out of reach of a search twenty deep, with the figure in
    # the store and nothing to cite it from.
    from src.retrieval.bm25 import BM25Retriever
    from src.retrieval.constants import TABLE_BOOST
    from src.retrieval.records import Query

    table = _passage("Statement (in millions)\n| | 2024 |\n| Total net sales | 391,035 |")
    chunks = [dataclasses.asdict(table)]
    for index in range(120):
        text = ("Total revenue discussion about annual performance and future business prospects."
                if index < 30 else "Employees technology operations research strategy competition.")
        chunks.append(dataclasses.asdict(_passage(
            text=text, chunk_id=f"{ACCESSION}_item7_{index}", content_type="prose")))
    retriever = BM25Retriever(chunks)
    ranked = [p.chunk_id for p in retriever.search(
        Query("total revenue", top_k=len(chunks), table_boost=TABLE_BOOST))]
    assert ranked.index(table.chunk_id) == 30
    found = supporting_passage(FACT, retriever)
    assert found is not None and found.chunk_id == table.chunk_id


def test_the_search_for_a_passage_stays_within_the_candidates_already_fetched():
    # Hybrid and a reranker fetch CANDIDATE_K candidates for any smaller
    # request, so looking through that many costs nothing more. Past it, a
    # reranker would score every extra passage.
    from src.retrieval.constants import CANDIDATE_K

    retriever = StubRetriever(_passage())
    supporting_passage(FACT, retriever)
    assert retriever.queries[0].top_k <= CANDIDATE_K


def test_the_supporting_search_carries_the_table_boost():
    # Every other numeric query gets it through ParsedQuestion.to_query; without
    # it prose repeating the line item can push the table out of the top k.
    from src.retrieval.constants import TABLE_BOOST

    retriever = StubRetriever(_passage())
    supporting_passage(FACT, retriever)
    assert retriever.queries[0].table_boost == TABLE_BOOST


def test_the_answer_records_the_route_for_the_harness(store):
    # evaluate() reads Answer.config.provider to count which route answered.
    answer = answer_from_facts(QUESTION, ("AAPL",), (2024,), StubRetriever(_passage()),
                               facts_file=store())
    assert answer.config.provider == FACTS_PROVIDER


def test_a_supplied_parse_is_not_read_again(store, monkeypatch):
    from src.rag.query import parse_question

    parsed = parse_question(QUESTION)
    monkeypatch.setattr("src.rag.answer.parse_question",
                        lambda *a, **k: pytest.fail("the question was read twice"))
    answer = answer_question(QUESTION, StubRetriever(_passage()), facts_file=store(),
                             parsed=parsed)
    assert answer.config.provider == FACTS_PROVIDER


def test_the_evaluation_command_can_turn_the_route_off(monkeypatch, tmp_path):
    # The ablation has to be runnable from the command line, not only from
    # evaluate()'s signature.
    import src.evaluation.cli as cli
    from src.retrieval.bm25 import BM25Retriever

    seen = {}
    monkeypatch.setattr(cli, "load_questions", lambda *a, **k: [])
    monkeypatch.setattr(BM25Retriever, "load", lambda **k: object())
    monkeypatch.setattr(cli, "evaluate", lambda *a, **k: seen.update(k) or
                        {"summary": {}, "by_answerability": {}, "results": []})
    cli.main(["q.jsonl", "--retriever", "bm25", "--run-id", "r",
              "--output", str(tmp_path / "report.json"), "--no-facts", "--provider", "ollama"])
    assert seen["use_facts"] is False


def test_the_evaluation_command_uses_the_route_by_default(monkeypatch, tmp_path):
    import src.evaluation.cli as cli
    from src.retrieval.bm25 import BM25Retriever

    seen = {}
    monkeypatch.setattr(cli, "load_questions", lambda *a, **k: [])
    monkeypatch.setattr(BM25Retriever, "load", lambda **k: object())
    monkeypatch.setattr(cli, "evaluate", lambda *a, **k: seen.update(k) or
                        {"summary": {}, "by_answerability": {}, "results": []})
    cli.main(["q.jsonl", "--retriever", "bm25", "--run-id", "r",
              "--output", str(tmp_path / "report.json"), "--provider", "ollama"])
    assert seen["use_facts"] is True
