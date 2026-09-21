"""The passage-size gate counts tokens as the encoder will (#64)."""

from __future__ import annotations

import pytest

from src.pipeline.chunk import chunk_tables
from src.pipeline.constants import CHUNK_CHAR_BUDGET
from src.pipeline.records import SectionRecord, TableRecord
from src.pipeline.verify import (
    OVERRUN_TOLERANCE,
    bge_token_counter,
    check_passage_sizes,
    check_statement_titles,
)

FILING = {"ticker": "AAA", "company": "Alpha Corp", "fiscal_year": 2024, "form": "10-K"}


def passage(text, content_type="prose"):
    return {**FILING, "item": "7", "title": "MD&A", "text": text, "n_chars": len(text),
            "content_type": content_type}


def counting(sizes):
    """A token counter that answers from a list, in order."""
    return lambda texts: list(sizes)


def test_passes_when_everything_fits():
    corpus = [passage("a"), passage("b", "table")]
    check = check_passage_sizes(corpus, count_tokens=counting([100, 200]))
    assert check.passed
    assert "prose 0 of 1" in check.detail and "table 0 of 1" in check.detail


def test_a_single_table_over_the_window_fails():
    corpus = [passage("a"), passage("b", "table")]
    check = check_passage_sizes(corpus, count_tokens=counting([100, 513]))
    assert not check.passed
    assert "table" in check.failures[0]


def test_prose_is_judged_on_its_own_share():
    n = 1000
    allowed = int(n * OVERRUN_TOLERANCE["prose"])
    corpus = [passage(str(i)) for i in range(n)]
    assert check_passage_sizes(corpus, counting([600] * allowed + [10] * (n - allowed))).passed
    assert not check_passage_sizes(corpus, counting([600] * (allowed + 1) + [10] * (n - allowed - 1))).passed


def test_the_header_is_counted():
    """The encoder reads the context header too, so the gate must."""
    seen = []
    check_passage_sizes([passage("body text")], count_tokens=lambda texts: seen.extend(texts) or [1])
    assert seen[0].startswith("Alpha Corp (AAA) FY2024 10-K, Item 7") and seen[0].endswith("body text")


def dense_section() -> SectionRecord:
    headers = ["(In millions)"] + [f"Q{q} {y}" for y in (2024, 2023) for q in (1, 2, 3, 4)]
    rows = [[f"Segment {i}"] + [f"${1000 * i + j:,}.{j}" for j in range(8)] for i in range(40)]
    table = TableRecord(table_index=0, caption="Quarterly results", headers=headers,
                        rows=rows, n_rows=len(rows), n_cols=len(headers))
    return SectionRecord(
        section_id="part_ii_item_8", part="II", item="8", title="Financial Statements",
        text="", n_chars=0, n_tables=1, n_data_tables=1, is_key_section=True, is_stub=False,
        resolved_from=None, confidence=None, detection_method=None, validated=True,
        tables=[table],
    )


def as_rows(chunks):
    return [{**FILING, "item": c.item, "title": c.title, "text": c.text,
             "n_chars": c.n_chars, "content_type": c.content_type} for c in chunks]


def test_reverting_the_table_budget_fails_the_gate():
    """The regression #64 asks the gate to catch, with the real tokenizer."""
    try:
        counter = bge_token_counter()
    except Exception as error:   # no network and no cached tokenizer
        pytest.skip(f"tokenizer unavailable: {error}")

    cut_right = as_rows(chunk_tables(dense_section(), accession_no="acc"))
    cut_wrong = as_rows(chunk_tables(dense_section(), accession_no="acc",
                                     table_budget=CHUNK_CHAR_BUDGET))
    assert check_passage_sizes(cut_right, count_tokens=counter).passed
    assert not check_passage_sizes(cut_wrong, count_tokens=counter).passed


# --- statement titles (#88) -----------------------------------------------------

def table_passage(accession: str, heading: str | None, item: str = "8") -> dict:
    text = "| Total assets | $364,980 |"
    return {"chunk_id": f"{accession}_part_ii_item_{item}_t000_00", "content_type": "table",
            "heading": heading, "item": item, "text": text, "n_chars": len(text)}


def filing_passages(accession: str, *headings: str | None, item: str = "8") -> list[dict]:
    return [table_passage(accession, heading, item) for heading in headings]


ALL_THREE = ("CONSOLIDATED BALANCE SHEETS", "CONSOLIDATED STATEMENTS OF OPERATIONS",
             "CONSOLIDATED STATEMENTS OF CASH FLOWS")


def test_a_filing_naming_all_three_statements_passes():
    assert check_statement_titles(filing_passages("acc0", *ALL_THREE, None)).passed


def test_a_filing_missing_a_statement_is_named_with_what_it_lacks():
    corpus = filing_passages("acc0", "CONSOLIDATED BALANCE SHEETS")
    check = check_statement_titles(corpus)
    assert not check.passed
    assert "acc0" in check.failures[0]
    assert "cash flow statement" in check.failures[0] and "income statement" in check.failures[0]


def test_statements_outside_item_8_still_count():
    """Oracle's Item 8 is a cross-reference: its statements are under Item 15."""
    assert check_statement_titles(filing_passages("acc0", *ALL_THREE, item="15")).passed


def test_one_bad_filing_among_many_stays_inside_the_tolerance():
    """One filer whose layout has no readable heading is not a corpus failure."""
    corpus = [p for i in range(39) for p in filing_passages(f"acc{i:02d}", *ALL_THREE)]
    corpus += filing_passages("bad", "Financial Statements")
    check = check_statement_titles(corpus)
    assert check.passed and "39 of 40" in check.detail


def test_enough_bad_filings_fail_the_gate():
    corpus = [p for i in range(17) for p in filing_passages(f"acc{i:02d}", *ALL_THREE)]
    corpus += [p for i in range(3) for p in filing_passages(f"bad{i}", "Financial Statements")]
    assert not check_statement_titles(corpus).passed


def test_an_untitled_note_does_not_fail_a_filing_that_named_its_statements():
    corpus = filing_passages("acc0", *ALL_THREE, "Segment detail", None, None)
    assert check_statement_titles(corpus).passed


def test_a_corpus_with_no_table_passages_fails_loudly():
    """No tables at all is the case the gate should be loudest about."""
    check = check_statement_titles([passage("some prose")])
    assert not check.passed
    assert "no table passages" in check.detail


def test_a_standalone_comprehensive_income_statement_is_not_the_income_statement():
    """A filer that sets comprehensive income apart still owes an income statement."""
    corpus = filing_passages(
        "acc0", "CONSOLIDATED BALANCE SHEETS",
        "CONSOLIDATED STATEMENTS OF COMPREHENSIVE INCOME",
        "CONSOLIDATED STATEMENTS OF CASH FLOWS",
    )
    check = check_statement_titles(corpus)
    assert not check.passed
    assert "income statement" in check.failures[0]


def test_a_heading_combining_operations_and_comprehensive_income_counts():
    """The combined heading is an income statement, and must still satisfy the gate."""
    corpus = filing_passages(
        "acc0", "CONSOLIDATED BALANCE SHEETS",
        "CONSOLIDATED STATEMENTS OF OPERATIONS AND COMPREHENSIVE INCOME",
        "CONSOLIDATED STATEMENTS OF CASH FLOWS",
    )
    assert check_statement_titles(corpus).passed


def test_a_comprehensive_income_statement_that_is_the_income_statement_counts():
    """ServiceNow presents one statement, which ASC 220 allows: its rows say so."""
    combined = {
        **table_passage("acc0", "CONSOLIDATED STATEMENTS OF COMPREHENSIVE INCOME"),
        "text": "\n".join([
            "| Total revenues | 10,984 |",
            "| Gross profit | 8,600 |",
            "| Income from operations | 1,340 |",
            "| Net income per share - basic | 6.12 |",
            "| Other comprehensive income (loss): |  |",
        ]),
    }
    corpus = [combined] + filing_passages(
        "acc0", "CONSOLIDATED BALANCE SHEETS", "CONSOLIDATED STATEMENTS OF CASH FLOWS",
    )
    assert check_statement_titles(corpus).passed


def test_a_pension_row_does_not_make_an_oci_statement_an_income_statement():
    """"Prior service cost of defined benefit plans" is an OCI row (TXN)."""
    oci = {
        **table_passage("acc0", "Consolidated Statements of Comprehensive Income"),
        "text": "\n".join([
            "| Net income | 4,075 |",
            "| Prior service cost of defined benefit plans: |  |",
            "| Foreign currency translation adjustments | 12 |",
        ]),
    }
    corpus = [oci] + filing_passages(
        "acc0", "CONSOLIDATED BALANCE SHEETS", "CONSOLIDATED STATEMENTS OF CASH FLOWS",
    )
    check = check_statement_titles(corpus)
    assert not check.passed and "income statement" in check.failures[0]
