"""The passage-size gate counts tokens as the encoder will (#64)."""

from __future__ import annotations

import pytest

from src.pipeline.chunk import chunk_tables
from src.pipeline.constants import CHUNK_CHAR_BUDGET
from src.pipeline.records import SectionRecord, TableRecord
from src.pipeline.verify import OVERRUN_TOLERANCE, bge_token_counter, check_passage_sizes

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
