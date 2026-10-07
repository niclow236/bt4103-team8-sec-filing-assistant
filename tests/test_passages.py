"""Reading passages back (#49): the filters ``python -m src.pipeline passages`` takes.

The command exists to answer "what would this index hold", so its filters have
to select what an index built with the same filters would. They are checked
here against the suite's synthetic corpus: 3 companies, 2 fiscal years each,
and in every filing 4 passages from each of Items 1, 1A and 7 and a 3-part
table in Item 8.
"""

from __future__ import annotations

import json

import pytest

from conftest import edit_filing
from src.pipeline import passages
from src.pipeline.chunk import iter_chunks


@pytest.fixture
def select(corpus, monkeypatch):
    """``passages.select`` reading the synthetic corpus, not ``data/processed/``."""
    monkeypatch.setattr(passages, "iter_chunks",
                        lambda **filters: iter_chunks(processed_dir=corpus, **filters))
    return passages.select


def test_no_filter_selects_every_passage_in_corpus_order(select, corpus):
    assert select() == list(iter_chunks(processed_dir=corpus))
    assert len(select()) == 90


@pytest.mark.parametrize("filters, count", [
    ({"tickers": ["AAA"]}, 30),
    ({"tickers": ["aaa", "CCC"]}, 60),
    ({"fiscal_years": [2024]}, 45),
    ({"fiscal_years": range(2023, 2025)}, 90),
    ({"items": ["8"]}, 18),
    ({"items": ["1a", "7"]}, 48),              # an Item is matched whatever its case
    ({"forms": ["10-K"]}, 90),
    ({"forms": ["10-Q"]}, 0),
    ({"content_type": "table"}, 18),
    ({"content_type": "prose"}, 72),
    ({"contains": "REVENUE BY SEGMENT"}, 18),   # and so is the text looked for
    ({"contains": "paragraph 3"}, 18),
    ({"key_items_only": True}, 90),
    ({"tickers": ["BBB"], "fiscal_years": [2023], "items": ["7"], "contains": "topic 2"}, 1),
    ({"tickers": ["ZZZ"]}, 0),
])
def test_each_filter_narrows_the_selection_and_together_they_all_hold(select, filters, count):
    matched = select(**filters)
    assert len(matched) == count
    for chunk in matched:
        if "tickers" in filters:
            assert chunk["ticker"] in {ticker.upper() for ticker in filters["tickers"]}
        if "fiscal_years" in filters:
            assert chunk["fiscal_year"] in set(filters["fiscal_years"])
        if "items" in filters:
            assert chunk["item"] in {item.upper() for item in filters["items"]}
        if "content_type" in filters:
            assert chunk["content_type"] == filters["content_type"]
        if "contains" in filters:
            assert filters["contains"].lower() in chunk["text"].lower()


def test_only_the_targeted_items_are_selected_when_asked(select, corpus):
    path = sorted(corpus.glob("AAA/*.json"))[0]
    edit_filing(path, lambda data: data["chunks"][0].update(is_key_section=False))

    assert len(select(key_items_only=True)) == 89
    assert len(select()) == 90


# --- how a selection is shown -----------------------------------------------


def test_a_passage_is_shown_with_what_a_citation_of_it_would_name(select, capsys):
    matched = select(tickers=["AAA"], fiscal_years=[2024], items=["1"], contains="paragraph 1")
    passages.report(matched)
    printed = capsys.readouterr().out

    assert "AAA  FY2024  10-K  Item 1 · Business" in printed
    assert "under: Heading 1" in printed
    assert f"{matched[0]['chunk_id']}  ({matched[0]['n_chars']} chars)" in printed
    assert matched[0]["text"] in printed
    assert f"1 passages across 1 filings, {matched[0]['n_chars']:,} characters" in printed


def test_a_table_passage_is_marked_as_one(select, capsys):
    passages.report(select(tickers=["AAA"], fiscal_years=[2024], content_type="table"))
    printed = capsys.readouterr().out
    assert printed.count("Item 8 · Financial Statements  [table]") == 3
    assert "[prose]" not in printed


def test_a_passage_that_answers_for_another_item_says_so(capsys):
    chunk = {
        "chunk_id": "acc_part_iv_item_15_000", "ticker": "ORCL", "fiscal_year": None,
        "filing_date": "2025-06-20", "form": "10-K", "item": None, "section_id": "signatures",
        "title": "Exhibits", "content_type": "prose", "heading": None,
        "incorporated_into": ["8"], "n_chars": 12, "text": "Short text.", "accession_no": "acc",
    }
    passages.report([chunk])
    printed = capsys.readouterr().out
    assert "also answers Item 8" in printed
    # With no fiscal year or Item on record, the filing year and the section stand in.
    assert "ORCL  FY2025  10-K  signatures · Exhibits" in printed
    assert "under:" not in printed


def test_a_long_passage_is_cut_at_a_line_break_unless_shown_in_full(capsys):
    lines = [f"| Row {number} | {1000 + number:,} |" for number in range(60)]
    chunk = {
        "chunk_id": "acc_t000_00", "ticker": "AAA", "fiscal_year": 2024, "filing_date": "",
        "form": "10-K", "item": "8", "section_id": "part_ii_item_8", "title": "Statements",
        "content_type": "table", "heading": None, "incorporated_into": [],
        "n_chars": len("\n".join(lines)), "text": "\n".join(lines), "accession_no": "acc",
    }

    passages.report([chunk])
    preview = capsys.readouterr().out
    shown = [line for line in preview.splitlines() if line.startswith("| Row")]
    assert 0 < len(shown) < 60 and shown == lines[:len(shown)]      # whole rows only
    hidden = len(chunk["text"]) - len("\n".join(shown)) - 1
    assert f"... {hidden} more characters, use --full to see them" in preview

    passages.report([chunk], full=True)
    assert capsys.readouterr().out.count("| Row") == 60


def test_the_limit_shows_the_first_few_and_says_how_many_matched(select, capsys):
    matched = select(tickers=["AAA"])

    passages.report(matched, limit=2)
    printed = capsys.readouterr().out
    assert printed.count(" chars)") == 2
    assert "30 passages across 2 filings" in printed
    assert "Showing the first 2. Use --limit to see more, or --limit 0 for all." in printed

    passages.report(matched, limit=0)
    everything = capsys.readouterr().out
    assert everything.count(" chars)") == 30 and "Showing the first" not in everything


def test_json_output_is_one_passage_a_line_and_nothing_else(select, capsys):
    matched = select(tickers=["CCC"], items=["8"])
    passages.report(matched, limit=0, as_json=True)

    lines = capsys.readouterr().out.splitlines()
    assert [json.loads(line) for line in lines] == matched
    passages.report(matched, limit=1, as_json=True)
    assert len(capsys.readouterr().out.splitlines()) == 1
