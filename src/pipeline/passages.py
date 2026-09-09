"""Show passages from ``data/processed/``, for spot-checking before indexing.

Run it from the project root:

    python -m src.pipeline passages --item 8 --tickers AAPL
    python -m src.pipeline passages --contains "going concern" --limit 3 --full
    python -m src.pipeline passages --content-type table --item 8 --json > tables.jsonl

Chunking decides what retrieval can ever return: a passage cut in the wrong
place, or one that turns out to be a page footer, is a wrong answer that no
amount of retrieval tuning fixes. The way to catch that is to read a few, and
before this the only way to read one was to open a JSON file and scroll. This
command puts a filter over the corpus instead, so a question like "what do
Item 8 table passages look like for Apple" is one line rather than a script.

It reads the corpus and changes nothing.
"""

from __future__ import annotations

import json
import textwrap

from .chunk import iter_chunks

# How much of a passage to show when it is not printed in full. Enough to tell
# whether the cut landed sensibly and whether the heading matches the text,
# which is what spot-checking is for.
PREVIEW_CHARS = 400


def select(
    tickers: list[str] | None = None,
    items: list[str] | None = None,
    forms: list[str] | None = None,
    fiscal_years: range | list[int] | None = None,
    content_type: str | None = None,
    contains: str | None = None,
    key_items_only: bool = False,
) -> list[dict]:
    """Every passage matching the filters, in corpus order.

    The filters are named parameters rather than a parsed argument namespace, so
    this is as usable from a notebook or the retrieval stage as from the command
    line, which is where the filters happen to come from today.
    """
    wanted_items = {item.upper() for item in items} if items else None
    wanted_forms = set(forms) if forms else None
    needle = contains.lower() if contains else None

    matched = []
    for chunk in iter_chunks(
        fiscal_years=fiscal_years,
        tickers=tickers,
        key_items_only=key_items_only,
    ):
        if wanted_items and (chunk["item"] or "").upper() not in wanted_items:
            continue
        if wanted_forms and chunk["form"] not in wanted_forms:
            continue
        if content_type and chunk["content_type"] != content_type:
            continue
        if needle and needle not in chunk["text"].lower():
            continue
        matched.append(chunk)
    return matched


def _show(chunk: dict, full: bool) -> None:
    """Print one passage with the identity a citation would need."""
    year = chunk["fiscal_year"] or chunk["filing_date"][:4]
    item = f"Item {chunk['item']}" if chunk["item"] else chunk["section_id"]
    kind = "" if chunk["content_type"] == "prose" else f"  [{chunk['content_type']}]"

    print(f"\n{'─' * 78}")
    print(f"{chunk['ticker']}  FY{year}  {chunk['form']}  {item} · {chunk['title']}{kind}")
    if chunk["heading"]:
        print(f"under: {chunk['heading']}")
    if chunk["incorporated_into"]:
        # Oracle's Item 8 points at Item 15, so a reader searching for Item 8
        # needs to know this passage answers for it.
        print(f"also answers Item {', '.join(chunk['incorporated_into'])}")
    print(f"{chunk['chunk_id']}  ({chunk['n_chars']} chars)")
    print()

    text = chunk["text"]
    if full or len(text) <= PREVIEW_CHARS:
        print(text)
    else:
        # Cut at a line break so a table keeps its rows intact and prose does
        # not break mid-word.
        head = text[:PREVIEW_CHARS]
        head = head[:head.rfind("\n") + 1] or head
        print(head.rstrip())
        print(f"... {len(text) - len(head)} more characters, use --full to see them")


def report(
    matched: list[dict], limit: int = 10, full: bool = False, as_json: bool = False,
) -> None:
    """Print the selected passages, or write them as JSON lines.

    ``limit`` of 0 means all of them. The JSON form is one object per line and
    carries no summary, so it can be piped into another tool unchanged.
    """
    shown = matched if limit == 0 else matched[:limit]

    if as_json:
        for chunk in shown:
            print(json.dumps(chunk))
        return

    for chunk in shown:
        _show(chunk, full)

    filings = {chunk["accession_no"] for chunk in matched}
    print(f"\n{'─' * 78}")
    summary = (f"{len(matched)} passages across {len(filings)} filings, "
               f"{sum(chunk['n_chars'] for chunk in matched):,} characters")
    print(textwrap.fill(summary, 78))
    if len(shown) < len(matched):
        print(f"Showing the first {len(shown)}. Use --limit to see more, or --limit 0 for all.")
