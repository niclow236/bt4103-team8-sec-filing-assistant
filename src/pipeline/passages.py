"""Show passages from ``data/processed/``, for spot-checking before indexing.

Run it from the project root:

    python -m src.pipeline passages --item 8 --ticker AAPL
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

from ..config import PROCESSED_DIR
from .chunk import iter_chunks
from .cli import build_passages_parser

# How much of a passage to show when it is not printed in full. Enough to tell
# whether the cut landed sensibly and whether the heading matches the text,
# which is what spot-checking is for.
PREVIEW_CHARS = 400


def select(args) -> list[dict]:
    """Every passage matching the filters, in corpus order."""
    fiscal_years = (
        range(args.fiscal_years[0], args.fiscal_years[1] + 1)
        if args.fiscal_years else None
    )
    wanted_items = {item.upper() for item in args.item} if args.item else None
    wanted_forms = set(args.forms) if args.forms else None
    needle = args.contains.lower() if args.contains else None

    matched = []
    for chunk in iter_chunks(
        fiscal_years=fiscal_years,
        tickers=args.tickers,
        key_items_only=args.key_items_only,
    ):
        if wanted_items and (chunk["item"] or "").upper() not in wanted_items:
            continue
        if wanted_forms and chunk["form"] not in wanted_forms:
            continue
        if args.content_type and chunk["content_type"] != args.content_type:
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


def main(argv: list[str] | None = None) -> None:
    args = build_passages_parser(__doc__.splitlines()[0] if __doc__ else "").parse_args(argv)

    if not any(PROCESSED_DIR.glob("*/*.json")):
        print("data/processed/ is empty. Run python -m src.pipeline.chunk first.")
        return

    matched = select(args)
    if not matched:
        print("No passage matches those filters.")
        return

    shown = matched if args.limit == 0 else matched[:args.limit]

    if args.json:
        for chunk in shown:
            print(json.dumps(chunk))
        return

    for chunk in shown:
        _show(chunk, args.full)

    filings = {chunk["accession_no"] for chunk in matched}
    print(f"\n{'─' * 78}")
    summary = (f"{len(matched)} passages across {len(filings)} filings, "
               f"{sum(chunk['n_chars'] for chunk in matched):,} characters")
    print(textwrap.fill(summary, 78))
    if len(shown) < len(matched):
        print(f"Showing the first {len(shown)}. Use --limit to see more, or --limit 0 for all.")


if __name__ == "__main__":
    main()
