"""Split downloaded filings into their numbered Items, into ``data/interim/``.

Run it from the project root, after ``src.pipeline.download``:

    python -m src.pipeline.parse                      # every filing in the manifest
    python -m src.pipeline.parse --tickers AAPL MSFT  # just these companies
    python -m src.pipeline.parse --force              # re-parse filings already done

A 10-K is one long HTML document, but a question is nearly always about one
part of it: Item 1A for risks, Item 7 for management's discussion, and so on.
This stage cuts each filing along those Item boundaries and writes one JSON
file per filing to ``data/interim/<TICKER>/``, carrying the filing's metadata
and one record per Item. Chunking, retrieval, and citations all read from here
rather than from the raw HTML.

The parsing runs entirely from the files already on disk, with no further calls
to EDGAR, so it can be re-run as often as the chunking strategy changes.

Not every Item holds the text it names. Many filers satisfy Item 8 with a
sentence pointing at the financial statements printed under Item 15, so the
Item parses cleanly but is nearly empty. Sections like that are marked
``is_stub``, and where the filing does hold the text under another Item, the
stub records a ``resolved_from`` pointer to it. The text is not copied across,
so the corpus keeps one copy of every passage and the chunking stage follows
the pointer instead.

Whatever cannot be resolved that way is reported in the run summary, along with
any key Item missing from the filing altogether, because a silently absent
Item 7 would otherwise look like a retrieval failure much later on.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, replace
from pathlib import Path

from edgar.documents import HTMLParser, ParserConfig

from ..config import INTERIM_DIR, PROJECT_ROOT, ensure_data_dirs
from .cli import build_parse_parser
from .constants import (
    EMPTY_ITEM_CHAR_LIMIT,
    EXTRA_ITEM_TITLES,
    FORM_STRUCTURES,
    INCORPORATION_FALLBACKS,
    INCORPORATION_PHRASES,
    KEY_ITEMS,
    STUB_CHAR_LIMIT,
    TABLE_MIN_CELLS,
)
from .download import load_manifest
from .records import FilingRecord, ParsedFiling, SectionRecord, TableRecord

logger = logging.getLogger(__name__)

_INCORPORATION = re.compile("|".join(INCORPORATION_PHRASES), re.IGNORECASE)


def item_title(form: str, part: str | None, item: str | None) -> str:
    """Look up the official title of an Item, for use in citations."""
    if not item:
        return ""

    structure = FORM_STRUCTURES.get(form)
    if structure is not None:
        entry = structure.get_item(f"ITEM {item}", f"PART {part}" if part else None)
        if entry and entry.get("Title"):
            return entry["Title"]

    # Some filers use Items the standard structure does not list, such as the
    # Item 4A several of them use for their executive officer list. Without this
    # the Item parses fine but cites with no title at all.
    return EXTRA_ITEM_TITLES.get(form, {}).get(item, "")


def _is_stub_text(text: str) -> bool:
    """Whether an Item holds nothing worth retrieving.

    Two different things look alike at a glance. An Item can be genuinely empty
    ("None.", "Not applicable."), or it can be a cross-reference that sends the
    reader to the proxy statement or to another Item. Both should be skipped.

    What must NOT be skipped is an Item that is simply short. Apple answers Item
    2 Properties in 488 characters of real fact about Cupertino, which a length
    threshold alone would throw away alongside the 303-character Item 12 that
    only points at the proxy statement. So length decides only the very shortest
    cases, and everything in between has to actually read as a cross-reference.
    """
    stripped = text.strip()
    if len(stripped) < EMPTY_ITEM_CHAR_LIMIT:
        return True
    return len(stripped) < STUB_CHAR_LIMIT and bool(_INCORPORATION.search(stripped))


def _format_cell(value: object) -> str:
    """Render one table cell the way the filing writes it."""
    if value is None:
        return ""
    # A whole number arrives from pandas as 245122.0; the filing prints 245,122.
    if isinstance(value, float):
        if value != value:          # NaN
            return ""
        if value == int(value):
            return f"{int(value):,}"
        return f"{value:,}"
    return str(value)


_YEAR_CELL = re.compile(r"^(19|20)\d{2}$")


def _looks_like_label(value: str) -> bool:
    """Is this cell a column label rather than a figure?

    A year counts as a label, because "2024" heading a column of figures is
    exactly what a financial table's header row looks like. Any other bare
    number does not, which is what keeps a row of figures from being mistaken
    for a header.
    """
    return bool(re.search(r"[A-Za-z]{3,}", value)) or bool(_YEAR_CELL.match(value))


def _promote_header_row(rendered):
    """Use the first row as the header where the table arrived without one.

    pandas numbers the columns 0, 1, 2 when it finds no header row in the HTML.
    A passage headed ``| 0 | 1 |`` says nothing about its figures, and a long
    table is split across passages with its header repeated on each one, so
    every slice after the first would otherwise carry no labels at all. Where
    the first row is plainly the header, promoting it fixes all of them.
    """
    if not all(str(column).strip().isdigit() for column in rendered.columns):
        return rendered
    first = [str(value).strip() for value in rendered.iloc[0]]
    filled = [value for value in first if value]
    if len(filled) < 2 or not all(_looks_like_label(value) for value in filled):
        return rendered

    promoted = rendered.iloc[1:]
    promoted.columns = first
    return promoted


def _extract_tables(section) -> list[TableRecord]:
    """Lift each table out of an Item as a table, not as flattened prose.

    The extractor's plain text runs a balance sheet together into a column of
    bare numbers with the row and column labels stripped away, so "245,122"
    arrives with nothing to say it is total revenue for 2024. Rebuilding each
    table from its own grid keeps those labels attached.

    A table that cannot be rebuilt is skipped rather than guessed at: the
    flattened version is still in the Item's text, so nothing is lost outright.
    """
    records: list[TableRecord] = []
    for index, table in enumerate(section.tables()):
        try:
            frame = table.to_dataframe()
        except Exception:
            logger.debug("table %d in %s could not be rebuilt", index, section.name)
            continue
        if frame is None or frame.empty or frame.size < TABLE_MIN_CELLS:
            continue

        # pandas renamed applymap to map in 2.1; support both.
        formatter = getattr(frame, "map", None) or frame.applymap
        try:
            rendered = formatter(_format_cell)
        except Exception:
            logger.debug("table %d in %s could not be rendered", index, section.name)
            continue

        # Filers lay out financial tables with spacer columns holding the
        # currency symbol or nothing at all, which arrive as empty columns that
        # push the year headers away from their figures. Dropping the columns
        # that hold no value anywhere puts each figure back under its own year.
        keep = [position for position in range(rendered.shape[1])
                if rendered.iloc[:, position].astype(str).str.strip().any()]
        if keep:
            rendered = rendered.iloc[:, keep]
        rendered = _promote_header_row(rendered)
        if rendered.empty or rendered.size < TABLE_MIN_CELLS:
            continue

        try:
            markdown = rendered.to_markdown()
        except Exception:
            logger.debug("table %d in %s could not be rendered", index, section.name)
            continue

        caption = str(getattr(table, "caption", "") or frame.index.name or "").strip()
        records.append(
            TableRecord(
                # Number the tables we actually kept, not their position among
                # every table found. A table that fails to rebuild is skipped
                # above, so counting `index` here would leave a record whose
                # number does not address it in this list: section.tables[n]
                # would return the wrong table, or raise.
                table_index=len(records),
                caption=caption,
                headers=[str(column) for column in rendered.columns],
                n_rows=int(rendered.shape[0]),
                n_cols=int(rendered.shape[1]),
                markdown=markdown,
            )
        )
    return records


def interim_path_for(record: FilingRecord, interim_dir: Path = INTERIM_DIR) -> Path:
    """Where a filing's parsed output belongs.

    The name mirrors the raw file so that the two are obviously a pair when the
    folders are opened side by side.
    """
    return interim_dir / record.ticker / (Path(record.path).stem + ".json")


def parse_filing(record: FilingRecord, project_root: Path = PROJECT_ROOT) -> ParsedFiling:
    """Split one downloaded filing into its Items."""
    source = project_root / record.path
    if source.suffix != ".html":
        # The download stage falls back to plain text for the rare filing with
        # no HTML document, and that has no Item markup to split on.
        raise ValueError(f"{source.name} is not HTML, so it has no Item structure to parse")

    # Telling the parser which form it is looking at is what lets it apply the
    # right Item layout, so a 10-Q is not read as a truncated 10-K.
    document = HTMLParser(ParserConfig(form=record.form)).parse(source.read_text(encoding="utf-8"))
    key_items = KEY_ITEMS.get(record.form, set())

    sections: list[SectionRecord] = []
    for section_id, section in document.sections.items():
        text = section.text()
        item = getattr(section, "item", None)
        sections.append(
            SectionRecord(
                section_id=section_id,
                part=getattr(section, "part", None),
                item=item,
                title=item_title(record.form, getattr(section, "part", None), item),
                text=text,
                n_chars=len(text),
                n_tables=len(section.tables()),
                is_key_section=item in key_items,
                is_stub=_is_stub_text(text),
                resolved_from=None,  # filled in by _resolve_stubs below
                confidence=getattr(section, "confidence", None),
                detection_method=getattr(section, "detection_method", None),
                validated=bool(getattr(section, "validated", False)),
                warnings=list(getattr(section, "warnings", []) or []),
                tables=_extract_tables(section),
            )
        )

    sections = _resolve_stubs(sections, record.form)

    return ParsedFiling(
        ticker=record.ticker,
        cik=record.cik,
        company=record.company,
        form=record.form,
        filing_date=record.filing_date,
        accession_no=record.accession_no,
        url=record.url,
        source_path=record.path,
        sections=sections,
        period_of_report=record.period_of_report,
    )


def _resolve_stubs(sections: list[SectionRecord], form: str) -> list[SectionRecord]:
    """Point each cross-referencing Item at the Item that holds its text.

    Only a stub is redirected, and only when the target Item is substantial, so
    a filing that genuinely prints its statements under Item 8 is left alone.
    """
    fallbacks = INCORPORATION_FALLBACKS.get(form, {})
    if not fallbacks:
        return sections

    by_item = {section.item: section for section in sections if section.item}

    resolved: list[SectionRecord] = []
    for section in sections:
        target_item = fallbacks.get(section.item or "")
        target = by_item.get(target_item) if target_item else None
        if section.is_stub and target is not None and not target.is_stub:
            section = replace(section, resolved_from=target.section_id)
        resolved.append(section)
    return resolved


def write_parsed(parsed: ParsedFiling, destination: Path) -> None:
    """Write one parsed filing as JSON."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(asdict(parsed), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def parse_all(
    records: list[FilingRecord],
    force: bool = False,
    interim_dir: Path = INTERIM_DIR,
) -> list[ParsedFiling]:
    """Parse every filing given, carrying on if one of them fails."""
    ensure_data_dirs()

    parsed_filings: list[ParsedFiling] = []
    for record in records:
        destination = interim_path_for(record, interim_dir)
        if destination.exists() and not force:
            logger.info("%s %s: already parsed, skipping", record.ticker, record.filing_date)
            continue

        try:
            parsed = parse_filing(record)
        except Exception:
            # A single unparseable filing should not cost us the rest of the
            # corpus; the summary at the end reports how many landed.
            logger.exception("%s %s: parsing failed, moving on", record.ticker, record.filing_date)
            continue

        write_parsed(parsed, destination)
        parsed_filings.append(parsed)

        key_sections = [s for s in parsed.sections if s.is_key_section]
        logger.info(
            "%s %s: %d sections, %d key, %d chars -> %s",
            record.ticker, record.filing_date, len(parsed.sections), len(key_sections),
            sum(s.n_chars for s in parsed.sections), destination.name,
        )

    return parsed_filings


def _report(parsed_filings: list[ParsedFiling]) -> None:
    """Print what was parsed and, more usefully, what looks wrong with it."""
    if not parsed_filings:
        print("\nNothing new was parsed. Use --force to re-parse filings already done.")
        return

    print(f"\nParsed {len(parsed_filings)} filings into {INTERIM_DIR}")

    # Tables are rebuilt so their figures keep their row and column labels. One
    # the rebuilder cannot handle is not lost, since the flattened copy stays in
    # the Item's text, but it is worth knowing how many there were.
    found = sum(section.n_tables for filing in parsed_filings for section in filing.sections)
    rebuilt = sum(len(section.tables) for filing in parsed_filings for section in filing.sections)
    if found:
        print(f"  {rebuilt} of {found} tables rebuilt with their labels intact "
              f"({rebuilt / found:.0%})")

    resolved = [
        (filing, section)
        for filing in parsed_filings
        for section in filing.sections
        if section.resolved_from
    ]
    if resolved:
        print(f"\n{len(resolved)} cross-referencing Items were resolved to the Item holding their text:")
        for filing, section in resolved:
            print(f"  {filing.ticker} {filing.filing_date}  Item {section.item} -> {section.resolved_from}")

    # A key Item that is absent, still empty, or badly detected will look like a
    # retrieval failure much later, so it is named here while it is still cheap
    # to do something about it. A missing Item has no section to inspect, so it
    # has to be found by comparing against the Items the form should have.
    problems: list[tuple[ParsedFiling, str, str]] = []
    for filing in parsed_filings:
        present = {section.item for section in filing.sections if section.item}
        for item in sorted(KEY_ITEMS.get(filing.form, set()) - present):
            problems.append((filing, f"Item {item}", "missing from the filing"))
        for section in filing.sections:
            if not section.is_key_section or section.resolved_from:
                continue
            if section.is_stub:
                problems.append((filing, f"Item {section.item}",
                                 f"{section.n_chars} chars, nothing to fall back on"))
            elif section.warnings:
                problems.append((filing, f"Item {section.item}", section.warnings[0]))

    if not problems:
        print("Every key Item came through with content.")
        return

    print(f"\n{len(problems)} key Items need a look:")
    for filing, item, reason in problems:
        print(f"  {filing.ticker} {filing.filing_date}  {item:<8} {reason}")


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    parser = build_parse_parser(__doc__.splitlines()[0] if __doc__ else "")
    args = parser.parse_args(argv)

    records = load_manifest()
    if not records:
        print("The manifest is empty. Run python -m src.pipeline.download first.")
        return

    if args.tickers:
        wanted = {ticker.upper() for ticker in args.tickers}
        records = [record for record in records if record.ticker in wanted]
    if args.forms:
        wanted_forms = set(args.forms)
        records = [record for record in records if record.form in wanted_forms]
    if not records:
        # Same reasoning as in chunk.main: say that the filter matched nothing,
        # rather than letting the report suggest --force.
        print(
            "No downloaded filing matches those filters. "
            "Run python -m src.pipeline.download for them first."
        )
        return

    logger.info("Parsing %d filings from the manifest", len(records))
    _report(parse_all(records, force=args.force))


if __name__ == "__main__":
    main()
