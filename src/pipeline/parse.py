"""Split downloaded filings into their numbered Items, into ``data/interim/``.

Run it from the project root, after ``src.pipeline.download``:

    python -m src.pipeline parse                      # every filing in the manifest
    python -m src.pipeline parse --tickers AAPL MSFT  # just these companies
    python -m src.pipeline parse --force              # re-parse filings already done

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
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from edgar.documents import HTMLParser, ParserConfig

from ..config import DIAGNOSTICS_DIR, INTERIM_DIR, PROJECT_ROOT, ensure_data_dirs
from .constants import (
    EMPTY_ITEM_CHAR_LIMIT,
    EXTRA_ITEM_TITLES,
    FORM_STRUCTURES,
    INCORPORATION_FALLBACKS,
    INCORPORATION_PHRASES,
    KEY_ITEMS,
    STUB_CHAR_LIMIT,
    TABLE_FRAGMENT_CHARS,
    TABLE_MIN_CELLS,
)
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


@dataclass(frozen=True)
class TableFailure:
    """One table the rebuilder could not turn into a labelled grid.

    Kept only to explain a run, never written into the interim files, because
    the HTML fragment it carries is far larger than the record it failed to
    produce. ``--table-debug`` writes these out for inspection.
    """

    section_id: str
    table_index: int      # position among every table in the Item, not among the rebuilt ones
    # A short, stable name for what went wrong, so a run can be summarised by
    # cause. ``reason`` carries the detail, including the numbers involved,
    # which is what makes it useless as a grouping key.
    kind: str
    reason: str
    n_rows: int
    n_cols: int
    html: str
    ticker: str = ""      # filled in by parse_filing, which knows the filing
    accession_no: str = ""


def _fragment(table) -> str:
    """The table's own HTML, trimmed to the part that shows the problem.

    ``html`` is a method on the extractor's table node rather than a property,
    so it has to be called; reading it as an attribute yields the repr of a
    bound method, which looks like markup at a glance and is useless.
    """
    source = getattr(table, "html", None)
    html = str(source() if callable(source) else source or "")
    if len(html) <= TABLE_FRAGMENT_CHARS:
        return html
    return html[:TABLE_FRAGMENT_CHARS] + f"\n<!-- truncated at {TABLE_FRAGMENT_CHARS} chars -->"


_THOUSANDS = re.compile(r"^(\d{1,3}),(\d{3})$")


def _uncomma_year_columns(grid: list[list[str]]) -> None:
    """Undo the thousands separator in a column that is a run of years.

    ``to_dataframe`` hands a year back as the number 2026, indistinguishable from
    2,026 of anything, and _format_cell then punctuates it as money. Debt and
    lease maturity schedules therefore list "2,026" where the filing says "2026".
    That is wrong on its face, and it also stops a search for the year from
    matching the row it belongs to, which is the part that costs retrieval.

    A column is rewritten only when at least three of its values form a run of
    consecutive years. Amounts do not climb by exactly one across three rows,
    so a column of figures cannot be mistaken for one of years.
    """
    width = max((len(row) for row in grid), default=0)
    for column in range(width):
        positions, years = [], []
        for index, row in enumerate(grid):
            if column >= len(row):
                continue
            match = _THOUSANDS.match(row[column].strip())
            if match:
                positions.append(index)
                years.append(int(match.group(1) + match.group(2)))

        if len(years) < 3 or not all(1990 <= year <= 2100 for year in years):
            continue
        if any(later - earlier != 1 for earlier, later in zip(years, years[1:])):
            continue
        for index, year in zip(positions, years):
            grid[index][column] = str(year)


def _clean_cell(value: object) -> str:
    """Flatten one cell to a single line of text.

    Filers wrap a long column label across lines, and the newline survives into
    the cell. A grid cell holding a newline breaks the row it belongs to when the
    table is rendered, splitting one row across two lines and leaving the cells
    after it under the wrong headers. A pipe does the same, so it is escaped
    rather than left to close the cell early.
    """
    text = "" if value is None else str(value)
    return " ".join(text.split()).replace("|", r"\|")


def _extract_tables(section) -> tuple[list[TableRecord], int, list[TableFailure]]:
    """Lift each table out of an Item as a table, not as flattened prose.

    The extractor's plain text runs a balance sheet together into a column of
    bare numbers with the row and column labels stripped away, so "245,122"
    arrives with nothing to say it is total revenue for 2024. Rebuilding each
    table from its own grid keeps those labels attached.

    A table that cannot be rebuilt is skipped rather than guessed at: the
    flattened version is still in the Item's text, so nothing is lost outright.

    Returns the rebuilt tables, how many of the Item's tables held a grid of
    data at all, and a record of each one that could not be rebuilt. Filers wrap
    bullet points in a one-cell table to indent them, and Item 1A is mostly
    built that way, so counting those as tables that failed to rebuild would be
    counting formatting as a fault.

    Each failure carries the table's own HTML, which is what makes the cause
    diagnosable: a merged cell, a spacer row, or a header split over two rows
    all look identical from the outside, and only the markup tells them apart.
    """
    records: list[TableRecord] = []
    failures: list[TableFailure] = []
    data_tables = 0
    for index, table in enumerate(section.tables()):
        try:
            frame = table.to_dataframe()
        except Exception as error:
            # This one raised before its shape was known, so it cannot be told
            # apart from a layout wrapper here. It is recorded, but the summary
            # counts it separately from the tables known to hold data.
            logger.debug("table %d in %s could not be rebuilt", index, section.name)
            failures.append(TableFailure(
                section_id=section.name, table_index=index,
                kind="to_dataframe raised",
                reason=f"to_dataframe raised {type(error).__name__}: {error}",
                n_rows=int(getattr(table, "row_count", 0) or 0),
                n_cols=int(getattr(table, "col_count", 0) or 0),
                html=_fragment(table),
            ))
            continue
        if frame is None or frame.empty or frame.size < TABLE_MIN_CELLS:
            # A wrapper around a sentence, not a table. Its text is in the Item
            # already, so it is not a rebuild that failed.
            continue
        data_tables += 1

        # pandas renamed applymap to map in 2.1; support both.
        formatter = getattr(frame, "map", None) or frame.applymap
        try:
            rendered = formatter(_format_cell)
        except Exception as error:
            logger.debug("table %d in %s could not be rendered", index, section.name)
            failures.append(TableFailure(
                section_id=section.name, table_index=index,
                kind="cell formatting raised",
                reason=f"cell formatting raised {type(error).__name__}: {error}",
                n_rows=int(frame.shape[0]), n_cols=int(frame.shape[1]),
                html=_fragment(table),
            ))
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
            # It held data on arrival but not after the empty spacer columns
            # were dropped, so the grid was mostly layout. Worth seeing, since
            # it is the case most likely to be a fault in the cleanup rather
            # than in the filing.
            failures.append(TableFailure(
                section_id=section.name, table_index=index,
                kind="too few cells after dropping empty columns",
                reason=f"only {rendered.size} cells left after dropping empty columns",
                n_rows=int(rendered.shape[0]), n_cols=int(rendered.shape[1]),
                html=_fragment(table),
            ))
            continue

        # The grid, not a rendered table: the row labels sit in the frame's index,
        # so they are moved into a first column and the header gains a cell above
        # them. Every row is then the same width as the header and can be indexed
        # by column, which is what lets the chunker split a table too wide to fit.
        try:
            header = [_clean_cell(rendered.index.name)]
            header += [_clean_cell(column) for column in rendered.columns]
            grid = [
                [_clean_cell(label)] + [_clean_cell(value) for value in row]
                for label, row in zip(rendered.index, rendered.to_numpy().tolist())
            ]
            _uncomma_year_columns(grid)
        except Exception as error:
            logger.debug("table %d in %s could not be rendered", index, section.name)
            failures.append(TableFailure(
                section_id=section.name, table_index=index,
                kind="grid build raised",
                reason=f"building the grid raised {type(error).__name__}: {error}",
                n_rows=int(rendered.shape[0]), n_cols=int(rendered.shape[1]),
                html=_fragment(table),
            ))
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
                headers=header,
                rows=grid,
                n_rows=len(grid),
                n_cols=len(header),
            )
        )
    return records, data_tables, failures


def interim_path_for(record: FilingRecord, interim_dir: Path = INTERIM_DIR) -> Path:
    """Where a filing's parsed output belongs.

    The name mirrors the raw file so that the two are obviously a pair when the
    folders are opened side by side.
    """
    return interim_dir / record.ticker / (Path(record.path).stem + ".json")


def parse_filing(
    record: FilingRecord,
    project_root: Path = PROJECT_ROOT,
    failures: list[TableFailure] | None = None,
) -> ParsedFiling:
    """Split one downloaded filing into its Items.

    Pass ``failures`` to collect the tables that could not be rebuilt. They are
    kept out of the returned record because their HTML is bulky and belongs to
    diagnosing a run, not to the corpus.
    """
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
        tables, n_data_tables, section_failures = _extract_tables(section)
        if failures is not None:
            failures.extend(
                replace(failure, ticker=record.ticker, accession_no=record.accession_no)
                for failure in section_failures
            )
        sections.append(
            SectionRecord(
                section_id=section_id,
                part=getattr(section, "part", None),
                item=item,
                title=item_title(record.form, getattr(section, "part", None), item),
                text=text,
                n_chars=len(text),
                n_tables=len(section.tables()),
                n_data_tables=n_data_tables,
                is_key_section=item in key_items,
                is_stub=_is_stub_text(text),
                resolved_from=None,  # filled in by _resolve_stubs below
                confidence=getattr(section, "confidence", None),
                detection_method=getattr(section, "detection_method", None),
                validated=bool(getattr(section, "validated", False)),
                warnings=list(getattr(section, "warnings", []) or []),
                tables=tables,
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


def write_table_failures(
    failures: list[TableFailure],
    diagnostics_dir: Path = DIAGNOSTICS_DIR,
) -> Path | None:
    """Write each failed table's HTML out, one file per table.

    The files sit next to an index listing the reason for each, so a run can be
    read either by scanning the reasons or by opening the markup of one table.
    """
    if not failures:
        return None

    destination = diagnostics_dir / "table_failures"
    destination.mkdir(parents=True, exist_ok=True)

    index_lines = []
    for failure in failures:
        name = (f"{failure.ticker}_{failure.accession_no}"
                f"_{failure.section_id}_{failure.table_index}.html")
        (destination / name).write_text(
            f"<!-- {failure.ticker} {failure.accession_no} {failure.section_id} "
            f"table {failure.table_index}\n     {failure.reason}\n     "
            f"{failure.n_rows} rows x {failure.n_cols} cols -->\n{failure.html}",
            encoding="utf-8",
        )
        index_lines.append(json.dumps({
            "file": name,
            **{key: value for key, value in asdict(failure).items() if key != "html"},
        }))

    (destination / "index.jsonl").write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    return destination


def parse_all(
    records: list[FilingRecord],
    force: bool = False,
    interim_dir: Path = INTERIM_DIR,
    failures: list[TableFailure] | None = None,
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
            parsed = parse_filing(record, failures=failures)
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


def report(parsed_filings: list[ParsedFiling],
            failures: list[TableFailure] | None = None,
            written_to: Path | None = None) -> None:
    """Print what was parsed and, more usefully, what looks wrong with it."""
    if not parsed_filings:
        print("\nNothing new was parsed. Use --force to re-parse filings already done.")
        return

    print(f"\nParsed {len(parsed_filings)} filings into {INTERIM_DIR}")

    # Tables are rebuilt so their figures keep their row and column labels. One
    # the rebuilder cannot handle is not lost, since the flattened copy stays in
    # the Item's text, but it is worth knowing how many there were.
    #
    # The rate is measured against the tables that hold data, not against every
    # <table> element. Filers wrap bullet points in a one-cell table to indent
    # them, and Item 1A is written almost entirely that way, so measuring
    # against the raw count would report page formatting as a failure and put
    # the figure roughly twenty points below the truth.
    sections = [section for filing in parsed_filings for section in filing.sections]
    found = sum(section.n_tables for section in sections)
    data_tables = sum(section.n_data_tables for section in sections)
    rebuilt = sum(len(section.tables) for section in sections)
    if data_tables:
        print(f"  {rebuilt} of {data_tables} data tables rebuilt with their labels intact "
              f"({rebuilt / data_tables:.0%})")
        print(f"  {found - data_tables} more were layout wrappers holding prose, not data, "
              "and were left in the Item's text")

    # Say why the rebuilds that failed did so. Without this the rate above is a
    # bare number: it says how many tables were lost but nothing about whether
    # the cause is one fixable pattern or twenty different ones.
    if failures:
        kinds: dict[str, int] = {}
        for failure in failures:
            kinds[failure.kind] = kinds.get(failure.kind, 0) + 1
        plural = "table" if len(failures) == 1 else "tables"
        print(f"  {len(failures)} {plural} could not be rebuilt:")
        for kind, count in sorted(kinds.items(), key=lambda pair: -pair[1]):
            print(f"    {count:>4}  {kind}")
        if written_to:
            print(f"  Their HTML is in {written_to}")
        else:
            print("  Re-run with --table-debug to write their HTML out for diagnosis")

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
