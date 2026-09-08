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

import argparse
import json
import logging
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from edgar.company_reports import TenK, TenQ
from edgar.documents import HTMLParser, ParserConfig

from .config import INTERIM_DIR, PROJECT_ROOT, ensure_data_dirs
from .download import FilingRecord, load_manifest

logger = logging.getLogger(__name__)

# The Items the project brief calls out, per form. Everything else in the
# filing is still parsed and stored; this only marks which sections the
# retrieval work is expected to lean on, so the summary can report on them.
KEY_ITEMS: dict[str, set[str]] = {
    "10-K": {"1", "1A", "7", "7A", "8"},
    "10-Q": {"1", "2", "3"},
}

# Where an Item commonly carries a cross-reference instead of the disclosure
# itself, and which Item actually holds the text. Oracle and NVIDIA both answer
# Item 8 by pointing at the financial statements filed under Item 15.
INCORPORATION_FALLBACKS: dict[str, dict[str, str]] = {
    "10-K": {"8": "15"},
}

# Item headings that carry a cross-reference instead of the disclosure itself
# run to a couple of sentences, while a real Item runs to thousands of
# characters. The gap between the two is wide enough that a flat threshold
# separates them reliably.
STUB_CHAR_LIMIT = 500

# The structures edgartools ships already name every Item, so the human-readable
# titles used in citations come from there rather than a second list of our own
# that could drift out of step with it.
_STRUCTURES = {"10-K": TenK.structure, "10-Q": TenQ.structure}


@dataclass(frozen=True)
class SectionRecord:
    """One Item of one filing."""

    section_id: str       # e.g. "part_ii_item_7", as the parser names it
    part: str | None      # "I", "II", ...
    item: str | None      # "1", "1A", "7A", ...
    title: str            # e.g. "Risk Factors"
    text: str
    n_chars: int
    n_tables: int
    is_key_section: bool
    is_stub: bool
    # For a stub whose text lives under a different Item, the section_id that
    # actually holds it. Chunking reads the text from there.
    resolved_from: str | None
    # What the parser thought of its own work, kept so that a bad answer can be
    # traced back to a badly detected section boundary.
    confidence: float | None
    detection_method: str | None
    validated: bool
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ParsedFiling:
    """One filing, split into Items, as written to data/interim/."""

    ticker: str
    cik: int
    company: str
    form: str
    filing_date: str
    accession_no: str
    url: str
    source_path: str      # the raw file this was parsed from
    sections: list[SectionRecord]


def item_title(form: str, part: str | None, item: str | None) -> str:
    """Look up the official title of an Item, for use in citations."""
    structure = _STRUCTURES.get(form)
    if structure is None or not item:
        return ""

    entry = structure.get_item(f"ITEM {item}", f"PART {part}" if part else None)
    return entry.get("Title", "") if entry else ""


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
                is_stub=len(text) < STUB_CHAR_LIMIT,
                resolved_from=None,  # filled in by _resolve_stubs below
                confidence=getattr(section, "confidence", None),
                detection_method=getattr(section, "detection_method", None),
                validated=bool(getattr(section, "validated", False)),
                warnings=list(getattr(section, "warnings", []) or []),
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


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--tickers", nargs="+",
        help="Only parse these companies. Defaults to every filing in the manifest.",
    )
    parser.add_argument(
        "--forms", nargs="+",
        help="Only parse these forms, for example --forms 10-K.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-parse filings that already have output in data/interim/.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    args = _parse_args(argv)

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

    logger.info("Parsing %d filings from the manifest", len(records))
    _report(parse_all(records, force=args.force))


if __name__ == "__main__":
    main()
