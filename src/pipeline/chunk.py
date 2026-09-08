"""Cut parsed Items into retrievable passages, into ``data/processed/``.

Run it from the project root, after ``src.pipeline.parse``:

    python -m src.pipeline.chunk                      # every filing in data/interim/
    python -m src.pipeline.chunk --tickers AAPL MSFT  # just these companies
    python -m src.pipeline.chunk --force              # re-chunk filings already done

Retrieval works on passages rather than whole Items. Item 1A alone runs to
77,000 characters in the median filing, which is far past what an embedding
model reads at once, and far past what is useful to put in front of an LLM as
evidence. This stage cuts each Item into passages of roughly 4,000 characters
and writes one JSON file per filing to ``data/processed/<TICKER>/``.

Three things shape where the cuts fall.

A passage never spans two Items, because a citation has to name the Item it
came from, which is the whole reason the parse stage exists.

Paragraphs are packed whole rather than sliced at a character count. Measured
over this corpus, only 6 of 127,885 paragraphs are longer than the budget, so
packing whole paragraphs keeps very nearly every passage on sentence boundaries
at no real cost in size.

A passage taken from the middle of Item 1A would otherwise lose the heading it
sits under, so the nearest heading above it is carried onto it.

Where an Item answers with a cross-reference rather than the disclosure itself,
the text is not duplicated. The stub yields no passages of its own, and instead
the Item that holds the text records the stub in ``incorporated_into``, so a
question about Item 8 reaches Oracle's Item 15 passages and the citation can say
where the text really sits.

Tables are passages in their own right. The parse stage rebuilds each financial
table as a grid, and this stage indexes those grids directly rather than the
flattened wreckage the extractor leaves in the prose, so a figure always arrives
under its own row and column labels. A table too long for one passage is split
by rows with its header repeated on every slice. Those passages are marked
``content_type="table"``, so retrieval can weight them when a question is
numeric, and the flattened copies are dropped from the prose so the same figures
are not indexed twice.
"""

from __future__ import annotations

import json
import logging
import re
import statistics
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path

from ..config import INTERIM_DIR, PROCESSED_DIR, PROJECT_ROOT, ensure_data_dirs
from .cli import build_chunk_parser
from .constants import (
    CHUNK_CHAR_BUDGET,
    CHUNK_CHAR_MINIMUM,
    CHUNK_CHAR_OVERLAP,
    FURNITURE_PATTERNS,
    HEADING_BODY_MINIMUM,
    HEADING_CHAR_LIMIT,
    PAGE_NUMBER_PATTERN,
    REJOIN_PAGE_BREAK_SPLITS,
    SKIP_UNNUMBERED_SECTIONS,
    TABLE_MAX_ROWS_PER_CHUNK,
)
from .records import ChunkedFiling, ChunkRecord, ParsedFiling, SectionRecord, TableRecord

logger = logging.getLogger(__name__)

# Each pattern carries its own anchors, so they combine into a single pass over
# a block rather than a loop over the list.
_FURNITURE = re.compile("|".join(FURNITURE_PATTERNS))
_PAGE_NUMBER = re.compile(PAGE_NUMBER_PATTERN)
# A figure worth tracking: four digits or more, so years and money count
# but a row index or a footnote marker does not.
_FIGURE = re.compile(r"\d[\d,]{3,}")


def interim_files(interim_dir: Path = INTERIM_DIR) -> list[Path]:
    """Every parsed filing on disk, in a stable order."""
    return sorted(interim_dir.glob("*/*.json"))


def load_parsed(path: Path) -> ParsedFiling:
    """Read one interim file back into the record the parse stage wrote.

    Rebuilding the dataclass rather than working with the raw dictionary means
    a file written by an older version of the parser fails here, loudly, rather
    than quietly producing passages with fields missing.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    sections = []
    for section in data.pop("sections"):
        # Tables are nested records, so they have to be rebuilt before the
        # section that holds them.
        section["tables"] = [TableRecord(**table) for table in section.get("tables", [])]
        sections.append(SectionRecord(**section))
    return ParsedFiling(**data, sections=sections)


def load_chunked(path: Path) -> ChunkedFiling:
    """Read one processed file back into the record the chunk stage wrote."""
    data = json.loads(path.read_text(encoding="utf-8"))
    chunks = [ChunkRecord(**chunk) for chunk in data.pop("chunks")]
    return ChunkedFiling(**data, chunks=chunks)


def iter_chunks(
    processed_dir: Path = PROCESSED_DIR,
    fiscal_years: range | list[int] | None = None,
    tickers: list[str] | None = None,
    key_items_only: bool = False,
) -> Iterator[dict]:
    """Yield every stored passage with its filing's identity already attached.

    This is the seam the retrieval stage builds on. A passage on its own knows
    which Item it came from but not which company or year, because that is
    stored once per filing rather than repeated on all two hundred of its
    passages. Indexing needs both on the same record, so they are joined here
    rather than in each place that reads the corpus.

    ``fiscal_years`` filters on the year a filing reports on, not the year it
    was filed. Passing it is what keeps "compare these companies in FY2024"
    from quietly answering across a mix of years.
    """
    wanted_fiscal = set(fiscal_years) if fiscal_years else None
    wanted_tickers = {ticker.upper() for ticker in tickers} if tickers else None

    for path in sorted(processed_dir.glob("*/*.json")):
        filing = load_chunked(path)
        if wanted_tickers and filing.ticker not in wanted_tickers:
            continue
        fiscal_year = (
            int(filing.period_of_report[:4])
            if filing.period_of_report[:4].isdigit() else None
        )
        if wanted_fiscal is not None and fiscal_year not in wanted_fiscal:
            continue

        for chunk in filing.chunks:
            if key_items_only and not chunk.is_key_section:
                continue
            yield {
                **asdict(chunk),
                "ticker": filing.ticker,
                "company": filing.company,
                "cik": filing.cik,
                "form": filing.form,
                "filing_date": filing.filing_date,
                "period_of_report": filing.period_of_report,
                "fiscal_year": fiscal_year,
                "accession_no": filing.accession_no,
                "url": filing.url,
            }


def processed_path_for(interim_file: Path, processed_dir: Path = PROCESSED_DIR) -> Path:
    """Where an interim file's passages belong.

    All three stages use the same ``<TICKER>/<filing>`` name, so raw, interim,
    and processed line up when the folders are opened side by side.
    """
    return processed_dir / interim_file.parent.name / interim_file.name


def _strip_furniture(blocks: list[str], has_tables: bool) -> list[str]:
    """Drop the navigation the extractor left inline in the text."""
    kept: list[str] = []
    for block in blocks:
        if _FURNITURE.search(block):
            continue
        if not has_tables and _PAGE_NUMBER.match(block):
            continue
        kept.append(block)
    return kept


def _rejoin_page_breaks(blocks: list[str]) -> list[str]:
    """Undo paragraph breaks the extractor inserted at a page boundary.

    A 10-K paragraph that runs across a printed page comes back as two blocks,
    because the page break looks like a paragraph break. Left alone, a passage
    can begin halfway through a sentence. Where one block stops without closing
    punctuation and the next opens in lower case, the two were one sentence.
    """
    if not REJOIN_PAGE_BREAK_SPLITS:
        return blocks

    joined: list[str] = []
    for block in blocks:
        if (
            joined
            and joined[-1]
            and joined[-1][-1] not in ".:;!?)”\"'"
            and block[:1].islower()
        ):
            joined[-1] = f"{joined[-1]} {block}"
        else:
            joined.append(block)
    return joined


def table_figures(tables: list[TableRecord]) -> set[str]:
    """Every multi-digit figure the rebuilt tables of an Item actually carry."""
    return {match.group() for table in tables for match in _FIGURE.finditer(table.markdown)}


def _is_table_debris(block: str, figures_in_tables: set[str]) -> bool:
    """Whether a block is the flattened wreckage of a table rather than prose.

    In a section that holds tables, the extractor's plain text runs each table
    together into strings like "Americas$167,045 3 %$162,560". Where that table
    was rebuilt properly, indexing the flattened copy as well would put the same
    figures in the corpus twice, once unreadable.

    The catch is that not every table can be rebuilt. Apple's statement of
    shareholders' equity comes back with its row labels but without its figures,
    so dropping the flattened copy there would delete numbers that exist nowhere
    else. A block is therefore only dropped once its own figures are confirmed
    present in a rebuilt table, which makes the loss impossible by construction.

    A real sentence is spared because it closes with punctuation, and a heading
    such as "Americas" is spared because it carries no figures at all.
    """
    if block[-1:] in ".:;!?":
        return False
    dense = sum(character.isdigit() or character in "$%()," for character in block)
    visible = sum(1 for character in block if not character.isspace())
    if visible == 0 or dense / visible < 0.3:
        return False

    figures = {match.group() for match in _FIGURE.finditer(block)}
    return bool(figures) and figures <= figures_in_tables


def _is_heading(block: str, following: str) -> bool:
    """Whether a block introduces the text beneath it rather than being text.

    A heading is short, is not written as a sentence, and has a body underneath
    it. That last test is what keeps table row labels such as "Total" out, since
    those are followed by a figure rather than by prose.
    """
    if not block or len(block) > HEADING_CHAR_LIMIT:
        return False
    if block[-1] in ".:;,":
        return False
    if not block[0].isupper():
        return False
    # Mostly digits or punctuation means a table cell, not a heading.
    if sum(character.isalpha() for character in block) < len(block) / 2:
        return False
    return len(following) >= HEADING_BODY_MINIMUM


def _headings_in_force(blocks: list[str]) -> list[str | None]:
    """The heading each block sits under, or None before the first one."""
    in_force: list[str | None] = []
    current: str | None = None
    for position, block in enumerate(blocks):
        following = blocks[position + 1] if position + 1 < len(blocks) else ""
        if _is_heading(block, following):
            current = block
        in_force.append(current)
    return in_force


def _overlap_tail(passage: list[int], blocks: list[str], overlap: int) -> list[int]:
    """The last whole paragraphs of a passage, within the overlap budget."""
    tail: list[int] = []
    size = 0
    for position in reversed(passage):
        if size + len(blocks[position]) > overlap:
            break
        tail.insert(0, position)
        size += len(blocks[position])
    return tail


def _pack_blocks(blocks: list[str], budget: int, overlap: int) -> list[list[int]]:
    """Group paragraphs into passages of about ``budget`` characters.

    Returns the positions of the paragraphs in each passage rather than their
    text, so the caller can look up whatever else it tracks per paragraph, such
    as the heading in force at that point.

    A paragraph is never cut, so a passage holding a single oversized paragraph
    runs over budget instead of splitting it, and joins whatever short paragraph
    precedes it rather than stranding it as a passage of its own. Every passage
    after the first opens with the closing paragraphs of the one before it.
    """
    passages: list[list[int]] = []
    current: list[int] = []
    size = 0

    for position, block in enumerate(blocks):
        if size >= CHUNK_CHAR_MINIMUM and size + len(block) > budget:
            passages.append(current)
            current = _overlap_tail(current, blocks, overlap)
            size = sum(len(blocks[carried]) for carried in current)
        current.append(position)
        size += len(block)

    if current:
        passages.append(current)
    return passages


def chunk_section(
    section: SectionRecord,
    accession_no: str,
    incorporated_into: list[str] | None = None,
    budget: int = CHUNK_CHAR_BUDGET,
    overlap: int = CHUNK_CHAR_OVERLAP,
) -> list[ChunkRecord]:
    """Cut one Item into passages."""
    blocks = [block.strip() for block in section.text.split("\n\n") if block.strip()]
    blocks = _strip_furniture(blocks, has_tables=section.n_tables > 0)
    blocks = _rejoin_page_breaks(blocks)
    # Where the Item's tables were rebuilt properly, the flattened copies
    # still sitting in the prose are pure noise, so they are dropped rather
    # than indexed alongside the readable version.
    if section.tables:
        figures = table_figures(section.tables)
        blocks = [block for block in blocks if not _is_table_debris(block, figures)]
    if not blocks:
        return []

    headings = _headings_in_force(blocks)

    passages: list[ChunkRecord] = []
    for index, positions in enumerate(_pack_blocks(blocks, budget, overlap)):
        text = "\n\n".join(blocks[position] for position in positions)
        passages.append(
            ChunkRecord(
                chunk_id=f"{accession_no}_{section.section_id}_{index:03d}",
                section_id=section.section_id,
                part=section.part,
                item=section.item,
                title=section.title,
                heading=headings[positions[0]],
                text=text,
                n_chars=len(text),
                chunk_index=index,
                is_key_section=section.is_key_section,
                incorporated_into=list(incorporated_into or []),
            )
        )
    return passages


def chunk_filing(
    parsed: ParsedFiling,
    source_path: str,
    key_items_only: bool = False,
    budget: int = CHUNK_CHAR_BUDGET,
    overlap: int = CHUNK_CHAR_OVERLAP,
) -> ChunkedFiling:
    """Cut one parsed filing into passages."""
    # The parse stage records, on the stub, which Item holds its text. Inverting
    # that gives the Item that holds the text the list of Items it also answers,
    # which is what lets the passage be stored once and still be cited as both.
    incorporated: dict[str, list[str]] = {}
    for section in parsed.sections:
        if section.resolved_from and section.item:
            incorporated.setdefault(section.resolved_from, []).append(section.item)

    passages: list[ChunkRecord] = []
    for section in parsed.sections:
        # A stub is either a cross-reference, whose text is chunked under the
        # Item it points at, or boilerplate deferring to the proxy statement.
        # Neither is worth indexing on its own.
        if section.is_stub or not section.text.strip():
            continue
        # A section the filing does not number cannot be cited as an Item, and a
        # citation is the point of the whole system. In this corpus that is only
        # the signature block, which answers nothing anyway.
        if SKIP_UNNUMBERED_SECTIONS and not section.item:
            continue
        if key_items_only and not section.is_key_section:
            continue

        also_answers = sorted(incorporated.get(section.section_id, []))
        passages.extend(
            chunk_section(
                section,
                accession_no=parsed.accession_no,
                incorporated_into=also_answers,
                budget=budget,
                overlap=overlap,
            )
        )
        passages.extend(
            chunk_tables(
                section,
                accession_no=parsed.accession_no,
                incorporated_into=also_answers,
                budget=budget,
            )
        )

    return ChunkedFiling(
        ticker=parsed.ticker,
        cik=parsed.cik,
        company=parsed.company,
        form=parsed.form,
        filing_date=parsed.filing_date,
        accession_no=parsed.accession_no,
        url=parsed.url,
        source_path=source_path,
        chunks=passages,
        period_of_report=parsed.period_of_report,
    )


def _slice_table_rows(rows: list[str], header_chars: int, budget: int) -> list[list[str]]:
    """Group table rows into slices that fit the passage budget.

    Rows are capped by count and by width together. A count alone is not enough,
    because a table can be wide as well as long: thirty rows of a twelve-column
    schedule run well past the budget on their own.
    """
    slices: list[list[str]] = []
    current: list[str] = []
    size = header_chars
    for row in rows:
        too_many = len(current) >= TABLE_MAX_ROWS_PER_CHUNK
        too_wide = current and size + len(row) > budget
        if too_many or too_wide:
            slices.append(current)
            current, size = [], header_chars
        current.append(row)
        size += len(row)
    if current:
        slices.append(current)
    return slices or [[]]


def chunk_tables(
    section: SectionRecord,
    accession_no: str,
    incorporated_into: list[str] | None = None,
    budget: int = CHUNK_CHAR_BUDGET,
) -> list[ChunkRecord]:
    """Turn each rebuilt table into passages that keep their column labels.

    A table is kept whole where it fits. A long or wide one is split by rows,
    and the header is repeated on every slice, so no passage of figures ever
    arrives without the labels that say what the figures are.
    """
    passages: list[ChunkRecord] = []
    for table in section.tables:
        lines = table.markdown.splitlines()
        if len(lines) < 3:
            continue
        # The first two lines are the column labels and the rule beneath them.
        header, rows = lines[:2], lines[2:]
        slices = _slice_table_rows(rows, sum(len(line) for line in header), budget)

        for part_number, row_slice in enumerate(slices):
            label = table.caption or section.title or f"Item {section.item}"
            if len(slices) > 1:
                label = f"{label} (part {part_number + 1} of {len(slices)})"
            text = "\n".join([label, "", *header, *row_slice])
            passages.append(
                ChunkRecord(
                    chunk_id=(
                        f"{accession_no}_{section.section_id}"
                        f"_t{table.table_index:03d}_{part_number:02d}"
                    ),
                    section_id=section.section_id,
                    part=section.part,
                    item=section.item,
                    title=section.title,
                    heading=table.caption or None,
                    text=text,
                    n_chars=len(text),
                    chunk_index=part_number,
                    is_key_section=section.is_key_section,
                    incorporated_into=list(incorporated_into or []),
                    content_type="table",
                    table_index=table.table_index,
                    table_caption=table.caption,
                )
            )
    return passages


def write_chunks(chunked: ChunkedFiling, destination: Path) -> None:
    """Write one filing's passages as JSON."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(asdict(chunked), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def chunk_all(
    paths: list[Path],
    force: bool = False,
    forms: list[str] | None = None,
    key_items_only: bool = False,
    budget: int = CHUNK_CHAR_BUDGET,
    overlap: int = CHUNK_CHAR_OVERLAP,
    processed_dir: Path = PROCESSED_DIR,
) -> list[ChunkedFiling]:
    """Chunk every interim file given, carrying on if one of them fails."""
    ensure_data_dirs()

    chunked_filings: list[ChunkedFiling] = []
    for path in paths:
        destination = processed_path_for(path, processed_dir)
        if destination.exists() and not force:
            logger.info("%s: already chunked, skipping", path.name)
            continue

        try:
            parsed = load_parsed(path)
        except Exception:
            # One unreadable interim file should not cost us the rest of the
            # corpus; the summary at the end reports how many landed.
            logger.exception("%s: could not be read, moving on", path.name)
            continue

        if forms and parsed.form not in forms:
            continue

        chunked = chunk_filing(
            parsed,
            source_path=path.relative_to(PROJECT_ROOT).as_posix(),
            key_items_only=key_items_only,
            budget=budget,
            overlap=overlap,
        )
        write_chunks(chunked, destination)
        chunked_filings.append(chunked)

        logger.info(
            "%s %s: %d passages from %d Items, %d chars -> %s",
            parsed.ticker, parsed.filing_date, len(chunked.chunks),
            len({passage.section_id for passage in chunked.chunks}),
            sum(passage.n_chars for passage in chunked.chunks), destination.name,
        )

    return chunked_filings


def _report(chunked_filings: list[ChunkedFiling]) -> None:
    """Print what was cut, and what the passages look like."""
    if not chunked_filings:
        print("\nNothing new was chunked. Use --force to re-chunk filings already done.")
        return

    passages = [passage for filing in chunked_filings for passage in filing.chunks]
    if not passages:
        print("\nNo section in those filings held text to chunk.")
        return

    sizes = sorted(passage.n_chars for passage in passages)
    from_key = sum(1 for passage in passages if passage.is_key_section)
    with_heading = sum(1 for passage in passages if passage.heading)

    print(f"\nChunked {len(chunked_filings)} filings into {PROCESSED_DIR}")
    print(f"  {len(passages):,} passages, {sum(sizes):,} characters")
    print(f"  {from_key:,} from key Items, {len(passages) - from_key:,} from the rest")
    print(f"  {with_heading:,} carry a heading ({with_heading / len(passages):.0%})")
    print(
        f"  size: median {statistics.median(sizes):,.0f}, "
        f"largest {sizes[-1]:,}, smallest {sizes[0]:,}"
    )

    # Where an Item answers with a cross-reference, its passages sit under the
    # Item that holds the text, so the redirect is worth naming rather than
    # leaving someone to wonder why Item 8 produced nothing.
    redirects = sorted({
        (filing.ticker, filing.filing_date, passage.item, ", ".join(passage.incorporated_into))
        for filing in chunked_filings
        for passage in filing.chunks
        if passage.incorporated_into
    })
    if redirects:
        print(f"\n{len(redirects)} Items hold text that another Item cross-refers to:")
        for ticker, filing_date, item, into in redirects:
            print(f"  {ticker} {filing_date}  Item {item} also answers Item {into}")

    # A passage far under budget is usually just a short Item, but a run of them
    # would mean the packer is flushing early and the budget needs a look.
    runts = [passage for passage in passages if passage.n_chars < 200]
    if runts:
        print(f"\n{len(runts)} passages are under 200 characters:")
        for passage in runts[:10]:
            print(f"  {passage.chunk_id}  {passage.n_chars} chars")


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    parser = build_chunk_parser(__doc__.splitlines()[0] if __doc__ else "")
    args = parser.parse_args(argv)

    paths = interim_files()
    if not paths:
        print("data/interim/ is empty. Run python -m src.pipeline.parse first.")
        return

    if args.tickers:
        wanted = {ticker.upper() for ticker in args.tickers}
        paths = [path for path in paths if path.parent.name in wanted]
        if not paths:
            # Without this the run would fall through to "use --force", which
            # points at the wrong problem: nothing was skipped, nothing is there.
            print(
                f"Nothing parsed yet for {', '.join(sorted(wanted))}. "
                "Run python -m src.pipeline.parse for those tickers first."
            )
            return

    logger.info("Chunking %d filings from %s", len(paths), INTERIM_DIR)
    _report(chunk_all(
        paths,
        force=args.force,
        forms=args.forms,
        key_items_only=args.key_items_only,
        budget=args.budget,
        overlap=args.overlap,
    ))


if __name__ == "__main__":
    main()
