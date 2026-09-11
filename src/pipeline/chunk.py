"""Cut parsed Items into retrievable passages, into ``data/processed/``.

Run it from the project root, after ``src.pipeline.parse``:

    python -m src.pipeline chunk                      # every filing in data/interim/
    python -m src.pipeline chunk --tickers AAPL MSFT  # just these companies
    python -m src.pipeline chunk --force              # re-chunk filings already done

Retrieval works on passages rather than whole Items. Item 1A alone runs to
77,000 characters in the median filing, which is far past what an embedding
model reads at once, and far past what is useful to put in front of an LLM as
evidence. This stage cuts each Item into passages of roughly 1,800 characters,
about 450 tokens, and writes one JSON file per filing to
``data/processed/<TICKER>/``.

Three things shape where the cuts fall.

A passage never spans two Items, because a citation has to name the Item it
came from, which is the whole reason the parse stage exists.

Paragraphs are packed whole rather than sliced at a character count, since a
paragraph is one idea and half an idea retrieves badly. Only 511 of 129,286
paragraphs in this corpus are longer than the budget, and those are split at
sentence ends, so all but a handful of passages begin and end on a sentence
boundary.

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

Every passage is sized to be read whole by the embedding model rather than
truncated by it. A table too long for one passage is split by rows and one too
wide by columns, with the header repeated on every piece, and the paragraph
separators count against the budget as well as the paragraphs. The result is
that 0.12% of passages exceed the 2,048 characters a 512-token model reads, by
at most 42 characters.
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
from .constants import (
    CHUNK_CHAR_BUDGET,
    CHUNK_CHAR_MINIMUM,
    CHUNK_CHAR_OVERLAP,
    DENSE_TEXT_AT,
    DENSE_TEXT_FROM,
    FURNITURE_PATTERNS,
    HEADING_BODY_MINIMUM,
    HEADING_CHAR_LIMIT,
    PAGE_NUMBER_PATTERN,
    PROSE_CHARS_PER_TOKEN,
    REJOIN_PAGE_BREAK_SPLITS,
    SKIP_UNNUMBERED_SECTIONS,
    TABLE_CHARS_PER_TOKEN,
    TABLE_MAX_ROWS_PER_CHUNK,
    table_budget_for,
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
                # Carried for the same reason the filing identity is: an index
                # records the settings its passages were cut with, and this is
                # where it can read them rather than assume the constants.
                "chunk_budget": filing.chunk_budget,
                "chunk_overlap": filing.chunk_overlap,
                "key_items_only": filing.key_items_only,
            }


def resolve_chunk_settings(
    seen: set[tuple[int | None, int | None]],
) -> tuple[int | None, int | None, str | None]:
    """What a corpus was cut with, from what its filings recorded.

    ``seen`` is the distinct ``(chunk_budget, chunk_overlap)`` pairs observed
    while walking the corpus. Returns the pair an index manifest should record,
    and a note to print when that answer is not a measurement, so both indexes
    report an unlabelled or mixed corpus the same way.

    There is deliberately no way for a caller to supply the values. An override
    can only agree with what the corpus recorded, in which case it adds
    nothing, or disagree, in which case it is wrong; and for a corpus that
    predates the recording, re-chunking takes seconds and turns the guess into
    a measurement.

    Three cases, and only the first is silent:

    One recorded pair is the answer.

    Nothing recorded means the files predate this, so the constants in force
    stand in, and the note says they are assumed rather than measured.

    More than one pair means the corpus was cut in more than one way, and a
    manifest has a single budget field. There is no correct value, so it
    records none. This includes the half-and-half case -- some filings
    recorded, some not -- because a file that did not record its settings is
    not evidence that they matched.
    """
    if len(seen) == 1:
        recorded_budget, recorded_overlap = next(iter(seen))
        if recorded_budget is None and recorded_overlap is None:
            return CHUNK_CHAR_BUDGET, CHUNK_CHAR_OVERLAP, (
                f"data/processed/ predates the recording of chunker settings, so "
                f"the manifest assumes the current constants "
                f"({CHUNK_CHAR_BUDGET}/{CHUNK_CHAR_OVERLAP}) rather than measuring "
                f"them. Re-chunk to record them: python -m src.pipeline chunk --force"
            )
        return recorded_budget, recorded_overlap, None

    listed = ", ".join(
        "unrecorded" if pair == (None, None) else f"{pair[0]}/{pair[1]}"
        for pair in sorted(seen, key=lambda pair: (pair[0] or 0, pair[1] or 0))
    )
    return None, None, (
        f"data/processed/ holds filings cut with different chunker settings "
        f"({listed}), so the manifest records none: one budget field cannot "
        f"describe a mixed corpus, and picking one would label every sweep row "
        f"with a number that is wrong for part of it. Re-chunk the whole corpus: "
        f"python -m src.pipeline chunk --force"
    )


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
    return {
        match.group()
        for table in tables
        for row in [table.headers, *table.rows]
        for cell in row
        for match in _FIGURE.finditer(cell)
    }


def render_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a grid as a markdown table.

    Cells are not padded to a common width. Alignment spaces would be the
    largest single item in a wide table, and they carry no meaning: a passage is
    read by an embedding model, which spends context on them and learns nothing.
    """
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(" --- " for _ in headers) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _row_width(headers: list[str], rows: list[list[str]], columns: list[int]) -> int:
    """How wide the widest rendered line would be, for these columns only."""
    return max(
        sum(len(row[column]) for column in columns) + 3 * len(columns) + 1
        for row in (headers, *rows)
    )


def _split_table_columns(
    headers: list[str], rows: list[list[str]], budget: int,
) -> list[tuple[list[str], list[list[str]]]]:
    """Split a table too wide for one row to fit into groups of columns.

    Slicing by row cannot help a table whose single row already runs past the
    budget, and a twenty-column schedule of quarterly figures does exactly that.
    Splitting by column does, and the first column, which holds the row labels,
    is repeated in every group: without it a group of figures has nothing saying
    which line item each one belongs to.

    A group is only closed once it holds a column besides the label, so a table
    whose label column alone exceeds the budget still yields whole rows rather
    than an endless run of one-column passages.
    """
    if len(headers) < 3 or _row_width(headers, rows, list(range(len(headers)))) <= budget:
        return [(headers, rows)]

    groups: list[list[int]] = []
    current: list[int] = []
    for column in range(1, len(headers)):
        if current and _row_width(headers, rows, [0, *current, column]) > budget:
            groups.append([0, *current])
            current = [column]
        else:
            current.append(column)
    if current:
        groups.append([0, *current])

    return [
        ([headers[column] for column in group],
         [[row[column] for column in group] for row in rows])
        for group in groups
    ]


def table_cells(tables: list[TableRecord]) -> dict[str, list[str]]:
    """The distinct cells of an Item's rebuilt tables, keyed by their first three
    characters.

    Keyed that way so a block can be matched against the thousands of cells an
    Item 8 holds by looking only at cells that could begin somewhere in it.
    Cells under three characters are left out: a lone "$" or ")" says nothing
    about whether a block came from the table.
    """
    index: dict[str, list[str]] = {}
    for cell in {cell for table in tables for row in [table.headers, *table.rows] for cell in row}:
        if len(cell) >= 3:
            index.setdefault(cell[:3], []).append(cell)
    return index


def _unexplained(block: str, cells: dict[str, list[str]]) -> str:
    """What is left of a block once every table cell found in it is removed.

    Longest cells first, so a row label is removed whole before a shorter cell
    that happens to sit inside it.
    """
    grams = {block[start:start + 3] for start in range(len(block) - 2)}
    found = sorted(
        {cell for gram in grams & cells.keys() for cell in cells[gram] if cell in block},
        key=len, reverse=True,
    )
    for cell in found:
        block = block.replace(cell, " ")
    return block


def _is_table_debris(
    block: str,
    figures_in_tables: set[str],
    cells: dict[str, list[str]] | None = None,
) -> bool:
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

    That test alone misses most of them, because flattening also runs cells
    into each other: "Balance -- July 31, 2020" followed by "0.4" becomes
    "20200.4", a figure no table holds, and one such join keeps the whole block.
    Those were 199 of the 286 passages the encoder was truncating. So a block
    is also debris when the table's own cells account for it: remove every cell
    found in it, and if no figure is left and at most a fifth of its letters
    and digits, the block was the table. The guarantee is the same one -- a
    figure present nowhere else survives the removal and keeps the block.

    A real sentence is spared because it closes with punctuation, and a heading
    such as "Americas" is spared because it carries no figures at all.
    """
    if block[-1:] in ".:;!?":
        return False
    dense = sum(character.isdigit() or character in "$%()," for character in block)
    visible = sum(1 for character in block if not character.isspace())
    if visible == 0:
        return False

    figures = {match.group() for match in _FIGURE.finditer(block)}
    if dense / visible >= 0.3 and figures and figures <= figures_in_tables:
        return True
    if cells is None or dense / visible < 0.15:
        return False

    # Whitespace normalised the way cells were, since the extractor writes
    # non-breaking spaces where the rebuilt grid has plain ones.
    remainder = _unexplained(" ".join(block.split()), cells)
    if _FIGURE.search(remainder):
        return False
    before = sum(character.isalnum() for character in block)
    after = sum(character.isalnum() for character in remainder)
    return after <= 0.2 * before


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


# Paragraphs are joined by a blank line, so every join after the first costs
# two characters of the budget.
_JOIN_CHARS = len("\n\n")


def _density(text: str) -> float:
    """The share of visible characters that are not letters."""
    visible = [character for character in text if not character.isspace()]
    if not visible:
        return 0.0
    return sum(not character.isalpha() for character in visible) / len(visible)


def _chars_per_token(block: str) -> float:
    """The characters per token to budget a paragraph at, from its density.

    PROSE_CHARS_PER_TOKEN below DENSE_TEXT_FROM, falling linearly to
    TABLE_CHARS_PER_TOKEN at DENSE_TEXT_AT; see constants.py for the
    measurements this follows.
    """
    progress = (_density(block) - DENSE_TEXT_FROM) / (DENSE_TEXT_AT - DENSE_TEXT_FROM)
    progress = min(max(progress, 0.0), 1.0)
    return PROSE_CHARS_PER_TOKEN - progress * (PROSE_CHARS_PER_TOKEN - TABLE_CHARS_PER_TOKEN)


def _cost(block: str) -> int:
    """What a paragraph costs against the prose budget, in prose characters.

    Its length for ordinary prose, and more for text dense with figures, which
    tokenises at up to two and a half times the rate. Charging it more is what
    keeps a passage of such text inside the encoder's window, where the budget
    in plain characters let 286 passages overrun it.
    """
    return round(len(block) * PROSE_CHARS_PER_TOKEN / _chars_per_token(block))


def _packed_size(passage: list[int], blocks: list[str]) -> int:
    """What the rendered passage costs against the budget, separators included."""
    if not passage:
        return 0
    return sum(_cost(blocks[position]) for position in passage) + _JOIN_CHARS * (len(passage) - 1)


def _overlap_tail(passage: list[int], blocks: list[str], overlap: int) -> list[int]:
    """The last whole paragraphs of a passage, within the overlap budget."""
    tail: list[int] = []
    size = 0
    for position in reversed(passage):
        if size + _cost(blocks[position]) > overlap:
            break
        tail.insert(0, position)
        size += _cost(blocks[position])
    return tail


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_CLAUSE_END = re.compile(r"(?<=[;:])\s+")


def _split_long_block(block: str, budget: int) -> list[str]:
    """Split a paragraph that is longer than the whole budget.

    Packing never cuts a paragraph, which is right for ordinary prose: a
    paragraph is one idea, and half an idea retrieves badly. But filings do run
    a single paragraph past the budget, and an embedding model then truncates it
    with no warning, so the tail is indexed as though it were never written.

    Sentences are kept whole, which is the property that matters for reading a
    passage back as a citation. A sentence that is itself over budget, which in
    filings means a long enumeration, is split at its clause boundaries instead.
    """
    if len(block) <= budget:
        return [block]

    parts: list[str] = []
    current = ""
    for sentence in _SENTENCE_END.split(block):
        pieces = [sentence]
        if len(sentence) > budget:
            pieces = _CLAUSE_END.split(sentence)
        for piece in pieces:
            for word_run in _split_on_words(piece, budget):
                if current and len(current) + 1 + len(word_run) > budget:
                    parts.append(current)
                    current = word_run
                else:
                    current = f"{current} {word_run}" if current else word_run
    if current:
        parts.append(current)
    return parts


def _split_on_words(text: str, budget: int) -> list[str]:
    """Last resort for a run of text with no sentence or clause boundary left.

    Rare, and always a long enumeration written as one clause. Splitting between
    words is the only cut left that does not land inside one, and returning the
    text untouched would put a passage back over the budget, which is the thing
    this is all for.
    """
    if len(text) <= budget:
        return [text]

    runs: list[str] = []
    current = ""
    for word in text.split():
        if current and len(current) + 1 + len(word) > budget:
            runs.append(current)
            current = word
        else:
            current = f"{current} {word}" if current else word
    if current:
        runs.append(current)
    return runs


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
        # The blank line between paragraphs is charged against the budget too,
        # because it is in the passage. Counting only the paragraphs lets a
        # passage of many short ones run hundreds of characters past the budget,
        # which is exactly the overshoot the budget exists to prevent. A
        # paragraph is charged its cost rather than its length, which is the
        # same thing for ordinary prose and more for text dense with figures.
        addition = _cost(block) + (_JOIN_CHARS if current else 0)
        if size >= CHUNK_CHAR_MINIMUM and size + addition > budget:
            passages.append(current)
            current = _overlap_tail(current, blocks, overlap)
            size = _packed_size(current, blocks)
            addition = _cost(block) + (_JOIN_CHARS if current else 0)
        current.append(position)
        size += addition

    if current:
        carried = _packed_size(current, blocks)
        if passages and carried < CHUNK_CHAR_MINIMUM:
            # The minimum guards the flush inside the loop but not the last
            # passage, which is whatever is left over when the blocks run out.
            # That leftover is padded by the overlap tail, except when the
            # paragraph just flushed was itself bigger than the overlap budget:
            # then the tail is empty and a two-character closing block becomes a
            # passage of its own, which is an embedding of nothing. Fold it into
            # the passage before it instead, which runs that one over budget by
            # less than the minimum.
            previous = passages[-1]
            previous.extend(position for position in current if position not in previous)
        else:
            passages.append(current)
    return passages


def prose_blocks(section: SectionRecord) -> tuple[list[str], list[str]]:
    """An Item's paragraphs as the chunker packs them, and those it drops as
    flattened copies of its rebuilt tables.

    One function for both, because ``verify`` has to agree with the chunker on
    what may be dropped: a paragraph missing from every passage is either
    debris this rule removed on purpose, or prose that was lost.
    """
    blocks = [block.strip() for block in section.text.split("\n\n") if block.strip()]
    blocks = _strip_furniture(blocks, has_tables=section.n_tables > 0)
    blocks = _rejoin_page_breaks(blocks)
    # Where the Item's tables were rebuilt properly, the flattened copies
    # still sitting in the prose are pure noise, so they are dropped rather
    # than indexed alongside the readable version.
    if not section.tables:
        return blocks, []
    figures = table_figures(section.tables)
    cells = table_cells(section.tables)
    kept: list[str] = []
    dropped: list[str] = []
    for block in blocks:
        (dropped if _is_table_debris(block, figures, cells) else kept).append(block)
    return kept, dropped


def chunk_section(
    section: SectionRecord,
    accession_no: str,
    incorporated_into: list[str] | None = None,
    budget: int = CHUNK_CHAR_BUDGET,
    overlap: int = CHUNK_CHAR_OVERLAP,
) -> list[ChunkRecord]:
    """Cut one Item into passages."""
    blocks, _ = prose_blocks(section)
    if not blocks:
        return []
    # After the debris filter, so a flattened table is judged as the one block
    # the extractor produced, and before headings, so every heading is measured
    # against the block that actually follows it. A dense paragraph is split to
    # the characters its cost allows, not to the prose budget in characters.
    blocks = [
        part
        for block in blocks
        for part in _split_long_block(
            block, round(budget * _chars_per_token(block) / PROSE_CHARS_PER_TOKEN)
        )
    ]

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
        # What these passages were cut with, recorded here because this is the
        # only place that knows. Everything downstream can otherwise do no
        # better than read the constants and hope they have not moved since.
        chunk_budget=budget,
        chunk_overlap=overlap,
        key_items_only=key_items_only,
    )


def _slice_table_rows(
    rows: list[list[str]], header_chars: int, budget: int,
) -> list[list[list[str]]]:
    """Group table rows into slices that fit the passage budget.

    Rows are capped by count and by width together. A count alone is not enough,
    because a table can be wide as well as long: thirty rows of a twelve-column
    schedule run well past the budget on their own. ``header_chars`` is the cost
    of the header and rule repeated on every slice, which is charged against the
    budget before any row is added.
    """
    slices: list[list[list[str]]] = []
    current: list[list[str]] = []
    size = header_chars
    for row in rows:
        # "| " before each cell and " " after it, the closing "|", and the
        # newline that puts the row on its own line. Leaving the newline out
        # let every slice overrun by one character per row.
        width = sum(len(cell) for cell in row) + 3 * len(row) + 2
        too_many = len(current) >= TABLE_MAX_ROWS_PER_CHUNK
        too_wide = current and size + width > budget
        if too_many or too_wide:
            slices.append(current)
            current, size = [], header_chars
        current.append(row)
        size += width
    if current:
        slices.append(current)
    return slices or [rows]


# The widest "(part n of m)" suffix a table is expected to need, reserved in
# the budget before a table is split. A table cut into a hundred parts or more
# overruns it by a character or two.
_PART_SUFFIX_RESERVE = " (part 99 of 99)"


def chunk_tables(
    section: SectionRecord,
    accession_no: str,
    incorporated_into: list[str] | None = None,
    budget: int = CHUNK_CHAR_BUDGET,
    table_budget: int | None = None,
) -> list[ChunkRecord]:
    """Turn each rebuilt table into passages that keep their column labels.

    A table is kept whole where it fits. A long one is split by rows and a wide
    one by columns, and in both cases the header is repeated on every piece, so
    no passage of figures ever arrives without the labels that say what the
    figures are. Splitting by column is what bounds a table whose single row is
    already wider than the budget, which no amount of row slicing reaches.

    ``budget`` is the prose budget, and the table budget is derived from it by
    ``constants.table_budget_for`` unless one is passed. Tables need their own
    because figures tokenise about twice as densely as prose, so a table cut to
    the prose budget overruns the embedding model's window and is silently
    truncated. Passing ``table_budget`` overrides the derivation, which is what
    a sweep comparing the two rates would do.
    """
    budget = table_budget if table_budget is not None else table_budget_for(budget)
    passages: list[ChunkRecord] = []
    for table in section.tables:
        if not table.rows or not table.headers:
            continue

        # The caption line opens every piece, so like the header it is charged
        # against the budget before anything is split, on both axes. Left
        # uncharged it put a third of table passages over the budget by about
        # its own width. The part suffix is reserved at its widest, because how
        # many parts there are is not known until the splitting it would change
        # has been done.
        label = table.caption or section.title or f"Item {section.item}"
        # Never below half the budget: a caption long enough to take more would
        # otherwise shatter its table into one-row pieces, which is worse than
        # letting that one caption overrun.
        room = max(budget - len(label) - len(_PART_SUFFIX_RESERVE) - len("\n\n"), budget // 2)

        # Columns first, then rows within each group, so that every piece is
        # inside the budget on both axes.
        pieces: list[tuple[list[str], list[list[str]]]] = []
        for headers, rows in _split_table_columns(table.headers, table.rows, room):
            # The header and the rule beneath it are repeated on every slice,
            # so their cost comes off the budget before any row is added.
            header_chars = len(render_table(headers, []))
            for row_slice in _slice_table_rows(rows, header_chars, room):
                pieces.append((headers, row_slice))

        for part_number, (headers, row_slice) in enumerate(pieces):
            title = label
            if len(pieces) > 1:
                title = f"{label} (part {part_number + 1} of {len(pieces)})"
            text = "\n".join([title, "", render_table(headers, row_slice)])
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


def report(chunked_filings: list[ChunkedFiling]) -> None:
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
