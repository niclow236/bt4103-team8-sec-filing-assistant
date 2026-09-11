"""The data records the pipeline passes between its stages.

Kept apart from the code that builds them so a later stage -- chunking,
retrieval, evaluation -- can read a manifest line or an interim file by
importing the shape alone, without pulling in the download or parse logic.
Each record is frozen: once a stage has written it, nothing downstream edits it
in place.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FilingRecord:
    """One downloaded filing, as written to data/raw/manifest.jsonl."""

    ticker: str
    cik: int
    company: str
    form: str
    filing_date: str
    accession_no: str
    url: str
    path: str  # relative to the project root, so the manifest stays portable
    # The fiscal period the filing covers, as EDGAR reports it, e.g. "2024-06-30".
    # The filing DATE is not the fiscal year: Alphabet, Meta, and Amazon file in
    # January or February, so their 2021 filing reports fiscal 2020. Any question
    # that pins a year across companies has to read this rather than filing_date.
    # Defaulted so a manifest written before this field existed still loads.
    period_of_report: str = ""

    @property
    def fiscal_year(self) -> int | None:
        """The fiscal year the filing reports on, from the period end date."""
        return int(self.period_of_report[:4]) if self.period_of_report[:4].isdigit() else None


@dataclass(frozen=True)
class TableRecord:
    """One table of one Item, kept as a table rather than flattened into prose.

    The extractor's plain-text rendering detaches every figure from its row and
    column labels, so a balance sheet becomes an unattributable run of numbers.
    Storing the table separately keeps "Total revenue" joined to 245,122 and to
    the year above it.

    The grid is stored rather than a rendered table, because how a table is laid
    out depends on the passage budget, which is a chunking decision. A table too
    wide for one row to fit is split by column, and that is only possible from
    the cells.
    """

    # Position in this Item's ``tables`` list, counting from 0, so that
    # ``section.tables[record.table_index] is record``. It counts the tables
    # that were rebuilt, not every table in the Item: one that could not be
    # rebuilt is not stored, and numbering around it would break that lookup.
    table_index: int
    caption: str          # the table's own caption, or the first column's label
    # The full header row, including the leading cell above the row labels, so
    # that ``headers`` and each entry of ``rows`` are the same width and either
    # can be indexed by column. Repeated on every passage cut from the table.
    headers: list[str]
    # The body, one list per row, first cell holding that row's label.
    rows: list[list[str]]
    n_rows: int
    n_cols: int


@dataclass(frozen=True)
class SectionRecord:
    """One Item of one filing."""

    section_id: str       # e.g. "part_ii_item_7", as the parser names it
    part: str | None      # "I", "II", ...
    item: str | None      # "1", "1A", "7A", ...
    title: str            # e.g. "Risk Factors"
    text: str
    n_chars: int
    # Every <table> element the extractor found in this Item. Most filers wrap
    # bullet points in a one-cell table to indent them, so this counts layout
    # markup as well as data.
    n_tables: int
    # Those of them that actually hold a grid of data. The difference is the
    # layout wrappers, whose text is already in ``text``, so skipping them
    # loses nothing. Rebuilding is measured against this, not against
    # ``n_tables``, which would score formatting as a failure.
    n_data_tables: int
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
    # Tables lifted out of this Item, in document order. The prose in ``text``
    # still holds the extractor's flattened version; these are what the chunking
    # stage indexes, because only these keep the labels attached.
    tables: list[TableRecord] = field(default_factory=list)


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
    period_of_report: str = ""   # see FilingRecord.period_of_report


@dataclass(frozen=True)
class ChunkRecord:
    """One retrievable passage of one Item, as written to data/processed/."""

    chunk_id: str         # "<accession_no>_<section_id>_<index>", unique corpus-wide
    section_id: str       # the Item this passage was cut from
    part: str | None
    item: str | None
    title: str            # the official Item title, for the citation
    heading: str | None   # the nearest heading above the passage, where there was one
    text: str
    n_chars: int
    chunk_index: int      # position within the Item, counting from 0
    is_key_section: bool
    # Items that answer with a cross-reference to this one. Oracle's Item 8
    # points at Item 15, so Item 15's passages carry ["8"] and the text is
    # stored once rather than copied into both Items.
    incorporated_into: list[str] = field(default_factory=list)
    # "prose" or "table". A table passage carries its column headers, so figures
    # in it stay attributable; retrieval can also weight or filter on this when a
    # question is numeric.
    content_type: str = "prose"
    table_index: int | None = None   # which table, for a table passage
    table_caption: str = ""


@dataclass(frozen=True)
class ChunkedFiling:
    """One filing, cut into retrievable passages, as written to data/processed/."""

    ticker: str
    cik: int
    company: str
    form: str
    filing_date: str
    accession_no: str
    url: str
    source_path: str      # the interim file this was chunked from, posix-style
    chunks: list[ChunkRecord]
    period_of_report: str = ""   # see FilingRecord.period_of_report
    # The settings these passages were actually cut with, rather than whatever
    # the constants say when someone later reads the file. The chunk stage takes
    # --budget and --overlap, so the two can differ, and an index built from
    # this corpus records the budget its rows are reported against: without this
    # the chunk-size sweep would label every row with the same wrong number.
    #
    # Optional, and None on a file written before they were recorded, so an
    # older data/processed/ still loads through load_chunked rather than failing
    # on a missing key. A reader that gets None knows only that the settings
    # were not recorded, which is the truth and is worth being able to say.
    chunk_budget: int | None = None
    chunk_overlap: int | None = None
    # Whether the run that produced these passages was narrowed to the key
    # Items. Recorded for the same reason, though a change here is already
    # caught by the corpus fingerprint, since it changes which passages exist.
    # The fingerprint detects it; this explains it.
    key_items_only: bool = False
