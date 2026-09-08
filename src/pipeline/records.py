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
