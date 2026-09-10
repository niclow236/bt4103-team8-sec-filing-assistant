"""Project-wide paths and environment setup.

Every package under ``src`` imports its paths from here so that no other file
has to guess where the project root is, and so the folder layout is changed in
one place if it ever moves. Stage-specific values live in that stage's own
module, for example ``src/pipeline/constants.py``.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from edgar import set_identity

# config.py lives at <project root>/src/config.py, so the root is the parent of
# the src package that contains this file.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"            # full downloaded filings (git-ignored)
INTERIM_DIR = DATA_DIR / "interim"    # parsed sections (git-ignored)
PROCESSED_DIR = DATA_DIR / "processed"  # chunks ready for indexing (git-ignored)
SAMPLE_DIR = DATA_DIR / "sample"      # small committed sample
# Output written to explain a run rather than to feed the next stage, such
# as the HTML of a table the parser could not rebuild. Git-ignored, and
# created on demand rather than by ensure_data_dirs, since most runs write
# nothing here.
DIAGNOSTICS_DIR = DATA_DIR / "diagnostics"

# Built search indexes: the corpus in the form a retriever can search, rather
# than the form it is stored in. Git-ignored and rebuildable, like every other
# data directory -- an index is derived from data/processed/ and is not worth
# versioning, but it IS worth knowing which corpus it came from, which is what
# the IndexManifest written beside it records. BM25 serialises to one pickle;
# Chroma wants a directory it manages itself.
INDEX_DIR = DATA_DIR / "index"
BM25_INDEX_FILE = INDEX_DIR / "bm25.pkl"
CHROMA_DIR = INDEX_DIR / "chroma"

CONFIG_DIR = PROJECT_ROOT / "config"
COMPANIES_FILE = CONFIG_DIR / "companies.txt"

# Terminal output from a run, kept so a number quoted in a report can be traced
# back to the run that produced it. Git-ignored: these are records of what
# happened on one machine, not shared source.
LOGS_DIR = PROJECT_ROOT / "logs"

# One JSON record per downloaded filing, so later stages know what exists on
# disk without having to walk the directory tree or re-query EDGAR.
MANIFEST_FILE = RAW_DIR / "manifest.jsonl"


class MissingIdentityError(RuntimeError):
    """Raised when EDGAR_IDENTITY is not set."""


def configure_edgar() -> str:
    """Load the local .env and hand our contact details to edgartools.

    The SEC requires every automated request to carry a contact string in the
    User-Agent header, and blocks traffic that does not have one. edgartools
    reads EDGAR_IDENTITY itself, but we set it explicitly so a missing value
    fails immediately with a clear message instead of part-way through a long
    download.

    Returns the identity string that was applied.
    """
    load_dotenv(PROJECT_ROOT / ".env")

    identity = os.getenv("EDGAR_IDENTITY", "").strip()
    if not identity:
        raise MissingIdentityError(
            "EDGAR_IDENTITY is not set. Copy .env.example to .env and fill in "
            'your name and email, for example: EDGAR_IDENTITY="Jane Tan '
            'jane@example.com". The SEC blocks requests without a contact string.'
        )

    set_identity(identity)
    return identity


def read_tickers(path: Path = COMPANIES_FILE) -> list[str]:
    """Read the ticker list, ignoring blank lines, comments, and inline notes.

    A line may carry a trailing comment, as in ``AAPL    # Apple``, so anything
    from the first ``#`` onwards is stripped before the ticker is read.
    """
    if not path.exists():
        raise FileNotFoundError(f"Ticker list not found: {path}")

    tickers: list[str] = []
    for line in path.read_text().splitlines():
        ticker = line.split("#", 1)[0].strip().upper()
        if ticker:
            tickers.append(ticker)
    return tickers


def ensure_data_dirs() -> None:
    """Create the data folders if they are not there yet."""
    for directory in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR, SAMPLE_DIR, INDEX_DIR):
        directory.mkdir(parents=True, exist_ok=True)
