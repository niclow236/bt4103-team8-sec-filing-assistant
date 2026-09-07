"""Shared paths and environment setup for the data pipeline.

Every pipeline module imports its paths from here so that no other file has to
guess where the project root is, and so the folder layout is changed in one
place if it ever moves.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from edgar import set_identity

# config.py lives at <project root>/src/pipeline/config.py, so the root is two
# directories up from the package that contains this file.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"            # full downloaded filings (git-ignored)
INTERIM_DIR = DATA_DIR / "interim"    # parsed sections (git-ignored)
PROCESSED_DIR = DATA_DIR / "processed"  # chunks ready for indexing (git-ignored)
SAMPLE_DIR = DATA_DIR / "sample"      # small committed sample

CONFIG_DIR = PROJECT_ROOT / "config"
COMPANIES_FILE = CONFIG_DIR / "companies.txt"

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
    for directory in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR, SAMPLE_DIR):
        directory.mkdir(parents=True, exist_ok=True)
