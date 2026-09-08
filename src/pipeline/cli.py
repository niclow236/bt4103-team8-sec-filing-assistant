"""Command-line argument parsers for the data-pipeline commands.

The parsers live here, apart from the modules that run them, so ``download.py``
and ``parse.py`` read as the work they do rather than as argument wiring, and so
every pipeline command declares its interface in one predictable place.
"""

from __future__ import annotations

import argparse

from .constants import DEFAULT_FORMS, DEFAULT_YEARS


def build_download_parser(description: str = "") -> argparse.ArgumentParser:
    """Arguments for ``python -m src.pipeline.download``."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--tickers", nargs="+",
        help="Tickers to download. Defaults to every entry in config/companies.txt.",
    )
    parser.add_argument(
        "--forms", nargs="+", default=list(DEFAULT_FORMS),
        help=f"Filing forms to fetch (default: {' '.join(DEFAULT_FORMS)}).",
    )
    parser.add_argument(
        "--years", nargs=2, type=int, metavar=("START", "END"),
        default=list(DEFAULT_YEARS),
        help="Inclusive range of filing years "
             f"(default: {DEFAULT_YEARS[0]} {DEFAULT_YEARS[1]}). "
             "Pass 0 0 to fetch every year on record.",
    )
    parser.add_argument(
        "--limit", type=int,
        help="Keep only the N most recent filings per company. Useful for a quick test.",
    )
    return parser


def build_parse_parser(description: str = "") -> argparse.ArgumentParser:
    """Arguments for ``python -m src.pipeline.parse``."""
    parser = argparse.ArgumentParser(description=description)
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
    return parser
