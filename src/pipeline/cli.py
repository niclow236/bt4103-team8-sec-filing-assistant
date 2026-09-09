"""Command-line argument parsers for the data-pipeline commands.

The parsers live here, apart from the modules that run them, so ``download.py``
and ``parse.py`` read as the work they do rather than as argument wiring, and so
every pipeline command declares its interface in one predictable place.
"""

from __future__ import annotations

import argparse

from .constants import (
    CHUNK_CHAR_BUDGET,
    CHUNK_CHAR_OVERLAP,
    DEFAULT_FILING_YEARS,
    DEFAULT_FISCAL_YEARS,
    DEFAULT_FORMS,
)


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
        "--fiscal-years", nargs=2, type=int, metavar=("START", "END"),
        default=list(DEFAULT_FISCAL_YEARS),
        help="Inclusive range of fiscal years the filings report on "
             f"(default: {DEFAULT_FISCAL_YEARS[0]} {DEFAULT_FISCAL_YEARS[1]}). "
             "This is the scope of the corpus, and the axis questions are asked "
             "on. Pass 0 0 to keep every year found.",
    )
    parser.add_argument(
        "--years", nargs=2, type=int, metavar=("START", "END"),
        default=list(DEFAULT_FILING_YEARS),
        help="Inclusive range of FILING years to search, which is how EDGAR "
             f"indexes (default: {DEFAULT_FILING_YEARS[0]} {DEFAULT_FILING_YEARS[1]}). "
             "It runs a year past --fiscal-years to reach companies that close "
             "in December and file in January. Pass 0 0 to search every year.",
    )
    parser.add_argument(
        "--limit", type=int,
        help="Keep only the N most recent filings per company. Useful for a quick test.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List what would be downloaded and stop, without fetching any "
             "document or writing to the manifest. Use it to check the scope "
             "before spending a long download on the wrong one.",
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
    parser.add_argument(
        "--table-debug", action="store_true",
        help="Write the HTML of every table that could not be rebuilt to "
             "data/diagnostics/table_failures/, with an index naming the reason "
             "for each. Use it to work out why a particular table will not "
             "convert: a merged cell and a spacer row look the same from the "
             "outside, and only the markup tells them apart.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Show the extractor's own commentary on how it located each Item. "
             "It is quiet by default because it narrates every strategy it "
             "tries, including the ones it abandons, which reads like a run of "
             "errors when the parse has in fact succeeded.",
    )
    return parser


def build_chunk_parser(description: str = "") -> argparse.ArgumentParser:
    """Arguments for ``python -m src.pipeline.chunk``."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--tickers", nargs="+",
        help="Only chunk these companies. Defaults to every filing in data/interim/.",
    )
    parser.add_argument(
        "--forms", nargs="+",
        help="Only chunk these forms, for example --forms 10-K.",
    )
    parser.add_argument(
        "--key-items-only", action="store_true",
        help="Chunk only the Items the project targets, rather than every section "
             "that holds text. Useful for comparing a narrow index against a full one.",
    )
    parser.add_argument(
        "--budget", type=int, default=CHUNK_CHAR_BUDGET, metavar="CHARS",
        help="Characters per passage, not tokens "
             f"(default: {CHUNK_CHAR_BUDGET} characters, about "
             f"{CHUNK_CHAR_BUDGET // 4} tokens of English prose).",
    )
    parser.add_argument(
        "--overlap", type=int, default=CHUNK_CHAR_OVERLAP, metavar="CHARS",
        help="Characters carried from one passage into the next, not tokens "
             f"(default: {CHUNK_CHAR_OVERLAP} characters). Whole paragraphs "
             "are moved, so this is a budget for the carried tail rather than "
             "an exact overlap.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-chunk filings that already have output in data/processed/.",
    )
    return parser


def build_passages_parser(description: str = "") -> argparse.ArgumentParser:
    """Arguments for ``python -m src.pipeline passages``."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--tickers", nargs="+",
        help="Only passages from these companies.",
    )
    parser.add_argument(
        "--item", nargs="+",
        help="Only passages from these Items, for example --item 8 or --item 1A 7.",
    )
    parser.add_argument(
        "--forms", nargs="+",
        help="Only passages from these forms, for example --forms 10-K.",
    )
    parser.add_argument(
        "--fiscal-years", nargs=2, type=int, metavar=("START", "END"),
        help="Only passages from filings reporting on these fiscal years.",
    )
    parser.add_argument(
        "--content-type", choices=("prose", "table"),
        help="Only prose passages, or only rebuilt tables.",
    )
    parser.add_argument(
        "--contains", metavar="TEXT",
        help="Only passages whose text contains this, matched without regard to case.",
    )
    parser.add_argument(
        "--key-items-only", action="store_true",
        help="Only passages from the Items the project targets.",
    )
    parser.add_argument(
        "--limit", type=int, default=10, metavar="N",
        help="Show at most N passages (default: 10). Pass 0 for all of them.",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Print each passage in full rather than its opening lines.",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Write matching passages as JSON lines, for piping into another tool.",
    )
    return parser
