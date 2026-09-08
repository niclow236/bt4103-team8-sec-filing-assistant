"""Download 10-K and 10-Q filings from SEC EDGAR into ``data/raw/``.

Run it from the project root:

    python -m src.pipeline.download                       # the whole project corpus
    python -m src.pipeline.download --years 2019 2020     # a different year range
    python -m src.pipeline.download --tickers AAPL MSFT --limit 2

With no arguments this downloads the project's agreed corpus: Form 10-K for
filing years 2021 to 2025, for every ticker in config/companies.txt. The
defaults live in DEFAULT_FORMS and DEFAULT_YEARS in src/pipeline/constants.py,
so the scope is set in one place rather than retyped on the command line.

Each filing is saved as its original HTML document, and one line describing it
is appended to ``data/raw/manifest.jsonl``. The download is resumable: a filing
already listed in the manifest is skipped, so re-running after an interruption
picks up where it left off rather than starting again.

Rate limiting is handled inside edgartools, which keeps requests under the
SEC's published limit, so this module does not add its own delays.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

from edgar import Company

from ..config import (
    MANIFEST_FILE,
    RAW_DIR,
    configure_edgar,
    ensure_data_dirs,
    read_tickers,
)
from .cli import build_download_parser
from .records import FilingRecord

logger = logging.getLogger(__name__)


def _safe_name(value: str) -> str:
    """Turn a value into something usable as part of a file name."""
    return "".join(char if char.isalnum() or char in "-_" else "-" for char in value)


def load_manifest(manifest_file: Path = MANIFEST_FILE) -> list[FilingRecord]:
    """Read every record already in the manifest, or return an empty list."""
    if not manifest_file.exists():
        return []

    records = []
    for line in manifest_file.read_text().splitlines():
        if line.strip():
            records.append(FilingRecord(**json.loads(line)))
    return records


def _append_to_manifest(record: FilingRecord, manifest_file: Path = MANIFEST_FILE) -> None:
    """Add one record to the manifest straight away.

    Writing after each filing rather than at the end means an interrupted run
    still leaves a manifest that matches what is actually on disk.
    """
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    with manifest_file.open("a") as handle:
        handle.write(json.dumps(asdict(record)) + "\n")


def download_company(
    ticker: str,
    forms: list[str],
    years: range | list[int] | None = None,
    limit: int | None = None,
    already_downloaded: set[str] | None = None,
    raw_dir: Path = RAW_DIR,
) -> list[FilingRecord]:
    """Download one company's filings and return a record for each new one.

    ``already_downloaded`` holds accession numbers to skip, which is how the
    resume behaviour works.
    """
    already_downloaded = already_downloaded or set()

    company = Company(ticker)
    filings = company.get_filings(
        form=forms,
        year=list(years) if years else None,
        # Newest first, so a --limit run gives the most recent filings.
        sort_by=[("filing_date", "descending")],
    )
    if filings is None or filings.empty:
        logger.warning("%s: no filings returned for forms %s", ticker, forms)
        return []

    if limit is not None:
        filings = filings.head(limit)

    company_dir = raw_dir / ticker
    company_dir.mkdir(parents=True, exist_ok=True)

    new_records: list[FilingRecord] = []
    for filing in filings:
        if filing.accession_no in already_downloaded:
            logger.info("%s: skipping %s, already downloaded", ticker, filing.accession_no)
            continue

        # A handful of older filings have no HTML document, so fall back to the
        # plain-text submission rather than losing the filing entirely.
        content = filing.html()
        suffix = ".html"
        if not content:
            content = filing.text()
            suffix = ".txt"
        if not content:
            logger.warning("%s: %s has no readable document, skipped", ticker, filing.accession_no)
            continue

        name = f"{_safe_name(filing.form)}_{filing.filing_date}_{_safe_name(filing.accession_no)}{suffix}"
        destination = company_dir / name
        destination.write_text(content, encoding="utf-8")

        record = FilingRecord(
            ticker=ticker,
            cik=filing.cik,
            company=filing.company,
            form=filing.form,
            filing_date=str(filing.filing_date),
            accession_no=filing.accession_no,
            url=filing.filing_url,
            path=str(destination.relative_to(raw_dir.parents[1])),
        )
        _append_to_manifest(record)
        new_records.append(record)
        already_downloaded.add(filing.accession_no)
        logger.info("%s: saved %s %s", ticker, filing.form, filing.filing_date)

    return new_records


def download_all(
    tickers: list[str],
    forms: list[str],
    years: range | list[int] | None = None,
    limit: int | None = None,
) -> list[FilingRecord]:
    """Download filings for every ticker, carrying on if one company fails."""
    ensure_data_dirs()
    already_downloaded = {record.accession_no for record in load_manifest()}
    logger.info("Manifest already holds %d filings", len(already_downloaded))

    all_new: list[FilingRecord] = []
    for ticker in tickers:
        try:
            new_records = download_company(
                ticker, forms, years=years, limit=limit,
                already_downloaded=already_downloaded,
            )
        except Exception:
            # One bad ticker should not end a download that may take a while,
            # so log it and move on. The summary at the end shows what landed.
            logger.exception("%s: download failed, moving on", ticker)
            continue
        all_new.extend(new_records)

    return all_new


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    parser = build_download_parser(__doc__.splitlines()[0] if __doc__ else "")
    args = parser.parse_args(argv)

    identity = configure_edgar()
    logger.info("Identifying to SEC EDGAR as: %s", identity)

    tickers = args.tickers or read_tickers()
    # "--years 0 0" is the escape hatch for an unfiltered download; anything
    # else, including the default, narrows the request to that range.
    years = range(args.years[0], args.years[1] + 1) if any(args.years) else None
    logger.info(
        "Scope: forms %s, years %s, %d companies",
        " ".join(args.forms),
        f"{args.years[0]}-{args.years[1]}" if years else "all",
        len(tickers),
    )

    new_records = download_all(tickers, args.forms, years=years, limit=args.limit)

    print(f"\nDownloaded {len(new_records)} new filings into {RAW_DIR}")
    print(f"Manifest: {MANIFEST_FILE}")


if __name__ == "__main__":
    main()
