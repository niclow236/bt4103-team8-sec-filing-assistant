"""Download 10-K and 10-Q filings from SEC EDGAR into ``data/raw/``.

Run it from the project root:

    python -m src.pipeline.download                            # the whole project corpus
    python -m src.pipeline.download --fiscal-years 2019 2020   # a different year range
    python -m src.pipeline.download --tickers AAPL MSFT --limit 2
    python -m src.pipeline.download --dry-run                   # preview only

With no arguments this downloads the project's agreed corpus: Form 10-K for
fiscal years 2021 to 2025, for every ticker in config/companies.txt. The
defaults live in DEFAULT_FORMS, DEFAULT_FISCAL_YEARS and DEFAULT_FILING_YEARS in
src/pipeline/constants.py, so the scope is set in one place rather than retyped
on the command line.

Note the two year ranges. EDGAR indexes filings by the date they were filed, but
a question is asked about a fiscal year, and the two differ by a year for any
company that closes its books in December. So the search runs over filing years
and the result is then narrowed to the fiscal years in scope, which is what
makes the corpus hold the same years for every company.

Each filing is saved as its original HTML document, and one line describing it
is appended to ``data/raw/manifest.jsonl``. The download is resumable: a filing
already listed in the manifest is skipped, so re-running after an interruption
picks up where it left off rather than starting again.

``--dry-run`` answers "what would this fetch?" without fetching it. Deciding
that still means asking EDGAR which filings exist, so the run makes one index
request per company, but it downloads no documents and writes nothing to disk
or to the manifest. That is where a scope mistake is cheap to notice: a wrong
year range or a mistyped ticker shows up as a preview, rather than as a long
download you have to unpick from the manifest afterwards.

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
    MissingIdentityError,
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


def _fiscal_year_of(filing) -> int | None:
    """The fiscal year a filing reports on, from its period of report."""
    period = str(getattr(filing, "period_of_report", "") or "")
    return int(period[:4]) if period[:4].isdigit() else None


def download_company(
    ticker: str,
    forms: list[str],
    years: range | list[int] | None = None,
    limit: int | None = None,
    already_downloaded: set[str] | None = None,
    raw_dir: Path = RAW_DIR,
    fiscal_years: range | list[int] | None = None,
    dry_run: bool = False,
) -> list[FilingRecord]:
    """Download one company's filings and return a record for each new one.

    ``already_downloaded`` holds accession numbers to skip, which is how the
    resume behaviour works. ``fiscal_years`` narrows the result to the years the
    filings report on, as opposed to ``years``, which narrows the EDGAR search
    to the years they were filed in.

    Under ``dry_run`` the same filings are selected and returned, but no
    document is fetched and nothing is written, so the caller can show what a
    real run would do.
    """
    already_downloaded = already_downloaded or set()
    wanted_fiscal = set(fiscal_years) if fiscal_years else None

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
    if not dry_run:
        company_dir.mkdir(parents=True, exist_ok=True)

    new_records: list[FilingRecord] = []
    for filing in filings:
        if filing.accession_no in already_downloaded:
            logger.info("%s: skipping %s, already downloaded", ticker, filing.accession_no)
            continue

        # The search window is set in filing years, so it reaches filings either
        # side of the fiscal years we want. Drop those here, before spending a
        # request on the document itself.
        fiscal_year = _fiscal_year_of(filing)
        if wanted_fiscal is not None and fiscal_year not in wanted_fiscal:
            logger.info(
                "%s: skipping %s, filed %s but reports fiscal %s, outside the scope",
                ticker, filing.accession_no, filing.filing_date, fiscal_year,
            )
            continue

        if dry_run:
            # Everything above this point is decided from the filing index, so
            # a preview can be built without fetching the document. The name is
            # written with the .html suffix a real run would almost always use;
            # the plain-text fallback below is rare enough that guessing it here
            # would be more misleading than assuming HTML.
            planned = company_dir / (
                f"{_safe_name(filing.form)}_{filing.filing_date}"
                f"_{_safe_name(filing.accession_no)}.html"
            )
            new_records.append(
                FilingRecord(
                    ticker=ticker,
                    cik=filing.cik,
                    company=filing.company,
                    form=filing.form,
                    filing_date=str(filing.filing_date),
                    accession_no=filing.accession_no,
                    url=filing.filing_url,
                    path=planned.relative_to(raw_dir.parents[1]).as_posix(),
                    period_of_report=str(getattr(filing, "period_of_report", "") or ""),
                )
            )
            already_downloaded.add(filing.accession_no)
            logger.info("%s: would download %s %s", ticker, filing.form, filing.filing_date)
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
            # as_posix so a manifest built on Windows still resolves on a
            # teammate's Mac, which is what makes the relative path portable.
            path=destination.relative_to(raw_dir.parents[1]).as_posix(),
            # The fiscal period the filing reports on, which is not the same as
            # the date it was filed. Recorded here so a later stage never has to
            # go back to EDGAR to find out which year a filing covers.
            period_of_report=str(getattr(filing, "period_of_report", "") or ""),
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
    fiscal_years: range | list[int] | None = None,
    dry_run: bool = False,
) -> list[FilingRecord]:
    """Download filings for every ticker, carrying on if one company fails."""
    if not dry_run:
        ensure_data_dirs()
    already_downloaded = {record.accession_no for record in load_manifest()}
    logger.info("Manifest already holds %d filings", len(already_downloaded))

    all_new: list[FilingRecord] = []
    for ticker in tickers:
        try:
            new_records = download_company(
                ticker, forms, years=years, limit=limit,
                already_downloaded=already_downloaded,
                fiscal_years=fiscal_years,
                dry_run=dry_run,
            )
        except Exception:
            # One bad ticker should not end a download that may take a while,
            # so log it and move on. The summary at the end shows what landed.
            logger.exception("%s: download failed, moving on", ticker)
            continue
        all_new.extend(new_records)

    return all_new


def fiscal_year_coverage(records: list[FilingRecord] | None = None) -> dict[int, set[str]]:
    """Which companies the corpus holds a filing for, per fiscal year."""
    coverage: dict[int, set[str]] = {}
    for record in records if records is not None else load_manifest():
        year = record.fiscal_year
        if year is not None:
            coverage.setdefault(year, set()).add(record.ticker)
    return coverage


def report_coverage(
    expected: set[str],
    scope: range | list[int] | None = None,
    records: list[FilingRecord] | None = None,
) -> None:
    """Print the fiscal years the corpus covers, and say which are incomplete.

    A question that pins one year across companies can only be answered where
    that year holds every company. Reporting it here means a gap is seen when
    the corpus is built, rather than inferred later from a thin answer.

    Pass ``records`` to describe a corpus other than the one on disk, which is
    how a dry run shows the coverage its download would end up with.
    """
    coverage = fiscal_year_coverage(records)
    if not coverage:
        return

    in_scope = set(scope) if scope else set(coverage)
    print("\nFiscal year coverage:")
    for year in sorted(coverage):
        have = coverage[year]
        missing = sorted(expected - have)
        mark = "" if not missing else f"  missing {' '.join(missing)}"
        note = "" if year in in_scope else "  (outside the scope, kept for single-company questions)"
        print(f"  FY{year}: {len(have):>2} of {len(expected)} companies{mark}{note}")

    complete = sorted(year for year in coverage if year in in_scope and expected <= coverage[year])
    if complete:
        print(
            f"  Comparable across all {len(expected)} companies: "
            f"FY{complete[0]} to FY{complete[-1]}"
            if complete == list(range(complete[0], complete[-1] + 1))
            else f"  Comparable across all {len(expected)} companies: "
                 + ", ".join(f"FY{year}" for year in complete)
        )
    else:
        print(f"  No fiscal year yet holds all {len(expected)} companies.")


def _report_plan(planned: list[FilingRecord], expected: set[str],
                 scope: range | list[int] | None, tickers: list[str]) -> None:
    """Say what a real run would download, and change nothing."""
    if not planned:
        # An empty plan has two very different causes, and saying the wrong one
        # sends someone hunting for a bug in the wrong place. Nothing left to
        # fetch is the happy case; nothing found at all usually means the
        # tickers or the year range are wrong, which is what a preview is for.
        held = [record for record in load_manifest() if record.ticker in set(tickers)]
        if held:
            print(f"\nNothing new to download. The manifest already holds "
                  f"{len(held)} filings for these companies.")
        else:
            print("\nNo filings matched this scope, and the manifest holds none "
                  "for these companies either. Check the tickers and the year range.")
    else:
        print(f"\nWould download {len(planned)} filings into {RAW_DIR}:")
        by_ticker: dict[str, list[FilingRecord]] = {}
        for record in planned:
            by_ticker.setdefault(record.ticker, []).append(record)
        for ticker in sorted(by_ticker):
            records = sorted(by_ticker[ticker], key=lambda record: record.filing_date)
            years = " ".join(
                f"FY{record.fiscal_year}" if record.fiscal_year else record.filing_date
                for record in records
            )
            print(f"  {ticker:6} {len(records)} filings   {years}")

    # The coverage a real run would leave behind, which is the point of the
    # preview: it shows a year range that comes out short before the download
    # is spent finding that out.
    report_coverage(expected, scope=scope, records=load_manifest() + planned)
    print("\nDry run: nothing was downloaded and the manifest is unchanged.")


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    parser = build_download_parser(__doc__.splitlines()[0] if __doc__ else "")
    args = parser.parse_args(argv)

    try:
        identity = configure_edgar()
    except MissingIdentityError as error:
        # An unset contact string is a setup step someone has not done yet, not
        # a fault in the code, so print what to do about it and stop. A stack
        # trace here would bury the one line that actually helps.
        raise SystemExit(f"\n{error}\n") from None
    logger.info("Identifying to SEC EDGAR as: %s", identity)

    tickers = args.tickers or read_tickers()
    # "--years 0 0" is the escape hatch for an unfiltered download; anything
    # else, including the default, narrows the request to that range.
    years = range(args.years[0], args.years[1] + 1) if any(args.years) else None
    # Likewise "--fiscal-years 0 0" keeps every year a filing reports on.
    fiscal_years = (
        range(args.fiscal_years[0], args.fiscal_years[1] + 1)
        if any(args.fiscal_years) else None
    )
    logger.info(
        "Scope: forms %s, fiscal years %s, searched over filing years %s, %d companies",
        " ".join(args.forms),
        f"{args.fiscal_years[0]}-{args.fiscal_years[1]}" if fiscal_years else "all",
        f"{args.years[0]}-{args.years[1]}" if years else "all",
        len(tickers),
    )

    new_records = download_all(
        tickers, args.forms, years=years, limit=args.limit, fiscal_years=fiscal_years,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        _report_plan(new_records, set(read_tickers()), fiscal_years, tickers)
        return

    print(f"\nDownloaded {len(new_records)} new filings into {RAW_DIR}")
    print(f"Manifest: {MANIFEST_FILE}")
    report_coverage(set(read_tickers()), scope=fiscal_years)


if __name__ == "__main__":
    main()
