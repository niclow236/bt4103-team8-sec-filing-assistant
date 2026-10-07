"""The download stage (#49): which filings are fetched, and what the manifest says of them.

EDGAR is replaced with a company that serves a fixed list of filings, so no
request is made. Every test runs in a temporary folder: the stage's own
defaults point at ``data/raw/``, and ``sandbox`` moves each of them and then
checks that the project's manifest is as it was.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from src import config
from src.pipeline import download
from src.pipeline.records import FilingRecord


@pytest.fixture(autouse=True)
def sandbox(tmp_path, monkeypatch):
    """Point everything the stage writes at ``tmp_path``, and check nothing else moved."""
    real = config.MANIFEST_FILE
    before = (real.stat().st_size, real.stat().st_mtime_ns) if real.exists() else None
    raw = tmp_path / "data" / "raw"
    manifest = raw / "manifest.jsonl"
    append, load = download._append_to_manifest, download.load_manifest
    company = download.download_company
    monkeypatch.setattr(download, "_append_to_manifest",
                        lambda record, manifest_file=manifest: append(record, manifest_file))
    monkeypatch.setattr(download, "load_manifest",
                        lambda manifest_file=manifest: load(manifest_file))
    # Its own default is data/raw/, which on a machine that holds a corpus is a
    # real folder: a call that names no folder saves under tmp_path instead.
    monkeypatch.setattr(download, "download_company", lambda ticker, forms, **settings:
                        company(ticker, forms, **{"raw_dir": raw, **settings}))
    monkeypatch.setattr(download, "ensure_data_dirs", lambda: None)
    monkeypatch.setattr(download, "RAW_DIR", raw)
    monkeypatch.setattr(download, "Company", NoEdgar)
    yield SimpleNamespace(raw=raw, manifest=manifest)
    after = (real.stat().st_size, real.stat().st_mtime_ns) if real.exists() else None
    assert after == before, "a download test wrote to the project's own manifest"


class NoEdgar:
    def __init__(self, ticker):
        pytest.fail(f"a test asked EDGAR for {ticker} without saying what it should answer")


class Filing:
    """One filing as EDGAR's index describes it, with the document it would serve."""

    def __init__(self, accession_no, filing_date, period_of_report, form="10-K",
                 html="<html>the filing</html>", text=""):
        self.accession_no = accession_no
        self.filing_date = filing_date
        self.period_of_report = period_of_report
        self.form = form
        self.cik = 320193
        self.company = "Apple Inc."
        self.filing_url = f"https://www.sec.gov/Archives/{accession_no}.htm"
        self._html, self._text = html, text
        self.fetched = 0

    def html(self):
        self.fetched += 1
        return self._html

    def text(self):
        return self._text


class Filings(list):
    @property
    def empty(self):
        return not self

    def head(self, count):
        return Filings(self[:count])


def serving(monkeypatch, *filings, seen=None):
    """Make EDGAR answer every company with these filings, newest first."""
    class Company:
        def __init__(self, ticker):
            self.ticker = ticker

        def get_filings(self, **search):
            if seen is not None:
                seen.append((self.ticker, search))
            return Filings(filings)

    monkeypatch.setattr(download, "Company", Company)


FY2025 = ("0000320193-25-000079", "2025-10-31", "2025-09-27")
FY2024 = ("0000320193-24-000123", "2024-11-01", "2024-09-28")
FY2020 = ("0000320193-20-000096", "2020-10-30", "2020-09-26")


def fetch(sandbox, **settings):
    return download.download_company("AAPL", ["10-K"], raw_dir=sandbox.raw, **settings)


# --- one company ------------------------------------------------------------


def test_a_filing_is_saved_as_its_document_and_recorded_in_the_manifest(sandbox, monkeypatch):
    serving(monkeypatch, Filing(*FY2025))

    (record,) = fetch(sandbox)

    saved = sandbox.raw / "AAPL" / "10-K_2025-10-31_0000320193-25-000079.html"
    assert saved.read_text(encoding="utf-8") == "<html>the filing</html>"
    assert record == FilingRecord(
        ticker="AAPL", cik=320193, company="Apple Inc.", form="10-K", filing_date="2025-10-31",
        accession_no="0000320193-25-000079",
        url="https://www.sec.gov/Archives/0000320193-25-000079.htm",
        # Relative to the project root with forward slashes, so a manifest built on
        # Windows still resolves on a teammate's Mac.
        path="data/raw/AAPL/10-K_2025-10-31_0000320193-25-000079.html",
        period_of_report="2025-09-27",
    )
    assert record.fiscal_year == 2025
    assert [json.loads(line) for line in sandbox.manifest.read_text().splitlines()] == [
        asdict(record)]
    assert download.load_manifest() == [record]


def test_the_search_is_over_filing_years_newest_first(sandbox, monkeypatch):
    seen = []
    serving(monkeypatch, Filing(*FY2025), seen=seen)

    fetch(sandbox, years=range(2021, 2027))

    assert seen == [("AAPL", {"form": ["10-K"], "year": [2021, 2022, 2023, 2024, 2025, 2026],
                              "sort_by": [("filing_date", "descending")]})]
    fetch(sandbox, years=None)
    assert seen[-1][1]["year"] is None


def test_a_filing_reporting_on_a_year_outside_the_scope_is_not_fetched(
        sandbox, monkeypatch, caplog):
    """The search runs over filing years, which reach a fiscal year either side of the scope."""
    inside, outside = Filing(*FY2024), Filing(*FY2020)
    serving(monkeypatch, inside, outside)

    with caplog.at_level(logging.INFO, logger="src.pipeline.download"):
        records = fetch(sandbox, fiscal_years=range(2021, 2026))

    assert [record.accession_no for record in records] == [FY2024[0]]
    assert (inside.fetched, outside.fetched) == (1, 0)
    assert "reports fiscal 2020, outside the scope" in caplog.text


def test_a_filing_already_in_the_manifest_is_skipped_so_a_run_can_be_resumed(
        sandbox, monkeypatch):
    first, second = Filing(*FY2025), Filing(*FY2024)
    serving(monkeypatch, first, second)
    held = {FY2025[0]}

    records = fetch(sandbox, already_downloaded=held)

    assert [record.accession_no for record in records] == [FY2024[0]]
    assert first.fetched == 0
    # What was just saved is held too, so the same run cannot fetch it twice.
    assert held == {FY2025[0], FY2024[0]}


def test_the_limit_keeps_the_most_recent_filings(sandbox, monkeypatch):
    serving(monkeypatch, Filing(*FY2025), Filing(*FY2024), Filing(*FY2020))
    assert [record.filing_date for record in fetch(sandbox, limit=2)] == [
        "2025-10-31", "2024-11-01"]


def test_a_filing_with_no_html_is_saved_as_its_plain_text(sandbox, monkeypatch):
    serving(monkeypatch, Filing(*FY2025, html="", text="PLAIN TEXT SUBMISSION"))

    (record,) = fetch(sandbox)

    assert record.path.endswith("_0000320193-25-000079.txt")
    assert (sandbox.raw / "AAPL" / "10-K_2025-10-31_0000320193-25-000079.txt").read_text(
        encoding="utf-8") == "PLAIN TEXT SUBMISSION"


def test_a_filing_with_no_readable_document_is_skipped_and_not_recorded(
        sandbox, monkeypatch, caplog):
    serving(monkeypatch, Filing(*FY2025, html="", text=""), Filing(*FY2024))

    with caplog.at_level(logging.WARNING, logger="src.pipeline.download"):
        records = fetch(sandbox)

    assert [record.accession_no for record in records] == [FY2024[0]]
    assert "has no readable document, skipped" in caplog.text
    assert len(sandbox.manifest.read_text().splitlines()) == 1


def test_a_company_edgar_returns_nothing_for_yields_nothing(sandbox, monkeypatch, caplog):
    serving(monkeypatch)
    with caplog.at_level(logging.WARNING, logger="src.pipeline.download"):
        assert fetch(sandbox) == []
    assert "AAPL: no filings returned for forms ['10-K']" in caplog.text
    assert not sandbox.manifest.exists()


def test_a_dry_run_selects_the_same_filings_and_writes_nothing(sandbox, monkeypatch):
    planned, skipped = Filing(*FY2024), Filing(*FY2020)
    serving(monkeypatch, planned, skipped)

    (record,) = fetch(sandbox, fiscal_years=range(2021, 2026), dry_run=True)

    assert record.path == "data/raw/AAPL/10-K_2024-11-01_0000320193-24-000123.html"
    assert record.fiscal_year == 2024
    assert planned.fetched == 0                       # the index was read, the document was not
    assert not sandbox.raw.exists() and not sandbox.manifest.exists()


def test_a_form_or_accession_with_an_odd_character_still_makes_a_file_name(
        sandbox, monkeypatch):
    serving(monkeypatch, Filing("0000320193/25:000079", "2025-10-31", "2025-09-27",
                                form="10-K/A"))
    (record,) = fetch(sandbox)
    assert record.path == "data/raw/AAPL/10-K-A_2025-10-31_0000320193-25-000079.html"
    assert (record.form, record.accession_no) == ("10-K/A", "0000320193/25:000079")


# --- every company ----------------------------------------------------------


def record(ticker, accession, period, filing_date="2025-02-01"):
    return FilingRecord(ticker=ticker, cik=1, company=f"{ticker} Corp", form="10-K",
                        filing_date=filing_date, accession_no=accession, url="u",
                        path=f"data/raw/{ticker}/{accession}.html", period_of_report=period)


def test_every_company_is_fetched_with_what_the_manifest_already_holds(sandbox, monkeypatch):
    download._append_to_manifest(record("AAA", "held-earlier", "2024-12-31"))
    calls = []

    def company(ticker, forms, **settings):
        calls.append((ticker, forms, settings))
        return [record(ticker, f"new-{ticker}", "2024-12-31")]

    monkeypatch.setattr(download, "download_company", company)

    new = download.download_all(["AAA", "BBB"], ["10-K"], years=range(2021, 2027), limit=3,
                                fiscal_years=range(2021, 2026))

    assert [item.accession_no for item in new] == ["new-AAA", "new-BBB"]
    assert [call[0] for call in calls] == ["AAA", "BBB"]
    for _, forms, settings in calls:
        assert forms == ["10-K"]
        assert settings["already_downloaded"] == {"held-earlier"}
        assert (settings["years"], settings["limit"]) == (range(2021, 2027), 3)
        assert (settings["fiscal_years"], settings["dry_run"]) == (range(2021, 2026), False)


def test_one_company_failing_does_not_end_the_download(sandbox, monkeypatch, caplog):
    def company(ticker, forms, **settings):
        if ticker == "BAD":
            raise ConnectionError("EDGAR did not answer")
        return [record(ticker, f"new-{ticker}", "2024-12-31")]

    monkeypatch.setattr(download, "download_company", company)

    with caplog.at_level(logging.ERROR, logger="src.pipeline.download"):
        new = download.download_all(["AAA", "BAD", "CCC"], ["10-K"])

    assert [item.ticker for item in new] == ["AAA", "CCC"]
    assert "BAD: download failed, moving on" in caplog.text


def test_only_a_real_run_makes_the_data_folders(sandbox, monkeypatch):
    made = []
    monkeypatch.setattr(download, "ensure_data_dirs", lambda: made.append(True))
    monkeypatch.setattr(download, "download_company", lambda *args, **settings: [])

    download.download_all(["AAA"], ["10-K"], dry_run=True)
    assert made == []
    download.download_all(["AAA"], ["10-K"])
    assert made == [True]


# --- what the corpus covers -------------------------------------------------


def test_a_manifest_line_written_before_the_period_was_recorded_still_loads(sandbox):
    sandbox.manifest.parent.mkdir(parents=True)
    old = {key: value for key, value in asdict(record("AAA", "old", "")).items()
           if key != "period_of_report"}
    sandbox.manifest.write_text(json.dumps(old) + "\n\n", encoding="utf-8")

    (loaded,) = download.load_manifest()
    assert (loaded.period_of_report, loaded.fiscal_year) == ("", None)
    assert download.load_manifest(sandbox.raw / "absent.jsonl") == []


def test_coverage_is_by_the_year_a_filing_reports_on_not_the_year_it_was_filed():
    """Filed in February 2025, reporting on 2024: it is the 2024 column it fills."""
    records = [record("AAA", "a24", "2024-12-31", "2025-02-01"),
               record("BBB", "b24", "2024-06-30", "2024-08-01"),
               record("AAA", "a23", "2023-12-31", "2024-02-01"),
               record("CCC", "c", "")]
    assert download.fiscal_year_coverage(records) == {2024: {"AAA", "BBB"}, 2023: {"AAA"}}


def test_the_coverage_table_names_the_company_a_year_is_missing(capsys):
    records = [record("AAA", "a24", "2024-12-31"), record("BBB", "b24", "2024-12-31"),
               record("AAA", "a23", "2023-12-31"), record("AAA", "a19", "2019-12-31")]

    download.report_coverage({"AAA", "BBB"}, scope=range(2023, 2025), records=records)
    printed = capsys.readouterr().out

    assert "FY2024:  2 of 2 companies\n" in printed
    assert "FY2023:  1 of 2 companies  missing BBB" in printed
    assert "FY2019:  1 of 2 companies  missing BBB  (outside the scope" in printed
    assert "Comparable across all 2 companies: FY2024 to FY2024" in printed


def test_the_coverage_table_says_when_no_year_is_complete_or_the_complete_ones_have_gaps(capsys):
    partial = [record("AAA", "a24", "2024-12-31")]
    download.report_coverage({"AAA", "BBB"}, records=partial)
    assert "No fiscal year yet holds all 2 companies." in capsys.readouterr().out

    gapped = [record(ticker, f"{ticker}{year}", f"{year}-12-31")
              for year in (2021, 2023) for ticker in ("AAA", "BBB")]
    gapped.append(record("AAA", "a22", "2022-12-31"))
    download.report_coverage({"AAA", "BBB"}, records=gapped)
    assert "Comparable across all 2 companies: FY2021, FY2023" in capsys.readouterr().out

    download.report_coverage({"AAA"}, records=[])
    assert capsys.readouterr().out == ""


def test_a_plan_lists_what_would_be_fetched_and_the_coverage_it_would_leave(sandbox, capsys):
    download._append_to_manifest(record("AAA", "held", "2023-12-31"))
    planned = [record("AAA", "a24", "2024-12-31", "2025-02-01"),
               record("BBB", "b24", "2024-12-31", "2025-02-03"),
               record("BBB", "b23", "2023-12-31", "2024-02-02")]

    download.report_plan(planned, {"AAA", "BBB"}, range(2023, 2025), ["AAA", "BBB"])
    printed = capsys.readouterr().out

    assert f"Would download 3 filings into {sandbox.raw}:" in printed
    assert "AAA    1 filings   FY2024" in printed
    assert "BBB    2 filings   FY2023 FY2024" in printed
    # What is held and what is planned, together: both years come out complete.
    assert "Comparable across all 2 companies: FY2023 to FY2024" in printed
    assert "Dry run: nothing was downloaded and the manifest is unchanged." in printed
    assert len(download.load_manifest()) == 1


def test_an_empty_plan_says_whether_there_was_nothing_left_or_nothing_found(sandbox, capsys):
    download.report_plan([], {"AAA"}, None, ["AAA"])
    assert "No filings matched this scope, and the manifest holds none" in capsys.readouterr().out

    download._append_to_manifest(record("AAA", "held", "2024-12-31"))
    download.report_plan([], {"AAA"}, None, ["AAA"])
    assert "Nothing new to download. The manifest already holds 1 filings" in (
        capsys.readouterr().out)
