"""The corpus gate (#49): each check passes a sound corpus and fails on the fault it is for.

``python -m src.pipeline verify`` is the last thing run before the corpus is
handed to retrieval, and it exits non-zero when the corpus is not fit to
index. ``test_verify.py`` covers the two checks that read passages alone. The
rest read all three stages' folders, the manifest and EDGAR, so here a small
corpus is written to a temporary folder by the stages' own records, each fault
is put into it by hand, and EDGAR is replaced by a company and a fact service
that answer from a fixed list.
"""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

from src.pipeline import verify
from src.pipeline.chunk import chunk_filing, iter_chunks, write_chunks
from src.pipeline.records import FilingRecord, ParsedFiling, SectionRecord, TableRecord
from src.pipeline.verify import (
    Check,
    _row_in,
    check_against_edgar,
    check_chunk_integrity,
    check_coverage,
    check_key_items,
    check_no_prose_lost,
    check_stage_parity,
    check_xbrl_figures,
    report,
    run_checks,
)

TICKERS = ["AAA", "BBB"]
YEARS = (2023, 2024)
STATEMENTS = ("CONSOLIDATED BALANCE SHEETS", "CONSOLIDATED STATEMENTS OF OPERATIONS",
              "CONSOLIDATED STATEMENTS OF CASH FLOWS")


def _name(ticker: str, year: int) -> str:
    return f"10-K_{year + 1}-02-01_{ticker}-{year}"


def _record(ticker: str, year: int) -> FilingRecord:
    return FilingRecord(
        ticker=ticker, cik=TICKERS.index(ticker) + 1, company=f"{ticker} Corp", form="10-K",
        filing_date=f"{year + 1}-02-01", accession_no=f"{ticker}-{year}",
        url=f"https://example.test/{ticker}-{year}",
        path=f"data/raw/{ticker}/{_name(ticker, year)}.html", period_of_report=f"{year}-12-31",
    )


def _assets(ticker: str, year: int) -> int:
    """Total assets in millions, different for every filing."""
    return 9000 + 100 * TICKERS.index(ticker) + year % 100


def _parsed(record: FilingRecord) -> ParsedFiling:
    year = int(record.period_of_report[:4])

    def prose(subject: str) -> str:
        return "\n\n".join(
            f"{record.ticker} {subject} paragraph {number} for fiscal {year} sets out one "
            "matter in enough detail that it is plainly a paragraph of the filing, and not a "
            "heading or a caption, which is what the check for lost prose looks for. It runs "
            "past two hundred characters even with the spaces taken out, as that check counts."
            for number in range(3))

    def section(item: str, title: str, text: str, tables=()) -> SectionRecord:
        return SectionRecord(
            section_id=f"part_x_item_{item.lower()}", part="I", item=item, title=title,
            text=text, n_chars=len(text), n_tables=len(tables), n_data_tables=len(tables),
            is_key_section=True, is_stub=False, resolved_from=None, confidence=None,
            detection_method=None, validated=True, tables=list(tables),
        )

    tables = [
        TableRecord(
            table_index=index, caption="", headers=["(In millions)", str(year)],
            rows=[["Total assets" if index == 0 else f"Line {index}",
                   f"{_assets(record.ticker, year) + index:,}"],
                  ["Other", f"{1000 + index:,}"]],
            n_rows=2, n_cols=2, statement_title=title,
        )
        for index, title in enumerate(STATEMENTS)
    ]
    return ParsedFiling(
        ticker=record.ticker, cik=record.cik, company=record.company, form=record.form,
        filing_date=record.filing_date, accession_no=record.accession_no, url=record.url,
        source_path=record.path, period_of_report=record.period_of_report,
        sections=[
            section("1", "Business", prose("business")),
            section("1A", "Risk Factors", prose("risk")),
            section("7", "Management's Discussion and Analysis", prose("discussion")),
            section("7A", "Market Risk", prose("market risk")),
            section("8", "Financial Statements", prose("statement"), tables),
        ],
    )


@pytest.fixture
def disk(tmp_path, monkeypatch):
    """Four filings at all three stages, and the gate pointed at them."""
    raw, interim, processed = (tmp_path / "data" / name for name in
                               ("raw", "interim", "processed"))
    records = [_record(ticker, year) for ticker in TICKERS for year in YEARS]
    for record in records:
        name = _name(record.ticker, int(record.period_of_report[:4]))
        parsed = _parsed(record)
        for folder, suffix, content in (
            (raw, ".html", "<html>the filing</html>"),
            (interim, ".json", json.dumps(asdict(parsed))),
        ):
            path = folder / record.ticker / f"{name}{suffix}"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        write_chunks(chunk_filing(parsed, source_path=f"data/interim/{record.ticker}/{name}.json"),
                     processed / record.ticker / f"{name}.json")

    interim_for, processed_for = verify.interim_path_for, verify.processed_path_for
    monkeypatch.setattr(verify, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(verify, "RAW_DIR", raw)
    monkeypatch.setattr(verify, "INTERIM_DIR", interim)
    monkeypatch.setattr(verify, "PROCESSED_DIR", processed)
    monkeypatch.setattr(verify, "DEFAULT_FISCAL_YEARS", YEARS)
    monkeypatch.setattr(verify, "interim_path_for", lambda record: interim_for(record, interim))
    monkeypatch.setattr(verify, "processed_path_for", lambda path: processed_for(path, processed))
    monkeypatch.setattr(verify, "iter_chunks", lambda: iter_chunks(processed_dir=processed))
    return SimpleNamespace(root=tmp_path, raw=raw, interim=interim, processed=processed,
                           records=records)


def edit(path, change) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    change(data)
    path.write_text(json.dumps(data), encoding="utf-8")


def stage_file(disk, stage: str, ticker: str = "AAA", year: int = 2023):
    return getattr(disk, stage) / ticker / f"{_name(ticker, year)}.json"


# --- coverage ---------------------------------------------------------------


def test_a_rectangular_corpus_passes_coverage(disk):
    check = check_coverage(disk.records, TICKERS)
    assert check.passed and check.failures == []
    assert check.detail == "4 filings, 2 companies x 2 fiscal years, expected 4"


def test_a_company_missing_from_one_year_is_named(disk):
    """Four of five years still reads as complete at the file count."""
    check = check_coverage(disk.records[:-1], TICKERS)
    assert not check.passed and check.failures == ["missing: BBB FY2024"]


def test_a_filing_counted_twice_or_outside_the_scope_is_named(disk):
    amended = replace(disk.records[0], accession_no="AAA-2023-amended")
    early = replace(disk.records[0], accession_no="AAA-2019", period_of_report="2019-12-31")
    undated = replace(disk.records[0], accession_no="AAA-undated", period_of_report="")

    check = check_coverage([*disk.records, amended, early, undated], TICKERS)

    assert check.failures == [
        "duplicated: AAA FY2023 x2",
        "outside FY2023-2024: AAA FY2019",
        "outside FY2023-2024: AAA FYNone",
    ]


# --- stage parity -----------------------------------------------------------


def test_a_filing_present_at_every_stage_passes_parity(disk):
    check = check_stage_parity(disk.records)
    assert check.passed
    assert check.detail == "4 filings present at raw, interim and processed"


def test_a_filing_that_did_not_reach_a_stage_is_named_with_the_stage(disk):
    stage_file(disk, "processed", "BBB", 2024).unlink()
    stage_file(disk, "interim", "AAA", 2024).unlink()
    (disk.raw / "AAA" / f"{_name('AAA', 2023)}.html").unlink()

    assert check_stage_parity(disk.records).failures == [
        "AAA 2024-02-01: no raw file",
        "AAA 2025-02-01: no interim file",
        "BBB 2025-02-01: no processed file",
    ]


def test_a_file_no_manifest_line_accounts_for_is_reported_at_each_stage(disk):
    for folder, name in ((disk.raw, "stray.html"), (disk.interim, "stray.json"),
                         (disk.processed, "stray.json")):
        (folder / "AAA" / name).write_text("{}", encoding="utf-8")

    assert check_stage_parity(disk.records).failures == [
        "interim: data/interim/AAA/stray.json is not in the manifest",
        "processed: data/processed/AAA/stray.json is not in the manifest",
        "raw: data/raw/AAA/stray.html is not in the manifest",
    ]


# --- key Items --------------------------------------------------------------


def test_filings_holding_every_targeted_item_pass(disk):
    check = check_key_items(disk.records)
    assert check.passed
    assert check.detail == "Items 1, 1A, 7, 7A, 8 present in all 4 filings"


def test_an_absent_item_is_named_with_its_filing(disk):
    edit(stage_file(disk, "interim"), lambda data: data["sections"].pop(3))
    assert check_key_items(disk.records).failures == [
        "AAA 2024-02-01: Item 7A absent or an unresolved stub"]


def test_a_cross_reference_counts_only_when_it_says_where_the_text_is(disk):
    """Oracle's Item 8 points at Item 15, which holds the statements. A stub that
    points nowhere is an empty Item."""
    def stub(resolved_from):
        return lambda data: data["sections"][4].update(is_stub=True, resolved_from=resolved_from)

    edit(stage_file(disk, "interim"), stub("part_iv_item_15"))
    assert check_key_items(disk.records).passed

    edit(stage_file(disk, "interim"), stub(None))
    assert check_key_items(disk.records).failures == [
        "AAA 2024-02-01: Item 8 absent or an unresolved stub"]


# --- chunk integrity --------------------------------------------------------


def test_sound_passages_pass_and_every_table_row_is_traced_to_its_source(disk):
    check = check_chunk_integrity(disk.records)
    passages = len(list(iter_chunks(processed_dir=disk.processed)))
    assert check.passed
    # Three statements of two rows in each of four filings.
    assert check.detail == (f"{passages:,} passages, {passages:,} unique ids, "
                            "24 table rows traced to source")


def first_chunk(kind: str = "prose"):
    def pick(data):
        return next(chunk for chunk in data["chunks"] if chunk["content_type"] == kind)
    return pick


@pytest.mark.parametrize("fault, said", [
    (lambda data: data["chunks"].append(dict(data["chunks"][0])),
     "duplicate chunk_id AAA-2023_part_x_item_1_000 (also in AAA 2024-02-01)"),
    (lambda data: first_chunk()(data).update(title=""),
     "AAA-2023_part_x_item_1_000: empty title, so it cannot be cited"),
    (lambda data: first_chunk()(data).update(item=None),
     "AAA-2023_part_x_item_1_000: empty item, so it cannot be cited"),
    (lambda data: first_chunk()(data).update(n_chars=1),
     "AAA-2023_part_x_item_1_000: n_chars disagrees with the text it holds"),
    (lambda data: first_chunk()(data).update(section_id="part_x_item_99"),
     "AAA-2023_part_x_item_1_000: section part_x_item_99 is not in the interim file"),
    (lambda data: first_chunk("table")(data).update(table_index=7),
     "AAA-2023_part_x_item_8_t000_00: table_index 7 is out of range for its Item"),
    (lambda data: first_chunk("table")(data).update(table_index=None),
     "AAA-2023_part_x_item_8_t000_00: table_index None is out of range for its Item"),
])
def test_each_way_a_passage_is_unusable_is_reported_against_its_id(disk, fault, said):
    edit(stage_file(disk, "processed"), fault)
    check = check_chunk_integrity(disk.records)
    assert not check.passed and said in check.failures


def test_a_table_row_that_is_in_no_source_table_is_a_figure_from_nowhere(disk):
    def invent(data):
        chunk = first_chunk("table")(data)
        chunk["text"] = chunk["text"].replace("9,023", "9,999")
        chunk["n_chars"] = len(chunk["text"])

    edit(stage_file(disk, "processed"), invent)
    assert check_chunk_integrity(disk.records).failures == [
        "AAA-2023_part_x_item_8_t000_00: a table row traces to no row of its source table"]


@pytest.mark.parametrize("cells, found", [
    (["Total assets", "9,023"], True),
    (["Total assets", "8,900"], True),          # a subset of the columns, in their order
    (["9,023", "Total assets"], False),         # the same cells the wrong way round
    (["Total assets", "1"], False),
    (["Other", "9,023"], False),                # cells of two different rows
    ([], True),
])
def test_a_row_is_traced_by_its_cells_in_order_since_a_wide_table_is_split_by_column(
        cells, found):
    source = [["Total assets", "9,023", "8,900"], ["Other", "1,000", "900"]]
    assert _row_in(cells, source) is found


# --- no prose lost ----------------------------------------------------------


def test_every_paragraph_of_a_chunked_item_is_in_some_passage(disk):
    check = check_no_prose_lost(disk.records)
    assert check.passed
    # Three paragraphs in each of five Items of four filings.
    assert check.detail == ("60 paragraphs of 200 characters or more all accounted for, "
                            "0 of them as flattened copies of a rebuilt table")


def test_a_paragraph_that_reaches_no_passage_is_reported_with_how_it_opens(disk):
    def lose(data):
        chunk = next(chunk for chunk in data["chunks"] if chunk["section_id"] == "part_x_item_7")
        chunk["text"] = chunk["text"].replace("paragraph 1 for fiscal", "paragraph one for fiscal")

    edit(stage_file(disk, "processed"), lose)
    (failure,) = check_no_prose_lost(disk.records).failures
    assert failure.startswith("AAA 2024-02-01 part_x_item_7: lost 'AAA discussion paragraph 1")


def test_a_short_block_is_not_held_to_the_check(disk):
    """Under 200 characters it is a heading or a caption, which the chunker may fold or drop."""
    edit(stage_file(disk, "interim"),
         lambda data: data["sections"][0].update(text=data["sections"][0]["text"] + "\n\n47"))
    assert check_no_prose_lost(disk.records).passed


# --- against EDGAR ----------------------------------------------------------


def serve(monkeypatch, filings_for):
    """Replace EDGAR's company lookup with one that answers from ``filings_for``."""
    class Listed(list):
        @property
        def empty(self):
            return not self

    class Company:
        def __init__(self, ticker):
            self.ticker = ticker

        def get_filings(self, **search):
            assert search == {"form": ["10-K"], "year": [2023, 2024, 2025]}
            answer = filings_for(self.ticker)
            return None if answer is None else Listed(answer)

    monkeypatch.setattr("edgar.Company", Company)


def on_edgar(record: FilingRecord, **changes):
    fields = {"accession_no": record.accession_no, "cik": record.cik, "form": record.form,
              "filing_date": record.filing_date, "period_of_report": record.period_of_report}
    return SimpleNamespace(**(fields | changes))


def test_a_corpus_that_agrees_with_edgar_passes(disk, monkeypatch):
    serve(monkeypatch, lambda ticker: [on_edgar(r) for r in disk.records if r.ticker == ticker])

    check = check_against_edgar(disk.records, TICKERS)

    assert check.passed
    assert check.detail == ("4 filings agree with EDGAR on cik, form, filing date and "
                            "period of report")


def test_a_filing_that_disagrees_with_edgar_names_the_field_and_both_values(disk, monkeypatch):
    serve(monkeypatch, lambda ticker: [
        on_edgar(r, period_of_report="2023-09-30") if r.accession_no == "AAA-2023" else on_edgar(r)
        for r in disk.records if r.ticker == ticker])

    assert check_against_edgar(disk.records, TICKERS).failures == [
        "AAA AAA-2023: period_of_report is '2023-12-31', EDGAR says '2023-09-30'"]


def test_a_filing_edgar_no_longer_lists_and_one_it_lists_that_we_lack(disk, monkeypatch):
    extra = SimpleNamespace(accession_no="AAA-2024-new", cik=1, form="10-K",
                            filing_date="2025-03-01", period_of_report="2024-12-31")
    old = SimpleNamespace(accession_no="AAA-2019", cik=1, form="10-K",
                          filing_date="2020-02-01", period_of_report="2019-12-31")
    serve(monkeypatch, lambda ticker: (
        [on_edgar(disk.records[1]), extra, old] if ticker == "AAA"
        else [on_edgar(r) for r in disk.records if r.ticker == ticker]))

    assert check_against_edgar(disk.records, TICKERS).failures == [
        "AAA AAA-2023: in our manifest but not on EDGAR",
        # In scope and never downloaded, which no local check could see. The 2019
        # filing is outside the scope, so it is not ours to hold.
        "AAA AAA-2024-new: EDGAR has a FY2024 10-K that is in scope and not downloaded",
    ]


def test_a_lookup_that_fails_is_a_failed_check_and_not_a_crash(disk, monkeypatch):
    def answer(ticker):
        if ticker == "AAA":
            raise ConnectionError("no route to EDGAR")
        return None

    serve(monkeypatch, answer)
    assert check_against_edgar(disk.records, TICKERS).failures == [
        "AAA: EDGAR lookup failed, ConnectionError: no route to EDGAR",
        "BBB: EDGAR returned no 10-K filings",
    ]


def facts_service(monkeypatch, reported, fail_for=()):
    """Replace the XBRL fact service: ``reported`` maps a CIK to ``{concept: [entries]}``."""
    asked = []

    def get(url, headers, timeout):
        asked.append((url, headers))
        cik = int(url.rsplit("CIK", 1)[1].split(".")[0])
        if cik in fail_for:
            raise TimeoutError("the fact service did not answer")
        units = {concept: {"units": {"USD": entries}}
                 for concept, entries in reported.get(cik, {}).items()}
        return SimpleNamespace(raise_for_status=lambda: None,
                               json=lambda: {"facts": {"us-gaap": units}})

    monkeypatch.setattr(verify.httpx, "get", get)
    monkeypatch.setattr(verify.time, "sleep", lambda seconds: None)
    return asked


def entry(accession: str, millions: float, end: str = "2024-12-31", form: str = "10-K"):
    return {"accn": accession, "form": form, "end": end, "val": int(millions * 1_000_000)}


def test_a_figure_edgar_was_told_is_found_in_the_passages_of_the_newest_filing(disk, monkeypatch):
    asked = facts_service(monkeypatch, {
        1: {"Assets": [entry("AAA-2024", _assets("AAA", 2024)),
                       entry("AAA-2023", 1234)]},           # an older filing's, not looked for
        2: {"Assets": [entry("BBB-2024", _assets("BBB", 2024))],
            "NetIncomeLoss": [entry("BBB-2024", 0.5)]},     # under a million: not a line item
    })
    corpus = list(iter_chunks(processed_dir=disk.processed))

    check = check_xbrl_figures(disk.records, "Jane Tan jane@example.com", corpus)

    assert check.passed
    assert check.detail == ("2 of 2 reported figures found in passages, "
                            "2 of them in a table passage")
    assert asked == [
        ("https://data.sec.gov/api/xbrl/companyfacts/CIK0000000001.json",
         {"User-Agent": "Jane Tan jane@example.com"}),
        ("https://data.sec.gov/api/xbrl/companyfacts/CIK0000000002.json",
         {"User-Agent": "Jane Tan jane@example.com"}),
    ]


def test_a_reported_figure_in_no_passage_is_one_nothing_could_ever_cite(disk, monkeypatch):
    facts_service(monkeypatch, {1: {"Assets": [entry("AAA-2024", 5555)],
                                    "Liabilities": [entry("AAA-2024", 1024, form="10-Q")]}},
                  fail_for={2})
    corpus = list(iter_chunks(processed_dir=disk.processed))

    check = check_xbrl_figures(disk.records, "id", corpus)

    assert check.failures == [
        "AAA FY2024: Assets = 5,555 million is reported to EDGAR but appears in no passage",
        "BBB: companyfacts fetch failed, TimeoutError: the fact service did not answer",
    ]
    assert check.detail == "0 of 1 reported figures found in passages, 0 of them in a table passage"


# --- the gate as a whole ----------------------------------------------------


@pytest.fixture
def gate(disk, monkeypatch):
    """``run_checks`` over the synthetic corpus, with EDGAR and the tokenizer replaced."""
    edgar = SimpleNamespace(calls=[])

    def against_edgar(records, tickers):
        edgar.calls.append("matches EDGAR")
        return Check("matches EDGAR", True, "agreed")

    def xbrl(records, identity, corpus):
        edgar.calls.append(("XBRL figures findable", identity, len(corpus)))
        return Check("XBRL figures findable", True, "found")

    monkeypatch.setattr(verify, "read_tickers", lambda: list(TICKERS))
    monkeypatch.setattr(verify, "load_manifest", lambda: list(disk.records))
    monkeypatch.setattr(verify, "configure_edgar", lambda: "Jane Tan jane@example.com")
    monkeypatch.setattr(verify, "check_against_edgar", against_edgar)
    monkeypatch.setattr(verify, "check_xbrl_figures", xbrl)
    monkeypatch.setattr(verify, "bge_token_counter", lambda: lambda texts: [10] * len(texts))
    return SimpleNamespace(disk=disk, edgar=edgar)


NAMES = ["coverage", "stage parity", "key Items", "chunk integrity", "no prose lost",
         "no figure lost", "passage sizes", "statement titles", "matches EDGAR",
         "XBRL figures findable"]


def test_a_sound_corpus_passes_all_ten_checks_cheapest_first(gate, capsys):
    checks = run_checks()

    assert [check.name for check in checks] == NAMES
    assert all(check.passed and not check.skipped for check in checks)
    passages = len(list(iter_chunks(processed_dir=gate.disk.processed)))
    assert gate.edgar.calls == ["matches EDGAR",
                                ("XBRL figures findable", "Jane Tan jane@example.com", passages)]
    assert report(checks) is True
    printed = capsys.readouterr().out
    assert printed.count("[PASS]") == 10
    assert "All 10 checks passed. The corpus is ready to index." in printed


def test_an_incomplete_corpus_skips_the_checks_that_would_only_repeat_the_fault(gate, capsys):
    """Listed as skipped, so three results cannot be read as a clean bill of health on ten."""
    stage_file(gate.disk, "processed", "BBB", 2024).unlink()

    checks = run_checks()

    assert [check.name for check in checks] == NAMES
    outcome = {check.name: "skipped" if check.skipped else check.passed for check in checks}
    assert outcome == {
        "coverage": True, "stage parity": False, "key Items": "skipped",
        "chunk integrity": "skipped", "no prose lost": "skipped", "no figure lost": "skipped",
        "passage sizes": "skipped", "statement titles": "skipped", "matches EDGAR": True,
        "XBRL figures findable": "skipped",
    }
    assert checks[2].detail == "not run: stage parity failed first"
    # EDGAR is still asked: that check reads the manifest, which is whole.
    assert gate.edgar.calls == ["matches EDGAR"]

    assert report(checks) is False
    printed = capsys.readouterr().out
    assert printed.count("[SKIP]") == 7 and printed.count("[FAIL]") == 1
    assert "1 of 10 checks failed: stage parity" in printed
    assert "7 were not run: key Items, chunk integrity, no prose lost" in printed
    assert "The corpus is not ready to index." in printed


def test_a_fault_in_one_passage_fails_the_gate_without_skipping_the_rest(gate):
    edit(stage_file(gate.disk, "processed"), lambda data: first_chunk()(data).update(title=""))

    checks = run_checks()

    failed = [check.name for check in checks if not check.passed]
    assert failed == ["chunk integrity"]
    assert not any(check.skipped for check in checks)
    assert report(checks) is False


def test_an_empty_manifest_is_one_failed_check_that_says_what_to_run(gate, monkeypatch, capsys):
    monkeypatch.setattr(verify, "load_manifest", lambda: [])

    (check,) = run_checks()

    assert (check.name, check.passed) == ("corpus present", False)
    assert check.failures == ["run python -m src.pipeline download first"]
    assert gate.edgar.calls == []
    assert report([check]) is False
    assert "1 of 1 checks failed: corpus present" in capsys.readouterr().out


def test_a_check_that_was_not_run_is_not_a_check_that_passed(capsys):
    """With nothing failed and one check skipped, the corpus is still not ready: a gate
    that ran nine checks of ten has not said the tenth would pass."""
    checks = [Check("coverage", True, "complete"),
              Check("XBRL figures findable", False, "not run", skipped=True)]

    assert report(checks) is False

    printed = capsys.readouterr().out
    assert "checks failed" not in printed
    assert "1 were not run: XBRL figures findable" in printed
    assert "The corpus is not ready to index." in printed
    assert "ready to index.\n" not in printed.replace("not ready to index.\n", "")


def test_a_long_list_of_failures_is_cut_to_ten_and_counted(capsys):
    failures = [f"missing: AAA FY{2000 + number}" for number in range(14)]
    passed = report([Check("coverage", False, "14 gaps", failures), Check("key Items", True, "ok")])

    printed = capsys.readouterr().out
    assert passed is False
    assert printed.count("missing: AAA") == 10
    assert "... and 4 more" in printed
    assert "1 of 2 checks failed: coverage" in printed
    assert "were not run" not in printed
