"""The pipeline's command line (#49): what each command passes on, and what it says.

``cli.py`` is the only module in the package that knows argparse exists. The
stages are functions with named arguments, so what can go wrong here is the
translation: a flag that never reaches the stage, a default that drifted from
the constant it restates, a filter that matches nothing and reads as success,
or a rebuild that carries on past a stage that stopped. The stages are replaced
with recorders, so nothing touches EDGAR or ``data/``.
"""

from __future__ import annotations

import argparse
import logging
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.config import MissingIdentityError
from src.pipeline import cli
from src.pipeline.constants import (
    CHUNK_CHAR_BUDGET,
    CHUNK_CHAR_OVERLAP,
    DEFAULT_FILING_YEARS,
    DEFAULT_FISCAL_YEARS,
    DEFAULT_FORMS,
)
from src.pipeline.records import FilingRecord


def parse(*arguments: str) -> argparse.Namespace:
    return cli.build_parser().parse_args(list(arguments))


def record(ticker: str = "AAA", form: str = "10-K") -> FilingRecord:
    return FilingRecord(
        ticker=ticker, cik=1, company=f"{ticker} Corp", form=form, filing_date="2025-02-01",
        accession_no=f"acc-{ticker}-{form}", url="https://example.test",
        path=f"data/raw/{ticker}/filing.html", period_of_report="2024-12-31",
    )


class Recorder:
    """Stands in for a stage function: remembers how it was called."""

    def __init__(self, returns=None):
        self.calls: list[tuple[tuple, dict]] = []
        self.returns = returns

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.returns

    @property
    def kwargs(self) -> dict:
        return self.calls[-1][1]

    @property
    def args(self) -> tuple:
        return self.calls[-1][0]


# --- the commands there are -------------------------------------------------


def test_every_stage_is_a_command_and_rebuild_runs_them_all():
    commands = cli.build_parser()._subparsers._group_actions[0].choices
    assert list(commands) == ["download", "parse", "chunk", "passages", "verify", "rebuild"]
    assert {name: commands[name].get_default("run") for name in commands} == {
        "download": cli.run_download, "parse": cli.run_parse, "chunk": cli.run_chunk,
        "passages": cli.run_passages, "verify": cli.run_verify, "rebuild": cli.run_rebuild,
    }


@pytest.mark.parametrize("name, stage", [
    ("download", cli.download_stage), ("parse", cli.parse_stage), ("chunk", cli.chunk_stage),
    ("passages", cli.passages_stage), ("verify", cli.verify_stage),
])
def test_a_commands_help_is_the_first_line_of_the_module_that_runs_it(name, stage):
    """One description, so the command line cannot drift from the code it describes."""
    command = cli.build_parser()._subparsers._group_actions[0].choices[name]
    first_line = stage.__doc__.strip().splitlines()[0]
    assert command.description == first_line.replace("``", "")
    assert "``" not in command.description


def test_the_defaults_are_the_constants_and_not_copies_of_them():
    download = parse("download")
    assert download.forms == list(DEFAULT_FORMS)
    assert download.fiscal_years == list(DEFAULT_FISCAL_YEARS)
    assert download.years == list(DEFAULT_FILING_YEARS)
    assert (download.limit, download.dry_run, download.tickers) == (None, False, None)
    chunk = parse("chunk")
    assert (chunk.budget, chunk.overlap) == (CHUNK_CHAR_BUDGET, CHUNK_CHAR_OVERLAP)
    assert (chunk.force, chunk.key_items_only, chunk.forms) == (False, False, None)
    assert parse("passages").limit == 10


def test_the_chunk_sizes_are_described_in_characters_not_tokens():
    """Both are measured in characters, and the help has to say which (#10)."""
    chunk = cli.build_parser()._subparsers._group_actions[0].choices["chunk"]
    options = {action.dest: action for action in chunk._actions}
    assert options["budget"].metavar == options["overlap"].metavar == "CHARS"
    assert f"{CHUNK_CHAR_BUDGET} characters" in options["budget"].help
    assert "not tokens" in options["budget"].help and "not tokens" in options["overlap"].help


def test_verify_takes_no_options_so_the_gate_cannot_be_narrowed():
    with pytest.raises(SystemExit):
        parse("verify", "--tickers", "AAA")
    verify = cli.build_parser()._subparsers._group_actions[0].choices["verify"]
    assert [action.dest for action in verify._actions] == ["help"]


@pytest.mark.parametrize("module", ["src.pipeline", "src.retrieval"])
def test_a_stage_run_with_no_command_lists_its_commands(module, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", [module])
    runpy.run_module(module, run_name="__main__")
    printed = capsys.readouterr().out
    assert "<command>" in printed and "usage: python -m " + module in printed


def test_a_command_keeps_a_log_of_what_it_printed(monkeypatch, capsys, tmp_path):
    started = []
    monkeypatch.setattr(cli, "start_run_log",
                        lambda name: started.append(name) or tmp_path / f"{name}.log")
    monkeypatch.setattr(cli.chunk_stage, "interim_files", lambda: [])

    cli.main(["chunk"])
    cli.main([])

    printed = capsys.readouterr().out
    assert started == ["chunk"]          # no command, no log
    assert "data/interim/ is empty" in printed
    assert f"Run log: {tmp_path / 'chunk.log'}" in printed


def test_the_log_path_is_printed_even_when_the_command_stops(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(cli, "start_run_log", lambda name: tmp_path / "verify.log")
    monkeypatch.setattr(cli.verify_stage, "run_checks", lambda: [])
    monkeypatch.setattr(cli.verify_stage, "report", lambda checks: False)

    with pytest.raises(SystemExit) as stopped:
        cli.main(["verify"])
    assert stopped.value.code == 1
    assert "Run log:" in capsys.readouterr().out


# --- download ---------------------------------------------------------------


@pytest.fixture
def download(monkeypatch):
    """The download command with EDGAR, the ticker list and the reports replaced."""
    stage = SimpleNamespace(
        download_all=Recorder(returns=[record()]), report_plan=Recorder(),
        report_coverage=Recorder(),
    )
    for name, recorder in vars(stage).items():
        monkeypatch.setattr(cli.download_stage, name, recorder)
    monkeypatch.setattr(cli, "configure_edgar", lambda: "Jane Tan jane@example.com")
    monkeypatch.setattr(cli, "read_tickers", lambda: ["AAA", "BBB"])
    return stage


def test_download_fetches_the_configured_companies_over_the_default_scope(download, capsys):
    cli.run_download(parse("download"))

    assert download.download_all.args == (["AAA", "BBB"], list(DEFAULT_FORMS))
    assert download.download_all.kwargs == {
        "years": range(DEFAULT_FILING_YEARS[0], DEFAULT_FILING_YEARS[1] + 1),
        "limit": None,
        "fiscal_years": range(DEFAULT_FISCAL_YEARS[0], DEFAULT_FISCAL_YEARS[1] + 1),
        "dry_run": False,
    }
    assert "Downloaded 1 new filings" in capsys.readouterr().out
    # The coverage table is over every company in config, whatever was asked for.
    assert download.report_coverage.args == ({"AAA", "BBB"},)
    assert download.report_plan.calls == []


def test_download_passes_its_narrowing_through(download):
    cli.run_download(parse("download", "--tickers", "MSFT", "--forms", "10-Q", "--limit", "2",
                           "--fiscal-years", "2019", "2020", "--years", "2019", "2021"))

    assert download.download_all.args == (["MSFT"], ["10-Q"])
    assert download.download_all.kwargs["years"] == range(2019, 2022)
    assert download.download_all.kwargs["fiscal_years"] == range(2019, 2021)
    assert download.download_all.kwargs["limit"] == 2


def test_zero_zero_searches_and_keeps_every_year(download):
    cli.run_download(parse("download", "--years", "0", "0", "--fiscal-years", "0", "0"))
    assert download.download_all.kwargs["years"] is None
    assert download.download_all.kwargs["fiscal_years"] is None


def test_a_dry_run_shows_the_plan_and_reports_nothing_as_downloaded(download, capsys):
    cli.run_download(parse("download", "--dry-run", "--tickers", "AAA"))

    assert download.download_all.kwargs["dry_run"] is True
    planned, expected, scope, tickers = download.report_plan.args
    assert planned == [record()] and expected == {"AAA", "BBB"} and tickers == ["AAA"]
    assert scope == range(DEFAULT_FISCAL_YEARS[0], DEFAULT_FISCAL_YEARS[1] + 1)
    assert download.report_coverage.calls == []
    assert "Downloaded" not in capsys.readouterr().out


def test_download_without_an_identity_stops_with_what_to_do_and_no_traceback(
        download, monkeypatch):
    def refuse():
        raise MissingIdentityError("EDGAR_IDENTITY is not set. Copy .env.example to .env")

    monkeypatch.setattr(cli, "configure_edgar", refuse)
    with pytest.raises(SystemExit) as stopped:
        cli.run_download(parse("download"))
    assert "EDGAR_IDENTITY is not set" in str(stopped.value.code)
    assert stopped.value.__suppress_context__
    assert download.download_all.calls == []


# --- parse ------------------------------------------------------------------


@pytest.fixture
def parsing(monkeypatch):
    stage = SimpleNamespace(
        parse_all=Recorder(returns=["parsed"]), report=Recorder(),
        write_table_failures=Recorder(returns=Path("data/diagnostics/table_failures")),
    )
    for name, recorder in vars(stage).items():
        monkeypatch.setattr(cli.parse_stage, name, recorder)
    manifest = [record("AAA"), record("BBB"), record("AAA", form="10-Q")]
    monkeypatch.setattr(cli.download_stage, "load_manifest", lambda: manifest)
    stage.manifest = manifest
    return stage


def test_parse_takes_every_filing_in_the_manifest_by_default(parsing):
    cli.run_parse(parse("parse"))

    (records,), settings = parsing.parse_all.calls[0]
    assert records == parsing.manifest
    assert settings["force"] is False and settings["failures"] == []
    # Nothing is written to data/diagnostics unless asked.
    assert parsing.write_table_failures.calls == []
    assert parsing.report.args == (["parsed"], [], None)


def test_parse_narrows_by_company_and_form(parsing):
    cli.run_parse(parse("parse", "--tickers", "aaa", "--forms", "10-K", "--force"))

    (records,), settings = parsing.parse_all.calls[0]
    assert records == [record("AAA")] and settings["force"] is True


def test_parse_writes_the_tables_it_could_not_rebuild_when_asked(parsing):
    cli.run_parse(parse("parse", "--table-debug"))

    failures = parsing.parse_all.kwargs["failures"]
    assert parsing.write_table_failures.args == (failures,)
    assert parsing.report.args == (["parsed"], failures, Path("data/diagnostics/table_failures"))


def test_parse_says_what_to_run_when_there_is_nothing_to_parse(parsing, monkeypatch, capsys):
    cli.run_parse(parse("parse", "--tickers", "ZZZ"))
    assert "No downloaded filing matches those filters" in capsys.readouterr().out

    monkeypatch.setattr(cli.download_stage, "load_manifest", lambda: [])
    cli.run_parse(parse("parse"))
    assert "The manifest is empty. Run python -m src.pipeline download" in capsys.readouterr().out
    assert parsing.parse_all.calls == []


def test_the_extractors_commentary_is_quiet_unless_asked_for(parsing):
    """It narrates the strategies it abandons, which reads like errors on a parse that worked."""
    edgar = logging.getLogger("edgar")
    before = edgar.level
    try:
        edgar.setLevel(logging.NOTSET)
        cli.run_parse(parse("parse", "--verbose"))
        assert edgar.level == logging.NOTSET
        cli.run_parse(parse("parse"))
        assert edgar.level == logging.WARNING
    finally:
        edgar.setLevel(before)


# --- chunk ------------------------------------------------------------------


@pytest.fixture
def chunking(monkeypatch, tmp_path):
    paths = [tmp_path / "AAA" / "one.json", tmp_path / "BBB" / "two.json"]
    stage = SimpleNamespace(chunk_all=Recorder(returns=["chunked"]), report=Recorder())
    monkeypatch.setattr(cli.chunk_stage, "interim_files", lambda: list(paths))
    monkeypatch.setattr(cli.chunk_stage, "chunk_all", stage.chunk_all)
    monkeypatch.setattr(cli.chunk_stage, "report", stage.report)
    stage.paths = paths
    return stage


def test_chunk_cuts_every_parsed_filing_at_the_shipped_sizes(chunking):
    cli.run_chunk(parse("chunk"))

    assert chunking.chunk_all.args == (chunking.paths,)
    assert chunking.chunk_all.kwargs == {
        "force": False, "forms": None, "key_items_only": False,
        "budget": CHUNK_CHAR_BUDGET, "overlap": CHUNK_CHAR_OVERLAP,
    }
    assert chunking.report.args == (["chunked"],)


def test_the_budget_and_overlap_typed_at_the_shell_reach_the_chunker(chunking):
    """``--budget`` is what makes a chunk-size sweep a one-command experiment (#45)."""
    cli.run_chunk(parse("chunk", "--budget", "1200", "--overlap", "200", "--force",
                        "--key-items-only", "--forms", "10-K", "--tickers", "bbb"))

    assert chunking.chunk_all.args == ([chunking.paths[1]],)
    assert chunking.chunk_all.kwargs == {
        "force": True, "forms": ["10-K"], "key_items_only": True, "budget": 1200,
        "overlap": 200,
    }


def test_chunk_tells_nothing_parsed_apart_from_nothing_left_to_do(chunking, monkeypatch, capsys):
    """Falling through to "use --force" would point at the wrong problem."""
    cli.run_chunk(parse("chunk", "--tickers", "ZZZ", "YYY"))
    assert "Nothing parsed yet for YYY, ZZZ" in capsys.readouterr().out

    monkeypatch.setattr(cli.chunk_stage, "interim_files", lambda: [])
    cli.run_chunk(parse("chunk"))
    assert "data/interim/ is empty. Run python -m src.pipeline parse" in capsys.readouterr().out
    assert chunking.chunk_all.calls == []


# --- passages and verify ----------------------------------------------------


@pytest.fixture
def reading(monkeypatch, tmp_path):
    stage = SimpleNamespace(select=Recorder(returns=[{"chunk_id": "c1"}]), report=Recorder())
    monkeypatch.setattr(cli.passages_stage, "select", stage.select)
    monkeypatch.setattr(cli.passages_stage, "report", stage.report)
    (tmp_path / "AAA").mkdir()
    (tmp_path / "AAA" / "filing.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(cli, "PROCESSED_DIR", tmp_path)
    return stage


def test_passages_passes_every_filter_and_how_to_show_the_result(reading):
    cli.run_passages(parse(
        "passages", "--tickers", "AAPL", "--item", "8", "1A", "--forms", "10-K",
        "--fiscal-years", "2023", "2024", "--content-type", "table", "--contains", "revenue",
        "--key-items-only", "--limit", "0", "--full", "--json"))

    assert reading.select.kwargs == {
        "tickers": ["AAPL"], "items": ["8", "1A"], "forms": ["10-K"],
        "fiscal_years": range(2023, 2025), "content_type": "table", "contains": "revenue",
        "key_items_only": True,
    }
    assert reading.report.args == ([{"chunk_id": "c1"}],)
    assert reading.report.kwargs == {"limit": 0, "full": True, "as_json": True}


def test_passages_without_filters_selects_everything_and_shows_ten(reading):
    cli.run_passages(parse("passages"))

    assert set(reading.select.kwargs.values()) == {None, False}
    assert reading.report.kwargs == {"limit": 10, "full": False, "as_json": False}


def test_passages_says_when_nothing_matches_or_nothing_is_chunked(
        reading, monkeypatch, capsys, tmp_path):
    reading.select.returns = []
    cli.run_passages(parse("passages", "--contains", "no such phrase"))
    assert "No passage matches those filters." in capsys.readouterr().out
    assert reading.report.calls == []

    monkeypatch.setattr(cli, "PROCESSED_DIR", tmp_path / "empty")
    cli.run_passages(parse("passages"))
    assert "data/processed/ is empty. Run python -m src.pipeline chunk" in capsys.readouterr().out


def test_verify_exits_non_zero_when_the_corpus_does_not_pass(monkeypatch):
    monkeypatch.setattr(cli.verify_stage, "run_checks", lambda: ["checks"])
    reported = Recorder(returns=True)
    monkeypatch.setattr(cli.verify_stage, "report", reported)

    cli.run_verify(parse("verify"))
    assert reported.args == (["checks"],)

    reported.returns = False
    with pytest.raises(SystemExit) as stopped:
        cli.run_verify(parse("verify"))
    assert stopped.value.code == 1


# --- rebuild ----------------------------------------------------------------


@pytest.fixture
def rebuilding(monkeypatch):
    """A rebuild whose stages only record that they ran, and with which arguments."""
    ran: list[tuple[str, argparse.Namespace | None]] = []
    for name in ("download", "parse", "chunk"):
        monkeypatch.setattr(cli, f"run_{name}",
                            lambda args, name=name: ran.append((name, args)))
    monkeypatch.setattr(cli.verify_stage, "run_checks", lambda: ran.append(("verify", None)))
    passed = SimpleNamespace(value=True)
    monkeypatch.setattr(cli.verify_stage, "report", lambda checks: passed.value)
    cleaned = Recorder(returns=[])
    monkeypatch.setattr(cli, "clean_data", cleaned)
    # ``rebuild --clean`` deletes data/raw, data/interim and data/processed, and
    # this suite also runs on machines that hold a corpus. Should the recorder
    # above ever stop standing in for it, the test fails here and deletes nothing.
    monkeypatch.setattr(cli.shutil, "rmtree",
                        lambda path, *args, **kwargs: pytest.fail(f"a test tried to delete {path}"))
    return SimpleNamespace(ran=ran, passed=passed, cleaned=cleaned)


def test_rebuild_runs_the_stages_in_order_and_then_the_gate(rebuilding, capsys):
    cli.run_rebuild(parse("rebuild"))

    assert [name for name, _ in rebuilding.ran] == ["download", "parse", "chunk", "verify"]
    assert rebuilding.cleaned.calls == []
    printed = capsys.readouterr().out
    for heading in ("DOWNLOAD", "PARSE", "CHUNK", "VERIFY", "Rebuild summary"):
        assert heading in printed
    assert "total" in printed


def test_rebuild_runs_each_stage_at_its_own_defaults_and_redoes_what_is_on_disk(rebuilding):
    """A stage skips what it already wrote, and a rebuild that skipped would prove nothing."""
    cli.run_rebuild(parse("rebuild"))

    stages = dict(rebuilding.ran)
    assert stages["download"] == parse("download")
    assert (stages["parse"].force, stages["chunk"].force) == (True, True)
    assert stages["chunk"].budget == CHUNK_CHAR_BUDGET
    assert stages["chunk"].overlap == CHUNK_CHAR_OVERLAP


def test_rebuild_clean_starts_from_nothing_so_force_is_not_needed(rebuilding, capsys):
    rebuilding.cleaned.returns = [cli.RAW_DIR, cli.PROCESSED_DIR]
    cli.run_rebuild(parse("rebuild", "--clean"))

    assert len(rebuilding.cleaned.calls) == 1
    stages = dict(rebuilding.ran)
    assert (stages["parse"].force, stages["chunk"].force) == (False, False)
    printed = capsys.readouterr().out
    assert "Removed data/raw" in printed and "Removed data/processed" in printed

    rebuilding.cleaned.returns = []
    cli.run_rebuild(parse("rebuild", "--clean"))
    assert "Nothing to clean" in capsys.readouterr().out


def test_rebuild_exits_non_zero_when_the_corpus_it_built_does_not_pass(rebuilding, capsys):
    rebuilding.passed.value = False
    with pytest.raises(SystemExit) as stopped:
        cli.run_rebuild(parse("rebuild"))
    assert stopped.value.code == 1
    # The summary is still printed: how long each stage took is worth having either way.
    assert "Rebuild summary" in capsys.readouterr().out


def test_rebuild_stops_at_the_stage_that_stopped_and_keeps_its_reason(rebuilding, monkeypatch):
    def no_identity(args):
        raise SystemExit("\nEDGAR_IDENTITY is not set.\n")

    monkeypatch.setattr(cli, "run_download", no_identity)
    with pytest.raises(SystemExit) as stopped:
        cli.run_rebuild(parse("rebuild"))

    message = str(stopped.value.code)
    assert "EDGAR_IDENTITY is not set." in message
    assert "DOWNLOAD stopped, so the rebuild cannot continue." in message
    assert rebuilding.ran == []          # nothing after it ran


def test_a_stage_that_exits_cleanly_does_not_stop_a_rebuild(capsys):
    def finished_early(args):
        raise SystemExit(0)

    assert cli._step("PARSE", finished_early, None) >= 0
    assert "PARSE finished in" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="CHUNK stopped"):
        cli._step("CHUNK", lambda args: (_ for _ in ()).throw(SystemExit(1)), None)


@pytest.mark.parametrize("seconds, shown", [(0.4, "0s"), (89, "89s"), (90, "1.5 min"),
                                            (1200, "20.0 min")])
def test_a_duration_is_shown_in_seconds_then_minutes(seconds, shown):
    assert cli._elapsed(seconds) == shown


def test_a_stages_defaults_come_from_its_own_parser(monkeypatch):
    """So an option added to a stage reaches a rebuild without being listed twice."""
    assert cli._defaults_for("chunk") == parse("chunk")
    forced = cli._defaults_for("chunk", force=True, budget=1200)
    assert (forced.force, forced.budget, forced.overlap) == (True, 1200, CHUNK_CHAR_OVERLAP)


def test_clean_removes_the_built_corpus_and_only_that(tmp_path, monkeypatch):
    made = Recorder()
    monkeypatch.setattr(cli, "ensure_data_dirs", made)
    raw, interim, processed = tmp_path / "raw", tmp_path / "interim", tmp_path / "processed"
    sample = tmp_path / "sample"
    for directory in (raw, processed, sample):
        (directory / "AAA").mkdir(parents=True)
        (directory / "AAA" / "filing.html").write_text("kept or not", encoding="utf-8")

    removed = cli.clean_data((raw, interim, processed))

    # interim was never there, so it is not reported as removed.
    assert removed == [raw, processed]
    assert not raw.exists() and not processed.exists()
    assert (sample / "AAA" / "filing.html").exists()
    assert len(made.calls) == 1           # the empty folders are put back for the stages
