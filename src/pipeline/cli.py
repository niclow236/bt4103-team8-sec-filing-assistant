"""The command line for the data pipeline: every command, and what each one runs.

This is the only module in the package that knows argparse exists. The stage
modules hold the work and know nothing about how they were invoked, which is what
lets the same functions be called from a notebook, from the retrieval stage, or
from a test without a fake argument namespace.

    python -m src.pipeline                    # list the commands
    python -m src.pipeline parse --help       # options for one of them
    python -m src.pipeline rebuild            # all of them, then the gate

Each command is declared once, in a ``_add_*`` function, and the help text
argparse shows comes from the stage module's own docstring. There is no second
table of descriptions to keep in step, because the previous arrangement had one
and all six entries had already drifted from the modules they described.

``rebuild`` lives here rather than in a module of its own because it is nothing
but these commands run in order: composition of the command line, not a stage of
the pipeline.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import time
from pathlib import Path

from ..config import (
    INTERIM_DIR,
    MANIFEST_FILE,
    PROCESSED_DIR,
    PROJECT_ROOT,
    RAW_DIR,
    MissingIdentityError,
    configure_edgar,
    ensure_data_dirs,
    read_tickers,
)
from ..utils import start_run_log
from . import chunk as chunk_stage
from . import download as download_stage
from . import parse as parse_stage
from . import passages as passages_stage
from . import verify as verify_stage
from .constants import (
    CHUNK_CHAR_BUDGET,
    CHUNK_CHAR_OVERLAP,
    DEFAULT_FILING_YEARS,
    DEFAULT_FISCAL_YEARS,
    DEFAULT_FORMS,
)

logger = logging.getLogger(__name__)


def _summary(module) -> str:
    """A command's one-line description, taken from the module that runs it.

    The docstrings are written for a reader of the source, so they mark paths up
    in reStructuredText. On a terminal ``data/raw/`` is just noise, so the
    backticks come off on the way through.
    """
    first = (module.__doc__ or "").strip().splitlines()[0]
    return first.replace("``", "")


# --- arguments --------------------------------------------------------------


def _add_download(subparsers) -> None:
    parser = subparsers.add_parser(
        "download", help=_summary(download_stage), description=_summary(download_stage),
    )
    parser.set_defaults(run=run_download)
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


def _add_parse(subparsers) -> None:
    parser = subparsers.add_parser(
        "parse", help=_summary(parse_stage), description=_summary(parse_stage),
    )
    parser.set_defaults(run=run_parse)
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


def _add_chunk(subparsers) -> None:
    parser = subparsers.add_parser(
        "chunk", help=_summary(chunk_stage), description=_summary(chunk_stage),
    )
    parser.set_defaults(run=run_chunk)
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


def _add_passages(subparsers) -> None:
    parser = subparsers.add_parser(
        "passages", help=_summary(passages_stage), description=_summary(passages_stage),
    )
    parser.set_defaults(run=run_passages)
    parser.add_argument("--tickers", nargs="+", help="Only passages from these companies.")
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


def _add_verify(subparsers) -> None:
    parser = subparsers.add_parser(
        "verify", help=_summary(verify_stage), description=_summary(verify_stage),
    )
    parser.set_defaults(run=run_verify)
    # No options on purpose. A gate you can narrow is one that gets narrowed
    # until it passes.


def _add_rebuild(subparsers) -> None:
    parser = subparsers.add_parser(
        "rebuild",
        help="Run download, parse, chunk and verify in one go",
        description="Run download, parse, chunk and verify in one go",
    )
    parser.set_defaults(run=run_rebuild)
    parser.add_argument(
        "--clean", action="store_true",
        help="Delete data/raw, data/interim and data/processed first, so the "
             "rebuild starts from nothing. Without it the stages resume, which "
             "is faster but does not prove an empty folder can be filled. "
             "data/sample and data/diagnostics are left alone.",
    )


# --- what each command runs -------------------------------------------------


def run_download(args) -> None:
    try:
        identity = configure_edgar()
    except MissingIdentityError as error:
        # An unset contact string is a setup step someone has not done yet, not
        # a fault in the code, so print what to do about it and stop. A stack
        # trace here would bury the one line that actually helps.
        raise SystemExit(f"\n{error}\n") from None
    logger.info("Identifying to SEC EDGAR as: %s", identity)

    tickers = args.tickers or read_tickers()
    # "--years 0 0" is the escape hatch for an unfiltered search; anything else,
    # including the default, narrows the request to that range. Likewise
    # "--fiscal-years 0 0" keeps every year a filing reports on.
    years = range(args.years[0], args.years[1] + 1) if any(args.years) else None
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

    new_records = download_stage.download_all(
        tickers, args.forms, years=years, limit=args.limit,
        fiscal_years=fiscal_years, dry_run=args.dry_run,
    )

    if args.dry_run:
        download_stage.report_plan(new_records, set(read_tickers()), fiscal_years, tickers)
        return

    print(f"\nDownloaded {len(new_records)} new filings into {RAW_DIR}")
    print(f"Manifest: {MANIFEST_FILE}")
    download_stage.report_coverage(set(read_tickers()), scope=fiscal_years)


def run_parse(args) -> None:
    # edgartools narrates its section detection at INFO: which candidates it
    # considered, which strategies it abandoned, which fallback it settled on.
    # Lines like "Could not find actual section for mda" are a note that its
    # first strategy missed and its second one worked, not a failure, but a
    # screen of them reads like one. Keep its warnings, drop the commentary,
    # and leave --verbose for when a parse actually needs diagnosing.
    if not args.verbose:
        logging.getLogger("edgar").setLevel(logging.WARNING)

    records = download_stage.load_manifest()
    if not records:
        print("The manifest is empty. Run python -m src.pipeline download first.")
        return

    if args.tickers:
        wanted = {ticker.upper() for ticker in args.tickers}
        records = [record for record in records if record.ticker in wanted]
    if args.forms:
        wanted_forms = set(args.forms)
        records = [record for record in records if record.form in wanted_forms]
    if not records:
        print(
            "No downloaded filing matches those filters. "
            "Run python -m src.pipeline download for them first."
        )
        return

    logger.info("Parsing %d filings from the manifest", len(records))
    # Always collected, so the summary can say why rebuilds failed; only written
    # to disk when asked, since the fragments are bulky.
    failures: list[parse_stage.TableFailure] = []
    parsed = parse_stage.parse_all(records, force=args.force, failures=failures)
    written_to = parse_stage.write_table_failures(failures) if args.table_debug else None
    parse_stage.report(parsed, failures, written_to)


def run_chunk(args) -> None:
    paths = chunk_stage.interim_files()
    if not paths:
        print("data/interim/ is empty. Run python -m src.pipeline parse first.")
        return

    if args.tickers:
        wanted = {ticker.upper() for ticker in args.tickers}
        paths = [path for path in paths if path.parent.name in wanted]
        if not paths:
            # Without this the run would fall through to "use --force", which
            # points at the wrong problem: nothing was skipped, nothing is there.
            print(
                f"Nothing parsed yet for {', '.join(sorted(wanted))}. "
                "Run python -m src.pipeline parse for those tickers first."
            )
            return

    logger.info("Chunking %d filings from %s", len(paths), INTERIM_DIR)
    chunk_stage.report(chunk_stage.chunk_all(
        paths,
        force=args.force,
        forms=args.forms,
        key_items_only=args.key_items_only,
        budget=args.budget,
        overlap=args.overlap,
    ))


def run_passages(args) -> None:
    if not any(PROCESSED_DIR.glob("*/*.json")):
        print("data/processed/ is empty. Run python -m src.pipeline chunk first.")
        return

    fiscal_years = (
        range(args.fiscal_years[0], args.fiscal_years[1] + 1)
        if args.fiscal_years else None
    )
    matched = passages_stage.select(
        tickers=args.tickers,
        items=args.item,
        forms=args.forms,
        fiscal_years=fiscal_years,
        content_type=args.content_type,
        contains=args.contains,
        key_items_only=args.key_items_only,
    )
    if not matched:
        print("No passage matches those filters.")
        return

    passages_stage.report(matched, limit=args.limit, full=args.full, as_json=args.json)


def run_verify(args) -> None:
    logging.getLogger("edgar").setLevel(logging.WARNING)
    if not verify_stage.report(verify_stage.run_checks()):
        raise SystemExit(1)


# --- rebuild: the commands above, in order ----------------------------------


def clean_data(directories: tuple[Path, ...] = (RAW_DIR, INTERIM_DIR, PROCESSED_DIR)) -> list[Path]:
    """Delete the built corpus, leaving anything else under data/ alone.

    Only the three folders the pipeline writes are removed. data/sample is
    committed and data/diagnostics holds what --table-debug wrote, and neither is
    this command's to throw away.
    """
    removed = []
    for directory in directories:
        if directory.exists():
            shutil.rmtree(directory)
            removed.append(directory)
    ensure_data_dirs()
    return removed


def _elapsed(seconds: float) -> str:
    return f"{seconds:.0f}s" if seconds < 90 else f"{seconds / 60:.1f} min"


def _step(name: str, run, args) -> float:
    """Run one command as part of a rebuild, timing it, and stop on failure."""
    print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")
    started = time.monotonic()
    try:
        run(args)
    except SystemExit as stopped:
        # A command that stops on purpose says why in its exit code, which for
        # download with no EDGAR identity is the whole explanation rather than a
        # number. Re-raising with only our own message would throw away the one
        # line that tells the reader what to do.
        if stopped.code not in (0, None):
            reason = stopped.code if isinstance(stopped.code, str) else ""
            raise SystemExit(f"{reason}\n{name} stopped, so the rebuild cannot continue.") from None
    elapsed = time.monotonic() - started
    print(f"\n{name} finished in {_elapsed(elapsed)}")
    return elapsed


def run_rebuild(args) -> None:
    if args.clean:
        removed = clean_data()
        for directory in removed:
            print(f"Removed {directory.relative_to(PROJECT_ROOT).as_posix()}")
        if not removed:
            print("Nothing to clean: data/raw, data/interim and data/processed are already absent")

    # --force is redundant after --clean, since there is nothing left to skip.
    force = not args.clean
    timings = {
        "DOWNLOAD": _step("DOWNLOAD", run_download, _defaults_for("download")),
        "PARSE": _step("PARSE", run_parse, _defaults_for("parse", force=force)),
        "CHUNK": _step("CHUNK", run_chunk, _defaults_for("chunk", force=force)),
    }

    print(f"\n{'=' * 78}\nVERIFY\n{'=' * 78}")
    started = time.monotonic()
    passed = verify_stage.report(verify_stage.run_checks())
    timings["VERIFY"] = time.monotonic() - started

    print(f"\n{'=' * 78}\nRebuild summary\n{'=' * 78}")
    for name, elapsed in timings.items():
        print(f"  {name.lower():<10} {_elapsed(elapsed)}")
    print(f"  {'total':<10} {_elapsed(sum(timings.values()))}")

    if not passed:
        raise SystemExit(1)


def _defaults_for(command: str, **overrides) -> argparse.Namespace:
    """The argument namespace a command would get with no flags at all.

    A rebuild runs each command at its defaults, and asking argparse for them
    rather than writing them out again keeps the two from drifting: a new option
    with a default is picked up here without anybody remembering to add it.
    """
    args = build_parser().parse_args([command])
    for name, value in overrides.items():
        setattr(args, name, value)
    return args


# --- dispatch ---------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Every command the pipeline has, and every option each one takes."""
    parser = argparse.ArgumentParser(
        prog="python -m src.pipeline",
        description="Build and check the SEC filing corpus.",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")
    for add in (_add_download, _add_parse, _add_chunk,
                _add_passages, _add_verify, _add_rebuild):
        add(subparsers)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return

    # Before basicConfig, so the log file captures log lines and not only prints.
    log_path = start_run_log(args.command)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    try:
        args.run(args)
    finally:
        print(f"Run log: {log_path}")


if __name__ == "__main__":
    main()
