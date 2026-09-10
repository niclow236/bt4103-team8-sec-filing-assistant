"""The command line for the retrieval stage: every command, and what each runs.

This is the only module in the package that knows argparse exists, the same
arrangement ``src/pipeline/cli.py`` uses one stage earlier. The stage modules
hold the work and know nothing about how they were invoked, which is what lets
the same functions be called from a notebook, from the evaluation harness, or
from a test without a fake argument namespace.

    python -m src.retrieval                   # list the commands
    python -m src.retrieval embed --help      # options for one of them
    python -m src.retrieval embed             # build the dense index
    python -m src.retrieval facts             # build the XBRL facts store

It is a separate command line from the pipeline's rather than more subcommands
on it, because the two stages are separated by what they build from: the
pipeline turns EDGAR into ``data/processed/``, and retrieval turns that into the
indexes a question is answered against.

``embed`` needs no network and no EDGAR identity, since it reads only what the
pipeline already wrote. ``facts`` does need both, because the figures it stores
are published through EDGAR rather than printed in the filing's HTML, so it is
the one command here that goes back to the source.
"""

from __future__ import annotations

import argparse
import logging

from ..pipeline.constants import CHUNK_CHAR_BUDGET, CHUNK_CHAR_OVERLAP
from ..utils import start_run_log
from . import embed as embed_stage
from . import facts as facts_stage
from .constants import EMBED_BATCH_SIZE

logger = logging.getLogger(__name__)


def _summary(module) -> str:
    """A command's one-line description, taken from the module that runs it.

    The docstrings are written for a reader of the source, so they mark paths up
    in reStructuredText. On a terminal ``data/index/`` is just noise, so the
    backticks come off on the way through.
    """
    first = (module.__doc__ or "").strip().splitlines()[0]
    return first.replace("``", "")


def _positive(value: str) -> int:
    """An argument that has to be at least one, rejected at parse time if not.

    Without this, ``--batch-size 0`` is accepted, the batch never fills, and the
    whole corpus accumulates in memory before a single encode call, which then
    dies inside the encoder rather than at the flag that caused it. argparse
    turns the same typo into one line naming the option.
    """
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not a whole number") from None
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {number}")
    return number


# --- arguments --------------------------------------------------------------


def _add_embed(subparsers) -> None:
    parser = subparsers.add_parser(
        "embed",
        help=_summary(embed_stage),
        description=_summary(embed_stage),
    )
    parser.set_defaults(run=run_embed)
    parser.add_argument(
        "--tickers",
        nargs="+",
        metavar="TICKER",
        help="Only these companies. Default: every company in data/processed/.",
    )
    parser.add_argument(
        "--fiscal-years",
        nargs="+",
        type=int,
        metavar="YEAR",
        help="Only filings reporting on these fiscal years.",
    )
    parser.add_argument(
        "--key-items-only",
        action="store_true",
        help="Only passages from the Items marked as key sections.",
    )
    parser.add_argument(
        "--batch-size",
        type=_positive,
        default=EMBED_BATCH_SIZE,
        help=f"Passages per encode call. Default: {EMBED_BATCH_SIZE}.",
    )
    parser.add_argument(
        "--threads",
        type=_positive,
        metavar="N",
        help=(
            "Threads for the encoder. Default: every logical processor, which "
            "measured about 20 per cent faster than torch's own default. Pass "
            "the physical core count if that is not true on your machine."
        ),
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help=(
            "Drop the collection and embed everything again. Use after the "
            "corpus has been re-chunked; without it a run resumes instead, and "
            "refuses to resume onto a corpus that has moved."
        ),
    )
    parser.add_argument(
        "--chunk-budget",
        type=_positive,
        default=CHUNK_CHAR_BUDGET,
        metavar="CHARS",
        help=(
            "Recorded in the manifest as the budget the corpus was cut with. "
            "data/processed/ does not carry it, so pass the value used for "
            "'python -m src.pipeline chunk --budget' if it was not the "
            f"default of {CHUNK_CHAR_BUDGET}."
        ),
    )
    parser.add_argument(
        "--chunk-overlap",
        type=_positive,
        default=CHUNK_CHAR_OVERLAP,
        metavar="CHARS",
        help=(
            "As --chunk-budget, for the overlap the corpus was cut with. "
            f"Default: {CHUNK_CHAR_OVERLAP}."
        ),
    )


def _add_facts(subparsers) -> None:
    parser = subparsers.add_parser(
        "facts",
        help=_summary(facts_stage),
        description=_summary(facts_stage),
    )
    parser.set_defaults(run=run_facts)
    parser.add_argument(
        "--tickers",
        nargs="+",
        metavar="TICKER",
        help="Only these companies. Default: every company in the manifest.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help=(
            "Pull every company again rather than skipping those already "
            "stored. Use after the corpus has gained filings."
        ),
    )


# --- commands ---------------------------------------------------------------


def run_facts(args) -> None:
    """Pull the XBRL facts store for the corpus."""
    facts_stage.build(tickers=args.tickers, refresh=args.refresh)


def run_embed(args) -> None:
    """Build the dense index from data/processed/."""
    embed_stage.build(
        tickers=args.tickers,
        fiscal_years=args.fiscal_years,
        key_items_only=args.key_items_only,
        rebuild=args.rebuild,
        batch_size=args.batch_size,
        threads=args.threads,
        chunk_budget=args.chunk_budget,
        chunk_overlap=args.chunk_overlap,
    )


# --- dispatch ---------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Every command the retrieval stage has, and every option each one takes."""
    parser = argparse.ArgumentParser(
        prog="python -m src.retrieval",
        description="Build and search the retrieval indexes.",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")
    for add in (_add_embed, _add_facts):
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
