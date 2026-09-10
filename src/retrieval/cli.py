"""The command line for the retrieval stage: every command, and what each runs.

This is the only module in the package that knows argparse exists, the same
arrangement ``src/pipeline/cli.py`` uses one stage earlier. The stage modules
hold the work and know nothing about how they were invoked, which is what lets
the same functions be called from a notebook, from the evaluation harness, or
from a test without a fake argument namespace.

    python -m src.retrieval                   # list the commands
    python -m src.retrieval embed --help      # options for one of them
    python -m src.retrieval embed             # build the dense index

It is a separate command line from the pipeline's rather than more subcommands
on it, because the two stages are separated by their inputs: the pipeline turns
EDGAR into ``data/processed/`` and needs an EDGAR identity to do it, while
retrieval turns ``data/processed/`` into an index and never touches the network.
Someone rebuilding an index should not need EDGAR credentials configured.
"""

from __future__ import annotations

import argparse
import logging

from ..utils import start_run_log
from . import embed as embed_stage
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
        type=int,
        default=EMBED_BATCH_SIZE,
        help=f"Passages per encode call. Default: {EMBED_BATCH_SIZE}.",
    )
    parser.add_argument(
        "--threads",
        type=int,
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
            "corpus has been re-chunked; without it a run resumes instead."
        ),
    )


# --- commands ---------------------------------------------------------------


def run_embed(args) -> None:
    """Build the dense index from data/processed/."""
    embed_stage.build(
        tickers=args.tickers,
        fiscal_years=args.fiscal_years,
        key_items_only=args.key_items_only,
        rebuild=args.rebuild,
        batch_size=args.batch_size,
        threads=args.threads,
    )


# --- dispatch ---------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Every command the retrieval stage has, and every option each one takes."""
    parser = argparse.ArgumentParser(
        prog="python -m src.retrieval",
        description="Build and search the retrieval indexes.",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")
    for add in (_add_embed,):
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
