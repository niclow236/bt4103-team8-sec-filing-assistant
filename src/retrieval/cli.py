"""The command line for the retrieval stage: every command, and what each runs.

This is the only module in the package that knows argparse exists, the same
arrangement ``src/pipeline/cli.py`` uses one stage earlier. The stage modules
hold the work and know nothing about how they were invoked, which is what lets
the same functions be called from a notebook, from the evaluation harness, or
from a test without a fake argument namespace.

    python -m src.retrieval                   # list the commands
    python -m src.retrieval embed --help      # options for one of them
    python -m src.retrieval embed             # build or update the dense index
    python -m src.retrieval bm25              # build the BM25 index
    python -m src.retrieval facts             # build the XBRL facts store
    python -m src.retrieval check             # are the indexes current?

It is a separate command line from the pipeline's rather than more subcommands
on it, because the two stages are separated by what they build from: the
pipeline turns EDGAR into ``data/processed/``, and retrieval turns that into the
indexes a question is answered against.

``embed``, ``bm25`` and ``check`` need no network and no EDGAR identity, since
they read only what the pipeline already wrote. ``facts`` does need both,
because the figures it stores are published through EDGAR rather than printed in
the filing's HTML, so it is the one command here that goes back to the source.
"""

from __future__ import annotations

import argparse
import logging

from ..utils import start_run_log
from . import bm25 as bm25_stage
from . import embed as embed_stage
from . import facts as facts_stage
from .constants import EMBED_BATCH_SIZE, EMBED_SORT_WINDOW

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
        help=(
            f"Passages per forward pass. Default: {EMBED_BATCH_SIZE}. This is "
            f"what the encoder runs at once, not what it is handed per call: "
            f"--sort-window sets that, and is what to lower if a machine runs "
            f"out of memory."
        ),
    )
    parser.add_argument(
        "--sort-window",
        type=_positive,
        default=EMBED_SORT_WINDOW,
        help=(
            f"Passages handed to the encoder per call, which it sorts by length "
            f"before cutting into batches of --batch-size. Default: "
            f"{EMBED_SORT_WINDOW}. A throughput knob: it changes no vector, but "
            f"it bounds what is held in memory and re-done after an interrupt."
        ),
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
            "Drop the collection and encode every passage again. Not needed "
            "after a re-chunk: a normal run re-encodes exactly the passages "
            "whose text changed. Needed after changing the embedding model."
        ),
    )


def _add_bm25(subparsers) -> None:
    parser = subparsers.add_parser(
        "bm25",
        help=_summary(bm25_stage),
        description=(
            f"{_summary(bm25_stage)} Rebuilds data/index/bm25.pkl from "
            "data/processed/ in full; there is no partial BM25 build, since its "
            "statistics are corpus-wide."
        ),
    )
    parser.set_defaults(run=run_bm25)


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
            "Pull the companies asked for again rather than skipping those "
            "already stored. Companies not asked for, and any whose request "
            "fails, keep what is stored."
        ),
    )


def _add_check(subparsers) -> None:
    parser = subparsers.add_parser(
        "check",
        help="Report whether each index matches the corpus on disk.",
        description=(
            "Run the same checks a retriever runs before loading an index, "
            "against the dense index and the BM25 index, and exit non-zero if "
            "either should not be searched."
        ),
    )
    parser.set_defaults(run=run_check)


# --- commands ---------------------------------------------------------------


def run_embed(args) -> None:
    """Build or update the dense index from data/processed/."""
    embed_stage.build(
        tickers=args.tickers,
        fiscal_years=args.fiscal_years,
        key_items_only=args.key_items_only,
        rebuild=args.rebuild,
        batch_size=args.batch_size,
        threads=args.threads,
        sort_window=args.sort_window,
    )


def run_bm25(args) -> None:
    """Build the BM25 index from data/processed/."""
    index = bm25_stage.build_index()
    manifest = index.manifest
    print(f"bm25:     {manifest.n_passages:,} passages from {manifest.n_filings} filings"
          f"  fingerprint {manifest.corpus_fingerprint[:12]}")
    print(f"written:  {manifest.path}")


def run_facts(args) -> None:
    """Pull the XBRL facts store for the corpus."""
    facts_stage.build(tickers=args.tickers, refresh=args.refresh)


def run_check(args) -> None:
    """Check both indexes against the corpus, the way their retrievers will."""
    failed = False

    problems = embed_stage.check_index()
    print("dense: " + ("current" if not problems else "NOT usable"))
    for problem in problems:
        print(f"  - {problem}")
    failed |= bool(problems)

    if not bm25_stage.BM25_INDEX_FILE.exists():
        print(f"bm25:  NOT usable\n  - no index at {bm25_stage.BM25_INDEX_FILE}. "
              f"Build it: python -m src.retrieval bm25")
        failed = True
    else:
        try:
            bm25_stage.load_index()
            print("bm25:  current")
        except ValueError as error:
            print("bm25:  NOT usable")
            for line in str(error).splitlines():
                print(f"  {line}")
            failed = True

    if failed:
        raise SystemExit(1)


# --- dispatch ---------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Every command the retrieval stage has, and every option each one takes."""
    parser = argparse.ArgumentParser(
        prog="python -m src.retrieval",
        description="Build and search the retrieval indexes.",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")
    for add in (_add_embed, _add_bm25, _add_facts, _add_check):
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
