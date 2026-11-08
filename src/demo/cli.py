"""``python -m src.demo``: make a machine ready for the demo, check it, start it.

    python -m src.demo prepare     # once, before the day: fetch, verify, build
    python -m src.demo check       # any time: is the snapshot here? sends nothing
    python -m src.demo start       # on the day: check, then open the app offline

What the snapshot is, and why, is in ``snapshot.py``. This file reads the
arguments, calls it, and prints what it says.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from src.config import PROJECT_ROOT

from . import snapshot

APP = PROJECT_ROOT / "src" / "app" / "main.py"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.demo",
        description="The demo's pinned snapshot: C4 on the frozen evaluation corpus (#51).",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser(
        "prepare",
        help="Fetch and verify the frozen corpus, build what does not match, cache the encoder.",
        description=(
            "Make this machine hold the demo snapshot, doing only what is missing. "
            "Needs the internet. A machine with no dense index encodes every passage, "
            "which takes hours on a laptop CPU, so run it well before the demo."
        ),
    )
    prepare.add_argument(
        "--archive", type=Path, default=None,
        help=(f"Use this copy of {snapshot.PIN.corpus_asset} instead of downloading it. "
              "It is checked against the release's hash either way."),
    )
    prepare.add_argument(
        "--replace", action="store_true",
        help=("Replace a corpus or facts store here that is not the frozen one. What was "
              "here is moved into data/replaced-<time>/ with copies of both indexes; "
              "nothing is deleted."),
    )
    prepare.set_defaults(run=run_prepare)

    check = commands.add_parser(
        "check",
        help="Say whether this machine holds the snapshot. Sends nothing over the network.",
    )
    check.set_defaults(run=run_check)

    start = commands.add_parser(
        "start",
        help="Check the snapshot, then open the app with Hugging Face offline.",
        description=(
            "Runs check, and opens the app only if it passes: on C4, answering with "
            f"{snapshot.PIN.provider} {snapshot.PIN.model}, with nothing downloaded. "
            "Arguments after -- go to streamlit, for example -- --server.port 8502."
        ),
    )
    start.add_argument("streamlit_args", nargs=argparse.REMAINDER,
                       help=argparse.SUPPRESS)
    start.set_defaults(run=run_start)
    return parser


def print_report(report: snapshot.Report) -> None:
    for line in report.lines():
        print(line)
    if report.ready:
        print("\nThe demo snapshot is ready." + (
            " Warnings above do not stop the demo." if report.warnings else ""))
    else:
        print(f"\nNot ready: {len(report.problems)} problem(s) above.")


def run_prepare(args: argparse.Namespace) -> int:
    try:
        snapshot.prepare(archive=args.archive, replace=args.replace)
    except snapshot.PrepareError as error:
        print(f"prepare stopped: {error}", file=sys.stderr)
        return 1
    print()
    report = snapshot.check()
    print_report(report)
    return 0 if report.ready else 1


def run_check(args: argparse.Namespace) -> int:
    report = snapshot.check()
    print_report(report)
    return 0 if report.ready else 1


def run_start(args: argparse.Namespace) -> int:
    report = snapshot.check()
    print_report(report)
    if not report.ready:
        print("The app was not started.", file=sys.stderr)
        return 1
    extra = [arg for arg in args.streamlit_args if arg != "--"]
    command = [
        sys.executable, "-m", "streamlit", "run", str(APP),
        # No file watcher on the day: nothing is being edited, and it is what
        # sets off transformers' lazy imports after the first question.
        "--server.fileWatcherType", "none",
        *extra,
    ]
    print("\nStarting the app: " + " ".join(command[2:]))
    return subprocess.call(command, cwd=PROJECT_ROOT, env=snapshot.demo_environment())


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.run(args)
