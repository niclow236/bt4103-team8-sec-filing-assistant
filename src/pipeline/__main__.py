"""Run any pipeline command as ``python -m src.pipeline <command>``.

Each stage is also runnable on its own, as ``python -m src.pipeline.parse``,
which is how the pipeline is documented. This adds one place that lists every
command, so ``python -m src.pipeline`` answers "what can this do?" without
anybody having to go looking through the package for modules with a main().
"""

from __future__ import annotations

import sys

COMMANDS = {
    "download": "Fetch filings from SEC EDGAR into data/raw/",
    "parse": "Split downloaded filings into their numbered Items, into data/interim/",
    "chunk": "Cut parsed Items into retrievable passages, into data/processed/",
    "passages": "Show passages from data/processed/, for spot-checking",
}


def _usage() -> None:
    print("Usage: python -m src.pipeline <command> [options]\n")
    print("Commands:")
    for name, description in COMMANDS.items():
        print(f"  {name:10} {description}")
    print("\nRun a command with --help for its own options.")


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help"):
        _usage()
        return

    command, rest = argv[0], argv[1:]
    if command not in COMMANDS:
        print(f"Unknown command: {command}\n")
        _usage()
        raise SystemExit(2)

    # Imported here rather than at module level so that one command's import
    # cost, and any import error in a stage being worked on, does not fall on
    # every other command.
    module = __import__(f"src.pipeline.{command}", fromlist=["main"])
    module.main(rest)


if __name__ == "__main__":
    main()
