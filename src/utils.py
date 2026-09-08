"""Shared helpers that no single stage owns.

At the moment this is run logging: keeping a copy of what a command printed, so
a figure quoted in a report or a slide can be traced back to the run that
produced it. Stage-specific helpers belong in that stage's own module; this file
is for things every package would otherwise duplicate.
"""

from __future__ import annotations

import atexit
import re
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import IO
from collections.abc import Generator

from .config import LOGS_DIR

# Colour and cursor codes, which belong on a terminal and not in a saved file.
_ANSI = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


def _timestamped(name: str, logs_dir: Path) -> Path:
    """Where this run's log goes, named so runs sort chronologically."""
    safe = "".join(char if char.isalnum() or char in "-_" else "-" for char in name)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return logs_dir / f"{safe}-{stamp}.log"


class _Tee:
    """Write to the terminal and to a file at once.

    The file copy has its escape codes stripped. Without that, a library that
    colours its output writes the colour codes into the log as well, and the
    saved file is unreadable in anything but a terminal. Passing the terminal's
    own ``isatty`` through means libraries still colour what they print, so
    stripping here rather than disabling colour keeps both outputs right.
    """

    def __init__(self, terminal: IO[str], log: IO[str]) -> None:
        self._terminal = terminal
        self._log = log

    def write(self, text: str) -> int:
        self._terminal.write(text)
        self._log.write(_ANSI.sub("", text))
        # Flushing on every write costs nothing at this volume and means the log
        # is complete even if the run is interrupted part way through.
        self._log.flush()
        return len(text)

    def flush(self) -> None:
        self._terminal.flush()
        self._log.flush()

    def isatty(self) -> bool:
        return self._terminal.isatty()

    def __getattr__(self, attribute: str):
        # Anything else a caller expects of a stream, such as encoding or fileno,
        # is answered by the real terminal stream.
        return getattr(self._terminal, attribute)


def start_run_log(name: str, logs_dir: Path = LOGS_DIR) -> Path:
    """Copy everything this process prints into ``logs/<name>-<timestamp>.log``.

    Call it as the first statement of a command, before ``logging.basicConfig``.
    Logging writes to stderr by default and binds to whichever stream is in
    place when its handler is built, so a call made afterwards captures print
    output but silently misses every log line.

    Returns the path being written to, so a command can say where its log went.
    """
    logs_dir.mkdir(parents=True, exist_ok=True)
    path = _timestamped(name, logs_dir)
    handle = path.open("w", encoding="utf-8")

    sys.stdout = _Tee(sys.stdout, handle)
    sys.stderr = _Tee(sys.stderr, handle)

    # Restoring the streams matters more than closing the file: without it an
    # interactive session, a notebook above all, keeps writing to a log that was
    # meant to cover one command.
    def _restore() -> None:
        sys.stdout = getattr(sys.stdout, "_terminal", sys.stdout)
        sys.stderr = getattr(sys.stderr, "_terminal", sys.stderr)
        handle.close()

    atexit.register(_restore)
    return path


@contextmanager
def run_log(name: str, logs_dir: Path = LOGS_DIR) -> Generator[Path]:
    """``start_run_log`` scoped to a block, for notebooks and ad-hoc scripts.

        with run_log("chunk-sweep") as path:
            ...

    A command that runs once and exits should use ``start_run_log`` instead,
    since there is nothing to scope. Use this where several runs happen in one
    process and each wants its own file.
    """
    logs_dir.mkdir(parents=True, exist_ok=True)
    path = _timestamped(name, logs_dir)
    saved_out, saved_err = sys.stdout, sys.stderr
    with path.open("w", encoding="utf-8") as handle:
        sys.stdout = _Tee(saved_out, handle)
        sys.stderr = _Tee(saved_err, handle)
        try:
            yield path
        finally:
            sys.stdout, sys.stderr = saved_out, saved_err
