"""Run logging (#49): a command's output is on the terminal and in a file, the same text.

A figure quoted in a report is traced back to the run that printed it through
``logs/<command>-<timestamp>.log``, so the file has to hold what was printed,
without the colour codes a terminal takes and an editor does not.
"""

from __future__ import annotations

import io
import logging
import re
import sys

import pytest

from src import utils
from src.utils import _Tee, run_log, start_run_log


def test_what_is_printed_inside_the_block_reaches_both_the_terminal_and_the_log(
        capsys, tmp_path):
    with run_log("chunk-sweep", logs_dir=tmp_path) as path:
        print("budget 1200: 44,688 passages")
        print("a warning", file=sys.stderr)

    shown = capsys.readouterr()
    assert shown.out == "budget 1200: 44,688 passages\n"
    assert shown.err == "a warning\n"
    assert path.read_text(encoding="utf-8") == "budget 1200: 44,688 passages\na warning\n"


def test_nothing_printed_after_the_block_lands_in_its_log(tmp_path):
    before = sys.stdout, sys.stderr
    with run_log("parse", logs_dir=tmp_path) as path:
        assert sys.stdout is not before[0] and sys.stderr is not before[1]
        print("inside")

    assert (sys.stdout, sys.stderr) == before
    print("after the block")
    assert path.read_text(encoding="utf-8") == "inside\n"


def test_the_streams_are_put_back_when_the_block_raises(tmp_path):
    before = sys.stdout, sys.stderr
    with pytest.raises(RuntimeError):
        with run_log("embed", logs_dir=tmp_path) as path:
            print("encoding")
            raise RuntimeError("interrupted")

    assert (sys.stdout, sys.stderr) == before
    # What was printed before the failure is in the file, which is when it is wanted.
    assert path.read_text(encoding="utf-8") == "encoding\n"


def test_the_log_is_named_for_the_command_and_the_time_it_ran(tmp_path):
    with run_log("parse", logs_dir=tmp_path / "logs") as path:
        pass
    assert path.parent == tmp_path / "logs"
    assert re.fullmatch(r"parse-\d{8}-\d{6}\.log", path.name)

    with run_log("rerank / sweep: v2", logs_dir=tmp_path) as odd:
        pass
    # A name is made safe for a file name, not refused.
    assert re.fullmatch(r"rerank---sweep--v2-\d{8}-\d{6}\.log", odd.name)


class Terminal(io.StringIO):
    """A stream that says it is a terminal, as a console does."""

    encoding = "cp1252"

    def isatty(self):
        return True


def test_colour_codes_stay_on_the_terminal_and_out_of_the_file():
    terminal, log = Terminal(), io.StringIO()
    tee = _Tee(terminal, log)

    written = tee.write("\x1b[32mPASS\x1b[0m coverage \x1b[1;31mFAIL\x1b[0m key Items\n")

    assert terminal.getvalue() == "\x1b[32mPASS\x1b[0m coverage \x1b[1;31mFAIL\x1b[0m key Items\n"
    assert log.getvalue() == "PASS coverage FAIL key Items\n"
    assert written == len(terminal.getvalue())
    # A library asks the stream whether to colour at all, and is answered by
    # the terminal: stripping in the file is what keeps both outputs right.
    assert tee.isatty() and tee.encoding == "cp1252"
    tee.flush()


def test_a_character_the_terminal_cannot_show_does_not_end_the_run():
    """A redirected Windows stream is not UTF-8, and filings are full of curly quotes."""
    class Narrow(io.StringIO):
        encoding = "ascii"

        def write(self, text):
            text.encode("ascii")              # raises on anything the code page lacks
            return super().write(text)

    terminal, log = Narrow(), io.StringIO()
    _Tee(terminal, log).write("“supply chain” ─ 12\n")

    # A replaced glyph on screen, and the real text in the file.
    assert terminal.getvalue() == "?supply chain? ? 12\n"
    assert log.getvalue() == "“supply chain” ─ 12\n"


def test_a_command_started_before_logging_is_configured_captures_its_log_lines(
        capsys, tmp_path, monkeypatch):
    """``start_run_log`` first, then the logging handler: a handler binds to the stream
    in place when it is built, so the other order logs to the terminal and not the file."""
    before = sys.stdout, sys.stderr
    registered = []
    monkeypatch.setattr(utils.atexit, "register", registered.append)

    path = start_run_log("chunk", logs_dir=tmp_path / "logs")
    handler = logging.StreamHandler()            # stderr as it is now: the copy
    logger = logging.getLogger("tests.run-log")
    logger.addHandler(handler)
    try:
        print("Chunked 75 filings")
        logger.warning("one filing could not be read")
        sys.stdout.flush()
    finally:
        logger.removeHandler(handler)
        (restore,) = registered
        restore()

    assert (sys.stdout, sys.stderr) == before
    shown = capsys.readouterr()
    assert shown.out == "Chunked 75 filings\n"
    assert shown.err == "one filing could not be read\n"
    assert path.read_text(encoding="utf-8") == (
        "Chunked 75 filings\none filing could not be read\n")
    assert re.fullmatch(r"chunk-\d{8}-\d{6}\.log", path.name)
