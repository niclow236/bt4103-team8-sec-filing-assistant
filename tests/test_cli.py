"""The retrieval command line."""

from __future__ import annotations

import argparse

import pytest

from src.retrieval.cli import _positive, build_parser


def test_positive_accepts_whole_numbers_from_one():
    assert _positive("1") == 1 and _positive("32") == 32


@pytest.mark.parametrize("value", ["0", "-4", "abc", "1.5"])
def test_positive_rejects_everything_else(value):
    with pytest.raises(argparse.ArgumentTypeError):
        _positive(value)


def test_the_four_commands_exist():
    commands = build_parser()._subparsers._group_actions[0].choices
    assert set(commands) == {"embed", "bm25", "facts", "check"}


def test_embed_takes_no_chunker_settings():
    """The corpus records them; an override could only repeat or contradict it."""
    embed = build_parser()._subparsers._group_actions[0].choices["embed"]
    flags = {option for action in embed._actions for option in action.option_strings}
    assert "--chunk-budget" not in flags and "--chunk-overlap" not in flags


def test_batch_size_zero_is_a_parse_error():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["embed", "--batch-size", "0"])
