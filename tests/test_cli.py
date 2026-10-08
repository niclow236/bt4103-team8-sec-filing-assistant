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


def test_retrieval_commands_exist():
    commands = build_parser()._subparsers._group_actions[0].choices
    assert set(commands) == {"embed", "bm25", "facts", "check", "benchmark"}


def test_embed_takes_no_chunker_settings():
    """The corpus records them; an override could only repeat or contradict it."""
    embed = build_parser()._subparsers._group_actions[0].choices["embed"]
    flags = {option for action in embed._actions for option in action.option_strings}
    assert "--chunk-budget" not in flags and "--chunk-overlap" not in flags


def test_batch_size_zero_is_a_parse_error():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["embed", "--batch-size", "0"])


def test_sort_window_zero_is_a_parse_error():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["embed", "--sort-window", "0"])


def test_run_embed_passes_the_sort_window_through(monkeypatch):
    """The window bounds memory, so the command line has to be able to lower it."""
    from src.retrieval import cli

    passed = {}
    monkeypatch.setattr(cli.embed_stage, "build", lambda **kwargs: passed.update(kwargs))
    args = build_parser().parse_args(
        ["embed", "--batch-size", "8", "--sort-window", "16"]
    )
    args.run(args)
    assert passed["batch_size"] == 8 and passed["sort_window"] == 16


# --- check: are the indexes current? (#49) ----------------------------------------
#
# ``check`` is the command to run after pulling a change to the pipeline. It
# runs the comparison each retriever runs before it will load, and exits
# non-zero if either index should not be searched.


@pytest.fixture
def indexes(corpus, tmp_path, fake_model, monkeypatch):
    """Both indexes built over the synthetic corpus, and ``check`` pointed at them."""
    from src.retrieval import cli
    from src.retrieval.bm25 import BM25Retriever

    chroma, index_file = tmp_path / "index" / "chroma", tmp_path / "index" / "bm25.pkl"
    cli.embed_stage.build(chroma_dir=chroma, processed_dir=corpus)
    BM25Retriever.build(processed_dir=corpus, index_path=index_file)
    check_index = cli.embed_stage.check_index
    monkeypatch.setattr(cli.embed_stage, "check_index", lambda: check_index(chroma, corpus))
    monkeypatch.setattr(cli.bm25_stage, "BM25_INDEX_FILE", index_file)
    monkeypatch.setattr(cli.bm25_stage, "load_index",
                        lambda: BM25Retriever.load(index_file, processed_dir=corpus))
    return cli, corpus, index_file


def test_check_passes_two_indexes_that_match_the_corpus(indexes, capsys):
    cli, _, _ = indexes
    cli.run_check(build_parser().parse_args(["check"]))
    assert capsys.readouterr().out.splitlines() == ["dense: current", "bm25:  current"]


def test_check_exits_non_zero_and_says_why_once_the_corpus_has_moved(indexes, capsys):
    from conftest import edit_filing

    cli, corpus, _ = indexes
    edit_filing(sorted(corpus.glob("*/*.json"))[0], lambda data: data["chunks"][0].update(
        {"text": "Rewritten after both indexes were built."}))

    with pytest.raises(SystemExit) as stopped:
        cli.run_check(build_parser().parse_args(["check"]))

    assert stopped.value.code == 1
    printed = capsys.readouterr().out
    assert "dense: NOT usable" in printed and "bm25:  NOT usable" in printed
    assert "the index does not match the corpus: 0 passages missing, 1 stale" in printed
    assert "corpus fingerprint: index built from" in printed
    assert "Rebuild it: python -m src.retrieval bm25" in printed


def test_check_reports_each_index_on_its_own(indexes, capsys):
    """One stale index does not hide that the other is current, or the other way round."""
    cli, _, index_file = indexes
    index_file.unlink()

    with pytest.raises(SystemExit) as stopped:
        cli.run_check(build_parser().parse_args(["check"]))

    assert stopped.value.code == 1
    printed = capsys.readouterr().out
    assert "dense: current" in printed
    assert f"no index at {index_file}. Build it: python -m src.retrieval bm25" in printed


# --- the other commands, and the entry point ------------------------------------------


def test_bm25_reports_what_it_built(monkeypatch, capsys):
    from types import SimpleNamespace

    from src.retrieval import cli

    manifest = SimpleNamespace(n_passages=28_289, n_filings=75, path="data/index/bm25.pkl",
                               corpus_fingerprint="a35764e4db4d428e9cbb3926281261ea")
    monkeypatch.setattr(cli.bm25_stage, "build_index", lambda: SimpleNamespace(manifest=manifest))

    cli.run_bm25(build_parser().parse_args(["bm25"]))

    printed = capsys.readouterr().out
    assert "bm25:     28,289 passages from 75 filings  fingerprint a35764e4db4d" in printed
    assert "written:  data/index/bm25.pkl" in printed


def test_facts_and_benchmark_pass_their_arguments_to_the_stage(monkeypatch, capsys):
    from src.retrieval import cli

    asked = {}
    monkeypatch.setattr(cli.facts_stage, "build", lambda **kwargs: asked.update(kwargs))
    monkeypatch.setattr(cli.benchmark_stage, "generate_xbrl_questions",
                        lambda: ["question"] * 12_579)

    cli.run_facts(build_parser().parse_args(["facts", "--tickers", "AAPL", "MSFT", "--refresh"]))
    cli.run_benchmark(build_parser().parse_args(["benchmark"]))

    assert asked == {"tickers": ["AAPL", "MSFT"], "refresh": True}
    assert "benchmark: 12,579 questions written to" in capsys.readouterr().out


def test_a_retrieval_command_keeps_a_log_and_no_command_lists_them(monkeypatch, capsys, tmp_path):
    from src.retrieval import cli

    started = []
    monkeypatch.setattr(cli, "start_run_log",
                        lambda name: started.append(name) or tmp_path / f"{name}.log")
    monkeypatch.setattr(cli.embed_stage, "build", lambda **kwargs: None)

    cli.main(["embed", "--tickers", "AAPL"])
    cli.main([])

    printed = capsys.readouterr().out
    assert started == ["embed"]
    assert f"Run log: {tmp_path / 'embed.log'}" in printed
    assert "usage: python -m src.retrieval" in printed
