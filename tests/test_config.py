"""Project-wide setup (#49): the ticker list, the EDGAR identity and the data folders."""

from __future__ import annotations

import os

import pytest

from src import config
from src.config import MissingIdentityError, configure_edgar, ensure_data_dirs, read_tickers


def test_the_ticker_list_ignores_comments_blank_lines_and_inline_notes(tmp_path):
    path = tmp_path / "companies.txt"
    path.write_text(
        "# Big tech\n"
        "AAPL    # Apple\n"
        "\n"
        "  msft\n"
        "#INTC   dropped: no Item headings\n"
        "GOOGL # Alphabet # the class A share\n",
        encoding="utf-8",
    )
    assert read_tickers(path) == ["AAPL", "MSFT", "GOOGL"]


def test_a_missing_ticker_list_is_an_error_that_names_the_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Ticker list not found"):
        read_tickers(tmp_path / "absent.txt")


def test_the_projects_own_list_is_the_fifteen_companies_the_readme_names():
    tickers = read_tickers()
    assert len(tickers) == len(set(tickers)) == 15
    assert {"AAPL", "MSFT", "ORCL", "PANW"} <= set(tickers)
    assert "INTC" not in tickers and "IBM" not in tickers


def test_the_identity_in_the_environment_is_handed_to_edgartools(monkeypatch):
    given = []
    monkeypatch.setattr(config, "set_identity", given.append)
    monkeypatch.setenv("EDGAR_IDENTITY", "  Jane Tan jane@example.com  ")

    assert configure_edgar() == "Jane Tan jane@example.com"
    assert given == ["Jane Tan jane@example.com"]


@pytest.mark.parametrize("value", [None, "", "   "])
def test_no_identity_fails_at_once_and_says_how_to_set_one(monkeypatch, value):
    """The SEC blocks a request with no contact string, part-way through a long download."""
    given = []
    monkeypatch.setattr(config, "set_identity", given.append)
    if value is None:
        monkeypatch.delenv("EDGAR_IDENTITY", raising=False)
    else:
        monkeypatch.setenv("EDGAR_IDENTITY", value)

    with pytest.raises(MissingIdentityError, match="Copy .env.example to .env"):
        configure_edgar()
    assert given == []


def test_a_setting_in_env_is_read_but_never_over_one_already_set(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("EDGAR_IDENTITY=From The File file@example.com\nTEAM8_TEST_ONLY_SETTING=file\n",
                   encoding="utf-8")
    monkeypatch.setenv("EDGAR_IDENTITY", "From The Shell shell@example.com")
    monkeypatch.delenv("TEAM8_TEST_ONLY_SETTING", raising=False)
    try:
        assert config.load_env(env) is True
        assert os.environ["EDGAR_IDENTITY"] == "From The Shell shell@example.com"
        assert os.environ["TEAM8_TEST_ONLY_SETTING"] == "file"
        assert config.load_env(tmp_path / "absent.env") is False
    finally:
        monkeypatch.delenv("TEAM8_TEST_ONLY_SETTING", raising=False)


def test_the_data_folders_are_made_and_making_them_again_changes_nothing(tmp_path, monkeypatch):
    folders = {name: tmp_path / "data" / part for name, part in (
        ("RAW_DIR", "raw"), ("INTERIM_DIR", "interim"), ("PROCESSED_DIR", "processed"),
        ("SAMPLE_DIR", "sample"), ("INDEX_DIR", "index"))}
    for name, path in folders.items():
        monkeypatch.setattr(config, name, path)

    ensure_data_dirs()
    (folders["RAW_DIR"] / "manifest.jsonl").write_text("kept\n", encoding="utf-8")
    ensure_data_dirs()

    assert all(path.is_dir() for path in folders.values())
    assert (folders["RAW_DIR"] / "manifest.jsonl").read_text(encoding="utf-8") == "kept\n"
    # Written only when something asks for it, so most runs leave it absent.
    assert not (tmp_path / "data" / "diagnostics").exists()


def test_every_path_is_under_the_project_root_the_package_sits_in():
    assert config.PROJECT_ROOT == (config.PROJECT_ROOT / "src" / "config.py").parents[1]
    assert (config.PROJECT_ROOT / "src" / "config.py").is_file()
    for path in (config.RAW_DIR, config.INTERIM_DIR, config.PROCESSED_DIR, config.INDEX_DIR,
                 config.BM25_INDEX_FILE, config.CHROMA_DIR, config.MANIFEST_FILE,
                 config.LOGS_DIR, config.COMPANIES_FILE):
        assert config.PROJECT_ROOT in path.parents
    assert config.MANIFEST_FILE.parent == config.RAW_DIR
    assert config.BM25_INDEX_FILE.parent == config.CHROMA_DIR.parent == config.INDEX_DIR
