"""The demo's pinned snapshot (#51): the pins, the offline check, prepare and start."""

from __future__ import annotations

import io
import json
import sys
import tarfile
from dataclasses import replace
from pathlib import Path

import pytest

from conftest import edit_filing, write_corpus
from src.config import PROJECT_ROOT
from src.demo import cli, snapshot
from src.demo.snapshot import PIN, Layout, PrepareError
from src.pipeline import verify
from src.rag.constants import (
    LLM_MODEL_ENV, LLM_PROVIDER_ENV, MISTRAL_API_KEY_ENV, SOCLAAS_API_KEY_ENV,
)
from src.retrieval import constants as retrieval_constants
from src.retrieval import embed

STUDY = PROJECT_ROOT / "docs" / "evaluation" / "full-matrix-20261021-manifest.json"

READY_ENV = {
    LLM_PROVIDER_ENV: PIN.provider, LLM_MODEL_ENV: PIN.model,
    SOCLAAS_API_KEY_ENV: "soclaas-test-key", MISTRAL_API_KEY_ENV: "mistral-test-key",
}


def _found(*_):
    return "cache/bge"


def _layout(root: Path) -> Layout:
    return Layout(
        processed_dir=root / "data" / "processed",
        bm25_index=root / "data" / "index" / "bm25.pkl",
        chroma_dir=root / "data" / "index" / "chroma",
        facts_file=root / "data" / "index" / "facts.parquet",
        data_dir=root / "data",
    )


def _build(layout: Layout) -> None:
    snapshot._build_bm25(layout)
    snapshot._build_dense(layout)


def _pin_for(layout: Layout) -> snapshot.DemoPin:
    """The real pin, with the corpus identities of the test's own corpus."""
    return replace(
        PIN,
        bm25_fingerprint=snapshot.corpus_identity(layout.processed_dir)[0],
        dense_fingerprint=snapshot.dense_manifest(layout.chroma_dir).corpus_fingerprint,
        facts_sha256=snapshot.sha256_of(layout.facts_file),
    )


@pytest.fixture
def ready(tmp_path, fake_model):
    """A machine holding a (small) snapshot: corpus, facts, both indexes."""
    layout = _layout(tmp_path)
    write_corpus(layout.processed_dir)
    layout.facts_file.parent.mkdir(parents=True, exist_ok=True)
    layout.facts_file.write_bytes(b"frozen facts")
    _build(layout)
    return layout, _pin_for(layout)


def _items(report):
    return {item.name: item for item in report.items}


# --- the pins ---------------------------------------------------------------

def test_the_pin_names_the_indexes_and_encoder_the_final_study_measured():
    """A typo in a 64-character literal would pin an index nobody measured."""
    study = json.loads(STUDY.read_text(encoding="utf-8"))
    dense, bm25, revisions = set(), set(), set()

    def walk(node):
        if isinstance(node, dict):
            if node.get("index_type") == "dense":
                dense.add(node["corpus_fingerprint"])
            if node.get("index_type") == "bm25":
                bm25.add(node["corpus_fingerprint"])
            # The encoder's runtime record: {"device": ..., "revision": ...}.
            if isinstance(node.get("dense"), dict) and "revision" in node["dense"]:
                revisions.add(node["dense"]["revision"])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(study)
    assert dense == {PIN.dense_fingerprint}
    assert bm25 == {PIN.bm25_fingerprint}
    assert revisions == {PIN.encoder_revision}


def test_the_code_constants_match_the_pin():
    assert retrieval_constants.EMBED_MODEL == PIN.encoder
    assert retrieval_constants.EMBED_REVISION == PIN.encoder_revision
    assert _items(snapshot.check(PIN, _layout(Path("missing")), environ=READY_ENV,
                                 find_encoder=_found))["code"].ok


def test_the_release_url_is_the_public_download_of_the_pinned_asset():
    assert PIN.corpus_url == (
        "https://github.com/niclow236/bt4103-team8-sec-filing-assistant/releases/download/"
        "data/evaluation-v1/sec-eval-150-12579-v1.tar.gz")


# --- check -----------------------------------------------------------------

def test_a_machine_holding_the_snapshot_is_ready(ready):
    layout, pin = ready
    report = snapshot.check(pin, layout, environ=READY_ENV, find_encoder=_found)
    assert report.ready, report.lines()
    assert [item.name for item in report.items] == [
        "code", "corpus", "facts", "bm25", "dense", "encoder", "answers", "fallback"]
    assert all(item.ok for item in report.items)
    assert "90 passages from 6 filings" in _items(report)["corpus"].detail


def test_an_empty_machine_is_told_to_run_prepare(tmp_path):
    report = snapshot.check(PIN, _layout(tmp_path), environ=READY_ENV, find_encoder=_found)
    items = _items(report)
    for name in ("corpus", "facts", "bm25", "dense"):
        assert not items[name].ok
        assert "python -m src.demo prepare" in items[name].detail
    assert len(report.problems) == 4


def test_a_changed_passage_fails_the_corpus(ready):
    layout, pin = ready
    first = sorted(layout.processed_dir.rglob("*.json"))[0]
    edit_filing(first, lambda data: data["chunks"][0].update(text="edited"))
    items = _items(snapshot.check(pin, layout, environ=READY_ENV, find_encoder=_found))
    assert not items["corpus"].ok
    assert "--replace" in items["corpus"].detail


def test_an_index_that_is_not_the_measured_one_fails(ready):
    layout, pin = ready
    other = replace(pin, bm25_fingerprint="0" * 64, dense_fingerprint="1" * 64)
    items = _items(snapshot.check(other, layout, environ=READY_ENV, find_encoder=_found))
    assert not items["bm25"].ok and "000000000000" in items["bm25"].detail
    assert not items["dense"].ok and "111111111111" in items["dense"].detail


def test_the_vectors_are_read_not_only_the_manifest(ready, monkeypatch):
    """A manifest that says the right thing over vectors that do not."""
    layout, pin = ready
    monkeypatch.setattr(embed, "check_index", lambda *args, **kwargs: ["3 passages are stale"])
    items = _items(snapshot.check(pin, layout, environ=READY_ENV, find_encoder=_found))
    assert not items["dense"].ok
    assert "3 passages are stale" in items["dense"].detail


def test_a_changed_facts_store_fails(ready):
    layout, pin = ready
    layout.facts_file.write_bytes(b"other facts")
    assert not _items(snapshot.check(pin, layout, environ=READY_ENV,
                                     find_encoder=_found))["facts"].ok


def test_an_encoder_missing_from_the_cache_fails(ready):
    layout, pin = ready
    items = _items(snapshot.check(pin, layout, environ=READY_ENV, find_encoder=lambda _: None))
    assert not items["encoder"].ok
    assert PIN.encoder_revision[:7] in items["encoder"].detail


def test_no_soclaas_key_fails_but_no_mistral_key_only_warns(ready):
    layout, pin = ready
    no_soclaas = {k: v for k, v in READY_ENV.items() if k != SOCLAAS_API_KEY_ENV}
    report = snapshot.check(pin, layout, environ=no_soclaas, find_encoder=_found)
    assert [item.name for item in report.problems] == ["answers"]

    no_mistral = {k: v for k, v in READY_ENV.items() if k != MISTRAL_API_KEY_ENV}
    report = snapshot.check(pin, layout, environ=no_mistral, find_encoder=_found)
    assert report.ready
    assert [item.name for item in report.warnings] == ["fallback"]
    assert "warn" in "\n".join(report.lines())


def test_a_changed_default_model_is_reported_as_drift(ready, monkeypatch):
    layout, pin = ready
    from src.rag import constants as rag

    monkeypatch.setattr(rag, "DEFAULT_SOCLAAS_MODEL", "some-other-model")
    items = _items(snapshot.check(pin, layout, environ=READY_ENV, find_encoder=_found))
    assert not items["code"].ok
    assert "DEFAULT_SOCLAAS_MODEL" in items["code"].detail


def test_an_app_that_opens_elsewhere_or_a_changed_c4_is_drift(ready, monkeypatch):
    layout, pin = ready
    from src import stack

    monkeypatch.setattr(stack, "DEFAULT_STACK", "C1")
    items = _items(snapshot.check(pin, layout, environ=READY_ENV, find_encoder=_found))
    assert not items["code"].ok
    assert "the app opens on C1" in items["code"].detail

    monkeypatch.setattr(stack, "DEFAULT_STACK", "C4")
    monkeypatch.setitem(stack.STACKS, "C4", replace(stack.STACKS["C4"], metadata_filter=False))
    items = _items(snapshot.check(pin, layout, environ=READY_ENV, find_encoder=_found))
    assert not items["code"].ok
    assert "no longer hybrid retrieval with the metadata filter" in items["code"].detail


def test_the_check_sends_nothing(ready, monkeypatch):
    layout, pin = ready

    def refuse(*args, **kwargs):
        raise AssertionError("check opened a connection")

    import socket

    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket.socket, "connect", refuse)
    assert snapshot.check(pin, layout, environ=READY_ENV, find_encoder=_found).ready


def test_the_cache_lookup_never_downloads(monkeypatch):
    seen = {}

    def snapshot_download(repo, **kwargs):
        seen.update(kwargs, repo=repo)
        raise FileNotFoundError("not cached")

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    assert snapshot.encoder_cached() is None
    assert seen == {"repo": PIN.encoder, "revision": PIN.encoder_revision,
                    "local_files_only": True}


# --- the environment the app runs in ---------------------------------------

def test_the_demo_environment_pins_the_model_and_goes_offline():
    base = {LLM_PROVIDER_ENV: "mistral", LLM_MODEL_ENV: "mistral-large-latest", "KEEP": "1"}
    env = snapshot.demo_environment(PIN, base)
    assert env[LLM_PROVIDER_ENV] == "soclaas"
    assert env[LLM_MODEL_ENV] == "qwen3.6:35b"   # not .env's Mistral model, handed to SOCLAAS
    assert env["HF_HUB_OFFLINE"] == "1" and env["TRANSFORMERS_OFFLINE"] == "1"
    assert env["KEEP"] == "1"
    assert base[LLM_PROVIDER_ENV] == "mistral"   # the caller's mapping is not changed


def test_the_fallback_is_mistrals_default_in_the_demo_environment():
    from src.rag.generate import config_from_env

    env = snapshot.demo_environment(PIN, {})
    assert config_from_env(environ=env).model == PIN.model
    assert config_from_env(provider="mistral", environ=env).model == PIN.fallback_model


# --- prepare ---------------------------------------------------------------

def _archive(tmp_path: Path, processed: Path, facts: bytes, extra=()) -> Path:
    """An archive shaped like the release: one top folder, the project's paths inside."""
    path = tmp_path / "release.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        for file in sorted(processed.rglob("*.json")):
            archive.add(file, f"snapshot/data/processed/{file.relative_to(processed).as_posix()}")
        info = tarfile.TarInfo("snapshot/data/index/facts.parquet")
        info.size = len(facts)
        archive.addfile(info, io.BytesIO(facts))
        info = tarfile.TarInfo("snapshot/benchmark/questions.jsonl")
        info.size = 2
        archive.addfile(info, io.BytesIO(b"{}"))
        for member, data in extra:
            archive.addfile(member, io.BytesIO(data) if data is not None else None)
    return path


@pytest.fixture
def frozen(tmp_path, fake_model):
    """The frozen corpus as a release archive, and the pin that names it."""
    source = write_corpus(tmp_path / "frozen" / "processed")
    facts = b"frozen facts"
    archive = _archive(tmp_path, source, facts)
    built = _layout(tmp_path / "reference")
    write_corpus(built.processed_dir)
    built.facts_file.parent.mkdir(parents=True)
    built.facts_file.write_bytes(facts)
    _build(built)
    pin = replace(_pin_for(built), corpus_sha256=snapshot.sha256_of(archive))
    return archive, pin


def test_prepare_fills_an_empty_machine(tmp_path, frozen):
    archive, pin = frozen
    layout = _layout(tmp_path / "machine")
    calls = []

    def download(url, target):
        calls.append(("download", url))
        target.write_bytes(archive.read_bytes())

    snapshot.prepare(pin, layout, download=download,
                     fetch_encoder=lambda pin: calls.append(("encoder", pin.encoder_revision)),
                     log=lambda _: None)

    assert calls == [("download", pin.corpus_url), ("encoder", pin.encoder_revision)]
    assert (layout.downloads / pin.corpus_asset).is_file()
    assert not (layout.data_dir / "benchmark").exists()   # only the corpus and facts are taken
    report = snapshot.check(pin, layout, environ=READY_ENV, find_encoder=_found)
    assert report.ready, report.lines()

    # A second prepare uses the archive it kept, and finds nothing to do.
    snapshot.prepare(pin, layout, download=lambda *_: pytest.fail("downloaded again"),
                     fetch_encoder=lambda _: None, log=lambda _: None)


def test_prepare_on_a_ready_machine_does_nothing(ready, monkeypatch):
    layout, pin = ready
    calls = []
    monkeypatch.setattr(snapshot, "_build_bm25", lambda *_: pytest.fail("rebuilt bm25"))
    monkeypatch.setattr(snapshot, "_build_dense", lambda *_: pytest.fail("rebuilt dense"))
    snapshot.prepare(pin, layout, download=lambda *_: pytest.fail("downloaded"),
                     fetch_encoder=lambda pin: calls.append("encoder"), log=lambda _: None)
    assert calls == ["encoder"]   # a no-op when it is cached, which only the library knows


def test_prepare_refuses_a_different_corpus_and_changes_nothing(tmp_path, frozen):
    archive, pin = frozen
    layout = _layout(tmp_path / "machine")
    write_corpus(layout.processed_dir)
    first = sorted(layout.processed_dir.rglob("*.json"))[0]
    edit_filing(first, lambda data: data["chunks"][0].update(text="a teammate's edit"))
    before = {path: path.read_bytes() for path in layout.data_dir.rglob("*") if path.is_file()}

    with pytest.raises(PrepareError, match="--replace"):
        snapshot.prepare(pin, layout, archive=archive, log=lambda _: None)
    after = {path: path.read_bytes() for path in layout.data_dir.rglob("*") if path.is_file()}
    assert after == before


def test_prepare_replace_moves_the_old_corpus_aside_and_keeps_the_index(tmp_path, frozen,
                                                                       fake_model):
    archive, pin = frozen
    layout = _layout(tmp_path / "machine")
    write_corpus(layout.processed_dir)
    first = sorted(layout.processed_dir.rglob("*.json"))[0]
    edit_filing(first, lambda data: data["chunks"][0].update(text="a teammate's edit"))
    layout.facts_file.parent.mkdir(parents=True)
    layout.facts_file.write_bytes(b"old facts")
    _build(layout)
    calls_before = fake_model.calls

    snapshot.prepare(pin, layout, archive=archive, replace=True,
                     fetch_encoder=lambda _: None, log=lambda _: None)

    backups = list(layout.data_dir.glob("replaced-*"))
    assert len(backups) == 1
    backup = backups[0]
    assert "a teammate's edit" in "".join(
        path.read_text(encoding="utf-8") for path in (backup / "processed").rglob("*.json"))
    assert (backup / "facts.parquet").read_bytes() == b"old facts"
    assert (backup / "index" / "bm25.pkl").is_file()
    assert (backup / "index" / "chroma").is_dir()
    assert (backup / "index" / "chroma.manifest.json").is_file()
    assert snapshot.check(pin, layout, environ=READY_ENV, find_encoder=_found).ready
    # The dense index was brought up to date in place: one passage encoded again,
    # not the whole corpus.
    assert fake_model.calls == calls_before + 1


def test_a_bad_archive_stops_prepare_before_anything_moves(tmp_path, frozen):
    archive, pin = frozen
    layout = _layout(tmp_path / "machine")
    write_corpus(layout.processed_dir)
    first = sorted(layout.processed_dir.rglob("*.json"))[0]
    edit_filing(first, lambda data: data["chunks"][0].update(text="a teammate's edit"))
    tampered = tmp_path / "tampered.tar.gz"
    tampered.write_bytes(archive.read_bytes() + b"x")

    with pytest.raises(PrepareError, match="sha256"):
        snapshot.prepare(pin, layout, archive=tampered, replace=True, log=lambda _: None)
    assert not list(layout.data_dir.glob("replaced-*"))
    assert sorted(layout.processed_dir.rglob("*.json"))


def test_a_failed_download_leaves_no_archive_behind(tmp_path, frozen):
    archive, pin = frozen
    layout = _layout(tmp_path / "machine")

    def broken(url, target):
        target.write_bytes(b"half")
        raise OSError("connection reset")

    with pytest.raises(OSError):
        snapshot.prepare(pin, layout, download=broken, log=lambda _: None)
    assert not (layout.downloads / pin.corpus_asset).exists()


def test_unsafe_archive_entries_are_not_extracted(tmp_path):
    layout = _layout(tmp_path / "machine")
    members = []
    for name in ("snapshot/../escape.json", "/abs/data/processed/x.json",
                 "snapshot/data/processed/../../escape.json"):
        info = tarfile.TarInfo(name)
        info.size = 2
        members.append(info)
    assert list(snapshot._wanted(members, layout, corpus=True, facts=True)) == []

    link = tarfile.TarInfo("snapshot/data/processed/AAA/link.json")
    link.type = tarfile.SYMTYPE
    link.linkname = "/etc/passwd"
    with pytest.raises(PrepareError, match="unexpected archive entry"):
        list(snapshot._wanted([link], layout, corpus=True, facts=True))


# --- the commands -----------------------------------------------------------

def test_start_refuses_when_the_check_fails(monkeypatch, capsys):
    failing = snapshot.Report((snapshot.Item("corpus", False, "not the frozen corpus"),))
    monkeypatch.setattr(snapshot, "check", lambda *a, **k: failing)
    monkeypatch.setattr(cli.subprocess, "call", lambda *a, **k: pytest.fail("started the app"))
    assert cli.main(["start"]) == 1
    assert "The app was not started." in capsys.readouterr().err


def test_start_opens_the_app_offline_on_the_pinned_model(monkeypatch):
    passing = snapshot.Report((snapshot.Item("corpus", True, "frozen"),))
    monkeypatch.setattr(snapshot, "check", lambda *a, **k: passing)
    seen = {}

    def call(command, cwd, env):
        seen.update(command=command, cwd=cwd, env=env)
        return 0

    monkeypatch.setattr(cli.subprocess, "call", call)
    assert cli.main(["start", "--", "--server.port", "8502"]) == 0
    assert seen["command"][:4] == [sys.executable, "-m", "streamlit", "run"]
    assert seen["command"][4] == str(PROJECT_ROOT / "src" / "app" / "main.py")
    assert seen["command"][-2:] == ["--server.port", "8502"]
    assert "--server.fileWatcherType" in seen["command"]
    assert seen["env"]["HF_HUB_OFFLINE"] == "1"
    assert seen["env"][LLM_PROVIDER_ENV] == "soclaas"
    assert seen["cwd"] == PROJECT_ROOT


def test_check_exits_non_zero_when_not_ready(monkeypatch, capsys):
    failing = snapshot.Report((snapshot.Item("dense", False, "stale"),))
    monkeypatch.setattr(snapshot, "check", lambda *a, **k: failing)
    assert cli.main(["check"]) == 1
    assert "Not ready: 1 problem(s)" in capsys.readouterr().out


def test_prepare_command_reports_a_refusal(monkeypatch, capsys):
    def refuse(**kwargs):
        raise PrepareError("not the frozen corpus")

    monkeypatch.setattr(snapshot, "prepare", refuse)
    assert cli.main(["prepare"]) == 1
    assert "prepare stopped: not the frozen corpus" in capsys.readouterr().err


# --- the encoder is loaded at its pinned checkpoint -------------------------

def test_the_project_encoder_loads_at_the_pinned_revision(monkeypatch):
    seen = []

    class Recorder:
        def __init__(self, name, revision=None):
            seen.append((name, revision))

        def get_embedding_dimension(self):
            return 768

    import sentence_transformers

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", Recorder)
    embed._load_model(threads=1)
    embed._load_model(threads=1, model_name="intfloat/e5-base-v2", expected_dimensions=768)
    assert seen == [(PIN.encoder, PIN.encoder_revision), ("intfloat/e5-base-v2", None)]


def test_the_token_counter_reads_the_same_checkpoint(monkeypatch):
    seen = {}

    class Tokenizer:
        @staticmethod
        def from_pretrained(name, revision=None):
            seen.update(name=name, revision=revision)
            return Tokenizer()

        def no_truncation(self):
            pass

        def no_padding(self):
            pass

    import tokenizers

    monkeypatch.setattr(tokenizers, "Tokenizer", Tokenizer)
    verify.bge_token_counter()
    assert seen == {"name": PIN.encoder, "revision": PIN.encoder_revision}
