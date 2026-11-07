from __future__ import annotations

import hashlib
import io
import json
import pickle
import subprocess
import tarfile
from dataclasses import asdict
from pathlib import Path

import pytest

from tools.archive_handover import (
    HandoverError,
    create_archive,
    sha256_file,
    validate_frozen_identities,
    validate_native_results,
    verify_archive,
)
from src.retrieval.records import IndexManifest


RESULT_BYTES = b'{"score": 1}\n'
RESULT_SHA256 = "fa6619ab8a8eee2536bab56faf4d131f1167a0c85904eefaa3bd3eb73e949623"


def _write(path: Path, content: str | bytes = "x\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")
    return path


def _report_manifest(path: Path, run_file: Path) -> Path:
    payload = {"native_artifacts_sha256": {run_file.name: RESULT_SHA256}}
    _write(path, json.dumps(payload))
    return path


def test_native_results_must_match_the_recorded_set_and_hashes(tmp_path: Path):
    run = tmp_path / "results" / "run"
    result = _write(run / "summary.json", RESULT_BYTES)
    report = _report_manifest(tmp_path / "report.json", result)

    assert validate_native_results(run, report) == {"summary.json": RESULT_SHA256}

    # Runtime debris and derived rescoring output are deliberately outside the
    # native set selected by the report generator.
    _write(run / "active.lock")
    _write(run / "worker-process-status.json")
    _write(run / "rescored" / "summary.json")
    assert validate_native_results(run, report) == {"summary.json": RESULT_SHA256}

    _write(run / "unrecorded.json", "{}\n")
    with pytest.raises(HandoverError, match="not recorded by report manifest"):
        validate_native_results(run, report)
    (run / "unrecorded.json").unlink()

    _write(result, '{"score": 0}\n')
    with pytest.raises(HandoverError, match="SHA-256 mismatch"):
        validate_native_results(run, report)


def test_missing_native_run_names_the_originals_release(
    tmp_path: Path,
):
    report = _write(
        tmp_path / "report.json",
        json.dumps({"native_artifacts_sha256": {"summary.json": "0" * 64}}),
    )
    with pytest.raises(HandoverError, match="data/full-matrix-20261021-v1"):
        validate_native_results(tmp_path / "absent", report)


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    )
    return completed.stdout.strip()


def _release_checkout(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "repo"
    root.mkdir()
    run_id = "full-matrix-20261021"
    native = _write(root / "results" / run_id / "summary.json", RESULT_BYTES)
    team = _write(root / "benchmark" / "questions.jsonl", '{"id":"team"}\n')
    generated = _write(root / "benchmark" / "generated.jsonl", '{"id":"xbrl"}\n')
    _write(root / "benchmark" / "schema.md", "# Schema\n")
    _write(root / "docs" / "full-ablation.md", "# Study\n")
    _write(root / "handover" / "README.md", "# Handover\n")
    _write(root / "results" / "README.md", "# Limitations\n")
    dense = {"index_type": "dense", "n_passages": 1}
    bm25_record = IndexManifest(
        index_type="bm25",
        path="data/index/bm25.pkl",
        corpus_fingerprint="b" * 64,
        n_passages=1,
        n_filings=1,
        built_at="2026-01-01T00:00:00+00:00",
        chunk_budget=1800,
        chunk_overlap=300,
    )
    bm25 = asdict(bm25_record)
    _write(root / "data" / "index" / "chroma.manifest.json", json.dumps(dense))
    with _write(root / "data" / "index" / "bm25.pkl", b"").open("wb") as stream:
        pickle.dump({"manifest": bm25_record}, stream)
    _write(
        root / "docs" / "evaluation" / f"{run_id}-manifest.json",
        json.dumps(
            {
                "run_id": run_id,
                "native_artifacts_sha256": {"summary.json": RESULT_SHA256},
                "experiment": {
                    "provenance": {
                        "inputs": [
                            {"path": "benchmark/questions.jsonl", "sha256": sha256_file(team)},
                            {"path": "benchmark/generated.jsonl", "sha256": sha256_file(generated)},
                        ]
                    },
                    "indexes": {"dense_manifest": dense, "bm25_manifest": bm25},
                },
            }
        ),
    )
    _write(root / "docs" / "evaluation" / f"{run_id}-summary.json", '{"limitations":[]}\n')
    _write(
        root / ".gitignore",
        "benchmark/generated.jsonl\ndata/index/\nresults/full-matrix-20261021/\n",
    )

    _git(root, "init")
    _git(root, "add", ".")
    _git(
        root,
        "-c",
        "user.name=Archive Test",
        "-c",
        "user.email=archive@example.invalid",
        "commit",
        "-m",
        "release",
    )
    _git(root, "tag", "final/v1")
    return root, _git(root, "rev-parse", "HEAD")


def test_frozen_benchmarks_and_indexes_must_match_the_reported_run(tmp_path: Path):
    root, _ = _release_checkout(tmp_path)
    report = root / "docs/evaluation/full-matrix-20261021-manifest.json"
    dense = json.loads((root / "data/index/chroma.manifest.json").read_text(encoding="utf-8"))
    bm25 = asdict(
        IndexManifest(
            index_type="bm25",
            path="data/index/bm25.pkl",
            corpus_fingerprint="b" * 64,
            n_passages=1,
            n_filings=1,
            built_at="2026-01-01T00:00:00+00:00",
            chunk_budget=1800,
            chunk_overlap=300,
        )
    )
    validate_frozen_identities(root, report, dense, bm25)

    dense["n_passages"] = 2
    with pytest.raises(HandoverError, match="not the dense index manifest"):
        validate_frozen_identities(root, report, dense, bm25)

    dense["n_passages"] = 1
    _write(root / "benchmark/generated.jsonl", '{"id":"changed"}\n')
    with pytest.raises(HandoverError, match="frozen input benchmark/generated.jsonl"):
        validate_frozen_identities(root, report, dense, bm25)


def test_create_binds_complete_archive_to_checked_out_tag_and_is_reproducible(
    tmp_path: Path,
):
    root, commit = _release_checkout(tmp_path)
    first = create_archive(root, "final/v1", "full-matrix-20261021", Path("one.tar.gz"))
    second = create_archive(root, "final/v1", "full-matrix-20261021", Path("two.tar.gz"))

    assert first.read_bytes() == second.read_bytes()
    checksum = first.with_name(first.name + ".sha256")
    assert checksum.read_text(encoding="ascii") == f"{sha256_file(first)}  {first.name}\n"
    manifest = verify_archive(
        first, expected_tag="final/v1", expected_commit=commit, checkout=root
    )
    assert manifest["run_id"] == "full-matrix-20261021"
    assert "benchmark/generated.jsonl" in manifest["files"]
    assert "results/full-matrix-20261021/summary.json" in manifest["files"]
    assert "index-manifests/bm25.manifest.json" in manifest["files"]


def test_create_refuses_a_tag_other_than_checked_out_head(tmp_path: Path):
    root, _ = _release_checkout(tmp_path)
    _write(root / "later.txt")
    _git(root, "add", "later.txt")
    _git(
        root,
        "-c",
        "user.name=Archive Test",
        "-c",
        "user.email=archive@example.invalid",
        "commit",
        "-m",
        "later",
    )
    with pytest.raises(HandoverError, match="check out the tag"):
        create_archive(root, "final/v1", "full-matrix-20261021", Path("bad.tar.gz"))


def test_create_refuses_a_dirty_tracked_file(tmp_path: Path):
    root, _ = _release_checkout(tmp_path)
    _write(root / "handover/README.md", "changed\n")
    with pytest.raises(HandoverError, match="tracked files differ"):
        create_archive(root, "final/v1", "full-matrix-20261021", Path("bad.tar.gz"))


def _rewrite_archive(source: Path, target: Path, edit) -> None:
    """Copy an archive, letting ``edit(relative, data)`` replace member bytes."""

    with tarfile.open(source, "r:gz") as reader:
        members = [(member, reader.extractfile(member).read()) for member in reader.getmembers()]
    with tarfile.open(target, "w:gz") as writer:
        for member, data in members:
            relative = member.name.split("/", 1)[1]
            data = edit(relative, data)
            member.size = len(data)
            writer.addfile(member, io.BytesIO(data))


def test_verify_rejects_native_result_swapped_and_rehashed(tmp_path: Path):
    root, commit = _release_checkout(tmp_path)
    archive = create_archive(root, "final/v1", "full-matrix-20261021", Path("good.tar.gz"))
    target = "results/full-matrix-20261021/summary.json"
    fake = b'{"fabricated": true}\n'

    def edit(relative: str, data: bytes) -> bytes:
        if relative == target:
            return fake
        if relative == "HANDOVER_MANIFEST.json":
            manifest = json.loads(data)
            manifest["files"][target] = {
                "sha256": hashlib.sha256(fake).hexdigest(),
                "size": len(fake),
            }
            return (json.dumps(manifest) + "\n").encode()
        return data

    swapped = tmp_path / "swapped.tar.gz"
    _rewrite_archive(archive, swapped, edit)
    with pytest.raises(HandoverError, match="native_artifacts_sha256"):
        verify_archive(
            swapped, expected_tag="final/v1", expected_commit=commit, checkout=root
        )


def test_verify_binds_the_archived_run_manifest_to_the_checkout(tmp_path: Path):
    root, commit = _release_checkout(tmp_path)
    archive = create_archive(root, "final/v1", "full-matrix-20261021", Path("good.tar.gz"))
    target = "results/full-matrix-20261021/summary.json"
    report_name = "docs/evaluation/full-matrix-20261021-manifest.json"
    fake = b'{"fabricated": true}\n'
    report = json.loads((root / report_name).read_text(encoding="utf-8"))
    report["native_artifacts_sha256"]["summary.json"] = hashlib.sha256(fake).hexdigest()
    fake_report = (json.dumps(report) + "\n").encode()

    def edit(relative: str, data: bytes) -> bytes:
        replacements = {target: fake, report_name: fake_report}
        if relative in replacements:
            return replacements[relative]
        if relative == "HANDOVER_MANIFEST.json":
            manifest = json.loads(data)
            for name, replacement in replacements.items():
                manifest["files"][name] = {
                    "sha256": hashlib.sha256(replacement).hexdigest(),
                    "size": len(replacement),
                }
            return (json.dumps(manifest) + "\n").encode()
        return data

    swapped = tmp_path / "swapped-both.tar.gz"
    _rewrite_archive(archive, swapped, edit)
    with pytest.raises(HandoverError, match="differs from this checkout"):
        verify_archive(
            swapped, expected_tag="final/v1", expected_commit=commit, checkout=root
        )


def test_create_archives_only_the_frozen_benchmark_files(tmp_path: Path):
    root, _ = _release_checkout(tmp_path)
    _write(root / "benchmark/stray.txt", "local only\n")
    archive = create_archive(root, "final/v1", "full-matrix-20261021", Path("good.tar.gz"))
    assert "benchmark/stray.txt" not in verify_archive(archive, checkout=root)["files"]


def test_create_refuses_a_symlinked_input(tmp_path: Path):
    root, _ = _release_checkout(tmp_path)
    secret = _write(tmp_path / "secret.txt", "token\n")
    generated = root / "benchmark/generated.jsonl"
    generated.unlink()
    try:
        generated.symlink_to(secret)
    except OSError as exc:
        pytest.skip(f"this Windows session cannot create symlinks: {exc}")
    with pytest.raises(HandoverError, match="symlinked"):
        create_archive(root, "final/v1", "full-matrix-20261021", Path("bad.tar.gz"))


@pytest.mark.parametrize(
    ("replacement", "message"),
    [(b"short\n", "size mismatch"), (b'{"score": 0}\n', "SHA-256 mismatch")],
)
def test_verify_rejects_truncated_or_changed_files(
    tmp_path: Path, replacement: bytes, message: str
):
    root, _ = _release_checkout(tmp_path)
    archive = create_archive(root, "final/v1", "full-matrix-20261021", Path("good.tar.gz"))
    target = "results/full-matrix-20261021/summary.json"
    changed = tmp_path / "changed.tar.gz"
    _rewrite_archive(
        archive,
        changed,
        lambda relative, data: replacement if relative == target else data,
    )
    with pytest.raises(HandoverError, match=message):
        verify_archive(changed, checkout=root)


def test_verify_rejects_parent_traversal_member(tmp_path: Path):
    root, _ = _release_checkout(tmp_path)
    archive = create_archive(root, "final/v1", "full-matrix-20261021", Path("good.tar.gz"))
    changed = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "r:gz") as source, tarfile.open(changed, "w:gz") as target:
        for member in source.getmembers():
            stream = source.extractfile(member)
            assert stream is not None
            target.addfile(member, stream)
        data = b"escape\n"
        unsafe = tarfile.TarInfo("sec-filing-assistant-handover-final-v1/../escape.txt")
        unsafe.size = len(data)
        target.addfile(unsafe, io.BytesIO(data))
    with pytest.raises(HandoverError, match="unsafe or unexpected"):
        verify_archive(changed, checkout=root)


def test_verify_reports_a_non_object_manifest_cleanly(tmp_path: Path):
    root, _ = _release_checkout(tmp_path)
    archive = create_archive(root, "final/v1", "full-matrix-20261021", Path("good.tar.gz"))
    changed = tmp_path / "bad-manifest.tar.gz"
    _rewrite_archive(
        archive,
        changed,
        lambda relative, data: b"[]\n" if relative == "HANDOVER_MANIFEST.json" else data,
    )
    with pytest.raises(HandoverError, match="must contain a JSON object"):
        verify_archive(changed, checkout=root)


def test_verify_rejects_an_unmanifested_file(tmp_path: Path):
    root, _ = _release_checkout(tmp_path)
    archive = create_archive(root, "final/v1", "full-matrix-20261021", Path("good.tar.gz"))
    changed = tmp_path / "changed.tar.gz"
    with tarfile.open(archive, "r:gz") as source, tarfile.open(changed, "w:gz") as target:
        for member in source.getmembers():
            stream = source.extractfile(member)
            assert stream is not None
            target.addfile(member, stream)
        data = b"not declared\n"
        extra = tarfile.TarInfo("sec-filing-assistant-handover-final-v1/extra.txt")
        extra.size = len(data)
        target.addfile(extra, io.BytesIO(data))

    with pytest.raises(HandoverError, match="file set mismatch"):
        verify_archive(changed)
