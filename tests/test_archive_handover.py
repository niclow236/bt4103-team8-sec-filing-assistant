from __future__ import annotations

import io
import json
import pickle
import subprocess
import tarfile
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


def _write(path: Path, content: str | bytes = "x\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")
    return path


def _report_manifest(path: Path, run_file: Path) -> Path:
    payload = {"native_artifacts_sha256": {run_file.name: sha256_file(run_file)}}
    _write(path, json.dumps(payload))
    return path


def test_native_results_must_match_the_recorded_set_and_hashes(tmp_path: Path):
    run = tmp_path / "results" / "run"
    result = _write(run / "summary.json", '{"score": 1}\n')
    report = _report_manifest(tmp_path / "report.json", result)

    assert validate_native_results(run, report) == {"summary.json": sha256_file(result)}

    # Runtime debris and derived rescoring output are deliberately outside the
    # native set selected by the report generator.
    _write(run / "active.lock")
    _write(run / "worker-process-status.json")
    _write(run / "rescored" / "summary.json")
    assert validate_native_results(run, report) == {"summary.json": sha256_file(result)}

    _write(run / "unrecorded.json", "{}\n")
    with pytest.raises(HandoverError, match="not recorded by report manifest"):
        validate_native_results(run, report)
    (run / "unrecorded.json").unlink()

    _write(result, '{"score": 0}\n')
    with pytest.raises(HandoverError, match="SHA-256 mismatch"):
        validate_native_results(run, report)


def test_missing_native_run_explains_that_it_must_come_from_measurement_machine(
    tmp_path: Path,
):
    report = _write(
        tmp_path / "report.json",
        json.dumps({"native_artifacts_sha256": {"summary.json": "0" * 64}}),
    )
    with pytest.raises(HandoverError, match="measurement machine"):
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
    native = _write(root / "results" / run_id / "summary.json", '{"score": 1}\n')
    team = _write(root / "benchmark" / "questions.jsonl", '{"id":"team"}\n')
    generated = _write(root / "benchmark" / "generated.jsonl", '{"id":"xbrl"}\n')
    _write(root / "benchmark" / "schema.md", "# Schema\n")
    _write(root / "docs" / "full-ablation.md", "# Study\n")
    _write(root / "handover" / "README.md", "# Handover\n")
    _write(root / "results" / "README.md", "# Limitations\n")
    dense = {"index_type": "dense", "n_passages": 1}
    bm25 = {"index_type": "bm25", "n_passages": 1}
    _write(root / "data" / "index" / "chroma.manifest.json", json.dumps(dense))
    with _write(root / "data" / "index" / "bm25.pkl", b"").open("wb") as stream:
        pickle.dump({"manifest": bm25}, stream)
    _write(
        root / "docs" / "evaluation" / f"{run_id}-manifest.json",
        json.dumps(
            {
                "run_id": run_id,
                "native_artifacts_sha256": {"summary.json": sha256_file(native)},
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
    bm25 = {"index_type": "bm25", "n_passages": 1}
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
    assert first.with_name(first.name + ".sha256").is_file()
    manifest = verify_archive(first, expected_tag="final/v1", expected_commit=commit)
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
