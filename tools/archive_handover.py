"""Build and verify the final handover release archive.

The native evaluation outputs and indexes are deliberately ignored by Git.  A
source tag therefore does not, by itself, preserve the evidence behind the
reported numbers.  This command is the release gate that binds those files to a
tag and refuses an incomplete or changed result directory.

Create an archive from a checkout of the release tag::

    python tools/archive_handover.py create --tag final/v1

Verify a downloaded archive without importing the project or its dependencies::

    python tools/archive_handover.py verify dist/sec-filing-assistant-handover-final-v1.tar.gz
"""

from __future__ import annotations

import argparse
import dataclasses
import gzip
import hashlib
import io
import json
import os
import pickle
import re
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    # ``python tools/archive_handover.py`` otherwise exposes tools/, not the
    # repository root, while unpickling the BM25 manifest imports src.*.
    sys.path.insert(0, str(ROOT))
DEFAULT_RUN_ID = "full-matrix-20261021"
MANIFEST_NAME = "HANDOVER_MANIFEST.json"
SCHEMA_VERSION = 1


class HandoverError(RuntimeError):
    """A release invariant was not satisfied."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str, root: Path = ROOT) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise HandoverError(f"git {' '.join(args)} failed: {detail}")
    return completed.stdout.strip()


def resolve_release(tag: str, root: Path = ROOT) -> tuple[str, str]:
    """Return (commit, commit timestamp), requiring TAG at checked-out HEAD."""

    if not tag.strip():
        raise HandoverError("release tag must not be empty")
    commit = _git("rev-parse", "--verify", f"refs/tags/{tag}^{{commit}}", root=root)
    head = _git("rev-parse", "HEAD", root=root)
    if head != commit:
        raise HandoverError(
            f"release tag {tag!r} resolves to {commit}, but HEAD is {head}; "
            f"check out the tag before archiving"
        )
    dirty = _git("status", "--porcelain", "--untracked-files=no", root=root)
    if dirty:
        raise HandoverError(
            "tracked files differ from the release tag; commit or restore them before archiving"
        )
    timestamp = _git("show", "-s", "--format=%cI", commit, root=root)
    return commit, timestamp


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HandoverError(f"cannot read JSON from {path}: {exc}") from exc


def _native_hashes(report_manifest: Path) -> dict[str, str]:
    payload = _read_json(report_manifest)
    hashes = payload.get("native_artifacts_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise HandoverError(
            f"{report_manifest} has no non-empty native_artifacts_sha256 object"
        )
    result: dict[str, str] = {}
    for relative, digest in hashes.items():
        candidate = PurePosixPath(str(relative))
        if candidate.is_absolute() or ".." in candidate.parts:
            raise HandoverError(f"unsafe native result path in manifest: {relative!r}")
        if not re.fullmatch(r"[0-9a-f]{64}", str(digest)):
            raise HandoverError(f"invalid SHA-256 for native result {relative!r}")
        result[candidate.as_posix()] = str(digest)
    return result


def _expected_inputs(report: Mapping[str, Any]) -> dict[str, str]:
    try:
        inputs = report["experiment"]["provenance"]["inputs"]
    except (KeyError, TypeError) as exc:
        raise HandoverError("report manifest has no experiment.provenance.inputs") from exc
    result: dict[str, str] = {}
    if not isinstance(inputs, list):
        raise HandoverError("report manifest provenance inputs must be a list")
    for record in inputs:
        if not isinstance(record, dict):
            raise HandoverError("invalid report manifest provenance input")
        path, digest = record.get("path"), record.get("sha256")
        if not isinstance(path, str) or not re.fullmatch(r"[0-9a-f]{64}", str(digest)):
            raise HandoverError("invalid path or SHA-256 in report manifest provenance input")
        result[path] = str(digest)
    required = {"benchmark/questions.jsonl", "benchmark/generated.jsonl"}
    if not required <= set(result):
        raise HandoverError("report manifest does not identify both frozen benchmark inputs")
    return result


def _named_objects(value: Any, name: str) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key == name:
                if not isinstance(child, dict):
                    raise HandoverError(f"report manifest {name} must be an object")
                found.append(child)
            found.extend(_named_objects(child, name))
    elif isinstance(value, list):
        for child in value:
            found.extend(_named_objects(child, name))
    return found


def _expected_index_manifest(report: Mapping[str, Any], name: str) -> dict[str, Any]:
    candidates = _named_objects(report, name)
    unique = {json.dumps(candidate, sort_keys=True) for candidate in candidates}
    if not unique:
        raise HandoverError(f"report manifest contains no {name}")
    if len(unique) != 1:
        raise HandoverError(f"report manifest contains conflicting {name} values")
    return json.loads(next(iter(unique)))


def validate_frozen_identities(
    root: Path,
    report_manifest: Path,
    dense_manifest: Mapping[str, Any],
    bm25_manifest: Mapping[str, Any],
) -> None:
    """Match benchmark and index identities to the reported measurement."""

    report = _read_json(report_manifest)
    if not isinstance(report, dict):
        raise HandoverError(f"{report_manifest} must contain a JSON object")
    for relative, expected in _expected_inputs(report).items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise HandoverError(
                f"frozen input {relative} does not match the SHA-256 recorded by {report_manifest}"
            )
    expected_dense = _expected_index_manifest(report, "dense_manifest")
    expected_bm25 = _expected_index_manifest(report, "bm25_manifest")
    if dict(dense_manifest) != expected_dense:
        raise HandoverError(
            "data/index/chroma.manifest.json is not the dense index manifest used by "
            f"the reported {report.get('run_id', 'run')}"
        )
    if dict(bm25_manifest) != expected_bm25:
        raise HandoverError(
            "data/index/bm25.pkl is not the BM25 index manifest used by "
            f"the reported {report.get('run_id', 'run')}"
        )


def validate_native_results(
    run_dir: Path, report_manifest: Path
) -> dict[str, str]:
    """Require every recorded native artifact and reject unrecorded files."""

    expected = _native_hashes(report_manifest)
    if not run_dir.is_dir():
        raise HandoverError(
            f"missing native run directory {run_dir}; copy the frozen run from the "
            "measurement machine before creating the release"
        )
    actual_files = {
        path.relative_to(run_dir).as_posix(): path
        for path in run_dir.rglob("*")
        if path.is_file() and _is_native_result(path.relative_to(run_dir))
    }
    missing = sorted(set(expected) - set(actual_files))
    extra = sorted(set(actual_files) - set(expected))
    changed = sorted(
        relative
        for relative in set(expected) & set(actual_files)
        if sha256_file(actual_files[relative]) != expected[relative]
    )
    problems = []
    if missing:
        problems.append("missing: " + ", ".join(missing))
    if extra:
        problems.append("not recorded by report manifest: " + ", ".join(extra))
    if changed:
        problems.append("SHA-256 mismatch: " + ", ".join(changed))
    if problems:
        raise HandoverError(
            f"native result validation failed for {run_dir}:\n  " + "\n  ".join(problems)
        )
    return expected


def _is_native_result(relative: Path) -> bool:
    """Match the native-file selection used by tools/report_full_matrix.py."""

    return (
        relative.suffix != ".lock"
        and not relative.name.endswith("process-status.json")
        and "rescored" not in relative.parts
    )


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {field.name: _jsonable(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, datetime):
        return value.isoformat()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise HandoverError(f"BM25 manifest contains unsupported value {type(value).__name__}")


def read_bm25_manifest(index_file: Path) -> dict[str, Any]:
    """Export only the manifest embedded in this project's trusted BM25 pickle."""

    if not index_file.is_file():
        raise HandoverError(f"missing BM25 index (and embedded manifest): {index_file}")
    try:
        with index_file.open("rb") as stream:
            payload = pickle.load(stream)  # noqa: S301 - project-created, local release input
    except (OSError, pickle.UnpicklingError, EOFError, AttributeError, ImportError) as exc:
        raise HandoverError(f"cannot read BM25 manifest from {index_file}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("manifest") is None:
        raise HandoverError(f"{index_file} carries no BM25 manifest")
    manifest = _jsonable(payload["manifest"])
    if not isinstance(manifest, dict):
        raise HandoverError(f"{index_file} has an invalid BM25 manifest")
    return manifest


def _require_files(root: Path, relatives: Iterable[str], purpose: str) -> list[Path]:
    files = [root / relative for relative in relatives]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise HandoverError(f"missing {purpose}: " + ", ".join(missing))
    return files


def _archive_inputs(root: Path, run_id: str) -> tuple[dict[str, Path], dict[str, bytes]]:
    """Collect source paths and generated bytes under their archive names."""

    report_manifest = root / "docs" / "evaluation" / f"{run_id}-manifest.json"
    report_summary = root / "docs" / "evaluation" / f"{run_id}-summary.json"
    required = _require_files(
        root,
        [
            "benchmark/questions.jsonl",
            "benchmark/generated.jsonl",
            "benchmark/schema.md",
            f"docs/evaluation/{run_id}-manifest.json",
            f"docs/evaluation/{run_id}-summary.json",
            "docs/full-ablation.md",
            "handover/README.md",
            "results/README.md",
            "data/index/chroma.manifest.json",
        ],
        "handover input",
    )
    run_dir = root / "results" / run_id
    validate_native_results(run_dir, report_manifest)

    paths: dict[str, Path] = {}
    for path in (root / "benchmark").rglob("*"):
        if path.is_file():
            paths[path.relative_to(root).as_posix()] = path
    for path in run_dir.rglob("*"):
        if path.is_file() and _is_native_result(path.relative_to(run_dir)):
            paths[path.relative_to(root).as_posix()] = path
    for path in required[3:8]:
        paths[path.relative_to(root).as_posix()] = path
    paths["index-manifests/chroma.manifest.json"] = root / "data/index/chroma.manifest.json"

    dense = _read_json(root / "data/index/chroma.manifest.json")
    if not isinstance(dense, dict):
        raise HandoverError("data/index/chroma.manifest.json must contain a JSON object")
    bm25 = read_bm25_manifest(root / "data/index/bm25.pkl")
    validate_frozen_identities(root, report_manifest, dense, bm25)
    generated = {
        "index-manifests/bm25.manifest.json": (
            json.dumps(bm25, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        ).encode("utf-8")
    }
    # Reading these here gives a clearer error if one became invalid JSON.
    _read_json(report_summary)
    return paths, generated


def _safe_tag(tag: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", tag).strip("-.")
    if not value:
        raise HandoverError(f"release tag {tag!r} cannot form an archive name")
    return value


def _tar_info(name: str, size: int, mtime: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name=name)
    info.size = size
    info.mtime = mtime
    info.mode = 0o644
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    return info


def create_archive(root: Path, tag: str, run_id: str, output: Path | None) -> Path:
    commit, commit_timestamp = resolve_release(tag, root)
    paths, generated = _archive_inputs(root, run_id)
    prefix = f"sec-filing-assistant-handover-{_safe_tag(tag)}"
    if output is None:
        output = root / "dist" / f"{prefix}.tar.gz"
    elif not output.is_absolute():
        output = root / output

    entries: dict[str, bytes | Path] = {**paths, **generated}
    files = {}
    for relative, source in sorted(entries.items()):
        if isinstance(source, Path):
            files[relative] = {"sha256": sha256_file(source), "size": source.stat().st_size}
        else:
            files[relative] = {"sha256": sha256_bytes(source), "size": len(source)}
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "release_tag": tag,
        "release_commit": commit,
        "release_commit_timestamp": commit_timestamp,
        "run_id": run_id,
        "contents": {
            "benchmark": "benchmark/ (handwritten and generated frozen questions)",
            "results": f"results/{run_id}/ (hash-verified native run)",
            "index_manifests": "dense and BM25 build identities; index binaries are rebuilt",
            "limitations": "results/README.md and the run summary's limitations array",
        },
        "files": files,
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    entries[MANIFEST_NAME] = manifest_bytes

    try:
        when = datetime.fromisoformat(commit_timestamp.replace("Z", "+00:00"))
    except ValueError as exc:
        raise HandoverError(f"invalid Git commit timestamp {commit_timestamp!r}") from exc
    mtime = int(when.astimezone(timezone.utc).timestamp())
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
                for relative, source in sorted(entries.items()):
                    data = source.read_bytes() if isinstance(source, Path) else source
                    name = f"{prefix}/{relative}"
                    archive.addfile(_tar_info(name, len(data), mtime), io.BytesIO(data))

    checksum = sha256_file(output)
    checksum_file = output.with_name(output.name + ".sha256")
    checksum_file.write_text(f"{checksum}  {output.name}\n", encoding="ascii", newline="\n")
    verify_archive(output, expected_tag=tag, expected_commit=commit)
    return output


def _member_relative(member: tarfile.TarInfo, prefix: str) -> str:
    path = PurePosixPath(member.name)
    if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != prefix:
        raise HandoverError(f"unsafe or unexpected archive member: {member.name!r}")
    if len(path.parts) < 2:
        raise HandoverError(f"archive member has no path below its top-level folder: {member.name!r}")
    return PurePosixPath(*path.parts[1:]).as_posix()


def verify_archive(
    archive_path: Path,
    *,
    expected_tag: str | None = None,
    expected_commit: str | None = None,
) -> dict[str, Any]:
    """Verify layout, release identity, exact file set, sizes, and SHA-256 hashes."""

    try:
        archive = tarfile.open(archive_path, mode="r:gz")
    except (OSError, tarfile.TarError) as exc:
        raise HandoverError(f"cannot open handover archive {archive_path}: {exc}") from exc
    with archive:
        members = archive.getmembers()
        if not members:
            raise HandoverError("handover archive is empty")
        prefixes = {PurePosixPath(member.name).parts[0] for member in members if member.name}
        if len(prefixes) != 1:
            raise HandoverError("handover archive must have exactly one top-level folder")
        prefix = next(iter(prefixes))
        by_relative: dict[str, tarfile.TarInfo] = {}
        for member in members:
            if not member.isfile():
                raise HandoverError(f"handover archive contains a non-file member: {member.name}")
            relative = _member_relative(member, prefix)
            if relative in by_relative:
                raise HandoverError(f"duplicate archive member: {relative}")
            by_relative[relative] = member
        if MANIFEST_NAME not in by_relative:
            raise HandoverError(f"handover archive has no {MANIFEST_NAME}")
        manifest_stream = archive.extractfile(by_relative[MANIFEST_NAME])
        assert manifest_stream is not None
        try:
            manifest = json.loads(manifest_stream.read().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HandoverError(f"invalid {MANIFEST_NAME}: {exc}") from exc

        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise HandoverError(f"unsupported handover schema {manifest.get('schema_version')!r}")
        if expected_tag is not None and manifest.get("release_tag") != expected_tag:
            raise HandoverError(
                f"archive release tag is {manifest.get('release_tag')!r}, expected {expected_tag!r}"
            )
        if expected_commit is not None and manifest.get("release_commit") != expected_commit:
            raise HandoverError(
                f"archive release commit is {manifest.get('release_commit')!r}, "
                f"expected {expected_commit!r}"
            )
        files = manifest.get("files")
        if not isinstance(files, dict) or not files:
            raise HandoverError(f"{MANIFEST_NAME} has no non-empty files object")
        actual_names = set(by_relative) - {MANIFEST_NAME}
        expected_names = set(files)
        if actual_names != expected_names:
            missing = sorted(expected_names - actual_names)
            extra = sorted(actual_names - expected_names)
            raise HandoverError(f"archive file set mismatch; missing={missing}, extra={extra}")
        required = {
            "benchmark/questions.jsonl",
            "benchmark/generated.jsonl",
            "index-manifests/chroma.manifest.json",
            "index-manifests/bm25.manifest.json",
            "results/README.md",
        }
        absent = sorted(required - actual_names)
        run_id = manifest.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise HandoverError(f"{MANIFEST_NAME} has no run_id")
        if not any(name.startswith(f"results/{run_id}/") for name in actual_names):
            absent.append(f"results/{run_id}/")
        if absent:
            raise HandoverError("archive is missing required handover content: " + ", ".join(absent))
        for relative in sorted(expected_names):
            record = files[relative]
            if not isinstance(record, dict):
                raise HandoverError(f"invalid file record for {relative}")
            stream = archive.extractfile(by_relative[relative])
            assert stream is not None
            data = stream.read()
            if len(data) != record.get("size"):
                raise HandoverError(f"size mismatch for {relative}")
            if sha256_bytes(data) != record.get("sha256"):
                raise HandoverError(f"SHA-256 mismatch for {relative}")
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create", help="validate inputs and build a tagged archive")
    create.add_argument("--tag", required=True, help="existing release tag checked out at HEAD")
    create.add_argument("--run-id", default=DEFAULT_RUN_ID)
    create.add_argument("--output", type=Path, help="archive path (default: dist/<tag>.tar.gz)")
    verify = subparsers.add_parser("verify", help="verify a downloaded archive")
    verify.add_argument("archive", type=Path)
    verify.add_argument("--tag", help="also require this release tag")
    verify.add_argument("--commit", help="also require this release commit")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "create":
            archive = create_archive(ROOT, args.tag, args.run_id, args.output)
            print(f"Created {archive}")
            print(f"Created {archive.with_name(archive.name + '.sha256')}")
        else:
            manifest = verify_archive(
                args.archive, expected_tag=args.tag, expected_commit=args.commit
            )
            print(
                f"Verified {args.archive}: tag {manifest['release_tag']}, "
                f"commit {manifest['release_commit']}, run {manifest['run_id']}"
            )
    except HandoverError as exc:
        print(f"ERROR: {exc}", file=os.sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
