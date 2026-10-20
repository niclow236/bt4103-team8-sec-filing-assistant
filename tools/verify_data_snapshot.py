"""Verify an extracted frozen data snapshot, using only Python's standard library.

Run from the snapshot/project root: python verify_data_snapshot.py
Or from the repository: python tools/verify_data_snapshot.py --root PATH
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


def verify(root: Path) -> dict:
    root = root.resolve()
    manifest = json.loads((root / "DATA_SNAPSHOT_MANIFEST.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "sec-evaluation-data-v1":
        raise ValueError("unsupported snapshot manifest schema")
    for name, expected in manifest["files"].items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe manifest path: {name}")
        path = root.joinpath(*relative.parts)
        if not path.is_file() or path.is_symlink() or root not in path.resolve().parents:
            raise ValueError(f"missing or unsafe snapshot file: {name}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if path.stat().st_size != expected["bytes"] or digest.hexdigest() != expected["sha256"]:
            raise ValueError(f"snapshot file differs: {name}")
    processed = {p.relative_to(root).as_posix() for p in (root / "data/processed").rglob("*.json")}
    expected_processed = {name for name in manifest["files"] if name.startswith("data/processed/")}
    if processed != expected_processed:
        raise ValueError("processed corpus contains extra/missing JSON files; extract into a fresh clone")
    chunks = []
    for name in sorted(processed):
        chunks.extend(json.loads((root / name).read_text(encoding="utf-8"))["chunks"])
    chunk_ids = {chunk["chunk_id"] for chunk in chunks}
    if len(chunks) != manifest["corpus"]["passages"] or len(chunk_ids) != len(chunks):
        raise ValueError("corpus passage count or unique IDs differ")
    if len(processed) != manifest["corpus"]["filings"]:
        raise ValueError("filing count differs")
    questions = []
    for name, count in manifest["question_files"].items():
        rows = [json.loads(line) for line in (root / name).read_text(encoding="utf-8").splitlines()]
        if len(rows) != count:
            raise ValueError(f"question count differs: {name}")
        questions.extend(rows)
    if len({q["question_id"] for q in questions}) != len(questions):
        raise ValueError("duplicate question IDs across benchmark files")
    for q in questions:
        evidence = set(q["supporting_chunk_ids"]) | set(q.get("hard_negative_chunk_ids", []))
        if not evidence <= chunk_ids:
            raise ValueError(f"question evidence is absent from this corpus: {q['question_id']}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    try:
        manifest = verify(args.root)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Verification failed: {error}\n")
    counts = manifest["question_files"]
    print(f"Verified {manifest['snapshot_id']}: {sum(counts.values()):,} questions, "
          f"{manifest['corpus']['passages']:,} passages, {manifest['corpus']['filings']} filings; "
          f"all {len(manifest['files'])} file hashes match.")


if __name__ == "__main__":
    main()
