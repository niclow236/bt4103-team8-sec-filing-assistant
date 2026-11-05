"""Inputs and settings needed to reproduce a local retrieval measurement."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from collections import Counter
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.config import PROJECT_ROOT
from src.pipeline.chunk import iter_chunks
from src.retrieval import constants
from src.retrieval.records import corpus_fingerprint

from .records import BenchmarkQuestion


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_git(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout if result.returncode == 0 else None


def _same_folder(first: str | Path, second: str | Path) -> bool:
    try:
        return os.path.samefile(first, second)
    except OSError:
        return False


def _git(*args: str) -> str | None:
    # Only the project's own repository counts. A copy without its own .git
    # that sits inside another repository would otherwise read as that one
    # (#50). GIT_CEILING_DIRECTORIES cannot say this for a path with a ":" in
    # it, so the repository git finds is checked for its top level instead. A
    # GIT_DIR set by the caller still wins.
    if "GIT_DIR" not in os.environ:
        top = _run_git("rev-parse", "--show-toplevel")
        # Only git's own newline is taken off, since a folder name may end in a
        # space, and the folders are compared as files: a case-insensitive disk
        # gives one folder spellings that compare unequal as paths.
        if top is None or not _same_folder(top.removesuffix("\n"), PROJECT_ROOT):
            return None
    output = _run_git(*args)
    return output.strip() if output is not None else None


@dataclass(frozen=True)
class RunSource:
    """The code a run measured: its commit, and the release tag on it (#50).

    Read once, when the run starts, since it is the code that process loaded
    that the numbers come from. The commit is kept beside the tag so a run on
    an untagged commit can still be traced, and matched to a tag made later.

    The tag is None when the commit carries none. Edits to committed files
    that are not committed yet add ``-dirty``, since the code that ran is then
    not quite the tagged code. Tags under ``data/`` name releases of the
    evaluation data, not of the code, and are passed over.
    """

    commit: str | None
    tag: str | None

    @classmethod
    def read(cls) -> RunSource:
        # HEAD is read once and the tag looked up on that commit, so a checkout
        # switched by another process in between cannot pair one commit with
        # another's tag. describe takes no --dirty with a commit named, so the
        # edits are asked of status, which refreshes the index as --dirty does.
        commit = _git("rev-parse", "HEAD")
        if commit is None:
            return cls(None, None)
        tag = _git("describe", "--tags", "--exact-match", "--exclude", "data/*", commit)
        if tag is not None and _git("status", "--porcelain", "--untracked-files=no"):
            tag += "-dirty"
        return cls(commit, tag)

    @classmethod
    def frozen(cls, provenance: Mapping[str, Any]) -> RunSource:
        """The source a run froze in its config; runs from before #50 have no tag."""
        return cls(provenance.get("source_commit"), provenance.get("source_tag"))

    def to_dict(self) -> dict[str, str | None]:
        return {"source_commit": self.commit, "source_tag": self.tag}


def corpus_records_sha256(chunks: Sequence[dict]) -> str:
    """Fingerprint text and metadata, including the gold evidence's scope."""
    return hashlib.sha256(json.dumps(
        sorted(chunks, key=lambda row: row["chunk_id"]), sort_keys=True,
    ).encode()).hexdigest()


def question_records_sha256(questions: Sequence[BenchmarkQuestion]) -> str:
    """Fingerprint the benchmark questions, in order, as a run records them."""
    return hashlib.sha256(
        json.dumps([asdict(q) for q in questions], sort_keys=True).encode()
    ).hexdigest()


def record_provenance(
    inputs: Sequence[Path], questions: Sequence[BenchmarkQuestion], processed_dir: Path,
    *, chunks: Sequence[dict] | None = None,
) -> dict[str, Any]:
    """Record file digests, corpus content and the actual installed settings.

    No environment variables or credentials are read. The source-file hashes
    distinguish dirty source trees even when the HEAD commit is unchanged.
    """
    if chunks is None:
        chunks = list(iter_chunks(processed_dir=processed_dir))
    settings = {
        name: getattr(constants, name) for name in (
            "EMBED_MODEL", "EMBED_DIMENSIONS", "EMBED_MAX_TOKENS",
            "EMBED_NORMALIZE", "QUERY_PREFIX", "PASSAGE_PREFIX", "CONTEXT_HEADER",
            "DISTANCE_METRIC", "BM25_K1", "BM25_B", "CANDIDATE_K", "RRF_K",
            "FUSION_WEIGHTS", "FIGURE_FUSION_WEIGHTS", "MIN_BM25_SCORE",
            "MIN_DENSE_SCORE", "MIN_FUSED_SCORE", "TABLE_BOOST",
        )
    }
    packages = {}
    for package in ("torch", "sentence-transformers", "transformers", "tokenizers",
                    "chromadb", "rank-bm25", "numpy", "filelock"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    return {
        **RunSource.read().to_dict(),
        "source_files_sha256": {
            path.relative_to(PROJECT_ROOT).as_posix(): sha256(path)
            for path in sorted((PROJECT_ROOT / "src").rglob("*.py"))
        },
        "inputs": [{"path": str(path), "sha256": sha256(path)} for path in inputs],
        "questions": len(questions),
        "by_benchmark": dict(sorted(Counter(q.benchmark for q in questions).items())),
        "question_records_sha256": question_records_sha256(questions),
        "corpus": {
            "processed_dir": str(processed_dir),
            "fingerprint": corpus_fingerprint(chunks),
            "records_sha256": corpus_records_sha256(chunks),
            "n_passages": len(chunks),
            "n_filings": len({chunk["accession_no"] for chunk in chunks}),
            "chunk_settings": sorted({
                (chunk.get("chunk_budget"), chunk.get("chunk_overlap")) for chunk in chunks
            }, key=str),
        },
        "retrieval_settings": settings,
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "machine": platform.machine(), "packages": packages},
    }
