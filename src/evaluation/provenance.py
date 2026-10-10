"""Inputs and settings needed to reproduce a local retrieval measurement."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from collections import Counter
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Sequence

from src.config import PROJECT_ROOT
from src.pipeline.chunk import iter_chunks
from src.retrieval import constants
from src.retrieval.records import corpus_fingerprint

from .records import BenchmarkQuestion


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str | None:
    result = subprocess.run(
        ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def record_provenance(
    inputs: Sequence[Path], questions: Sequence[BenchmarkQuestion], processed_dir: Path,
) -> dict[str, Any]:
    """Record file digests, corpus content and the actual installed settings.

    No environment variables or credentials are read. The source-file hashes
    distinguish dirty source trees even when the HEAD commit is unchanged.
    """
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
                    "chromadb", "rank-bm25", "numpy"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    return {
        "source_commit": _git("rev-parse", "HEAD"),
        "source_files_sha256": {
            path.relative_to(PROJECT_ROOT).as_posix(): sha256(path)
            for path in sorted((PROJECT_ROOT / "src").rglob("*.py"))
        },
        "inputs": [{"path": str(path), "sha256": sha256(path)} for path in inputs],
        "questions": len(questions),
        "by_source": dict(sorted(Counter(q.source for q in questions).items())),
        "question_records_sha256": hashlib.sha256(
            json.dumps([asdict(q) for q in questions], sort_keys=True).encode()
        ).hexdigest(),
        "corpus": {
            "processed_dir": str(processed_dir),
            "fingerprint": corpus_fingerprint(chunks),
            "records_sha256": hashlib.sha256(
                json.dumps(sorted(chunks, key=lambda row: row["chunk_id"]),
                           sort_keys=True).encode()
            ).hexdigest(),
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
