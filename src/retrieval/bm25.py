"""BM25 keyword retrieval over the processed filing passages."""

from __future__ import annotations

import hashlib
import pickle
import re
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from ..config import DATA_DIR
from ..pipeline.chunk import iter_chunks
from .records import IndexManifest, Query, RetrievedPassage

INDEX_PATH = DATA_DIR / "index" / "bm25.pkl"
_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Tokenize text consistently for both indexing and querying."""
    return _TOKEN.findall(text.lower())


def _corpus_fingerprint(chunks: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(str(chunk["chunk_id"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(chunk["text"]).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _matches(chunk: Mapping[str, Any], query: Query) -> bool:
    """Apply all hard filters before a BM25 score is calculated."""
    if query.tickers and str(chunk["ticker"]).upper() not in {
        ticker.upper() for ticker in query.tickers
    }:
        return False
    if query.fiscal_years and chunk.get("fiscal_year") not in query.fiscal_years:
        return False
    if query.items and str(chunk.get("item") or "").upper() not in {
        item.upper() for item in query.items
    }:
        return False
    if query.content_type and chunk.get("content_type", "prose") != query.content_type:
        return False
    if query.key_items_only and not chunk.get("is_key_section", False):
        return False
    return True


class BM25Retriever:
    """Build, persist, and query a BM25 index of filing passages."""

    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self.chunks = chunks
        self._tokens = [tokenize(chunk["text"]) for chunk in chunks]
        self._bm25 = BM25Okapi(self._tokens)

    @classmethod
    def build(
        cls,
        processed_dir: Path | None = None,
        index_path: Path = INDEX_PATH,
    ) -> BM25Retriever:
        """Build an index from ``iter_chunks`` and serialize it to disk."""
        chunks = list(iter_chunks(processed_dir=processed_dir)) if processed_dir else list(iter_chunks())
        if not chunks:
            raise ValueError("Cannot build a BM25 index: no processed passages were found")

        retriever = cls(chunks)
        manifest = IndexManifest(
            index_type="bm25",
            path=str(index_path),
            corpus_fingerprint=_corpus_fingerprint(chunks),
            n_passages=len(chunks),
            n_filings=len({chunk["accession_no"] for chunk in chunks}),
            built_at=datetime.now(timezone.utc).isoformat(),
        )
        index_path.parent.mkdir(parents=True, exist_ok=True)
        with index_path.open("wb") as file:
            pickle.dump({"chunks": chunks, "manifest": manifest}, file)
        return retriever

    @classmethod
    def load(cls, index_path: Path = INDEX_PATH) -> BM25Retriever:
        """Load a previously serialized BM25 index."""
        with index_path.open("rb") as file:
            payload = pickle.load(file)
        return cls(payload["chunks"])

    def search(self, query: Query) -> list[RetrievedPassage]:
        """Return the highest-scoring passages matching the query filters."""
        if query.top_k <= 0:
            return []

        candidates = [
            (position, chunk)
            for position, chunk in enumerate(self.chunks)
            if _matches(chunk, query)
        ]
        if not candidates:
            return []

        candidate_tokens = [self._tokens[position] for position, _ in candidates]
        scores = BM25Okapi(candidate_tokens).get_scores(tokenize(query.text))
        ranked = sorted(
            zip(scores, (chunk for _, chunk in candidates)),
            key=lambda result: result[0],
            reverse=True,
        )[:query.top_k]
        return [
            RetrievedPassage.from_chunk(
                chunk,
                score=float(score),
                rank=rank,
                retriever="bm25",
            )
            for rank, (score, chunk) in enumerate(ranked, start=1)
        ]


def build_index(
    processed_dir: Path | None = None,
    index_path: Path = INDEX_PATH,
) -> BM25Retriever:
    """Convenience wrapper for building the project BM25 index."""
    return BM25Retriever.build(processed_dir=processed_dir, index_path=index_path)


def load_index(index_path: Path = INDEX_PATH) -> BM25Retriever:
    """Convenience wrapper for loading the project BM25 index."""
    return BM25Retriever.load(index_path=index_path)