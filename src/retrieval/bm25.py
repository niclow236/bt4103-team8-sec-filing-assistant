"""BM25 keyword retrieval over the processed filing passages.

The sparse baseline the dense and hybrid rows are measured against, so what
matters most here is that its numbers mean the same thing from one query to the
next. Two decisions follow from that.

The index is fitted once, over the whole corpus, and never refitted per query.
BM25 scores a passage against the statistics of the collection it sits in -- how
rare a term is, and how long the average passage runs -- so fitting over the
rows a filter happened to admit makes those statistics a property of the filter.
Pin a query to one ticker and that company's own name appears in nearly every
candidate, which floors its IDF; rank_bm25 then substitutes ``epsilon *
average_idf`` for the negative value and the term the user actually asked about
stops discriminating. The scale moves with the filter too, which would leave
``MIN_BM25_SCORE`` uncalibratable and the ablation's rows incomparable. So the
filter chooses which scores are returned, never which scores are computed.

The ordering, the floor, the rank numbering and the projection into
``RetrievedPassage`` all come from ``base.py``, as does the metadata filter.
This module contributes tokenization, the fit, and the index on disk; anything
the four retrievers share is imported rather than restated, because a
comparison whose rows differ in their tie-break is not measuring retrieval.
"""

from __future__ import annotations

import pickle
import re
from collections.abc import Iterable, Iterator, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from src.pipeline.chunk import iter_chunks, resolve_chunk_settings

from .base import matches, rank, resolve_k
from .constants import BM25, BM25_B, BM25_INDEX_FILE, BM25_K1, MIN_BM25_SCORE
from .records import (
    IndexManifest,
    Query,
    RetrievedPassage,
    corpus_fingerprint,
    manifest_path,
)

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Tokenize text consistently for both indexing and querying.

    The same function on both sides is the whole requirement: a query token that
    was produced by different rules than the index tokens matches nothing, and
    does so silently.
    """
    return _TOKEN.findall(text.lower())


def _read_corpus(processed_dir: Path | None) -> Iterator[dict]:
    """Every processed passage, from the default corpus or a given directory.

    ``iter_chunks`` defaults ``processed_dir`` to ``PROCESSED_DIR`` rather than
    to None, so None has to become "no argument" here instead of being passed
    straight through.
    """
    if processed_dir is None:
        return iter_chunks()
    return iter_chunks(processed_dir=processed_dir)


class BM25Retriever:
    """Build, persist, and query a BM25 index of filing passages.

    Satisfies ``base.Retriever`` structurally: a ``name`` and a
    ``search(query, k=None)``, so the ablation harness and any wrapping
    retriever hold this the same way they hold the dense one.
    """

    name = BM25

    def __init__(
        self,
        chunks: Iterable[Mapping[str, Any]],
        manifest: IndexManifest | None = None,
    ) -> None:
        self.chunks = [dict(chunk) for chunk in chunks]
        self.manifest = manifest
        self._tokens = [tokenize(chunk["text"]) for chunk in self.chunks]
        # rank_bm25 divides by the corpus size to get the average document
        # length, so an empty corpus raises rather than building an index that
        # matches nothing. build() refuses one; a hand-made or truncated index
        # file can still produce one, and search() answers it with no results.
        self._bm25 = BM25Okapi(self._tokens, k1=BM25_K1, b=BM25_B) if self._tokens else None

    @classmethod
    def build(
        cls,
        processed_dir: Path | None = None,
        index_path: Path = BM25_INDEX_FILE,
    ) -> BM25Retriever:
        """Build an index from ``iter_chunks`` and serialize it to disk.

        The chunker settings the manifest records are measured off the corpus,
        which stores what it was cut with. ``resolve_chunk_settings`` holds the
        rule for a corpus that predates that or was cut two different ways, so
        this index and the dense one report both cases identically.
        """
        chunks = list(_read_corpus(processed_dir))
        if not chunks:
            raise ValueError("Cannot build a BM25 index: no processed passages were found")

        chunk_budget, chunk_overlap, note = resolve_chunk_settings(
            {(chunk.get("chunk_budget"), chunk.get("chunk_overlap")) for chunk in chunks}
        )
        if note:
            print(f"  NOTE  {note}")

        manifest = IndexManifest(
            index_type=BM25,
            path=manifest_path(index_path),
            corpus_fingerprint=corpus_fingerprint(chunks),
            n_passages=len(chunks),
            n_filings=len({chunk["accession_no"] for chunk in chunks}),
            built_at=datetime.now(timezone.utc).isoformat(),
            chunk_budget=chunk_budget,
            chunk_overlap=chunk_overlap,
        )
        index_path.parent.mkdir(parents=True, exist_ok=True)
        with index_path.open("wb") as file:
            pickle.dump({"chunks": chunks, "manifest": manifest}, file)
        return cls(chunks, manifest=manifest)

    @classmethod
    def load(
        cls,
        index_path: Path = BM25_INDEX_FILE,
        processed_dir: Path | None = None,
        verify: bool = True,
    ) -> BM25Retriever:
        """Load a serialized index, refusing one the corpus has moved past.

        The check is the reason ``build`` writes a manifest at all. Re-chunking
        without re-indexing leaves an index that still answers, from passages
        whose ids no longer exist -- confidently, and with nothing in the output
        to say so. Comparing fingerprints turns that into an error naming both
        sides.

        ``verify=False`` skips reading the corpus, which is what a test with a
        hand-built index wants; it is not what a run that produces numbers for
        the report wants.
        """
        with index_path.open("rb") as file:
            payload = pickle.load(file)
        manifest = payload.get("manifest")

        if verify:
            if manifest is None:
                raise ValueError(
                    f"BM25 index at {index_path} carries no manifest, so it cannot be "
                    "checked against the corpus. Rebuild it: python -m src.retrieval bm25"
                )
            problems = manifest.mismatches(
                corpus_fingerprint=corpus_fingerprint(_read_corpus(processed_dir))
            )
            if problems:
                detail = "\n".join(f"  - {problem}" for problem in problems)
                raise ValueError(
                    f"BM25 index at {index_path} does not match the corpus on disk:\n"
                    f"{detail}\nRebuild it: python -m src.retrieval bm25"
                )

        return cls(payload["chunks"], manifest=manifest)

    def search(self, query: Query, k: int | None = None) -> list[RetrievedPassage]:
        """Return the highest-scoring passages matching the query filters.

        The filters are applied before scoring, as #22 requires, but the scores
        themselves come from the corpus-wide fit -- see the module docstring for
        why those are two different things.
        """
        wanted = resolve_k(query, k)
        if wanted <= 0 or self._bm25 is None:
            return []

        candidates = [
            (position, chunk)
            for position, chunk in enumerate(self.chunks)
            if matches(chunk, query)
        ]
        if not candidates:
            return []

        scores = self._bm25.get_scores(tokenize(query.text))
        return rank(
            ((chunk, float(scores[position])) for position, chunk in candidates),
            retriever=self.name,
            k=wanted,
            min_score=MIN_BM25_SCORE,
        )


def build_index(
    processed_dir: Path | None = None,
    index_path: Path = BM25_INDEX_FILE,
) -> BM25Retriever:
    """Convenience wrapper for building the project BM25 index."""
    return BM25Retriever.build(processed_dir=processed_dir, index_path=index_path)


def load_index(
    index_path: Path = BM25_INDEX_FILE,
    processed_dir: Path | None = None,
) -> BM25Retriever:
    """Convenience wrapper for loading the project BM25 index."""
    return BM25Retriever.load(index_path=index_path, processed_dir=processed_dir)
