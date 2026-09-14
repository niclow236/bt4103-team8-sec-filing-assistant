"""Hybrid BM25 and dense retrieval using reciprocal rank fusion."""

from __future__ import annotations

from dataclasses import replace

from .base import Retriever, resolve_k
from .constants import BM25, CANDIDATE_K, DENSE, FUSION_WEIGHTS, HYBRID, MIN_FUSED_SCORE, RRF_K
from .records import Query, RetrievedPassage


class HybridRetriever:
    """Fuse BM25 and dense result lists without comparing their score scales."""

    name = HYBRID

    def __init__(self, bm25: Retriever, dense: Retriever) -> None:
        self.bm25 = bm25
        self.dense = dense

    def search(self, query: Query, k: int | None = None) -> list[RetrievedPassage]:
        wanted = resolve_k(query, k)
        if wanted <= 0:
            return []

        depth = max(CANDIDATE_K, wanted)
        result_lists = (
            (self.bm25.search(query, k=depth), BM25),
            (self.dense.search(query, k=depth), DENSE),
        )
        fused: dict[str, tuple[RetrievedPassage, float, list[str]]] = {}
        for passages, method in result_lists:
            weight = FUSION_WEIGHTS[method]
            for position, passage in enumerate(passages, start=1):
                score = weight / (RRF_K + position)
                if passage.chunk_id not in fused:
                    fused[passage.chunk_id] = (passage, score, [method])
                else:
                    previous, total, sources = fused[passage.chunk_id]
                    fused[passage.chunk_id] = (previous, total + score, sources + [method])

        ordered = sorted(fused.values(), key=lambda entry: (-entry[1], entry[0].chunk_id))
        kept = [entry for entry in ordered if MIN_FUSED_SCORE is None or entry[1] >= MIN_FUSED_SCORE]
        return [
            replace(
                passage,
                score=score,
                rank=rank,
                retriever=self.name,
                sources=tuple(sources),
            )
            for rank, (passage, score, sources) in enumerate(kept[:wanted], start=1)
        ]


__all__ = ["HybridRetriever"]
