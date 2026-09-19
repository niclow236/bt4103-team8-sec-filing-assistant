"""Cross-encoder reranking over a first-stage retriever's candidates."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Sequence

from .base import WrappingRetriever
from .constants import CANDIDATE_K, RERANK, RERANK_BATCH_SIZE, RERANK_MAX_TOKENS, RERANK_MODEL
from .records import Query, RetrievedPassage


@lru_cache(maxsize=None)
def load_model(model_name: str = RERANK_MODEL) -> Any:
    """Load one shared cross-encoder model for the app and CLI."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name, max_length=RERANK_MAX_TOKENS)


class CrossEncoderReranker(WrappingRetriever):
    """Re-score candidates with a cross-encoder and keep the best passages."""

    name = RERANK

    def __init__(
        self,
        inner,
        model_name: str = RERANK_MODEL,
        model: Any | None = None,
        candidate_k: int = CANDIDATE_K,
        min_score: float | None = None,
        batch_size: int = RERANK_BATCH_SIZE,
    ) -> None:
        super().__init__(inner, candidate_k=candidate_k, min_score=min_score)
        self.model_name = model_name
        self.model = model if model is not None else load_model(model_name)
        self.batch_size = batch_size

    def score(self, query: Query, passages: Sequence[RetrievedPassage]) -> Sequence[float]:
        pairs = [(query.text, passage.text) for passage in passages]
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        return [float(score) for score in scores]


Reranker = CrossEncoderReranker

__all__ = ["CrossEncoderReranker", "Reranker", "load_model"]
