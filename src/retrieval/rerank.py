"""Opt-in cross-encoder experiment; no change to the shipped C4 stack."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from time import perf_counter
from typing import Any, Sequence

from .base import matches, reorder, resolve_k, validate_table_boost
from .constants import CANDIDATE_K
from .records import Query, RetrievedPassage


@dataclass(frozen=True)
class RerankConfig:
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    revision: str = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
    candidate_k: int = CANDIDATE_K
    max_length: int = 512
    batch_size: int = 32
    device: str | None = None

    def __post_init__(self) -> None:
        for name in ("candidate_k", "max_length", "batch_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("model", "revision"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError("model and revision must be explicit non-empty strings")
        if self.device is not None and (not isinstance(self.device, str) or not self.device.strip()):
            raise ValueError("device must be non-empty or None")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {
            "input": "(Query.text, RetrievedPassage.text); no added header or prompt",
            "scores": "raw logits; sign-aware Query.table_boost; no score floor",
            "ties": "chunk_id ascending",
            "truncation": "tokenizer longest_first, including query and special tokens",
            "latency": "predict and reorder only; diagnostic token counting excluded",
        }


@dataclass(frozen=True)
class RerankResult:
    passages: list[RetrievedPassage]
    diagnostics: dict[str, Any]


class CrossEncoderReranker:
    """Load once and re-score a supplied pool, allowing a paired comparison.

    Supplying candidates separately ensures the experiment uses exactly one
    filtered hybrid pool for both rows. Scores on all candidates are retained
    in diagnostics so rankings can be rescored without loading the model.
    """

    name = "hybrid-reranked-experimental"

    def __init__(self, config: RerankConfig = RerankConfig(), *, model: Any = None) -> None:
        self.config = config
        self._model = model

    def load(self) -> Any:
        if self._model is None:
            from sentence_transformers import CrossEncoder
            from torch.nn import Identity

            self._model = CrossEncoder(
                self.config.model, revision=self.config.revision,
                max_length=self.config.max_length, device=self.config.device,
                activation_fn=Identity(), trust_remote_code=False,
            )
        return self._model

    def rerank(
        self, query: Query, candidates: Sequence[RetrievedPassage], *, k: int | None = None,
    ) -> RerankResult:
        wanted = resolve_k(query, k)
        validate_table_boost(query.table_boost)
        if wanted > self.config.candidate_k:
            raise ValueError("final k must not exceed candidate_k")
        if len(candidates) > self.config.candidate_k:
            raise ValueError("candidate pool exceeds candidate_k")
        if len({p.chunk_id for p in candidates}) != len(candidates):
            raise ValueError("candidate pool contains duplicate chunk IDs")
        if any(not matches(asdict(p), query) for p in candidates):
            raise ValueError("candidate pool violates the query metadata filters")
        if wanted <= 0 or not candidates:
            return RerankResult([], {"latency_ms": 0.0, "candidates": [], "truncated": 0})

        model = self.load()
        from torch.nn import Identity

        # Count the exact paired input, without truncating the diagnostics.
        # This extra pass is not part of inference and must not affect its gate.
        diagnostic_started = perf_counter()
        encoded = model.tokenizer(
            [query.text] * len(candidates), [p.text for p in candidates],
            truncation=False, padding=False, return_length=True,
            verbose=False,
        )
        lengths = [int(n) for n in encoded["length"]]
        started = perf_counter()
        diagnostic_tokenization_ms = (started - diagnostic_started) * 1000
        scores = [float(n) for n in model.predict(
            [(query.text, p.text) for p in candidates],
            batch_size=self.config.batch_size, show_progress_bar=False,
            activation_fn=Identity(),
        )]
        if len(scores) != len(candidates) or len(lengths) != len(candidates):
            raise ValueError("cross-encoder returned a different number of scores or token lengths")
        if not all(isfinite(n) for n in scores):
            raise ValueError("cross-encoder returned non-finite scores")
        passages = reorder(
            zip(candidates, scores), retriever=self.name, k=wanted,
            table_boost=query.table_boost,
        )
        elapsed = (perf_counter() - started) * 1000
        return RerankResult(passages, {
            "latency_ms": elapsed,
            "diagnostic_tokenization_ms": diagnostic_tokenization_ms,
            "device": str(getattr(model, "device", self.config.device)),
            "truncated": sum(n > self.config.max_length for n in lengths),
            "candidates": [
                {"chunk_id": p.chunk_id, "hybrid_rank": p.rank,
                 "hybrid_score": p.score, "raw_logit": score,
                 "input_tokens": n, "truncated": n > self.config.max_length}
                for p, score, n in zip(candidates, scores, lengths)
            ],
        })
