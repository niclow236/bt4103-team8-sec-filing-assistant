"""Stable records shared by the benchmark loader and evaluation harness."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from src.retrieval.records import RetrievedPassage


class BenchmarkValidationError(ValueError):
    """Raised when a benchmark record does not satisfy the benchmark contract."""


def _required_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkValidationError(f"{field_name} must be a non-empty string")
    return value


def _string_tuple(value: Any, field_name: str, *, allow_empty: bool = True) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise BenchmarkValidationError(f"{field_name} must be a list of non-empty strings")
    if not allow_empty and not value:
        raise BenchmarkValidationError(f"{field_name} must contain at least one chunk ID")
    if len(set(value)) != len(value):
        raise BenchmarkValidationError(f"{field_name} must not contain duplicate values")
    return tuple(value)


@dataclass(frozen=True)
class BenchmarkQuestion:
    """One ground-truth question used to evaluate retrieval."""

    question_id: str
    question: str
    expected_answer: str
    supporting_chunk_ids: tuple[str, ...]
    hard_negative_chunk_ids: tuple[str, ...]
    ticker: str
    fiscal_year: int
    question_type: str
    difficulty: str
    source: str

    REQUIRED_FIELDS = frozenset(
        {
            "question_id",
            "question",
            "expected_answer",
            "supporting_chunk_ids",
            "hard_negative_chunk_ids",
            "ticker",
            "fiscal_year",
            "question_type",
            "difficulty",
            "source",
        }
    )

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> BenchmarkQuestion:
        """Validate and convert one decoded JSON object."""
        unknown = set(data) - cls.REQUIRED_FIELDS
        missing = cls.REQUIRED_FIELDS - set(data)
        if missing:
            raise BenchmarkValidationError(
                f"missing required fields: {', '.join(sorted(missing))}"
            )
        if unknown:
            raise BenchmarkValidationError(
                f"unknown fields: {', '.join(sorted(unknown))}"
            )

        fiscal_year = data["fiscal_year"]
        if isinstance(fiscal_year, bool) or not isinstance(fiscal_year, int) or fiscal_year < 1900:
            raise BenchmarkValidationError("fiscal_year must be an integer year")

        supporting = _string_tuple(
            data["supporting_chunk_ids"], "supporting_chunk_ids", allow_empty=False
        )
        hard_negatives = _string_tuple(data["hard_negative_chunk_ids"], "hard_negative_chunk_ids")
        overlap = set(supporting) & set(hard_negatives)
        if overlap:
            raise BenchmarkValidationError(
                f"supporting and hard-negative IDs overlap: {', '.join(sorted(overlap))}"
            )

        return cls(
            question_id=_required_text(data["question_id"], "question_id"),
            question=_required_text(data["question"], "question"),
            expected_answer=_required_text(data["expected_answer"], "expected_answer"),
            supporting_chunk_ids=supporting,
            hard_negative_chunk_ids=hard_negatives,
            ticker=_required_text(data["ticker"], "ticker").upper(),
            fiscal_year=fiscal_year,
            question_type=_required_text(data["question_type"], "question_type"),
            difficulty=_required_text(data["difficulty"], "difficulty"),
            source=_required_text(data["source"], "source"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible representation used by the benchmark."""
        return asdict(self) | {
            "supporting_chunk_ids": list(self.supporting_chunk_ids),
            "hard_negative_chunk_ids": list(self.hard_negative_chunk_ids),
        }


@dataclass(frozen=True)
class RunResult:
    """Per-question retrieval output consumed by later evaluation metrics."""

    question_id: str
    retriever: str
    retrieved_chunk_ids: tuple[str, ...]
    retrieved_scores: tuple[float, ...]
    latency_ms: float | None = None
    config: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.retrieved_chunk_ids) != len(self.retrieved_scores):
            raise ValueError("retrieved_chunk_ids and retrieved_scores must have equal lengths")
        if len(set(self.retrieved_chunk_ids)) != len(self.retrieved_chunk_ids):
            raise ValueError("retrieved_chunk_ids must not contain duplicates")

    @classmethod
    def from_passages(
        cls,
        question_id: str,
        passages: list[RetrievedPassage],
        *,
        latency_ms: float | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> RunResult:
        """Build a result without coupling metrics to a retriever class."""
        retriever = passages[0].retriever if passages else ""
        if any(passage.retriever != retriever for passage in passages):
            raise ValueError("all passages in a RunResult must have the same retriever")
        return cls(
            question_id=question_id,
            retriever=retriever,
            retrieved_chunk_ids=tuple(passage.chunk_id for passage in passages),
            retrieved_scores=tuple(float(passage.score) for passage in passages),
            latency_ms=latency_ms,
            config={} if config is None else config,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible representation for results files."""
        return {
            "question_id": self.question_id,
            "retriever": self.retriever,
            "retrieved_chunk_ids": list(self.retrieved_chunk_ids),
            "retrieved_scores": list(self.retrieved_scores),
            "latency_ms": self.latency_ms,
            "config": dict(self.config),
        }