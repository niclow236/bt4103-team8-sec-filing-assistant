"""Load and validate the hand-written benchmark questions."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from src.config import PROJECT_ROOT, PROCESSED_DIR
from src.pipeline.chunk import iter_chunks

from .records import BenchmarkQuestion, BenchmarkValidationError

DEFAULT_QUESTIONS_PATH = PROJECT_ROOT / "benchmark" / "questions.jsonl"


def _available_chunk_ids(
    chunk_ids: Iterable[str] | Iterable[Mapping[str, Any]] | None,
    processed_dir: Path,
) -> set[str]:
    if chunk_ids is None:
        available = {chunk["chunk_id"] for chunk in iter_chunks(processed_dir=processed_dir)}
        if not available:
            raise FileNotFoundError(
                f"No chunked filings found under {processed_dir}; run "
                "`python -m src.pipeline chunk` first, or pass chunk_ids explicitly"
            )
        return available
    return {
        item if isinstance(item, str) else item["chunk_id"]
        for item in chunk_ids
    }


def load_questions(
    path: Path = DEFAULT_QUESTIONS_PATH,
    *,
    chunk_ids: Iterable[str] | Iterable[Mapping[str, Any]] | None = None,
    processed_dir: Path = PROCESSED_DIR,
) -> list[BenchmarkQuestion]:
    """Load JSONL questions and fail loudly on invalid or stale references.

    ``chunk_ids`` is injectable for tests and for callers that already hold a
    corpus. In normal use, IDs are resolved against the current processed
    corpus through :func:`iter_chunks`.
    """
    if not path.exists():
        raise FileNotFoundError(f"Benchmark questions file not found: {path}")

    available = _available_chunk_ids(chunk_ids, processed_dir)
    questions: list[BenchmarkQuestion] = []
    seen_ids: set[str] = set()
    # utf-8-sig drops the byte-order mark some editors write. Records are split on
    # \r\n, \r and \n only: JSON cannot hold those raw inside a string, but it can
    # hold U+2028, U+2029 and U+0085, which str.splitlines() would also split on.
    text = path.read_text(encoding="utf-8-sig")
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as error:
            raise BenchmarkValidationError(
                f"{path}:{line_number}: invalid JSON: {error.msg}"
            ) from error
        if not isinstance(data, Mapping):
            raise BenchmarkValidationError(f"{path}:{line_number}: expected a JSON object")

        try:
            question = BenchmarkQuestion.from_mapping(data)
        except BenchmarkValidationError as error:
            raise BenchmarkValidationError(f"{path}:{line_number}: {error}") from error

        if question.question_id in seen_ids:
            raise BenchmarkValidationError(
                f"{path}:{line_number}: duplicate question_id {question.question_id!r}"
            )
        seen_ids.add(question.question_id)

        referenced_ids = set(question.supporting_chunk_ids) | set(question.hard_negative_chunk_ids)
        unresolved = referenced_ids - available
        if unresolved:
            raise BenchmarkValidationError(
                f"{path}:{line_number}: unknown chunk IDs: {', '.join(sorted(unresolved))}"
            )
        questions.append(question)

    if not questions:
        raise BenchmarkValidationError(f"{path}: benchmark contains no questions")
    return questions