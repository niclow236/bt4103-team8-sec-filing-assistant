"""Load and validate the benchmark questions, and generate the XBRL benchmark."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import PROJECT_ROOT, PROCESSED_DIR
from src.pipeline.chunk import iter_chunks
from src.retrieval.facts import FACTS_FILE, load_facts

from .records import BenchmarkQuestion, BenchmarkValidationError

DEFAULT_QUESTIONS_PATH = PROJECT_ROOT / "benchmark" / "questions.jsonl"
DEFAULT_GENERATED_QUESTIONS_PATH = PROJECT_ROOT / "benchmark" / "generated.jsonl"


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


def _normalise_xbrl_label(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "figure"
    text = re.sub(r"\s+", " ", text)
    return text.strip(".") or "figure"


def _format_expected_answer(row: Mapping[str, Any]) -> str:
    raw = row.get("raw_value")
    if raw not in (None, ""):
        return str(raw).strip()
    value = row.get("value")
    if value is None or pd.isna(value):
        return "unknown"
    return str(value).strip()


def _supporting_chunk_ids_for(
    accession_no: str,
    *,
    tickers: Iterable[str] | None = None,
    processed_dir: Path = PROCESSED_DIR,
) -> list[str]:
    available = []
    for chunk in iter_chunks(processed_dir=processed_dir, tickers=list(tickers) if tickers else None):
        if chunk.get("accession_no") == accession_no:
            available.append(chunk["chunk_id"])
    return sorted(available)


def generate_xbrl_questions(
    facts_file: Path = FACTS_FILE,
    *,
    processed_dir: Path = PROCESSED_DIR,
    output_path: Path = DEFAULT_GENERATED_QUESTIONS_PATH,
) -> list[BenchmarkQuestion]:
    """Generate the mechanical XBRL benchmark from the current facts store.

    Each fact becomes one numeric benchmark question. A supporting chunk is any
    real processed passage from the same filing, and the generated JSONL is
    written to ``benchmark/generated.jsonl`` by default. The output follows the
    same schema as the hand-written benchmark and sets ``source`` to ``xbrl``.
    """
    frame = load_facts(facts_file)
    current = frame[frame["is_current_year"].fillna(False)]
    if current.empty:
        raise ValueError(f"No current-year facts found in {facts_file}")

    available_chunk_ids = {
        chunk["chunk_id"]
        for chunk in iter_chunks(processed_dir=processed_dir)
    }
    questions: list[BenchmarkQuestion] = []

    for row in current.to_dict("records"):
        accession = str(row.get("accession", "")).strip()
        if not accession:
            continue
        supporting = _supporting_chunk_ids_for(accession, processed_dir=processed_dir)
        if not supporting:
            continue

        ticker = str(row.get("ticker", "")).strip().upper()
        fiscal_year = row.get("fiscal_year")
        label = _normalise_xbrl_label(row.get("concept", "figure"))
        question_id = (
            f"xbrl-{ticker.lower()}-{int(fiscal_year)}-"
            f"{re.sub(r'[^a-z0-9]+', '-', label.lower()).strip('-')}-"
            f"{accession}"
        )
        question = BenchmarkQuestion(
            question_id=question_id,
            question=f"What was {label} for {ticker} in FY{int(fiscal_year)}?",
            expected_answer=_format_expected_answer(row),
            supporting_chunk_ids=tuple(supporting[:3]),
            hard_negative_chunk_ids=(),
            ticker=ticker,
            fiscal_year=int(fiscal_year),
            question_type="numeric",
            difficulty="mechanical",
            source="xbrl",
        )

        unresolved = set(question.supporting_chunk_ids) - available_chunk_ids
        if unresolved:
            raise BenchmarkValidationError(
                f"generated question {question.question_id!r} references unknown chunk IDs: "
                f"{', '.join(sorted(unresolved))}"
            )
        questions.append(question)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        for question in questions:
            stream.write(json.dumps(question.to_dict(), ensure_ascii=False, allow_nan=False) + "\n")

    return questions


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