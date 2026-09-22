"""Load and validate benchmark questions, and generate the XBRL benchmark."""

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
    return {item if isinstance(item, str) else item["chunk_id"] for item in chunk_ids}


def _normalise_xbrl_label(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "figure"
    text = re.sub(r"\s+", " ", str(value).strip())
    return text.strip(". ") or "figure"


def _format_expected_answer(row: Mapping[str, Any]) -> str:
    raw = row.get("raw_value")
    if raw in (None, ""):
        raw = row.get("value")
        if raw is None or pd.isna(raw):
            return "unknown"
    unit = str(row.get("unit") or "").strip()
    return f"{str(raw).strip()} {unit}".strip()


def _value_needles(raw: Any) -> set[str]:
    text = str(raw or "").strip()
    if not text:
        return set()
    try:
        value = float(text)
    except ValueError:
        return {text}
    needles = set()
    for divisor in (1, 1_000, 1_000_000):
        scaled = value / divisor
        if abs(scaled - round(scaled)) < 1e-9:
            whole = abs(int(round(scaled)))
            needles.update({str(whole), f"{whole:,}"})
    return {needle for needle in needles if len(needle) >= 3}


def _supporting_chunk_ids_for(row: Mapping[str, Any], candidates: list[dict]) -> list[str]:
    needles = _value_needles(row.get("raw_value"))
    return [
        chunk["chunk_id"]
        for chunk in candidates
        if any(needle in chunk["text"] for needle in needles)
    ][:3]


def generate_xbrl_questions(
    facts_file: Path = FACTS_FILE,
    *,
    processed_dir: Path = PROCESSED_DIR,
    output_path: Path = DEFAULT_GENERATED_QUESTIONS_PATH,
) -> list[BenchmarkQuestion]:
    """Generate the mechanical XBRL benchmark from the current facts store."""
    frame = load_facts(facts_file)
    current = frame[frame["is_current_year"].fillna(False)]
    if current.empty:
        raise ValueError(f"No current-year facts found in {facts_file}")

    chunks_by_accession: dict[str, list[dict]] = {}
    for chunk in iter_chunks(processed_dir=processed_dir):
        chunks_by_accession.setdefault(chunk["accession_no"], []).append(chunk)

    questions: list[BenchmarkQuestion] = []
    seen_ids: set[str] = set()
    for row in current.to_dict("records"):
        accession = str(row.get("accession", "")).strip()
        if not accession:
            continue
        supporting = _supporting_chunk_ids_for(row, chunks_by_accession.get(accession, []))
        if not supporting:
            continue

        ticker = str(row.get("ticker", "")).strip().upper()
        fiscal_year = row.get("fiscal_year")
        raw_label = row.get("label")
        if raw_label is None or (isinstance(raw_label, float) and pd.isna(raw_label)):
            raw_label = row.get("concept", "figure")
        label = _normalise_xbrl_label(raw_label)
        question_id = (
            f"xbrl-{ticker.lower()}-{int(fiscal_year)}-"
            f"{re.sub(r'[^a-z0-9]+', '-', label.lower()).strip('-')}-"
            f"{accession}"
        )
        question = BenchmarkQuestion(
            question_id=question_id,
            question=f"What was {label} for {ticker} in FY{int(fiscal_year)}?",
            expected_answer=_format_expected_answer(row),
            supporting_chunk_ids=tuple(supporting),
            hard_negative_chunk_ids=(),
            ticker=ticker,
            fiscal_year=int(fiscal_year),
            question_type="numeric",
            difficulty="mechanical",
            source="xbrl",
        )
        if question.question_id in seen_ids:
            raise BenchmarkValidationError(
                f"generated duplicate question_id {question.question_id!r}"
            )
        seen_ids.add(question.question_id)
        questions.append(BenchmarkQuestion.from_mapping(question.to_dict()))

    if not questions:
        raise ValueError(
            f"No benchmark questions generated from {facts_file}; no processed filing under "
            f"{processed_dir} matches a current-year fact — run `python -m src.pipeline chunk` first"
        )

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
    """Load JSONL questions and fail loudly on invalid or stale references."""
    if not path.exists():
        raise FileNotFoundError(f"Benchmark questions file not found: {path}")

    available = _available_chunk_ids(chunk_ids, processed_dir)
    questions: list[BenchmarkQuestion] = []
    seen_ids: set[str] = set()
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
