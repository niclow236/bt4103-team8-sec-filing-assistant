"""Run the C0-C4 retrieval ablation and write reproducible result files."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from statistics import mean
from typing import Any

from src.config import PROCESSED_DIR, PROJECT_ROOT
from src.pipeline.chunk import iter_chunks
from src.pipeline.constants import CHUNK_CHAR_BUDGET
from src.retrieval.base import Retriever
from src.retrieval.records import Query

from .benchmark import DEFAULT_QUESTIONS_PATH, load_questions
from .metrics import score_question
from .records import BenchmarkQuestion, RunResult

DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results"
CONFIGURATIONS: dict[str, dict[str, Any]] = {
    "C0": {
        "name": "naive BM25",
        "retriever": "bm25-fixed-size",
        "chunking": "fixed-size",
        "metadata_filter": False,
    },
    "C1": {
        "name": "section-aware BM25",
        "retriever": "bm25",
        "chunking": "section-aware",
        "metadata_filter": False,
    },
    "C2": {
        "name": "dense retrieval",
        "retriever": "dense",
        "chunking": "section-aware",
        "metadata_filter": False,
    },
    "C3": {
        "name": "hybrid retrieval",
        "retriever": "hybrid",
        "chunking": "section-aware",
        "metadata_filter": False,
    },
    "C4": {
        "name": "hybrid retrieval with metadata filters",
        "retriever": "hybrid",
        "chunking": "section-aware",
        "metadata_filter": True,
    },
}


def _safe_run_id(run_id: str) -> str:
    value = run_id.strip()
    if not value or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError("run_id must contain only letters, numbers, ., _, and -")
    return value


def _query(question: BenchmarkQuestion, *, top_k: int, metadata_filter: bool) -> Query:
    if not metadata_filter:
        return Query(question.question, top_k=top_k)

    from src.rag.query import parse_question

    query = parse_question(question.question).to_query(top_k=top_k)
    if question.ticker is not None:
        query = replace(query, tickers=(question.ticker,))
    if question.fiscal_year is not None:
        query = replace(query, fiscal_years=(question.fiscal_year,))
    return query


def _summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    metric_names = ("recall", "ndcg", "mrr", "hard_negative_accuracy")
    return {
        "questions": len(rows),
        **{
            metric: mean(
                row[metric] for row in rows if row[metric] is not None
            ) if any(row[metric] is not None for row in rows) else None
            for metric in metric_names
        },
    }


def run_configuration(
    questions: Sequence[BenchmarkQuestion],
    retriever: Retriever,
    *,
    config_id: str,
    output_dir: Path,
    top_k: int = 10,
) -> dict[str, Any]:
    """Run one retriever configuration and write its question-level results."""
    if config_id not in CONFIGURATIONS:
        raise ValueError(f"unknown configuration: {config_id}")
    if top_k < 1:
        raise ValueError("top_k must be positive")

    configuration = dict(CONFIGURATIONS[config_id])
    configuration["id"] = config_id
    configuration["top_k"] = top_k
    rows: list[dict[str, Any]] = []
    for question in questions:
        query = _query(
            question,
            top_k=top_k,
            metadata_filter=configuration["metadata_filter"],
        )
        passages = retriever.search(query)
        result = RunResult.from_passages(
            question.question_id,
            passages,
            retriever=retriever.name,
            config=configuration,
        )
        scoring_question = question
        if config_id == "C0" and hasattr(retriever, "relevant_chunk_ids"):
            scoring_question = replace(
                question,
                supporting_chunk_ids=tuple(retriever.relevant_chunk_ids(question)),
                hard_negative_chunk_ids=tuple(retriever.relevant_chunk_ids(question, hard=True)),
            )
        metrics = score_question(scoring_question, result, k=top_k)
        rows.append(
            {
                "question_id": question.question_id,
                "question_type": question.question_type,
                "retriever": retriever.name,
                "config": configuration,
                "retrieved_chunk_ids": list(result.retrieved_chunk_ids),
                "retrieved_scores": list(result.retrieved_scores),
                **{name: metrics[name] for name in (
                    "recall", "ndcg", "mrr", "hard_negative_accuracy"
                )},
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "questions.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {"config": configuration, **_summary(rows)}
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def run_ablation(
    questions: Sequence[BenchmarkQuestion],
    retrievers: Mapping[str, Retriever],
    *,
    run_id: str,
    results_root: Path = DEFAULT_RESULTS_ROOT,
    top_k: int = 10,
    configurations: Sequence[str] = tuple(CONFIGURATIONS),
) -> dict[str, Any]:
    """Run selected C0-C4 rows and write ``results/<run-id>/``."""
    run_dir = results_root / _safe_run_id(run_id)
    if run_dir.exists():
        raise FileExistsError(f"{run_dir} already holds a run; re-run under a new --run-id")
    summaries: list[dict[str, Any]] = []
    for config_id in configurations:
        if config_id not in CONFIGURATIONS:
            raise ValueError(f"unknown configuration: {config_id}")
        retriever_key = config_id if config_id in retrievers else CONFIGURATIONS[config_id]["retriever"]
        if retriever_key not in retrievers:
            raise ValueError(f"missing retriever for {config_id}: {retriever_key}")
        summaries.append(
            run_configuration(
                questions,
                retrievers[retriever_key],
                config_id=config_id,
                output_dir=run_dir / config_id,
                top_k=top_k,
            )
        )

    summary_rows = []
    for summary in summaries:
        row = {"config": summary["config"]["id"], "name": summary["config"]["name"]}
        row.update({key: summary[key] for key in (
            "questions", "recall", "ndcg", "mrr", "hard_negative_accuracy"
        )})
        summary_rows.append(row)
    with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    manifest = {"run_id": run_id, "top_k": top_k, "configurations": summaries}
    (run_dir / "summary.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


class FixedSizeBM25Retriever:
    """BM25 over filing-wide fixed-size windows for the naive C0 baseline."""

    name = "bm25-fixed-size"

    def __init__(self, processed_dir: Path, budget: int = CHUNK_CHAR_BUDGET) -> None:
        from src.retrieval.bm25 import BM25Retriever

        source_chunks = list(iter_chunks(processed_dir=processed_dir))
        grouped: dict[str, list[dict[str, Any]]] = {}
        for chunk in source_chunks:
            grouped.setdefault(chunk["accession_no"], []).append(chunk)

        fixed_chunks: list[dict[str, Any]] = []
        self._source_ids: dict[str, set[str]] = {}
        for accession, chunks in grouped.items():
            spans: list[tuple[int, int, str]] = []
            offset = 0
            for chunk in chunks:
                spans.append((offset, offset + len(chunk["text"]), chunk["chunk_id"]))
                offset += len(chunk["text"]) + 1
            text = "\n".join(chunk["text"] for chunk in chunks)
            for start in range(0, len(text), budget):
                end = start + budget
                fixed_id = f"fixed-{accession}-{start:08d}"
                window = text[start:end]
                source_ids = {
                    chunk_id
                    for first, last, chunk_id in spans
                    if first < end and last > start
                }
                template = chunks[0]
                fixed_chunks.append({**template, "chunk_id": fixed_id, "text": window})
                self._source_ids[fixed_id] = source_ids
        self._retriever = BM25Retriever(fixed_chunks)

    def search(self, query: Query, k: int | None = None):
        return [
            replace(passage, retriever=self.name)
            for passage in self._retriever.search(query, k=k)
        ]

    def relevant_chunk_ids(
        self,
        question: BenchmarkQuestion,
        *,
        hard: bool = False,
    ) -> list[str]:
        source_ids = (
            question.hard_negative_chunk_ids if hard else question.supporting_chunk_ids
        )
        return [
            fixed_id
            for fixed_id, origins in self._source_ids.items()
            if origins.intersection(source_ids)
        ]


def _load_retrievers(processed_dir: Path) -> dict[str, Retriever]:
    from src.retrieval.bm25 import BM25Retriever
    from src.retrieval.dense import DenseRetriever
    from src.retrieval.hybrid import HybridRetriever

    bm25 = BM25Retriever.load(processed_dir=processed_dir)
    dense = DenseRetriever.load(processed_dir=processed_dir)
    return {
        "bm25-fixed-size": FixedSizeBM25Retriever(processed_dir),
        "bm25": bm25,
        "dense": dense,
        "hybrid": HybridRetriever(bm25, dense),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("questions", type=Path, nargs="?", default=DEFAULT_QUESTIONS_PATH)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--config", choices=tuple(CONFIGURATIONS), action="append")
    args = parser.parse_args(argv)
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    questions = load_questions(args.questions, processed_dir=args.processed_dir)
    run_ablation(
        questions,
        _load_retrievers(args.processed_dir),
        run_id=args.run_id,
        results_root=args.results_root,
        top_k=args.top_k,
        configurations=args.config or tuple(CONFIGURATIONS),
    )


if __name__ == "__main__":
    main()
