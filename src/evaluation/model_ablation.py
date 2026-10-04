"""Embedding and generation ablation runners for issue 46."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from types import SimpleNamespace
from typing import Any

from src.config import PROCESSED_DIR, PROJECT_ROOT
from src.retrieval.records import Query
from src.stack import build_retriever, build_stack, stack_config
from src.rag.constants import DEFAULT_MISTRAL_MODEL

from .benchmark import DEFAULT_QUESTIONS_PATH, load_questions
from .harness import RunInterrupted, RunStopped, evaluate
from .metrics import score_question
from .records import BenchmarkQuestion, RunResult
from .run import _safe_run_id

RESULTS_ROOT = PROJECT_ROOT / "results"


@dataclass(frozen=True)
class EmbeddingExperiment:
    id: str
    name: str
    model: str
    index_dir: str
    dimensions: int
    query_prefix: str
    passage_prefix: str
    max_tokens: int


@dataclass(frozen=True)
class GenerationExperiment:
    id: str
    name: str
    provider: str
    model: str


EMBEDDING_EXPERIMENTS = (
    EmbeddingExperiment(
        "E1", "BGE base", "BAAI/bge-base-en-v1.5", "data/index/chroma", 768,
        "Represent this sentence for searching relevant passages: ", "", 512,
    ),
    EmbeddingExperiment(
        "E2", "MiniLM", "sentence-transformers/all-MiniLM-L6-v2", "data/index/chroma/E2", 384, "", "", 256,
    ),
    EmbeddingExperiment(
        "E3", "E5 base", "intfloat/e5-base-v2", "data/index/chroma/E3", 768, "query: ", "passage: ", 512,
    ),
)
GENERATION_EXPERIMENTS = (
    GenerationExperiment("G1", "Local Ollama", "ollama", "llama3.2:3b"),
    GenerationExperiment("G2", "Hosted Mistral", "mistral", DEFAULT_MISTRAL_MODEL),
)


def _safe_run_id(run_id: str) -> str:
    value = run_id.strip()
    if not value or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError("run_id must contain only letters, numbers, ., _, and -")
    return value


def _mean_metric(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [row[key] for row in rows if row.get(key) is not None]
    return mean(values) if values else None


def _write_table(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "config", "name", "questions", "recall", "ndcg", "mrr",
        "hard_negative_accuracy", "abstention_rate", "provider", "model",
        "embedding_model",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _configuration_manifest(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    metric_keys = ("questions", "recall", "ndcg", "mrr", "hard_negative_accuracy", "abstention_rate")
    return [
        {
            "config": {
                "id": row["config"], "name": row["name"],
                "provider": row.get("provider"), "model": row.get("model"),
                "embedding_model": row.get("embedding_model"),
            },
            **{key: row.get(key) for key in metric_keys},
        }
        for row in rows
    ]


def run_embedding_ablation(
    questions: Sequence[BenchmarkQuestion],
    stack_factory: Callable[[EmbeddingExperiment], Any],
    *,
    run_id: str,
    results_root: Path = RESULTS_ROOT,
    top_k: int = 10,
) -> dict[str, Any]:
    """Measure E1-E3 using stacks selected solely by embedding configuration."""
    run_dir = results_root / _safe_run_id(run_id)
    if run_dir.exists():
        raise FileExistsError(f"{run_dir} already holds a run; use a new --run-id")
    stacks = [(experiment, stack_factory(experiment)) for experiment in EMBEDDING_EXPERIMENTS]
    run_dir.mkdir(parents=True)
    rows = []
    for experiment, stack in stacks:
        result_rows = []
        for question in questions:
            from dataclasses import replace
            from src.rag.query import parse_question

            query = parse_question(question.question).to_query(top_k=top_k)
            if question.ticker is not None:
                query = replace(query, tickers=(question.ticker,))
            if question.fiscal_year is not None:
                query = replace(query, fiscal_years=(question.fiscal_year,))
            passages = stack.retriever.search(query)

            result = RunResult.from_passages(
                question.question_id,
                passages,
                retriever=stack.retriever.name,
                latency_ms=None,
                config={"embedding_model": experiment.model, "config": "C4"},
            )
            metrics = score_question(question, result, k=top_k)
            result_rows.append({
                "question_id": question.question_id,
                "question_type": question.question_type,
                "retriever": result.retriever,
                "retrieved_chunk_ids": list(result.retrieved_chunk_ids),
                "retrieved_scores": list(result.retrieved_scores),
                "k": top_k,
                **{key: metrics[key] for key in ("recall", "ndcg", "mrr", "hard_negative_accuracy")},
            })
        row = {
            "config": experiment.id,
            "name": experiment.name,
            "questions": len(result_rows),
            "recall": None,
            "ndcg": None,
            "mrr": None,
            "hard_negative_accuracy": None,
            "abstention_rate": None,
            "provider": None,
            "model": None,
            "embedding_model": experiment.model,
        }
        for metric in ("recall", "ndcg", "mrr", "hard_negative_accuracy"):
            row[metric] = _mean_metric(result_rows, metric)
        rows.append(row)
        output_dir = run_dir / experiment.id
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "report.json").write_text(
            json.dumps({"config": experiment.__dict__, "results": result_rows}, indent=2)
            + "\n",
            encoding="utf-8",
        )
    _write_table(run_dir / "summary.csv", rows)
    manifest = {"run_id": run_id, "top_k": top_k, "configurations": _configuration_manifest(rows)}
    (run_dir / "summary.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def run_generation_ablation(
    questions: Sequence[BenchmarkQuestion],
    stack_factory: Callable[[GenerationExperiment], Any],
    *,
    run_id: str,
    results_root: Path = RESULTS_ROOT,
    top_k: int = 10,
) -> dict[str, Any]:
    """Measure G1-G2 using stacks selected solely by provider configuration."""
    run_dir = results_root / _safe_run_id(run_id)
    if run_dir.exists():
        raise FileExistsError(f"{run_dir} already holds a run; use a new --run-id")
    stacks = [(experiment, stack_factory(experiment)) for experiment in GENERATION_EXPERIMENTS]
    run_dir.mkdir(parents=True)
    rows = []
    for experiment, stack in stacks:
        try:
            report = evaluate(
                questions, stack.retriever, stack.generation, run_id=experiment.id,
                min_score=stack.config.min_score, top_k=stack.config.top_k,
                llm=stack.llm, use_facts=stack.config.use_facts,
                use_decomposition=stack.config.use_decomposition, stack=stack.config,
            )
        except (RunStopped, RunInterrupted) as error:
            report = error.report
            output_dir = run_dir / experiment.id
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            rows.append({
                "config": experiment.id, "name": experiment.name,
                "questions": report["summary"]["total"], "recall": None,
                "ndcg": None, "mrr": None, "hard_negative_accuracy": None,
                "abstention_rate": report["summary"]["abstention_rate"],
                "provider": experiment.provider, "model": experiment.model,
                "embedding_model": None,
            })
            _write_table(run_dir / "summary.csv", rows)
            (run_dir / "summary.json").write_text(
                json.dumps({"run_id": run_id, "top_k": top_k, "configurations": _configuration_manifest(rows)}, indent=2)
                + "\n",
                encoding="utf-8",
            )
            raise
        rows.append({
            "config": experiment.id,
            "name": experiment.name,
            "questions": report["summary"]["total"],
            "recall": None,
            "ndcg": None,
            "mrr": None,
            "hard_negative_accuracy": None,
            "abstention_rate": report["summary"]["abstention_rate"],
            "provider": experiment.provider,
            "model": experiment.model,
            "embedding_model": None,
        })
        output_dir = run_dir / experiment.id
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
    _write_table(run_dir / "summary.csv", rows)
    manifest = {"run_id": run_id, "top_k": stack.config.top_k, "configurations": _configuration_manifest(rows)}
    (run_dir / "summary.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def _default_embedding_stack(
    experiment: EmbeddingExperiment,
    *,
    processed_dir: Path = PROCESSED_DIR,
):
    """Build the hybrid retriever against the selected E-index."""
    return SimpleNamespace(
        config=stack_config("C4"),
        retriever=build_retriever(
            "hybrid",
            processed_dir=processed_dir,
            chroma_dir=PROJECT_ROOT / experiment.index_dir,
            embedding_model=experiment.model,
            embedding_dimensions=experiment.dimensions,
            embedding_query_prefix=experiment.query_prefix,
            embedding_passage_prefix=experiment.passage_prefix,
        ),
    )


def _default_generation_stack(
    experiment: GenerationExperiment,
    *,
    processed_dir: Path = PROCESSED_DIR,
):
    parts = _GENERATION_PARTS.setdefault(str(processed_dir), {})
    return build_stack(
        "C4", processed_dir=processed_dir, provider=experiment.provider, model=experiment.model,
        parts=parts,
    )


_GENERATION_PARTS: dict[str, dict[str, Any]] = {}


def prepare_embedding_indexes(processed_dir: Path) -> None:
    """Build one dense index per E-row using only the registry settings."""
    from src.retrieval import embed

    for experiment in EMBEDDING_EXPERIMENTS:
        embed.build(
            processed_dir=processed_dir,
            chroma_dir=PROJECT_ROOT / experiment.index_dir,
            model_name=experiment.model,
            dimensions=experiment.dimensions,
            passage_prefix=experiment.passage_prefix,
            max_tokens=experiment.max_tokens,
            rebuild=False,
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix", choices=("embedding", "generation"))
    parser.add_argument("questions", type=Path, nargs="?", default=DEFAULT_QUESTIONS_PATH)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--processed-dir", type=Path)
    parser.add_argument(
        "--prepare-indexes",
        action="store_true",
        help="Build the three configured embedding indexes before E1-E3.",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--limit", type=int, help="Evenly sample this many questions.")
    args = parser.parse_args(argv)
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    try:
        run_dir = args.results_root / _safe_run_id(args.run_id)
    except ValueError as error:
        parser.error(str(error))
    if run_dir.exists():
        parser.error(f"{run_dir} already holds a run; use a new --run-id")
    questions = load_questions(args.questions, processed_dir=args.processed_dir or PROCESSED_DIR)
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        step = max(1, len(questions) // args.limit)
        questions = questions[::step][:args.limit]
    if args.matrix == "embedding":
        if args.prepare_indexes:
            prepare_embedding_indexes(args.processed_dir or PROCESSED_DIR)
        manifest = run_embedding_ablation(
            questions,
            lambda experiment: _default_embedding_stack(
                experiment, processed_dir=args.processed_dir or PROCESSED_DIR
            ),
            run_id=args.run_id,
            results_root=args.results_root, top_k=args.top_k,
        )
    else:
        try:
            manifest = run_generation_ablation(
                questions,
                lambda experiment: _default_generation_stack(
                    experiment, processed_dir=args.processed_dir or PROCESSED_DIR
                ),
                run_id=args.run_id,
                results_root=args.results_root, top_k=args.top_k,
            )
        except (RunStopped, RunInterrupted) as error:
            raise SystemExit(f"{error}; partial results are under {run_dir}") from None
    print((run_dir / "summary.csv").read_text(encoding="utf-8"), end="")
    print(f"written: {run_dir}")


if __name__ == "__main__":
    main()
