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
from typing import Any

from src.config import PROCESSED_DIR, PROJECT_ROOT
from src.retrieval.records import Query
from src.stack import build_retriever, build_stack

from .benchmark import DEFAULT_QUESTIONS_PATH, load_questions
from .harness import evaluate
from .metrics import score_question
from .records import BenchmarkQuestion

RESULTS_ROOT = PROJECT_ROOT / "results"


@dataclass(frozen=True)
class EmbeddingExperiment:
    id: str
    name: str
    model: str
    index_dir: str
    dimensions: int
    query_prefix: str


@dataclass(frozen=True)
class GenerationExperiment:
    id: str
    name: str
    provider: str
    model: str


EMBEDDING_EXPERIMENTS = (
    EmbeddingExperiment(
        "E1", "BGE base", "BAAI/bge-base-en-v1.5", "data/index/chroma/E1", 768,
        "Represent this sentence for searching relevant passages: ",
    ),
    EmbeddingExperiment(
        "E2", "MiniLM", "sentence-transformers/all-MiniLM-L6-v2", "data/index/chroma/E2", 384, "",
    ),
    EmbeddingExperiment(
        "E3", "E5 base", "intfloat/e5-base-v2", "data/index/chroma/E3", 768, "query: ",
    ),
)
GENERATION_EXPERIMENTS = (
    GenerationExperiment("G1", "Local Ollama", "ollama", "llama3.2:3b"),
    GenerationExperiment("G2", "Hosted Mistral", "mistral", "ministral-8b-latest"),
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
    rows = []
    for experiment in EMBEDDING_EXPERIMENTS:
        stack = stack_factory(experiment)
        result_rows = []
        for question in questions:
            query = Query(question.question, top_k=top_k)
            passages = stack.retriever.search(query)
            from .records import RunResult

            result = RunResult.from_passages(
                question.question_id,
                passages,
                retriever=stack.retriever.name,
                config={"embedding_model": experiment.model},
            )
            result_rows.append(score_question(question, result, k=top_k))
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
    manifest = {"run_id": run_id, "matrix": "embedding", "rows": rows}
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
    rows = []
    for experiment in GENERATION_EXPERIMENTS:
        stack = stack_factory(experiment)
        report = evaluate(
            questions,
            stack.retriever,
            stack.generation,
            run_id=experiment.id,
            top_k=top_k,
            llm=stack.llm,
        )
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
    manifest = {"run_id": run_id, "matrix": "generation", "rows": rows}
    (run_dir / "summary.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def _default_embedding_stack(experiment: EmbeddingExperiment):
    """Build the hybrid retriever against the selected E-index."""
    return type("RetrievalStack", (), {
        "retriever": build_retriever(
            "hybrid",
            chroma_dir=PROJECT_ROOT / experiment.index_dir,
            embedding_model=experiment.model,
            embedding_dimensions=experiment.dimensions,
            embedding_query_prefix=experiment.query_prefix,
        )
    })()


def _default_generation_stack(experiment: GenerationExperiment):
    return build_stack("C4", provider=experiment.provider, model=experiment.model)


def prepare_embedding_indexes(processed_dir: Path) -> None:
    """Build one dense index per E-row using only the registry settings."""
    from src.retrieval import embed

    for experiment in EMBEDDING_EXPERIMENTS:
        embed.build(
            processed_dir=processed_dir,
            chroma_dir=PROJECT_ROOT / experiment.index_dir,
            model_name=experiment.model,
            dimensions=experiment.dimensions,
            rebuild=True,
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
    args = parser.parse_args(argv)
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    questions = load_questions(args.questions)
    if args.matrix == "embedding":
        if args.prepare_indexes:
            prepare_embedding_indexes(args.processed_dir or PROCESSED_DIR)
        run_embedding_ablation(
            questions, _default_embedding_stack, run_id=args.run_id,
            results_root=args.results_root, top_k=args.top_k,
        )
    else:
        run_generation_ablation(
            questions, _default_generation_stack, run_id=args.run_id,
            results_root=args.results_root, top_k=args.top_k,
        )


if __name__ == "__main__":
    main()
