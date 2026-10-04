"""Compare E1-E3 embeddings and G1-G2 providers with reproducible reports."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from shlex import join as shell_join
from time import perf_counter
from types import SimpleNamespace
from typing import Any

from src.config import PROCESSED_DIR, PROJECT_ROOT
from src.rag import ProviderUnavailable
from src.rag.constants import DEFAULT_MODEL, DEFAULT_MISTRAL_MODEL
from src.retrieval.embedding_config import EmbeddingConfig
from src.stack import RESULTS_ROOT, build_retriever, build_stack, stack_config

from .benchmark import DEFAULT_QUESTIONS_PATH, load_questions
from .harness import RunInterrupted, RunStopped, evaluate
from .metrics import score_question
from .records import BenchmarkQuestion, RunResult
from .results import available_run_dir, write_comparison
from .run import _query, _safe_run_id, _summary


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

    @property
    def embedding(self) -> EmbeddingConfig:
        return EmbeddingConfig(
            model=self.model, dimensions=self.dimensions,
            index_dir=PROJECT_ROOT / self.index_dir, query_prefix=self.query_prefix,
            passage_prefix=self.passage_prefix, max_tokens=self.max_tokens,
        )


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
    GenerationExperiment("G1", "Local Ollama", "ollama", DEFAULT_MODEL),
    GenerationExperiment("G2", "Hosted Mistral", "mistral", DEFAULT_MISTRAL_MODEL),
)
MODEL_FIELDS = (
    "abstention_rate", "provider", "model", "embedding_model", "retriever",
    "max_tokens", "n_indexed", "n_truncated", "truncation_rate", "build_ms",
    "median_latency_ms", "retried_questions", "llm_questions", "llm_abstention_rate",
)


def _write_report(run_dir: Path, config_id: str, report: dict[str, Any]) -> None:
    output_dir = run_dir / config_id
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def _embedding_diagnostics(retriever: Any, experiment: EmbeddingExperiment) -> dict[str, Any]:
    """Read truncation from the validated index, rather than a possibly stale file."""
    dense = getattr(retriever, "dense", retriever)
    collection = getattr(dense, "collection", None)
    manifest = getattr(dense, "manifest", None)
    limit = getattr(manifest, "max_tokens", None) or experiment.max_tokens
    if collection is None:
        return {"max_tokens": limit, "n_indexed": None, "n_truncated": None, "truncation_rate": None}
    from src.retrieval.embed import TOKENS_FIELD

    indexed = collection.count()
    truncated = len(collection.get(where={TOKENS_FIELD: {"$gt": limit}}, include=[])["ids"])
    return {"max_tokens": limit, "n_indexed": indexed, "n_truncated": truncated,
            "truncation_rate": truncated / indexed if indexed else None}


def run_embedding_ablation(
    questions: Sequence[BenchmarkQuestion],
    stack_factory: Callable[[EmbeddingExperiment], Any],
    *, run_id: str, results_root: Path = RESULTS_ROOT, top_k: int = 10,
    build_times: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Measure filtered hybrid and dense-only retrieval over the same E-indexes.

    A custom stack with only ``retriever`` measures that retriever. The default
    stack supplies ``dense_retriever`` too, isolating the encoder from fusion.
    """
    if top_k < 1:
        raise ValueError("top_k must be positive")
    run_dir = available_run_dir(results_root, run_id)
    stacks = [(experiment, stack_factory(experiment)) for experiment in EMBEDDING_EXPERIMENTS]
    run_dir.mkdir(parents=True)
    summaries = []
    for experiment, stack in stacks:
        diagnostics = _embedding_diagnostics(stack.retriever, experiment)
        retrievers = [(experiment.id, stack.retriever)]
        if getattr(stack, "dense_retriever", None) is not None:
            retrievers.append((f"{experiment.id}-dense", stack.dense_retriever))
        for config_id, retriever in retrievers:
            configuration = {
                **asdict(experiment), "id": config_id,
                "name": f"{experiment.name} ({retriever.name})",
                "embedding_model": experiment.model, "retriever": retriever.name,
                "metadata_filter": True, "top_k": top_k,
            }
            result_rows = []
            for question in questions:
                query = _query(question, top_k=top_k, metadata_filter=True)
                started = perf_counter()
                passages = retriever.search(query)
                latency_ms = (perf_counter() - started) * 1000
                result = RunResult.from_passages(
                    question.question_id, passages, retriever=retriever.name,
                    latency_ms=latency_ms, config=configuration,
                )
                result_rows.append({
                    **score_question(question, result, k=top_k), "config": configuration,
                    "retrieved_chunk_ids": list(result.retrieved_chunk_ids),
                    "retrieved_scores": list(result.retrieved_scores),
                })
            summary = {
                "config": configuration, **_summary(result_rows), **diagnostics,
                "build_ms": (build_times or {}).get(experiment.id),
                "median_latency_ms": median(row["latency_ms"] for row in result_rows) if result_rows else None,
            }
            summaries.append(summary)
            _write_report(run_dir, config_id, {**summary, "results": result_rows})
    return write_comparison(run_dir, run_id=run_id, top_k=top_k, summaries=summaries,
                            extra_fields=MODEL_FIELDS, matrix="embedding")


def _generation_summary(experiment: GenerationExperiment, report: dict[str, Any]) -> dict[str, Any]:
    results = report["results"]
    llm_rows = [row for row in results if row.get("route") == experiment.provider]
    latencies = [row["answer"]["latency_ms"] for row in results
                 if row.get("answer", {}).get("latency_ms") is not None]
    return {
        "config": {**asdict(experiment), "stack": report.get("stack")},
        "questions": report["summary"]["total"],
        "recall": None, "ndcg": None, "mrr": None, "hard_negative_accuracy": None,
        "abstention_rate": report["summary"]["abstention_rate"],
        "median_latency_ms": median(latencies) if latencies else None,
        "retried_questions": sum(row.get("attempts", 1) > 1 for row in results),
        "llm_questions": len(llm_rows),
        "llm_abstention_rate": mean(row["answer"]["abstained"] for row in llm_rows) if llm_rows else None,
        "stopped": report.get("stopped"),
    }


def run_generation_ablation(
    questions: Sequence[BenchmarkQuestion],
    stack_factory: Callable[[GenerationExperiment], Any],
    *, run_id: str, results_root: Path = RESULTS_ROOT,
) -> dict[str, Any]:
    """Measure both providers with each stack's complete answer settings."""
    run_dir = available_run_dir(results_root, run_id)
    stacks = [(experiment, stack_factory(experiment)) for experiment in GENERATION_EXPERIMENTS]
    if len({stack.config.top_k for _, stack in stacks}) != 1:
        raise ValueError("generation stacks must use the same top_k")
    top_k = stacks[0][1].config.top_k
    run_dir.mkdir(parents=True)
    summaries = []
    stopped = None
    for experiment, stack in stacks:
        try:
            report = evaluate(
                questions, stack.retriever, stack.generation, run_id=experiment.id,
                min_score=stack.config.min_score, top_k=stack.config.top_k,
                llm=stack.llm, use_facts=stack.config.use_facts,
                use_decomposition=stack.config.use_decomposition,
                use_refusal=stack.config.use_refusal, stack=stack.config,
            )
        except (RunStopped, RunInterrupted) as error:
            report, stopped = error.report, error
        summaries.append(_generation_summary(experiment, report))
        _write_report(run_dir, experiment.id, report)
        if stopped is not None:
            break
    manifest = write_comparison(run_dir, run_id=run_id, top_k=top_k, summaries=summaries,
                                extra_fields=MODEL_FIELDS, matrix="generation")
    if stopped is not None:
        raise stopped
    return manifest


def _default_embedding_stack(
    experiment: EmbeddingExperiment, *, processed_dir: Path = PROCESSED_DIR,
    parts: dict[Any, Any] | None = None,
):
    """Build both views over one encoder, sharing BM25 between E rows."""
    retriever = build_retriever(
        "hybrid", processed_dir=processed_dir,
        embedding=experiment.embedding, parts=parts,
    )
    return SimpleNamespace(config=stack_config("C4"), retriever=retriever,
                           dense_retriever=getattr(retriever, "dense", None))


def _default_generation_stack(
    experiment: GenerationExperiment, *, processed_dir: Path = PROCESSED_DIR,
    parts: dict[Any, Any] | None = None, use_facts: bool = False,
):
    return build_stack(
        "C4", processed_dir=processed_dir, provider=experiment.provider,
        model=experiment.model, parts=parts, use_facts=use_facts,
    )


def prepare_embedding_indexes(processed_dir: Path, *, rebuild: bool = False) -> dict[str, float]:
    """Incrementally prepare each E-index; return elapsed build time per model."""
    from src.retrieval import embed

    times = {}
    for experiment in EMBEDDING_EXPERIMENTS:
        started = perf_counter()
        embed.build(
            processed_dir=processed_dir, chroma_dir=PROJECT_ROOT / experiment.index_dir,
            model_name=experiment.model, dimensions=experiment.dimensions,
            passage_prefix=experiment.passage_prefix, max_tokens=experiment.max_tokens,
            rebuild=rebuild,
        )
        times[experiment.id] = (perf_counter() - started) * 1000
    return times


def _sample(questions: Sequence[BenchmarkQuestion], limit: int) -> list[BenchmarkQuestion]:
    """Even spacing over the full file, including both ends when possible."""
    if limit >= len(questions):
        return list(questions)
    if limit == 1:
        return [questions[len(questions) // 2]]
    return [questions[i * (len(questions) - 1) // (limit - 1)] for i in range(limit)]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix", choices=("embedding", "generation"))
    parser.add_argument("questions", type=Path, nargs="?", default=DEFAULT_QUESTIONS_PATH)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--prepare-indexes", action="store_true", help="Incrementally prepare E1-E3 indexes.")
    parser.add_argument("--rebuild-indexes", action="store_true", help="Re-encode E-indexes; requires --prepare-indexes.")
    parser.add_argument("--top-k", type=int, help="Override embedding depth (default 10); generation uses C4's budget.")
    parser.add_argument("--limit", type=int, help="Evenly sample this many questions over the full file.")
    facts = parser.add_mutually_exclusive_group()
    facts.add_argument("--no-facts", dest="use_facts", action="store_false", help="Send numeric questions to generation (default).")
    facts.add_argument("--with-facts", dest="use_facts", action="store_true", help="Include the C4 facts shortcut; routes are reported separately.")
    parser.set_defaults(use_facts=False)
    args = parser.parse_args(argv)
    if args.top_k is not None and args.top_k < 1:
        parser.error("--top-k must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.matrix == "generation" and (args.top_k is not None or args.prepare_indexes or args.rebuild_indexes):
        parser.error("--top-k and index preparation options apply only to embedding")
    if args.rebuild_indexes and not args.prepare_indexes:
        parser.error("--rebuild-indexes requires --prepare-indexes")
    try:
        run_dir = available_run_dir(args.results_root, args.run_id)
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    questions = load_questions(args.questions, processed_dir=args.processed_dir)
    if args.limit is not None:
        questions = _sample(questions, args.limit)
    parts: dict[Any, Any] = {}
    if args.matrix == "embedding":
        times = prepare_embedding_indexes(args.processed_dir, rebuild=args.rebuild_indexes) if args.prepare_indexes else None
        try:
            run_embedding_ablation(
                questions, lambda experiment: _default_embedding_stack(
                    experiment, processed_dir=args.processed_dir, parts=parts),
                run_id=args.run_id, results_root=args.results_root, top_k=args.top_k or 10,
                build_times=times,
            )
        except ValueError as error:
            command = shell_join([
                "python", "-m", "src.evaluation.model_ablation", "embedding", str(args.questions),
                "--prepare-indexes", "--processed-dir", str(args.processed_dir),
                "--results-root", str(args.results_root), "--run-id", _safe_run_id(args.run_id),
            ])
            parser.error(f"{error}; prepare E-indexes with {command}")
    else:
        try:
            run_generation_ablation(
                questions, lambda experiment: _default_generation_stack(
                    experiment, processed_dir=args.processed_dir, parts=parts, use_facts=args.use_facts),
                run_id=args.run_id, results_root=args.results_root,
            )
        except (RunStopped, RunInterrupted) as error:
            raise SystemExit(f"{error}; partial results are under {run_dir}") from None
        except ProviderUnavailable as error:
            parser.error(str(error))
    print((run_dir / "summary.csv").read_text(encoding="utf-8"), end="")
    print(f"written: {run_dir}")


if __name__ == "__main__":
    main()
