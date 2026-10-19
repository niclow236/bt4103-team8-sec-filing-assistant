"""Compare E1-E3 embeddings and G1-G3 providers with reproducible reports."""

from __future__ import annotations

import argparse
import json
import logging
import threading
from collections.abc import Callable, Sequence
from dataclasses import MISSING, asdict, dataclass, fields, replace
from math import ceil
from pathlib import Path
from statistics import mean, median
from tempfile import TemporaryDirectory
from shlex import join as shell_join
from time import perf_counter
from types import SimpleNamespace
from typing import Any

from src.config import PROCESSED_DIR, PROJECT_ROOT
from src.rag import ProviderUnavailable
from src.rag.generate import without_answer_locally
from src.rag.records import GenerationConfig
from src.rag.constants import (
    DEFAULT_MODEL, DEFAULT_MISTRAL_MODEL, DEFAULT_SOCLAAS_MODEL, PROVIDERS,
)
from src.retrieval.embedding_config import EmbeddingConfig
from src.retrieval.facts import FACTS_FILE
from src.stack import RESULTS_ROOT, build_retriever, build_stack, stack_config

from .benchmark import DEFAULT_QUESTIONS_PATH, load_questions
from .harness import RunInterrupted, RunStopped, evaluate, logger as harness_log
from .metrics import score_question
from .records import UNANSWERABLE, BenchmarkQuestion, RunResult
from .results import available_run_dir, write_comparison
from .run import _query, _safe_run_id, _summary, question_fields


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
        "E2", "MiniLM", "sentence-transformers/all-MiniLM-L6-v2", "data/index/chroma-E2", 384, "", "", 256,
    ),
    EmbeddingExperiment(
        "E3", "E5 base", "intfloat/e5-base-v2", "data/index/chroma-E3", 768, "query: ", "passage: ", 512,
    ),
)
GENERATION_EXPERIMENTS = (
    GenerationExperiment("G1", "Local Ollama", "ollama", DEFAULT_MODEL),
    GenerationExperiment("G2", "Hosted Mistral", "mistral", DEFAULT_MISTRAL_MODEL),
    GenerationExperiment("G3", "Hosted SOCLAAS", "soclaas", DEFAULT_SOCLAAS_MODEL),
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
                    **question_fields(question),
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
    """Measure every provider with each stack's complete answer settings."""
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
        question_by_id = {question.question_id: question for question in questions}
        for row in report.get("results", []):
            question = question_by_id.get(row.get("question_id"))
            if question is not None:
                row.update(question_fields(question))
        summaries.append(_generation_summary(experiment, report))
        _write_report(run_dir, experiment.id, report)
        if stopped is not None:
            break
    manifest = write_comparison(run_dir, run_id=run_id, top_k=top_k, summaries=summaries,
                                extra_fields=MODEL_FIELDS, matrix="generation")
    if stopped is not None:
        raise stopped
    return manifest


# --- G rows over a full-matrix run's recorded evidence (#163) ----------------------
#
# A G row run above answers through the app's own C4 stack. To set a generator
# beside the answers a full-matrix run (#161) recorded for C4, it has to answer
# over the same evidence, with the same settings and the same judge, so that the
# generator is the only change. The replay below does that: it hands the run's
# recorded C4 rankings to the harness in place of a retriever, with the matrix's
# C4 settings (no facts, decomposition or refusal shortcuts), and judges with the
# model and rubric that judged the recorded answers. The recorded answers are
# summarised beside it by the same function, as the G row that wrote them.

REPLAY_ROW = "C4"
REPLAY_FIELDS = (
    "provider", "model", "judge", "correctness", "correctness_scored", "faithfulness",
    "unanswerable_abstention", "answerable_abstention", "citation_precision",
    "invalid_citation_answers", "schema_failures", "truncated", "judge_failures",
    "median_generate_ms", "p95_generate_ms", "mean_input_tokens", "mean_output_tokens",
    "retried_questions", "stopped",
)


@dataclass(frozen=True)
class RecordedAnswers:
    """A full-matrix run's C4 answers and the evidence they were written over.

    ``questions`` are the team questions, each scored against the gold passages
    the run recorded for C4; ``retriever`` replays the run's C4 rankings;
    ``generation`` is the config that wrote ``rows`` and, with the
    ``answer-quality-v1`` rubric, judged them. ``facts`` is the facts store, as
    read from ``facts_file`` when each question's query was checked against
    it, or None where there was none; a replay writes it into its run
    directory and parses with that copy, so a rebuild of the store while the
    replay runs cannot change a query part-way through.
    """

    run_dir: Path
    questions: tuple[BenchmarkQuestion, ...]
    retriever: Any
    generation: GenerationConfig
    rows: tuple[dict[str, Any], ...]
    facts_file: Path = FACTS_FILE
    facts: bytes | None = None


class _RecordedRankings:
    """Replays every question's recorded C4 ranking, found by the query it answered."""

    def __init__(self, rows: Sequence[dict[str, Any]], chunks: dict[str, dict]) -> None:
        from src.evaluation.full_ablation import ReplayRetriever

        self.replays = [ReplayRetriever(row, chunks) for row in rows]
        self.name = ReplayRetriever.name

    def search(self, query: Any, k: int | None = None) -> list[Any]:
        for replay in self.replays:
            try:
                return replay.search(query, k)
            except ValueError:
                continue
        raise ValueError("answer query differs from every measured query")

    def has_candidates(self, query: Any) -> None:
        return None


def recorded_answers(run_dir: Path, *, processed_dir: Path = PROCESSED_DIR,
                     chunks: Sequence[dict] | None = None,
                     facts_file: Path = FACTS_FILE) -> RecordedAnswers:
    """The C4 answers a complete full-matrix run recorded, ready to replay.

    Refused unless the benchmark snapshot, the matrix settings and the corpus
    are the ones the run froze, its retrieval covered every team question, and
    its C4 answers are complete, in order, and every team question's query,
    built as the harness will build it, is the one the run recorded. A changed
    facts store or company list can change a query; caught here, it stops the
    replay before any model is asked, rather than part-way through. The check
    reads the store from ``facts_file`` once, and the replay parses with what
    it read. ``chunks`` is the corpus, read from ``processed_dir`` unless given.
    """
    from src.evaluation.full_ablation import (
        FULL_MATRIX_SCHEMA, MATRIX, MatrixConfig, check_transactions, read_rows, scored_for,
        team_questions, validate_snapshot,
    )
    from src.evaluation.provenance import corpus_records_sha256
    from src.evaluation.quality import RUBRIC_ID
    from src.pipeline.chunk import iter_chunks

    if not run_dir.is_dir():
        raise ValueError(f"{run_dir} does not exist: replay reads a full-matrix run's folder, "
                         f"which stays on the machine that ran it, so copy it here first")
    if not (run_dir / "config.json").is_file():
        raise ValueError(f"{run_dir} is not a full-matrix run: it has no {run_dir / 'config.json'}")
    config = json.loads((run_dir / "config.json").read_text())
    if not isinstance(config, dict) or config.get("schema") != FULL_MATRIX_SCHEMA:
        raise ValueError(f"{run_dir} is not a full-matrix run: its config.json does not record "
                         f"the schema {FULL_MATRIX_SCHEMA!r}")
    if not (run_dir / "answers-config.json").is_file():
        raise ValueError(f"{run_dir} has no answers yet: it has no "
                         f"{run_dir / 'answers-config.json'}; finish its `answers` step first")
    questions = [BenchmarkQuestion.from_mapping(row) for row in read_rows(run_dir / "benchmark.jsonl")]
    validate_snapshot(config, questions)
    # A setting the matrix gained after the run, such as rerank after
    # full-matrix-20261017, was not recorded and ran at its default.
    defaults = {field.name: field.default for field in fields(MatrixConfig)
                if field.default is not MISSING}
    frozen = config.get("matrix", {}).get(REPLAY_ROW) or {}
    current = MATRIX[REPLAY_ROW].to_dict()
    if not frozen or any(frozen.get(name, defaults.get(name, MISSING)) != value
                         for name, value in current.items()) or set(frozen) - set(current):
        raise ValueError(f"{REPLAY_ROW}'s settings differ from the run's; use its recorded source")
    team = team_questions(questions)
    transactions = read_rows(run_dir / "pairs.jsonl")
    check_transactions(transactions, questions)
    measured = {t["question"]["question_id"]: t["rows"][REPLAY_ROW] for t in transactions}
    if any(question.question_id not in measured for question in team):
        raise ValueError("the run's team retrieval is incomplete")
    answers_config = run_dir / "answers-config.json"
    answered = json.loads(answers_config.read_text())
    for name in ("judge_rubric", "generation"):
        if not isinstance(answered, dict) or name not in answered:
            raise ValueError(f"{answers_config} does not record {name!r}, which a complete "
                             f"`answers` step writes")
    if answered["judge_rubric"] != RUBRIC_ID:
        raise ValueError(f"the run was judged with {answered['judge_rubric']!r}, not {RUBRIC_ID!r}")
    try:
        generation = GenerationConfig(**answered["generation"])
    except TypeError:
        generation = None
    # The config does not check its fields, and a provider the generator does
    # not know fails only when the judge is built from it.
    if generation is None or generation.provider not in PROVIDERS or not all(
            isinstance(value, str) for value in (generation.model, generation.prompt_template_id)):
        raise ValueError(f"{answers_config}'s 'generation' is not a generation config: "
                         f"{answered['generation']!r}")
    rows = read_rows(run_dir / REPLAY_ROW / "answers.jsonl")
    if [row["question_id"] for row in rows] != [question.question_id for question in team]:
        raise ValueError(f"the run's {REPLAY_ROW} answers do not cover its team questions in order")
    snapshot = list(chunks) if chunks is not None else list(iter_chunks(processed_dir=processed_dir))
    if corpus_records_sha256(snapshot) != config["provenance"]["corpus"]["records_sha256"]:
        raise ValueError("this corpus differs from the one the run measured")
    scored = tuple(scored_for(question, measured[question.question_id]) for question in team)
    # Read once. A store that does not exist, as on a fresh clone, is parsed
    # as missing, the way the run that recorded the queries would have.
    facts = Path(facts_file).read_bytes() if Path(facts_file).is_file() else None
    with TemporaryDirectory(prefix="replay-facts-") as scratch:
        checked = _facts_copy(facts, facts_file, Path(scratch))
        _check_queries(scored, measured, checked, store=Path(facts_file))
    return RecordedAnswers(
        run_dir=run_dir, questions=scored,
        retriever=_RecordedRankings([measured[q.question_id] for q in team],
                                    {chunk["chunk_id"]: chunk for chunk in snapshot}),
        generation=generation, rows=tuple(rows),
        facts_file=Path(facts_file), facts=facts,
    )


def _facts_copy(facts: bytes | None, facts_file: Path, directory: Path) -> Path:
    """The facts store as read, written into ``directory`` under its own name.

    Where there was no store to read, the path is returned unwritten, so it is
    parsed as missing however long the caller runs: a store built at
    ``facts_file`` in the meantime is not read.
    """
    copy = directory / Path(facts_file).name
    if facts is not None:
        copy.write_bytes(facts)
    return copy


def _check_queries(scored: Sequence[BenchmarkQuestion], measured: dict[str, dict],
                   snapshot: Path, *, store: Path) -> None:
    """Refuse a run whose recorded C4 query for any question is not the one the
    harness would build now, parsing with ``snapshot``, the copy of ``store``
    that the refusal names."""
    from src.evaluation.full_ablation import MATRIX, query_record
    from src.evaluation.harness import benchmark_query
    from src.rag.query import parse_question

    stack = MATRIX[REPLAY_ROW]
    for question in scored:
        parsed = parse_question(question.question, facts_file=snapshot)
        query = stack.scoped(benchmark_query(parsed, question, top_k=stack.top_k))
        if query_record(query) != measured[question.question_id]["query"]:
            raise ValueError(
                f"{question.question_id}'s query {query.text!r} differs from the one the run "
                f"recorded; the facts store ({store}) or config/companies.txt has changed "
                f"since the run, so its rankings cannot be replayed")


def _nearest_rank(ordered: Sequence[float], share: float) -> float | None:
    """The nearest-rank percentile of values already in order: the smallest
    value at or above ``share`` of them."""
    return ordered[max(ceil(share * len(ordered)), 1) - 1] if ordered else None


def _mean_of(values: Sequence[Any]) -> float | None:
    known = [value for value in values if value is not None]
    return mean(known) if known else None


def _replay_summary(experiment: GenerationExperiment, rows: Sequence[dict[str, Any]], *,
                    judge: GenerationConfig, origin: str) -> dict[str, Any]:
    """What #163 compares generators on, over one G row's answers.

    Correctness and faithfulness are the judge's means over the answers it
    scored. A schema failure is an answer that failed the schema without being
    cut at the output ceiling; a truncated one is counted on its own. An
    answer with an invalid or unresolved citation marker counts once however
    many it has. Latency is generation alone, with the 95th percentile taken
    by nearest rank, and tokens are the generator's, not the judge's.
    """
    from .quality import quality_summary

    quality = quality_summary(list(rows))
    answers = [row["answer"] for row in rows]
    generate_ms = sorted(row["latency_ms"]["generate"] for row in rows)

    def abstention(selected):
        return mean(row["answer"]["abstained"] for row in selected) if selected else None

    return {
        "config": {**asdict(experiment), "origin": origin},
        "questions": len(rows),
        "recall": None, "ndcg": None, "mrr": None, "hard_negative_accuracy": None,
        "judge": f"{judge.provider}/{judge.model}",
        "correctness": quality["correctness"]["mean"],
        "correctness_scored": quality["correctness"]["scored"],
        "faithfulness": quality["faithfulness"]["mean"],
        "unanswerable_abstention": abstention(
            [row for row in rows if row["question_type"] == UNANSWERABLE]),
        "answerable_abstention": abstention(
            [row for row in rows if row["question_type"] != UNANSWERABLE]),
        "citation_precision": quality["citation_precision"]["mean"],
        "invalid_citation_answers": sum(
            bool(row["quality"]["invalid_markers"] or row["quality"]["unresolved_markers"])
            for row in rows),
        "schema_failures": sum(bool(a["parse_error"]) and not a["truncated"] for a in answers),
        "truncated": sum(bool(a["truncated"]) for a in answers),
        "judge_failures": sum(row["quality"]["method"] == "judge_failed" for row in rows),
        "median_generate_ms": median(generate_ms) if generate_ms else None,
        "p95_generate_ms": _nearest_rank(generate_ms, 0.95),
        "mean_input_tokens": _mean_of([row["cost"]["input_tokens"] for row in rows]),
        "mean_output_tokens": _mean_of([row["cost"]["output_tokens"] for row in rows]),
        "retried_questions": sum(row.get("attempts", 1) > 1 for row in rows),
    }


class _WithoutAnswerLocally(logging.Filter):
    """Drops the advice to set ``LLM_PROVIDER`` from a notice logged on the
    thread that made it, as :func:`without_answer_locally` does from a
    failure's message. Another thread's run may be one LLM_PROVIDER chooses."""

    def __init__(self) -> None:
        super().__init__()
        self.thread = threading.get_ident()

    def filter(self, record: logging.LogRecord) -> bool:
        # A filter runs on the thread that logs, and record.thread is None
        # when logging.logThreads is off.
        if threading.get_ident() == self.thread:
            record.msg, record.args = without_answer_locally(record.getMessage()), ()
        return True


def run_generation_replay(
    recorded: RecordedAnswers, experiments: Sequence[GenerationExperiment], *, run_id: str,
    results_root: Path = RESULTS_ROOT, llm_for: Callable[[GenerationConfig], Any] | None = None,
    judge: Any | None = None,
) -> dict[str, Any]:
    """Answer the recorded C4 evidence with each G row, judged as the recorded answers were.

    Each row's config is the recorded one with only the provider and model
    changed, and a row repeating the recorded answers' own model is named
    ``<id>-replay``. ``llm_for`` builds each chat model, :func:`checked_model`
    unless given; every one is built, and the judge with them, before the run
    directory is made, so a missing key stops the run before it writes
    anything. A stopped or interrupted row keeps its answers, as
    :func:`run_generation_ablation` does.
    """
    from src.evaluation.full_ablation import MATRIX
    from src.evaluation.quality import LLMJudge
    from src.rag.generate import checked_model

    run_dir = available_run_dir(results_root, run_id)
    llm_for = llm_for or checked_model
    judge = judge if judge is not None else LLMJudge(recorded.generation)
    recorded_model = (recorded.generation.provider, recorded.generation.model)

    def wrote_recorded(experiment: GenerationExperiment) -> bool:
        return (experiment.provider, experiment.model) == recorded_model

    # A row repeating the model that wrote the recorded answers is named apart
    # from them, so the table never shows two rows with one id (#163).
    answering = [(replace(experiment, id=f"{experiment.id}-replay")
                  if wrote_recorded(experiment) else experiment,
                  replace(recorded.generation, provider=experiment.provider, model=experiment.model))
                 for experiment in experiments]
    models = [llm_for(config) for _, config in answering]
    run_dir.mkdir(parents=True)
    # The store the queries were checked against, kept with the run.
    facts_file = _facts_copy(recorded.facts, recorded.facts_file, run_dir)
    stack = MATRIX[REPLAY_ROW]
    origin = f"{recorded.run_dir.name}/{REPLAY_ROW}"
    recorded_experiment = next(
        (experiment for experiment in GENERATION_EXPERIMENTS if wrote_recorded(experiment)),
        GenerationExperiment("recorded", "Recorded answers", *recorded_model))
    summaries = [_replay_summary(recorded_experiment, recorded.rows, judge=judge.config,
                                 origin=origin)]
    question_by_id = {question.question_id: question for question in recorded.questions}
    stopped = None
    # The harness's retry notices are the provider's advice too (#180).
    notices = _WithoutAnswerLocally()
    for (experiment, config), llm in zip(answering, models):
        harness_log.addFilter(notices)
        try:
            report = evaluate(
                list(recorded.questions), recorded.retriever, config,
                run_id=f"{run_id}-{experiment.id}", llm=llm, top_k=stack.top_k, stack=stack,
                use_facts=False, use_refusal=False, use_decomposition=stack.use_decomposition,
                use_verification=stack.use_verification, facts_file=facts_file,
                judge=judge,
            )
        except (RunStopped, RunInterrupted) as error:
            report = error.report
            # The row named the provider, so LLM_PROVIDER is no way out (#180).
            # Raised afresh, as its message was made from the report before this.
            report["stopped"]["error"] = without_answer_locally(report["stopped"]["error"])
            stopped = type(error)(report)
            stopped.__cause__ = error.__cause__
        finally:
            harness_log.removeFilter(notices)
        if recorded.facts is None:
            # The path it parsed with was never written: there was no store.
            report["facts_file"] = None
        for row in report.get("results", []):
            question = question_by_id.get(row.get("question_id"))
            if question is not None:
                row.update(question_fields(question))
        _write_report(run_dir, experiment.id, report)
        summaries.append(_replay_summary(experiment, report.get("results", []),
                                         judge=judge.config, origin="replayed")
                         | {"stopped": report.get("stopped")})
        if stopped is not None:
            break
    manifest = write_comparison(
        run_dir, run_id=run_id, top_k=stack.top_k, summaries=summaries,
        extra_fields=REPLAY_FIELDS, matrix="generation-replay",
        evidence={"run": recorded.run_dir.name, "row": REPLAY_ROW}, judge=judge.to_dict(),
    )
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


def _require_default_corpus(processed_dir: Path) -> None:
    if processed_dir.resolve() != PROCESSED_DIR.resolve():
        raise ValueError(
            "--prepare-indexes brings data/index/chroma, the app's own index, in line "
            "with --processed-dir and removes every other vector; use the default corpus"
        )


def prepare_embedding_indexes(processed_dir: Path, *, rebuild: bool = False) -> dict[str, float]:
    """Incrementally prepare the default corpus's E-indexes; return build times."""
    _require_default_corpus(processed_dir)
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


def replay_main(argv: list[str]) -> None:
    """``replay RUN_DIR --rows G3``: G rows over a full-matrix run's recorded C4 evidence."""
    parser = argparse.ArgumentParser(
        prog="python -m src.evaluation.model_ablation replay",
        description="Answer a complete full-matrix run's recorded C4 evidence with each named G "
                    "row, judged by the model and rubric that judged the run's own answers, and "
                    "table them beside those answers.")
    parser.add_argument("run_dir", type=Path, help="A complete full-matrix run, such as "
                                                   "results/full-matrix-20261017.")
    parser.add_argument("--rows", nargs="+", required=True,
                        choices=[experiment.id for experiment in GENERATION_EXPERIMENTS])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    args = parser.parse_args(argv)
    try:
        run_dir = available_run_dir(args.results_root, args.run_id)
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    experiments = [experiment for experiment in GENERATION_EXPERIMENTS if experiment.id in args.rows]
    try:
        run_generation_replay(recorded_answers(args.run_dir, processed_dir=args.processed_dir),
                              experiments, run_id=args.run_id, results_root=args.results_root)
    # Each row and the judge name their own provider, so LLM_PROVIDER is no way out.
    except (RunStopped, RunInterrupted) as error:
        raise SystemExit(f"{without_answer_locally(str(error))}; partial results are under "
                         f"{run_dir}") from None
    except (ValueError, ProviderUnavailable) as error:
        parser.error(without_answer_locally(str(error)))
    print((run_dir / "summary.csv").read_text(encoding="utf-8"), end="")
    print(f"written: {run_dir}")


def main(argv: list[str] | None = None) -> None:
    import sys

    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["replay"]:
        return replay_main(argv[1:])
    # `replay` is sent to its own parser above, so this one only points to it.
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="replay: answer a full-matrix run's recorded C4 evidence with G rows; see "
               "python -m src.evaluation.model_ablation replay --help")
    parser.add_argument("matrix", choices=("embedding", "generation"))
    parser.add_argument("questions", type=Path, nargs="?", default=DEFAULT_QUESTIONS_PATH)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--prepare-indexes", action="store_true", help="Incrementally prepare E1-E3 indexes from the default corpus.")
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
    if args.prepare_indexes:
        try:
            _require_default_corpus(args.processed_dir)
        except ValueError as error:
            parser.error(str(error))
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
                "--prepare-indexes", "--processed-dir", str(PROCESSED_DIR),
                "--results-root", str(args.results_root), "--run-id", _safe_run_id(args.run_id),
            ])
            parser.error(
                "an index the E rows need is not ready. If it is E2 or E3, prepare it "
                f"from the default corpus with:\n  {command}\n"
                "The advice below names the command for the default index and for BM25.\n"
                f"{error}"
            )
    else:
        try:
            run_generation_ablation(
                questions, lambda experiment: _default_generation_stack(
                    experiment, processed_dir=args.processed_dir, parts=parts, use_facts=args.use_facts),
                run_id=args.run_id, results_root=args.results_root,
            )
        except (RunStopped, RunInterrupted) as error:
            raise SystemExit(f"{error}; partial results are under {run_dir}") from None
        # Each G row names its own provider, so LLM_PROVIDER is no way out (#183).
        except ProviderUnavailable as error:
            parser.error(without_answer_locally(str(error)))
    print((run_dir / "summary.csv").read_text(encoding="utf-8"), end="")
    print(f"written: {run_dir}")


if __name__ == "__main__":
    main()
