"""The experimental C0–C6 matrix (#44), separate from app adoption.

Retrieval covers both complete benchmark files; answers cover the complete team
file. A transactional JSONL row holds every configuration for one question.
The accepted C4/C5 reference can be imported after validating its frozen inputs,
query policy, corpus, settings and index identities. Imported timings retain
their original workload, rather than being presented as contemporaneous trials.
"""
from __future__ import annotations

import argparse
import json
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.rag.records import Answer

from filelock import FileLock

from src.config import PROCESSED_DIR
from src.pipeline.chunk import iter_chunks
from src.rag.decompose import decompose, merge
from src.retrieval.constants import CANDIDATE_K, FINAL_K
from src.retrieval.records import Query, RetrievedPassage
from src.retrieval.rerank import CrossEncoderReranker, RerankConfig
from src.stack import RESULTS_ROOT, STACKS, StackConfig, build_retriever

from .artifacts import dump_json as dump, read_jsonl
from .benchmark import load_questions
from .metrics import RETRIEVAL_METRICS, score_question
from .provenance import corpus_records_sha256, question_records_sha256, record_provenance, sha256
from .records import TEAM, BenchmarkQuestion, RunResult, benchmark_of
from .rerank_ablation import _comparable_manifest, _index_manifests, read_pairs, verify_rankings
from .results import _safe_run_id, write_comparison
from .run import FixedSizeBM25Retriever, _query, question_fields

METRICS = RETRIEVAL_METRICS


@dataclass(frozen=True)
class MatrixConfig(StackConfig):
    """Keep parsing/keyword/table policy fixed when removing metadata filters."""
    use_verification: bool = False
    rerank: bool = False

    def scoped(self, query: Query) -> Query:
        return query if self.metadata_filter else replace(
            query, tickers=(), fiscal_years=(), items=(), content_type=None)

    def to_dict(self) -> dict[str, Any]:
        return super().to_dict() | {"use_verification": self.use_verification, "rerank": self.rerank}


MATRIX = {
    **{key: replace(MatrixConfig(**value.to_dict()), top_k=FINAL_K, use_facts=False, use_decomposition=False,
                    use_refusal=False) for key, value in STACKS.items()},
    "C5": replace(MatrixConfig(**STACKS["C4"].to_dict()), id="C5", name="C4 + experimental MiniLM reranking",
                  top_k=FINAL_K, use_facts=False, use_decomposition=False, use_refusal=False, rerank=True),
    "C6": replace(MatrixConfig(**STACKS["C4"].to_dict()), id="C6", name="C4 + filing decomposition and verification",
                  top_k=FINAL_K, use_facts=False, use_decomposition=True, use_refusal=False,
                  use_verification=True, rerank=False),
}


# What a full-matrix run's config.json records as its schema.
FULL_MATRIX_SCHEMA = "full-matrix-v2"
REPLAYABLE_SCHEMAS = ("full-matrix-v1", FULL_MATRIX_SCHEMA)


def selected_configurations(args: argparse.Namespace) -> tuple[str, ...]:
    """The matrix rows a command works on: every row unless explicitly selected."""
    ids = tuple(getattr(args, "configurations", None) or MATRIX)
    if len(set(ids)) != len(ids) or any(cid not in MATRIX for cid in ids):
        raise ValueError("configurations must be unique matrix row IDs")
    return ids


def scored_for(question: BenchmarkQuestion, row: dict) -> BenchmarkQuestion:
    """The question scored against the gold passages a matrix row recorded for it."""
    return replace(question, supporting_chunk_ids=tuple(row["scoring_supporting_chunk_ids"]),
                   hard_negative_chunk_ids=tuple(row["scoring_hard_negative_chunk_ids"]))


def query_record(query: Query) -> dict:
    """A query as a run records it in pairs.jsonl, for comparing with a recorded one."""
    return json.loads(json.dumps(asdict(query)))


def read_rows(path: Path, *, recover: bool = False) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        return read_jsonl(stream, recover=recover)


def write_rows(path: Path, rows: Sequence[dict]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def check_transactions(transactions: Sequence[dict], questions: Sequence[BenchmarkQuestion]) -> None:
    if [t["question"] for t in transactions] != [q.to_dict() for q in questions[:len(transactions)]]:
        raise ValueError("stored matrix is not an exact benchmark prefix")
    for t in transactions:
        if set(t["rows"]) != set(MATRIX):
            raise ValueError("transaction does not contain every matrix row")
        for cid,row in t["rows"].items():
            if row["config"]["id"] != cid or row["question_id"] != t["question"]["question_id"]:
                raise ValueError("matrix row identity differs")


def validate_snapshot(config: dict, questions: Sequence[BenchmarkQuestion]) -> None:
    """A benchmark snapshot must remain the one the manifest froze."""
    provenance=config["provenance"]
    if (len(questions)!=provenance["questions"]
            or len({q.question_id for q in questions})!=len(questions)
            or question_records_sha256(questions)!=provenance["question_records_sha256"]):
        raise ValueError("benchmark snapshot differs from the frozen manifest")


def frozen_manifest_matches(path: Path, config: dict) -> bool:
    """Keep measurement identity, allowing runtime observations/paths to differ."""
    frozen = json.loads(path.read_text())
    return comparable_config(frozen) == comparable_config(json.loads(json.dumps(config)))


def comparable_config(config: dict) -> dict:
    value = json.loads(json.dumps(config))
    value.pop("workload_note", None)
    for key in ("provenance", "source"):
        if key in value:
            value[key] = _comparable_manifest({"provenance": value[key]})["provenance"]
    return value


RANKING_SOURCES = ("src/retrieval/", "src/evaluation/run.py", "src/rag/query.py",
                   "src/rag/constants.py", "src/stack.py", "src/config.py")


def ranking_sources(provenance: dict) -> dict[str, str]:
    hashes = provenance.get("source_files_sha256", {})
    required = ("src/retrieval/hybrid.py", "src/retrieval/rerank.py",
                "src/evaluation/run.py", "src/rag/query.py", "src/stack.py")
    if any(path not in hashes for path in required):
        raise ValueError("reference retrieval source hashes are missing")
    return {path: digest for path, digest in hashes.items() if path.startswith(RANKING_SOURCES)}


def summarize(rows: list[dict]) -> dict:
    result = {"questions": len(rows), "metric_denominators": {}}
    for metric in METRICS:
        values = [r[metric] for r in rows if r[metric] is not None]
        result[metric] = mean(values) if values else None
        result["metric_denominators"][metric] = len(values)
    result["median_latency_ms"] = median(r["latency_ms"] for r in rows) if rows else None
    return result


def reference_rows(
    path: Path, questions: Sequence[BenchmarkQuestion], provenance: dict,
    hybrid: Any, reranker: CrossEncoderReranker, *, expected_pairs_sha256: str,
) -> tuple[list[dict], dict]:
    if not re.fullmatch(r"[0-9a-fA-F]{64}", expected_pairs_sha256 or ""):
        raise ValueError("--reference requires --reference-pairs-sha256 from the archived manifest")
    actual_digest = sha256(path / "pairs.jsonl.gz")
    if actual_digest != expected_pairs_sha256.lower():
        raise ValueError("reference pairs differ from the archived digest")
    config = json.loads((path / "config.json").read_text())
    runtime = json.loads((path / "runtime.json").read_text())
    original = config["provenance"]
    for key in ("question_records_sha256", "retrieval_settings"):
        if original[key] != provenance[key]:
            raise ValueError(f"reference {key} differs")
    if ranking_sources(original) != ranking_sources(provenance):
        raise ValueError("reference retrieval source differs; measure a new compatible reference")
    for package in ("torch","sentence-transformers","transformers","tokenizers","chromadb","rank-bm25","numpy"):
        if original.get("environment",{}).get("packages",{}).get(package) != provenance.get("environment",{}).get("packages",{}).get(package):
            raise ValueError(f"reference package version differs: {package}")
    if original["corpus"]["records_sha256"] != provenance["corpus"]["records_sha256"]:
        raise ValueError("reference corpus differs")
    # The new latency descriptor changes reporting, not rankings. A frozen
    # reference keeps its original timing definition; all scoring parameters
    # (including checkpoint/revision, depth and token limit) must still match.
    recorded_reranker = {key:value for key,value in config["reranker"].items() if key != "latency"}
    current_reranker = {key:value for key,value in reranker.config.to_dict().items() if key != "latency"}
    if config["top_k"] != FINAL_K or recorded_reranker != current_reranker:
        raise ValueError("reference reranking configuration differs")
    if any(runtime.get(key) != value for key, value in _index_manifests(hybrid).items()):
        raise ValueError("reference indexes differ")
    rows = read_pairs(path / "pairs.jsonl.gz")
    if [r["question"] for r in rows] != [q.to_dict() for q in questions]:
        raise ValueError("reference must contain the exact complete question sequence")
    for q, row in zip(questions, rows):
        if row["query"] != query_record(_query(q, top_k=FINAL_K, metadata_filter=True)):
            raise ValueError("reference query policy differs")
    try:
        verify_rankings(rows, config["top_k"], candidate_k=config["reranker"]["candidate_k"])
    except (KeyError, TypeError) as error:
        raise ValueError("reference saved rankings are incomplete or invalid") from error
    return rows, {"source_commit": original["source_commit"], "config_sha256": sha256(path / "config.json"),
                  "pairs_sha256": actual_digest, "expected_pairs_sha256": expected_pairs_sha256.lower(),
                  "runtime": runtime,
                  "reranker_latency_definition": config["reranker"].get(
                      "latency", "unrecorded in reference; preserve original timing measurement"),
                  "timing_note": "Imported original shared-retrieval/reranking timings and workload; not a new timing trial."}


def restored(chunks: dict, ids: list[str], scores: list[float], name: str, sources: Sequence[Sequence[str]] | None = None) -> list[RetrievedPassage]:
    if len(ids) != len(scores) or len(set(ids)) != len(ids):
        raise ValueError("invalid saved ranking")
    source_map=dict(zip(ids,sources or [[] for _ in ids]))
    return [RetrievedPassage.from_chunk(chunks[cid], score, i, name, tuple(source_map[cid]))
            for i, (cid, score) in enumerate(zip(ids, scores), 1)]


def ranking(
    question: BenchmarkQuestion, query: Query, passages: Sequence[RetrievedPassage],
    config_id: str, latency: float, *, scoring: BenchmarkQuestion | None = None,
    origin: str = "measured", **extra: Any,
) -> dict:
    scored = scoring or question
    config = MATRIX[config_id].to_dict() | {"candidate_k": CANDIDATE_K}
    name = passages[0].retriever if passages else config["retriever"]
    result = RunResult.from_passages(question.question_id, passages, retriever=name,
                                     latency_ms=latency, config=config)
    metrics = score_question(scored, result, k=FINAL_K)
    return {**question_fields(question), "query": asdict(query), "config": config,
            "retriever": name, "retrieved_chunk_ids": list(result.retrieved_chunk_ids),
            "retrieved_scores": list(result.retrieved_scores), "latency_ms": latency,
            "retrieved_sources":[list(p.sources) for p in passages],
            "scoring_supporting_chunk_ids": list(scored.supporting_chunk_ids),
            "scoring_hard_negative_chunk_ids": list(scored.hard_negative_chunk_ids),
            "origin": origin, **{m: metrics[m] for m in METRICS}, **extra}


class CachedPart:
    """One component search reused by C1/C2 and fusion; timing is explicit."""
    def __init__(self, inner: Any) -> None:
        self.inner, self.name = inner, inner.name
        self.entries = {}

    def search(self, query: Query, k: int | None = None) -> list[RetrievedPassage]:
        key = (query, k)
        if key not in self.entries:
            started = perf_counter()
            passages = self.inner.search(query, k=k)
            self.entries[key] = (passages, (perf_counter() - started) * 1000)
        return self.entries[key][0]


def measure_question(
    question: BenchmarkQuestion, fixed: FixedSizeBM25Retriever, hybrid: Any,
    reranker: CrossEncoderReranker, chunks: dict, reference: dict | None = None,
) -> dict:
    from src.retrieval.hybrid import HybridRetriever

    scoped = _query(question, top_k=FINAL_K, metadata_filter=True)
    raw = MATRIX["C3"].scoped(scoped)
    rows = {}
    started = perf_counter()
    naive = fixed.search(raw, k=FINAL_K)
    elapsed = (perf_counter() - started) * 1000
    supporting = tuple(fixed.relevant_chunk_ids(question))
    hard = tuple(fixed.relevant_chunk_ids(question, hard=True))
    projected = replace(question, supporting_chunk_ids=supporting,
                        hard_negative_chunk_ids=tuple(cid for cid in hard if cid not in supporting))
    rows["C0"] = ranking(question, raw, naive, "C0", elapsed, scoring=projected,
                            projected_label_conflicts=sum(cid in supporting for cid in hard))
    sparse, dense = CachedPart(hybrid.bm25), CachedPart(hybrid.dense)
    components = {}
    for config_id, component in (("C1", sparse), ("C2", dense)):
        pool = component.search(raw, k=CANDIDATE_K)
        elapsed = component.entries[(raw, CANDIDATE_K)][1]
        components[config_id] = elapsed
        rows[config_id] = ranking(question, raw, pool[:FINAL_K], config_id, elapsed,
                                    timing_note="Component top-50 search shared with C3; top 16 scored.")
    started = perf_counter()
    fused = HybridRetriever(sparse, dense).search(raw, k=FINAL_K)
    elapsed = (perf_counter() - started) * 1000 + sum(components.values())
    rows["C3"] = ranking(question, raw, fused, "C3", elapsed,
                            timing_note="Shared component search times plus measured fusion overhead.")
    if reference is None:
        started = perf_counter()
        pool = hybrid.search(scoped, k=CANDIDATE_K)
        retrieve_ms = (perf_counter() - started) * 1000
        rows["C4"] = ranking(question, scoped, pool[:FINAL_K], "C4", retrieve_ms)
        if MATRIX["C5"].rerank:
            ranked = reranker.rerank(scoped, pool)
            rows["C5"] = ranking(question, scoped, ranked.passages, "C5",
                                 retrieve_ms + ranked.diagnostics["latency_ms"],
                                 rerank_diagnostics=ranked.diagnostics)
        else:
            rows["C5"] = ranking(question, scoped, pool[:FINAL_K], "C5", retrieve_ms)
    else:
        for cid in ("C4", "C5"):
            r = reference[cid]
            metadata={p["chunk_id"]:p for p in reference["rerank"]["candidates"]}
            passages = restored(chunks, r["retrieved_chunk_ids"], r["retrieved_scores"],
                                 CrossEncoderReranker.name if MATRIX[cid].rerank else "hybrid",
                                 [metadata[c]["sources"] for c in r["retrieved_chunk_ids"]])
            rows[cid] = ranking(question, scoped, passages, cid, r["latency_ms"], origin="frozen_reference",
                                 reference_question_id=question.question_id)
    sub_questions = decompose(scoped) if MATRIX["C6"].use_decomposition else ()
    if not sub_questions:
        base = "C5" if MATRIX["C6"].rerank else "C4"
        rows["C6"] = {**rows[base], "config": MATRIX["C6"].to_dict() | {
                       "candidate_k": CANDIDATE_K}, "origin": f"{base}_identical_no_split", "sub_results": []}
    else:
        started = perf_counter()
        results, saved = [], []
        for sub in sub_questions:
            pool = hybrid.search(sub.query, k=CANDIDATE_K)
            ranked = reranker.rerank(sub.query, pool) if MATRIX["C6"].rerank else None
            passages = ranked.passages if ranked is not None else pool[:FINAL_K]
            results.append((sub, passages))
            saved.append({"label": sub.label, "query": asdict(sub.query),
                          "retrieved_chunk_ids": [p.chunk_id for p in passages],
                          "retrieved_scores": [p.score for p in passages],
                          "retrieved_sources":[list(p.sources) for p in passages],
                          "rerank_diagnostics": ranked.diagnostics if ranked is not None else None})
        passages, provenance = merge(results, FINAL_K)
        rows["C6"] = ranking(question, scoped, list(passages), "C6", (perf_counter()-started)*1000,
                                sub_results=saved, sub_question_provenance=provenance)
    return {"question": question.to_dict(), "rows": rows}


def export_retrieval(
    run_dir: Path, transactions: Sequence[dict], expected: int, *, run_id: str | None = None,
    config_file: str = "config.json", write_question_outputs: bool = True,
) -> None:
    if len(transactions) != expected:
        raise ValueError("cannot summarize an incomplete retrieval matrix")
    summaries, benchmarks = [], []
    for cid in MATRIX:
        rows = [t["rows"][cid] for t in transactions]
        directory = run_dir / cid
        directory.mkdir(exist_ok=True)
        if write_question_outputs:
            write_rows(directory / "questions.jsonl", rows)
        groups = {name: summarize([r for r in rows if benchmark_of(r["source"]) == name])
                  for name in sorted({benchmark_of(r["source"]) for r in rows})}
        summary = {"config": rows[0]["config"], **summarize(rows), "by_benchmark": groups,
                   "by_type": {kind: summarize([r for r in rows if r["question_type"] == kind])
                               for kind in sorted({r["question_type"] for r in rows})}}
        dump(directory / "summary.json", summary)
        summaries.append(summary)
        benchmarks += [{"config": summary["config"], "benchmark": name, **g} for name,g in groups.items()]
    cutoff=summaries[0]["config"]["top_k"]
    identifier=run_id or run_dir.name
    write_comparison(run_dir, run_id=identifier, top_k=cutoff, summaries=summaries,
                     extra_fields=("median_latency_ms",), phase="retrieval", complete=True,
                     config_file=config_file, evaluation_only=True)
    write_comparison(run_dir / "by-benchmark", run_id=identifier, top_k=cutoff,
                     summaries=benchmarks, extra_fields=("benchmark",), complete=True, evaluation_only=True)


def verified_baseline(run_dir: Path, snapshot: Sequence[dict], fixed: FixedSizeBM25Retriever | None = None) -> list[dict]:
    """Bind C0 answer text to the deterministic windows that retrieval measured."""
    if fixed is None:
        fixed = FixedSizeBM25Retriever(Path("unused"), chunks=snapshot)
    expected = list(fixed.baseline_chunks)
    saved = read_rows(run_dir / "baseline.jsonl")
    if corpus_records_sha256(saved) != corpus_records_sha256(expected):
        raise ValueError("C0 baseline changed")
    return saved


def retrieval(args: argparse.Namespace) -> None:
    snapshot = list(iter_chunks(processed_dir=args.processed_dir))
    chunks = {chunk["chunk_id"]: chunk for chunk in snapshot}
    questions = [q for path in args.questions for q in load_questions(
        path, processed_dir=args.processed_dir, chunk_ids=chunks)]
    if not questions or len({q.question_id for q in questions}) != len(questions):
        raise ValueError("benchmark questions must be nonempty and unique across files")
    provenance = record_provenance(args.questions, questions, args.processed_dir, chunks=snapshot)
    fixed = FixedSizeBM25Retriever(args.processed_dir, chunks=snapshot)
    baseline = list(fixed.baseline_chunks)
    hybrid = build_retriever("hybrid", processed_dir=args.processed_dir, parts={})
    if corpus_records_sha256(hybrid.bm25.chunks) != corpus_records_sha256(snapshot):
        raise ValueError("hybrid index differs from the frozen corpus snapshot")
    reranker = CrossEncoderReranker(RerankConfig(device=args.device))
    references, source = (None, None) if args.reference is None else reference_rows(
        args.reference, questions, provenance, hybrid, reranker,
        expected_pairs_sha256=args.reference_pairs_sha256)
    config = {
        "schema": FULL_MATRIX_SCHEMA, "provenance": provenance,
        "matrix": {cid: row.to_dict() for cid, row in MATRIX.items()},
        "reranker": reranker.config.to_dict(), "reference": source,
        "baseline_records_sha256": corpus_records_sha256(baseline),
        "workload_note": args.workload_note,
        "answer_scope": "Complete non-mechanical team benchmark; no facts shortcut or refusal shortcut.",
    }
    run_dir = args.results_root / _safe_run_id(args.run_id)
    if args.resume:
        if not run_dir.is_dir() or not frozen_manifest_matches(run_dir / "config.json", config):
            raise ValueError("resume requires the identical frozen measurement identity and run directory")
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        dump(run_dir / "config.json", config)
        write_rows(run_dir / "benchmark.jsonl", [q.to_dict() for q in questions])
    with FileLock(run_dir / "retrieval.lock", timeout=0):
        transactions = read_rows(run_dir / "pairs.jsonl", recover=True)
        check_transactions(transactions, questions)
        if args.resume:
            verified_baseline(run_dir, snapshot, fixed)
        else:
            write_rows(run_dir / "baseline.jsonl", baseline)
        runtime_path = run_dir / "runtime.json"
        previous = json.loads(runtime_path.read_text())["identity"] if runtime_path.exists() else None
        indexes = _index_manifests(hybrid)
        if args.resume:
            if previous is not None and any(previous.get(key) != value for key, value in indexes.items()):
                raise ValueError("runtime model or index identity changed")
            if previous is None and transactions:
                raise ValueError("cannot verify resumed indexes without the original runtime.json")
        if len(transactions) == len(questions):
            write_rows(run_dir / "pairs.jsonl", transactions)
            export_retrieval(run_dir, transactions, len(questions))
            return
        # Warm the encoders on a real nonempty pool, outside measured rows.
        warm_started = perf_counter()
        hybrid.search(MATRIX["C3"].scoped(_query(questions[0], top_k=FINAL_K, metadata_filter=True)),
                      k=CANDIDATE_K)
        model = reranker.load()
        warmup_question = None
        for question in questions:
            scoped = _query(question, top_k=FINAL_K, metadata_filter=True)
            pool = hybrid.search(scoped, k=CANDIDATE_K)
            if pool:
                reranker.rerank(scoped, pool)
                warmup_question = question.question_id
                break
        runtime = {**indexes, "dense": hybrid.dense.runtime_info,
                   "cross_encoder_device": str(model.device)}
        if source and source["runtime"]["dense_revision"] != runtime["dense"]["revision"]:
            raise ValueError("loaded dense revision differs from the reference")
        if args.resume and previous is not None and previous != runtime:
            raise ValueError("runtime model or index identity changed")
        segment = {
            "identity": runtime, "load_and_warmup_ms": (perf_counter() - warm_started) * 1000,
            "resumed_after_questions": len(transactions), "workload_note": args.workload_note,
            "warmup_question_id": warmup_question, "cross_encoder_warmed": warmup_question is not None,
            "environment": provenance["environment"], "inputs": provenance["inputs"],
            "processed_dir": str(args.processed_dir),
        }
        number = 0
        while (run_dir / f"runtime-{number:04d}.json").exists():
            number += 1
        dump(run_dir / f"runtime-{number:04d}.json", segment)
        if not runtime_path.exists():
            dump(runtime_path, segment)
        write_rows(run_dir / "pairs.jsonl", transactions)
        with (run_dir / "pairs.jsonl").open("a", encoding="utf-8") as stream:
            for i, question in enumerate(questions[len(transactions):], len(transactions)):
                transaction = measure_question(question, fixed, hybrid, reranker, chunks,
                                               None if references is None else references[i])
                stream.write(json.dumps(transaction, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                transactions.append(transaction)
                if (i + 1) % 25 == 0 or i + 1 == len(questions):
                    print(f"C0–C6: {i + 1}/{len(questions)}", flush=True)
        export_retrieval(run_dir, transactions, len(questions))


class ReplayRetriever:
    """Replay the frozen rankings through the real answer/judge pipeline."""
    name = "recorded-matrix-evidence"

    def __init__(self, row: dict, chunks: dict) -> None:
        self.row, self.chunks = row, chunks

    def search(self, query: Query, k: int | None = None) -> list[RetrievedPassage]:
        wanted = query.top_k if k is None else k
        for row in [self.row, *self.row.get("sub_results", [])]:
            if query_record(query) == row["query"]:
                return restored(self.chunks,row["retrieved_chunk_ids"],row["retrieved_scores"],self.name,
                                row.get("retrieved_sources"))[:wanted]
        raise ValueError("answer query differs from the measured query")

    def has_candidates(self, query: Query) -> None:
        return None


def team_questions(questions: Sequence[BenchmarkQuestion]) -> list[BenchmarkQuestion]:
    return [q for q in questions if q.benchmark == TEAM]


def answer_summary(rows: Sequence[dict], cid: str) -> dict:
    from .harness import _checks
    from .quality import quality_summary, resource_summary

    return {"questions":len(rows),"quality":quality_summary(rows),"resources":resource_summary(rows),
            "checks":_checks(rows),"verification_enabled":MATRIX[cid].use_verification}


def saved_answer(data: dict) -> Answer:
    """Restore the validated records for a completed answer awaiting its judge."""
    from src.rag.records import (Answer,Citation,SentenceCitations,GenerationConfig,
                                VerificationCheck,VerificationResult)
    values=dict(data)
    values["config"]=GenerationConfig(**data["config"])
    values["citations"]=tuple(Citation(**c) for c in data["citations"])
    values["passages"]=tuple(RetrievedPassage(**{**p,"sources":tuple(p["sources"])}) for p in data["passages"])
    values["sentences"]=tuple(SentenceCitations(**{k:v for k,v in s.items() if k not in ("flagged","citations")},
                                  citations=tuple(Citation(**c) for c in s["citations"])) for s in data["sentences"])
    verification=data["verification"]
    values["verification"]=None if verification is None else VerificationResult(
        question_type=verification["question_type"],version=verification["version"],
        checks=tuple(VerificationCheck(**c) for c in verification["checks"]))
    return Answer(**values)


def merge_cost(previous: dict, current: dict) -> dict:
    result=dict(current)
    for key in ("usd","input_tokens","output_tokens","observed_input_tokens","observed_output_tokens"):
        result[key]=None if previous[key] is None or current[key] is None else previous[key]+current[key]
    for key in ("model_calls","usage_calls","successful_calls"):result[key]=previous[key]+current[key]
    result["complete"]=result["usd"] is not None
    result["reason"]=None if result["complete"] else previous["reason"] or current["reason"]
    return result


def finish_pending_judge(path: Path, question: BenchmarkQuestion, judge: Any) -> dict:
    from .harness import _ModelCalls
    from .quality import JudgeInvalid, query_cost

    row=json.loads(path.read_text())
    answer=saved_answer(row["answer"])
    if row["question_id"] != question.question_id:
        raise ValueError("pending judge question differs")
    calls=_ModelCalls()
    try:
        result,_=calls.ask(question.question_id+":judge",lambda:judge(question,answer),
                          lambda r:{"input_tokens":r.get("input_tokens"),"output_tokens":r.get("output_tokens")})
        row["quality"].update({k:result[k] for k in ("faithfulness","correctness","faithfulness_reason","correctness_reason")},
                              method="llm_judge",error=None)
        if question.question_type=="unanswerable":row["quality"].update(correctness=0.0,correctness_reason="Answered an unanswerable question")
    except JudgeInvalid as error:
        row["quality"].update(method="judge_failed",error=str(error))
    finally:
        overhead=row["judge_overhead"]
        overhead["latency_ms"]+=calls.ms
        overhead["retry_wait_ms"]+=calls.waited_ms
        overhead["total_ms"]+=calls.ms+calls.waited_ms
        overhead["cost"]=merge_cost(overhead["cost"],query_cost(judge.config.provider,calls.usages,
                              attempts=calls.calls,prices=None,successful_calls=calls.successes))
        row["judge_resumed"]=True
        dump(path,row)
    return row


def answers(args: argparse.Namespace) -> None:
    from src.evaluation.harness import evaluate, RunStopped, RunInterrupted
    from src.evaluation.quality import LLMJudge
    from src.rag.generate import config_from_env, checked_model

    run_dir = args.run_dir
    original = json.loads((run_dir / "config.json").read_text())
    questions = [BenchmarkQuestion.from_mapping(r) for r in read_rows(run_dir / "benchmark.jsonl")]
    validate_snapshot(original,questions)
    if "matrix" in original and original["matrix"] != {
        cid: config.to_dict() for cid, config in MATRIX.items()
    }:
        raise ValueError("answer matrix settings differ from frozen retrieval; use its recorded source")
    team = team_questions(questions)
    configuration_ids = selected_configurations(args)
    transactions = read_rows(run_dir / "pairs.jsonl", recover=True)
    check_transactions(transactions,questions)
    selected = {t["question"]["question_id"]:t for t in transactions}
    if any(q.question_id not in selected for q in team):
        raise ValueError("complete team retrieval must finish before answers start")
    chunks = list(iter_chunks(processed_dir=args.processed_dir))
    if corpus_records_sha256(chunks) != original["provenance"]["corpus"]["records_sha256"]:
        raise ValueError("answer corpus differs from frozen retrieval corpus")
    baseline = verified_baseline(run_dir, chunks)
    indexed = {c["chunk_id"]: c for c in [*chunks, *baseline]}
    generation = config_from_env(provider=args.provider, model=args.model)
    configuration = {"retrieval_config_sha256":sha256(run_dir/"config.json"),
                     "generation":generation.to_dict(), "judge_rubric":"answer-quality-v1",
                     "facts_sha256":sha256(args.facts_file),
                     "questions":[q.to_dict() for q in team],
                     "source":record_provenance([],team,args.processed_dir,chunks=chunks),
                     "timing_note":"Answer pipeline replays recorded rankings; real retrieval timing is in the matrix rows."}
    with FileLock(run_dir / "answers.lock", timeout=0):
        target = run_dir / "answers-config.json"
        if target.exists():
            if not frozen_manifest_matches(target, configuration):
                raise ValueError("answer resume requires identical settings/source/facts")
        else:
            dump(target, configuration)
        number = 0
        while (run_dir / f"answers-runtime-{number:04d}.json").exists():
            number += 1
        dump(run_dir / f"answers-runtime-{number:04d}.json", {
            "source": configuration["source"], "facts_path": str(args.facts_file),
            "configurations": list(configuration_ids),
            "workload_note": getattr(args, "workload_note", "Answer replay over recorded evidence"),
        })
        llm = checked_model(generation, None)
        judge = LLMJudge(generation, llm=llm)
        for cid in configuration_ids:
            directory = run_dir/cid
            directory.mkdir(exist_ok=True)
            path = directory/"answers.jsonl"
            stored = read_rows(path,recover=True)
            if [r["question_id"] for r in stored] != [q.question_id for q in team[:len(stored)]]:
                raise ValueError("answer rows are not the exact team prefix")
            pending = directory / "pending-judge.json"
            if pending.exists() and any(
                row["question_id"] == json.loads(pending.read_text())["question_id"] for row in stored
            ):
                pending.unlink()
            write_rows(path,stored)
            with path.open("a",encoding="utf-8") as stream:
                for q in team[len(stored):]:
                    measured = selected[q.question_id]["rows"][cid]
                    scored = scored_for(q,measured)
                    if pending.exists():
                        row=finish_pending_judge(pending,scored,judge)
                    else:
                        try:
                            report = evaluate([scored],ReplayRetriever(measured,indexed),generation,
                                              run_id=f"{run_dir.name}-{cid}",llm=llm,top_k=FINAL_K,
                                              stack=MATRIX[cid],use_facts=False,use_refusal=False,
                                              use_decomposition=MATRIX[cid].use_decomposition,
                                              use_verification=MATRIX[cid].use_verification,
                                              facts_file=args.facts_file,judge=judge)
                        except (RunStopped,RunInterrupted) as error:
                            if error.report["results"] and error.report["stopped"].get("stage")=="judge":
                                dump(pending,error.report["results"][0])
                            raise
                        row=report["results"][0]
                    row=row | {"source":q.source,"matrix_config":measured["config"],
                          "measured_retrieval_latency_ms":measured["latency_ms"],
                          "retrieval_origin":measured["origin"]}
                    row["evidence_rerank_enabled"] = MATRIX[cid].rerank
                    stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n")
                    stream.flush()
                    stored.append(row)
                    if pending.exists():pending.unlink()
                    print(f"{cid} answers: {len(stored)}/{len(team)}",flush=True)
            dump(directory/"answer-summary.json",answer_summary(stored,cid))
        answered = [cid for cid in MATRIX if (run_dir/cid/"answer-summary.json").exists()]
        dump(run_dir/"answer-summary.json",{
            "complete":answered==list(MATRIX), "team_questions":len(team),
            "configurations":{cid:json.loads((run_dir/cid/"answer-summary.json").read_text())
                              for cid in answered},
        })


def rescore(args: argparse.Namespace) -> None:
    """Recompute recorded retrieval and answer aggregates without any model."""
    directory=args.run_dir
    config=json.loads((directory/"config.json").read_text())
    questions=[BenchmarkQuestion.from_mapping(q) for q in read_rows(directory/"benchmark.jsonl")]
    validate_snapshot(config,questions)
    transactions=read_rows(directory/"pairs.jsonl")
    check_transactions(transactions,questions)
    if len(transactions)!=config["provenance"]["questions"]:
        raise ValueError("cannot rescore an incomplete retrieval matrix")
    for q,t in zip(questions,transactions):
        for row in t["rows"].values():
            scored=scored_for(q,row)
            result=RunResult(q.question_id,row["retriever"],tuple(row["retrieved_chunk_ids"]),
                             tuple(row["retrieved_scores"]),row["latency_ms"])
            metrics=score_question(scored,result,k=row["config"]["top_k"])
            row.update({m:metrics[m] for m in METRICS})
    target=directory/"rescored"
    target.mkdir(exist_ok=True)
    export_retrieval(target,transactions,len(questions),run_id=directory.name,config_file="../config.json",
                     write_question_outputs=False)
    team=team_questions(questions)
    configuration_ids = selected_configurations(args)
    summaries={}
    for cid in configuration_ids:
        rows=read_rows(directory/cid/'answers.jsonl')
        if [r['question_id'] for r in rows]!=[q.question_id for q in team]:
            raise ValueError(f"cannot rescore incomplete {cid} team answers")
        summaries[cid]=answer_summary(rows,cid)
    dump(target/'answer-summary.json',{'complete':set(configuration_ids)==set(MATRIX),
         'team_questions':len(team),'configurations':summaries})


def main(argv: list[str] | None = None) -> None:
    from src.retrieval.facts import FACTS_FILE

    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest="command",required=True)
    run=commands.add_parser("retrieval")
    run.add_argument("questions",type=Path,nargs="+")
    run.add_argument("--run-id",required=True)
    run.add_argument("--results-root",type=Path,default=RESULTS_ROOT)
    run.add_argument("--processed-dir",type=Path,default=PROCESSED_DIR)
    run.add_argument("--reference",type=Path)
    run.add_argument("--reference-pairs-sha256",help="Expected pairs hash from the archived C5 manifest")
    run.add_argument("--device",default="mps")
    run.add_argument("--workload-note",required=True)
    run.add_argument("--resume",action="store_true")
    answer=commands.add_parser("answers")
    answer.add_argument("run_dir",type=Path)
    answer.add_argument("--processed-dir",type=Path,default=PROCESSED_DIR)
    answer.add_argument("--facts-file",type=Path,default=FACTS_FILE)
    answer.add_argument("--provider",choices=("mistral","ollama"),default="mistral")
    answer.add_argument("--model")
    answer.add_argument("--configurations",nargs="+",choices=tuple(MATRIX),
                        help="Answer only these matrix rows (default: every row)")
    answer.add_argument("--workload-note", default="Answer replay over recorded evidence")
    scoring=commands.add_parser("rescore")
    scoring.add_argument("run_dir",type=Path)
    scoring.add_argument("--configurations",nargs="+",choices=tuple(MATRIX),
                         help="Rescore answers only for these matrix rows (default: every row)")
    args=parser.parse_args(argv)
    if args.command == "retrieval":
        if bool(args.reference) != bool(args.reference_pairs_sha256):
            parser.error("--reference and --reference-pairs-sha256 must be supplied together")
    {"retrieval":retrieval,"answers":answers,"rescore":rescore}[args.command](args)


if __name__=="__main__":
    main()
