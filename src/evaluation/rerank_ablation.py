"""Paired C4/C5 retrieval experiment: run, resume, or rescore stored rankings."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from typing import Any, Sequence

import numpy as np
from filelock import FileLock, Timeout

from src.config import PROCESSED_DIR
from src.pipeline.chunk import iter_chunks
from src.retrieval.base import Retriever
from src.retrieval.constants import CANDIDATE_K, FINAL_K
from src.retrieval.rerank import CrossEncoderReranker, RerankConfig
from src.stack import RESULTS_ROOT, build_retriever

from .benchmark import load_questions
from .metrics import score_question
from .provenance import corpus_records_sha256, record_provenance
from .records import BenchmarkQuestion, RunResult
from .results import _safe_run_id
from .run import _query

METRICS = ("recall", "ndcg", "mrr", "hard_negative_accuracy")
POLICY = {
    "decision_top_k": 16,
    "decision_candidate_k": 50,
    "primary_sources": ["handwritten", "xbrl"],
    "min_ndcg_gain": 0.02,
    "max_recall_mrr_regression": 0.01,
    "max_llm_assisted_ndcg_regression": 0.02,
    "max_type_ndcg_regression": 0.02,
    "min_type_questions": 10,
    "max_hard_negative_regression": 0.01,
    "max_rerank_median_ms": 500,
    "max_rerank_p95_ms": 1000,
    "uncertainty": "95% percentile paired bootstrap of connected filing clusters",
    "bootstrap_samples": 2000,
    "seed": 154,
}


def _dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _rank_record(question: BenchmarkQuestion, passages: list, latency: float, config: str) -> dict:
    result = RunResult.from_passages(
        question.question_id, passages, retriever="hybrid" if config == "C4" else "hybrid-reranked-experimental",
        latency_ms=latency, config={"id": config},
    )
    return {"retrieved_chunk_ids": list(result.retrieved_chunk_ids),
            "retrieved_scores": list(result.retrieved_scores), "latency_ms": latency}


def _scores(row: dict, config: str, k: int) -> dict:
    question = BenchmarkQuestion.from_mapping(row["question"])
    record = row[config]
    result = RunResult(question.question_id, config,
                       tuple(record["retrieved_chunk_ids"]), tuple(record["retrieved_scores"]),
                       record["latency_ms"])
    return score_question(question, result, k=k)


def _clusters(rows: Sequence[dict]) -> list[str]:
    """Join filings linked by a multi-filing question before resampling.

    This avoids treating overlapping questions from the same filing as
    independent. Unanswerable questions without a filing get their own cluster.
    """
    parent: dict[str, str] = {}

    def root(key: str) -> str:
        parent.setdefault(key, key)
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    keys = []
    for row in rows:
        filings = row["filings"] or ["question:" + row["question"]["question_id"]]
        first = root(filings[0])
        for filing in filings[1:]:
            parent[root(filing)] = root(first)
        keys.append(filings[0])
    return [root(key) for key in keys]


def paired_metric(rows: Sequence[dict], metric: str, k: int, policy: dict = POLICY) -> dict:
    selected = []
    for row in rows:
        a, b = _scores(row, "C4", k)[metric], _scores(row, "C5", k)[metric]
        if a is not None and b is not None:
            selected.append((row, a, b))
    if not selected:
        return {"questions": 0, "C4": None, "C5": None, "delta": None,
                "wins": 0, "losses": 0, "ties": 0, "ci95": None, "clusters": 0}
    grouped: dict[str, list[float]] = defaultdict(list)
    # Build clusters from all rows so connections are retained even when a
    # question itself has no gold for this metric.
    identities = dict(zip((r["question"]["question_id"] for r in rows), _clusters(rows)))
    diffs = []
    for row, a, b in selected:
        diff = b - a
        diffs.append(diff)
        grouped[identities[row["question"]["question_id"]]].append(diff)
    values = list(grouped.values())
    sums = np.array([sum(v) for v in values])
    sizes = np.array([len(v) for v in values])
    ci = None
    if len(values) >= 2:
        indices = np.random.default_rng(policy["seed"]).integers(
            0, len(values), size=(policy["bootstrap_samples"], len(values)),
        )
        draws = sums[indices].sum(axis=1) / sizes[indices].sum(axis=1)
        ci = [float(x) for x in np.quantile(draws, [0.025, 0.975])]
    return {"questions": len(selected), "C4": mean(a for _, a, _ in selected),
            "C5": mean(b for _, _, b in selected), "delta": mean(diffs),
            "wins": sum(d > 1e-12 for d in diffs), "losses": sum(d < -1e-12 for d in diffs),
            "ties": sum(abs(d) <= 1e-12 for d in diffs), "ci95": ci, "clusters": len(values)}


def summarize(
    rows: Sequence[dict], k: int, policy: dict = POLICY, *, candidate_k: int = CANDIDATE_K,
) -> dict:
    groups: dict[str, dict[str, list]] = {
        "by_source": defaultdict(list), "by_type": defaultdict(list),
        "by_source_type": defaultdict(list), "by_gold_content": defaultdict(list),
        "by_gold_truncation": defaultdict(list),
    }
    for row in rows:
        groups["by_source"][row["question"]["source"]].append(row)
        groups["by_type"][row["question"]["question_type"]].append(row)
        groups["by_source_type"][row["question"]["source"] + ":" + row["question"]["question_type"]].append(row)
        groups["by_gold_content"][row["gold_content"]].append(row)
        gold_ids = set(row["question"]["supporting_chunk_ids"])
        gold_in_pool = [p for p in row["rerank"]["candidates"] if p["chunk_id"] in gold_ids]
        truncated_gold = ("unlabelled" if not gold_ids else "no_gold_in_pool" if not gold_in_pool
                          else "gold_in_pool_truncated" if any(p["truncated"] for p in gold_in_pool)
                          else "gold_in_pool_untruncated")
        groups["by_gold_truncation"][truncated_gold].append(row)

    def group_summary(subset: Sequence[dict]) -> dict:
        return {"questions": len(subset), **{m: paired_metric(subset, m, k, policy) for m in METRICS}}

    report = {"questions": len(rows), "top_k": k, "candidate_k": candidate_k, "policy": policy,
              "overall": group_summary(rows),
              **{name: {key: group_summary(subset) for key, subset in sorted(group.items())}
                 for name, group in groups.items()}}
    times = [r["rerank"]["latency_ms"] for r in rows if r["rerank"]["candidates"]]
    report["latency_ms"] = {
        "rerank_questions": len(times), "empty_pool_questions": len(rows) - len(times),
        "rerank_median": median(times) if times else None,
        "rerank_p95": float(np.quantile(times, .95)) if times else None,
        "C4_median": median(r["C4"]["latency_ms"] for r in rows) if rows else None,
        "C5_median": median(r["C5"]["latency_ms"] for r in rows) if rows else None,
    }
    total = sum(len(r["rerank"]["candidates"]) for r in rows)
    truncated = sum(r["rerank"]["truncated"] for r in rows)
    report["truncation"] = {"candidates": total, "truncated": truncated,
                            "rate": truncated / total if total else None}
    report["top_k_diagnostics"] = {}
    # Runs recorded before gold filings were stored cannot report these.
    recorded = all("gold_filings" in row for row in rows)
    for config in ("C4", "C5"):
        tables = truncs = wrong_company = wrong_year = wrong_filing = 0
        outside_query_tickers = outside_query_years = eligible_gold_passages = known_year_passages = 0
        for row in rows:
            candidates = {p["chunk_id"]: p for p in row["rerank"]["candidates"]}
            # The query filters already exclude other companies and years, so
            # compare against the companies and years the gold evidence is from.
            tickers = {ticker for ticker, _ in row.get("gold_filings", ())}
            years = {year for _, year in row.get("gold_filings", ()) if year is not None}
            filings = {tuple(filing) for filing in row.get("gold_filings", ())}
            known_years = bool(filings) and all(year is not None for _, year in filings)
            query = row["query"]
            for cid in row[config]["retrieved_chunk_ids"][:k]:
                p = candidates[cid]
                tables += p["content_type"] == "table"
                truncs += p["truncated"]
                wrong_company += bool(tickers) and p["ticker"] not in tickers
                wrong_year += known_years and p["fiscal_year"] not in years
                wrong_filing += bool(filings) and not any(
                    p["ticker"] == ticker and (year is None or p["fiscal_year"] == year)
                    for ticker, year in filings)
                eligible_gold_passages += bool(filings)
                known_year_passages += known_years
                outside_query_tickers += bool(query["tickers"]) and p["ticker"] not in query["tickers"]
                outside_query_years += bool(query["fiscal_years"]) and p["fiscal_year"] not in query["fiscal_years"]
        report["top_k_diagnostics"][config] = {
            "tables": tables, "truncated": truncs,
            "outside_gold_tickers": wrong_company if recorded else None,
            "outside_gold_years": wrong_year if recorded else None,
            "outside_gold_filings": wrong_filing if recorded else None,
            "gold_scope_passages": eligible_gold_passages if recorded else None,
            "gold_year_scope_passages": known_year_passages if recorded else None,
            "outside_query_tickers": outside_query_tickers,
            "outside_query_years": outside_query_years,
        }
    reasons = []
    for source in policy["primary_sources"]:
        group = report["by_source"].get(source)
        if group is None:
            reasons.append(f"missing primary source {source}")
            continue
        n = group["ndcg"]
        if n["delta"] is None or n["delta"] < policy["min_ndcg_gain"] or n["ci95"] is None or n["ci95"][0] <= 0:
            reasons.append(f"{source}: nDCG gain/uncertainty criterion not met")
        for metric in ("recall", "mrr"):
            if group[metric]["delta"] is None or group[metric]["delta"] < -policy["max_recall_mrr_regression"]:
                reasons.append(f"{source}: {metric} regression criterion not met")
    assisted = report["by_source"].get("llm_assisted")
    if assisted is None:
        reasons.append("missing source llm_assisted")
    elif (assisted["ndcg"]["delta"] is None
          or assisted["ndcg"]["delta"] < -policy["max_llm_assisted_ndcg_regression"]):
        reasons.append("llm_assisted: nDCG regression criterion not met")
    for kind, group in (report["by_type"] | report["by_source_type"]).items():
        n = group["ndcg"]
        if n["questions"] >= policy["min_type_questions"] and n["delta"] < -policy["max_type_ndcg_regression"]:
            reasons.append(f"{kind}: nDCG regression criterion not met")
    h = report["overall"]["hard_negative_accuracy"]
    if h["delta"] is not None and h["delta"] < -policy["max_hard_negative_regression"]:
        reasons.append("hard-negative accuracy regression criterion not met")
    for statistic in ("median", "p95"):
        value = report["latency_ms"][f"rerank_{statistic}"]
        if value is None or value > policy[f"max_rerank_{statistic}_ms"]:
            reasons.append(f"reranking {statistic} latency criterion not met")
    report["decision"] = {"advance_to_answer_evaluation": not reasons, "reasons": reasons,
                          "default": "C4", "adoption": "requires paired answer evaluation and team review"}
    # Legacy reference manifests declared @16 in the protocol before this
    # field existed. A future app default must not reinterpret their gates.
    declared_k = policy.get("decision_top_k", 16)
    declared_candidates = policy.get("decision_candidate_k", 50)
    exploratory = []
    if k != declared_k:
        exploratory.append(f"sensitivity rescore; decided at k={declared_k}")
    if candidate_k != declared_candidates:
        exploratory.append(f"exploratory pool of {candidate_k}; decided over {declared_candidates} candidates")
    if exploratory:
        report["decision"] = {"advance_to_answer_evaluation": None, "reasons": exploratory, "default": "C4"}
    return report


def read_pairs(path: Path, *, recover: bool = False) -> list[dict]:
    """Read strictly, or recover complete newline-terminated rows after a kill."""
    rows = []
    try:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                if recover and not line.endswith("\n"):
                    break
                if line.strip():
                    rows.append(json.loads(line))
    except EOFError:
        if not recover:
            raise
    return rows


def _repair_pairs(path: Path, rows: Sequence[dict]) -> None:
    # Atomic replacement keeps the cut stream intact if recovery is interrupted.
    whole = path.with_name(path.name + ".tmp")
    with gzip.open(whole, "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    whole.replace(path)


def _index_manifests(hybrid: Retriever) -> dict:
    result = {}
    for name in ("dense", "bm25"):
        manifest = getattr(getattr(hybrid, name, None), "manifest", None)
        result[name + "_manifest"] = asdict(manifest) if manifest is not None else None
    return result


def _comparable_manifest(manifest: dict) -> dict:
    result = json.loads(json.dumps(manifest))
    # Workload is a segment observation, not a retrieval setting. Preserve the
    # original note in config and record the new one in runtime history.
    result["provenance"].pop("workload_note", None)
    return result


def run_pairs(
    questions: Sequence[BenchmarkQuestion], hybrid: Retriever, reranker: CrossEncoderReranker,
    *, output_dir: Path, provenance: dict, chunks: Sequence[dict], top_k: int = FINAL_K,
    resume: bool = False,
) -> dict:
    if not questions or len({q.question_id for q in questions}) != len(questions):
        raise ValueError("questions must be nonempty with unique IDs")
    if not 0 < top_k <= reranker.config.candidate_k:
        raise ValueError("top_k must be positive and no larger than candidate_k")
    if reranker.config.candidate_k > CANDIDATE_K:
        # Hybrid fuses at max(CANDIDATE_K, k), so a deeper pool would reorder
        # the C4 row away from the ranking the app ships.
        raise ValueError(f"candidate_k must not exceed the shipped C4 depth {CANDIDATE_K}")
    if not resume:
        output_dir.mkdir(parents=True, exist_ok=False)
    elif not output_dir.is_dir():
        raise ValueError("resume requires an existing run directory")
    try:
        with FileLock(output_dir / "run.lock", timeout=0):
            return _run_pairs_locked(questions, hybrid, reranker, output_dir=output_dir,
                                     provenance=provenance, chunks=chunks, top_k=top_k, resume=resume)
    except Timeout as exc:
        raise ValueError("run directory is already locked by another evaluation") from exc


def _run_pairs_locked(
    questions: Sequence[BenchmarkQuestion], hybrid: Retriever, reranker: CrossEncoderReranker,
    *, output_dir: Path, provenance: dict, chunks: Sequence[dict], top_k: int, resume: bool,
) -> dict:
    manifest = {"provenance": provenance, "reranker": reranker.config.to_dict(),
                "top_k": top_k, "policy": POLICY, "query_policy": "C4 evaluation.run._query"}
    manifest = json.loads(json.dumps(manifest))
    pairs_path = output_dir / "pairs.jsonl.gz"
    rows = []
    original_runtime = {}
    repair_needed = False
    indexes = _index_manifests(hybrid)
    if resume:
        original = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
        if _comparable_manifest(original) != _comparable_manifest(manifest):
            raise ValueError("resume requires identical source, inputs, corpus, settings and policy")
        if pairs_path.exists():
            try:
                rows = read_pairs(pairs_path)
            except EOFError:
                rows = read_pairs(pairs_path, recover=True)
                repair_needed = True
        if [r["question"] for r in rows] != [q.to_dict() for q in questions[:len(rows)]]:
            raise ValueError("stored questions are not an exact prefix of the requested benchmark")
        runtime_path = output_dir / "runtime.json"
        if runtime_path.exists():
            original_runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
            if any(original_runtime.get(key) != value for key, value in indexes.items()):
                raise ValueError("resume requires identical dense and BM25 index manifests")
        elif rows:
            raise ValueError("cannot verify resumed indexes without the original runtime.json")
        if len(rows) == len(questions):
            if repair_needed:
                _repair_pairs(pairs_path, rows)
            report = summarize(rows, top_k, candidate_k=reranker.config.candidate_k)
            _dump(output_dir / "summary.json", report)
            return report
    else:
        _dump(output_dir / "config.json", manifest)
    indexed = {p["chunk_id"]: p for p in chunks}
    started = perf_counter()
    reranker.load()
    model_load_ms = (perf_counter() - started) * 1000
    # Warm both models on one real question, without recording it as a result.
    query = _query(questions[0], top_k=top_k, metadata_filter=True)
    started = perf_counter()
    pool = hybrid.search(query, k=reranker.config.candidate_k)
    reranker.rerank(query, pool)
    dense_info = getattr(getattr(hybrid, "dense", None), "runtime_info", {})
    recorded_revision = original_runtime.get("dense_revision")
    if resume and recorded_revision is not None and dense_info.get("revision") != recorded_revision:
        raise ValueError("resume requires the original dense encoder revision")
    if repair_needed:
        _repair_pairs(pairs_path, rows)
    runtime = {
        "cross_encoder_load_ms": model_load_ms, "warmup_ms": (perf_counter() - started) * 1000,
        "resumed_after_questions": len(rows),
        **indexes,
        "cross_encoder_device": str(getattr(reranker.load(), "device", reranker.config.device)),
        "dense_device": dense_info.get("device"), "dense_revision": dense_info.get("revision"),
        "workload_note": provenance.get("workload_note"),
    }
    segment_number = 0
    while (output_dir / f"runtime-{segment_number:04d}.json").exists():
        segment_number += 1
    _dump(output_dir / f"runtime-{segment_number:04d}.json", runtime)
    if not (output_dir / "runtime.json").exists():
        _dump(output_dir / "runtime.json", runtime)
    with gzip.open(pairs_path, "at" if resume else "wt", encoding="utf-8") as stream:
        for i, question in enumerate(questions[len(rows):], start=len(rows) + 1):
            query = _query(question, top_k=top_k, metadata_filter=True)
            started = perf_counter()
            pool = hybrid.search(query, k=reranker.config.candidate_k)
            retrieve_ms = (perf_counter() - started) * 1000
            ranked = reranker.rerank(query, pool)
            metadata = {p.chunk_id: {"ticker": p.ticker, "fiscal_year": p.fiscal_year,
                                    "content_type": p.content_type, "sources": list(p.sources)} for p in pool}
            for p in ranked.diagnostics["candidates"]:
                p.update(metadata[p["chunk_id"]])
            gold = [indexed[cid] for cid in question.supporting_chunk_ids]
            row = {"question": question.to_dict(), "query": asdict(query),
                   "filings": sorted({p["accession_no"] for p in gold}),
                   "gold_filings": _gold_filings(gold),
                   "gold_content": "+".join(sorted({p["content_type"] for p in gold})) or "none",
                   "C4": _rank_record(question, pool[:top_k], retrieve_ms, "C4"),
                   "C5": _rank_record(question, ranked.passages,
                                       retrieve_ms + ranked.diagnostics["latency_ms"], "C5"),
                   "rerank": ranked.diagnostics}
            for config in ("C4", "C5"):
                row[config]["metrics"] = _scores(row, config, top_k)
            rows.append(row)
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            if i % 25 == 0 or i == len(questions):
                print(f"paired C4/C5: {i}/{len(questions)}", flush=True)
    report = summarize(rows, top_k, candidate_k=reranker.config.candidate_k)
    _dump(output_dir / "summary.json", report)
    return report


def _gold_filings(gold: Sequence[dict]) -> list[tuple[str, int | None]]:
    return sorted({(p["ticker"], p["fiscal_year"]) for p in gold}, key=str)


def enrich_gold_filings(rows: Sequence[dict], chunks: Sequence[dict], expected_digest: str | None) -> dict:
    """Derive missing diagnostic metadata without changing recorded results."""
    actual = corpus_records_sha256(chunks)
    if expected_digest is None or actual != expected_digest:
        raise ValueError("gold diagnostics require the exact frozen corpus records digest")
    indexed = {p["chunk_id"]: p for p in chunks}
    updated = 0
    for row in rows:
        if "gold_filings" in row:
            continue
        gold = [indexed[cid] for cid in row["question"]["supporting_chunk_ids"]]
        row["gold_filings"] = _gold_filings(gold)
        updated += 1
    return {"corpus_records_sha256": actual, "derived_rows": updated,
            "method": "supporting chunk IDs joined to the verified frozen corpus; original pairs unchanged"}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    run = actions.add_parser("run")
    run.add_argument("questions", nargs="+", type=Path)
    run.add_argument("--run-id", required=True)
    run.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    run.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    run.add_argument("--top-k", type=int, default=FINAL_K)
    run.add_argument("--candidate-k", type=int, default=RerankConfig().candidate_k)
    run.add_argument("--batch-size", type=int, default=RerankConfig().batch_size)
    run.add_argument("--device")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--workload-note", required=True, help="record other workloads and timing conditions")
    score = actions.add_parser("rescore")
    score.add_argument("run_dir", type=Path)
    score.add_argument("--top-k", type=int)
    score.add_argument("--processed-dir", type=Path,
                       help="derive missing gold-filing diagnostics from the exact frozen corpus")
    args = parser.parse_args(argv)
    if args.action == "rescore":
        manifest = json.loads((args.run_dir / "config.json").read_text(encoding="utf-8"))
        k = manifest["top_k"] if args.top_k is None else args.top_k
        if not 0 < k <= manifest["top_k"]:
            parser.error("rescoring k must be positive and no greater than stored final k")
        rows = read_pairs(args.run_dir / "pairs.jsonl.gz")
        if len(rows) != manifest["provenance"]["questions"]:
            parser.error("run is incomplete; resume it before final rescoring")
        digest = hashlib.sha256(json.dumps([r["question"] for r in rows], sort_keys=True).encode()).hexdigest()
        expected = manifest["provenance"].get("question_records_sha256")
        if expected is not None and digest != expected:
            parser.error("stored questions do not match the frozen benchmark digest")
        enrichment = None
        if args.processed_dir is not None:
            enrichment = enrich_gold_filings(
                rows, list(iter_chunks(processed_dir=args.processed_dir)),
                manifest["provenance"].get("corpus", {}).get("records_sha256"),
            )
        report = summarize(rows, k, manifest["policy"], candidate_k=manifest["reranker"]["candidate_k"])
        if enrichment is not None:
            report["gold_diagnostics_provenance"] = enrichment
        _dump(args.run_dir / f"rescored-k{k}.json", report)
        return
    config = RerankConfig(candidate_k=args.candidate_k, batch_size=args.batch_size, device=args.device)
    if not args.workload_note.strip():
        parser.error("workload-note must describe the timing conditions")
    if not 0 < args.top_k <= config.candidate_k:
        parser.error("top-k must be positive and no greater than candidate-k")
    if config.candidate_k > CANDIDATE_K:
        parser.error(f"candidate-k must not exceed the shipped C4 depth {CANDIDATE_K}")
    output = args.results_root / _safe_run_id(args.run_id)
    if output.exists() and not args.resume:
        parser.error("run directory exists; use a new ID or --resume")
    questions = [q for path in args.questions for q in load_questions(path, processed_dir=args.processed_dir)]
    if len({q.question_id for q in questions}) != len(questions):
        parser.error("duplicate question IDs across benchmark files")
    provenance = record_provenance(args.questions, questions, args.processed_dir)
    provenance["workload_note"] = args.workload_note
    report = run_pairs(questions, build_retriever("hybrid", processed_dir=args.processed_dir),
                       CrossEncoderReranker(config), output_dir=output, provenance=provenance,
                       chunks=list(iter_chunks(processed_dir=args.processed_dir)),
                       top_k=args.top_k, resume=args.resume)
    print(json.dumps(report["decision"], indent=2))
    print(output)


if __name__ == "__main__":
    main()
