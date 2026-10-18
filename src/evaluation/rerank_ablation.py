"""Paired C4/C5 retrieval experiment: run, resume, or rescore stored rankings."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from dataclasses import asdict
from math import isfinite
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from typing import Any, Sequence

import numpy as np
from filelock import FileLock, Timeout

from src.config import PROCESSED_DIR
from src.pipeline.chunk import iter_chunks
from src.retrieval.base import Retriever, _boosted, validate_table_boost
from src.retrieval.constants import CANDIDATE_K, FINAL_K
from src.retrieval.rerank import CrossEncoderReranker, RerankConfig
from src.stack import RESULTS_ROOT, build_retriever

from .benchmark import load_questions
from .metrics import score_question
from .provenance import corpus_records_sha256, question_records_sha256, record_provenance, sha256
from .records import TEAM, XBRL, BenchmarkQuestion, BenchmarkValidationError, RunResult, benchmark_of
from .results import _safe_run_id
from .run import _query

METRICS = ("recall", "ndcg", "mrr", "hard_negative_accuracy")
POLICY = {
    "decision_top_k": 16,
    "decision_candidate_k": 50,
    "primary_benchmarks": [TEAM, XBRL],
    "decision_benchmark_counts": {TEAM: 150, XBRL: 12579},
    "decision_question_records_sha256": "84763df25b8c320ef5f1a60ab8e554db35b4e959eff975bdf81fc898bfc90339",
    "min_ndcg_gain": 0.02,
    "max_recall_mrr_regression": 0.01,
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


def _all_scores(rows: Sequence[dict], k: int) -> dict:
    # Recompute from saved rankings rather than trusting cached metrics. One
    # score per configuration/question/cutoff is shared by every breakdown.
    return {row["question"]["question_id"]: {
        config: _scores(row, config, k) for config in ("C4", "C5")
    } for row in rows}


def paired_metric(
    rows: Sequence[dict], metric: str, k: int, policy: dict = POLICY,
    *, scores: dict | None = None, identities: dict | None = None,
) -> dict:
    if scores is None:
        scores = _all_scores(rows, k)
    selected = []
    for row in rows:
        scored = scores[row["question"]["question_id"]]
        a, b = scored["C4"][metric], scored["C5"][metric]
        if a is not None and b is not None:
            selected.append((row, a, b))
    if not selected:
        return {"questions": 0, "C4": None, "C5": None, "delta": None,
                "wins": 0, "losses": 0, "ties": 0, "ci95": None, "clusters": 0}
    grouped: dict[str, list[float]] = defaultdict(list)
    # Build clusters from all rows so connections are retained even when a
    # question itself has no gold for this metric.
    if identities is None:
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
    scores = _all_scores(rows, k)
    groups: dict[str, dict[str, list]] = {
        "by_benchmark": defaultdict(list), "by_type": defaultdict(list),
        "by_benchmark_type": defaultdict(list), "by_gold_content": defaultdict(list),
        "by_gold_truncation": defaultdict(list),
    }
    for row in rows:
        benchmark = benchmark_of(row["question"].get("source"))
        groups["by_benchmark"][benchmark].append(row)
        groups["by_type"][row["question"]["question_type"]].append(row)
        groups["by_benchmark_type"][benchmark + ":" + row["question"]["question_type"]].append(row)
        groups["by_gold_content"][row["gold_content"]].append(row)
        gold_ids = set(row["question"]["supporting_chunk_ids"])
        gold_in_pool = [p for p in row["rerank"]["candidates"] if p["chunk_id"] in gold_ids]
        truncated_gold = ("unlabelled" if not gold_ids else "no_gold_in_pool" if not gold_in_pool
                          else "gold_in_pool_truncated" if any(p["truncated"] for p in gold_in_pool)
                          else "gold_in_pool_untruncated")
        groups["by_gold_truncation"][truncated_gold].append(row)

    def group_summary(subset: Sequence[dict]) -> dict:
        # Keep each subset's historical connected-filing clusters. Building
        # global clusters would change the published group confidence bounds.
        identities = dict(zip((r["question"]["question_id"] for r in subset), _clusters(subset)))
        return {"questions": len(subset), **{
            m: paired_metric(subset, m, k, policy, scores=scores, identities=identities)
            for m in METRICS}}

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
    for benchmark in policy["primary_benchmarks"]:
        group = report["by_benchmark"].get(benchmark)
        if group is None:
            reasons.append(f"missing benchmark {benchmark}")
            continue
        n = group["ndcg"]
        if n["delta"] is None or n["delta"] < policy["min_ndcg_gain"] or n["ci95"] is None or n["ci95"][0] <= 0:
            reasons.append(f"{benchmark}: nDCG gain/uncertainty criterion not met")
        for metric in ("recall", "mrr"):
            if group[metric]["delta"] is None or group[metric]["delta"] < -policy["max_recall_mrr_regression"]:
                reasons.append(f"{benchmark}: {metric} regression criterion not met")
    for kind, group in (report["by_type"] | report["by_benchmark_type"]).items():
        n = group["ndcg"]
        if n["questions"] >= policy["min_type_questions"] and n["delta"] < -policy["max_type_ndcg_regression"]:
            reasons.append(f"{kind}: nDCG regression criterion not met")
    h = report["overall"]["hard_negative_accuracy"]
    if h["delta"] is None:
        reasons.append("hard-negative accuracy could not be measured")
    elif h["delta"] < -policy["max_hard_negative_regression"]:
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
    if not exploratory:
        counts = {name:group["questions"] for name,group in report["by_benchmark"].items()}
        digest = question_records_sha256([BenchmarkQuestion.from_mapping(row["question"]) for row in rows])
        if (counts != policy.get("decision_benchmark_counts")
                or digest != policy.get("decision_question_records_sha256")):
            exploratory.append("exploratory benchmark cohort; adoption requires the declared complete benchmark counts and question digest")
    if exploratory:
        report["quality_gate_reasons"] = list(reasons)
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


def _logit_ranking(row: dict, k: int, table_boost: float) -> dict:
    validate_table_boost(table_boost)
    scored = []
    for candidate in row["rerank"]["candidates"]:
        score = float(candidate["raw_logit"])
        if not isfinite(score):
            raise ValueError("sensitivity requires finite saved raw logits")
        if candidate["content_type"] == "table":
            score = _boosted(score, table_boost)
        scored.append((score, candidate["chunk_id"]))
    ranked = sorted(scored, key=lambda pair: (-pair[0], pair[1]))[:k]
    return {"retrieved_chunk_ids": [cid for _, cid in ranked],
            "retrieved_scores": [score for score, _ in ranked],
            "latency_ms": row["C5"]["latency_ms"]}


def verify_rankings(rows: Sequence[dict], k: int, *, candidate_k: int = 50) -> None:
    """Check stored rankings against the recorded pool/logits, at stored k.

    This detects inconsistent edits/corruption, not coordinated fabrication of
    both model outputs and rankings. Frozen input/model/artifact provenance is
    retained separately. Checking the full stored depth also protects a rescore
    requested at a smaller cutoff.
    """
    for row in rows:
        candidates = row["rerank"]["candidates"]
        ids = [p["chunk_id"] for p in candidates]
        if len(ids) != len(set(ids)) or len(ids) > candidate_k:
            raise ValueError("saved candidate pool has duplicate IDs or exceeds the ranking depth")
        for p in candidates:
            for field in ("hybrid_score", "raw_logit"):
                value = p[field]
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
                    raise ValueError("saved candidate pool contains non-finite or invalid ranking scores")
        n = min(k, len(candidates))
        for cid in ("C4", "C5"):
            if (len(row[cid]["retrieved_chunk_ids"]) != n
                    or len(row[cid]["retrieved_scores"]) != n):
                raise ValueError("recorded rankings do not retain the expected number of passages and scores")
        if (ids[:n] != row["C4"]["retrieved_chunk_ids"]
                or [p["hybrid_score"] for p in candidates[:n]] != row["C4"]["retrieved_scores"]):
            raise ValueError("saved candidate pool does not reproduce the recorded C4 ranking")
        reconstructed = _logit_ranking(row, n, row["query"]["table_boost"])
        if any(reconstructed[key] != row["C5"][key]
               for key in ("retrieved_chunk_ids", "retrieved_scores")):
            raise ValueError("saved logits/boost do not reproduce the recorded C5 ranking")


def table_boost_sensitivity(rows: Sequence[dict], k: int, manifest: dict) -> dict:
    """Change only the final logit table boost; never infer new timings/adoption.

    This can diagnose a historical source-split run as well as a current one.
    Both are grouped as team/xbrl, with the original policy retained as context.
    """
    verify_rankings(rows, manifest.get("top_k", k), candidate_k=manifest["reranker"]["candidate_k"])
    unboosted = [{**row, "C5": _logit_ranking(row, k, 1.0)} for row in rows]
    original_policy = manifest["policy"]
    policy = POLICY | {key: original_policy[key] for key in ("seed", "bootstrap_samples")}
    reports = {}
    for name, pairs in (("original", rows), ("without_table_boost", unboosted)):
        report = summarize(pairs, k, policy, candidate_k=manifest["reranker"]["candidate_k"])
        report.pop("latency_ms")
        report["decision"] = {"advance_to_answer_evaluation": None,
                              "reasons": ["table-boost sensitivity only; no adoption or latency decision"],
                              "default": "C4"}
        reports[name] = report
    return {"kind": "table_boost_sensitivity", "questions": len(rows), "top_k": k,
            "method": "same candidates/raw logits; remove only final logit table boost; chunk-ID ties",
            "limitation": "does not remove candidate-retrieval boosts or measure new latency/answers",
            "original_policy": original_policy, **reports}


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
    provenance = result["provenance"]
    provenance.pop("workload_note", None)
    provenance.get("environment", {}).pop("platform", None)
    provenance.get("corpus", {}).pop("processed_dir", None)
    if "inputs" in provenance:
        provenance["inputs"] = [item["sha256"] for item in provenance["inputs"]]
    return result


def _validate_depths(top_k: int, candidate_k: int) -> None:
    if not 0 < top_k <= candidate_k:
        raise ValueError("top_k must be positive and no larger than candidate_k")
    if candidate_k > CANDIDATE_K:
        raise ValueError(f"candidate_k must not exceed the shipped C4 depth {CANDIDATE_K}")


def run_pairs(
    questions: Sequence[BenchmarkQuestion], hybrid: Retriever, reranker: CrossEncoderReranker,
    *, output_dir: Path, provenance: dict, chunks: Sequence[dict], top_k: int = FINAL_K,
    resume: bool = False,
) -> dict:
    if not questions or len({q.question_id for q in questions}) != len(questions):
        raise ValueError("questions must be nonempty with unique IDs")
    _validate_depths(top_k, reranker.config.candidate_k)
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
        verify_rankings(rows, top_k, candidate_k=reranker.config.candidate_k)
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
    # Empty pools never call predict. Find a real nonempty pool so first-use
    # inference cannot leak into a measured row, including after a resume.
    warmup_question_id = None
    started = perf_counter()
    for warm in questions:
        query = _query(warm, top_k=top_k, metadata_filter=True)
        pool = hybrid.search(query, k=reranker.config.candidate_k)
        if pool:
            reranker.rerank(query, pool)
            warmup_question_id = warm.question_id
            break
    dense_info = getattr(getattr(hybrid, "dense", None), "runtime_info", {})
    recorded_revision = original_runtime.get("dense_revision")
    if resume and recorded_revision is not None and dense_info.get("revision") != recorded_revision:
        raise ValueError("resume requires the original dense encoder revision")
    if repair_needed:
        _repair_pairs(pairs_path, rows)
    runtime = {
        "cross_encoder_load_ms": model_load_ms, "warmup_ms": (perf_counter() - started) * 1000,
        "resumed_after_questions": len(rows),
        "warmup_question_id": warmup_question_id,
        "cross_encoder_warmed": warmup_question_id is not None,
        **indexes,
        "cross_encoder_device": str(getattr(reranker.load(), "device", reranker.config.device)),
        "dense_device": dense_info.get("device"), "dense_revision": dense_info.get("revision"),
        "workload_note": provenance.get("workload_note"),
        "environment": provenance.get("environment"),
        "inputs": provenance.get("inputs"),
        "processed_dir": provenance.get("corpus", {}).get("processed_dir"),
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
    verify_rankings(rows, top_k, candidate_k=reranker.config.candidate_k)
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
    sensitivity = actions.add_parser("table-boost-sensitivity")
    sensitivity.add_argument("run_dir", type=Path)
    sensitivity.add_argument("--top-k", type=int)
    args = parser.parse_args(argv)
    if args.action in ("rescore", "table-boost-sensitivity"):
        manifest = json.loads((args.run_dir / "config.json").read_text(encoding="utf-8"))
        k = manifest["top_k"] if args.top_k is None else args.top_k
        if not 0 < k <= manifest["top_k"]:
            parser.error("rescoring k must be positive and no greater than stored final k")
        if args.action == "rescore" and "primary_benchmarks" not in manifest["policy"]:
            parser.error("run predates the team/xbrl benchmark policy (#161); "
                         "rescore it on its recorded source commit")
        rows = read_pairs(args.run_dir / "pairs.jsonl.gz")
        if len(rows) != manifest["provenance"]["questions"]:
            parser.error("run is incomplete; resume it before final rescoring")
        expected = manifest["provenance"].get("question_records_sha256")
        try:
            digest = question_records_sha256([BenchmarkQuestion.from_mapping(r["question"]) for r in rows])
        except (BenchmarkValidationError, TypeError, KeyError):
            digest = None
        if expected is not None and digest != expected:
            parser.error("stored questions do not match the frozen benchmark digest")
        if args.action == "table-boost-sensitivity":
            try:
                report = table_boost_sensitivity(rows, k, manifest)
            except (ValueError, KeyError, TypeError) as error:
                parser.error(str(error))
            report["provenance"] = {
                "source_commit": manifest["provenance"].get("source_commit"),
                "question_records_sha256": expected,
                "config_sha256": sha256(args.run_dir / "config.json"),
                "pairs_sha256": sha256(args.run_dir / "pairs.jsonl.gz"),
            }
            _dump(args.run_dir / f"table-boost-sensitivity-k{k}.json", report)
            return
        try:
            verify_rankings(rows, manifest["top_k"], candidate_k=manifest["reranker"]["candidate_k"])
        except (ValueError, KeyError, TypeError) as error:
            parser.error(str(error))
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
    if not args.workload_note.strip():
        parser.error("workload-note must describe the timing conditions")
    try:
        config = RerankConfig(candidate_k=args.candidate_k, batch_size=args.batch_size, device=args.device)
        _validate_depths(args.top_k, config.candidate_k)
    except ValueError as error:
        parser.error(str(error))
    output = args.results_root / _safe_run_id(args.run_id)
    if output.exists() and not args.resume:
        parser.error("run directory exists; use a new ID or --resume")
    if args.resume and not (output / "config.json").is_file():
        parser.error(f"nothing to resume at {output}: check --run-id; if interrupted before "
                     "config.json was written, use a new run ID")
    questions = [q for path in args.questions for q in load_questions(path, processed_dir=args.processed_dir)]
    if len({q.question_id for q in questions}) != len(questions):
        parser.error("duplicate question IDs across benchmark files")
    chunks = list(iter_chunks(processed_dir=args.processed_dir))
    provenance = record_provenance(args.questions, questions, args.processed_dir, chunks=chunks)
    provenance["workload_note"] = args.workload_note
    try:
        report = run_pairs(questions, build_retriever("hybrid", processed_dir=args.processed_dir),
                           CrossEncoderReranker(config), output_dir=output, provenance=provenance,
                           chunks=chunks, top_k=args.top_k, resume=args.resume)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    print(json.dumps(report["decision"], indent=2))
    print(output)


if __name__ == "__main__":
    main()
