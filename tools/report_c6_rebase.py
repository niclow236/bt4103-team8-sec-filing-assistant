"""Publish the paired #173 acceptance study from frozen native artifacts.

Run from the repository root:
    python tools/report_c6_rebase.py results/c6-rebase-full-20261019

This makes no model calls and never overwrites the native run. The CLI rescore
must have completed first; all published aggregates are checked against it.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
import sys
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.artifacts import dump_json
from src.evaluation.full_ablation import (
    MATRIX, METRICS, answer_summary, check_transactions, read_rows,
    summarize, validate_snapshot,
)
from src.evaluation.provenance import sha256
from src.evaluation.records import BenchmarkQuestion

ROWS = ("C4", "C6")
QUALITY = ("correctness", "faithfulness", "answerability_correct", "citation_precision", "citation_recall")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def portable(value):
    if isinstance(value, dict):
        return {key: portable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [portable(item) for item in value]
    if isinstance(value, str) and value.startswith(str(ROOT) + "/"):
        return value[len(str(ROOT)) + 1:]
    return value


def quality_effect(rows: dict[str, list[dict]], ids: set[str]) -> dict:
    selected = {cid: [row for row in rows[cid] if row["question_id"] in ids] for cid in ROWS}
    result = {"questions": len(ids), **{
        cid: answer_summary(selected[cid], cid)["quality"] for cid in ROWS
    }}
    deltas = {}
    for metric in QUALITY:
        paired = {cid: {row["question_id"]: row["quality"].get(metric) for row in selected[cid]} for cid in ROWS}
        eligible = [qid for qid in sorted(ids) if all(paired[cid][qid] is not None for cid in ROWS)]
        deltas[metric] = {
            "paired_scored": len(eligible),
            "mean_delta": mean(paired["C6"][qid] - paired["C4"][qid] for qid in eligible) if eligible else None,
        }
    result["C6_minus_C4_paired"] = deltas
    return result


def reference_consistency(transactions: list[dict], accepted: dict) -> dict | None:
    """Optional post-run check; never import a historical ranking or timing."""
    path = ROOT / "results/c5-full-20261017/pairs.jsonl.gz"
    if not path.exists():
        return None
    digest = sha256(path)
    if digest != accepted["artifacts_sha256"]["pairs.jsonl.gz"]:
        raise ValueError("historical reference pairs differ from their trusted manifest")
    counts = Counter()
    max_delta = 0.0
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line, transaction in zip(stream, transactions, strict=True):
            original = json.loads(line)
            if BenchmarkQuestion.from_mapping(original["question"]) != BenchmarkQuestion.from_mapping(transaction["question"]):
                raise ValueError("historical reference question sequence differs")
            old, fresh = original["C4"], transaction["rows"]["C4"]
            counts["questions"] += 1
            if old["retrieved_chunk_ids"] == fresh["retrieved_chunk_ids"]:
                counts["identical_ranked_ids"] += 1
                deltas = [abs(a-b) for a, b in zip(old["retrieved_scores"], fresh["retrieved_scores"], strict=True)]
                max_delta = max(max_delta, max(deltas, default=0.0))
            if old["retrieved_scores"] == fresh["retrieved_scores"]:
                counts["identical_scores"] += 1
    return {
        **counts, "pairs_sha256": digest,
        "maximum_absolute_score_delta_when_ids_match": max_delta,
        "note": "Post-run diagnostic only; no reference rankings or timings imported into the fresh paired measurements.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir
    config = read_json(run_dir / "config.json")
    provenance = config["provenance"]
    reference_manifest = read_json(ROOT / "docs/evaluation/c5-full-20261017-manifest.json")
    accepted = reference_manifest["experiment"]["provenance"]
    if tuple(config["matrix"]) != ROWS or config["reranker"] is not None or config["reference"] is not None:
        raise ValueError("expected a fresh C4/C6-only study without a reranker or imported reference")
    if config["matrix"] != {cid: MATRIX[cid].to_dict() for cid in ROWS}:
        raise ValueError("matrix settings differ from the measured source")
    for key in ("question_records_sha256", "questions", "by_benchmark", "retrieval_settings"):
        if provenance[key] != accepted[key]:
            raise ValueError(f"complete #155 benchmark/settings identity differs: {key}")
    if provenance["corpus"]["records_sha256"] != accepted["corpus"]["records_sha256"]:
        raise ValueError("corpus differs from the complete #155 study")
    if [p["sha256"] for p in provenance["inputs"]] != [p["sha256"] for p in accepted["inputs"]]:
        raise ValueError("frozen input file bytes differ")
    source = provenance["source_commit"]
    for path, digest in provenance["source_files_sha256"].items():
        committed = subprocess.check_output(["git", "show", f"{source}:{path}"], cwd=ROOT)
        if hashlib.sha256(committed).hexdigest() != digest or sha256(ROOT / path) != digest:
            raise ValueError(f"source does not match the committed measurement: {path}")
    questions = [BenchmarkQuestion.from_mapping(q) for q in read_rows(run_dir / "benchmark.jsonl")]
    validate_snapshot(config, questions)
    transactions = read_rows(run_dir / "pairs.jsonl")
    check_transactions(transactions, questions, ROWS)
    if len(transactions) != 12729:
        raise ValueError("complete retrieval is required")
    team = {q.question_id for q in questions if q.benchmark == "team"}
    if len(team) != 150:
        raise ValueError("complete team benchmark is required")
    split = {t["question"]["question_id"] for t in transactions if t["rows"]["C6"]["sub_results"]}
    identical = Counter()
    retrieval = {cid: [] for cid in ROWS}
    for question, transaction in zip(questions, transactions):
        c4, c6 = (transaction["rows"][cid] for cid in ROWS)
        if c6.get("rerank_diagnostics") is not None or any(
            sub["rerank_diagnostics"] is not None for sub in c6["sub_results"]
        ):
            raise ValueError("unexpected C6 reranking diagnostics")
        if question.question_id not in split:
            for key in ("retrieved_chunk_ids", "retrieved_scores", "latency_ms", *METRICS):
                if c4[key] != c6[key]:
                    raise ValueError(f"unsplit C6 differs from C4: {question.question_id}: {key}")
            identical[question.benchmark] += 1
        for cid in ROWS:
            if transaction["rows"][cid]["config"] != config["matrix"][cid] | {"candidate_k": 50}:
                raise ValueError("per-question configuration differs")
            retrieval[cid].append(transaction["rows"][cid])
    if not split <= team:
        raise ValueError("unexpected decomposition of a mechanical question")
    stored = read_json(run_dir / "summary.json")
    rescored = read_json(run_dir / "rescored/summary.json")
    if stored["configurations"] != rescored["configurations"]:
        raise ValueError("retrieval rescore differs")
    answers = {cid: read_rows(run_dir / cid / "answers.jsonl") for cid in ROWS}
    expected_ids = [q.question_id for q in questions if q.question_id in team]
    for cid, rows in answers.items():
        if [r["question_id"] for r in rows] != expected_ids:
            raise ValueError(f"incomplete or reordered {cid} answers")
        if any(r["evidence_rerank_enabled"] or r["matrix_config"] != config["matrix"][cid] | {"candidate_k": 50} for r in rows):
            raise ValueError("answers do not record the measured C4/C6 settings")
        if read_json(run_dir / cid / "answer-summary.json") != answer_summary(rows, cid):
            raise ValueError("answer summary differs from native outputs")
    answer_report = read_json(run_dir / "answer-summary.json")
    if answer_report != read_json(run_dir / "rescored/answer-summary.json") or not answer_report["complete"]:
        raise ValueError("answer rescore differs or the paired study is incomplete")
    runtime = read_json(run_dir / "runtime.json")
    if runtime["identity"]["cross_encoder_device"] is not None or runtime["cross_encoder_warmed"]:
        raise ValueError("the acceptance study must not load the cross-encoder")
    native = {
        path.relative_to(run_dir).as_posix(): sha256(path)
        for path in sorted(run_dir.rglob("*"))
        if path.is_file() and "rescored" not in path.relative_to(run_dir).parts and path.suffix != ".lock"
    }
    verification = {cid: Counter(
        check["status"] for row in answers[cid] if row["answer"]["verification"]
        for check in row["answer"]["verification"]["checks"]
    ) for cid in ROWS}
    effects = {}
    for label, ids in {"team": team, "xbrl": {q.question_id for q in questions if q.benchmark == "xbrl"}, "decomposed_team": split}.items():
        groups = {cid: summarize([r for r in retrieval[cid] if r["question_id"] in ids]) for cid in ROWS}
        effects[label] = {**groups, "C6_minus_C4": {
            metric: groups["C6"][metric] - groups["C4"][metric] if groups["C4"][metric] is not None and groups["C6"][metric] is not None else None
            for metric in METRICS
        }}
    report = {
        "run_id": run_dir.name, "schema": config["schema"],
        "configurations_measured": list(ROWS), "retrieval_questions_per_row": len(questions),
        "by_benchmark": provenance["by_benchmark"], "team_answers_per_row": len(team),
        "decomposition_questions": len(split), "decomposition_question_ids": sorted(split),
        "C6_identical_to_C4_without_decomposition": dict(identical),
        "retrieval": stored, "retrieval_effect_relative_to_C4": effects,
        "answers": answer_report,
        "answer_effect_relative_to_C4": {
            "all_team": quality_effect(answers, team), "decomposed_team": quality_effect(answers, split),
            "unsplit_team": quality_effect(answers, team - split),
        },
        "evaluation_methods": {cid: dict(Counter(r["quality"]["method"] for r in answers[cid])) for cid in ROWS},
        "verification_outputs": {cid: sum(r["answer"]["verification"] is not None for r in answers[cid]) for cid in ROWS},
        "verification_claim_status_counts": {cid: dict(counts) for cid, counts in verification.items()},
        "limitations": [
            "This is a fresh paired C4/C6 study, not a replacement measurement of all seven matrix rows.",
            "The archived C5-based C6 and this C4-based C6 use distinct definitions and remain separately identified.",
            "Mechanical labels are facts-derived relevance proxies; mechanical answer quality is not measured.",
            "Hosted generation and self-judging are nondeterministic; verification annotates claims and does not rewrite answers.",
            "Answer differences on identical retrieval evidence cannot be attributed to decomposition or verification alone.",
            "Timing is a recorded M1 laptop workload with overlapping hosted answer preparation, not isolated throughput.",
        ],
    }
    hardware_path = run_dir / "hardware.txt"
    reference_check = reference_consistency(transactions, reference_manifest)
    if reference_check is not None:
        report["C4_consistency_with_accepted_C5_study"] = reference_check
    hardware_lines = hardware_path.read_text().splitlines() if hardware_path.exists() else []
    manifest = {
        "run_id": run_dir.name, "experiment": portable(config),
        "report_script": {"path": "tools/report_c6_rebase.py", "sha256": sha256(Path(__file__))},
        "hardware": {
            "cpu": hardware_lines[0] if hardware_lines else provenance["environment"]["machine"],
            "memory_gib": int(hardware_lines[1]) / 1024**3 if len(hardware_lines) > 1 else None,
            "dense_device": runtime["identity"]["dense"]["device"],
        },
        "runtime_segments": [portable(read_json(p)) for p in sorted(run_dir.glob("runtime-*.json"))],
        "answers_config": portable(read_json(run_dir / "answers-config.json")),
        "answer_runtime_segments": [portable(read_json(p)) for p in sorted(run_dir.glob("answers-runtime-*.json"))],
        "native_artifacts_sha256": native,
        "generation_packages": {name: version(name) for name in ("langchain-core", "langchain-mistralai", "httpx", "httpx-sse", "pydantic")},
        "validation": {
            "complete_frozen_155_cohort_and_corpus": True, "all_source_hashes_match_committed_measurement": True,
            "exact_retrieval_rescore": True, "exact_answer_rescore": True,
            "cross_encoder_loaded": False,
        },
        "shareable_note": "Repository paths normalized; raw native artifact hashes retained. Native per-question outputs and corpus stay local and ignored.",
    }
    archived = {}
    for name in (
        "full-matrix-20261017-retrieval.json", "full-matrix-20261017-answers.json",
        "full-matrix-20261017-manifest.json", "c5-full-20261017-summary.json",
        "c5-full-20261017-manifest.json", "c6-rebase-team-20261019-summary.json",
    ):
        path = Path("docs/evaluation") / name
        committed = subprocess.check_output(["git", "show", f"{source}:{path}"], cwd=ROOT)
        if (ROOT / path).read_bytes() != committed:
            raise ValueError(f"an archived report was changed: {name}")
        archived[name] = sha256(ROOT / path)
    manifest["archived_artifacts_unchanged_sha256"] = archived
    proof = run_dir / "review-validation.json"
    if proof.exists():
        validation = read_json(proof)
        for path, digest in validation["preserved_native_artifacts_sha256"].items():
            if sha256(run_dir / path) != digest:
                raise ValueError(f"completed-resume artifact no longer matches: {path}")
        manifest["validation"].update(validation)
    manifest["reproduce_at_source"] = {
        "commit": source,
        "retrieval": "HF_HUB_OFFLINE=1 python -m src.evaluation.full_ablation retrieval benchmark/questions.jsonl benchmark/generated.jsonl --configurations C4 C6 --run-id NEW_RUN_ID --device mps --workload-note 'describe workload'",
        "answers": "python -m src.evaluation.full_ablation answers results/NEW_RUN_ID --configurations C4 C6 --provider mistral --model ministral-8b-2512",
        "rescore": "python -m src.evaluation.full_ablation rescore results/NEW_RUN_ID",
        "report": "python tools/report_c6_rebase.py results/NEW_RUN_ID",
        "report_script_note": "Use the report script from this PR's reporting commit; the measurement source commit predates that script. Keep src/ at its frozen hashes.",
    }
    output = ROOT / "docs/evaluation"
    dump_json(output / f"{run_dir.name}-summary.json", portable(report))
    dump_json(output / f"{run_dir.name}-manifest.json", manifest)
    print(json.dumps({"run":run_dir.name, "questions":len(questions), "team_answers":{cid:len(rows) for cid,rows in answers.items()}, "split":len(split)}))


if __name__ == "__main__":
    main()
