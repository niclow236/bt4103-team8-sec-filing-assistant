"""Saved runs under results/ as the tables and charts the app and report show (#48).

The Results page and this module's command read a run directory through the
same functions, so a number on the page and the number in a generated report
come from one calculation. Nothing here reruns an evaluation.
"""

from __future__ import annotations

import json
from math import isnan
from pathlib import Path
from statistics import mean, median
from typing import Any

from src.evaluation.metrics import RETRIEVAL_METRICS

RESULT_METRICS = RETRIEVAL_METRICS
METRIC_LABELS = {"recall": "Recall", "ndcg": "nDCG", "mrr": "MRR",
                 "hard_negative_accuracy": "Hard Negative Accuracy"}
BENCHMARKS = (("Team benchmark", "handwritten"),
              ("Mechanical XBRL benchmark", "mechanical"))


def benchmark_kind(row: dict[str, Any]) -> str:
    """Classify a saved row once, consistently for every Results view."""
    source = str(row.get("source", "")).lower()
    difficulty = str(row.get("difficulty", "")).lower()
    if difficulty == "mechanical" or source == "xbrl":
        return "mechanical"
    if str(row.get("question_id", "")).lower().startswith("xbrl-"):
        return "mechanical"
    return "handwritten"


def read_result_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        if path.name == "report.json":
            loaded = json.loads(path.read_text(encoding="utf-8"))
            rows = loaded.get("results", []) if isinstance(loaded, dict) else []
            return [row for row in rows if isinstance(row, dict)]
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict):
                rows.append(row)
        return rows
    except (OSError, ValueError):
        return []


def saved_benchmark_metrics(rows: list[dict[str, Any]], provider: str | None = None) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(benchmark_kind(row), []).append(row)
    output: dict[str, dict[str, Any]] = {}
    for kind, items in grouped.items():
        values: dict[str, Any] = {"questions": len(items)}
        for metric in RESULT_METRICS:
            measured = [row[metric] for row in items if row.get(metric) is not None]
            values[metric] = mean(measured) if measured else None
        answers = [row["answer"] for row in items if isinstance(row.get("answer"), dict)]
        if answers:
            latencies = [answer["latency_ms"] for answer in answers
                         if answer.get("latency_ms") is not None]
            values.update({
                "abstention_rate": mean(answer.get("abstained", False) for answer in answers),
                "median_latency_ms": median(latencies) if latencies else None,
                "llm_questions": sum(provider is not None and row.get("route") == provider for row in items),
            })
        output[kind] = values
    return output


def incomplete(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str | None:
    """Describe partial results without treating a summary-only run as truncated."""
    stopped = summary.get("stopped")
    if stopped:
        where = stopped.get("question_id") if isinstance(stopped, dict) else None
        return f"stopped at {where}" if where else "stopped part-way"
    expected = summary.get("questions")
    if rows and isinstance(expected, int) and len(rows) < expected:
        return f"{len(rows)} of {expected} question rows could be read"
    return None


def read_runs(results_root: Path) -> list[dict[str, Any]]:
    """Read saved evaluation runs, including older files, without rerunning them."""
    if not results_root.is_dir():
        return []
    runs: list[dict[str, Any]] = []
    for run_dir in sorted((path for path in results_root.iterdir() if path.is_dir()),
                          key=lambda path: path.stat().st_mtime, reverse=True):
        try:
            manifest = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(manifest, dict):
            continue
        configurations = []
        for summary in manifest.get("configurations", ()):
            if not isinstance(summary, dict):
                continue
            config = summary.get("config") or {}
            if not isinstance(config, dict):
                continue
            config_id = str(config.get("id") or "")
            # Saved IDs name one child directory, never a path or a Windows drive.
            if not config_id or config_id in {".", ".."} or any(
                    character in config_id for character in ("/", "\\", ":")):
                continue
            config_dir = run_dir / config_id
            if config_dir.resolve().parent != run_dir.resolve():
                continue
            rows = read_result_rows(config_dir / "questions.jsonl")
            if not rows:
                rows = read_result_rows(config_dir / "report.json")
            configurations.append({
                "run_id": run_dir.name,
                "config_id": config_id,
                "name": str(config.get("name") or config_id),
                "questions": int(summary.get("questions") or len(rows)),
                "incomplete": incomplete(summary, rows),
                "benchmark_metrics": (saved_benchmark_metrics(rows, config.get("provider"))
                                      if rows else summary.get("by_benchmark") or {}),
            })
        if configurations:
            runs.append({"run_id": run_dir.name,
                         "configurations": configurations})
    return runs


def _without_empty_columns(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    filled = {key for row in rows for key, value in row.items()
              if value is not None and not (isinstance(value, float) and isnan(value))}
    return [{key: value for key, value in row.items() if key in filled} for row in rows]


def retrieval_table(configurations: list[dict[str, Any]], benchmark: str) -> list[dict[str, Any]]:
    rows = []
    for item in configurations:
        saved = item["benchmark_metrics"].get(benchmark, {})
        rows.append({"Configuration": item["config_id"], "Name": item["name"],
                     "Questions": saved.get("questions", item["questions"]),
                     **{label: saved.get(metric)
                        for metric, label in METRIC_LABELS.items()}})
    return _without_empty_columns(rows)


def model_table(configurations: list[dict[str, Any]], benchmark: str) -> list[dict[str, Any]]:
    rows = []
    for item in configurations:
        saved = item["benchmark_metrics"].get(benchmark, {})
        if saved.get("abstention_rate") is None:
            continue
        rows.append({"Configuration": item["config_id"], "Name": item["name"],
                     "Questions": saved.get("questions"),
                     "Abstention rate": saved.get("abstention_rate"),
                     "Median latency (ms)": saved.get("median_latency_ms"),
                     "LLM questions": saved.get("llm_questions")})
    return _without_empty_columns(rows)


def chart_rows(configurations: list[dict[str, Any]], benchmark: str) -> list[dict[str, Any]]:
    """Return only the four retrieval metrics; counts are not scores."""
    rows = []
    for item in configurations:
        saved = item["benchmark_metrics"].get(benchmark, {})
        for metric, label in METRIC_LABELS.items():
            value = saved.get(metric)
            if value is not None:
                rows.append({"Configuration": item["config_id"],
                             "Metric": label,
                             "Value": float(value)})
    return rows


def retrieval_configurations(configurations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The configurations with at least one retrieval score on either benchmark."""
    return [item for item in configurations
            if any(any(item["benchmark_metrics"].get(kind, {}).get(metric) is not None
                       for metric in METRIC_LABELS)
                   for _, kind in BENCHMARKS)]
