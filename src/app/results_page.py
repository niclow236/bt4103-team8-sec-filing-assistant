"""Read-only Results page for saved benchmark runs (#41)."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd
import streamlit as st

from src.stack import RESULTS_ROOT

METRICS = ("recall", "ndcg", "mrr", "hard_negative_accuracy")


@st.cache_data(show_spinner=False)
def load_result_runs(results_root: Path = RESULTS_ROOT) -> list[dict[str, Any]]:
    """Load completed run summaries and question rows without recomputation."""
    if not results_root.is_dir():
        return []
    runs: list[dict[str, Any]] = []
    for run_dir in sorted(results_root.iterdir(), key=lambda path: path.name, reverse=True):
        manifest_path = run_dir / "summary.json"
        if not run_dir.is_dir() or not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        configurations = []
        for config_summary in manifest.get("configurations", ()):
            config = config_summary.get("config") or {}
            config_id = str(config.get("id") or "")
            if not config_id:
                continue
            questions = _read_questions(run_dir / config_id / "questions.jsonl")
            configurations.append({
                "run_id": str(manifest.get("run_id") or run_dir.name),
                "config_id": config_id,
                "name": str(config.get("name") or config_id),
                "questions": int(config_summary.get("questions") or len(questions)),
                "metrics": {metric: config_summary.get(metric) for metric in METRICS},
                "question_rows": questions,
            })
        if configurations:
            runs.append({"run_id": str(manifest.get("run_id") or run_dir.name),
                         "configurations": configurations})
    return runs


def _read_questions(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _benchmark_kind(row: dict[str, Any]) -> str:
    """Keep generated XBRL questions separate from hand-written questions."""
    if str(row.get("difficulty", "")).lower() == "mechanical":
        return "Mechanical (XBRL)"
    if str(row.get("source", "")).lower() == "xbrl":
        return "Mechanical (XBRL)"
    return "Hand-written"


def _table(configurations: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in configurations:
        rows.append({"Configuration": item["config_id"], "Name": item["name"],
                     "Questions": item["questions"],
                     **{metric.replace("_", " ").title(): item["metrics"].get(metric)
                        for metric in METRICS}})
    return pd.DataFrame(rows)


def _render_benchmark(title: str, configurations: list[dict[str, Any]]) -> None:
    st.subheader(title)
    if not configurations:
        st.info("No saved results for this benchmark.")
        return
    st.dataframe(_table(configurations), use_container_width=True, hide_index=True)
    chart_rows = []
    for item in configurations:
        for metric, value in item["metrics"].items():
            if value is not None:
                chart_rows.append({"Configuration": item["config_id"],
                                   "Metric": metric.replace("_", " ").title(),
                                   "Value": float(value)})
    if chart_rows:
        chart = pd.DataFrame(chart_rows).pivot(index="Configuration", columns="Metric",
                                                values="Value").fillna(0)
        st.bar_chart(chart)


def render_results_page() -> None:
    """Render saved ablation tables and charts, split by benchmark source."""
    st.header("Results")
    st.caption("Saved benchmark results are read from results/<run-id>/; no evaluation is rerun.")
    runs = load_result_runs()
    if not runs:
        st.info("No completed benchmark runs were found under results/.")
        return
    run_id = st.selectbox("Run", [run["run_id"] for run in runs])
    selected = next(run for run in runs if run["run_id"] == run_id)
    manual: list[dict[str, Any]] = []
    mechanical: list[dict[str, Any]] = []
    for item in selected["configurations"]:
        grouped: dict[str, list[dict[str, Any]]] = {"Hand-written": [], "Mechanical (XBRL)": []}
        for row in item["question_rows"]:
            grouped[_benchmark_kind(row)].append(row)
        for kind, rows in grouped.items():
            if rows:
                copy = {**item, "questions": len(rows)}
                # Recalculate the displayed metrics for this benchmark only;
                # the saved comparison summary may contain both sources.
                copy["metrics"] = {
                    metric: _mean_metric(rows, metric) for metric in METRICS
                }
                if kind == "Hand-written":
                    manual.append(copy)
                else:
                    mechanical.append(copy)
    tab_manual, tab_mechanical = st.tabs(["Hand-written benchmark", "Mechanical XBRL benchmark"])
    with tab_manual:
        _render_benchmark("Hand-written benchmark", manual)
    with tab_mechanical:
        _render_benchmark("Mechanical XBRL benchmark", mechanical)


def _mean_metric(rows: list[dict[str, Any]], metric: str) -> float | None:
    values = [row.get(metric) for row in rows if row.get(metric) is not None]
    return mean(values) if values else None
