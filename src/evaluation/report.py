"""Saved runs under results/ as the tables and charts the app and report show (#48).

The Results page and this module's command read a run directory through the
same functions, so a number on the page and the number in a generated report
come from one calculation. Nothing here reruns an evaluation.
"""

from __future__ import annotations

import argparse
import json
from html import escape
from math import isnan
from pathlib import Path
from statistics import mean, median
from typing import Any

from src.config import PROJECT_ROOT
from src.evaluation.metrics import RETRIEVAL_METRICS

RESULT_METRICS = RETRIEVAL_METRICS
METRIC_LABELS = {"recall": "Recall", "ndcg": "nDCG", "mrr": "MRR",
                 "hard_negative_accuracy": "Hard Negative Accuracy"}
# Answer quality as full_ablation's answer-summary.json records it, team questions only.
QUALITY_LABELS = {"correctness": "Correctness", "faithfulness": "Faithfulness",
                  "answerability_correct": "Answerability",
                  "citation_precision": "Citation precision",
                  "citation_recall": "Citation recall"}
# Generated reports sit beside the hand-curated summaries the docs link to.
REPORT_ROOT = PROJECT_ROOT / "docs" / "evaluation"
TEXT_COLUMNS = ("Configuration", "Name", "Claim checks")
BENCHMARKS = (("Team benchmark", "handwritten"),
              ("Mechanical XBRL benchmark", "mechanical"))
# Every name a report can write; a regenerated report removes those it no longer has.
GENERATED = ("ablation.md", *(f"{kind}-retrieval.svg" for _, kind in BENCHMARKS))


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


# full_ablation names its saved split by benchmark source; rows are classified by kind.
SAVED_SPLIT_NAMES = {"team": "handwritten", "xbrl": "mechanical"}


def _saved_split(by_benchmark: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(by_benchmark, dict):
        return {}
    return {SAVED_SPLIT_NAMES.get(name, name): metrics for name, metrics in by_benchmark.items()}


def read_run(run_dir: Path) -> dict[str, Any] | None:
    """One saved run, or None when its folder holds no readable configurations."""
    try:
        manifest = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(manifest, dict):
        return None
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
                                  if rows else _saved_split(summary.get("by_benchmark"))),
        })
    if not configurations:
        return None
    return {"run_id": run_dir.name, "configurations": configurations,
            "answer_quality": _answer_quality(run_dir / "answer-summary.json")}


def _answer_quality(path: Path) -> dict[str, dict[str, Any]]:
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    configurations = saved.get("configurations") if isinstance(saved, dict) else None
    if not isinstance(configurations, dict):
        return {}
    return {str(config_id): summary for config_id, summary in configurations.items()
            if isinstance(summary, dict)}


def read_runs(results_root: Path) -> list[dict[str, Any]]:
    """Read saved evaluation runs, including older files, without rerunning them."""
    if not results_root.is_dir():
        return []
    runs = (read_run(run_dir) for run_dir in sorted(
        (path for path in results_root.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime, reverse=True))
    return [run for run in runs if run is not None]


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


def answer_quality_table(run: dict[str, Any], benchmark: str) -> list[dict[str, Any]]:
    """Judged answer quality per configuration, each mean beside how many outputs it scored.

    The matrix judges answers to team questions only, so the mechanical
    benchmark has none.
    """
    if benchmark != "handwritten":
        return []
    rows = []
    for item in run["configurations"]:
        saved = run.get("answer_quality", {}).get(item["config_id"])
        if saved is None:
            continue
        row = {"Configuration": item["config_id"], "Name": item["name"],
               "Outputs": saved.get("questions")}
        quality = saved.get("quality") or {}
        for metric, label in QUALITY_LABELS.items():
            measured = quality.get(metric) or {}
            row[label] = measured.get("mean")
            row[f"{label} scored"] = measured.get("scored")
        checks = saved.get("checks") or {}
        row["Claim checks"] = ", ".join(f"{status} {count}"
                                        for status, count in sorted(checks.items())) or None
        rows.append(row)
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


def run_warnings(run: dict[str, Any]) -> list[str]:
    """What a reader must know before trusting a run's tables, page and report alike."""
    warnings = []
    unsplit = [item["config_id"] for item in run["configurations"]
               if not item["benchmark_metrics"]]
    if unsplit:
        warnings.append(
            f"{len(unsplit)} of this run's {len(run['configurations'])} configurations "
            "have a summary but no usable per-question results on this machine "
            f"(results/{run['run_id']}/<configuration>/questions.jsonl or report.json), so their "
            "scores cannot be split by benchmark and are not shown: " + ", ".join(unsplit))
    partial = [f"{item['config_id']} ({item['incomplete']})"
               for item in run["configurations"] if item.get("incomplete")]
    if partial:
        warnings.append("Incomplete results: " + ", ".join(partial)
                        + ". These scores are partial and are not comparable to a complete run.")
    return warnings


def _cell(value: Any) -> str:
    if value is None or (isinstance(value, float) and isnan(value)):
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value).replace("|", "\\|")


def markdown_table(rows: list[dict[str, Any]]) -> str:
    columns = list(rows[0])
    lines = ["| " + " | ".join(columns) + " |",
             "| " + " | ".join("---" if column in TEXT_COLUMNS else "---:"
                               for column in columns) + " |"]
    lines += ["| " + " | ".join(_cell(row.get(column)) for column in columns) + " |"
              for row in rows]
    return "\n".join(lines) + "\n"


# One colour per metric, from a palette that stays distinct under the common
# colour-vision deficiencies, on a white plate so it reads in a dark viewer too.
METRIC_COLOURS = {"Recall": "#4269d0", "nDCG": "#efb118", "MRR": "#ff725c",
                  "Hard Negative Accuracy": "#6cc5b0"}


def svg_chart(rows: list[dict[str, Any]], title: str) -> str:
    """A grouped bar chart of ``chart_rows`` on a 0-1 axis: one group per configuration."""
    configurations = list(dict.fromkeys(row["Configuration"] for row in rows))
    metrics = [label for label in METRIC_LABELS.values()
               if any(row["Metric"] == label for row in rows)]
    left, top, plot_height, group_width = 48, 56, 240, max(24 * len(metrics) + 24, 72)
    width = left + group_width * len(configurations) + 16
    height = top + plot_height + 40
    bar_width = (group_width - 24) / len(metrics)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="sans-serif" font-size="12" '
        f'role="img" aria-label="{escape(title)}">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{left}" y="20" font-size="14" font-weight="bold" fill="#1f1f1f">'
        f'{escape(title)}</text>',
    ]
    legend_x = left
    for label in metrics:
        parts.append(f'<rect x="{legend_x}" y="30" width="10" height="10" '
                     f'fill="{METRIC_COLOURS[label]}"/>')
        parts.append(f'<text x="{legend_x + 14}" y="39" fill="#1f1f1f">{escape(label)}</text>')
        legend_x += 14 + 7 * len(label) + 16
    for tick in (0, 0.25, 0.5, 0.75, 1):
        y = top + plot_height * (1 - tick)
        parts.append(f'<line x1="{left}" x2="{width - 16}" y1="{y:g}" y2="{y:g}" '
                     f'stroke="#d9d9d9"/>')
        parts.append(f'<text x="{left - 6}" y="{y + 4:g}" text-anchor="end" '
                     f'fill="#555555">{tick:g}</text>')
    values = {(row["Configuration"], row["Metric"]): row["Value"] for row in rows}
    for group, configuration in enumerate(configurations):
        x = left + group * group_width + 12
        for index, label in enumerate(metrics):
            value = values.get((configuration, label))
            if value is None:
                continue
            bar_height = plot_height * value
            parts.append(
                f'<rect x="{x + index * bar_width:g}" y="{top + plot_height - bar_height:g}" '
                f'width="{bar_width - 2:g}" height="{bar_height:g}" '
                f'fill="{METRIC_COLOURS[label]}">'
                f'<title>{escape(configuration)} {escape(label)}: {value:.4f}</title></rect>')
        parts.append(f'<text x="{x + (group_width - 24) / 2:g}" y="{top + plot_height + 18}" '
                     f'text-anchor="middle" fill="#1f1f1f">{escape(configuration)}</text>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def report_files(run: dict[str, Any]) -> dict[str, str]:
    """Every file the report is made of, by name, so a rerun rewrites all of them."""
    files: dict[str, str] = {}
    sections = [f"# Results: {run['run_id']}\n",
                *(f"> **Warning:** {warning}\n" for warning in run_warnings(run))]
    retrieval = retrieval_configurations(run["configurations"])
    for title, kind in BENCHMARKS:
        matching = [item for item in retrieval if kind in item["benchmark_metrics"]]
        models = model_table(run["configurations"], kind)
        quality = answer_quality_table(run, kind)
        if not matching and not models and not quality:
            continue
        section = f"## {title}\n"
        if matching:
            section += "\n" + markdown_table(retrieval_table(matching, kind))
        chart = chart_rows(matching, kind)
        if chart:
            name = f"{kind}-retrieval.svg"
            files[name] = svg_chart(chart, f"{title}: retrieval metrics")
            section += f"\n![{title} retrieval metrics]({name})\n"
        if models:
            section += "\n### Answer-model ablation\n\n" + markdown_table(models)
        if quality:
            section += "\n### Answer quality\n\n" + markdown_table(quality)
        sections.append(section)
    files["ablation.md"] = "\n".join(sections)
    return files


def write_report(run: dict[str, Any], out_dir: Path) -> None:
    """Write the report, removing any file an earlier report wrote that this one does not."""
    files = report_files(run)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in set(GENERATED) - set(files):
        (out_dir / stale).unlink(missing_ok=True)
    for name, text in files.items():
        (out_dir / name).write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m src.evaluation.report",
        description="Turn one saved run directory into its ablation tables and metric charts.")
    parser.add_argument("run_dir", type=Path, help="a run under results/, e.g. results/full-matrix-20261017")
    parser.add_argument("--out", type=Path,
                        help="where to write the report (default: docs/evaluation/<run-id>/)")
    args = parser.parse_args(argv)
    run = read_run(args.run_dir)
    if run is None:
        parser.error(f"{args.run_dir} holds no summary.json with a readable configuration")
    write_report(run, args.out or REPORT_ROOT / run["run_id"])


if __name__ == "__main__":
    main()
