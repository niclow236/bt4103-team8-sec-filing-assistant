"""Saved runs under results/ as the tables and charts the app and report show (#48).

The Results page and this module's command read a run directory through the
same functions, so a number on the page and the number in a generated report
come from one calculation. Nothing here reruns an evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
from html import escape
from math import isnan
from pathlib import Path
from statistics import mean, median
from typing import Any

from src.config import PROJECT_ROOT
from src.evaluation.harness import answer_scores
from src.evaluation.metrics import RETRIEVAL_METRICS

RELEVANCE_METRICS = ("recall", "ndcg", "mrr")
METRIC_LABELS = {"recall": "Recall", "ndcg": "nDCG", "mrr": "MRR",
                 "hard_negative_accuracy": "Hard Negative Accuracy"}
# Answer quality as full_ablation's answer-summary.json records it, team questions only.
QUALITY_LABELS = {"correctness": "Correctness", "faithfulness": "Faithfulness",
                  "answerability_correct": "Answerability",
                  "citation_precision": "Citation precision",
                  "citation_recall": "Citation recall"}
# The manifest every evaluation command writes at a run's root; a folder that
# holds one is a saved run.
SUMMARY = "summary.json"
# Generated reports sit beside the hand-curated summaries the docs link to.
REPORT_ROOT = PROJECT_ROOT / "docs" / "evaluation"
TEXT_COLUMNS = ("Configuration", "Name", "Claim checks")
# The two benchmark kinds a saved question row is classified into.
TEAM, MECHANICAL = "handwritten", "mechanical"
BENCHMARKS = (("Team benchmark", TEAM),
              ("Mechanical XBRL benchmark", MECHANICAL))
# Every name a report can write; a regenerated report removes those it no longer has.
GENERATED = ("ablation.md", *(f"{kind}-{chart}.svg" for _, kind in BENCHMARKS
                               for chart in ("retrieval", "answer-quality")))


def benchmark_kind(row: dict[str, Any]) -> str:
    """Classify a saved row once, consistently for every Results view."""
    source = str(row.get("source", "")).lower()
    difficulty = str(row.get("difficulty", "")).lower()
    if difficulty == "mechanical" or source == "xbrl":
        return MECHANICAL
    if str(row.get("question_id", "")).lower().startswith("xbrl-"):
        return MECHANICAL
    return TEAM


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
        values: dict[str, Any] = {"questions": len(items), "metric_denominators": {}}
        # Read leniently, unlike full_ablation.summarize: a row whose score is not
        # one is left out of the mean and its count rather than failing. Recall,
        # nDCG and MRR score the same questions (metrics.score_question sets the
        # three together), so a row with any of them set to a non-score leaves
        # all three.
        relevance_rows = [row for row in items if all(
            row.get(metric) is None or _score(row.get(metric)) is not None
            for metric in RELEVANCE_METRICS)]
        for metric in RETRIEVAL_METRICS:
            source = relevance_rows if metric in RELEVANCE_METRICS else items
            measured = [score for row in source if (score := _score(row.get(metric))) is not None]
            values[metric] = mean(measured) if measured else None
            values["metric_denominators"][metric] = len(measured)
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
SAVED_SPLIT_NAMES = {"team": TEAM, "xbrl": MECHANICAL}


def _saved_split(by_benchmark: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(by_benchmark, dict):
        return {}
    return {SAVED_SPLIT_NAMES.get(name, name): metrics for name, metrics in by_benchmark.items()}


def _rescored_splits(path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    """Each configuration's benchmark split from full_ablation rescore's summary."""
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    configurations = manifest.get("configurations") if isinstance(manifest, dict) else None
    splits = {}
    for summary in configurations if isinstance(configurations, list) else ():
        config = summary.get("config") if isinstance(summary, dict) else None
        if isinstance(config, dict):
            splits[str(config.get("id"))] = {
                kind: metrics for kind, metrics in _saved_split(summary.get("by_benchmark")).items()
                if isinstance(metrics, dict)}
    return splits


def read_run(run_dir: Path) -> dict[str, Any] | None:
    """One saved run, or None when its folder holds no readable configurations."""
    try:
        manifest = json.loads((run_dir / SUMMARY).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(manifest, dict):
        return None
    rescored = _rescored_splits(run_dir / "rescored" / SUMMARY)
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
        metrics = (saved_benchmark_metrics(rows, config.get("provider"))
                   if rows else _saved_split(summary.get("by_benchmark")))
        # rescore recomputes retrieval under rescored/ and leaves the rows as
        # they were, so its figures replace theirs.
        for kind, rescored_metrics in rescored.get(config_id, {}).items():
            metrics[kind] = {**metrics.get(kind, {}), **rescored_metrics}
        configurations.append({
            "run_id": run_dir.name,
            "config_id": config_id,
            "name": str(config.get("name") or config_id),
            "questions": int(summary.get("questions") or len(rows)),
            "incomplete": incomplete(summary, rows),
            "benchmark_metrics": metrics,
        })
    if not configurations:
        return None
    # The saved summary is only for a run whose per-question answers are not here:
    # rescore writes its fresh one elsewhere, so the top-level one can be stale.
    saved_quality = _answer_quality(run_dir / "answer-summary.json")
    answer_quality, unscored = {}, []
    for item in configurations:
        config_id = item["config_id"]
        answers = run_dir / config_id / "answers.jsonl"
        answered = _answered(answers)
        if answered is None and answers.is_file():
            unscored.append(config_id)
        quality = answered or saved_quality.get(config_id)
        if quality is not None:
            answer_quality[config_id] = quality
    return {"run_id": run_dir.name, "configurations": configurations,
            "answer_quality": answer_quality, "unscored_answers": unscored}


def _answered(path: Path) -> dict[str, Any] | None:
    """Recompute answer quality from saved per-question answers, as rescore does."""
    rows = read_result_rows(path)
    if not rows:
        return None
    # As with retrieval rows, a judged score that is not a number from 0 to 1 is
    # left out of its mean and count; valid rows score exactly as rescore does.
    rows = [{**row, "quality": {**row["quality"], **{
                metric: _score(row["quality"].get(metric)) for metric in QUALITY_LABELS}}}
            if isinstance(row.get("quality"), dict) else row for row in rows]
    try:
        return answer_scores(rows)
    # A row of the wrong shape anywhere in it, a missing key or a check that is
    # not a record, falls back rather than taking the Results page down.
    except (KeyError, TypeError, AttributeError, OverflowError):
        return None


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


def _without_empty_columns(rows: list[dict[str, Any]],
                           paired: dict[str, str] | None = None) -> list[dict[str, Any]]:
    """Drop columns no row fills; a column in ``paired`` stays while its partner is filled."""
    filled = {key for row in rows for key, value in row.items()
              if value is not None and not (isinstance(value, float) and isnan(value))}
    filled |= {key for key, partner in (paired or {}).items() if partner in filled}
    return [{key: value for key, value in row.items() if key in filled} for row in rows]


def retrieval_table(configurations: list[dict[str, Any]], benchmark: str) -> list[dict[str, Any]]:
    rows = []
    for item in configurations:
        saved = item["benchmark_metrics"].get(benchmark, {})
        counts = _record(saved.get("metric_denominators"))
        scores = {metric: _score(saved.get(metric)) for metric in METRIC_LABELS}
        relevance = any(scores[metric] is not None for metric in RELEVANCE_METRICS)
        hard_negatives = scores["hard_negative_accuracy"] is not None
        # Recall, nDCG and MRR score the same questions: those with a gold passage.
        rows.append({"Configuration": item["config_id"], "Name": item["name"],
                     "Questions": saved.get("questions", item["questions"]),
                     "Relevance-scored": counts.get("recall") if relevance else None,
                     **{label: scores[metric] for metric, label in METRIC_LABELS.items()
                        if metric != "hard_negative_accuracy"},
                     "Hard negatives": (counts.get("hard_negative_accuracy")
                                        if hard_negatives else None),
                     METRIC_LABELS["hard_negative_accuracy"]: scores["hard_negative_accuracy"]})
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


def _record(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _score(value: Any) -> float | None:
    """A saved score as a number, or None for anything else: text, true, NaN.

    Every score here is a proportion, so one outside 0-1 is not a score either.
    The comparison rejects NaN and infinities, and never converts an integer
    too large for a float.
    """
    if isinstance(value, (int, float)) and not isinstance(value, bool) and 0 <= value <= 1:
        return float(value) + 0.0  # + 0.0 turns -0.0 into 0.0
    return None


def answer_quality_table(run: dict[str, Any], benchmark: str) -> list[dict[str, Any]]:
    """Judged answer quality per configuration, each mean beside how many outputs it scored.

    The matrix judges answers to team questions only, so the mechanical
    benchmark has none.
    """
    if benchmark != TEAM:
        return []
    rows = []
    for item in run["configurations"]:
        saved = run.get("answer_quality", {}).get(item["config_id"])
        if saved is None:
            continue
        row = {"Configuration": item["config_id"], "Name": item["name"],
               "Outputs": saved.get("questions")}
        # A saved summary is read as it is found: a part of the wrong shape is left blank.
        quality = _record(saved.get("quality"))
        for metric, label in QUALITY_LABELS.items():
            measured = _record(quality.get(metric))
            row[label] = _score(measured.get("mean"))
            row[f"{label} scored"] = measured.get("scored")
        checks = _record(saved.get("checks"))
        row["Claim checks"] = ", ".join(f"{status} {count}"
                                        for status, count in sorted(checks.items())) or None
        rows.append(row)
    # A mean over no scored outputs is shown as a dash beside its 0, not dropped.
    return _without_empty_columns(rows, {label: f"{label} scored"
                                         for label in QUALITY_LABELS.values()})


def chart_rows(configurations: list[dict[str, Any]], benchmark: str) -> list[dict[str, Any]]:
    """Return only the four retrieval metrics; counts are not scores."""
    rows = []
    for item in configurations:
        saved = item["benchmark_metrics"].get(benchmark, {})
        for metric, label in METRIC_LABELS.items():
            value = _score(saved.get(metric))
            if value is not None:
                rows.append({"Configuration": item["config_id"],
                             "Metric": label,
                             "Value": value})
    return rows


def quality_chart_rows(run: dict[str, Any], benchmark: str) -> list[dict[str, Any]]:
    """The judged answer-quality means of ``answer_quality_table``, one row per bar."""
    return [{"Configuration": row["Configuration"], "Metric": label, "Value": row[label]}
            for row in answer_quality_table(run, benchmark)
            for label in QUALITY_LABELS.values() if row.get(label) is not None]


def retrieval_configurations(configurations: list[dict[str, Any]],
                             benchmark: str) -> list[dict[str, Any]]:
    """The configurations a benchmark's retrieval table lists: those saved for
    it, with at least one retrieval score on either benchmark."""
    return [item for item in configurations
            if benchmark in item["benchmark_metrics"]
            and any(any(_score(item["benchmark_metrics"].get(kind, {}).get(metric)) is not None
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
    unanswered = []
    for item in run["configurations"]:
        answered = run.get("answer_quality", {}).get(item["config_id"], {}).get("questions")
        retrieved = item["benchmark_metrics"].get(TEAM, {}).get("questions")
        if isinstance(answered, int) and isinstance(retrieved, int) and answered < retrieved:
            unanswered.append(f"{item['config_id']} ({answered} of {retrieved} team questions answered)")
    unscored = run.get("unscored_answers", [])
    if unscored:
        warnings.append("Unscored answers: " + ", ".join(unscored)
                        + ". Their answers.jsonl could not be read or scored, so their answer "
                        "quality is the saved answer-summary.json, which may be out of date, "
                        "or is not shown.")
    if unanswered:
        warnings.append("Incomplete answers: " + ", ".join(unanswered)
                        + ". Their answer quality is partial and is not comparable to a complete run.")
    return warnings


def _cell(value: Any) -> str:
    if value is None or (isinstance(value, float) and isnan(value)):
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    # One table row per line: line breaks become spaces, and a pipe is escaped.
    return " ".join(str(value).split()).replace("|", "\\|")


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
# Paired with every charted label, in chart order; zip(strict=True) refuses a
# label added to either map without a colour of its own.
METRIC_COLOURS = dict(zip((*METRIC_LABELS.values(), *QUALITY_LABELS.values()),
                          ("#4269d0", "#efb118", "#ff725c", "#6cc5b0", "#3ca951",
                           "#ff8ab7", "#a463f2", "#97bbf5", "#9c6b4e"), strict=True))


def svg_chart(rows: list[dict[str, Any]], title: str) -> str:
    """A grouped bar chart of ``chart_rows`` on a 0-1 axis: one group per configuration."""
    configurations = list(dict.fromkeys(row["Configuration"] for row in rows))
    metrics = [label for label in METRIC_COLOURS if any(row["Metric"] == label for row in rows)]
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
    for title, kind in BENCHMARKS:
        matching = retrieval_configurations(run["configurations"], kind)
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
        quality_chart = quality_chart_rows(run, kind)
        if quality_chart:
            name = f"{kind}-answer-quality.svg"
            files[name] = svg_chart(quality_chart, f"{title}: answer quality")
            section += f"\n![Answer quality]({name})\n"
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


def _out_folder(value: str) -> Path:
    # Path("") is the current folder, which a forgotten value should not mean.
    if not value:
        raise argparse.ArgumentTypeError("--out needs a folder")
    return Path(value)


def _folders_up(path: Path) -> tuple[Path, ...]:
    """``path`` with symlinks resolved, then each folder above it."""
    resolved = Path(os.path.realpath(path))
    return (resolved, *resolved.parents)


def _inside(path: Path, folder: Path) -> bool:
    """Whether ``path`` is ``folder`` or under it, by file identity, so neither a
    symlink nor a case-insensitive filesystem's other spelling gets past."""
    return any(candidate.exists() and os.path.samefile(candidate, folder)
               for candidate in _folders_up(path))


def _saved_run_holding(path: Path) -> Path | None:
    """The saved run ``path`` is or lies under, if any: the nearest folder with a
    summary.json. A lookup on disk, so a symlink or a case-insensitive
    filesystem's other spelling of a run is found too."""
    return next((candidate for candidate in _folders_up(path)
                 if (candidate / SUMMARY).is_file()), None)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m src.evaluation.report",
        description="Turn one saved run directory into its ablation tables and metric charts.")
    parser.add_argument("run_dir", type=Path, help="a run under results/, e.g. results/full-matrix-20261017")
    parser.add_argument("--out", type=_out_folder,
                        help="where to write the report (default: docs/evaluation/<run-id>/)")
    args = parser.parse_args(argv)
    # Normalised, not resolved: "results/x/C4/.." is run x, and a symlinked
    # run keeps the name it is listed under.
    if not args.run_dir.is_dir():
        parser.error(f"{args.run_dir} does not exist or is not a folder")
    run = read_run(Path(os.path.abspath(args.run_dir)))
    if run is None:
        parser.error(f"{args.run_dir} holds no summary.json with a readable configuration")
    out = args.out or REPORT_ROOT / run["run_id"]
    try:
        if out.exists() and not out.is_dir():
            parser.error(f"--out {out} is not a folder")
        # A run's own files are hashed by the tools that publish it, so a
        # report beside them would change what they check.
        # The run's own configuration folders hold summary.json too, so the run
        # being read is looked for first, then any other saved run.
        if _inside(out, args.run_dir):
            parser.error(f"--out {out} is inside the run it reads; write the report elsewhere")
        holder = _saved_run_holding(out)
        if holder is not None:
            parser.error(f"--out {out} is inside a saved run ({holder}); write the report elsewhere")
        write_report(run, out)
    except OSError as error:
        parser.error(f"cannot write the report to {out}: {error.strerror or error}")


if __name__ == "__main__":
    main()
