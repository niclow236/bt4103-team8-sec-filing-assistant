"""Shared run-directory validation and C/E/G comparison files."""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

COMPARISON_FIELDS = (
    "config", "name", "questions", "recall", "ndcg", "mrr",
    "hard_negative_accuracy",
)


def _safe_run_id(run_id: str) -> str:
    value = run_id.strip()
    if not value or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError("run_id must contain only letters, numbers, ., _, and -")
    return value


def available_run_dir(results_root: Path, run_id: str) -> Path:
    run_dir = results_root / _safe_run_id(run_id)
    if run_dir.exists():
        raise FileExistsError(f"{run_dir} already holds a run; use a new --run-id")
    return run_dir


def write_comparison(
    run_dir: Path, *, run_id: str, top_k: int,
    summaries: Sequence[Mapping[str, Any]], extra_fields: Sequence[str] = (),
    **metadata: Any,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    fields = (*COMPARISON_FIELDS, *extra_fields)
    with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for summary in summaries:
            config = summary["config"]
            writer.writerow({**config, **summary, "config": config["id"], "name": config["name"]})
    manifest = {"run_id": run_id, "top_k": top_k, "configurations": list(summaries), **metadata}
    (run_dir / "summary.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
