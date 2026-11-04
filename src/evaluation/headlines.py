"""Headline statistics from one saved run, never from live model calls (#165).

The passage count uses the same top-ten question measure as the Results
chart and breakdown. Correctness is a mean with partial credit, not a count
of correct answers. Baseline deltas require identical scored question IDs.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from statistics import mean, median
from typing import Any, NamedTuple

from src.evaluation import breakdown, report
from src.stack import DEFAULT_STACK


class Headline(NamedTuple):
    """One tile's value, support and optional change against the C0 baseline."""

    key: str
    value: float
    scored: int
    count: int | None = None
    delta: float | None = None


def _values(questions: Mapping[str, Any], key: str) -> dict[str, float]:
    """Finite, eligible values for one measure, retaining IDs for comparison."""
    if key == "found":
        return {qid: float(value) for qid, outcome in questions.items()
                if (value := breakdown.FOUND.score(outcome)) is not None}
    if key == "correctness":
        return {qid: value for qid, outcome in questions.items()
                if (value := report.saved_score(outcome.correctness)) is not None}
    if key == "refused":
        return {qid: float(outcome.abstained) for qid, outcome in questions.items()
                if isinstance(outcome.abstained, bool)}
    return {qid: outcome.latency_ms / 1000 for qid, outcome in questions.items()
            if outcome.latency_ms is not None}


def headline_tiles(run: Mapping[str, Any], benchmark: str,
                   outcomes: Callable[[str, str], Mapping[str, Any]]) -> list[Headline]:
    """The app default's available statistics on ``benchmark`` in ``run``.

    No substitution of another configuration when the default is absent.
    Missing observations stay missing; real zeroes remain visible. Correctness
    can use the shared report's saved team summary when per-question scores
    are absent, but such an aggregate has no IDs to justify a baseline delta.
    """
    ids = {item["config_id"] for item in run["configurations"]}
    if DEFAULT_STACK not in ids or not run.get("run_dir"):
        return []

    def read(config_id: str) -> dict[str, Any]:
        return {qid: item for qid, item in outcomes(run["run_dir"], config_id).items()
                if item.benchmark == benchmark}

    questions = read(DEFAULT_STACK)
    baseline = read(breakdown.BASELINE) if breakdown.BASELINE in ids else {}
    tiles = []
    for key in ("found", "correctness", "refused", "latency"):
        values, before = _values(questions, key), _values(baseline, key)
        if not values:
            if key == "correctness":
                rows = report.answer_quality_table(dict(run), benchmark)
                row = next((item for item in rows if item["Configuration"] == DEFAULT_STACK), {})
                score = report.saved_score(row.get("Correctness"))
                scored = row.get("Correctness scored")
                if score is not None and report.is_count(scored):
                    tiles.append(Headline(key, score, scored))
            continue
        aggregate = median if key == "latency" else mean
        value = aggregate(values.values())
        delta = value - aggregate(before.values()) if values.keys() == before.keys() else None
        count = sum(bool(value) for value in values.values()) if key in {"found", "refused"} else None
        tiles.append(Headline(key, value, len(values), count, delta))
    return tiles
