"""Score a finished chunk-size sweep again at an equal amount of text (#45).

    python notebooks/retrieval/chunk_sweep_equal_text.py chunk-sweep-fy2025-20261007
    python notebooks/retrieval/chunk_sweep_equal_text.py chunk-sweep-fy2025-20261007 3000 5000

The sweep's second cutoff is a number of passages: as many of a size as are
sure to fit the prompt the app sends, 24, 16, 12 and 7. BM25's passages fill
that to about 18,500 characters at every size. Dense and hybrid return
table passages, which are shorter than the budget, so the same cutoff held
13,654 characters at 1,200 and 7,232 at 4,000 for hybrid. The larger sizes are
compared there on less text, and a gap between them is partly that.

This reads the per-question files a run wrote under ``results/<run-id>/``. A
question is found when a supporting passage is among the first passages of its
ranking whose texts add up to no more than a given number of characters. The
first passage always counts, so no size is left with nothing.

A run stores each ranking to its deeper cutoff only: 24, 16, 12 and 10
passages. Past the text those hold, a size is being read short, so each line
also gives the share of a size's rankings that ran out before the budget was
full. Read a budget where that is near zero at every size.

Each row of a run's ``questions.jsonl`` holds the passages that support its
question and the length of every passage in its ranking, as they were when
the run measured it. Those are what is scored, so nothing under
``data/sweep/`` is read and a later run cannot change what this one scores to.

The two runs of 7 October 2026 were written before a row held either. For such
a run the lengths are read from ``data/sweep/<size>/processed`` and the
supporting passages from ``data/sweep/<size>/benchmark.jsonl``, which a later
run cuts and writes again, so both are checked first: the build's fingerprint
against the one the run recorded, and the supporting passages against what the
run scored with them. A row records whether its question was found at each of
the run's two cutoffs, and its Recall, and all three are worked out again from
the supporting passages read now. One row that disagrees and the run is
refused. That catches a changed label wherever it moved a row's score, which is
not every change a label can make: a supporting passage added or dropped
beyond both cutoffs of a ranking, or outside it, with the count the same, is
not seen. A run that stored its own has no such gap.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping, Sequence
from math import isclose
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.config import SWEEP_DIR  # noqa: E402
from src.evaluation.metrics import recall_at_k  # noqa: E402
from src.evaluation.records import RunResult  # noqa: E402
from src.pipeline.chunk import iter_chunks  # noqa: E402
from src.retrieval.records import corpus_fingerprint  # noqa: E402
from src.stack import RESULTS_ROOT  # noqa: E402

RESULTS_CSV = ROOT / "notebooks" / "retrieval" / "results" / "chunk_sweep_equal_text.csv"

# Characters of text. The smallest ranking a run stores is ten passages of the
# largest size, which for dense search held about 7,400 characters.
BUDGETS = (3000, 4000, 5000, 6000, 7000)

# What a row holds once a run stores what it was scored against.
STORED = ("supporting_chunk_ids", "retrieved_chars")


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def from_the_build(
    rankings: Sequence[dict[str, Any]], config: Mapping[str, Any], build: Mapping[str, Any],
    sweep_dir: Path, run_id: str,
) -> list[dict[str, Any]]:
    """Rows of a run that stored neither, filled in from the build it was measured on.

    The build under ``sweep_dir`` is trusted only as far as the run can vouch
    for it: its corpus by the fingerprint the run recorded, and its benchmark
    by the scores the run gave each ranking with it (see the module docstring
    for what that does not catch).
    """
    size = config["chunk_budget"]
    corpus = list(iter_chunks(processed_dir=sweep_dir / str(size) / "processed"))
    if corpus_fingerprint(corpus) != build["corpus_fingerprint"]:
        raise SystemExit(
            f"{sweep_dir / str(size)} is no longer the build {run_id} measured: a later run "
            f"cut it again, and {run_id} was written before a run stored the length of "
            "each passage it ranked. Run that sweep again under a new run id, then this."
        )
    lengths = {row["chunk_id"]: len(row["text"]) for row in corpus}
    supporting = {
        question["question_id"]: question["supporting_chunk_ids"]
        for question in _rows(sweep_dir / str(size) / "benchmark.jsonl")
    }

    top_k, context_k = config["top_k"], config["context_k"]
    filled = []
    for row in rankings:
        labels = supporting.get(row["question_id"])
        ranked = row["retrieved_chunk_ids"]
        scored_as = None if not labels else (
            bool(set(labels) & set(ranked[:top_k])),
            bool(set(labels) & set(ranked[:context_k])),
            recall_at_k(
                RunResult(row["question_id"], row["retriever"], tuple(ranked[:top_k]),
                          tuple(row["retrieved_scores"][:top_k])),
                labels, k=top_k),
        )
        recorded = (row["hit"], row["context_hit"], row["recall"])
        if scored_as is None or scored_as[:2] != recorded[:2] or not isclose(
                scored_as[2], recorded[2], abs_tol=1e-9):
            raise SystemExit(
                f"{sweep_dir / str(size) / 'benchmark.jsonl'} is no longer the benchmark "
                f"{run_id} was scored against: for {row['question_id']} ({config['id']}) it "
                "gives supporting passages the run's own scores do not agree with. A later "
                f"run wrote it again, and {run_id} was written before a run stored each "
                "question's supporting passages. Run that sweep again under a new run id, "
                "then this."
            )
        filled.append({**row, "supporting_chunk_ids": labels,
                       "retrieved_chars": [lengths[chunk_id] for chunk_id in ranked]})
    return filled


def rescore(
    run_dir: Path, budgets: Sequence[int] = BUDGETS, sweep_dir: Path = SWEEP_DIR,
) -> list[dict[str, Any]]:
    """A run's rankings scored at each number of characters: one row for each
    retriever, chunk size and number of characters."""
    if not budgets or any(budget < 1 for budget in budgets):
        raise ValueError("a number of characters must be 1 or more")
    manifest = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    builds = {build["chunk_budget"]: build for build in manifest["builds"]}

    scored = []
    for summary in manifest["configurations"]:
        config = summary["config"]
        rankings = _rows(run_dir / config["id"] / "questions.jsonl")
        if not all(name in row for row in rankings for name in STORED):
            rankings = from_the_build(
                rankings, config, builds[config["chunk_budget"]], sweep_dir, manifest["run_id"])
        for budget in budgets:
            found = ran_out = taken_in_all = 0
            for row in rankings:
                taken: list[str] = []
                held = 0
                for chunk_id, length in zip(row["retrieved_chunk_ids"], row["retrieved_chars"]):
                    if taken and held + length > budget:
                        break
                    taken.append(chunk_id)
                    held += length
                else:
                    # Every stored passage fitted, so there may have been room for more.
                    ran_out += 1
                found += bool(set(row["supporting_chunk_ids"]) & set(taken))
                taken_in_all += len(taken)
            scored.append({
                "run_id": manifest["run_id"], "retriever": config["retriever"],
                "characters": budget, "chunk_budget": config["chunk_budget"],
                "questions": len(rankings),
                "hit_rate": found / len(rankings),
                "passages_taken": taken_in_all / len(rankings),
                "ran_out": ran_out / len(rankings),
            })
    return scored


def main(argv: Sequence[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    usage = __doc__.split("\n\n")[1]
    if not arguments:
        raise SystemExit(usage)
    run_dir = RESULTS_ROOT / arguments[0]
    try:
        budgets = tuple(int(value) for value in arguments[1:]) or BUDGETS
    except ValueError:
        raise SystemExit(f"a number of characters must be a whole number\n{usage}") from None
    if not (run_dir / "summary.json").is_file():
        raise SystemExit(f"no finished run at {run_dir}: there is no summary.json in it")
    try:
        scored = rescore(run_dir, budgets, SWEEP_DIR)
    except ValueError as error:
        raise SystemExit(f"{error}\n{usage}") from None
    except FileNotFoundError as error:
        raise SystemExit(
            f"{error.filename} is missing. A run's per-question files are not committed, so "
            "this needs the machine the run was measured on."
        ) from None

    table = pd.DataFrame(scored)
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS_CSV, index=False, encoding="utf-8-sig")
    print(f"{scored[0]['run_id']}: supporting passage within the first N characters of a ranking")
    for name in ("hit_rate", "ran_out"):
        wide = table.pivot(index=["retriever", "characters"], columns="chunk_budget", values=name)
        print(f"\n{name}")
        print(wide.to_string(float_format=lambda value: f"{value:.1%}"))
    print("\nSaved", RESULTS_CSV.relative_to(ROOT))


if __name__ == "__main__":
    main()
