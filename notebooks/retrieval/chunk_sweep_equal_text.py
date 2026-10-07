"""Score a finished chunk-size sweep again at an equal amount of text (#45).

    python notebooks/retrieval/chunk_sweep_equal_text.py chunk-sweep-fy2025-20261007
    python notebooks/retrieval/chunk_sweep_equal_text.py chunk-sweep-fy2025-20261007 3000 5000

The sweep's second cutoff is a number of passages: as many of a size as are
sure to fit the prompt the app sends, 24, 16, 12 and 7. BM25's passages fill
that to about 18,500 characters at every size. Dense and hybrid return
table passages, which are shorter than the budget, so the same cutoff held
13,654 characters at 1,200 and 7,232 at 4,000 for hybrid. The larger sizes are
compared there on less text, and a gap between them is partly that.

This reads the per-question files a run wrote under ``results/<run-id>/`` and
the builds it was measured on under ``data/sweep/``. A question is found when
a supporting passage is among the first passages of its ranking whose texts
add up to no more than a given number of characters. The first passage always
counts, so no size is left with nothing.

A run stores each ranking to its deeper cutoff only: 24, 16, 12 and 10
passages. Past the text those hold, a size is being read short, so each line
also gives the share of a size's rankings that ran out before the budget was
full. Read a budget where that is near zero at every size.

The builds have to be the ones the run measured, and a later run over other
filings cuts ``data/sweep/`` again. Each build's fingerprint is checked against
the one the run recorded before anything is scored.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.config import SWEEP_DIR  # noqa: E402
from src.pipeline.chunk import iter_chunks  # noqa: E402
from src.retrieval.records import corpus_fingerprint  # noqa: E402
from src.stack import RESULTS_ROOT  # noqa: E402

RESULTS_CSV = ROOT / "notebooks" / "retrieval" / "results" / "chunk_sweep_equal_text.csv"

# Characters of text. The smallest ranking a run stores is ten passages of the
# largest size, which for dense search held about 7,400 characters.
BUDGETS = (3000, 4000, 5000, 6000, 7000)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__.split("\n\n")[1])
    run_dir = RESULTS_ROOT / sys.argv[1]
    budgets = tuple(int(value) for value in sys.argv[2:]) or BUDGETS
    manifest = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

    lengths: dict[int, dict[str, int]] = {}
    supporting: dict[int, dict[str, set[str]]] = {}
    for build in manifest["builds"]:
        size = build["chunk_budget"]
        rows = list(iter_chunks(processed_dir=SWEEP_DIR / str(size) / "processed"))
        if corpus_fingerprint(rows) != build["corpus_fingerprint"]:
            raise SystemExit(
                f"data/sweep/{size}/ is no longer the build {manifest['run_id']} measured: a "
                "later run cut it again. Run that sweep again under a new run id, then this."
            )
        lengths[size] = {row["chunk_id"]: len(row["text"]) for row in rows}
        questions = (SWEEP_DIR / str(size) / "benchmark.jsonl").read_text(encoding="utf-8")
        supporting[size] = {
            question["question_id"]: set(question["supporting_chunk_ids"])
            for question in map(json.loads, questions.splitlines())
        }

    scored = []
    for summary in manifest["configurations"]:
        config = summary["config"]
        size = config["chunk_budget"]
        lines = (run_dir / config["id"] / "questions.jsonl").read_text(encoding="utf-8")
        rankings = [json.loads(line) for line in lines.splitlines()]
        for budget in budgets:
            found = ran_out = taken_in_all = 0
            for row in rankings:
                taken: list[str] = []
                held = 0
                for chunk_id in row["retrieved_chunk_ids"]:
                    if taken and held + lengths[size][chunk_id] > budget:
                        break
                    taken.append(chunk_id)
                    held += lengths[size][chunk_id]
                else:
                    # Every stored passage fitted, so there may have been room for more.
                    ran_out += 1
                found += bool(supporting[size][row["question_id"]] & set(taken))
                taken_in_all += len(taken)
            scored.append({
                "run_id": manifest["run_id"], "retriever": config["retriever"],
                "characters": budget, "chunk_budget": size, "questions": len(rankings),
                "hit_rate": found / len(rankings),
                "passages_taken": taken_in_all / len(rankings),
                "ran_out": ran_out / len(rankings),
            })

    table = pd.DataFrame(scored)
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS_CSV, index=False, encoding="utf-8-sig")
    print(f"{manifest['run_id']}: supporting passage within the first N characters of a ranking")
    for name in ("hit_rate", "ran_out"):
        wide = table.pivot(index=["retriever", "characters"], columns="chunk_budget", values=name)
        print(f"\n{name}")
        print(wide.to_string(float_format=lambda value: f"{value:.1%}"))
    print("\nSaved", RESULTS_CSV.relative_to(ROOT))


if __name__ == "__main__":
    main()
