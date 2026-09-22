"""Which fusion weights to give a question that asks for a figure (#86).

Reciprocal rank fusion with equal weights rewards the passage both retrievers
return. BM25 rarely returns a financial statement table, which names neither
the company nor the fiscal year and writes the year as "December 31, 2025", so
a table only the dense retriever ranks highly loses to prose both agree on.
The traced examples: Meta's FY2025 total assets, dense rank 4 and hybrid 14;
Apple's FY2022 accounts payable, 8 and 20; Meta's FY2022 total liabilities,
11 and 23.

This runs the 48 test questions through hybrid retrieval at each candidate
weight pair, and reports the same measures as ``search_text_comparison.py``:
whether the expected figure reached the top 8, where it ranked, how much of a
prose answer's wording the top 8 held, and whether the right Item was in it.
Dense keeps weight 1.0 throughout and BM25's is the swept value, since only
their ratio matters to the ranking.

Per-question rows land in ``notebooks/retrieval/results/fusion_weight_sweep.csv``
and the summary prints. Prose questions are measured at every pair too: the
shipped setting may not cost them anything, and this is what says whether it
does.

    python notebooks/retrieval/fusion_weight_sweep.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

from search_text_comparison import load_questions, measure, summarize  # noqa: E402
from src.rag.query import parse_question  # noqa: E402
from src.retrieval.bm25 import BM25Retriever  # noqa: E402
from src.retrieval.constants import BM25, CANDIDATE_K, DENSE  # noqa: E402
from src.retrieval.dense import DenseRetriever  # noqa: E402
from src.retrieval.hybrid import HybridRetriever  # noqa: E402

RESULTS_CSV = ROOT / "notebooks" / "retrieval" / "results" / "fusion_weight_sweep.csv"

# Dense is held at 1.0 and BM25 swept, so 1.0 is today's equal weighting and
# 0.0 is dense alone. The ones between say whether the gain is BM25 being
# quieter or BM25 being absent, which a single "dense only" row cannot.
BM25_WEIGHTS = (1.0, 0.7, 0.5, 0.3, 0.0)


def main() -> None:
    questions = load_questions()
    bm25, dense = BM25Retriever.load(), DenseRetriever.load()

    rows = []
    for weight in BM25_WEIGHTS:
        weights = {BM25: weight, DENSE: 1.0}
        # Both weightings are set to the swept pair, so every question in the
        # run is fused the same way and the figure and prose columns are
        # comparable. The shipped code applies the pair to figure questions
        # only; that split is the decision this measurement informs.
        retriever = HybridRetriever(bm25, dense, weights=weights, figure_weights=weights)
        for question in questions:
            query = parse_question(question["question"]).to_query(top_k=CANDIDATE_K)
            passages = retriever.search(query)
            rows.append({
                "qid": question["qid"],
                "retriever": "hybrid",
                "text": f"bm25={weight}",
                **measure(question, passages),
            })
        print(f"bm25 weight {weight}: {len(questions)} questions", flush=True)

    table = pd.DataFrame(rows)
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS_CSV, index=False, encoding="utf-8-sig")
    print("\nSaved", RESULTS_CSV.relative_to(ROOT))
    print(summarize(table).to_string())


if __name__ == "__main__":
    main()
