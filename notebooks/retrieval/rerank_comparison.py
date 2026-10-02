"""Whether reranking hybrid's candidates puts more answers in the prompt.

``CrossEncoderReranker`` (#21) is built and tested, and nothing on the answer
path uses it: the app searches with Hybrid or BM25, the evaluation harness
offers bm25, dense and hybrid, and the ablation matrix stops at C4. This
measures what wiring it in would buy, on the questions the other retrieval
settings were measured on:

- hybrid: the top FINAL_K as the app searches now.
- hybrid + rerank: hybrid's top CANDIDATE_K, re-scored by the cross-encoder,
  as ``CrossEncoderReranker(hybrid).search`` returns them. The table boost a
  figure question carries is applied to the cross-encoder's scores too,
  because ``WrappingRetriever.search`` passes it on.
- hybrid + rerank, scores as they are: the same scores with no boost on them,
  to see what that lean does on the cross-encoder's scale.

Two question sets, as in ``table_boost_sweep.py``, whose measures these are:
the 48 hand-written questions, and PER_FILING questions from each filing of
the XBRL benchmark with that script's seed. Fewer per filing than the other
sweeps use, because the cross-encoder reads 50 passages for every question
and takes seconds over it on a laptop; the seconds are reported too.

Rows land in ``notebooks/retrieval/results/rerank_comparison.csv`` (the 48)
and ``rerank_comparison_xbrl.csv`` (the benchmark), and the summaries print.

    python -m src.retrieval benchmark          # writes benchmark/generated.jsonl
    python notebooks/retrieval/rerank_comparison.py
"""

from __future__ import annotations

import random
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

from search_text_comparison import load_questions as load_written, measure, summarize  # noqa: E402
from src.evaluation.benchmark import (  # noqa: E402
    DEFAULT_GENERATED_QUESTIONS_PATH,
    load_questions as load_generated,
)
from src.evaluation.metrics import score_question  # noqa: E402
from src.evaluation.records import RunResult  # noqa: E402
from src.rag.query import parse_question  # noqa: E402
from src.retrieval.base import reorder  # noqa: E402
from src.retrieval.bm25 import BM25Retriever  # noqa: E402
from src.retrieval.constants import CANDIDATE_K, FINAL_K, RERANK  # noqa: E402
from src.retrieval.dense import DenseRetriever  # noqa: E402
from src.retrieval.hybrid import HybridRetriever  # noqa: E402
from src.retrieval.rerank import CrossEncoderReranker  # noqa: E402
from table_boost_sweep import SEED, generated_query  # noqa: E402

RESULTS_DIR = ROOT / "notebooks" / "retrieval" / "results"
WRITTEN_CSV = RESULTS_DIR / "rerank_comparison.csv"
XBRL_CSV = RESULTS_DIR / "rerank_comparison_xbrl.csv"

PER_FILING = 3

HYBRID = "hybrid"
RERANKED = "hybrid + rerank"
UNBOOSTED = "hybrid + rerank, scores as they are"


def sample_generated(questions):
    """PER_FILING questions from each filing, the same ones every run."""
    by_filing = defaultdict(list)
    for question in questions:
        by_filing[(question.ticker, question.fiscal_year)].append(question)
    rng = random.Random(SEED)
    sample = []
    for filing in sorted(by_filing):
        held = by_filing[filing]
        sample += rng.sample(held, min(PER_FILING, len(held)))
    return sample


def searches(reranker, query):
    """The top CANDIDATE_K three ways, and the seconds the cross-encoder took.

    One search and one scoring pass: the candidates are hybrid's, the scores
    the cross-encoder's, and ``base.reorder`` orders them as
    ``WrappingRetriever.search`` does, with the query's boost and without.
    """
    found = reranker.inner.search(query, k=CANDIDATE_K)
    started = time.perf_counter()
    scores = reranker.score(query, found) if found else []
    seconds = time.perf_counter() - started
    ordered = {
        HYBRID: found,
        RERANKED: reorder(zip(found, scores, strict=True), retriever=RERANK, k=CANDIDATE_K,
                          table_boost=query.table_boost),
        UNBOOSTED: reorder(zip(found, scores, strict=True), retriever=RERANK, k=CANDIDATE_K,
                           table_boost=1.0),
    }
    return ordered, seconds


def check_reorder_is_the_reranker(reranker, queries):
    """The reordered list must be what the reranker itself returns."""
    for query in queries:
        ordered, _ = searches(reranker, query)
        searched = reranker.search(query, k=CANDIDATE_K)
        if [p.chunk_id for p in ordered[RERANKED]] != [p.chunk_id for p in searched]:
            raise SystemExit(f"the reordered list differs from CrossEncoderReranker.search on "
                             f"{query.text!r}: the comparison would be measuring something else")
    print(f"check: matches CrossEncoderReranker.search on {len(queries)} questions", flush=True)


def written_rows(reranker, questions):
    rows = []
    for q in questions:
        query = parse_question(q["question"]).to_query(top_k=CANDIDATE_K)
        ordered, seconds = searches(reranker, query)
        for name, passages in ordered.items():
            rows.append({"qid": q["qid"], "retriever": name, "text": "as shipped",
                         "wants_figures": query.wants_figures, "rerank_seconds": round(seconds, 2),
                         **measure(q, passages)})
    return rows


def generated_rows(reranker, questions, kind_of):
    rows = []
    for number, question in enumerate(questions, start=1):
        query = generated_query(question)
        kinds = {kind_of[chunk_id] for chunk_id in question.supporting_chunk_ids}
        ordered, seconds = searches(reranker, query)
        for name, passages in ordered.items():
            top = passages[:FINAL_K]
            # A RunResult is named for the retriever its passages carry, which
            # is "hybrid" or "rerank" and not this comparison's row label.
            metrics = score_question(
                question,
                RunResult.from_passages(question.question_id, top,
                                        retriever=top[0].retriever if top else name),
                k=FINAL_K,
            )
            rows.append({"qid": question.question_id, "retriever": name,
                         "wants_figures": query.wants_figures,
                         "supported_by": "tables" if kinds == {"table"} else
                                         "prose" if "table" not in kinds else "both",
                         "recall": metrics["recall"], "ndcg": round(metrics["ndcg"], 4),
                         "mrr": round(metrics["mrr"], 4),
                         "tables": sum(passage.content_type == "table" for passage in top),
                         "rerank_seconds": round(seconds, 2)})
        if number % 25 == 0:
            print(f"  {number:,} of {len(questions):,}", flush=True)
    return rows


def summarize_generated(table):
    out = {}
    for name, rows in table.groupby("retriever", sort=False):
        hit = rows.recall > 0
        row = {
            f"Supporting chunk in top {FINAL_K} (of {len(rows):,})": int(hit.sum()),
            "share": round(hit.mean(), 3),
            "Recall, mean": round(rows.recall.mean(), 3),
            "nDCG, mean": round(rows.ndcg.mean(), 3),
            "Reciprocal rank, mean": round(rows.mrr.mean(), 3),
            f"Tables among the top {FINAL_K}, mean": round(rows.tables.mean(), 1),
        }
        for support in ("tables", "both", "prose"):
            mine = rows.supported_by == support
            row[f"  supported by {support} ({int(mine.sum()):,})"] = int(hit[mine].sum())
        out[name] = row
    return pd.DataFrame(out).T


def main() -> None:
    bm25 = BM25Retriever.load()
    reranker = CrossEncoderReranker(HybridRetriever(bm25, DenseRetriever.load()))
    kind_of = {chunk["chunk_id"]: chunk["content_type"] for chunk in bm25.chunks}

    written = load_written()
    generated = sample_generated(load_generated(DEFAULT_GENERATED_QUESTIONS_PATH))
    check_reorder_is_the_reranker(
        reranker,
        [parse_question(q["question"]).to_query(top_k=CANDIDATE_K) for q in written[:3]]
        + [generated_query(question) for question in generated[:3]],
    )

    print(f"hand-written: {len(written)} questions", flush=True)
    written_table = pd.DataFrame(written_rows(reranker, written))
    print(f"XBRL benchmark: {len(generated):,} questions, {PER_FILING} from each filing",
          flush=True)
    generated_table = pd.DataFrame(generated_rows(reranker, generated, kind_of))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    written_table.to_csv(WRITTEN_CSV, index=False, encoding="utf-8-sig")
    generated_table.to_csv(XBRL_CSV, index=False, encoding="utf-8-sig")
    print("\nSaved", WRITTEN_CSV.relative_to(ROOT), "and", XBRL_CSV.relative_to(ROOT))
    report(written_table, generated_table)


def report(written_table, generated_table) -> None:
    """Print the two summaries for a finished run."""
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print("\n--- 48 hand-written questions ---")
        print(summarize(written_table).T.to_string())
        print(f"\n--- XBRL benchmark (#24), scored with #25's metrics at {FINAL_K} ---")
        print(summarize_generated(generated_table).to_string())
        seconds = pd.concat([written_table, generated_table]).drop_duplicates("qid").rerank_seconds
        print(f"\nCross-encoder, seconds per question over {CANDIDATE_K} candidates: "
              f"median {seconds.median():.1f}, slowest {seconds.max():.1f}")


if __name__ == "__main__":
    # Re-print a finished run's summaries without searching again.
    if "--report" in sys.argv:
        report(pd.read_csv(WRITTEN_CSV, encoding="utf-8-sig"),
               pd.read_csv(XBRL_CSV, encoding="utf-8-sig"))
    else:
        main()
