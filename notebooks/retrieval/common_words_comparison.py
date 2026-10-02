"""Whether the keyword search should score the words that most passages contain.

BM25 weighs a word of the question by how rare it is. Okapi's IDF goes
negative for a word in more than half the passages, and rank_bm25 does not
score such a word at nothing: it gives it a floor, a quarter of the average
IDF. On this corpus ten words are over half the passages ("the", "of", "and",
"to", "in", "for", "a", "as", "on", "our") and the floor is 1.89, where
"revenue" weighs 1.61 and "total" 1.04. So in "What was the total value of
Goodwill at the end?" the two "the" and the "of" count for more than the
line item, and a statement table, which holds none of them, sinks below prose
that holds all of them: Salesforce's balance sheet ranked 260th of 397.

This runs both ways of scoring over the same questions, with the shipped
table boost and the same filters, so the only difference is whether the
common words are scored:

- every word: the search as it was.
- without the common words: what ``BM25Retriever.search`` does now.

Two question sets, as in ``table_boost_sweep.py``, whose sample and measures
these are:

- The 48 hand-written questions, with the checks
  ``search_text_comparison.py`` uses.
- The XBRL benchmark, PER_FILING questions from each filing with a fixed seed,
  scored with the retrieval metrics (#25) at FINAL_K and split by what
  supports the question.

Dense retrieval does not change, so it is searched once per question and the
real ``HybridRetriever`` fuses it with each keyword search.
``check_replay_is_exact`` compares that with real BM25 and hybrid searches
before the numbers are trusted.

Rows land in ``notebooks/retrieval/results/common_words_comparison.csv`` (the
48) and ``common_words_comparison_xbrl.csv`` (the benchmark), and the
summaries print.

    python -m src.retrieval benchmark          # writes benchmark/generated.jsonl
    python notebooks/retrieval/common_words_comparison.py
"""

from __future__ import annotations

import copy
import sys
from dataclasses import replace
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
from src.retrieval.bm25 import BM25Retriever  # noqa: E402
from src.retrieval.constants import BM25, CANDIDATE_K, DENSE, FINAL_K, HYBRID  # noqa: E402
from src.retrieval.dense import DenseRetriever  # noqa: E402
from src.retrieval.hybrid import HybridRetriever  # noqa: E402
from table_boost_sweep import PER_FILING, Replayed, generated_query, sample_generated  # noqa: E402

RESULTS_DIR = ROOT / "notebooks" / "retrieval" / "results"
WRITTEN_CSV = RESULTS_DIR / "common_words_comparison.csv"
XBRL_CSV = RESULTS_DIR / "common_words_comparison_xbrl.csv"

EVERY_WORD = "every word"
WITHOUT_COMMON = "without the common words"


def scoring_every_word(bm25):
    """The same index, scoring a question on all of its words, as it used to."""
    every = copy.copy(bm25)
    every._common = frozenset()
    return every


def searches(keyword, dense, query):
    """The top CANDIDATE_K for BM25 alone and for hybrid, under each scoring.

    Yields ``(retriever name, scoring, passages)``. Each first stage is
    searched once and replayed, as ``table_boost_sweep.py`` does: BM25 for
    every admitted passage, and dense for its top tables and its top others,
    which is the union ``DenseRetriever.search`` builds for a boosted query
    and holds the top of everything for an unboosted one.
    """
    plain = replace(query, table_boost=1.0)
    vectors = Replayed(DENSE, dense.search(replace(plain, content_type="table"), k=CANDIDATE_K)
                       + dense.search(replace(plain, content_type="prose"), k=CANDIDATE_K))
    for scoring, bm25 in keyword.items():
        sparse = Replayed(BM25, bm25.search(plain, k=len(bm25.chunks)))
        yield BM25, scoring, sparse.search(query)
        yield HYBRID, scoring, HybridRetriever(sparse, vectors).search(query)


def check_replay_is_exact(keyword, dense, queries):
    """The replayed lists must be what the real retrievers return."""
    for query in queries:
        for name, scoring, passages in searches(keyword, dense, query):
            real = keyword[scoring] if name == BM25 else HybridRetriever(keyword[scoring], dense)
            if [p.chunk_id for p in passages] != [p.chunk_id for p in real.search(query)]:
                raise SystemExit(
                    f"the replayed list differs from a real {name} search ({scoring}) on "
                    f"{query.text!r}: the comparison would be measuring something else"
                )
    print(f"replay check: matches a real search for BM25 and hybrid under both scorings, on "
          f"{len(queries)} questions", flush=True)


def written_rows(keyword, dense, questions):
    rows = []
    for q in questions:
        query = parse_question(q["question"]).to_query(top_k=CANDIDATE_K)
        for name, scoring, passages in searches(keyword, dense, query):
            rows.append({"qid": q["qid"], "retriever": name, "text": scoring,
                         "wants_figures": query.wants_figures, **measure(q, passages)})
    return rows


def generated_rows(keyword, dense, questions, kind_of):
    """One row per question, retriever and scoring, with the metrics at FINAL_K."""
    rows = []
    for number, question in enumerate(questions, start=1):
        query = generated_query(question)
        kinds = {kind_of[chunk_id] for chunk_id in question.supporting_chunk_ids}
        for name, scoring, passages in searches(keyword, dense, query):
            top = passages[:FINAL_K]
            metrics = score_question(
                question,
                RunResult.from_passages(question.question_id, top, retriever=name),
                k=FINAL_K,
            )
            rows.append({"qid": question.question_id, "retriever": name, "scoring": scoring,
                         "wants_figures": query.wants_figures,
                         "supported_by": "tables" if kinds == {"table"} else
                                         "prose" if "table" not in kinds else "both",
                         "recall": metrics["recall"], "ndcg": round(metrics["ndcg"], 4),
                         "mrr": round(metrics["mrr"], 4),
                         "tables": sum(passage.content_type == "table" for passage in top)})
        if number % 250 == 0:
            print(f"  {number:,} of {len(questions):,}", flush=True)
    return rows


def summarize_generated(table):
    """One row per retriever and scoring: the metrics at FINAL_K, and how many
    questions had a supporting chunk in the top FINAL_K, split by support."""
    out = {}
    for (name, scoring), rows in table.groupby(["retriever", "scoring"], sort=False):
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
        out[f"{name} / {scoring}"] = row
    return pd.DataFrame(out).T


def main() -> None:
    bm25, dense = BM25Retriever.load(), DenseRetriever.load()
    keyword = {EVERY_WORD: scoring_every_word(bm25), WITHOUT_COMMON: bm25}
    kind_of = {chunk["chunk_id"]: chunk["content_type"] for chunk in bm25.chunks}
    print(f"in more than half of the {len(bm25.chunks):,} passages, and not scored: "
          f"{', '.join(sorted(bm25._common))}", flush=True)

    written = load_written()
    generated = sample_generated(load_generated(DEFAULT_GENERATED_QUESTIONS_PATH))
    check_replay_is_exact(
        keyword, dense,
        [parse_question(q["question"]).to_query(top_k=CANDIDATE_K) for q in written[:6]]
        + [generated_query(question) for question in generated[:6]],
    )

    print(f"hand-written: {len(written)} questions", flush=True)
    written_table = pd.DataFrame(written_rows(keyword, dense, written))
    print(f"XBRL benchmark: {len(generated):,} questions, {PER_FILING} from each filing",
          flush=True)
    generated_table = pd.DataFrame(generated_rows(keyword, dense, generated, kind_of))

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


if __name__ == "__main__":
    # Re-print a finished run's summaries without searching again.
    if "--report" in sys.argv:
        report(pd.read_csv(WRITTEN_CSV, encoding="utf-8-sig"),
               pd.read_csv(XBRL_CSV, encoding="utf-8-sig"))
    else:
        main()
