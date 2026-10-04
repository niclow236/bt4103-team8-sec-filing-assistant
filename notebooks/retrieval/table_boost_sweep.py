"""What to set TABLE_BOOST to, now that the XBRL benchmark can measure it.

``TABLE_BOOST`` multiplies a table passage's score for a question that asks for
a figure, before the cut to k. It stayed at 1.0, which is off, from the day #22
added the knob, on purpose: boosting tables always looks better on the few
numeric questions somebody happens to try, so the value was left for the
mechanical XBRL benchmark (#24) to set. This is that measurement, and
``retrieval/constants.py`` records what it chose.

Two question sets, because they fail differently, as in ``final_k_sweep.py``:

- The 48 hand-written questions in ``notebooks/test_data/test_questions.csv``,
  with the checks ``search_text_comparison.py`` uses: whether the expected
  figure reached the top FINAL_K and where it ranked, how much of a prose
  answer's wording that top held, and whether the right Item was in it.
- The XBRL benchmark, PER_FILING questions drawn from each filing with a fixed
  seed, scored with the retrieval metrics (#25) at FINAL_K. Its supporting
  chunks are the passages that print the figure, and about one question in
  thirteen is supported only by prose, so the summary is also split by what
  supports the question: a lean toward tables is expected to cost those, and
  this says how much.

The boost is applied as the shipped code applies it: to a question the parser
reads as asking for a figure, and to no other. Two prose questions among the
48 are read that way ("What hardware products generated revenue for Google"),
so the prose column moves if the boost hurts them.

Each question's first-stage scores are collected once and every boost is
replayed over them, rather than searching again per boost. The replay is the
project's own code: ``base.reorder`` applies the boost and the cut exactly as
``base.rank`` does inside a retriever, and the real ``HybridRetriever`` fuses
the two replayed lists. ``check_replay_is_exact`` compares it with real
searches at every boost before the sweep is trusted.

Rows land in ``notebooks/retrieval/results/table_boost_sweep.csv`` (the 48, one
row per question, retriever and boost) and ``table_boost_sweep_xbrl.csv`` (the
benchmark, one row per question and retriever), and the summaries print.

    python -m src.retrieval benchmark          # writes benchmark/generated.jsonl
    python notebooks/retrieval/table_boost_sweep.py
"""

from __future__ import annotations

import random
import sys
from collections import defaultdict
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
from src.retrieval.base import reorder, resolve_k  # noqa: E402
from src.retrieval.bm25 import BM25Retriever  # noqa: E402
from src.retrieval.constants import BM25, CANDIDATE_K, DENSE, FINAL_K, HYBRID  # noqa: E402
from src.retrieval.dense import DenseRetriever  # noqa: E402
from src.retrieval.hybrid import HybridRetriever  # noqa: E402

RESULTS_DIR = ROOT / "notebooks" / "retrieval" / "results"
WRITTEN_CSV = RESULTS_DIR / "table_boost_sweep.csv"
XBRL_CSV = RESULTS_DIR / "table_boost_sweep_xbrl.csv"

# 1.0 is the boost off, and the row the others are read against. The steps are
# fine near 1.0 because that is where the curve bends: dense scores sit in a
# narrow band, about 0.45 for an off-topic query's best passage and 0.68 to
# 0.74 for an answerable one's (retrieval/constants.py), so a multiplier of 1.2
# already carries a loosely related table past the best prose passage.
BOOSTS = (1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.5, 2.0)

# The benchmark holds a question for every figure a filing prints, far more
# than a sweep needs. The same number from each filing, so that a filing with
# many tagged facts does not outweigh one with few, drawn with a fixed seed so
# that a re-run measures the same questions.
PER_FILING = 20
SEED = 4103


class Replayed:
    """A first-stage retriever that returns one question's collected passages.

    ``search`` applies the query's table boost and the cut to k through
    ``base.reorder``, which orders, boosts and numbers passages exactly as
    ``base.rank`` does for a real retriever, so the list it returns for a boost
    is the list the real retriever returns for it. ``kept`` is what the real
    retriever would have had in hand: every admitted passage for BM25, which
    scores them all, and the top CANDIDATE_K tables and top CANDIDATE_K others
    for dense, which is the union ``DenseRetriever.search`` builds for a boost.
    """

    def __init__(self, name, kept):
        self.name = name
        self.kept = kept

    def search(self, query, k=None):
        return reorder(((passage, passage.score) for passage in self.kept),
                       retriever=self.name, k=resolve_k(query, k), table_boost=query.table_boost)


def first_stage(bm25, dense, query):
    """One question's BM25 and dense passages, unboosted, as replayable retrievers."""
    plain = replace(query, table_boost=1.0)
    sparse = bm25.search(plain, k=len(bm25.chunks))
    tables = dense.search(replace(plain, content_type="table"), k=CANDIDATE_K)
    others = dense.search(replace(plain, content_type="prose"), k=CANDIDATE_K)
    return Replayed(BM25, sparse), Replayed(DENSE, tables + others)


def searches(bm25, dense, query):
    """The top CANDIDATE_K at every boost, for BM25 alone and for hybrid.

    Yields ``(retriever name, boost, passages)``. A question that does not ask
    for a figure is searched without the boost at every setting, since that is
    what ``ParsedQuestion.to_query`` gives it.
    """
    sparse, vectors = first_stage(bm25, dense, query)
    fused = HybridRetriever(sparse, vectors)
    for boost in BOOSTS:
        boosted = at_boost(query, boost)
        yield BM25, boost, sparse.search(boosted)
        yield HYBRID, boost, fused.search(boosted)


def at_boost(query, boost):
    """The query as the shipped code would send it were TABLE_BOOST this value."""
    return replace(query, table_boost=boost if query.wants_figures else 1.0, top_k=CANDIDATE_K)


def check_replay_is_exact(bm25, dense, queries):
    """The replay must return what the real retrievers return, at every boost.

    The sweep rests on it: if replaying a boost over collected scores gave a
    different list from searching with that boost, every row below would
    describe a retriever nobody runs.
    """
    real = {BM25: bm25, HYBRID: HybridRetriever(bm25, dense)}
    for query in queries:
        for name, boost, replayed in searches(bm25, dense, query):
            searched = real[name].search(at_boost(query, boost))
            if [p.chunk_id for p in replayed] != [p.chunk_id for p in searched]:
                raise SystemExit(
                    f"replay is not exact for {name} at boost {boost} on {query.text!r}: "
                    "the sweep would be measuring something other than the retrievers"
                )
    print(f"replay check: matches a real search at each of {BOOSTS} for BM25 and hybrid, "
          f"on {len(queries)} questions", flush=True)


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


def generated_query(question):
    """The harness's own query for a benchmark question: parsed, then the
    benchmark's ticker and year over what the parser read."""
    query = parse_question(question.question).to_query(top_k=CANDIDATE_K)
    if question.ticker is not None:
        query = replace(query, tickers=(question.ticker,))
    if question.fiscal_year is not None:
        query = replace(query, fiscal_years=(question.fiscal_year,))
    return query


def written_rows(bm25, dense, questions):
    rows = []
    for q in questions:
        query = parse_question(q["question"]).to_query(top_k=CANDIDATE_K)
        for name, boost, passages in searches(bm25, dense, query):
            rows.append({"qid": q["qid"], "retriever": name, "text": f"boost={boost}",
                         "wants_figures": query.wants_figures, **measure(q, passages)})
    return rows


def supported_by(question, kind_of):
    """Whether a benchmark question's supporting chunks are tables, prose or both."""
    kinds = {kind_of[chunk_id] for chunk_id in question.supporting_chunk_ids}
    return "tables" if kinds == {"table"} else "prose" if "table" not in kinds else "both"


def at_final_k(question, name, passages):
    """A benchmark question's metrics over the top FINAL_K of one search, and
    how many of those passages are tables."""
    top = passages[:FINAL_K]
    metrics = score_question(
        question,
        RunResult.from_passages(question.question_id, top, retriever=name),
        k=FINAL_K,
    )
    return {"recall": metrics["recall"], "ndcg": round(metrics["ndcg"], 4),
            "mrr": round(metrics["mrr"], 4),
            "tables": sum(passage.content_type == "table" for passage in top)}


def generated_rows(bm25, dense, questions, kind_of):
    """One row per question and retriever, with the metrics at each boost side by side."""
    rows = []
    for number, question in enumerate(questions, start=1):
        query = generated_query(question)
        by_retriever = defaultdict(dict)
        for name, boost, passages in searches(bm25, dense, query):
            by_retriever[name].update({
                f"{metric}@{boost}": value
                for metric, value in at_final_k(question, name, passages).items()})
        for name, metrics in by_retriever.items():
            rows.append({"qid": question.question_id, "retriever": name,
                         "wants_figures": query.wants_figures,
                         "supported_by": supported_by(question, kind_of),
                         **metrics})
        if number % 250 == 0:
            print(f"  {number:,} of {len(questions):,}", flush=True)
    return rows


def summarize_generated(table):
    """One row per retriever and boost: the metrics at FINAL_K, and the share of
    questions whose supporting chunk reached the top FINAL_K, split by support."""
    out = {}
    for name, rows in table.groupby("retriever", sort=False):
        for boost in BOOSTS:
            hit = rows[f"recall@{boost}"] > 0
            row = {
                f"Supporting chunk in top {FINAL_K}": round(hit.mean(), 3),
                "Recall, mean": round(rows[f"recall@{boost}"].mean(), 3),
                "nDCG, mean": round(rows[f"ndcg@{boost}"].mean(), 3),
                "Reciprocal rank, mean": round(rows[f"mrr@{boost}"].mean(), 3),
                # What the lean does to the prompt: how many of its passages
                # are tables.
                f"Tables among the top {FINAL_K}, mean": round(rows[f"tables@{boost}"].mean(), 1),
            }
            for support in ("tables", "both", "prose"):
                mine = rows.supported_by == support
                row[f"  supported by {support} ({int(mine.sum()):,})"] = round(hit[mine].mean(), 3)
            out[f"{name} / boost={boost}"] = row
    return pd.DataFrame(out).T


def main() -> None:
    bm25, dense = BM25Retriever.load(), DenseRetriever.load()
    kind_of = {chunk["chunk_id"]: chunk["content_type"] for chunk in bm25.chunks}

    written = load_written()
    generated = sample_generated(load_generated(DEFAULT_GENERATED_QUESTIONS_PATH))
    check_replay_is_exact(
        bm25, dense,
        [parse_question(q["question"]).to_query(top_k=CANDIDATE_K) for q in written[:6]]
        + [generated_query(question) for question in generated[:6]],
    )

    print(f"hand-written: {len(written)} questions", flush=True)
    written_table = pd.DataFrame(written_rows(bm25, dense, written))
    print(f"XBRL benchmark: {len(generated):,} questions, {PER_FILING} from each filing",
          flush=True)
    generated_table = pd.DataFrame(generated_rows(bm25, dense, generated, kind_of))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    written_table.to_csv(WRITTEN_CSV, index=False, encoding="utf-8-sig")
    generated_table.to_csv(XBRL_CSV, index=False, encoding="utf-8-sig")
    print("\nSaved", WRITTEN_CSV.relative_to(ROOT), "and", XBRL_CSV.relative_to(ROOT))
    report(written_table, generated_table)


def report(written_table, generated_table, summarize_benchmark=None) -> None:
    """Print the two summaries for a finished run.

    ``summarize_benchmark`` sums up the benchmark table. It is this script's
    own unless another is given: ``common_words_comparison.py`` prints through
    here with its own, since its table has a row per scoring where this one
    has a column per boost.
    """
    summarize_benchmark = summarize_benchmark or summarize_generated
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print("\n--- 48 hand-written questions ---")
        print(summarize(written_table).T.to_string())
        print(f"\n--- XBRL benchmark (#24), scored with #25's metrics at {FINAL_K} ---")
        print(summarize_benchmark(generated_table).to_string())


if __name__ == "__main__":
    # Re-print a finished run's summaries without searching again.
    if "--report" in sys.argv:
        report(pd.read_csv(WRITTEN_CSV, encoding="utf-8-sig"),
               pd.read_csv(XBRL_CSV, encoding="utf-8-sig"))
    else:
        main()
