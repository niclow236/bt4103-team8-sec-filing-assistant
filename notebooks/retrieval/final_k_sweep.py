"""Where to set FINAL_K now that a hosted model removes the context limit (#85).

FINAL_K is 8 partly because of Ollama. A grounded prompt over eight passages
runs 2,700 to 3,400 tokens, Ollama's laptop default window is 4,096, and a
prompt longer than the window is cut from the front without telling the caller.
Ministral 3 14B on Mistral's API takes 256K tokens, so the cap is now a choice
to measure rather than a constraint to respect.

What the 2026-09-18 Mistral evaluation found, and what this answers: 12 of the
28 figure questions failed because the table holding the figure never reached
the prompt, and for 7 of them hybrid search had ranked that table 12th to 23rd.
A top 20 would have included 6. More passages also cost prompt tokens and can
distract the model, so the right value is measured, not guessed.

Two question sets, because they fail differently:

- The 48 hand-written questions in ``notebooks/test_data/test_questions.csv``,
  with the Mistral evaluation's own checks, so the numbers line up with
  ``docs/mistral-free-tier-evaluation.md``: whether the expected figure reached
  the prompt, how much of a prose answer's wording it held, and whether the
  right Item was in it. Small, but written by hand against real answers.
- The mechanical XBRL benchmark (#24), scored with the retrieval metrics (#25):
  Recall, nDCG and reciprocal rank over the supporting chunk. Generated rather
  than written, and large, so it says what 48 questions cannot -- whether a
  gain holds across the corpus or is a handful of questions moving.

And what each K costs, which is the other half of the decision: the size of the
prompt ``build_prompt`` actually renders over those passages.

Each question is searched once at the largest K and the ranking sliced. That is
exact rather than approximate: ``HybridRetriever`` fuses at
``max(CANDIDATE_K, k)``, which is 50 for every K here, so the fused order does
not depend on k and a shorter K is a prefix of a longer one.
``check_slicing_is_exact`` verifies that against real searches before the sweep
runs, rather than leaving it as an argument in a docstring.

Per-question rows land in ``notebooks/retrieval/results/final_k_sweep.csv`` and
the summaries print.

    python -m src.retrieval benchmark          # writes benchmark/generated.jsonl
    python notebooks/retrieval/final_k_sweep.py
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

from search_text_comparison import load_questions as load_written, measure  # noqa: E402
from src.evaluation.benchmark import (  # noqa: E402
    DEFAULT_GENERATED_QUESTIONS_PATH,
    load_questions as load_generated,
)
from src.evaluation.metrics import score_question  # noqa: E402
from src.evaluation.records import RunResult  # noqa: E402
from src.rag.constants import MAX_OUTPUT_TOKENS, NUM_CTX  # noqa: E402
from src.rag.prompt import build_prompt  # noqa: E402
from src.rag.query import parse_question  # noqa: E402
from src.retrieval.bm25 import BM25Retriever  # noqa: E402
from src.retrieval.constants import CANDIDATE_K  # noqa: E402
from src.retrieval.dense import DenseRetriever  # noqa: E402
from src.retrieval.hybrid import HybridRetriever  # noqa: E402

RESULTS_CSV = ROOT / "notebooks" / "retrieval" / "results" / "final_k_sweep.csv"

# The values #85 asks for. 8 is today's setting and the row the others are read
# against; 20 is where the Mistral trace found the tables that never arrived.
K_VALUES = (8, 12, 16, 20)

# Tokens are counted with llama3.2's own tokenizer, because the limit this has
# to clear is Ollama's NUM_CTX and a token there is whatever that tokenizer
# says it is. Meta's repository is gated, so this is the ungated mirror of the
# same tokenizer; it is a download on first use and cached afterwards. Offline,
# the run falls back to characters at the ratio below and says so, because the
# NUM_CTX conclusion turns on the difference.
TOKENIZER_NAME = "unsloth/Llama-3.2-3B-Instruct"
# The fallback ratio, measured against that tokenizer over these prompts: 4.53
# characters per token on average, 3.90 to 5.36 across questions. Rounded down,
# so the fallback overstates rather than understates a prompt. It is a fallback
# and not the measurement -- a 19% error in this ratio is the difference
# between a top 20 that clears NUM_CTX and one that does not.
CHARS_PER_TOKEN = 4.5


def token_counter():
    """How to count a prompt's tokens, and a line saying which way it counted."""
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    except Exception as error:  # offline, or the mirror moved
        print(f"tokens: {TOKENIZER_NAME} unavailable ({type(error).__name__}), "
              f"estimating at {CHARS_PER_TOKEN} characters per token", flush=True)
        return lambda text: round(len(text) / CHARS_PER_TOKEN)
    print(f"tokens: counted with {TOKENIZER_NAME}, the tokenizer llama3.2:3b uses",
          flush=True)
    return lambda text: len(tokenizer(text)["input_ids"])


def prompt_size(question, passages, count_tokens):
    """What this many passages cost, in the prompt the generator would send."""
    if not passages:
        return {"prompt_chars": 0, "prompt_tokens": 0}
    text = build_prompt(question, passages).as_text()
    return {"prompt_chars": len(text), "prompt_tokens": count_tokens(text)}


def check_slicing_is_exact(retriever, questions, n=5):
    """Searching at 20 and slicing to 8 must equal searching at 8.

    The sweep searches once per question and slices, which is sound only while
    the ranking does not depend on k. Hybrid fusion should not depend on it at
    these values, but "should not" is what this checks.
    """
    for q in questions[:n]:
        parsed = parse_question(q["question"])
        deep = retriever.search(parsed.to_query(top_k=max(K_VALUES)))
        for k in K_VALUES:
            shallow = retriever.search(parsed.to_query(top_k=k))
            if [p.chunk_id for p in shallow] != [p.chunk_id for p in deep[:k]]:
                raise SystemExit(
                    f"slicing is not exact for {q['qid']} at k={k}: searching once "
                    "per question and slicing is not valid here, so the sweep would "
                    "be comparing rankings rather than cutoffs"
                )
    print(f"slicing check: a top {max(K_VALUES)} sliced to {K_VALUES} matches a direct "
          f"search at each, on {n} questions", flush=True)


def written_rows(retriever, questions, count_tokens):
    """The 48 hand-written questions, measured the way the Mistral run measured them."""
    rows = []
    for q in questions:
        # Searched at CANDIDATE_K so figure_rank spans the same top 50 the
        # other scripts in this directory report, whatever K the row is for.
        passages = retriever.search(parse_question(q["question"]).to_query(top_k=CANDIDATE_K))
        for k in K_VALUES:
            rows.append({"question_set": "written", "qid": q["qid"], "k": k,
                         **measure(q, passages, k=k),
                         **prompt_size(q["question"], passages[:k], count_tokens)})
    return rows


def generated_rows(retriever, questions, count_tokens):
    """The XBRL benchmark (#24), scored with the retrieval metrics (#25)."""
    rows = []
    for number, question in enumerate(questions, start=1):
        # The harness's own query: parsed, then the benchmark's ticker and year
        # override what the parser read, so a generated question is searched
        # exactly as evaluate() would search it.
        query = parse_question(question.question).to_query(top_k=max(K_VALUES))
        if question.ticker is not None:
            query = replace(query, tickers=(question.ticker,))
        if question.fiscal_year is not None:
            query = replace(query, fiscal_years=(question.fiscal_year,))
        passages = retriever.search(query)
        for k in K_VALUES:
            top = passages[:k]
            metrics = score_question(
                question,
                RunResult.from_passages(question.question_id, top, retriever=retriever.name),
                k=k,
            )
            recall = metrics["recall"]
            rows.append({"question_set": "xbrl", "qid": question.question_id, "k": k,
                         "recall": recall, "ndcg": metrics["ndcg"], "mrr": metrics["mrr"],
                         "supporting_in_top_k": recall is not None and recall > 0,
                         **prompt_size(question.question, top, count_tokens)})
        if number % 250 == 0:
            print(f"  {number:,} of {len(questions):,}", flush=True)
    return rows


def summarize_written(table):
    out = {}
    for k, rows in table.groupby("k"):
        # Cast first. The two question sets share one frame, so the columns only
        # one of them fills arrive as object dtype with NaN in the other's rows,
        # and ``~`` on an object column negates bitwise: True becomes -2, which
        # pandas then looks up as a column name.
        is_figure = rows.figure_question.astype(bool)
        figure, prose = rows[is_figure], rows[~is_figure]
        out[f"top {k}"] = {
            f"Figure in the prompt (of {len(figure)})": int(figure.figure_in_top_k.eq(True).sum()),
            f"Prose: expected terms, mean (of {len(prose)})": round(
                prose.terms_in_top_k.astype(float).mean(), 3),
            f"Right section (of {len(rows)})": int(rows.section_in_top_k.astype(bool).sum()),
            "Prompt characters, median": int(rows.prompt_chars.median()),
            "Prompt tokens, median": int(rows.prompt_tokens.median()),
            "Prompt tokens, max": int(rows.prompt_tokens.max()),
        }
    return pd.DataFrame(out)


def summarize_generated(table):
    out = {}
    for k, rows in table.groupby("k"):
        out[f"top {k}"] = {
            f"Supporting chunk in the prompt (of {len(rows):,})": int(
                rows.supporting_in_top_k.astype(bool).sum()),
            "Supporting chunk in the prompt, share": round(
                rows.supporting_in_top_k.astype(bool).mean(), 3),
            "Recall, mean": round(rows.recall.astype(float).mean(), 3),
            "nDCG, mean": round(rows.ndcg.astype(float).mean(), 3),
            "Reciprocal rank, mean": round(rows.mrr.astype(float).mean(), 3),
            "Prompt characters, median": int(rows.prompt_chars.median()),
            "Prompt tokens, median": int(rows.prompt_tokens.median()),
            "Prompt tokens, max": int(rows.prompt_tokens.max()),
        }
    return pd.DataFrame(out)


def summarize_context(table):
    """How close each K comes to Ollama's window once the answer's ceiling is beside it."""
    budget = NUM_CTX - MAX_OUTPUT_TOKENS
    out = {}
    for k, rows in table.groupby("k"):
        over = rows.prompt_tokens.gt(budget)
        out[f"top {k}"] = {
            f"Prompts over NUM_CTX - MAX_OUTPUT_TOKENS ({budget:,}) (of {len(rows):,})": int(over.sum()),
            "Largest prompt as a share of the window": round(
                rows.prompt_tokens.max() / NUM_CTX, 3),
        }
    return pd.DataFrame(out)


def main():
    bm25, dense = BM25Retriever.load(), DenseRetriever.load()
    retriever = HybridRetriever(bm25, dense)

    count_tokens = token_counter()
    written = load_written()
    check_slicing_is_exact(retriever, written)

    print(f"hand-written: {len(written)} questions", flush=True)
    rows = written_rows(retriever, written, count_tokens)

    generated = load_generated(DEFAULT_GENERATED_QUESTIONS_PATH)
    print(f"XBRL benchmark: {len(generated):,} questions", flush=True)
    rows += generated_rows(retriever, generated, count_tokens)

    table = pd.DataFrame(rows)
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS_CSV, index=False, encoding="utf-8-sig")
    print("\nSaved", RESULTS_CSV.relative_to(ROOT))

    report(table)


def report(table):
    """Print the three summaries for a finished run."""
    print("\n--- 48 hand-written questions ---")
    print(summarize_written(table[table["question_set"] == "written"]).to_string())
    print("\n--- XBRL benchmark (#24), scored with #25's metrics ---")
    print(summarize_generated(table[table["question_set"] == "xbrl"]).to_string())
    print("\n--- against Ollama's context window ---")
    print(summarize_context(table).to_string())


if __name__ == "__main__":
    # Re-print a finished run's summaries without searching again.
    if "--report" in sys.argv:
        report(pd.read_csv(RESULTS_CSV, encoding="utf-8-sig"))
    else:
        main()
