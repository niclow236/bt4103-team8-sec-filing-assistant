"""Search with the whole question or with the question trimmed of its filters (#87).

Runs the 48 test questions in ``notebooks/test_data/test_questions.csv``
through BM25, dense and hybrid retrieval three ways, with the same filters
each time, so the only difference is the text:

- full: every retriever searches the question as asked.
- trimmed: every retriever searches ``ParsedQuestion.search_text``.
- shipped: what ``ParsedQuestion.to_query`` builds, where BM25 searches the
  trimmed text and dense the question. For BM25 alone this is the trimmed
  row, and for dense alone the full row; it differs only for hybrid.

No model is called. The checks are the Mistral evaluation's, so the numbers
line up with ``docs/mistral-free-tier-evaluation.md``:

- figure questions: whether a passage holding every expected figure is in the
  top 8, and the rank of the first one in the top 50.
- prose questions: the share of the expected answer's items the top-8
  passages mention.
- every question: whether the top 8 include the expected Item.

Run from the project root:

    python notebooks/retrieval/search_text_comparison.py

It writes one row per question, retriever and text to
``notebooks/retrieval/results/search_text_comparison.csv`` and prints the
summary.
"""

from __future__ import annotations

import csv
import re
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.rag.query import parse_question  # noqa: E402
from src.retrieval.bm25 import BM25Retriever  # noqa: E402
from src.retrieval.constants import CANDIDATE_K, FINAL_K  # noqa: E402
from src.retrieval.dense import DenseRetriever  # noqa: E402
from src.retrieval.hybrid import HybridRetriever  # noqa: E402

QUESTIONS_CSV = ROOT / "notebooks" / "test_data" / "test_questions.csv"
RESULTS_CSV = ROOT / "notebooks" / "retrieval" / "results" / "search_text_comparison.csv"

# --- the Mistral notebook's checks, unchanged --------------------------------
FIGURE = re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?|\d[\d,]*(?:\.\d+)?\s?%")
NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
STOP = {"the", "and", "for", "its", "their", "our", "with", "from", "that", "this", "which", "other", "including",
        "related", "could", "may", "might", "would", "any", "all", "are", "was", "were", "has", "have", "been",
        "into", "such", "than", "not", "per", "of", "in", "to", "as", "on", "by", "or", "an", "at", "is", "it", "be", "we"}


def plain(number):
    number = number.replace(",", "")
    return number.rstrip("0").rstrip(".") if "." in number else number


def figures(text):
    return [plain(NUMBER.search(m).group()) for m in FIGURE.findall(text)]


def figure_in_text(expected, text):
    wanted = figures(expected)
    if not wanted:
        return None
    digits = re.sub(r"[,\s]", "", text)
    return all(re.search(r"(?<!\d)" + re.escape(w) + r"(?!\d)", digits) for w in wanted)


def words(text):
    found = set()
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        if len(word) >= 2 and word not in STOP:
            found.add(word[:-1] if len(word) > 4 and word.endswith("s") else word)
    return found


def terms_mentioned(expected, text):
    if figures(expected):
        return None
    items = [w for w in (words(part) for part in re.split(r",|;|\band\b|\bor\b", expected)) if w]
    if not items:
        return None
    have = words(text)
    return sum(len(i & have) / len(i) >= (1.0 if len(i) <= 2 else 0.5) for i in items) / len(items)


# --- the run -------------------------------------------------------------------

def load_questions():
    with open(QUESTIONS_CSV, encoding="utf-8-sig", newline="") as f:
        sheet = [r for r in csv.DictReader(f) if r["Question"].strip()]
    questions = []
    for n, r in enumerate(sheet, start=1):
        item = re.search(r"item\s+(\d+[a-z]?)", r["Item"], re.IGNORECASE)
        questions.append({
            "qid": f"Q{n:02d}",
            "question": r["Question"].strip(),
            "ticker": r["Ticker"].strip().upper(),
            "fiscal_year": int(r["Fiscal Year"]),
            "item": item.group(1).upper() if item else None,
            "expected": r["Answer"].strip(),
        })
    return questions


def first_rank(passages, test):
    return next((rank for rank, p in enumerate(passages, start=1) if test(p)), None)


def measure(q, passages):
    top = passages[:FINAL_K]
    right_item = lambda p: (p.ticker == q["ticker"] and p.fiscal_year == q["fiscal_year"]
                            and (q["item"] is None or p.item == q["item"]))
    figure_rank = (first_rank(passages, lambda p: figure_in_text(q["expected"], p.text))
                   if figures(q["expected"]) else None)
    return {
        "figure_question": bool(figures(q["expected"])),
        "figure_in_top_k": figure_in_text(q["expected"], " ".join(p.text for p in top)),
        "figure_rank": figure_rank,
        "section_in_top_k": any(right_item(p) for p in top),
        "terms_in_top_k": terms_mentioned(q["expected"], " ".join(p.text for p in top)),
    }


def main():
    questions = load_questions()
    bm25, dense = BM25Retriever.load(), DenseRetriever.load()
    retrievers = {"bm25": bm25, "dense": dense, "hybrid": HybridRetriever(bm25, dense)}

    rows = []
    for q in questions:
        parsed = parse_question(q["question"])
        shipped = parsed.to_query(top_k=CANDIDATE_K)
        variants = {
            "full": replace(shipped, keyword_text=None),
            "trimmed": replace(shipped, text=parsed.search_text, keyword_text=None),
            "shipped": shipped,
        }
        for text_kind, query in variants.items():
            for name, retriever in retrievers.items():
                passages = retriever.search(query)
                rows.append({"qid": q["qid"], "retriever": name, "text": text_kind,
                             "search_text": parsed.search_text, **measure(q, passages)})
        print(f"{q['qid']}  {parsed.search_text}", flush=True)

    table = pd.DataFrame(rows)
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS_CSV, index=False, encoding="utf-8-sig")
    print("\nSaved", RESULTS_CSV.relative_to(ROOT))
    print(summarize(table).to_string())


def summarize(table):
    out = {}
    for (name, kind), rows in table.groupby(["retriever", "text"], sort=False):
        fig, prose = rows[rows.figure_question], rows[~rows.figure_question]
        ranks = fig.figure_rank.dropna()
        out[f"{name} / {kind}"] = {
            f"Figure in top {FINAL_K} (of {len(fig)})": int(fig.figure_in_top_k.eq(True).sum()),
            f"Figure in top {CANDIDATE_K}": len(ranks),
            "Figure rank, median when found": ranks.median(),
            f"Prose: expected terms in top {FINAL_K}, mean (of {len(prose)})": round(prose.terms_in_top_k.mean(), 3),
            f"Right section in top {FINAL_K} (of {len(rows)})": int(rows.section_in_top_k.sum()),
        }
    return pd.DataFrame(out)


if __name__ == "__main__":
    main()
