"""The 48 test questions through the app's own answer path, checked automatically.

``notebooks/mistral/mistral_generation_test.ipynb`` calls retrieval and the
model itself, so its numbers leave out what ``answer_question`` does before a
model is asked: the facts route (#34), which answers a figure question from
the XBRL store, and the split of a multi-filing question (#35). This runs the
path the app runs, so a change to retrieval, routing or the prompt can be
read as a change in what a user is shown:

    parsed = parse_question(question)                       # as src/app/app.py reads it
    answer_question(question, retriever, config,
                    query=parsed.to_query(top_k=FINAL_K), parsed=parsed)

The checks are the Mistral notebook's, so the two line up:

- figure questions (28): whether the answer states every expected figure,
  within rounding ("$49.6 billion" for "$49,584 million") and to the exact
  figure; whether it abstained; and, where it answered without the figure,
  that it stated a wrong one or none. A figure the facts route writes in full
  ("$64,115,000,000") is the same figure as "$64,115 million".
- prose questions (20): the share of the expected answer's items the answer
  mentions.
- both: whether the passages the answer was built from held the figure or the
  items at all, which is what separates a retrieval miss from a model's.

They are proxies. The figure check matches numbers and the prose check matches
words; neither judges an answer in context.

A hosted model does not repeat itself. Mistral's API returned three different
answers to one prompt at temperature 0, with or without a seed, and for a
question whose figure was not among its sources it abstained in some and
stated a wrong figure in others. So one run cannot show a small difference:
``--runs`` asks every question that many times, and the summary gives each
run beside the others. Retrieval is the same in every run, and so is an
answer from the facts route.

    python notebooks/answers/app_path_accuracy.py --provider mistral --label hybrid

Each run of 48 takes about three minutes on Mistral's free plan. Rows land in
``notebooks/answers/results/<label>.csv``, one per question and run, with the
answer's text, and the summary prints. ``--report`` prints a finished run's
summary again.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notebooks" / "retrieval"))

import pandas as pd  # noqa: E402

from search_text_comparison import (  # noqa: E402
    figure_in_text,
    figures,
    load_questions,
    terms_mentioned,
)
from src.evaluation.harness import answer_with_retries  # noqa: E402
from src.rag import (  # noqa: E402
    ProviderUnavailable,
    answer_question,
    chat_model,
    config_from_env,
    parse_question,
)
from src.rag.constants import FACTS_PROVIDER, PROVIDERS  # noqa: E402
from src.retrieval.bm25 import BM25Retriever  # noqa: E402
from src.retrieval.constants import BM25, FINAL_K, HYBRID  # noqa: E402

RESULTS_DIR = ROOT / "notebooks" / "answers" / "results"

# Between requests, for a hosted free plan's per-second limit, as the Mistral
# notebook waits. Not counted in an answer's time.
MIN_INTERVAL_S = 1.1

# A dollar amount or a percentage as an answer writes one: "$49,584 million",
# "$49.6 billion", "$64,115,000,000", "17.6%", "18 percent".
AMOUNT = re.compile(
    r"(\$?)\s?(\d[\d,]*(?:\.\d+)?)\s*(trillion|billion|million|thousand|%|percent)?",
    re.IGNORECASE,
)
TO_MILLIONS = {"trillion": 1e6, "billion": 1e3, "million": 1.0, "thousand": 1e-3}
# How far a stated figure may sit from the expected one and still be it
# rounded: a thousandth of a dollar amount, 0.05 of a percentage point. The
# Mistral notebook's tolerances.
DOLLAR_TOLERANCE = 0.001
PERCENT_TOLERANCE = 0.05


def amounts(text: str) -> list[tuple[str, tuple[float, ...]]]:
    """The dollar amounts, in millions, and the percentages a text states.

    A dollar amount with no scale after it is read two ways, since both are
    written: a statement's own "391,035", which is in millions, and a figure
    written out in full, "$391,035,000,000", as the facts route writes one.
    Markdown emphasis is read through, so "**683.9** million" keeps its scale.
    """
    found = []
    for dollar, number, unit in AMOUNT.findall(text.replace("*", "")):
        value, unit = float(number.replace(",", "")), unit.lower()
        if unit in ("%", "percent"):
            found.append(("%", (value,)))
        elif unit in TO_MILLIONS:
            found.append(("$", (value * TO_MILLIONS[unit],)))
        elif dollar:
            found.append(("$", (value, value / 1e6)))
    return found


def figures_stated(expected: str, answer: str) -> tuple[float | None, float | None]:
    """The share of the expected figures an answer states: within rounding, and exactly.

    ``(None, None)`` where the expected answer holds no figure.
    """
    wanted = amounts(expected)
    if not wanted:
        return None, None
    given = amounts(answer)
    close = exact = 0
    for kind, (value, *_) in wanted:
        stated = [v for k, values in given if k == kind for v in values]
        tolerance = PERCENT_TOLERANCE if kind == "%" else abs(value) * DOLLAR_TOLERANCE
        close += any(abs(v - value) <= tolerance for v in stated)
        exact += any(abs(v - value) <= abs(value) * 1e-9 for v in stated)
    return close / len(wanted), exact / len(wanted)


def load_retriever(name: str):
    bm25 = BM25Retriever.load()
    if name == BM25:
        return bm25
    from src.retrieval.dense import DenseRetriever
    from src.retrieval.hybrid import HybridRetriever

    return HybridRetriever(bm25, DenseRetriever.load())


def read_as_the_app_does(question: str):
    """The question as ``src/app/app.py`` reads it, so that the route it takes,
    the lean toward tables and the checks are the ones a user gets. One place,
    for the three scripts here to follow the app from."""
    return parse_question(question)


def paced(last_request: list[float]) -> float:
    """Wait out what is left of ``MIN_INTERVAL_S`` since the last request, and
    return the time this one starts, which becomes the last."""
    gap = MIN_INTERVAL_S - (time.perf_counter() - last_request[0])
    if gap > 0:
        time.sleep(gap)
    last_request[0] = time.perf_counter()
    return last_request[0]


def ask(question: str, retriever, config, llm, last_request: list[float], **options):
    """One answer by the app's path and the seconds it took, asked again where
    the provider was busy by the evaluation harness's own retry
    (``answer_with_retries``). ``options`` go to ``answer_question``, for a
    script that measures the path with one of its switches off."""
    parsed = read_as_the_app_does(question)

    def once():
        started = paced(last_request)
        answer = answer_question(question, retriever, config, llm=llm,
                                 query=parsed.to_query(top_k=FINAL_K), parsed=parsed,
                                 **options)
        return answer, time.perf_counter() - started

    return answer_with_retries(question, once)[0]


def run_and_save(results_csv: Path, answer_all, show) -> None:
    """Run ``answer_all(rows, last_request)``, which adds a row to ``rows`` for
    each answer, save the rows to ``results_csv`` and ``show`` them as a table.

    The answers before a provider failure or Ctrl-C are kept: a rate limit at
    question 40 of a third run should not cost the two runs before it. The
    rows are saved and shown as for a finished run, and the script then exits
    saying where it stopped and where they went. The three scripts here end
    this way.

    A stopped run is saved beside ``results_csv``, as ``<name>.partial.csv``,
    and never over it: ``results/`` is not in git, so a finished file of the
    same label is the only copy, and four rows of a rerun stopped at question
    5 would have replaced it.
    """
    rows, last_request, stopped = [], [0.0], None
    try:
        answer_all(rows, last_request)
    except (ProviderUnavailable, KeyboardInterrupt) as error:
        stopped = error
    saved = results_csv if stopped is None else results_csv.with_suffix(".partial.csv")
    if rows:
        table = pd.DataFrame(rows)
        saved.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(saved, index=False, encoding="utf-8-sig")
        print("\nSaved", saved.relative_to(ROOT) if saved.is_relative_to(ROOT) else saved)
        show(table)
    if stopped is not None:
        where = f", saved as {saved.name}" if rows else ""
        sys.exit(f"stopped after {len(rows)} answers{where}: "
                 f"{type(stopped).__name__}: {stopped}")


def outcome_of(row: dict) -> str:
    if row["abstained"]:
        return "abstained"
    if not row["figure_question"]:
        return f"terms {row['terms_in_answer']:.2f}"
    if row["figure_stated"] == 1.0:
        return "right" if row["figure_exact"] == 1.0 else "right, rounded"
    return "partly right" if row["figure_stated"] else "wrong or no figure"


def run(questions, retriever, config, llm, run_number: int, last_request: list[float],
        rows: list[dict]) -> None:
    """Ask every question once, adding a row to ``rows`` as each is answered, so
    the answers before a failure are still there to save."""
    for q in questions:
        answer, seconds = ask(q["question"], retriever, config, llm, last_request)
        evidence = " ".join(p.text for p in answer.passages)
        stated, exact = figures_stated(q["expected"], answer.text)
        row = {
            "run": run_number, "qid": q["qid"], "figure_question": bool(figures(q["expected"])),
            # The provider that wrote the answer: "facts" where the figure was
            # looked up and no model was asked.
            "route": answer.config.provider, "model": answer.config.model,
            "abstained": answer.abstained, "abstention_reason": answer.abstention_reason,
            "figure_in_passages": figure_in_text(q["expected"], evidence) if evidence else False,
            "figure_stated": stated, "figure_exact": exact,
            "terms_in_passages": terms_mentioned(q["expected"], evidence) if evidence else None,
            "terms_in_answer": terms_mentioned(q["expected"], answer.text),
            "seconds": round(seconds, 2), "question": q["question"], "expected": q["expected"],
            "answer": answer.text,
        }
        rows.append(row)
        print(f"run {run_number} {q['qid']} | {row['route']:<7} | {seconds:5.1f} s | "
              f"{outcome_of(row)}", flush=True)


def summarize(table: pd.DataFrame) -> pd.DataFrame:
    """One column per run. The figure rows count questions, of 28; a question
    counts as right only where every expected figure was stated."""
    out = {}
    for number, rows in table.groupby("run"):
        figure, prose = rows[rows.figure_question], rows[~rows.figure_question]
        answered = figure[~figure.abstained]
        modelled = figure[figure.route != FACTS_PROVIDER]
        # An object column, so a count prints as 26 and not as 26.000 beside a mean.
        out[f"run {number}"] = pd.Series({
            f"Figures: right, rounding allowed (of {len(figure)})": int(
                (figure.figure_stated == 1).sum()),
            "Figures: right, to the exact figure": int((figure.figure_exact == 1).sum()),
            "Figures: answered with a wrong figure or none": int(
                (answered.figure_stated < 1).sum()),
            "Figures: abstained": int(figure.abstained.sum()),
            "Figures: answered from the facts store": int(
                (figure.route == FACTS_PROVIDER).sum()),
            f"Figures: in the passages, where a model answered (of {len(modelled)})": int(
                modelled.figure_in_passages.eq(True).sum()),
            f"Prose: expected terms in the answer, mean (of {len(prose)})": round(
                prose.terms_in_answer.astype(float).mean(), 3),
            "Prose: expected terms in the passages, mean": round(
                prose.terms_in_passages.astype(float).mean(), 3),
            "Prose: abstained": int(prose.abstained.sum()),
            "Seconds per answer, median": round(rows.seconds.median(), 1),
        }, dtype=object)
    return pd.DataFrame(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--retriever", choices=(BM25, HYBRID), default=HYBRID,
                        help="the app's Retrieval method (default: %(default)s)")
    parser.add_argument("--provider", choices=PROVIDERS, default=None,
                        help="which provider answers; default LLM_PROVIDER in .env")
    parser.add_argument("--model", default=None,
                        help="one of that provider's models; default its own")
    parser.add_argument("--runs", type=int, default=3,
                        help="how many times to ask every question (default: %(default)s)")
    parser.add_argument("--label", required=True,
                        help="names the results file: results/<label>.csv")
    parser.add_argument("--report", action="store_true",
                        help="print a finished run's summary again, without asking anything")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")
    results_csv = RESULTS_DIR / f"{args.label}.csv"
    if args.report:
        print(summarize(pd.read_csv(results_csv, encoding="utf-8-sig")).to_string())
        return

    # Built before the indexes load, so a mistyped LLM_PROVIDER or a missing
    # MISTRAL_API_KEY stops the run at once, as python -m src.evaluation does.
    try:
        config = config_from_env(model=args.model, provider=args.provider)
        llm = chat_model(config)
    except (ValueError, ProviderUnavailable) as error:
        parser.error(str(error))
    retriever = load_retriever(args.retriever)
    questions = load_questions()
    print(f"{len(questions)} questions, {args.runs} run(s) | retriever {retriever.name} | "
          f"{config.provider} {config.model} | FINAL_K {FINAL_K}", flush=True)

    def answer_all(rows: list[dict], last_request: list[float]) -> None:
        for number in range(1, args.runs + 1):
            run(questions, retriever, config, llm, number, last_request, rows)

    run_and_save(results_csv, answer_all, lambda table: print(summarize(table).to_string()))


if __name__ == "__main__":
    main()
