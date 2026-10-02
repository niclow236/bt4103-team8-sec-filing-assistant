"""Whether refusing unanswerable questions ever refuses one that has an answer.

``answer_question`` refuses a question the parser reads as asking for advice,
a prediction or a current price, or as naming only companies the corpus holds
no filings for, without searching (``answer.REFUSALS``). That is only safe if
the parser never reads an answerable question that way, since a refused
question gets no answer at all where a searched one would have got one.

This reads every question there is an answer for and counts the ones that
would be refused. No index is loaded and no model is asked:

- the 48 hand-written questions,
- the plain question for every filing and every name of every line item the
  facts route answers (``headline_figures.py --every-name``),
- the plain question for every filing and every line item in
  ``line_item_figures.py``,
- the generated XBRL benchmark, where it has been built
  (``python -m src.retrieval benchmark``).

It then reads a short list of questions that should be refused, and of
questions that look like them and should not be, and prints how each is read.

    python notebooks/answers/refusal_check.py

With ``--provider`` it also asks a model each of those through the app's
path, with Hybrid, once with the refusal off and once with it on, and writes
what came back to ``results/refusal-probes.csv``: what a user was shown for
each before the refusal, and what they are shown with it.

    python notebooks/answers/refusal_check.py --provider mistral
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

from app_path_accuracy import (  # noqa: E402
    RESULTS_DIR,
    ask,
    load_retriever,
    read_as_the_app_does,
)
from headline_figures import questions as headline_questions  # noqa: E402
from line_item_figures import questions as line_item_questions  # noqa: E402
from src.evaluation.benchmark import DEFAULT_GENERATED_QUESTIONS_PATH  # noqa: E402
from src.pipeline.constants import DEFAULT_FISCAL_YEARS  # noqa: E402
from src.rag import ProviderUnavailable, chat_model, config_from_env  # noqa: E402
from src.rag.answer import REFUSALS  # noqa: E402
from src.rag.constants import PROVIDERS  # noqa: E402
from src.retrieval.constants import HYBRID  # noqa: E402
from src.retrieval.facts import load_facts  # noqa: E402

TEST_QUESTIONS = ROOT / "notebooks" / "test_data" / "test_questions.csv"
PROBES_CSV = RESULTS_DIR / "refusal-probes.csv"

# Questions no filing in the corpus answers, and the reason each is refused.
SHOULD_REFUSE = {
    "Should I buy Amazon stock?": "beyond_the_filings",
    "Is Meta a good investment?": "beyond_the_filings",
    "What will Microsoft's revenue be next year?": "beyond_the_filings",
    "What is Apple's current stock price?": "beyond_the_filings",
    "Based on what Apple disclosed, predict next quarter's revenue": "beyond_the_filings",
    "What was Intel's total revenue in FY2024?": "company_not_in_corpus",
    "What was Tesla's net income in FY2023?": "company_not_in_corpus",
    "How many employees did NVIDIA have in FY2024?": "company_not_in_corpus",
}
# Questions that share their words and that a filing does answer, or may.
SHOULD_SEARCH = (
    "What risks did Apple disclose about its stock price?",
    "What did Microsoft say about forecasting risk in FY2024?",
    "What does Apple expect its capital expenditures to be next year?",
    "What guidance did Cisco give for fiscal 2025 revenue?",
    "How does Apple describe competition with NVIDIA?",
    "Compare Intel and Apple revenue in FY2024",
    # Only the year is outside the corpus, and the FY2021 statements print FY2020.
    "What was Apple's total revenue in FY2020?",
)


def refusal(question: str) -> str | None:
    """The reason the app's path would refuse the question with, or None."""
    return REFUSALS.get(read_as_the_app_does(question).unanswerable_because)


def answerable() -> dict[str, list[str]]:
    """Every question there is an answer for, by where it comes from."""
    with open(TEST_QUESTIONS, encoding="utf-8-sig", newline="") as stream:
        written = [row["Question"].strip() for row in csv.DictReader(stream)
                   if row["Question"].strip()]
    frame = load_facts()
    first, last = DEFAULT_FISCAL_YEARS
    sets = {
        "48 hand-written questions": written,
        "headline questions, every name": [
            q["question"] for q in headline_questions(frame, every_name=True)],
        "line-item questions, every year": [
            q["question"] for q in line_item_questions(frame, range(first, last + 1))],
    }
    if DEFAULT_GENERATED_QUESTIONS_PATH.exists():
        with open(DEFAULT_GENERATED_QUESTIONS_PATH, encoding="utf-8") as stream:
            sets["generated XBRL benchmark"] = [
                json.loads(line)["question"] for line in stream if line.strip()]
    return sets


def ask_the_probes(config, llm) -> pd.DataFrame:
    """Each probe asked through the app's path, with the refusal off and on."""
    retriever = load_retriever(HYBRID)
    probes = [(question, "refuse") for question in SHOULD_REFUSE]
    probes += [(question, "search") for question in SHOULD_SEARCH]
    rows, last_request = [], [0.0]
    for question, expected in probes:
        row = {"question": question, "should": expected}
        for name, use_refusal in (("without", False), ("with", True)):
            answer, _ = ask(question, retriever, config, llm, last_request,
                            use_refusal=use_refusal)
            row[f"{name}: abstained"] = answer.abstained
            row[f"{name}: reason"] = answer.abstention_reason or ""
            row[f"{name}: answer"] = answer.text
        rows.append(row)
        print(f"  {len(rows)} of {len(probes)}", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--provider", choices=PROVIDERS, default=None,
                        help="also ask this provider's model each probe, refusal off and on")
    parser.add_argument("--model", default=None,
                        help="one of that provider's models; default its own")
    args = parser.parse_args()
    if args.provider:
        # Built before anything is read, so a missing key stops the run at once.
        try:
            config = config_from_env(model=args.model, provider=args.provider)
            llm = chat_model(config)
        except (ValueError, ProviderUnavailable) as error:
            parser.error(str(error))

    total = refused = 0
    for name, questions in answerable().items():
        wrongly = [(q, refusal(q)) for q in questions]
        wrongly = [(q, reason) for q, reason in wrongly if reason is not None]
        total, refused = total + len(questions), refused + len(wrongly)
        print(f"{name}: {len(questions):,} questions, {len(wrongly)} would be refused", flush=True)
        for question, reason in wrongly[:20]:
            print(f"    {reason}: {question}")
    print(f"answerable questions: {total:,}, refused: {refused}")

    print("\nquestions that should be refused:")
    for question, expected in SHOULD_REFUSE.items():
        got = refusal(question)
        print(f"  {'ok  ' if got == expected else 'MISS'} {got or 'searched':22s} {question}")
    print("questions that should be searched:")
    for question in SHOULD_SEARCH:
        got = refusal(question)
        print(f"  {'ok  ' if got is None else 'MISS'} {got or 'searched':22s} {question}")

    if args.provider:
        print(f"\nasking {config.provider} {config.model}, refusal off and on:", flush=True)
        table = ask_the_probes(config, llm)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        table.to_csv(PROBES_CSV, index=False, encoding="utf-8-sig")
        print("Saved", PROBES_CSV.relative_to(ROOT))
        for expected, rows in table.groupby("should", sort=False):
            print(f"  should {expected} ({len(rows)}): abstained without the refusal "
                  f"{int(rows['without: abstained'].sum())}, with it "
                  f"{int(rows['with: abstained'].sum())}")


if __name__ == "__main__":
    main()
