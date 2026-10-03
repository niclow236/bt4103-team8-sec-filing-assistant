"""The plain question for every filing and every figure the facts route supports.

"What was <Company>'s <metric> in fiscal year <year>?", for each of the 75
filings and each line item in ``FINANCIAL_METRICS``, asked through the app's
answer path and graded against the XBRL store. The 48 hand-written questions
cover eight companies and most are answered right, so they cannot show much
more. This covers all fifteen, and it asks the most common kind of question
there is: a headline figure.

What it separates is the two ways such a question is answered:

- from the facts store (#34), with the figure looked up and a passage that
  prints it cited beside it. That answer is exact and takes no model, but the
  route gives way whenever it cannot find the figure or a passage to cite.
- by a model, from retrieved passages, where the route gave way.

Without ``--provider`` no model is asked: the run says how many questions the
route answers and how many it leaves, which needs neither a key nor a
network, and is the same on every run. With one, the questions the route
leaves are asked of the model as well, and graded: right where the answer
states the store's figure, rounded where it states it to fewer digits ("$46.8
billion" for 46,752 million), wrong where it states neither, or abstained.

The expected figure is the store's, read by the route's own lookup, so a
question is graded only where the store holds one figure for it. Where it
holds none (a software company has no inventories) or two that disagree, the
question is still asked and is reported as having no store figure.

A filer words a line one way and a question another: Amazon's statement says
"net sales" and a question says "revenue". ``--every-name`` asks each
question once for every name the line item has in ``FINANCIAL_METRICS``,
instead of the first name only, which says whether the route copes with a
question in words the filing does not use.

Each answer is also put through ``verify_answer`` as the app puts it, and the
row records what the answer card would say of it: a mismatch, support, or
neither. Since the grade says whether the answer is right, the two together
say how often the checker marks a right answer a mismatch and how often it
catches a wrong one.

    python notebooks/answers/headline_figures.py --label facts-only
    python notebooks/answers/headline_figures.py --every-name --label every-name
    python notebooks/answers/headline_figures.py --provider mistral --label hybrid

Rows land in ``notebooks/answers/results/headline-<label>.csv``, one per
question, and the summaries print. ``--report`` prints a finished run's again.
"""

from __future__ import annotations

import argparse
import re
import sys
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

from app_path_accuracy import (  # noqa: E402
    RESULTS_DIR,
    load_retriever,
    paced,
    read_as_the_app_does,
    run_and_save,
)
from src.config import read_tickers  # noqa: E402
from src.evaluation.harness import answer_with_retries  # noqa: E402
from src.pipeline.constants import DEFAULT_FISCAL_YEARS  # noqa: E402
from src.rag import (  # noqa: E402
    ProviderUnavailable,
    answer_question,
    chat_model,
    config_from_env,
    verify_answer,
    worst_check,
)
from src.rag.constants import (  # noqa: E402
    COMPANY_ALIASES,
    FACTS_PROVIDER,
    FINANCIAL_METRICS,
    PROVIDERS,
)
from src.rag.numeric import lookup_fact  # noqa: E402
from src.retrieval.constants import BM25, FINAL_K, HYBRID  # noqa: E402
from src.retrieval.facts import load_facts  # noqa: E402

QUESTION = "What was {company}'s {metric} in fiscal year {year}?"

# A dollar amount as an answer writes one: "$391,035 million", "$46.8 billion",
# "$391,035,000,000", or a figure with its scale and no dollar sign.
AMOUNT = re.compile(
    r"(\$)?\s?(\d[\d,]*(?:\.(\d+))?)(?:\s*(trillion|billion|million|thousand)\b)?",
    re.IGNORECASE,
)
SCALES = {"trillion": 10**12, "billion": 10**9, "million": 10**6, "thousand": 10**3}
# A figure with no scale word is read at each scale a statement prints in,
# since a model copies "391,035" from a table headed "in millions" as it stands.
UNSTATED_SCALES = (1, 10**3, 10**6)
# The coarsest rounding that still counts as stating the figure: a last digit
# worth no more than a hundredth of it. "$46.8 billion" for 46,752 million is
# the figure rounded; "$47 billion" is not specific enough to check.
ROUNDING_LIMIT = Decimal("0.01")

OUTCOMES = ("from the facts store", "model: right", "model: right, rounded",
            "model: wrong figure or none", "model: abstained", "left to a model")
# What the checker says of an answer, in the order the summary prints them. A
# run recorded before "unchecked" was told apart has those answers under
# "unverified".
CHECKER = ("supported", "unverified", "unchecked", "mismatch")


class NotAsked(Exception):
    """Raised in place of a model's answer when the run asks no model."""


class NoModel:
    """Stands where the chat model would, and refuses: what a question reaches
    when the facts route gives way and the run has no provider."""

    def stream(self, *args, **kwargs):
        raise NotAsked


def states(text: str, expected: Decimal) -> str | None:
    """Whether a text states the figure: "exact", "rounded", or None.

    Sign-blind, since "a net loss of $2,722 million" states -2,722 million. A
    rounded figure is compared at the precision it shows, as ``verify_answer``
    compares one. Markdown emphasis is read through: a model that writes
    "**683.9** million" has stated 683.9 million.
    """
    target, best = abs(expected), None
    for dollar, number, decimals, unit in AMOUNT.findall(text.replace("*", "")):
        unit = unit.lower()
        if not (dollar or unit):
            continue
        value = Decimal(number.replace(",", ""))
        for scale in (SCALES[unit],) if unit else UNSTATED_SCALES:
            stated, step = value * scale, Decimal(scale) / 10 ** len(decimals)
            if stated == target:
                return "exact"
            if abs(stated - target) <= step / 2 and step <= target * ROUNDING_LIMIT:
                best = "rounded"
    return best


def questions(frame, every_name: bool = False) -> list[dict]:
    """One question per filing and line item, with the store's figure for it.

    With ``every_name``, one per name the line item has rather than its first
    name only.
    """
    rows = []
    first, last = DEFAULT_FISCAL_YEARS
    for ticker in sorted(read_tickers()):
        company = COMPANY_ALIASES[ticker][0].title()
        for year in range(first, last + 1):
            for key, metric in FINANCIAL_METRICS.items():
                names = metric.aliases if every_name else metric.aliases[:1]
                for number, name in enumerate(names):
                    question = QUESTION.format(company=company, metric=name, year=year)
                    fact = lookup_fact(question, (ticker,), (year,), frame=frame)
                    suffix = f"-{number}" if every_name else ""
                    rows.append({"id": f"{ticker}-{year}-{key}{suffix}", "ticker": ticker,
                                 "fiscal_year": year, "metric": key, "question": question,
                                 "expected": None if fact is None else str(fact.value)})
    return rows


def ask(question: str, retriever, config, llm, last_request: list[float]):
    """The answer by the app's path, or None where it was left to a model that
    this run does not ask."""
    parsed = read_as_the_app_does(question)

    def once():
        if not isinstance(llm, NoModel):
            paced(last_request)
        try:
            return answer_question(question, retriever, config, llm=llm,
                                   query=parsed.to_query(top_k=FINAL_K), parsed=parsed)
        except NotAsked:
            return None

    # Asked again where the provider was busy, on the evaluation harness's terms.
    return answer_with_retries(question, once)[0]


def outcome_of(answer, expected: str | None) -> str:
    if answer is None:
        return "left to a model"
    if answer.config.provider == FACTS_PROVIDER:
        return "from the facts store"
    if answer.abstained:
        return "model: abstained"
    stated = None if expected is None else states(answer.text, Decimal(expected))
    return {"exact": "model: right", "rounded": "model: right, rounded"}.get(
        stated, "model: wrong figure or none")


def checker_says(answer, question: str) -> str:
    """What the app's checker makes of an answer, in one word.

    "mismatch" where a check contradicts any figure the answer states.
    "unverified" where none does and some figure could not be checked.
    "supported" where a passage or the facts store supports every figure, and
    "unchecked" where the answer states no figure to check. The words and the
    rule are the evaluation harness's (``verify.worst_check``), which counts a
    run's answers by them. Blank for an abstention and for a question no model
    was asked, which have nothing to check.
    """
    if answer is None or answer.abstained:
        return ""
    checked = verify_answer(answer, parsed=read_as_the_app_does(question))
    return worst_check(checked.verification.checks)


def with_no_store_figure(table: pd.DataFrame) -> pd.DataFrame:
    """A question the store holds no single figure for is counted apart, since
    nothing it is answered with can be graded."""
    return table.assign(outcome=table.outcome.where(
        table.expected.notna() | (table.outcome == "from the facts store"), "no store figure"))


def summarize(table: pd.DataFrame) -> pd.DataFrame:
    """Questions by line item and outcome."""
    table = with_no_store_figure(table)
    columns = [name for name in (*OUTCOMES, "no store figure") if (table.outcome == name).any()]
    counts = pd.crosstab(table.metric, table.outcome).reindex(
        index=list(FINANCIAL_METRICS), columns=columns, fill_value=0)
    counts["questions"] = counts.sum(axis=1)
    counts.loc["all"] = counts.sum()
    return counts


def summarize_checker(table: pd.DataFrame) -> pd.DataFrame | None:
    """Answers by outcome and by what the checker says of them, or None for a
    run recorded before the checker was."""
    if "checker" not in table.columns:
        return None
    checked = with_no_store_figure(table)
    checked = checked[checked.checker.fillna("") != ""]
    if checked.empty:
        return None
    return pd.crosstab(checked.outcome, checked.checker).reindex(
        index=[name for name in (*OUTCOMES, "no store figure") if (checked.outcome == name).any()],
        columns=[name for name in CHECKER if (checked.checker == name).any()], fill_value=0)


def report(table: pd.DataFrame) -> None:
    print(summarize(table).to_string())
    checker = summarize_checker(table)
    if checker is not None:
        print("\nWhat the checker says of the answers:")
        print(checker.to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--retriever", choices=(BM25, HYBRID), default=HYBRID,
                        help="the app's Retrieval method (default: %(default)s)")
    parser.add_argument("--provider", choices=PROVIDERS, default=None,
                        help="ask this provider what the facts route leaves; default: ask no model")
    parser.add_argument("--model", default=None,
                        help="one of that provider's models; default its own")
    parser.add_argument("--every-name", action="store_true",
                        help="ask with every name a line item has, not its first name only")
    parser.add_argument("--label", required=True,
                        help="names the results file: results/headline-<label>.csv")
    parser.add_argument("--report", action="store_true",
                        help="print a finished run's summary again, without asking anything")
    args = parser.parse_args()
    results_csv = RESULTS_DIR / f"headline-{args.label}.csv"
    if args.report:
        report(pd.read_csv(results_csv, encoding="utf-8-sig",
                           dtype={"expected": str, "checker": str}))
        return

    # Built before the indexes load, so a mistyped LLM_PROVIDER or a missing
    # MISTRAL_API_KEY stops the run at once, as python -m src.evaluation does.
    try:
        config = config_from_env(model=args.model, provider=args.provider)
        llm = chat_model(config) if args.provider else NoModel()
    except (ValueError, ProviderUnavailable) as error:
        parser.error(str(error))
    retriever = load_retriever(args.retriever)
    asked = questions(load_facts(), every_name=args.every_name)
    model = f"asking {config.provider} {config.model}" if args.provider else "no model asked"
    print(f"{len(asked)} questions | retriever {retriever.name} | {model} | FINAL_K {FINAL_K}",
          flush=True)

    def answer_all(rows: list[dict], last_request: list[float]) -> None:
        for number, q in enumerate(asked, start=1):
            answer = ask(q["question"], retriever, config, llm, last_request)
            rows.append({**q, "outcome": outcome_of(answer, q["expected"]),
                         "checker": checker_says(answer, q["question"]),
                         "answer": "" if answer is None else answer.text})
            if number % 55 == 0:
                print(f"  {number:,} of {len(asked):,}", flush=True)

    run_and_save(results_csv, answer_all, report)


if __name__ == "__main__":
    main()
