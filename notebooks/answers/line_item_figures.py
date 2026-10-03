"""The plain question for line items the facts route does not answer.

"What was Microsoft's accounts receivable in fiscal year 2024?" The facts
route answers eleven line items (``headline_figures.py`` measures those). A
filing reports many more, and a question about one of them is answered by a
model from retrieved passages. Whether the statement that prints the figure is
among those passages depends on how the question is read: only a question read
as asking for a figure is searched with the lean toward tables
(``TABLE_BOOST``), and the parser reads one that way from a list of cues.

This asks the plain question for each line item in ``LINE_ITEMS`` and each
filing of the chosen fiscal years, through the app's answer path, and grades
the answer against the XBRL store as ``headline_figures.py`` grades one: right
where it states the store's figure, rounded where it states it to fewer digits,
wrong where it states neither, or abstained. A question is asked only where
the store holds one USD value for the line item in that filing.

The store's concept is one reading of the name. "Share repurchases" is the
cash paid in the year on the cash flow statement, and a filing also reports
what was bought under its repurchase programme, which is a different figure
and a fair answer. So the grade is a floor, and the same floor for every run,
which is what a comparison between two states of the code needs.

    python notebooks/answers/line_item_figures.py --provider mistral --label before
    python notebooks/answers/line_item_figures.py --provider mistral --years 2023 2024 \
        --label two-years

Rows land in ``notebooks/answers/results/line-items-<label>.csv``, one per
question, and the summary prints. ``--report`` prints a finished run's again.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

from app_path_accuracy import (  # noqa: E402
    RESULTS_DIR,
    load_retriever,
    read_as_the_app_does,
    run_and_save,
)
from headline_figures import ask, checker_says, outcome_of  # noqa: E402
from src.config import read_tickers  # noqa: E402
from src.pipeline.constants import DEFAULT_FISCAL_YEARS  # noqa: E402
from src.rag import ProviderUnavailable, chat_model, config_from_env  # noqa: E402
from src.rag.constants import COMPANY_ALIASES, PROVIDERS  # noqa: E402
from src.retrieval.constants import BM25, FINAL_K, HYBRID  # noqa: E402
from src.retrieval.facts import current_year, load_facts  # noqa: E402

QUESTION = "What was {company}'s {name} in fiscal year {year}?"

# The name a question uses, and the XBRL concepts that carry the figure. Names
# an analyst would use for lines of the three statements, none of them one the
# facts route answers.
LINE_ITEMS = {
    "gross profit": ("GrossProfit",),
    "income before income taxes": (
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItems"
        "NoncontrollingInterest",),
    "income tax expense": ("IncomeTaxExpenseBenefit",),
    "provision for income taxes": ("IncomeTaxExpenseBenefit",),
    "interest expense": ("InterestExpense",),
    "sales and marketing expense": ("SellingAndMarketingExpense",),
    "stock-based compensation": ("ShareBasedCompensation",
                                 "AllocatedShareBasedCompensationExpense"),
    "depreciation and amortization": ("DepreciationDepletionAndAmortization",
                                      "DepreciationAndAmortization"),
    "accounts receivable": ("AccountsReceivableNetCurrent",),
    "marketable securities": ("MarketableSecuritiesCurrent",
                              "AvailableForSaleSecuritiesDebtSecuritiesCurrent"),
    "property and equipment": ("PropertyPlantAndEquipmentNet",),
    "intangible assets": ("IntangibleAssetsNetExcludingGoodwill",
                          "FiniteLivedIntangibleAssetsNet"),
    "capital expenditures": ("PaymentsToAcquirePropertyPlantAndEquipment",),
    "purchases of property and equipment": ("PaymentsToAcquirePropertyPlantAndEquipment",),
    "interest paid": ("InterestPaidNet",),
    "share repurchases": ("PaymentsForRepurchaseOfCommonStock",),
}

OUTCOMES = ("model: right", "model: right, rounded", "model: wrong figure or none",
            "model: abstained")


def questions(frame, years) -> list[dict]:
    """One question per filing and line item the store holds one USD value for."""
    current = current_year(frame)
    current = current.assign(name=current["concept"].astype(str).str.split(":").str[-1])
    rows = []
    for ticker in sorted(read_tickers()):
        company = COMPANY_ALIASES[ticker][0].title()
        for year in years:
            for name, concepts in LINE_ITEMS.items():
                held = current[(current.ticker == ticker) & (current.fiscal_year == year)
                               & current.name.isin(concepts)
                               & (current.unit.astype(str).str.upper() == "USD")]
                values = {str(value).strip() for value in held.raw_value if str(value).strip()}
                if len(values) != 1:
                    continue
                question = QUESTION.format(company=company, name=name, year=year)
                rows.append({"id": f"{ticker}-{year}-{name.replace(' ', '_')}", "ticker": ticker,
                             "fiscal_year": year, "line_item": name, "question": question,
                             "expected": values.pop()})
    return rows


def summarize(table: pd.DataFrame) -> pd.DataFrame:
    """Questions by line item and outcome, with how many were read as asking
    for a figure and how many of the sixteen passages were tables."""
    columns = [name for name in OUTCOMES if (table.outcome == name).any()]
    counts = pd.crosstab(table.line_item, table.outcome).reindex(
        index=list(LINE_ITEMS), columns=columns, fill_value=0).dropna(how="all")
    counts["questions"] = counts.sum(axis=1)
    counts["read as a figure question"] = table.groupby("line_item").read_as_figure.sum()
    counts.loc["all"] = counts.sum()
    counts["tables among the passages, mean"] = table.groupby("line_item").tables.mean().round(1)
    counts.loc["all", "tables among the passages, mean"] = round(table.tables.mean(), 1)
    return counts[counts.questions > 0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--retriever", choices=(BM25, HYBRID), default=HYBRID,
                        help="the app's Retrieval method (default: %(default)s)")
    parser.add_argument("--provider", choices=PROVIDERS, default=None,
                        help="the provider that answers; default: LLM_PROVIDER in .env")
    parser.add_argument("--model", default=None,
                        help="one of that provider's models; default its own")
    parser.add_argument("--years", type=int, nargs="+", default=[2024],
                        help="fiscal years to ask about (default: %(default)s)")
    parser.add_argument("--label", required=True,
                        help="names the results file: results/line-items-<label>.csv")
    parser.add_argument("--report", action="store_true",
                        help="print a finished run's summary again, without asking anything")
    args = parser.parse_args()
    results_csv = RESULTS_DIR / f"line-items-{args.label}.csv"
    if args.report:
        print(summarize(pd.read_csv(results_csv, encoding="utf-8-sig",
                                    dtype={"expected": str})).to_string())
        return
    first, last = DEFAULT_FISCAL_YEARS
    if any(not first <= year <= last for year in args.years):
        parser.error(f"--years must be within {first} to {last}")

    # Built before the indexes load, so a mistyped LLM_PROVIDER or a missing
    # MISTRAL_API_KEY stops the run at once, as python -m src.evaluation does.
    try:
        config = config_from_env(model=args.model, provider=args.provider)
        llm = chat_model(config)
    except (ValueError, ProviderUnavailable) as error:
        parser.error(str(error))
    retriever = load_retriever(args.retriever)
    asked = questions(load_facts(), sorted(set(args.years)))
    print(f"{len(asked)} questions | retriever {retriever.name} | asking {config.provider} "
          f"{config.model} | FINAL_K {FINAL_K}", flush=True)

    def answer_all(rows: list[dict], last_request: list[float]) -> None:
        for number, q in enumerate(asked, start=1):
            answer = ask(q["question"], retriever, config, llm, last_request)
            rows.append({**q, "outcome": outcome_of(answer, q["expected"]),
                         "read_as_figure": read_as_the_app_does(q["question"]).wants_figures,
                         "tables": sum(p.content_type == "table" for p in answer.passages),
                         "checker": checker_says(answer, q["question"]),
                         "answer": answer.text})
            if number % 10 == 0:
                print(f"  {number:,} of {len(asked):,}", flush=True)

    run_and_save(results_csv, answer_all, lambda table: print(summarize(table).to_string()))


if __name__ == "__main__":
    main()
