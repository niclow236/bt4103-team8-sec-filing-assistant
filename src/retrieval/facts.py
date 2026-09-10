"""Store the figures a filing reported, as figures, in ``data/index/facts.parquet``.

Run it from the project root:

    python -m src.retrieval facts
    python -m src.retrieval facts --tickers AAPL MSFT
    python -m src.retrieval facts --refresh

"What was Apple's FY2024 revenue?" should not be answered by retrieving prose
and hoping the right number survives chunking, embedding and generation. The
filer already published that figure as structured XBRL, tagged with the concept
it answers and the period it covers, so the answer is a lookup rather than a
search. This module is the store that lookup reads.

It is deliberately narrow. Retrieval over passages answers "what does the
company say about X"; this answers "what number did they report for X", and the
two meet in the app, where a numeric question is routed here and the passage
that shows the figure is cited beside it.

**A filing reports three years at once, so ``fiscal_year`` is not the year the
figure describes.** It is the year of the filing the figure was published in. A
FY2024 10-K prints FY2022, FY2023 and FY2024 revenue side by side, and all three
arrive here with ``fiscal_year`` 2024. Filtering on it alone and taking one row
answers "Apple's FY2024 revenue" with FY2022's figure, and reads as confident and
correct, which is the same failure the hard pre-filter in ``base.py`` exists to
prevent one layer up.

``is_current_year`` marks the rows describing the year the filing reports on, and
:func:`current_year` filters to them. Use it, or match ``period_end`` against
``period_of_report`` yourself. The comparatives are kept rather than dropped
because "how did revenue move over three years" is answered from one filing when
they are there, and needs three when they are not.

Every fact kept here is traceable to a filing this project holds. The SEC's
companyfacts endpoint returns a company's whole history, across 10-Q and 8-K as
well, back to 2009 for Apple. A fact from a filing that is not in
``data/raw/manifest.jsonl`` cannot be cited by this project, because there is no
passage to show beside it and no document the reader can open, so it is dropped
rather than stored as a figure with nowhere to point.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, fields as dataclass_fields
from pathlib import Path

from ..config import configure_edgar
from ..pipeline.download import load_manifest
from .constants import INDEX_DIR

logger = logging.getLogger(__name__)

FACTS_FILE = INDEX_DIR / "facts.parquet"

# us-gaap is the financial taxonomy. The endpoint also returns dei, which is
# entity metadata -- the share count on the cover page, the filer category --
# and answers no financial question, so it is not carried.
TAXONOMY = "us-gaap"

# What is kept from each fact. The first group identifies the filing it came
# from, which is what makes a stored figure citable; the second is the figure
# itself; the third is the period it covers, which is what stops a comparative
# from being read as the year in question.
COLUMNS = (
    "ticker", "cik", "company", "accession", "form", "filing_date",
    "period_of_report",
    "concept", "label", "value", "raw_value", "unit", "scale",
    "fiscal_year", "fiscal_period", "period_start", "period_end", "period_type",
    "is_current_year",
    "statement_type", "is_audited",
)


def corpus_filings() -> dict[str, object]:
    """Every filing this project holds, keyed on accession number.

    The accession is the join between the two halves of the project: it is what
    the manifest records, what the chunker carries onto every passage, and what
    a fact arrives with. A figure and the passage showing it can therefore be
    put side by side without matching on dates or company names.
    """
    return {record.accession_no: record for record in load_manifest()}


def _rows_for(entity_facts, filings: dict, ticker: str) -> list[dict]:
    """Turn one company's facts into rows, dropping what cannot be cited."""
    rows: list[dict] = []
    for fact in entity_facts.get_all_facts():
        if fact.taxonomy != TAXONOMY:
            continue
        record = filings.get(fact.accession)
        if record is None:
            # A real fact from a filing outside this corpus: a 10-Q, or a year
            # before the scope. Nothing here can cite it.
            continue
        rows.append({
            "ticker": ticker,
            "cik": int(record.cik),
            "company": record.company,
            "accession": fact.accession,
            "form": record.form,
            "filing_date": record.filing_date,
            "period_of_report": record.period_of_report,
            "concept": fact.concept,
            "label": fact.label or "",
            # numeric_value is the figure to compute with; value is what the
            # filing printed. Both are kept because a fact can be non-numeric,
            # and because a figure quoted back to the reader should read as the
            # filing wrote it.
            "value": _as_float(fact.numeric_value),
            "raw_value": "" if fact.value is None else str(fact.value),
            "unit": fact.unit or "",
            "scale": fact.scale,
            "fiscal_year": fact.fiscal_year,
            "fiscal_period": fact.fiscal_period or "",
            "period_start": fact.period_start,
            "period_end": fact.period_end,
            "period_type": fact.period_type or "",
            # Whether this fact describes the year the filing reports on, or one
            # of the comparatives printed beside it. See the module docstring:
            # this is the column that stops "Apple's FY2024 revenue" returning
            # FY2022's figure, and it is computed here so that every caller gets
            # it right rather than each one rediscovering the trap.
            "is_current_year": _same_day(fact.period_end, record.period_of_report),
            "statement_type": fact.statement_type or "",
            "is_audited": bool(fact.is_audited),
        })
    return rows


def _same_day(left, right) -> bool:
    """Whether two dates name the same day, whatever type they arrive as.

    ``period_end`` comes back as a date and ``period_of_report`` as the string
    the manifest stored, so they are compared on their first ten characters
    rather than by equality, which would be False for every row.
    """
    if left is None or not right:
        return False
    return str(left)[:10] == str(right)[:10]


def _as_float(value) -> float | None:
    """The fact's numeric value, or None where it has none."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build(
    tickers: list[str] | None = None,
    refresh: bool = False,
    facts_file: Path = FACTS_FILE,
) -> Path:
    """Pull company facts for the corpus and write the parquet store.

    Resumable in the way that matters here: one company is one request, so a run
    that stops half way has completed whole companies. Those are already in the
    parquet and are skipped on the next run unless ``refresh`` is passed, which
    is what to use when the corpus has gained filings.

    The SEC's rate limit is enforced by edgartools, which throttles and caches
    its own requests. Hand-rolling a second limiter on top would not make the
    run politer, only harder to reason about when one of them is wrong.
    """
    import pandas as pd

    configure_edgar()
    from edgar.entity import get_company_facts

    filings = corpus_filings()
    if not filings:
        raise RuntimeError(
            "data/raw/manifest.jsonl is empty. Run the download stage first: "
            "python -m src.pipeline download"
        )

    wanted = {t.upper() for t in tickers} if tickers else None
    by_ticker: dict[str, int] = {}
    for record in filings.values():
        if wanted is None or record.ticker in wanted:
            by_ticker[record.ticker] = int(record.cik)

    existing = None
    done: set[str] = set()
    if facts_file.exists() and not refresh:
        existing = pd.read_parquet(facts_file)
        done = set(existing["ticker"].unique())
        if done:
            print(f"resuming: {len(done)} companies already stored")

    rows: list[dict] = []
    pending = [t for t in sorted(by_ticker) if t not in done]
    for position, ticker in enumerate(pending, start=1):
        try:
            entity_facts = get_company_facts(by_ticker[ticker])
        except Exception as error:
            # One company having no facts published is not a reason to lose the
            # fourteen already pulled, so it is reported and the run goes on.
            logger.warning("no company facts for %s: %s", ticker, error)
            print(f"  {position}/{len(pending)}  {ticker}: no facts ({error})")
            continue
        found = _rows_for(entity_facts, filings, ticker)
        rows.extend(found)
        print(f"  {position}/{len(pending)}  {ticker}: {len(found):,} facts", flush=True)

    frame = pd.DataFrame(rows, columns=list(COLUMNS))
    if existing is not None and not existing.empty:
        frame = pd.concat([existing, frame], ignore_index=True)
    # One concept reported once per filing: the endpoint repeats a fact across
    # the filings that restate it, and two rows identical on these five would be
    # the same figure counted twice by anything that aggregates.
    frame = frame.drop_duplicates(
        subset=["ticker", "accession", "concept", "period_end", "value"]
    )

    facts_file.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(facts_file, index=False)

    print()
    print(f"facts:    {len(frame):,} rows, {frame['concept'].nunique():,} concepts")
    print(f"filings:  {frame['accession'].nunique()} of {len(filings)} in the manifest")
    print(f"written:  {facts_file}")
    return facts_file


def current_year(frame):
    """Only the facts describing the year their filing reports on.

    The one-line form of the warning in the module docstring. A question about a
    named fiscal year wants this; a question about a trend does not, since the
    comparatives are what make a three-year movement answerable from a single
    filing.
    """
    return frame[frame["is_current_year"]]


def load_facts(facts_file: Path = FACTS_FILE):
    """Read the store back, or raise saying how to build it."""
    import pandas as pd

    if not facts_file.exists():
        raise FileNotFoundError(
            f"No facts store at {facts_file}. Build it with: "
            "python -m src.retrieval facts"
        )
    return pd.read_parquet(facts_file)
