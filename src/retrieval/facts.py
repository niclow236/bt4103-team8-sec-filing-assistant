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

**Ending on the fiscal year end is not enough either.** A 10-K can also tag a
fourth-quarter figure, and the quarter ends on the same day the year does:
Amazon's FY2022 10-K carries an annual impairment charge and a Q4 one, both
ending 31 December 2022. ``fiscal_period`` cannot separate them, since it is the
filing's period ("FY") on every fact. The duration can, so a duration fact
counts as the reported year only when it runs about a year.

``is_current_year`` marks the rows describing the year the filing reports on,
and :func:`current_year` filters to them. The comparatives are kept rather than
dropped because "how did revenue move over three years" is answered from one
filing when they are there, and needs three when they are not.

Every fact kept here is traceable to a filing this project holds. The SEC's
companyfacts endpoint returns a company's whole history, across 10-Q and 8-K as
well, back to 2009 for Apple. A fact from a filing that is not in
``data/raw/manifest.jsonl`` cannot be cited by this project, because there is no
passage to show beside it and no document the reader can open, so it is dropped
rather than stored as a figure with nowhere to point.
"""

from __future__ import annotations

import logging
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

# How long a duration has to run to be the fiscal year rather than a part of it.
# A 52/53-week fiscal year runs 364 or 371 days, a calendar year 365 or 366, and
# a quarter 90 to 98, so any window between the two separates them; this one is
# wide enough for every calendar here and far from both. Measured on this
# corpus: every current-year duration fact is either 90-91 or 363-365 days.
ANNUAL_MIN_DAYS = 350
ANNUAL_MAX_DAYS = 380

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

# Two rows equal on all of these are the same figure counted twice. period_start
# is part of the key because an annual and a fourth-quarter figure end on the
# same day and can carry the same value -- a charge booked entirely in Q4 -- and
# without it one of the two would be discarded at random. unit is part of it
# because one concept can be reported in more than one.
DUPLICATE_KEY = [
    "ticker", "accession", "concept", "unit", "period_start", "period_end", "value",
]


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
            "statement_type": fact.statement_type or "",
            "is_audited": bool(fact.is_audited),
        })
    return rows


def _as_float(value) -> float | None:
    """The fact's numeric value, or None where it has none."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def mark_current_year(frame):
    """Set ``is_current_year``: the fact describes the year its filing reports on.

    Two conditions, both required. The fact's period ends on the filing's
    period of report, which rules out the comparatives. And it is either an
    instant -- a balance at year end -- or a duration of about a year, which
    rules out a fourth quarter ending the same day. See the module docstring
    for why each is needed.

    The one definition of the column, used when the store is built and again
    when it is read, so a store written before the rule changed is corrected on
    load instead of answering with its old flags.
    """
    import pandas as pd

    end = frame["period_end"].astype(str).str[:10]
    report = frame["period_of_report"].astype(str).str[:10]
    start = pd.to_datetime(frame["period_start"], errors="coerce")
    days = (pd.to_datetime(end, errors="coerce") - start).dt.days
    annual = days.between(ANNUAL_MIN_DAYS, ANNUAL_MAX_DAYS)
    frame = frame.copy()
    frame["is_current_year"] = (end == report) & (start.isna() | annual)
    return frame


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

    A run only ever replaces the companies it successfully pulled. Everything
    else in the store is carried over as it was: the companies a ``--tickers``
    run was not asked about, and any company whose request failed or came back
    with nothing citable. So ``--refresh --tickers AAPL`` refreshes Apple and
    leaves the other fourteen alone, and one failed request during a full
    refresh costs that company nothing it already had.

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

    existing = pd.read_parquet(facts_file) if facts_file.exists() else None
    stored = set(existing["ticker"].unique()) if existing is not None else set()
    if stored and not refresh:
        print(f"resuming: {len(stored)} companies already stored")
    pending = [t for t in sorted(by_ticker) if refresh or t not in stored]

    pulled: dict[str, list[dict]] = {}
    failed: list[str] = []
    for position, ticker in enumerate(pending, start=1):
        try:
            entity_facts = get_company_facts(by_ticker[ticker])
        except Exception as error:
            logger.warning("no company facts for %s: %s", ticker, error)
            print(f"  {position}/{len(pending)}  {ticker}: request failed ({error})")
            failed.append(ticker)
            continue
        found = _rows_for(entity_facts, filings, ticker)
        if not found:
            # This company's filings are in the manifest, so an answer with
            # nothing citable in it is an anomaly rather than a result, and not
            # a reason to discard what an earlier run stored.
            print(f"  {position}/{len(pending)}  {ticker}: no citable facts returned")
            failed.append(ticker)
            continue
        pulled[ticker] = found
        print(f"  {position}/{len(pending)}  {ticker}: {len(found):,} facts", flush=True)

    frames = []
    if existing is not None and not existing.empty:
        frames.append(existing[~existing["ticker"].isin(list(pulled))])
    if pulled:
        frames.append(pd.DataFrame(
            [row for rows in pulled.values() for row in rows],
            columns=[column for column in COLUMNS if column != "is_current_year"],
        ))
    if not frames:
        raise RuntimeError("No company facts were pulled and none were stored before.")
    frame = pd.concat(frames, ignore_index=True)

    # Applied to everything, carried-over rows included, so a filing that has
    # since left the manifest takes its facts with it.
    untraceable = ~frame["accession"].isin(list(filings))
    if untraceable.any():
        print(f"dropped:  {int(untraceable.sum()):,} facts from filings no longer in the manifest")
        frame = frame[~untraceable]
    frame = frame.drop_duplicates(subset=DUPLICATE_KEY)
    frame = mark_current_year(frame)[list(COLUMNS)]

    facts_file.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(facts_file, index=False)

    print()
    print(f"facts:    {len(frame):,} rows, {frame['concept'].nunique():,} concepts, "
          f"{int(frame['is_current_year'].sum()):,} describing their filing's own year")
    print(f"filings:  {frame['accession'].nunique()} of {len(filings)} in the manifest")
    if failed:
        print(f"kept as stored, not refreshed: {', '.join(failed)}")
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
    """Read the store back, or raise saying how to build it.

    ``is_current_year`` is recomputed on the way out, so a store built before
    the fourth-quarter rule existed answers with the current rule rather than
    the flags it was written with.
    """
    import pandas as pd

    if not facts_file.exists():
        raise FileNotFoundError(
            f"No facts store at {facts_file}. Build it with: "
            "python -m src.retrieval facts"
        )
    return mark_current_year(pd.read_parquet(facts_file))
