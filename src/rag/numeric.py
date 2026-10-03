"""Answer a numeric question from the facts store, and cite the table that shows it.

"What was Microsoft's FY2024 total revenue?" has an exact answer that the filer
already published as tagged XBRL. Asking a model to read it out of a retrieved
passage puts a language model between a question with one right answer and the
number itself, which is where a figure gets rounded, transposed, or taken from
the wrong column of a table that prints three years side by side. This module
takes the other route: look the figure up, then retrieve the passage that prints
it so the reader can check it against the filing.

The route is deliberately narrow, and returns None rather than guessing:

- One company and one fiscal year. "Compare Apple and Microsoft" needs a figure
  each and a sentence setting them against one another, which is #35's job.
- One metric, from ``constants.FINANCIAL_METRICS``. A question naming two, or
  naming none of them, is not a lookup.
- One value. Where two concepts for the same metric disagree -- a filer tagging
  both ``Revenues`` and ``RevenueFromContractWithCustomerExcludingAssessedTax``
  with different figures -- the store cannot say which the question meant, and
  answering would be picking one at random and sounding certain about it.
- A passage from that same filing that actually prints the figure. Without one
  there is nothing to cite, and a figure shown beside a citation that does not
  contain it is worse than no answer.

Every one of those is a fall-through, not a failure: :func:`answer_from_facts`
returns None and :func:`answer_question` retrieves and generates as it would
have anyway. The route only ever replaces an answer it can fully support.

The answer it builds is this project's own sentence, not a model's, so it is
recorded as having come from the facts store rather than from whichever model
was configured -- see ``constants.FACTS_PROVIDER``. Nothing here calls a model,
which is also why it returns in the time of one retrieval, or two where the
first finds nothing to cite, rather than the minutes a local model takes to
read a prompt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from math import isfinite
from pathlib import Path
from time import perf_counter
from typing import Any

from ..retrieval.facts import FACTS_FILE, current_year, load_facts, printed_forms_by_scale
from ..retrieval.records import Query, RetrievedPassage
from ..retrieval.constants import TABLE_BOOST
from .constants import (
    ACCESSION_PATTERN,
    COMPANY_ALIASES,
    FACT_PASSAGE_K,
    FACT_SCOPE_UNSUPPORTED,
    FACTS_PROVIDER,
    FACTS_SENTENCE,
    FACTS_SOURCE,
    FACTS_TEMPLATE_ID,
    FINANCIAL_METRICS,
    METRIC_QUALIFIER,
    OUT_OF_SCOPE_ALIASES,
    QUESTION_SCAFFOLDING,
    SEGMENT_ALIASES,
    TABLE_SCALE,
    UNIT_ALIASES,
)
from .records import Answer, Citation, GenerationConfig, SentenceCitations, render_sentence

# The word a table uses for each scale, so a figure divided by a million is
# only accepted where the passage says it is in millions.
_SCALE_WORDS = {1_000: ("thousand", "thousands"),
                1_000_000: ("million", "millions"),
                1_000_000_000: ("billion", "billions")}

# A scale word straight after a figure, which is what makes "$5.2 billion"
# a different number from a figure of 5.2.
_SCALE_AFTER = re.compile(r"\s*(?:thousand|million|billion|trillion)s?\b", re.I)

# "per share" straight after a line item's name, which makes it a per-share
# amount and no longer the line item: "net income per share", "net loss per
# share, basic and diluted", "net income per diluted share".
_PER_SHARE_AFTER = re.compile(
    r"\s+per\s+(?:(?:basic|diluted|common|ordinary)\s+)*share\b", re.I)

# How a filing writes a negative: a minus sign, either side of the currency
# symbol, or the accounting form, wrapping the figure in parentheses.
_CURRENCY = r"(?:us\$|usd|s\$|sgd|eur|gbp|jpy|[$€£¥])"
_MINUS_BEFORE = re.compile(rf"(?:[-−]\s*{_CURRENCY}?|{_CURRENCY}\s*[-−])\s*$", re.I)
_OPEN_BEFORE = re.compile(rf"\(\s*{_CURRENCY}?\s*$", re.I)
_CLOSE_AFTER = re.compile(r"\s*\)")

# The company names and tickers a question is scoped by, which carry no line
# item and so are taken out before what is left is weighed. A segment alias is
# deliberately not here: "AWS" names a part of Amazon, so a question using it
# is asking for something the whole-company figure does not answer, and it has
# to survive to be counted against the question.
_ENTITY_NAMES = re.compile(
    r"\b(?:" + "|".join(
        re.escape(name).replace(r"\ ", r"\s+")
        for name in sorted(
            (
                {alias for aliases in COMPANY_ALIASES.values() for alias in aliases}
                - set(SEGMENT_ALIASES)
                | {alias for aliases in OUT_OF_SCOPE_ALIASES.values() for alias in aliases}
                | set(COMPANY_ALIASES)
            ),
            key=len, reverse=True,
        )
    ) + r")\b",
    re.I,
)


@dataclass(frozen=True)
class Fact:
    """One figure from the store, with everything needed to state and cite it."""

    metric: str            # the key in FINANCIAL_METRICS the question asked for
    concept: str           # the XBRL concept the filer tagged it with
    # What to call the line item in the answer. The metric's first alias, not
    # the store's ``label``: that column holds the taxonomy's standard label,
    # so revenue reads "Revenue from Contract with Customer, Excluding
    # Assessed Tax", which is neither what the filing prints nor what a
    # keyword search for the table should be given.
    label: str
    value: Decimal         # the figure, in base units
    unit: str              # "USD", "USD/shares", ...; this project's name for it
    ticker: str
    company: str
    fiscal_year: int
    accession: str         # the filing that reported it, which is what may cite it
    period_end: str        # the day the period the figure covers ended

    @property
    def figure(self) -> str:
        """The figure as the answer states it."""
        return format_figure(self.value, self.unit)

    @property
    def period(self) -> str:
        """The period the figure covers, as the answer states it."""
        ended = self.period_end[:10]
        return f"fiscal year {self.fiscal_year}" + (f", ended {ended}" if ended else "")


def format_figure(value: Decimal, unit: str) -> str:
    """A figure written out exactly, in the unit the filer reported it in.

    Exactly: the stored figure with separators, never rounded to a friendlier
    scale. "$391,035,000,000" is harder to read than "$391.0 billion" and is
    the number the filing can be checked against, which is the whole point of
    answering from the store. The scale a reader wants is in the table the
    citation points at.
    """
    magnitude = abs(value)
    whole = magnitude == magnitude.to_integral_value()
    # Fixed-point on both branches: a Decimal keeps its exponent through
    # to_integral_value, and "," alone would print 3.91035E+11 back out as
    # itself -- a figure neither a reader nor verify.py's parser can read.
    digits = f"{magnitude.to_integral_value() if whole else magnitude:,f}"
    # The sign goes outside the currency symbol, as a filing writes it: a loss
    # is -$1,500,000, never $-1,500,000.
    sign = "-" if value < 0 else ""
    if unit == "USD":
        return f"{sign}${digits}"
    if unit == "USD/shares":
        return f"{sign}${digits} per share"
    if unit == "ratio":
        return f"{sign}{digits}"
    return f"{sign}{digits} {unit}".strip()


@lru_cache(maxsize=2)
def cached_facts(path: Path, mtime_ns: int):
    """The facts store, read once per file per change.

    Keyed on the modification time as well as the path, so a rebuilt store is
    picked up rather than served stale. Every caller only ever filters the
    frame, never writes to it, so one copy is safe to share -- and reading it
    per question meant a full parquet read and a ``to_datetime`` over every
    row for each question in a benchmark run, including the ones that then
    fall through to retrieval.
    """
    return load_facts(path)


@lru_cache(maxsize=None)
def _alias(alias: str) -> re.Pattern[str]:
    """A metric's alias as a whole word or phrase, in any case: the one pattern
    every reader of a question or a claim finds it by."""
    return re.compile(r"\b" + re.escape(alias) + r"\b", re.I)


def metrics_in(text: str) -> set[str]:
    """Every metric whose words appear in the text, by whole-word match.

    The plain reading, with no judgement about what the text is asking for:
    ``verify.py`` reads a claim this way, because "revenue was $391,035
    million" is a claim about revenue however it is phrased.
    :func:`find_metric` adds the judgement a question needs. One definition
    rather than two, so a change to how an alias is matched cannot land on one
    side only and leave the router answering what the checker will not check.

    A name inside a longer name of another metric is part of that name:
    "diluted net income per share" names earnings per share, and the "net
    income" in it does not name net income. A text that also says "net income"
    on its own names both.

    A name with "per share" after it is a per-share amount of the line item
    and not the line item. "Net income per share" is what several filers call
    earnings per share, and without "basic" or "diluted" in front it does not
    say which, so it names neither of those and not net income either. Read
    as net income, a right answer in those words was checked against the
    store's net income and marked a mismatch.
    """
    found = {
        key: [match.span() for alias in metric.aliases
              for match in _alias(alias).finditer(text)]
        for key, metric in FINANCIAL_METRICS.items()
    }

    def inside_another(key: str, span: tuple[int, int]) -> bool:
        return any(
            other != key and start <= span[0] and span[1] <= end and (start, end) != span
            for other, spans in found.items() for start, end in spans
        )

    return {key for key, spans in found.items()
            if any(not inside_another(key, span) and not _per_share_amount(text, key, span)
                   for span in spans)}


def _per_share_amount(text: str, key: str, span: tuple[int, int]) -> bool:
    """Whether the name at ``span`` is a line item's with "per share" after it."""
    return (not FINANCIAL_METRICS[key].unit.endswith("/shares")
            and _PER_SHARE_AFTER.match(text, span[1]) is not None)


def names_a_per_share_amount(text: str) -> bool:
    """Whether the text names a line item per share, as "net income per share" does.

    For a text :func:`metrics_in` finds no line item in. Such a figure is a
    per-share one of a line item the text does not pin down, so the checker
    cannot say what to compare it with, and reading it as a plain dollar
    amount would set "$12.36" against a table in millions and call it wrong.
    """
    return any(
        _per_share_amount(text, key, match.span())
        for key, metric in FINANCIAL_METRICS.items() for alias in metric.aliases
        for match in _alias(alias).finditer(text)
    )


def _longest_alias(metric: str, question: str) -> str:
    """The longest of the metric's aliases that the question actually uses."""
    matched = [
        alias for alias in FINANCIAL_METRICS[metric].aliases
        if _alias(alias).search(question)
    ]
    return max(matched, key=len)


def _asks_only_for(question: str, alias: str) -> bool:
    """Whether the question asks for this line item and nothing narrower.

    Take the metric's own words out, take out the company and the year the
    question is scoped by, and what is left has to be scaffolding: the frame
    of a question and the verbs of reporting. Anything else is the question
    asking about something the whole-company annual figure is not --
    "**Services** net sales", "revenue in **Greater China**", "net income
    **per share**", "total liabilities **and** shareholders' equity", "how
    does Apple **recognise** revenue", "what **drove** net sales".

    This is positive evidence, which is what the route needs: a blocklist of
    segment names would have to grow one name at a time across fifteen
    companies and would still be a list of the ones somebody thought of.
    """
    rest = _alias(alias).sub(" ", question)
    rest = _ENTITY_NAMES.sub(" ", rest)
    words = re.findall(r"[A-Za-z0-9']+", rest)
    return all(_is_scaffolding(word) for word in words)


def _is_scaffolding(word: str) -> bool:
    """Whether one leftover word carries no line item of its own."""
    plain = word.lower()
    if plain in QUESTION_SCAFFOLDING:
        return True
    # A possessive is scaffolding only where its owner is: "the company's",
    # "this year's". "LinkedIn's revenue" asks for a part of Microsoft, and
    # the company's own name has already been taken out by _ENTITY_NAMES,
    # leaving a bare "'s" -- so a possessive that still has an owner here is
    # one that names something other than the company being asked about.
    if plain.endswith("'s") and len(plain) > 2:
        return _is_scaffolding(plain[:-2])
    # A year in any of the forms a question writes one, and the bare possessive
    # left behind where a company name was taken out.
    return bool(re.fullmatch(r"(?:fye?|fiscal)?'?\d{2,4}|'s?", plain))


def find_metric(question: str) -> str | None:
    """The one metric a question asks for, or None where it asks for something else.

    Three tests, and a question has to pass all of them. It must name exactly
    one metric -- "how did revenue and net income move" is two lookups and a
    sentence joining them, which this route does not write. It must not name a
    scope a stored annual figure cannot answer, nor a qualifier that makes the
    line item a different one; see ``constants.FACT_SCOPE_UNSUPPORTED`` and
    ``METRIC_QUALIFIER``. And it must ask for that line item and nothing
    narrower, which :func:`_asks_only_for` is the positive test for.
    """
    if FACT_SCOPE_UNSUPPORTED.search(question) or METRIC_QUALIFIER.search(question):
        return None
    found = metrics_in(question)
    if len(found) != 1:
        return None
    metric = found.pop()
    return metric if _asks_only_for(question, _longest_alias(metric, question)) else None


def lookup_fact(
    question: str,
    tickers: tuple[str, ...],
    fiscal_years: tuple[int, ...],
    *,
    facts_file: Path = FACTS_FILE,
    frame: Any | None = None,
) -> Fact | None:
    """The figure a numeric question asks for, or None where the store cannot say.

    ``frame`` is a facts frame already in memory, for a caller answering many
    questions or a test; otherwise the store is read from ``facts_file``. A
    missing or unreadable store is None like any other miss, because a
    teammate who has not built it should get a retrieved answer rather than an
    exception from a route they did not know they were on.
    """
    metric = find_metric(question)
    if metric is None or len(tickers) != 1 or len(fiscal_years) != 1:
        return None
    ticker, year = tickers[0].upper(), int(fiscal_years[0])

    if frame is None:
        try:
            path = Path(facts_file)
            frame = cached_facts(path, path.stat().st_mtime_ns)
        except Exception:
            # Every way of failing to read the store is a miss, deliberately:
            # a corrupt file raises whatever the parquet engine chooses to
            # raise, and this route is a shortcut past retrieval, never a new
            # way for a question to fail. The question goes to retrieval, which
            # is where it would have gone if the store held nothing for it.
            return None
    required = {"ticker", "fiscal_year", "concept", "unit", "value", "accession",
                "company", "label", "period_end", "raw_value", "is_current_year"}
    if not required.issubset(frame.columns):
        return None

    wanted = FINANCIAL_METRICS[metric]
    rows = current_year(frame)
    rows = rows[(rows["ticker"].astype(str).str.upper() == ticker)
                & (rows["fiscal_year"] == year)
                & rows["concept"].astype(str).str.split(":").str[-1].isin(wanted.concepts)]

    candidates: list[Fact] = []
    for row in rows.to_dict("records"):
        if UNIT_ALIASES.get(str(row["unit"]).lower(), str(row["unit"])) != wanted.unit:
            continue
        # raw_value is what the filing published; value is the same figure as a
        # float. The string is parsed first so a figure survives as it was
        # filed rather than through binary floating point.
        for source in (row.get("raw_value"), row.get("value")):
            try:
                value = Decimal(str(source).strip())
            except (InvalidOperation, AttributeError, TypeError):
                continue
            if value.is_finite():
                break
        else:
            continue
        candidates.append(Fact(
            metric=metric,
            concept=str(row["concept"]).split(":")[-1],
            label=wanted.aliases[0],
            value=value,
            unit=wanted.unit,
            ticker=ticker,
            company=str(row.get("company") or "").strip() or ticker,
            fiscal_year=year,
            accession=str(row["accession"]).strip(),
            period_end=str(row.get("period_end") or "")[:10],
        ))

    if not candidates:
        return None
    # Two concepts disagreeing is the store saying it does not know which one
    # the question meant. Picking either would read as certain and be a coin
    # toss, so the question goes to retrieval instead.
    if len({fact.value for fact in candidates}) > 1:
        return None
    return candidates[0]


def _shown_negative(before: str, after: str) -> bool:
    """Whether the figure printed at this position is a negative one."""
    return bool(
        _MINUS_BEFORE.search(before)
        or (_OPEN_BEFORE.search(before) and _CLOSE_AFTER.match(after))
    )


def _row_names(before: str, labels: tuple[str, ...]) -> bool:
    """Whether the figure sits in a table row labelled with one of the metric's names."""
    row = before[before.rfind("\n") + 1:]
    return row.lstrip().startswith("|") and any(
        re.search(rf"\b{re.escape(label)}\b", row, re.I) for label in labels
    )


def prints_figure(
    text: str,
    by_scale: dict[int, set[str]],
    *,
    negative: bool = False,
    labels: tuple[str, ...] = (),
) -> bool:
    """Whether a passage prints the figure, rather than merely containing its digits.

    Three things a bare substring search gets wrong, and each of them puts a
    figure next to a citation that does not show it.

    "391,035" is inside "1,391,035" and "7,000" is inside "17,000", so a match
    has to begin and end at a number boundary.

    A needle is a figure divided by a thousand, a million or a billion, so
    "100" stands for $100,000,000 only where the passage says it is in
    millions -- in a heading, as a statement table gives it, or beside the
    figure, as prose does ("$391,035 million"). A needle at scale 1 is the
    figure itself and needs neither, but it does have to be free of a scale
    word of its own: "5.2" in "$5.2 billion" is not a figure of 5.2. A
    declared heading is deliberately not applied to scale 1, because an
    income statement headed "in millions, except per share amounts" prints
    its earnings per share unscaled in the same table.

    Most statement passages never name their scale, though: the "(In
    millions ...)" caption sits outside the table the chunker keeps, and 73%
    of the corpus's table passages carry no scale word at all. Where a passage
    declares none, a scaled figure is accepted in a table row that names the
    metric (``labels``), which is how a reader knows "| Total net sales |
    391,035 |" is revenue in millions. A passage that declares a different
    scale is still refused.

    A needle with a decimal point in it, "56.9" for a filer that prints its
    millions to one place, is held to more than that: a declared scale is not
    enough, because a table in millions holds percentages and rates written
    the same way. It has to sit in a row that names the metric, or carry its
    scale word beside it.

    And the needles carry no sign, so the sign is checked here: a loss of
    $1,500 is printed "(1,500)" or "-1,500", and a passage showing a positive
    1,500 is a different line item. A figure whose sign disagrees with the
    fact's is passed over rather than cited.
    """
    heading = TABLE_SCALE.search(text)
    for divisor, needles in by_scale.items():
        words = _SCALE_WORDS.get(divisor, ())
        declared = bool(words) and heading is not None and heading[1].lower() in words
        for needle in needles:
            for match in re.finditer(
                rf"(?<![\d.,]){re.escape(needle)}(?![\d,%]|\.\d)", text
            ):
                before, after = text[:match.start()], text[match.end():]
                if _shown_negative(before, after) != negative:
                    continue
                if divisor == 1:
                    if _SCALE_AFTER.match(after) is None:
                        return True
                    continue
                named = _row_names(before, labels)
                if (re.match(rf"\s*(?:{'|'.join(words)})\b", after, re.I)
                        or (declared and (named or "." not in needle))
                        or (heading is None and named)):
                    return True
    return False


def supporting_passage(
    fact: Fact,
    retriever: Any,
    *,
    top_k: int = FACT_PASSAGE_K,
    min_score: float | None = None,
    search_text: str | None = None,
) -> RetrievedPassage | None:
    """A passage from the fact's own filing that prints the fact's figure.

    ``search_text`` is the metric alias the question used, when available:
    Amazon's statement says "net sales", not the canonical "total revenue".

    One search over both kinds of passage, preferring a table among whatever
    prints the figure: a statement table is where a figure sits under the
    label and year that give it meaning, and a table passage keeps those
    headers. Prose is accepted on the same terms -- it has to print the figure
    -- since what is wanted is a citation showing the number, not a particular
    kind of passage. Searching tables and then everything would have run two
    searches to look through the same passages twice, on top of the one
    ``answer_question`` runs when this finds nothing.

    The passage must come from the filing the fact was reported in, which the
    chunk_id carries: a FY2023 filing prints FY2023's revenue too, and citing
    it for the FY2024 figure would point the reader at the wrong number on the
    right subject.

    A passage has to clear the same bar the retrieval path sets before it is
    cited: a usable score and some text, and ``min_score`` where the caller
    set one. A route that cited a passage the caller's own floor would have
    rejected would answer where the same question, asked the same way, abstains.
    """
    by_scale = printed_forms_by_scale(fact.value)
    if not by_scale:
        return None
    label = search_text if search_text is not None else fact.label
    query = Query(
        text=f"{label} {fact.ticker} FY{fact.fiscal_year}",
        # The line item alone for keyword search. The ticker and the year are
        # already hard filters, and #87 measured that leaving them in the
        # keyword text buries the statement tables under the prose repeating
        # them.
        keyword_text=label,
        tickers=(fact.ticker,),
        fiscal_years=(fact.fiscal_year,),
        top_k=top_k,
        wants_figures=True,
        # The same boost every other numeric query carries, since the passage
        # wanted here is a statement table: without it prose that repeats the
        # line item can push the table out of the top top_k entirely.
        table_boost=TABLE_BOOST,
    )
    matches = [
        passage for passage in retriever.search(query)
        if isfinite(passage.score) and passage.text.strip()
        and (min_score is None or passage.score >= min_score)
        and (found := ACCESSION_PATTERN.match(passage.chunk_id))
        and found[0] == fact.accession
        and prints_figure(passage.text, by_scale, negative=fact.value < 0,
                          labels=FINANCIAL_METRICS[fact.metric].aliases)
    ]
    tables = [passage for passage in matches if passage.content_type == "table"]
    for preferred in (tables, matches):
        if preferred:
            return preferred[0]
    return None


def answer_from_facts(
    question: str,
    tickers: tuple[str, ...],
    fiscal_years: tuple[int, ...],
    retriever: Any,
    *,
    facts_file: Path = FACTS_FILE,
    frame: Any | None = None,
    top_k: int = FACT_PASSAGE_K,
    min_score: float | None = None,
) -> Answer | None:
    """The whole route: look the figure up, find the passage that prints it, state it.

    None where any step cannot be completed, which the caller answers by
    retrieving and generating as usual. See the module docstring for what each
    step refuses and why. ``min_score`` is the caller's floor on a passage's
    score, applied here as the retrieval path applies it.

    The passage is searched for under the name the question used for the line
    item, and where that finds none, once more under its other names: the
    figure is the same whatever the filer calls the line, and a question need
    not use the filer's word for it.
    """
    started = perf_counter()
    fact = lookup_fact(question, tickers, fiscal_years, facts_file=facts_file, frame=frame)
    if fact is None:
        return None
    asked = _longest_alias(fact.metric, question)
    passage = supporting_passage(
        fact, retriever, top_k=top_k, min_score=min_score, search_text=asked,
    )
    if passage is None:
        # The filer may word the line differently from the question: Amazon's
        # statement says "net sales", so a search for "total revenue" does not
        # reach it. One more search, for the line item's other names together,
        # before giving the question to a model.
        others = " ".join(
            alias for alias in FINANCIAL_METRICS[fact.metric].aliases if alias != asked
        )
        if others:
            passage = supporting_passage(
                fact, retriever, top_k=top_k, min_score=min_score, search_text=others,
            )
    if passage is None:
        return None

    shown = (passage,)
    text = FACTS_SENTENCE.format(
        company=fact.company, label=fact.label, figure=fact.figure, period=fact.period,
    )
    citation = Citation(marker=1, chunk_id=passage.chunk_id, resolved=True)
    sentence = SentenceCitations(
        raw_text=render_sentence(text, (1,)),
        text=render_sentence(text, (1,)),
        citations=(citation,),
    )
    return Answer(
        question=question,
        text=sentence.text,
        citations=(citation,),
        passages=shown,
        abstained=False,
        config=GenerationConfig(
            provider=FACTS_PROVIDER, model=FACTS_SOURCE,
            prompt_template_id=FACTS_TEMPLATE_ID,
        ),
        latency_ms=(perf_counter() - started) * 1000.0,
        sentences=(sentence,),
    )


__all__ = [
    "Fact",
    "answer_from_facts",
    "find_metric",
    "format_figure",
    "lookup_fact",
    "prints_figure",
    "supporting_passage",
]
