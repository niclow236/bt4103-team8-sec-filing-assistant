"""Query understanding: read a question, and build the retrieval Query it needs.

A question arrives as prose -- "how did Apple's revenue change from FY2022 to
FY2024" -- and the retrievers take a ``Query``, whose filters have to hold
before scoring starts. This module is the boundary between the two. It pulls
the companies and fiscal years out of the question, resolves them against the
corpus's scope, decides what kind of question it is, and hands back a
``ParsedQuestion`` that exposes every one of those decisions, so the app can
show "Filtered to AAPL, FY2022-FY2024" next to the answer and the user can see
when the reading was wrong.

The reading is rule-based on purpose. The scope is fifteen named companies
and five fiscal years, which is small enough to match exactly and not a thing
to guess at: a model that resolved "Meta" to Microsoft one time in fifty would
pass the filter check and return the right answer for the wrong company. The
rules use the lists in ``constants.py`` and the local facts store's XBRL labels
for additional financial line items.

An entity the rules see but cannot resolve -- NVIDIA, which was dropped from
the scope, or FY2015, which is before it -- never raises. The filter for it is
left open, so the search runs over the whole corpus, and the mention is
reported in ``unresolved`` so the app can say why the answer is not about it.
A question is only ever refused for being empty, which is a caller's bug and
not a user's.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..config import read_tickers
from ..pipeline.constants import DEFAULT_FISCAL_YEARS
from ..retrieval.constants import TABLE_BOOST
from ..retrieval.facts import FACTS_FILE
from ..retrieval.records import Query
from .constants import (
    ADVICE_CUES,
    BEYOND_FILING_NOUNS,
    COMPANY_ALIASES,
    COMPARATIVE_CUES,
    CORPORATE_SUFFIXES,
    COUNT_AFTER,
    CURRENCY_BEFORE,
    FUTURE_CUES,
    FIGURE_LINE_ITEM_CUES,
    FIGURE_METRIC_CUES,
    MAGNITUDE_AFTER,
    NUMERIC_CUES,
    OUT_OF_SCOPE_ALIASES,
    OWNING_VERBS,
    PREDICTION_VERBS,
    QUANTITY_BEFORE,
    QUESTION_TYPES,
    QUESTION_SCAFFOLDING,
    RELATION_NOUNS,
    REPORTING_VERBS,
    SEGMENT_ALIASES,
    TEMPORAL_CUES,
    UNANSWERABLE_BECAUSE,
    UnanswerableBecause,
)

# A year as a question writes it. Four alternatives, tried in this order:
#   FY2024, FY24, FYE2024      -- a fiscal prefix, then two or four digits
#   fiscal 2024, fiscal year 24
#   '24, ’24                   -- an apostrophe, straight or curly, marks a two-digit
#                                 year, but not a quoted number like '10-K'
#   2024                       -- a bare number, but only one that looks like a year
# A bare two-digit number is never a year: "24" in "24 percent" is a figure.
_YEAR = re.compile(
    r"\b(?:fye?)\s?(?P<fy>\d{4}|\d{2})\b"
    r"|\bfiscal(?:\s+year)?\s+(?P<fiscal>\d{4}|\d{2})\b"
    r"|['’](?P<apos>\d{2})(?![\w'’]|-[A-Za-z])"
    r"|\b(?P<bare>(?:19|20)\d{2})\b",
    re.IGNORECASE,
)

# "FY22-24", "FY2022-24" and "FY22 to 24": the second year borrows the first
# one's prefix. Rewritten to "FY22 - FY24" before scanning, so the scanner sees
# two years; a two-digit second year reads as 20xx whatever the first one was.
_SHORT_RANGE = re.compile(
    r"\b(fye?\s?)(\d{4}|\d{2})\s*(-|–|—|to|through)\s*(\d{2})\b", re.IGNORECASE
)

# What can sit between two years to make them a range rather than a pair.
# "and" is only a range after "between": "in 2022 and 2024" is two years, and
# "between 2022 and 2024" is three.
_RANGE_JOIN = re.compile(r"^(?:-|–|—|to|through|thru|until|till)$", re.IGNORECASE)
_BETWEEN = re.compile(r"\bbetween\s*$", re.IGNORECASE)

# The widest range a question is allowed to expand to. Wider than the corpus,
# and there to stop "1999 to 2024" turning into twenty-six filters.
_MAX_RANGE = 10

# What marks a bare four-digit number as something other than a filing year:
# a currency before it, a magnitude after it, a count after it with no
# possessive before it, a comparison of quantity before it with a plural after
# it, or "Act of" before it. See the tables in ``constants.py`` and
# ``_is_not_a_year``.
_CURRENCY_BEFORE = re.compile(
    "(?:" + "|".join(re.escape(c) for c in CURRENCY_BEFORE) + r")\s*$", re.IGNORECASE
)
_MAGNITUDE_AFTER = re.compile(
    r"^\s*(?:" + "|".join(re.escape(u) for u in MAGNITUDE_AFTER) + r")(?!\w)", re.IGNORECASE
)
_COUNT_AFTER = re.compile(
    r"^\s*(?:" + "|".join(re.escape(u) for u in COUNT_AFTER) + r")(?!\w)", re.IGNORECASE
)
_QUANTITY_BEFORE = re.compile(
    r"\b(?:" + "|".join(re.escape(q).replace(r"\ ", r"\s+") for q in QUANTITY_BEFORE)
    + r")\s+$",
    re.IGNORECASE,
)
_POSSESSIVE_BEFORE = re.compile(r"(?:['’]s|\b(?:its|their|the))\s+$", re.IGNORECASE)
_PLURAL_AFTER = re.compile(r"^\s+[a-z]+s\b", re.IGNORECASE)
_LAW_BEFORE = re.compile(r"\bact\s+of\s+$", re.IGNORECASE)

# A prediction verb used as a request: at the start, after a clause break, or
# after "can you" / "could you" / "please". "what does Apple predict" has the
# verb after a subject and is not a request.
# Not before "of": "Loss Contingency, Estimate of Possible Loss" is a line
# item's name, and the comma in it does not make "Estimate" an instruction.
# Nor is the full stop that ends a corporate suffix a clause break: "what did
# Apple Inc. estimate" has its verb after a subject like any other.
_PREDICTION_REQUEST = re.compile(
    r"(?:^|" + "".join(rf"(?<!\b{suffix})" for suffix in CORPORATE_SUFFIXES) + r"[,;:.?!]\s*"
    r"|\b(?:can|could|would|will)\s+you\s+(?:please\s+)?|\bplease\s+)"
    r"(?:" + "|".join(PREDICTION_VERBS) + r")\b(?!\s+of\b)",
    re.IGNORECASE,
)

# A reporting verb, but not one addressed to the engine: "what did Apple
# expect" is about the filing, "what do you expect" is not.
_REPORTING = re.compile(
    r"(?<!\byou\s)(?<!\w)(?:" + "|".join(re.escape(v) for v in REPORTING_VERBS) + r")(?!\w)",
    re.IGNORECASE,
)

# A reporting verb only makes the question about what the filing said when the
# filing is the one reporting. Not after "will", "would", "shall", "going to"
# or "'ll", up to three words back: "what will Apple report next year" asks
# about a filing that does not exist yet. And not when "expected" or
# "anticipated" describes the thing asked for rather than what was said:
# "Apple's expected stock price", "revenue is expected to".
_FUTURE_BEFORE = re.compile(
    r"(?:\b(?:will|would|shall|going\s+to)|['’]ll)\s+(?:[\w'’.-]+\s+){0,3}$", re.IGNORECASE
)
_DESCRIBED_BEFORE = re.compile(
    r"(?:['’]s|\b(?:is|are|be|been|its|their|the|an?))\s+$", re.IGNORECASE
)
_DESCRIBING_FORMS = frozenset({"expected", "anticipated"})

# A company named as the one a figure belongs to is named in one of three
# ways. As the subject of a verb that makes the figure its own, after an
# auxiliary: "did NVIDIA have", "does Intel report". As a possessive, "Intel's
# total revenue" (``_POSSESSIVE_AFTER``). Or after "of", "the total revenue of
# Intel". Named any other way, the company is what some other filing is being
# asked about: "revenue from Intel", "named NVIDIA as a competitor", "did
# NVIDIA account for more than 10% of any company's revenue", "customers of
# Intel".
_SUBJECT_BEFORE = re.compile(
    r"\b(?:did|does|do|has|have|had|will|would|can|could)\s+$", re.IGNORECASE
)
_OWNS_AFTER = re.compile(r"\s+(?:" + "|".join(OWNING_VERBS) + r")", re.IGNORECASE)
_OF_BEFORE = re.compile(r"\b(\w+)\s+of\s+$", re.IGNORECASE)
# What follows the company when it ends its phrase. "purchases of NVIDIA chips"
# goes on to name something else, which the figure belongs to.
_PHRASE_ENDS_AFTER = re.compile(
    r"\s*(?:$|[?,.;:]|\s(?:in|for|during|as|at|over|across)\b|\s(?:FY)?\d)", re.IGNORECASE
)

# Applied after recognised company names and possessives have been removed.
# Arbitrary words before "total" could swallow "the rationale behind Apple’s".
_ASKS_TOTAL = re.compile(
    r"\bwhat\s+(?:was|were|is|are)\s+"
    r"(?:(?:the|its|their)\s+)?total\b",
    re.IGNORECASE,
)

_FIGURE_SCAFFOLDING = QUESTION_SCAFFOLDING | frozenset(
    "compare and between from through over since change changed will next last as".split()
)


@dataclass(frozen=True)
class ParsedQuestion:
    """What the parser read out of one question, and the Query it builds.

    ``tickers`` and ``fiscal_years`` are the filters that resolved, already in
    the corpus's form, and empty means the search is unrestricted on that
    axis. ``unresolved`` is every company or year the question named that the
    corpus does not hold, as the question wrote it, so the app can show it
    back. ``wants_figures`` is whether the question asks for a number, which is
    what turns the table boost on; it is separate from ``question_type``
    because a comparison of two companies' revenue wants the tables as much
    as a plain numeric question does, and the type can only say one thing.
    """

    question: str
    question_type: str
    tickers: tuple[str, ...]
    fiscal_years: tuple[int, ...]
    unresolved: tuple[str, ...]
    wants_figures: bool
    # The question with the companies and years the filters apply taken out
    # (#87), which is what keyword search matches. Inside the filtered filing
    # those words tell no passage apart, but BM25 still scores the prose that
    # repeats them above the statement tables that never do. None, for a
    # ParsedQuestion built by hand, means the question itself. See
    # ``_search_text`` for what is removed and what stays.
    search_text: str | None = None
    # Why an unanswerable question is one, and None for every other type:
    # "request" where it asks for advice or a prediction, "company" where it
    # asks for a figure of a company the corpus does not hold, "topic" where
    # it is about a share price, next year or a company outside the corpus,
    # and "year" where it names only fiscal years the corpus does not hold.
    # ``answer_question`` refuses the first two without searching. It still
    # searches for the other two, since a filing may answer either: one prints
    # the price it paid for its own shares and what it owes next year, names
    # the companies it competes with, and prints the two years before its
    # own, so FY2020 may be in the FY2021 statements. See
    # ``constants.UNANSWERABLE_BECAUSE``.
    unanswerable_because: UnanswerableBecause | None = None

    def __post_init__(self) -> None:
        if self.question_type not in QUESTION_TYPES:
            raise ValueError(
                f"question_type must be one of {', '.join(QUESTION_TYPES)}, "
                f"got {self.question_type!r}"
            )
        if self.unanswerable_because not in (None, *UNANSWERABLE_BECAUSE):
            raise ValueError(
                f"unanswerable_because must be one of {', '.join(UNANSWERABLE_BECAUSE)}, "
                f"got {self.unanswerable_because!r}"
            )
        if self.unanswerable_because is not None and self.question_type != "unanswerable":
            raise ValueError("unanswerable_because is only for an unanswerable question")
        object.__setattr__(self, "tickers", tuple(self.tickers))
        object.__setattr__(self, "fiscal_years", tuple(self.fiscal_years))
        object.__setattr__(self, "unresolved", tuple(self.unresolved))
        if self.search_text is None:
            object.__setattr__(self, "search_text", self.question)

    @property
    def filters(self) -> dict[str, Any]:
        """The filters that resolved, as ``Query.filters`` would report them.

        Only the axes that are actually restricted appear, so an empty mapping
        means the whole corpus is searched. The app renders this; a retriever
        reads the same thing off the Query.
        """
        active: dict[str, Any] = {}
        if self.tickers:
            active["ticker"] = list(self.tickers)
        if self.fiscal_years:
            active["fiscal_year"] = list(self.fiscal_years)
        return active

    @property
    def refused(self) -> UnanswerableBecause | None:
        """Why this reading is refused without a search, or None where it is searched.

        Advice and a prediction are refused whatever is searched. An outside
        company's own figure is refused unless a company is named to search:
        a reading scoped to one chosen by hand (``scoped_to``), as the app's
        sidebar lets a user choose, is searched, since that company's filings
        may well mention the one the question named. A topic or a year the
        corpus may not hold is always searched. The one place this is decided,
        so the line under an answer cannot disagree with what was done.
        """
        because = self.unanswerable_because
        if because == "request" or (because == "company" and not self.tickers):
            return because
        return None

    @property
    def not_in_corpus(self) -> str | None:
        """The companies and years the question named that the corpus does not
        hold, as one line, or None where it holds them all.

        The one wording of it: :meth:`describe`, and the app's sidebar and
        filters line, all show this.
        """
        if not self.unresolved:
            return None
        return "Not in the corpus: " + ", ".join(self.unresolved)

    def describe(self) -> tuple[str, ...]:
        """One line per thing the parser decided, for showing under the answer.

        Written for a person rather than a log: "Companies: AAPL, MSFT" rather
        than the filters mapping. The unresolved line is the one that earns
        its place, since it is the only way the user learns that the answer
        ignored a company they asked about.

        A question read as unanswerable and searched all the same says so:
        the line is shown under the answer a filing gave, and "unanswerable"
        alone would contradict the answer above it. Whether it is searched is
        ``refused``'s to say, which is what ``answer_question`` acts on.
        """
        kind = self.question_type
        if self.unanswerable_because is not None and self.refused is None:
            kind += ", searched in case a filing answers it"
        lines = [f"Question type: {kind}"]
        if self.tickers:
            lines.append("Companies: " + ", ".join(self.tickers))
        if self.fiscal_years:
            lines.append("Fiscal years: " + ", ".join(f"FY{y}" for y in self.fiscal_years))
        if not self.tickers and not self.fiscal_years:
            lines.append("Filters: none, searching every company and year")
        if self.not_in_corpus:
            lines.append(self.not_in_corpus)
        return tuple(lines)

    def to_query(self, *, top_k: int | None = None) -> Query:
        """The retrieval request for this question.

        The filters go on as read. The table boost goes on when the question
        wants a figure, and only then: ``constants.TABLE_BOOST`` is documented
        as the value to set for a numeric question and never as a default, so
        a prose question is scored with the boost off.

        Keyword search gets ``search_text`` and dense search the question as
        asked, which is the split that measured best (see
        ``Query.keyword_text``). The prompt is built from the question by
        whoever calls ``build_prompt``, so the model still reads what the user
        asked.
        """
        fields: dict[str, Any] = {
            "text": self.question,
            "keyword_text": self.search_text if self.search_text != self.question else None,
            "tickers": self.tickers,
            "fiscal_years": self.fiscal_years,
            "table_boost": TABLE_BOOST if self.wants_figures else 1.0,
            "wants_figures": self.wants_figures,
        }
        if top_k is not None:
            fields["top_k"] = top_k
        return Query(**fields)

    def scoped_to(self, query: Query) -> ParsedQuestion:
        """This reading in the companies and years ``query`` names.

        What an answer is checked against, and what the app describes under
        it. The two can differ from what the question named: the app's sidebar
        lets a user change them, and the evaluation harness replaces them with
        the benchmark's. One definition, so the app and the harness cannot
        come to check an answer against different scopes.
        """
        return replace(self, tickers=query.tickers, fiscal_years=query.fiscal_years)


def parse_question(
    question: str,
    *,
    known_tickers: Iterable[str] | None = None,
    fiscal_years: tuple[int, int] = DEFAULT_FISCAL_YEARS,
    facts_file: Path | None = FACTS_FILE,
) -> ParsedQuestion:
    """Read one question into a ParsedQuestion.

    ``known_tickers`` is the scope the companies resolve against, defaulting to
    config/companies.txt; a test passes its own. ``fiscal_years`` is the
    inclusive range a year has to fall in to become a filter.
    ``facts_file`` supplies additional numeric cues from its XBRL labels.
    A missing or unreadable store leaves the built-in cues available.
    ``None`` disables label lookup, for callers that only need entity filters.
    """
    if not question or not question.strip():
        raise ValueError("question must not be blank")
    scope = frozenset(t.upper() for t in known_tickers) if known_tickers is not None else _scope()

    tickers, out_of_scope = _companies(question, scope)
    years, bad_years = _years(question, fiscal_years)
    figure_text = _figure_text(question, scope)
    wants_figures = bool(
        _any_cue(question, NUMERIC_CUES)
        or _ASKS_TOTAL.search(figure_text)
        or _asks_for_label(figure_text, _FIGURE_METRIC_PATTERN)
        or (facts_file is not None and _mentions_fact_label(figure_text, facts_file))
    )

    only_out_of_scope = bool(out_of_scope) and not tickers
    only_bad_years = bool(bad_years) and not years
    question_type = _classify(
        question,
        n_tickers=len(tickers),
        n_years=len(years),
        named_only_out_of_scope=only_out_of_scope,
        named_only_bad_years=only_bad_years,
        wants_figures=wants_figures,
    )
    because: UnanswerableBecause | None = None
    if question_type == "unanswerable":
        if _asks_for_advice_or_a_prediction(question):
            because = "request"
        elif only_out_of_scope and wants_figures and _names_whose_figure(question, out_of_scope):
            because = "company"
        elif only_out_of_scope or _asks_beyond_the_filing(question):
            because = "topic"
        else:
            because = "year"
    return ParsedQuestion(
        question=question.strip(),
        question_type=question_type,
        tickers=tickers,
        fiscal_years=years,
        unresolved=tuple(out_of_scope) + tuple(bad_years),
        wants_figures=wants_figures,
        search_text=_search_text(question.strip(), tickers, years),
        unanswerable_because=because,
    )


@lru_cache(maxsize=1)
def _scope() -> frozenset[str]:
    """The tickers in config/companies.txt, read once per process."""
    return frozenset(read_tickers())


# --- companies --------------------------------------------------------------

def _alias_pattern(aliases: Iterable[str]) -> re.Pattern[str]:
    """One pattern matching any alias as a whole word or phrase.

    Longest first, so "meta platforms" is tried before "meta" and the match
    covers the whole name. Spaces in an alias match any run of whitespace.
    """
    ordered = sorted(aliases, key=len, reverse=True)
    body = "|".join(re.escape(a).replace(r"\ ", r"\s+") for a in ordered)
    return re.compile(rf"\b(?:{body})\b", re.IGNORECASE)


_IN_SCOPE_PATTERNS = {t: _alias_pattern(a) for t, a in COMPANY_ALIASES.items()}
_OUT_OF_SCOPE_PATTERNS = {n: _alias_pattern(a) for n, a in OUT_OF_SCOPE_ALIASES.items()}


@lru_cache(maxsize=4)
def _ticker_pattern(tickers: frozenset[str]) -> re.Pattern[str]:
    """A ticker written as a ticker. Case-sensitive: upper case only, so
    ordinary words that happen to be tickers ("now" in "how is revenue
    recognised now") do not resolve."""
    body = "|".join(re.escape(t) for t in sorted(tickers, key=len, reverse=True))
    return re.compile(rf"\b({body})\b")


def _companies(question: str, scope: frozenset[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The tickers the question names, in order of first mention, and the names it
    uses for companies the corpus does not hold, as written and in the same order.

    A company the alias table knows but the scope does not -- one dropped from
    companies.txt -- is recognised and then reported, not skipped. Skipping it
    would search the other fourteen filings for a company that is not there
    and say nothing about it, which is the failure the unresolved list exists
    to prevent.
    """
    found: dict[str, int] = {}      # ticker -> position of first mention
    excluded: dict[str, int] = {}   # the mention as written -> its position

    # Every ticker that could be written as one: in scope, or known to the
    # alias table and dropped from it.
    for match in _ticker_pattern(scope | frozenset(COMPANY_ALIASES)).finditer(question):
        ticker = match.group(1)
        if ticker in scope:
            found.setdefault(ticker, match.start())
        else:
            excluded.setdefault(match.group(0), match.start())

    for ticker, pattern in _IN_SCOPE_PATTERNS.items():
        match = pattern.search(question)
        if not match:
            continue
        if ticker in scope:
            found[ticker] = min(found.get(ticker, match.start()), match.start())
        else:
            excluded.setdefault(match.group(0), match.start())

    for pattern in _OUT_OF_SCOPE_PATTERNS.values():
        match = pattern.search(question)
        if match:
            excluded.setdefault(match.group(0), match.start())

    by_position = lambda items: tuple(k for k, _ in sorted(items, key=lambda kv: kv[1]))
    return by_position(found.items()), by_position(excluded.items())


# --- fiscal years -----------------------------------------------------------

def _year_value(match: re.Match[str]) -> int:
    """The year a match names, as four digits."""
    digits = next(v for v in match.groupdict().values() if v is not None)
    return int(digits) if len(digits) == 4 else 2000 + int(digits)


def _is_not_a_year(text: str, match: re.Match[str]) -> bool:
    """Whether a bare four-digit match is a figure, or a year that dates
    something other than a filing.

    Only the bare form is in doubt: "FY2024" and "'24" say they are years.
    "$2024 million" has a currency before it and a magnitude after; either
    alone is enough, since "$2024" and "2024 million" are both figures. A count
    after the number is a figure too, unless a possessive before it makes the
    number date the count: "2000 employees" is a figure, "Meta's 2023
    headcount" is a year. A comparison before it with a plural after it is a
    figure whatever the noun, as in "more than 2000 suppliers". And "Act of
    2022" dates a law, which is no reason to filter the filings.
    """
    if match.group("bare") is None:
        return False
    before, after = text[:match.start()], text[match.end():]
    if (
        _CURRENCY_BEFORE.search(before)
        or _MAGNITUDE_AFTER.match(after)
        or _LAW_BEFORE.search(before)
    ):
        return True
    if _COUNT_AFTER.match(after):
        return _POSSESSIVE_BEFORE.search(before) is None
    return bool(_QUANTITY_BEFORE.search(before) and _PLURAL_AFTER.match(after))


def _years(question: str, bounds: tuple[int, int]) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """The fiscal years the question names, within bounds and in ascending order,
    and the years it named outside them, as written."""
    text = _SHORT_RANGE.sub(r"\1\2 \3 \1\4", question)
    # Figures are dropped before ranges are read, so "$2000 to $2024 million"
    # cannot become a range of years between two amounts.
    matches = [m for m in _YEAR.finditer(text) if not _is_not_a_year(text, m)]
    low, high = bounds

    named: set[int] = set()
    bad: list[str] = []
    for i, match in enumerate(matches):
        year = _year_value(match)
        if not low <= year <= high:
            # Reported as the question wrote it, so "FY15" is shown as "FY15"
            # and not as a 2015 the user never typed.
            if match.group(0) not in bad:
                bad.append(match.group(0))
        named.add(year)

        # A range is two years with a joiner between them and nothing else.
        # Every year inside it is named too, but only the endpoints can be
        # unresolved: the user wrote those, and not the ones in between. The
        # endpoints are sorted first, so "between FY2025 and FY2021" is the
        # same five years as the other way round rather than just the two.
        if i + 1 < len(matches):
            gap = text[match.end():matches[i + 1].start()].strip()
            preceded_by_between = _BETWEEN.search(text[:match.start()]) is not None
            is_range = bool(_RANGE_JOIN.match(gap)) or (gap.lower() == "and" and preceded_by_between)
            start, end = sorted((year, _year_value(matches[i + 1])))
            if is_range and start < end <= start + _MAX_RANGE:
                named.update(range(start, end + 1))

    return tuple(sorted(y for y in named if low <= y <= high)), tuple(bad)


# --- search text ------------------------------------------------------------

# What follows a company name to make it possessive: "Apple's", "Meta Platforms'".
_POSSESSIVE_AFTER = re.compile(r"['’]s\b|['’](?!\w)")
# What follows an alias to make it part of a product or segment name rather
# than the company: "Google Cloud", "Microsoft 365". Not a four-digit number,
# which is a year: "Apple 2024" is Apple in 2024.
_NAME_CONTINUES = re.compile(r"\s+(?:[A-Z]|\d(?!\d{3}\b))")
# The word that introduces a year, or joins two into a range, and means nothing
# once the year is gone: "in FY2024", "from 2022 to 2024", "FY22 - FY24".
_YEAR_LEAD_IN = re.compile(
    r"(?:\b(?:in|for|during|from|to|through|between|and|of)|[-–—])\s*$", re.IGNORECASE
)
# A month and day before a year make it a date, which stays: "December 31,
# 2025" is how a balance sheet heads its column, so it is the one way of
# writing the year that matches a statement table.
_DATE_BEFORE = re.compile(
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d{1,2},?\s*$",
    re.IGNORECASE,
)
_SPACE_BEFORE_PUNCTUATION = re.compile(r"\s+([,.;:?!])")
_REPEATED_PUNCTUATION = re.compile(r"([,;:])(?:\s*[,;:])+")


def _search_text(question: str, tickers: tuple[str, ...], years: tuple[int, ...]) -> str:
    """The question without the companies and years its filters already apply.

    Only what resolved is removed: a company or a year the corpus does not hold
    filters nothing, so its name is still the best thing to search for. A
    ticker, an alias with its possessive, and a year with the word that
    introduces it all go. An alias that names a segment or a product stays
    (see ``constants.SEGMENT_ALIASES``), and so does a year inside a date,
    which is how a statement table writes it. If nothing but punctuation would be
    left, the question is searched as asked.
    """
    # The same rewrite _years reads, so "FY22-24" loses both years.
    text = _SHORT_RANGE.sub(r"\1\2 \3 \1\4", question)
    spans: list[tuple[int, int]] = []

    if tickers:
        for match in _ticker_pattern(frozenset(tickers)).finditer(text):
            spans.append((match.start(), _past_possessive(text, match.end())))
        for ticker in tickers:
            pattern = _IN_SCOPE_PATTERNS.get(ticker)
            for match in pattern.finditer(text) if pattern else ():
                end = match.end()
                if match.group(0).lower() in SEGMENT_ALIASES:
                    continue
                if not _POSSESSIVE_AFTER.match(text, end) and _NAME_CONTINUES.match(text, end):
                    continue
                spans.append((match.start(), _past_possessive(text, end)))

    for match in _YEAR.finditer(text):
        if _is_not_a_year(text, match) or _year_value(match) not in years:
            continue
        if _DATE_BEFORE.search(text, 0, match.start()):
            continue
        lead_in = _YEAR_LEAD_IN.search(text, 0, match.start())
        spans.append((lead_in.start() if lead_in else match.start(), match.end()))

    if not spans:
        return question
    kept, position = [], 0
    for start, end in sorted(spans):
        if start > position:
            kept.append(text[position:start])
        position = max(position, end)
    kept.append(text[position:])
    trimmed = " ".join("".join(kept).split())
    trimmed = _REPEATED_PUNCTUATION.sub(r"\1", _SPACE_BEFORE_PUNCTUATION.sub(r"\1", trimmed))
    return trimmed if re.search(r"\w", trimmed) else question


def _past_possessive(text: str, end: int) -> int:
    """Where a name ends once the possessive after it is included."""
    match = _POSSESSIVE_AFTER.match(text, end)
    return match.end() if match else end


# --- classification ---------------------------------------------------------

def _label_words(text: str) -> str:
    """Normalise punctuation and spacing without matching parts of words."""
    return " ".join(re.findall(r"\w+", text.casefold()))


@lru_cache(maxsize=4)
def _figure_entities(scope: frozenset[str]) -> re.Pattern[str]:
    names = scope | frozenset(COMPANY_ALIASES) | {
        alias for group in (COMPANY_ALIASES, OUT_OF_SCOPE_ALIASES)
        for aliases in group.values() for alias in aliases
    }
    pattern = _alias_pattern(names)
    return re.compile(pattern.pattern + r"(?:['’]s\b|['’](?!\w))?", re.I)


def _figure_text(question: str, scope: frozenset[str]) -> str:
    """Remove only named entities and dates before checking a figure request."""
    text = _figure_entities(scope).sub(" ", question)
    return " ".join(_YEAR.sub(" ", _SHORT_RANGE.sub(r"\1\2 \3 \1\4", text)).split())


# Matched against a question already reduced to its words, so a cue written
# with punctuation ("stock-based", "property, plant and equipment") is reduced
# the same way first.
_FIGURE_METRIC_PATTERN = _alias_pattern(
    (*FIGURE_METRIC_CUES, *(_label_words(cue) for cue in FIGURE_LINE_ITEM_CUES))
)


def _asks_for_label(text: str, pattern: re.Pattern[str]) -> bool:
    """A full line item plus figure-question scaffolding, with no prose topic.

    Even two-word labels can name a topic: "commercial paper program" asks
    about a program, whereas "what was commercial paper" asks for its balance.
    Keep the legacy cues separate so their existing classifications stay put.
    """
    normal = _label_words(text)
    if not pattern.search(normal):
        return False
    return set(pattern.sub(" ", normal).split()).issubset(_FIGURE_SCAFFOLDING)


@lru_cache(maxsize=4)
def _fact_label_pattern(path: Path, mtime_ns: int, size: int) -> re.Pattern[str] | None:
    """Read only the label column and compile once per successful store revision.

    Full labels are cues, not their individual words: "Assets Held for Sale"
    must not turn every mention of "sale" into a request for a figure. The
    metric aliases in constants cover everyday names such as "net sales".
    """
    import pandas as pd

    labels = pd.read_parquet(path, columns=["label"])["label"].dropna().unique()
    cues = {_label_words(label) for label in labels if isinstance(label, str)} - {""}
    return _alias_pattern(cues) if cues else None


def _mentions_fact_label(question: str, facts_file: Path) -> bool:
    try:
        path = Path(facts_file).absolute()
        stat = path.stat()
        pattern = _fact_label_pattern(path, stat.st_mtime_ns, stat.st_size)
    except Exception:
        # Failure is deliberately outside the cached function: a temporary
        # read failure must be retried even if the file's timestamp is unchanged.
        return False
    return pattern is not None and _asks_for_label(question, pattern)


def _any_cue(question: str, cues: Iterable[str]) -> bool:
    """Whether any cue appears as a whole word or phrase, ignoring case.

    Whole words, so "will" is not found in "goodwill" and "amd" is not found
    in "amdahl". A cue with punctuation, like "vs.", is matched literally.
    """
    lowered = question.lower()
    return any(
        re.search(rf"(?<!\w){re.escape(cue)}(?!\w)", lowered) is not None for cue in cues
    )


def _reports_on_the_filing(question: str) -> bool:
    """Whether some reporting verb in the question is the filing's, as above."""
    for match in _REPORTING.finditer(question):
        before = question[:match.start()]
        if _FUTURE_BEFORE.search(before):
            continue
        if match.group(0).lower() in _DESCRIBING_FORMS and _DESCRIBED_BEFORE.search(before):
            continue
        return True
    return False


def _asks_for_advice_or_a_prediction(question: str) -> bool:
    """Whether the question asks the engine for advice or a prediction of its own.

    The two requests no filing answers however they are worded: "based on
    what Apple disclosed, predict next quarter's revenue" still asks for a
    prediction. These are the questions ``answer_question`` refuses unsearched.
    """
    return bool(_any_cue(question, ADVICE_CUES) or _PREDICTION_REQUEST.search(question))


def _names_whose_figure(question: str, names: Iterable[str]) -> bool:
    """Whether one of ``names``, as the question writes them, is the company
    whose figure it asks for: a possessive, the subject of a verb that makes
    the figure its own, or the company after "of".

    "What was Intel's total revenue?", "How many employees did NVIDIA have?"
    and "What was the total revenue of Intel?" ask for an outside company's
    own figure, which no filing in the corpus reports. "Which companies
    reported revenue from Intel as a customer?" and "Did NVIDIA account for
    more than 10% of any company's revenue?" ask the filings the corpus does
    hold about them. A wording this does not list, "Intel revenue in FY2024?",
    is searched, and declining it is left to the model.
    """
    for name in names:
        for match in re.finditer(r"(?<!\w)" + re.escape(name) + r"(?!\w)", question):
            before, end = question[:match.start()], match.end()
            if _POSSESSIVE_AFTER.match(question, end):
                return True
            if _SUBJECT_BEFORE.search(before) and _OWNS_AFTER.match(question, end):
                return True
            of = _OF_BEFORE.search(before)
            if (of and of.group(1).lower() not in RELATION_NOUNS
                    and _PHRASE_ENDS_AFTER.match(question, end)):
                return True
    return False


def _asks_beyond_the_filing(question: str) -> bool:
    """Whether the question asks for something no 10-K can give.

    Three things a filing cannot give: advice, a new prediction, and what it
    does not carry -- a current price, next year. The first two are read this
    way however the question is worded (``_asks_for_advice_or_a_prediction``).
    The third only when the question asks for the thing itself: with a
    reporting verb in it, "what risks did Apple disclose about its stock
    price" is a question about Item 1A, and the noun is just its topic.

    The third reading is a guess from the words, and filings do print some
    share prices and some of next year: ``parse_question`` records it as
    "topic", which is searched, and the first two as "request", which is not.
    """
    if _asks_for_advice_or_a_prediction(question):
        return True
    if _reports_on_the_filing(question):
        return False
    return _any_cue(question, BEYOND_FILING_NOUNS) or _any_cue(question, FUTURE_CUES)


def _classify(
    question: str,
    *,
    n_tickers: int,
    n_years: int,
    named_only_out_of_scope: bool,
    named_only_bad_years: bool,
    wants_figures: bool,
) -> str:
    """One label, from the first rule that fires. The order is the contract:
    see ``constants.QUESTION_TYPES`` for why comparative outranks temporal."""
    if named_only_out_of_scope or named_only_bad_years or _asks_beyond_the_filing(question):
        return "unanswerable"
    if n_tickers >= 2:
        return "comparative"
    if n_years >= 2 or _any_cue(question, TEMPORAL_CUES):
        return "temporal"
    if _any_cue(question, COMPARATIVE_CUES):
        return "comparative"
    if wants_figures:
        return "numeric"
    return "factual"
