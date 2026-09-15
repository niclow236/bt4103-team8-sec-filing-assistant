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
rules are lists in ``constants.py``, so extending them is editing a table.

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
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from ..config import read_tickers
from ..pipeline.constants import DEFAULT_FISCAL_YEARS
from ..retrieval.constants import TABLE_BOOST
from ..retrieval.records import Query
from .constants import (
    COMPANY_ALIASES,
    COMPARATIVE_CUES,
    NUMERIC_CUES,
    OUT_OF_SCOPE_ALIASES,
    QUESTION_TYPES,
    TEMPORAL_CUES,
    UNANSWERABLE_CUES,
)

# A year as a question writes it. Four alternatives, tried in this order:
#   FY2024, FY24, FYE2024      -- a fiscal prefix, then two or four digits
#   fiscal 2024, fiscal year 24
#   '24                        -- an apostrophe is enough to mark a two-digit year
#   2024                       -- a bare number, but only one that looks like a year
# A bare two-digit number is never a year: "24" in "24 percent" is a figure.
_YEAR = re.compile(
    r"\b(?:fye?)\s?(?P<fy>\d{4}|\d{2})\b"
    r"|\bfiscal(?:\s+year)?\s+(?P<fiscal>\d{4}|\d{2})\b"
    r"|'(?P<apos>\d{2})\b"
    r"|\b(?P<bare>(?:19|20)\d{2})\b",
    re.IGNORECASE,
)

# "FY22-24" and "FY22 to 24": the second year borrows the first one's prefix.
# Rewritten to "FY22 - FY24" before scanning, so the scanner sees two years.
_SHORT_RANGE = re.compile(
    r"\b(fye?\s?)(\d{2})\s*(-|–|—|to|through)\s*(\d{2})\b", re.IGNORECASE
)

# What can sit between two years to make them a range rather than a pair.
# "and" is only a range after "between": "in 2022 and 2024" is two years, and
# "between 2022 and 2024" is three.
_RANGE_JOIN = re.compile(r"^(?:-|–|—|to|through|thru|until|till)$", re.IGNORECASE)
_BETWEEN = re.compile(r"\bbetween\s*$", re.IGNORECASE)

# The widest range a question is allowed to expand to. Wider than the corpus,
# and there to stop "1999 to 2024" turning into twenty-six filters.
_MAX_RANGE = 10


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

    def __post_init__(self) -> None:
        if self.question_type not in QUESTION_TYPES:
            raise ValueError(
                f"question_type must be one of {', '.join(QUESTION_TYPES)}, "
                f"got {self.question_type!r}"
            )
        object.__setattr__(self, "tickers", tuple(self.tickers))
        object.__setattr__(self, "fiscal_years", tuple(self.fiscal_years))
        object.__setattr__(self, "unresolved", tuple(self.unresolved))

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

    def describe(self) -> tuple[str, ...]:
        """One line per thing the parser decided, for showing under the answer.

        Written for a person rather than a log: "Companies: AAPL, MSFT" rather
        than the filters mapping. The unresolved line is the one that earns
        its place, since it is the only way the user learns that the answer
        ignored a company they asked about.
        """
        lines = [f"Question type: {self.question_type}"]
        if self.tickers:
            lines.append("Companies: " + ", ".join(self.tickers))
        if self.fiscal_years:
            lines.append("Fiscal years: " + ", ".join(f"FY{y}" for y in self.fiscal_years))
        if not self.tickers and not self.fiscal_years:
            lines.append("Filters: none, searching every company and year")
        if self.unresolved:
            lines.append("Not in the corpus: " + ", ".join(self.unresolved))
        return tuple(lines)

    def to_query(self, *, top_k: int | None = None) -> Query:
        """The retrieval request for this question.

        The filters go on as read. The table boost goes on when the question
        wants a figure, and only then: ``constants.TABLE_BOOST`` is documented
        as the value to set for a numeric question and never as a default, so
        a prose question is scored with the boost off.
        """
        fields: dict[str, Any] = {
            "text": self.question,
            "tickers": self.tickers,
            "fiscal_years": self.fiscal_years,
            "table_boost": TABLE_BOOST if self.wants_figures else 1.0,
        }
        if top_k is not None:
            fields["top_k"] = top_k
        return Query(**fields)


def parse_question(
    question: str,
    *,
    known_tickers: Iterable[str] | None = None,
    fiscal_years: tuple[int, int] = DEFAULT_FISCAL_YEARS,
) -> ParsedQuestion:
    """Read one question into a ParsedQuestion.

    ``known_tickers`` is the scope the companies resolve against, defaulting to
    config/companies.txt; a test passes its own. ``fiscal_years`` is the
    inclusive range a year has to fall in to become a filter.
    """
    if not question or not question.strip():
        raise ValueError("question must not be blank")
    scope = frozenset(t.upper() for t in known_tickers) if known_tickers is not None else _scope()

    tickers, out_of_scope = _companies(question, scope)
    years, bad_years = _years(question, fiscal_years)
    wants_figures = _any_cue(question, NUMERIC_CUES)

    question_type = _classify(
        question,
        n_tickers=len(tickers),
        n_years=len(years),
        named_only_out_of_scope=bool(out_of_scope) and not tickers,
        named_only_bad_years=bool(bad_years) and not years,
        wants_figures=wants_figures,
    )
    return ParsedQuestion(
        question=question.strip(),
        question_type=question_type,
        tickers=tickers,
        fiscal_years=years,
        unresolved=tuple(out_of_scope) + tuple(bad_years),
        wants_figures=wants_figures,
    )


def build_query(question: str, *, top_k: int | None = None, **options: Any) -> Query:
    """Parse a question and return only its Query, for a caller that wants nothing else.

    ``options`` are passed to :func:`parse_question`.
    """
    return parse_question(question, **options).to_query(top_k=top_k)


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
def _ticker_pattern(scope: frozenset[str]) -> re.Pattern[str]:
    """A ticker written as a ticker, for any ticker in scope. Case-sensitive:
    upper case only, so ordinary words that happen to be tickers ("now" in
    "how is revenue recognised now") do not resolve."""
    body = "|".join(re.escape(t) for t in sorted(scope, key=len, reverse=True))
    return re.compile(rf"\b({body})\b")


def _companies(question: str, scope: frozenset[str]) -> tuple[tuple[str, ...], list[str]]:
    """The tickers the question names, in order of first mention, and the names it
    uses for companies the corpus does not hold."""
    found: dict[str, int] = {}   # ticker -> position of first mention

    if scope:
        for match in _ticker_pattern(scope).finditer(question):
            found.setdefault(match.group(1), match.start())

    for ticker, pattern in _IN_SCOPE_PATTERNS.items():
        if ticker not in scope:
            continue
        match = pattern.search(question)
        if match:
            found[ticker] = min(found.get(ticker, match.start()), match.start())

    out_of_scope = [
        name for name, pattern in _OUT_OF_SCOPE_PATTERNS.items() if pattern.search(question)
    ]
    ordered = tuple(t for t, _ in sorted(found.items(), key=lambda kv: kv[1]))
    return ordered, out_of_scope


# --- fiscal years -----------------------------------------------------------

def _year_value(match: re.Match[str]) -> int:
    """The year a match names, as four digits."""
    digits = next(v for v in match.groupdict().values() if v is not None)
    return int(digits) if len(digits) == 4 else 2000 + int(digits)


def _years(question: str, bounds: tuple[int, int]) -> tuple[tuple[int, ...], list[str]]:
    """The fiscal years the question names, within bounds and in ascending order,
    and the years it named outside them, as written."""
    text = _SHORT_RANGE.sub(r"\1\2 \3 \1\4", question)
    matches = list(_YEAR.finditer(text))
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
        # unresolved: the user wrote those, and not the ones in between.
        if i + 1 < len(matches):
            gap = text[match.end():matches[i + 1].start()].strip()
            preceded_by_between = _BETWEEN.search(text[:match.start()]) is not None
            is_range = bool(_RANGE_JOIN.match(gap)) or (gap.lower() == "and" and preceded_by_between)
            end = _year_value(matches[i + 1])
            if is_range and year < end <= year + _MAX_RANGE:
                named.update(range(year, end + 1))

    return tuple(sorted(y for y in named if low <= y <= high)), bad


# --- classification ---------------------------------------------------------

def _any_cue(question: str, cues: Iterable[str]) -> bool:
    """Whether any cue appears as a whole word or phrase, ignoring case.

    Whole words, so "will" is not found in "goodwill" and "amd" is not found
    in "amdahl". A cue with punctuation, like "vs.", is matched literally.
    """
    lowered = question.lower()
    return any(
        re.search(rf"(?<!\w){re.escape(cue)}(?!\w)", lowered) is not None for cue in cues
    )


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
    if named_only_out_of_scope or named_only_bad_years or _any_cue(question, UNANSWERABLE_CUES):
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
