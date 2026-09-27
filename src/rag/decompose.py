"""Split a question that asks about several filings into one search per filing.

"How did Apple's AI risk disclosure change between FY2023 and FY2024, and how
does it differ from Microsoft's?" is four questions wearing one coat. Asked as
a single search it returns the eight passages that best match the wording, and
nothing makes those eight cover four filings: the pre-filter admits all four,
the scorer ranks across them, and whichever filing phrases the topic most like
the question takes the whole budget. The answer then compares FY2024 with
FY2024 and calls it a change.

So the question is decomposed. One sub-question per filing the question asks
about, each a copy of the same ``Query`` narrowed to one company and one year,
and the passage budget is shared between them rather than competed for. The
sub-questions' filters are disjoint by construction -- each names a different
company, a different year, or both -- so no passage can be returned by two of
them, which is what makes the merge a plain interleave: reciprocal rank fusion
over disjoint lists reduces to ordering by rank, with none of the agreement
between lists that fusion exists to reward.

What it does not do is decompose by topic. "AI risk" and "supply chain risk"
in one question stay one search, because splitting on meaning needs a model to
do the splitting, and a model that mis-splits sends a confident search after a
question nobody asked. The axes here are the two the corpus is indexed on and
the parser already reads exactly -- company and fiscal year -- so a
decomposition is a fact about the question rather than a guess at it.

On by default, and off with ``use_decomposition=False``, which is the without
half of the ablation row this exists for. It only ever fires on a question
that names more than one company or more than one year; everything else takes
the single search it always took.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
from typing import Any

from ..retrieval.base import resolve_k
from ..retrieval.records import Query, RetrievedPassage
from .constants import MAX_CROSS_SPLIT


@dataclass(frozen=True)
class SubQuestion:
    """One filing's worth of a question: the narrowed query, and what to call it."""

    label: str      # "AAPL FY2024", "MSFT", "FY2023": the filters, as a reader reads them
    query: Query

    @property
    def tickers(self) -> tuple[str, ...]:
        return self.query.tickers

    @property
    def fiscal_years(self) -> tuple[int, ...]:
        return self.query.fiscal_years


@dataclass(frozen=True)
class Decomposition:
    """The sub-questions a question was split into, and the evidence they found.

    ``passages`` is the merged set, re-ranked from 1 so it reads like any other
    result list. ``provenance`` maps each passage's chunk_id to the label of the
    sub-question that returned it, recorded at merge time rather than inferred
    afterwards, so the app can group the evidence by filing and the ablation
    can say which half of a comparison an answer actually rested on.
    """

    sub_questions: tuple[SubQuestion, ...]
    passages: tuple[RetrievedPassage, ...]
    provenance: dict[str, str]

    @property
    def labels(self) -> tuple[str, ...]:
        """What the question was split into, in order."""
        return tuple(sub.label for sub in self.sub_questions)

    def by_sub_question(self) -> dict[str, tuple[RetrievedPassage, ...]]:
        """The merged evidence grouped back under the sub-question that found it.

        Every sub-question appears, including one that found nothing: a
        comparison resting on evidence from only one side of it is a thing the
        reader has to be able to see.
        """
        grouped: dict[str, list[RetrievedPassage]] = {sub.label: [] for sub in self.sub_questions}
        for passage in self.passages:
            grouped[self.provenance[passage.chunk_id]].append(passage)
        return {label: tuple(found) for label, found in grouped.items()}


def label_for(tickers: tuple[str, ...], fiscal_years: tuple[int, ...]) -> str:
    """A sub-question's name, from the filters that make it one.

    Read by a person, so "AAPL FY2024" rather than the filters mapping. Empty
    on both axes cannot happen for a sub-question, since a sub-question exists
    only where something was narrowed.
    """
    parts = [" ".join(tickers)] if tickers else []
    parts += [" ".join(f"FY{year}" for year in fiscal_years)] if fiscal_years else []
    return " ".join(part for part in parts if part)


def decompose(query: Query, *, limit: int = MAX_CROSS_SPLIT) -> tuple[SubQuestion, ...]:
    """The sub-questions a query splits into, or empty where it does not split.

    A query naming one company and one year is already about one filing and is
    returned as nothing to do. Naming several of either splits on that axis;
    naming several of both splits on the pair, which is what the comparison of
    a change over time between two companies actually asks for.

    Two things bound a split, both of them the passage budget. ``limit`` caps
    the pairs, because the pairs are filings nobody named: where two companies
    and five years would be ten, the split falls back to companies alone and
    keeps every year in each sub-question, which is a weaker reading of the
    question than the pairs would be and a stronger one than five years of
    whichever company writes most like the question.

    And no split may leave a sub-question with nothing. A split into more
    filings than there are passages to hand out gives the last of them none,
    which is a comparison quietly missing a side while carrying a label saying
    it has one; a question naming more companies than the budget can cover is
    better served by the single search it would have had.
    """
    tickers, years = query.tickers, query.fiscal_years
    if len(tickers) < 2 and len(years) < 2:
        return ()

    if len(tickers) >= 2 and len(years) >= 2 and len(tickers) * len(years) <= limit:
        pairs = [((ticker,), (year,)) for ticker, year in product(tickers, years)]
    elif len(tickers) >= 2:
        pairs = [((ticker,), years) for ticker in tickers]
    else:
        pairs = [(tickers, (year,)) for year in years]

    if len(pairs) > max(query.top_k, 1):
        return ()

    return tuple(
        SubQuestion(
            label=label_for(sub_tickers, sub_years),
            query=replace(query, tickers=sub_tickers, fiscal_years=sub_years),
        )
        for sub_tickers, sub_years in pairs
    )


def merge(
    results: list[tuple[SubQuestion, list[RetrievedPassage]]], wanted: int
) -> tuple[tuple[RetrievedPassage, ...], dict[str, str]]:
    """Interleave the sub-results, best of each first, down to ``wanted`` passages.

    Round by round: every sub-question's best remaining passage before any
    sub-question's second. That is what shares the budget rather than letting
    one filing win it, and on disjoint result lists it is what reciprocal rank
    fusion would compute anyway. A sub-question that runs out drops out of the
    rounds, so its unused share goes to the others rather than going unused --
    a company the corpus holds fewer passages for should not shrink the
    evidence for the one it holds more of.

    The passages are re-ranked from 1 in merged order. Their scores are left as
    each sub-search set them, and are not comparable across sub-questions: they
    came from searches over different filings, which is the same reason a BM25
    score and a cosine similarity are not compared.
    """
    merged: list[RetrievedPassage] = []
    provenance: dict[str, str] = {}
    queues = [(sub, list(passages)) for sub, passages in results]
    while len(merged) < wanted and any(passages for _, passages in queues):
        for sub, passages in queues:
            if not passages:
                continue
            passage = passages.pop(0)
            if passage.chunk_id in provenance:
                continue
            provenance[passage.chunk_id] = sub.label
            merged.append(passage)
            if len(merged) == wanted:
                break
    return (
        tuple(replace(passage, rank=rank) for rank, passage in enumerate(merged, start=1)),
        provenance,
    )


def search_decomposed(
    query: Query,
    retriever: Any,
    *,
    k: int | None = None,
    limit: int = MAX_CROSS_SPLIT,
) -> Decomposition | None:
    """Search each sub-question and merge the results, or None if there are none.

    None means the question is about one filing and the caller should search it
    the way it always did, rather than this returning a one-element
    decomposition that every caller then has to special-case.

    Each sub-search asks for the whole budget rather than its share of it, so
    the merge chooses from a full ranking per filing instead of from a
    pre-truncated one, and so a sub-question whose neighbours found little can
    contribute more than its share.
    """
    sub_questions = decompose(query, limit=limit)
    if not sub_questions:
        return None
    wanted = resolve_k(query, k)
    results = [(sub, retriever.search(sub.query, k=wanted)) for sub in sub_questions]
    passages, provenance = merge(results, wanted)
    return Decomposition(
        sub_questions=sub_questions, passages=passages, provenance=provenance,
    )


__all__ = [
    "Decomposition",
    "SubQuestion",
    "decompose",
    "label_for",
    "merge",
    "search_decomposed",
]
