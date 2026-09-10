"""The contract every retriever satisfies, and the work all four would repeat.

Objective 2 is a comparison, so the ablation harness runs one question through
BM25, dense, hybrid and reranked retrieval and lines the numbers up. That only
stays a loop if the four are interchangeable. If they are not, the harness forks
once per method, every new configuration is another branch, and the ablation
matrix becomes a rewrite instead of a parameter.

So this module holds the seam:

``Retriever`` is the protocol. A retriever is anything with a ``name`` and a
``search``, which means the harness, the RAG engine and the app hold one type
and never ask which method they were handed. It is a Protocol rather than a base
class deliberately -- nothing has to inherit from anything, so a retriever built
around Chroma and one built around rank_bm25 satisfy the same contract without
being forced into one class hierarchy.

``matches``, ``candidates``, ``rank`` and ``reorder`` are the work the four would
otherwise each write. Every retriever has to apply the metadata filters before it
scores (#22), sort, drop what falls under a floor, cut to k, and number the
results from 1. Four copies of that is four places for the rank numbering or the
tie-break to drift, and a comparison whose rows differ for a reason that has
nothing to do with retrieval.

``WrappingRetriever`` is how reranking composes. A reranker wraps whatever it is
given and re-scores what came back, so there is one reranker rather than a
reranked variant of each method, and ``Reranked(Hybrid(...))`` is a row in the
ablation matrix rather than a class.

Nothing here scores anything or opens an index. This module imports the records,
the constants and ``iter_chunks``; it does not import a retriever, and a
retriever imports it.

A note on the signature. The issue this module answers describes
``search(query, k, filters)``. The filters live on the ``Query`` instead, because
``records.Query`` already carries them and a filter that can arrive through two
routes is a filter that gets dropped on one of them. So a Query is the whole
request -- text, k and where it is allowed to look -- and the ``k`` argument
stays as an override for the one caller that needs it: hybrid and reranking
retrieve deep and return shallow, so they ask their inner retriever for
CANDIDATE_K while the Query the user wrote says 8.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import replace
from typing import Any, Protocol, runtime_checkable

from src.pipeline.chunk import iter_chunks

from .constants import CANDIDATE_K
from .records import Query, RetrievedPassage


@runtime_checkable
class Retriever(Protocol):
    """Anything that can turn a :class:`Query` into ranked passages.

    Structural, so an implementation satisfies this by having the two members
    below and not by inheriting anything. ``isinstance(obj, Retriever)`` works
    at runtime and is what the harness uses to fail early on a misconfigured
    row, rather than half-way through an ablation that has already been running
    for an hour.

    What an implementation promises, beyond the shape:

    - The list is sorted best first and ``rank`` counts from 1 with no gaps,
      because the metrics in #25 read the rank rather than the position.
    - The Query's filters are applied BEFORE scoring, not after (#22).
    - ``retriever`` on every passage returned equals this retriever's ``name``,
      so the harness can attribute a passage without threading its own state.
    - Fewer than k results is a normal answer, not an error. A filtered corpus
      can hold fewer than k passages, and an empty list is what a filter that
      matches nothing should return.

    ``rank`` and ``reorder`` in this module keep the first three of those true
    for whoever uses them.
    """

    # "bm25", "dense", "hybrid", "rerank" -- one of constants.RETRIEVERS, and
    # the value written onto every passage this retriever returns.
    name: str

    def search(self, query: Query, k: int | None = None) -> list[RetrievedPassage]:
        """The best passages for this query, best first.

        ``k`` overrides ``query.top_k`` for this call and is how a wrapping
        retriever asks for candidate depth without editing the caller's Query.
        Leave it as None and the Query decides.
        """
        ...


def resolve_k(query: Query, k: int | None = None) -> int:
    """How many passages this call should return.

    One line, in one place, because the alternative is four retrievers each
    deciding whether an explicit k beats the Query's own -- and the ablation
    quietly comparing a top-8 row against a top-10 one.
    """
    return query.top_k if k is None else k


def matches(chunk: Mapping[str, Any], query: Query) -> bool:
    """Whether one corpus row is inside what the Query allows.

    The full pre-filter, including the two ``iter_chunks`` does not take. That
    gap is the reason this exists: ``iter_chunks`` filters on tickers, fiscal
    years and key sections, so a retriever that hands it a Query and stops there
    searches a corpus wider than the question asked for, silently and with no
    error to notice. Items and content type are checked here, above it, the way
    ``passages.py`` already checks them.

    Empty means unrestricted, matching ``Query``: no tickers means every ticker,
    not no tickers at all.
    """
    if query.tickers and chunk.get("ticker") not in {t.upper() for t in query.tickers}:
        return False
    if query.fiscal_years and chunk.get("fiscal_year") not in set(query.fiscal_years):
        return False
    if query.items and (chunk.get("item") or "").upper() not in {i.upper() for i in query.items}:
        return False
    if query.content_type and chunk.get("content_type") != query.content_type:
        return False
    if query.key_items_only and not chunk.get("is_key_section"):
        return False
    return True


def candidates(query: Query, chunks: Iterable[Mapping[str, Any]] | None = None) -> Iterator[dict]:
    """Every corpus row the Query allows, as ``iter_chunks`` rows.

    This is the searchable corpus for one query: the filter first, the scoring
    inside it. Pass ``chunks`` to filter a corpus already held in memory -- an
    index built once at startup, say -- and leave it out to read from
    ``data/processed/``, which is what a retriever that scores the corpus
    directly wants.

    The ticker and fiscal-year filters are pushed down into ``iter_chunks`` when
    the corpus is read from disk, so a query pinned to one company does not load
    the other fourteen filings' passages only to discard them.
    """
    if chunks is None:
        chunks = iter_chunks(
            tickers=list(query.tickers) or None,
            fiscal_years=list(query.fiscal_years) or None,
            key_items_only=query.key_items_only,
        )
    for chunk in chunks:
        if matches(chunk, query):
            yield dict(chunk)


def _ordered(
    entries: list[tuple[float, str, Any]],
    min_score: float | None,
    k: int | None,
) -> list[tuple[float, str, Any]]:
    """Apply the floor, sort best first, and cut to k.

    The sort breaks ties on ``chunk_id`` rather than leaving them to insertion
    order. Two passages on the same score is not a rare case here -- BM25 gives
    a lot of them, and fused scores collide whenever two passages were found at
    the same ranks -- and a run whose top-8 depends on which order the corpus
    happened to be walked in is a run that cannot be reproduced. The results in
    the report are quoted from these lists.

    ``min_score`` of None applies no floor, which is what every threshold in
    constants.py starts at until #26 measures one.
    """
    kept = [entry for entry in entries if min_score is None or entry[0] >= min_score]
    kept.sort(key=lambda entry: (-entry[0], entry[1]))
    return kept if k is None else kept[:k]


def rank(
    scored: Iterable[tuple[Mapping[str, Any], float]],
    retriever: str,
    k: int | None = None,
    min_score: float | None = None,
    table_boost: float = 1.0,
) -> list[RetrievedPassage]:
    """Turn scored corpus rows into a ranked result set.

    The last step of a first-stage retriever: BM25 and dense both arrive here
    holding ``(chunk, score)`` pairs and leave holding ranked passages, so the
    ordering, the floor, the numbering and the projection into
    ``RetrievedPassage`` happen once rather than twice.

    ``table_boost`` multiplies the score of a table passage, and is the hook
    #22 asks for: a numeric question can lean toward the passages that keep
    figures under their row and column labels. It defaults to off rather than to
    ``constants.TABLE_BOOST``, because a boost belongs to a question that is
    numeric and not to every question a retriever is ever asked -- the caller
    that knows the question is numeric passes ``TABLE_BOOST`` in.
    """
    entries = [
        (
            score * table_boost if chunk.get("content_type") == "table" else score,
            chunk["chunk_id"],
            chunk,
        )
        for chunk, score in scored
    ]
    return [
        RetrievedPassage.from_chunk(chunk, score=score, rank=position, retriever=retriever)
        for position, (score, _, chunk) in enumerate(_ordered(entries, min_score, k), start=1)
    ]


def reorder(
    scored: Iterable[tuple[RetrievedPassage, float]],
    retriever: str,
    k: int | None = None,
    min_score: float | None = None,
) -> list[RetrievedPassage]:
    """Re-rank passages that have already been retrieved, under new scores.

    What a second stage needs: fusion and reranking are handed passages rather
    than corpus rows, and score them on a scale of their own. The passage keeps
    its text and its citation and takes the new score, the new rank and the new
    method's name.

    Provenance survives the rescoring. A passage that arrived with ``sources``
    keeps them, since a fused passage already records which methods surfaced it;
    a passage that arrived without them gets the method that produced it, so
    reranking a plain BM25 result set still leaves a trail back to BM25 after
    ``retriever`` has been overwritten with "rerank".
    """
    entries = [(score, passage.chunk_id, passage) for passage, score in scored]
    return [
        replace(
            passage,
            score=score,
            rank=position,
            retriever=retriever,
            sources=passage.sources or (passage.retriever,),
        )
        for position, (score, _, passage) in enumerate(_ordered(entries, min_score, k), start=1)
    ]


class WrappingRetriever(ABC):
    """A retriever that asks another one for candidates and re-scores them.

    Composition rather than inheritance, and that is the whole point of the
    class. Reranking is one behaviour that applies to every method, so it is
    written once and wrapped around whichever retriever the ablation row names:
    ``Reranked(bm25)`` and ``Reranked(hybrid)`` are two rows of the matrix and
    one class, where a ``RerankedBM25`` and a ``RerankedHybrid`` would be two
    classes drifting apart. It nests, too, since a wrapper is itself a
    ``Retriever``.

    It is also the "broad then narrow" shape from the architecture: ask the
    inner retriever for ``candidate_k``, re-score that set, return the k the
    caller asked for. Retrieving the final k directly measures worse, because
    the passage that answers the question is often outside a first-stage top-8
    and only a second stage that has seen it can pull it up.

    A subclass supplies ``name`` and :meth:`score`. #21 supplies the
    cross-encoder; nothing here loads a model, so this class stays testable with
    a scorer that is three lines long.
    """

    name = "wrapping"

    def __init__(
        self,
        inner: Retriever,
        candidate_k: int = CANDIDATE_K,
        min_score: float | None = None,
    ) -> None:
        self.inner = inner
        self.candidate_k = candidate_k
        self.min_score = min_score

    def search(self, query: Query, k: int | None = None) -> list[RetrievedPassage]:
        """Candidates from the inner retriever, re-scored and cut to k.

        The depth is never below what the caller asked for. Asking for 50
        candidates and being asked for 80 results would otherwise return 50, and
        a wrapper that quietly returns fewer passages than the row beside it is
        the kind of unfairness an ablation cannot see.
        """
        wanted = resolve_k(query, k)
        found = self.inner.search(query, k=max(self.candidate_k, wanted))
        if not found:
            return []
        scores = self.score(query, found)
        return reorder(
            zip(found, scores, strict=True),
            retriever=self.name,
            k=wanted,
            min_score=self.min_score,
        )

    @abstractmethod
    def score(self, query: Query, passages: Sequence[RetrievedPassage]) -> Sequence[float]:
        """Score each candidate against the query, in the order given.

        Returns one score per passage, on whatever scale the wrapper works in --
        a cross-encoder's logits, for instance. Ordering is not this method's
        job: :meth:`search` sorts, applies the floor and renumbers.
        """
        ...


__all__ = [
    "Retriever",
    "WrappingRetriever",
    "candidates",
    "matches",
    "rank",
    "reorder",
    "resolve_k",
]
