"""Every retriever searches inside the Query's filters, not around them (#22).

One set of assertions, run against BM25, dense and hybrid in turn, so that the
contract on ``base.Retriever`` is checked on the retrievers the ablation
actually runs rather than restated in each of their test files. Each is real:
BM25 is fitted and dense is built over the synthetic corpus from conftest, with
the fake encoder standing in for bge, and hybrid fuses those two.

What is asserted is which passages come back, never which one wins, because
the fake encoder's vectors carry no meaning. The one test about ranking is on
BM25, whose scores do.
"""

from __future__ import annotations

from dataclasses import asdict, replace

import pytest

from src.pipeline.chunk import iter_chunks
from src.retrieval import embed
from src.retrieval.base import Retriever, matches, rank
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.records import Query


@pytest.fixture
def bm25(corpus, tmp_path):
    return BM25Retriever.build(processed_dir=corpus, index_path=tmp_path / "bm25.pkl")


@pytest.fixture
def dense(corpus, tmp_path, fake_model):
    chroma = tmp_path / "chroma"
    embed.build(chroma_dir=chroma, processed_dir=corpus, batch_size=8, sort_window=8)
    return DenseRetriever.load(chroma_dir=chroma, processed_dir=corpus)


@pytest.fixture(params=["bm25", "dense", "hybrid"])
def retriever(request) -> Retriever:
    if request.param == "hybrid":
        return HybridRetriever(request.getfixturevalue("bm25"), request.getfixturevalue("dense"))
    return request.getfixturevalue(request.param)


# Asked for more than the 90-passage corpus holds, so a result set is the whole
# admitted set and not whichever part of it scored highest.
FILTERS = [
    Query("segment revenue risk", top_k=200),
    Query("segment revenue risk", top_k=200, tickers=("aaa",)),
    Query("segment revenue risk", top_k=200, tickers=("AAA", "CCC"), fiscal_years=(2024,)),
    Query("segment revenue risk", top_k=200, items=("1a", "7")),
    Query("segment revenue risk", top_k=200, content_type="table"),
    Query("segment revenue risk", top_k=200, key_items_only=True),
    Query("segment revenue risk", top_k=200, tickers=("BBB",), fiscal_years=(2023,),
          items=("8",), content_type="table"),
]


def test_it_is_a_retriever(retriever):
    assert isinstance(retriever, Retriever)


@pytest.mark.parametrize("query", FILTERS, ids=lambda q: str(q.filters))
def test_the_search_covers_exactly_what_the_filter_admits(retriever, corpus, query):
    found = [passage.chunk_id for passage in retriever.search(query)]
    expected = {row["chunk_id"] for row in iter_chunks(processed_dir=corpus) if matches(row, query)}
    assert len(found) == len(set(found))
    assert set(found) == expected


def test_a_narrow_filter_still_fills_k(retriever):
    """k passages from inside the filter, not the survivors of a global top k."""
    found = retriever.search(Query("risk factors", top_k=5, tickers=("BBB",), fiscal_years=(2023,)))
    assert len(found) == 5
    assert {(passage.ticker, passage.fiscal_year) for passage in found} == {("BBB", 2023)}
    assert [passage.rank for passage in found] == [1, 2, 3, 4, 5]


def test_a_filter_narrower_than_k_returns_what_it_admits(retriever):
    query = Query("segment revenue", top_k=8, tickers=("BBB",), fiscal_years=(2023,),
                  items=("8",), content_type="table")
    found = retriever.search(query)
    assert len(found) == 3
    assert all(passage.content_type == "table" and passage.item == "8" for passage in found)


def test_a_filter_that_admits_nothing_returns_nothing(retriever):
    assert retriever.search(Query("revenue", tickers=("ZZZ",))) == []
    assert retriever.search(Query("revenue", fiscal_years=(1999,))) == []


def test_the_wrong_year_cannot_answer_when_the_year_is_set(bm25):
    """The failure the issue opens with, on the one retriever whose scores mean something.

    The question's wording matches the FY2023 passage best, so unfiltered it wins.
    Pinned to FY2024 it must not appear at all, and the FY2024 twin takes its place.
    """
    text = "AAA risk factors paragraph 0 topic 0 fiscal 2023"
    unfiltered = bm25.search(Query(text, top_k=1))
    assert (unfiltered[0].ticker, unfiltered[0].fiscal_year, unfiltered[0].item) == ("AAA", 2023, "1A")

    pinned = bm25.search(Query(text, top_k=10, tickers=("AAA",), fiscal_years=(2024,)))
    assert {passage.fiscal_year for passage in pinned} == {2024}
    assert (pinned[0].ticker, pinned[0].item) == ("AAA", "1A")


# --- weighting toward tables ------------------------------------------------


# Worded for the prose, so unboosted the tables sit below the top five.
NUMERIC = Query("AAA 2024 risk factors revenue", top_k=5)


@pytest.fixture(params=["bm25", "dense"])
def first_stage(request) -> Retriever:
    return request.getfixturevalue(request.param)


@pytest.mark.parametrize("filters", [{}, {"tickers": ("AAA", "BBB")}], ids=["all", "filtered"])
def test_the_boost_is_applied_before_the_cut_to_k(first_stage, filters):
    """The boosted top k is the top k of every admitted passage, boosted.

    Worked out by hand from an unboosted search over the whole admitted set, so
    a retriever that boosted only the k it had already cut -- the easy mistake
    for dense, where Chroma does the cutting -- returns a different list.
    """
    query = replace(NUMERIC, **filters)
    everything = first_stage.search(replace(query, top_k=200))
    expected = rank(((asdict(p), p.score) for p in everything),
                    retriever=first_stage.name, k=5, table_boost=3.0)

    found = first_stage.search(replace(query, table_boost=3.0))
    assert [p.chunk_id for p in found] == [p.chunk_id for p in expected]
    assert [p.score for p in found] == pytest.approx([p.score for p in expected])

    unboosted = {p.chunk_id for p in everything[:5]}
    lifted = [p for p in found if p.chunk_id not in unboosted]
    assert lifted and all(p.content_type == "table" for p in lifted)


def test_the_boost_leaves_the_filter_alone(retriever, corpus):
    query = replace(NUMERIC, top_k=200, tickers=("CCC",), items=("7", "8"), table_boost=3.0)
    expected = {row["chunk_id"] for row in iter_chunks(processed_dir=corpus) if matches(row, query)}
    assert {p.chunk_id for p in retriever.search(query)} == expected


def test_hybrid_fuses_lists_already_ordered_under_the_boost(bm25, dense):
    hybrid = HybridRetriever(bm25, dense)
    plain = hybrid.search(NUMERIC)
    boosted = hybrid.search(replace(NUMERIC, table_boost=1000.0))
    assert any(p.content_type == "prose" for p in plain)
    assert all(p.content_type == "table" for p in boosted)
