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
from src.retrieval.records import Query, RetrievedPassage


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


# --- keyword text ---------------------------------------------------------------

def test_bm25_matches_the_keyword_text_when_one_is_set(bm25):
    keyword = Query("AAA risk factors paragraph 0", keyword_text="Revenue by segment", top_k=5)
    assert [p.chunk_id for p in bm25.search(keyword)] == [
        p.chunk_id for p in bm25.search(Query("Revenue by segment", top_k=5))
    ]


def test_dense_ignores_the_keyword_text(dense):
    query = Query("AAA risk factors paragraph 0", top_k=5)
    keyword = replace(query, keyword_text="Revenue by segment")
    assert [p.chunk_id for p in dense.search(keyword)] == [p.chunk_id for p in dense.search(query)]


# --- words most passages contain --------------------------------------------

STATEMENT = "| Goodwill | 48,568 | 47,937 |"
# Twelve passages of prose that share the words a question is asked in, each
# with words of its own, so that the average word is rare and the floor
# rank_bm25 gives a common one is worth having.
PROSE = [
    f"At the end of the year the board reviewed the value of the {plan} plan "
    f"for {who} and {what}."
    for plan, who, what in (
        ("pension", "retirees", "actuaries"), ("hiring", "engineers", "campuses"),
        ("leasing", "warehouses", "landlords"), ("hedging", "currencies", "forwards"),
        ("licensing", "patents", "royalties"), ("marketing", "campaigns", "agencies"),
        ("sourcing", "suppliers", "components"), ("staffing", "contractors", "shifts"),
        ("pricing", "discounts", "resellers"), ("training", "managers", "courses"),
        ("travel", "airlines", "hotels"), ("audit", "controls", "findings"),
    )
]


def _keyword_index(*texts):
    """A BM25 index over passages with these texts, all from one filing."""
    return BM25Retriever([
        asdict(RetrievedPassage(
            chunk_id=f"0000000001-24-000001_part_ii_item_8_{number:03d}", text=text, score=0.0,
            rank=1, retriever="bm25", ticker="AAA", company="AAA Corp", fiscal_year=2024,
            item="8", title="Financial Statements", url="https://example.test/filing"))
        for number, text in enumerate(texts)
    ])


def test_words_in_most_passages_do_not_outweigh_the_one_that_names_the_line_item():
    """A statement row holds the line item and none of the words around it.

    "the", "of", "at", "end", "year" and "value" are each in twelve of the
    thirteen passages. Scored, as rank_bm25 scores them, the six together
    outweigh "goodwill" and every prose passage ranks above the row that
    answers the question.
    """
    index = _keyword_index(STATEMENT, *PROSE)
    found = index.search(Query("What was the value of goodwill at the end of the year?", top_k=13))
    assert found[0].text == STATEMENT
    assert all(passage.score == 0 for passage in found[1:])


def test_a_question_made_only_of_common_words_is_still_scored_on_them():
    index = _keyword_index(STATEMENT, *PROSE)
    found = index.search(Query("of the", top_k=13))
    assert [passage.text for passage in found[:12] if passage.score > 0] == [
        passage.text for passage in found[:12]]
    assert found[12].text == STATEMENT and found[12].score == 0


def test_the_common_words_are_the_corpus_s_own_not_a_list():
    # "goodwill" in every passage tells none of them apart, and is left out as
    # "the" is; a question about it is then scored on its other words.
    index = _keyword_index(*(f"Goodwill and the {word} review." for word in ("first", "second")),
                           "Goodwill impairment testing.")
    found = index.search(Query("goodwill impairment", top_k=3))
    assert found[0].text == "Goodwill impairment testing."
    assert [passage.score for passage in found[1:]] == [0, 0]


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
