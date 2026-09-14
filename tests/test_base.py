"""The shared pre-filter and ranking in base.py (#22), on rows built by hand.

``matches`` is the one definition of what a Query allows. BM25 calls it on
rows, ``embed.where_for`` is pinned to it in ``test_embed``, and
``test_prefilter`` checks every retriever against it, so a mistake here is a
mistake in all of them at once -- which is why each filter is tested on its own
before any retriever is.
"""

from __future__ import annotations

import pytest

from src.pipeline.chunk import iter_chunks
from src.retrieval.base import WrappingRetriever, candidates, matches, rank
from src.retrieval.constants import PREFILTER_FIELDS
from src.retrieval.records import Query


def row(**fields):
    """One iter_chunks-shaped row: Apple FY2024 Item 1A prose unless told otherwise."""
    return {
        "chunk_id": "c", "text": "t", "ticker": "AAPL", "company": "Apple",
        "fiscal_year": 2024, "item": "1A", "title": "Risk Factors", "url": "u",
        "content_type": "prose", "is_key_section": True, **fields,
    }


# --- matches ----------------------------------------------------------------


def test_a_query_without_filters_admits_every_row():
    unrestricted = Query("q")
    assert matches(row(), unrestricted)
    assert matches(row(item=None, fiscal_year=None, is_key_section=False), unrestricted)


@pytest.mark.parametrize(
    ("query", "admitted", "excluded"),
    [
        (Query("q", tickers=("aapl",)), row(), row(ticker="MSFT")),
        (Query("q", fiscal_years=("2024",)), row(), row(fiscal_year=2023)),
        (Query("q", fiscal_years=(2024,)), row(), row(fiscal_year=None)),
        (Query("q", items=("1a",)), row(), row(item="7")),
        (Query("q", items=("1A",)), row(), row(item=None)),
        (Query("q", content_type="table"), row(content_type="table"), row()),
        (Query("q", content_type="prose"), row(), row(content_type="table")),
        (Query("q", key_items_only=True), row(), row(is_key_section=False)),
    ],
    ids=["ticker", "fiscal_year", "no_fiscal_year", "item", "no_item",
         "table", "prose", "is_key_section"],
)
def test_each_filter_admits_and_excludes(query, admitted, excluded):
    assert matches(admitted, query)
    assert not matches(excluded, query)


def test_values_within_a_filter_are_alternatives():
    query = Query("q", tickers=("AAPL", "MSFT"), items=("1A", "7"))
    assert matches(row(ticker="MSFT", item="7"), query)


def test_filters_on_different_fields_must_all_hold():
    """The right company in the wrong year is the case the pre-filter exists for."""
    query = Query("q", tickers=("AAPL",), fiscal_years=(2024,))
    assert matches(row(), query)
    assert not matches(row(fiscal_year=2023), query)
    assert not matches(row(ticker="MSFT"), query)


def test_every_prefilter_field_can_be_set_from_a_query():
    """The fields #22 names are the fields a Query can restrict, no more and no fewer."""
    every = Query("q", tickers=("A",), fiscal_years=(2024,), items=("1",),
                  content_type="table", key_items_only=True)
    assert set(every.filters) == set(PREFILTER_FIELDS)


# --- candidates -------------------------------------------------------------


QUERIES = [
    Query("q"),
    Query("q", tickers=("aaa",)),
    Query("q", tickers=("AAA", "CCC"), fiscal_years=(2024,)),
    Query("q", items=("1a", "7")),
    Query("q", content_type="table"),
    Query("q", key_items_only=True),
    Query("q", tickers=("BBB",), fiscal_years=(2023,), items=("8",), content_type="table"),
]


@pytest.mark.parametrize("query", QUERIES, ids=lambda q: str(q.filters))
def test_candidates_read_from_disk_are_the_rows_matches_admits(corpus, query):
    """Pushing ticker and year down into iter_chunks must not change the answer.

    Item and content type are not parameters of iter_chunks, so this is also the
    test that they are still applied above it rather than dropped on the way.
    """
    found = [chunk["chunk_id"] for chunk in candidates(query, processed_dir=corpus)]
    expected = [chunk["chunk_id"] for chunk in iter_chunks(processed_dir=corpus)
                if matches(chunk, query)]
    assert found == expected
    assert found or query.filters


def test_candidates_filter_rows_already_in_memory_and_copy_them():
    held = [row(chunk_id="a"), row(chunk_id="b", item="7")]
    found = list(candidates(Query("q", items=("7",)), chunks=held))
    assert [chunk["chunk_id"] for chunk in found] == ["b"]
    found[0]["text"] = "edited"
    assert held[1]["text"] == "t"


# --- rank and the table boost -----------------------------------------------


def test_without_a_boost_tables_and_prose_rank_on_their_scores():
    ranked = rank([(row(chunk_id="prose"), 1.0), (row(chunk_id="table", content_type="table"), 0.9)],
                  retriever="bm25")
    assert [p.chunk_id for p in ranked] == ["prose", "table"]
    assert [p.rank for p in ranked] == [1, 2]


def test_a_boost_lifts_a_table_passage_and_leaves_prose_alone():
    ranked = rank([(row(chunk_id="prose"), 1.0), (row(chunk_id="table", content_type="table"), 0.9)],
                  retriever="bm25", table_boost=1.5)
    assert [p.chunk_id for p in ranked] == ["table", "prose"]
    assert [p.score for p in ranked] == pytest.approx([1.35, 1.0])
    assert ranked[0].content_type == "table"


def test_a_boost_lifts_a_negative_score_rather_than_sinking_it():
    """A cosine similarity can be below zero, where multiplying pushes it down."""
    ranked = rank([(row(chunk_id="prose"), -0.25), (row(chunk_id="table", content_type="table"), -0.3)],
                  retriever="dense", table_boost=1.5)
    assert [p.chunk_id for p in ranked] == ["table", "prose"]
    assert ranked[0].score == pytest.approx(-0.2)


def test_the_floor_applies_to_the_boosted_score():
    ranked = rank([(row(chunk_id="table", content_type="table"), 0.4)],
                  retriever="bm25", min_score=0.5, table_boost=1.5)
    assert [p.chunk_id for p in ranked] == ["table"]


@pytest.mark.parametrize("boost", [0.0, -1.0])
def test_a_boost_that_is_not_positive_is_refused(boost):
    with pytest.raises(ValueError, match="table_boost"):
        rank([(row(), 1.0)], retriever="bm25", table_boost=boost)


# --- the boost through a second stage ---------------------------------------


class Inner:
    """A first stage that returns one prose and one table passage, prose first."""

    name = "bm25"

    def search(self, query, k=None):
        return rank([(row(chunk_id="prose"), 2.0), (row(chunk_id="table", content_type="table"), 1.0)],
                    retriever=self.name, k=k)


class Logits(WrappingRetriever):
    """Scores the way a cross-encoder can: below zero, prose ahead of the table."""

    name = "rerank"

    def score(self, query, passages):
        return [-1.0 if passage.content_type == "table" else -0.8 for passage in passages]


def test_a_second_stage_applies_the_querys_boost_to_its_own_scores():
    """Re-scoring throws the first stage's boost away, so the wrapper applies it again."""
    plain = Logits(Inner()).search(Query("q", top_k=2))
    boosted = Logits(Inner()).search(Query("q", top_k=2, table_boost=2.0))
    assert [p.chunk_id for p in plain] == ["prose", "table"]
    assert [p.chunk_id for p in boosted] == ["table", "prose"]
    assert boosted[0].score == pytest.approx(-0.5)
