"""The dense retriever: the filter, the ranking, and the stale-index refusal.

Every test builds a real index over the synthetic corpus with the deterministic
fake encoder from conftest, then searches it, so the whole file runs in seconds
and needs no model. The fake's vectors carry no meaning -- they are seeded from
the text -- so nothing here asserts that a semantically right passage wins.
What is asserted is the machinery around the scoring: that the filter selects
the same passages BM25's does, that the ranking is ordered and numbered the way
``base`` promises, that a distance comes back as a similarity, and that an
index the corpus has moved past is refused with the reason named.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import edit_filing, vector_for
from src.pipeline.chunk import iter_chunks
from src.retrieval import embed
from src.retrieval.base import Retriever, matches
from src.retrieval.dense import DenseRetriever, similarity_of
from src.retrieval.records import Query, RetrievedPassage


def build(corpus, chroma, **kwargs):
    return embed.build(chroma_dir=chroma, processed_dir=corpus,
                       batch_size=8, sort_window=8, **kwargs)


@pytest.fixture
def chroma(tmp_path):
    return tmp_path / "chroma"


@pytest.fixture
def retriever(corpus, chroma, fake_model):
    """A loaded retriever over a freshly built, current index."""
    build(corpus, chroma)
    return DenseRetriever.load(chroma_dir=chroma, processed_dir=corpus)


# --- the contract -----------------------------------------------------------


def test_it_satisfies_the_retriever_protocol(retriever):
    """The ablation harness holds this the same way it holds BM25."""
    assert isinstance(retriever, Retriever)
    assert retriever.name == "dense"


def test_search_returns_retrieved_passages(retriever):
    found = retriever.search(Query("revenue by segment", top_k=5))
    assert found and all(isinstance(passage, RetrievedPassage) for passage in found)
    assert all(passage.retriever == "dense" for passage in found)
    assert all(passage.text and passage.ticker and passage.url for passage in found)


def test_results_are_ordered_and_numbered_from_one(retriever):
    found = retriever.search(Query("risk factors", top_k=8))
    assert [passage.rank for passage in found] == list(range(1, len(found) + 1))
    assert [p.score for p in found] == sorted((p.score for p in found), reverse=True)


def test_k_overrides_the_query(retriever):
    assert len(retriever.search(Query("anything", top_k=10), k=3)) == 3


def test_k_of_zero_returns_nothing_without_touching_the_index(retriever):
    assert retriever.search(Query("anything", top_k=0)) == []


def test_asking_for_more_than_the_corpus_holds_is_not_an_error(retriever, corpus):
    held = len(list(iter_chunks(processed_dir=corpus)))
    assert len(retriever.search(Query("anything", top_k=held + 50))) == held


# --- the filter -------------------------------------------------------------


FILTERS = [
    Query("segment revenue", top_k=200),
    Query("segment revenue", top_k=200, tickers=("AAA",)),
    Query("segment revenue", top_k=200, tickers=("aaa", "ccc"), fiscal_years=(2024,)),
    Query("segment revenue", top_k=200, items=("1a", "7")),
    Query("segment revenue", top_k=200, content_type="table"),
    Query("segment revenue", top_k=200, key_items_only=True),
    Query("segment revenue", top_k=200, tickers=("BBB",), fiscal_years=(2023,),
          items=("8",), content_type="table"),
]


@pytest.mark.parametrize("query", FILTERS, ids=lambda q: str(q.filters))
def test_the_filter_selects_what_base_matches_selects(retriever, corpus, query):
    """The pre-filter (#22), and the same corpus BM25 would have scored.

    Asked for more passages than the corpus holds, so this compares the whole
    admitted set rather than whichever k of it scored highest.
    """
    found = {passage.chunk_id for passage in retriever.search(query)}
    expected = {row["chunk_id"] for row in iter_chunks(processed_dir=corpus)
                if matches(row, query)}
    assert found == expected


def test_a_filter_that_matches_nothing_returns_nothing(retriever):
    assert retriever.search(Query("x", tickers=("ZZZ",))) == []


def test_the_filter_is_applied_before_scoring_not_after(retriever, corpus):
    """A filtered search returns k passages, not what survives filtering k.

    The failure this pins: taking the global top k and dropping the disallowed
    ones leaves fewer than k, and can leave none at all.
    """
    narrow = Query("segment revenue", top_k=5, tickers=("BBB",))
    found = retriever.search(narrow)
    assert len(found) == 5
    assert {passage.ticker for passage in found} == {"BBB"}


# --- scores -----------------------------------------------------------------


def test_the_score_is_a_similarity_not_a_distance(corpus, chroma, fake_model, monkeypatch):
    """The nearest passage scores highest, and a cosine distance is inverted.

    The encoder is pinned to one stored passage's own vector, so that passage is
    at distance 0 and must come back first with a score of 1. Left as a
    distance, the score would be 0 and it would rank last.
    """
    build(corpus, chroma)
    rows = list(iter_chunks(processed_dir=corpus))
    target = rows[7]
    pinned = vector_for(embed.embed_text(target))

    class Pinned:
        def encode(self, texts, **kwargs):
            return np.stack([pinned for _ in texts])

    monkeypatch.setattr(embed, "_load_model", lambda threads=None: (Pinned(), 1))
    found = DenseRetriever.load(chroma_dir=chroma, processed_dir=corpus).search(
        Query("ignored, the encoder is pinned", top_k=3)
    )
    assert found[0].chunk_id == target["chunk_id"]
    assert found[0].score == pytest.approx(1.0, abs=1e-5)


def test_similarity_of_inverts_cosine_distance():
    assert similarity_of(0.0) == 1.0
    assert similarity_of(1.0) == 0.0
    assert similarity_of(2.0) == -1.0


def test_a_floor_drops_what_falls_under_it(corpus, chroma, fake_model):
    build(corpus, chroma)
    loose = DenseRetriever.load(chroma_dir=chroma, processed_dir=corpus)
    strict = DenseRetriever.load(chroma_dir=chroma, processed_dir=corpus, min_score=2.0)
    assert loose.search(Query("anything", top_k=5))
    assert strict.search(Query("anything", top_k=5)) == []


# --- the stale-index guard --------------------------------------------------


def test_a_current_index_loads(corpus, chroma, fake_model):
    build(corpus, chroma)
    assert DenseRetriever.load(chroma_dir=chroma, processed_dir=corpus).manifest is not None


def test_it_refuses_an_index_the_corpus_has_moved_past(corpus, chroma, fake_model):
    """Re-chunking without re-embedding is the failure the manifest exists for."""
    build(corpus, chroma)
    edited = sorted(corpus.glob("*/*.json"))[0]
    edit_filing(edited, lambda data: data["chunks"][0].update(
        {"text": "Rewritten after the index was built."}
    ))
    with pytest.raises(ValueError) as error:
        DenseRetriever.load(chroma_dir=chroma, processed_dir=corpus)
    assert "does not match the corpus" in str(error.value)


def test_the_refusal_names_a_model_mismatch(corpus, chroma, fake_model, monkeypatch):
    build(corpus, chroma)
    monkeypatch.setattr(embed, "EMBED_MODEL", "some-other/encoder")
    with pytest.raises(ValueError) as error:
        DenseRetriever.load(chroma_dir=chroma, processed_dir=corpus)
    assert "model" in str(error.value) and "some-other/encoder" in str(error.value)


def test_the_refusal_names_a_dimension_mismatch(corpus, chroma, fake_model, monkeypatch):
    build(corpus, chroma)
    monkeypatch.setattr(embed, "EMBED_DIMENSIONS", 1024)
    with pytest.raises(ValueError) as error:
        DenseRetriever.load(chroma_dir=chroma, processed_dir=corpus)
    assert "dimensions" in str(error.value) and "1024" in str(error.value)


def test_the_refusal_names_a_missing_manifest(corpus, chroma, fake_model):
    build(corpus, chroma)
    embed.manifest_file_for(chroma).unlink()
    with pytest.raises(ValueError) as error:
        DenseRetriever.load(chroma_dir=chroma, processed_dir=corpus)
    assert "no manifest" in str(error.value)


def test_it_refuses_when_there_is_no_index_at_all(tmp_path, corpus):
    with pytest.raises(ValueError) as error:
        DenseRetriever.load(chroma_dir=tmp_path / "nothing", processed_dir=corpus)
    assert "python -m src.retrieval embed" in str(error.value)


def test_verify_false_loads_an_index_the_corpus_has_moved_past(corpus, chroma, fake_model):
    """The escape hatch a test with a hand-built index needs, and nothing else."""
    build(corpus, chroma)
    edited = sorted(corpus.glob("*/*.json"))[0]
    edit_filing(edited, lambda data: data["chunks"][0].update({"text": "Rewritten."}))
    loaded = DenseRetriever.load(chroma_dir=chroma, processed_dir=corpus, verify=False)
    assert loaded.search(Query("anything", top_k=3))
