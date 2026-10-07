import pytest

from src.pipeline.chunk import iter_chunks
from src.retrieval import embed
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.constants import CANDIDATE_K, RRF_K
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.records import Query, RetrievedPassage


def passage(chunk_id: str, retriever: str, rank: int) -> RetrievedPassage:
    return RetrievedPassage(
        chunk_id=chunk_id, text=chunk_id, score=1.0, rank=rank,
        retriever=retriever, ticker="AAPL", company="Apple", fiscal_year=2024,
        item="7", title="MD&A", url="https://example.com",
    )


class FakeRetriever:
    def __init__(self, name, results):
        self.name = name
        self.results = results
        self.calls = []

    def search(self, query, k=None):
        self.calls.append(k)
        return self.results


def test_hybrid_fuses_results_and_preserves_sources():
    bm25 = FakeRetriever("bm25", [passage("both", "bm25", 1), passage("bm25-only", "bm25", 2)])
    dense = FakeRetriever("dense", [passage("both", "dense", 1), passage("dense-only", "dense", 2)])

    results = HybridRetriever(bm25, dense).search(Query("revenue"), k=3)

    assert bm25.calls == [CANDIDATE_K]
    assert dense.calls == [CANDIDATE_K]
    assert [item.chunk_id for item in results] == ["both", "bm25-only", "dense-only"]
    assert results[0].sources == ("bm25", "dense")
    assert results[0].retriever == "hybrid"
    assert [item.rank for item in results] == [1, 2, 3]


# --- the order fusion gives (#49) ------------------------------------------------
#
# Reciprocal rank fusion scores a passage sum(weight / (RRF_K + rank)) over the
# methods that returned it, and RRF_K is 60. The expected orders below are
# worked out by hand from that, so a change to the formula, the constant, the
# tie-break or the cut shows up as a test that names what moved.


def listed(name: str, *chunk_ids: str) -> FakeRetriever:
    """A method that returns these passages in this order."""
    return FakeRetriever(name, [passage(chunk_id, name, rank)
                                for rank, chunk_id in enumerate(chunk_ids, start=1)])


def test_the_constant_the_expected_orders_are_worked_out_from():
    assert RRF_K == 60 and CANDIDATE_K == 50


def test_a_fused_score_is_the_sum_of_reciprocal_ranks():
    bm25 = listed("bm25", "a", "b", "c")
    dense = listed("dense", "c", "a", "d")

    results = HybridRetriever(bm25, dense).search(Query("revenue"))

    scores = {item.chunk_id: item.score for item in results}
    assert scores == pytest.approx({
        "a": 1 / 61 + 1 / 62,   # first in one list, second in the other
        "c": 1 / 63 + 1 / 61,   # third and first
        "b": 1 / 62,            # second, in one list only
        "d": 1 / 63,            # third, in one list only
    })
    assert [item.chunk_id for item in results] == ["a", "c", "b", "d"]
    assert [item.rank for item in results] == [1, 2, 3, 4]


def test_a_passage_both_methods_found_outranks_one_that_tops_a_single_list():
    """Third in both lists is 2/63, about 0.0317; first in one is 1/61, about 0.0164."""
    bm25 = listed("bm25", "bm25-best", "x1", "agreed")
    dense = listed("dense", "dense-best", "x2", "agreed")

    results = HybridRetriever(bm25, dense).search(Query("revenue"))

    assert results[0].chunk_id == "agreed"
    assert results[0].score == pytest.approx(2 / 63)
    assert results[0].sources == ("bm25", "dense")
    assert {results[1].chunk_id, results[2].chunk_id} == {"bm25-best", "dense-best"}


def test_passages_on_the_same_score_are_ordered_by_chunk_id():
    """Two passages found at the same rank by one method each tie at 1/61, whatever
    order the methods were asked in, so the id decides and a rerun gives the same list."""
    forwards = HybridRetriever(listed("bm25", "m", "b"), listed("dense", "a", "z"))
    backwards = HybridRetriever(listed("bm25", "a", "z"), listed("dense", "m", "b"))

    expected = ["a", "m", "b", "z"]
    assert [item.chunk_id for item in forwards.search(Query("revenue"))] == expected
    assert [item.chunk_id for item in backwards.search(Query("revenue"))] == expected


def test_the_cut_to_k_comes_after_fusion_not_before():
    """A passage outside each method's own top 2 is still the best of the fused list."""
    bm25 = listed("bm25", "b1", "b2", "agreed")
    dense = listed("dense", "d1", "d2", "agreed")

    results = HybridRetriever(bm25, dense).search(Query("revenue"), k=2)

    assert [item.chunk_id for item in results] == ["agreed", "b1"]
    assert [item.rank for item in results] == [1, 2]


def test_each_method_is_asked_for_the_candidate_depth_whatever_k_is():
    bm25, dense = listed("bm25", "a"), listed("dense", "a")
    hybrid = HybridRetriever(bm25, dense)

    hybrid.search(Query("revenue", top_k=3))
    hybrid.search(Query("revenue"), k=CANDIDATE_K + 30)

    # A k past the candidate depth is asked for in full, since a fused top 80
    # cannot be built from two lists of 50.
    assert bm25.calls == dense.calls == [CANDIDATE_K, CANDIDATE_K + 30]


@pytest.mark.parametrize("k", [0, -1])
def test_a_k_of_nothing_searches_nothing(k):
    bm25, dense = listed("bm25", "a"), listed("dense", "a")
    assert HybridRetriever(bm25, dense).search(Query("revenue"), k=k) == []
    assert bm25.calls == dense.calls == []


def test_weights_scale_each_methods_reciprocal_ranks():
    bm25, dense = listed("bm25", "keyword", "shared"), listed("dense", "vector", "shared")

    equal = HybridRetriever(bm25, dense).search(Query("revenue"))
    leaning = HybridRetriever(bm25, dense, weights={"bm25": 0.3, "dense": 1.0}).search(
        Query("revenue"))

    assert [item.chunk_id for item in equal] == ["shared", "keyword", "vector"]
    assert [item.chunk_id for item in leaning] == ["shared", "vector", "keyword"]
    scores = {item.chunk_id: item.score for item in leaning}
    assert scores == pytest.approx({
        "shared": 0.3 / 62 + 1 / 62, "vector": 1 / 61, "keyword": 0.3 / 61})


def test_a_question_asking_for_a_figure_is_fused_with_its_own_weights():
    bm25, dense = listed("bm25", "keyword"), listed("dense", "vector")
    hybrid = HybridRetriever(bm25, dense, figure_weights={"bm25": 0.0, "dense": 1.0})

    prose = hybrid.search(Query("What are the risks?"))
    figure = hybrid.search(Query("What was revenue?", wants_figures=True))

    assert [item.chunk_id for item in prose] == ["keyword", "vector"]
    assert [item.chunk_id for item in figure] == ["vector", "keyword"]
    assert figure[1].score == 0.0


def test_a_floor_on_the_fused_score_keeps_only_what_clears_it(monkeypatch):
    """Above 1/61 a passage has to have been found by both methods."""
    monkeypatch.setattr("src.retrieval.hybrid.MIN_FUSED_SCORE", 1 / 61 + 1e-9)
    bm25, dense = listed("bm25", "agreed", "keyword"), listed("dense", "agreed", "vector")

    results = HybridRetriever(bm25, dense).search(Query("revenue"))

    assert [item.chunk_id for item in results] == ["agreed"]


def test_a_fused_passage_keeps_what_it_cites_and_says_which_methods_found_it():
    bm25 = listed("bm25", "shared", "keyword")
    dense = listed("dense", "vector", "shared")

    results = {item.chunk_id: item for item in HybridRetriever(bm25, dense).search(Query("q"))}

    assert all(item.retriever == "hybrid" for item in results.values())
    assert results["shared"].sources == ("bm25", "dense")
    assert results["keyword"].sources == ("bm25",)
    assert results["vector"].sources == ("dense",)
    kept = ("text", "ticker", "company", "fiscal_year", "item", "title", "url")
    for item in results.values():
        assert all(getattr(item, field) == getattr(passage(item.chunk_id, "bm25", 1), field)
                   for field in kept)


class Admitting(FakeRetriever):
    """A method that can say whether a Query's filters admit any passage."""

    def __init__(self, name, admits):
        super().__init__(name, [])
        self.admits = admits

    def has_candidates(self, query):
        return self.admits


@pytest.mark.parametrize("bm25, dense, expected", [
    (True, False, True), (False, True, True), (False, False, False),
])
def test_filters_admit_a_passage_if_either_method_holds_one(bm25, dense, expected):
    hybrid = HybridRetriever(Admitting("bm25", bm25), Admitting("dense", dense))
    assert hybrid.has_candidates(Query("revenue")) is expected


def test_a_method_that_cannot_say_leaves_the_answer_unknown_not_empty():
    silent = FakeRetriever("dense", [])
    assert HybridRetriever(Admitting("bm25", False), silent).has_candidates(Query("q")) is None
    assert HybridRetriever(Admitting("bm25", True), silent).has_candidates(Query("q")) is True


def test_the_real_indexes_are_fused_in_the_order_the_formula_gives(corpus, tmp_path, fake_model):
    """Over a BM25 and a dense index of one corpus, not stand-ins: the fused list is
    what summing reciprocal ranks over the two methods' own lists gives."""
    bm25 = BM25Retriever(list(iter_chunks(processed_dir=corpus)))
    embed.build(chroma_dir=tmp_path / "chroma", processed_dir=corpus)
    dense = DenseRetriever.load(chroma_dir=tmp_path / "chroma", processed_dir=corpus)
    query = Query("topic 2 of Item 7 in fiscal 2024", top_k=12, tickers=("AAA", "BBB"))

    expected: dict[str, float] = {}
    for method in (bm25, dense):
        for rank, item in enumerate(method.search(query, k=CANDIDATE_K), start=1):
            expected[item.chunk_id] = expected.get(item.chunk_id, 0.0) + 1 / (RRF_K + rank)
    order = sorted(expected, key=lambda chunk_id: (-expected[chunk_id], chunk_id))[:12]

    results = HybridRetriever(bm25, dense).search(query)

    assert [item.chunk_id for item in results] == order
    assert [item.score for item in results] == pytest.approx([expected[c] for c in order])
    assert [item.rank for item in results] == list(range(1, 13))
    assert {item.ticker for item in results} <= {"AAA", "BBB"}
