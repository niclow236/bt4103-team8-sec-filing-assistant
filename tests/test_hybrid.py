from src.retrieval.constants import CANDIDATE_K
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
