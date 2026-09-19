from src.retrieval.base import rank
from src.retrieval.records import Query
from src.retrieval.rerank import CrossEncoderReranker


def chunk(chunk_id):
    return {
        "chunk_id": chunk_id, "text": chunk_id, "ticker": "AAPL", "company": "Apple",
        "fiscal_year": 2024, "item": "7", "title": "MD&A", "url": "https://example.com",
    }


class Inner:
    name = "hybrid"

    def __init__(self):
        self.calls = []

    def search(self, query, k=None):
        self.calls.append(k)
        return rank([(chunk("first"), 1.0), (chunk("second"), 0.9)], self.name, k=k)


class FakeCrossEncoder:
    def __init__(self):
        self.calls = []

    def predict(self, pairs, batch_size, show_progress_bar):
        self.calls.append((pairs, batch_size, show_progress_bar))
        return [0.1, 0.9]


def test_reranker_scores_candidates_and_returns_top_k():
    inner = Inner()
    model = FakeCrossEncoder()
    results = CrossEncoderReranker(inner, model=model).search(Query("question", top_k=1))

    assert inner.calls == [50]
    assert model.calls[0][0] == [("question", "first"), ("question", "second")]
    assert [result.chunk_id for result in results] == ["second"]
    assert results[0].retriever == "rerank"
    assert results[0].sources == ("hybrid",)
