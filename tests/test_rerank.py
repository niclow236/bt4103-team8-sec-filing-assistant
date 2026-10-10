from dataclasses import replace

import pytest

from src.retrieval.base import Retriever
from src.retrieval.records import Query, RetrievedPassage
from src.retrieval.rerank import CrossEncoderReranker, RerankConfig, RerankRetriever


def passage(cid, score=1, content_type="prose", ticker="AAA", year=2024):
    return RetrievedPassage(cid, "Evidence " + cid, score, 1, "hybrid", ticker,
                            "Alpha Corp", year, "8", "Statements", "https://example.test",
                            content_type, ("bm25", "dense"))


class Model:
    device = "test"

    def __init__(self, scores, lengths=None):
        self.scores = scores
        self.lengths = lengths
        self.calls = []

    def tokenizer(self, queries, texts, **kwargs):
        self.calls.append(("tokens", queries, texts, kwargs))
        return {"length": self.lengths or [25] * len(texts)}

    def predict(self, pairs, **kwargs):
        self.calls.append(("predict", pairs, kwargs))
        return self.scores


class Inner:
    name = "hybrid"

    def __init__(self, passages):
        self.passages = passages
        self.calls = []

    def search(self, query, k=None):
        self.calls.append((query, k))
        return self.passages[:k]

    def has_candidates(self, query):
        return bool(self.passages)


def test_ties_raw_negative_table_boost_and_provenance():
    model = Model([-2, -2, -3], [512, 513, 100])
    scorer = CrossEncoderReranker(RerankConfig(candidate_k=3), model=model)
    originals = [passage("z"), passage("a"), passage("t", content_type="table")]
    result = scorer.rerank(Query("raw full question", top_k=3, table_boost=2), originals)
    assert [p.chunk_id for p in result.passages] == ["t", "a", "z"]
    assert [p.score for p in result.passages] == [-1.5, -2, -2]
    assert [p.rank for p in result.passages] == [1, 2, 3]
    assert result.passages[0] == replace(originals[2], score=-1.5, rank=1, retriever=scorer.name)
    assert result.diagnostics["truncated"] == 1
    assert result.diagnostics["candidates"][1]["raw_logit"] == -2
    assert result.diagnostics["latency_ms"] >= 0
    assert model.calls[1][1] == [("raw full question", p.text) for p in originals]
    assert originals[2].retriever == "hybrid"


def test_adapter_passes_all_filters_and_candidate_depth():
    q = Query("full", top_k=1, tickers=("AAA",), fiscal_years=(2024,),
              items=("8",), content_type="table", keyword_text="trimmed", table_boost=1.2)
    inner = Inner([passage("b", content_type="table"), passage("a", content_type="table")])
    scorer = CrossEncoderReranker(RerankConfig(candidate_k=2), model=Model([1, 3]))
    r = RerankRetriever(inner, scorer)
    assert isinstance(r, Retriever)
    assert r.has_candidates(q)
    assert [p.chunk_id for p in r.search(q)] == ["a"]
    assert inner.calls == [(q, 2)]
    assert [p.rank for p in r.search(q, k=2)] == [1, 2]


@pytest.mark.parametrize("change", [
    {"ticker": "BBB"}, {"fiscal_year": 2023}, {"item": "1A"}, {"content_type": "prose"},
])
def test_filter_violation_fails_before_model_scoring(change):
    model = Model([1])
    scorer = CrossEncoderReranker(model=model)
    q = Query("q", top_k=1, tickers=("AAA",), fiscal_years=(2024,), items=("8",), content_type="table")
    with pytest.raises(ValueError, match="metadata filters"):
        scorer.rerank(q, [replace(passage("a", content_type="table"), **change)])
    assert not model.calls


@pytest.mark.parametrize("field,value", [("candidate_k", 0), ("max_length", -1),
    ("batch_size", True), ("batch_size", 1.5), ("model", ""), ("revision", " "), ("device", "")])
def test_invalid_config(field, value):
    with pytest.raises(ValueError):
        RerankConfig(**{field: value})


@pytest.mark.parametrize("scores", [[float("nan")], [float("inf")], [], [1, 2]])
def test_invalid_model_output(scores):
    r = CrossEncoderReranker(model=Model(scores))
    with pytest.raises(ValueError):
        r.rerank(Query("q", top_k=1), [passage("a")])


def test_empty_and_zero_k_do_not_load_model():
    r = CrossEncoderReranker()
    assert r.rerank(Query("q", top_k=1), []).passages == []
    assert r.rerank(Query("q", top_k=0), [passage("a")]).passages == []
    assert r._model is None


def test_invalid_pool_and_budget():
    r = CrossEncoderReranker(RerankConfig(candidate_k=1))
    with pytest.raises(ValueError, match="final k"):
        r.rerank(Query("q", top_k=2), [])
    with pytest.raises(ValueError, match="exceeds"):
        r.rerank(Query("q", top_k=1), [passage("a"), passage("b")])
    r = CrossEncoderReranker(RerankConfig(candidate_k=2))
    with pytest.raises(ValueError, match="duplicate"):
        r.rerank(Query("q", top_k=1), [passage("a"), passage("a")])
    with pytest.raises(ValueError, match="table_boost"):
        r.rerank(Query("q", top_k=1, table_boost=float("nan")), [passage("a")])


def test_load_is_pinned_and_cached(monkeypatch):
    import sentence_transformers

    calls = []
    model = Model([1])

    def build(*args, **kwargs):
        calls.append((args, kwargs))
        return model

    monkeypatch.setattr(sentence_transformers, "CrossEncoder", build)
    r = CrossEncoderReranker(RerankConfig(device="cpu"))
    assert r.load() is r.load() is model
    assert len(calls) == 1
    assert calls[0][1]["revision"] == r.config.revision
    assert calls[0][1]["trust_remote_code"] is False
    assert calls[0][1]["activation_fn"](-3) == -3
