"""The RAG records: what they refuse, and what they carry through to_dict."""

import dataclasses

import pytest

from src.rag.records import Answer, Citation, GenerationConfig
from src.retrieval.records import RetrievedPassage

CONFIG = GenerationConfig(provider="ollama", model="llama3.1:8b", prompt_template_id="grounded_v1")


def _passage(chunk_id: str, rank: int) -> RetrievedPassage:
    return RetrievedPassage(
        chunk_id=chunk_id, text=f"text of {chunk_id}", score=1.0 / rank, rank=rank,
        retriever="hybrid", ticker="AAA", company="Alpha Corp", fiscal_year=2024,
        item="7", title="Management's Discussion", url="https://example.test/a",
        sources=("bm25", "dense"),
    )


PASSAGES = (_passage("p1", 1), _passage("p2", 2), _passage("p3", 3))


def _answer(**overrides) -> Answer:
    fields = dict(
        question="What was revenue?",
        text="Revenue was $100 [1]. It grew [2].",
        citations=(Citation(1, "p1", True), Citation(2, "p2", True)),
        passages=PASSAGES,
        abstained=False,
        config=CONFIG,
        latency_ms=12.5,
    )
    fields.update(overrides)
    return Answer(**fields)


# --- frozen ------------------------------------------------------------------

@pytest.mark.parametrize("record", [CONFIG, Citation(1, "p1", True), _answer()])
def test_records_are_frozen(record):
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(record, dataclasses.fields(record)[0].name, "changed")


# --- GenerationConfig --------------------------------------------------------

def test_config_defaults_to_deterministic_generation():
    assert CONFIG.temperature == 0.0
    assert isinstance(GenerationConfig("a", "m", "t", temperature=1).temperature, float)


def test_config_refuses_a_negative_temperature():
    with pytest.raises(ValueError, match="temperature"):
        GenerationConfig("a", "m", "t", temperature=-0.1)


def test_config_to_dict_is_round_trippable():
    data = CONFIG.to_dict()
    assert data == {"provider": "ollama", "model": "llama3.1:8b",
                    "prompt_template_id": "grounded_v1", "temperature": 0.0}
    assert GenerationConfig(**data) == CONFIG


# --- Citation ----------------------------------------------------------------

def test_unresolved_citation_keeps_its_marker_without_a_passage():
    invented = Citation(marker=9, chunk_id=None, resolved=False)
    assert invented.resolved is False
    assert invented.to_dict() == {"marker": 9, "chunk_id": None, "resolved": False}


@pytest.mark.parametrize("chunk_id, resolved", [("p1", False), (None, True)])
def test_citation_resolved_must_agree_with_chunk_id(chunk_id, resolved):
    with pytest.raises(ValueError, match="resolved|unresolved"):
        Citation(1, chunk_id, resolved)


@pytest.mark.parametrize("marker", [0, -1, "1", True])
def test_citation_marker_is_a_positive_integer(marker):
    with pytest.raises(ValueError, match="marker"):
        Citation(marker, "p1", True)


# --- Answer ------------------------------------------------------------------

def test_answer_normalises_lists_to_tuples():
    answer = _answer(citations=[Citation(1, "p1", True)], passages=list(PASSAGES))
    assert isinstance(answer.citations, tuple)
    assert isinstance(answer.passages, tuple)


def test_answer_refuses_a_resolved_citation_to_an_unseen_passage():
    with pytest.raises(ValueError, match="do not match the numbered passages"):
        _answer(citations=(Citation(1, "ghost", True),))


@pytest.mark.parametrize("citation", [
    Citation(1, "p3", True),    # in range, wrong passage
    Citation(9, "p1", True),    # resolved past the last source
    Citation(2, None, False),   # unresolved though source 2 exists
])
def test_answer_refuses_a_citation_that_disagrees_with_its_position(citation):
    with pytest.raises(ValueError, match="do not match the numbered passages"):
        _answer(citations=(citation,))


def test_answer_allows_an_unresolved_marker():
    answer = _answer(citations=(Citation(1, "p1", True), Citation(9, None, False)))
    assert answer.unresolved_markers == (9,)
    assert answer.cited_passages == (PASSAGES[0],)


def test_answer_refuses_duplicate_markers_and_duplicate_passages():
    with pytest.raises(ValueError, match="one entry per marker"):
        _answer(citations=(Citation(1, "p1", True), Citation(1, "p2", True)))
    with pytest.raises(ValueError, match="duplicate chunk_ids"):
        _answer(passages=(PASSAGES[0], PASSAGES[0]))


def test_cited_passages_follow_marker_order_not_citation_order():
    answer = _answer(citations=(Citation(3, "p3", True), Citation(1, "p1", True)))
    assert [p.chunk_id for p in answer.cited_passages] == ["p1", "p3"]


def test_abstention_carries_no_citations():
    answer = _answer(text="The provided passages do not say.", citations=(), abstained=True)
    assert answer.abstained is True
    assert answer.cited_passages == ()


def test_answer_to_dict_is_json_shaped():
    data = _answer().to_dict()
    assert data["config"] == CONFIG.to_dict()
    assert data["citations"] == [{"marker": 1, "chunk_id": "p1", "resolved": True},
                                 {"marker": 2, "chunk_id": "p2", "resolved": True}]
    assert [p["chunk_id"] for p in data["passages"]] == ["p1", "p2", "p3"]
    assert data["passages"][0]["sources"] == ["bm25", "dense"]   # a list, not a tuple
    assert data["abstained"] is False
    assert data["latency_ms"] == 12.5


def test_config_rides_on_the_answer():
    # The harness reads what generated an answer off the answer itself.
    other = GenerationConfig(provider="anthropic", model="claude-sonnet-5",
                             prompt_template_id="grounded_v1")
    assert _answer(config=other).config is other
