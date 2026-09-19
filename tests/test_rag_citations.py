"""Issue #31: source identity, stored labels, and visible sentence warnings."""

import dataclasses
import json

import pytest

from src.rag import render_citation, resolve_citations
from src.rag.constants import ABSTAIN_PHRASE, PROMPT_TEMPLATE_ID
from src.rag.prompt import build_prompt
from src.rag.records import Citation, CitedSentence, Generation, GenerationConfig, GroundedAnswer
from src.retrieval.embed import chunk_from_record, metadata_for
from src.retrieval.records import RetrievedPassage


CONFIG = GenerationConfig("ollama", "test-model", PROMPT_TEMPLATE_ID)


def _passage(chunk_id="p1", rank=1, **overrides):
    fields = dict(
        chunk_id=chunk_id, text="Revenue was $5.2 billion.", score=1.0,
        rank=rank, retriever="hybrid", ticker="AAA", company="Alpha Corp",
        fiscal_year=2024, item="7", title="Management's Discussion",
        url="https://example.test/filing", cik=123, form="10-K", part="II",
        filing_date="2025-02-01",
    )
    return RetrievedPassage(**(fields | overrides))


def _generation(*sentences, answerable=True, **overrides):
    answer = GroundedAnswer(
        answerable=answerable,
        sentences=tuple(CitedSentence(text=text, sources=sources) for text, sources in sentences),
    )
    fields = dict(
        text=answer.render(), answer=answer, raw=answer.model_dump_json(),
        config=CONFIG, latency_ms=12.5, input_tokens=50, output_tokens=20,
        stop_reason="stop",
    )
    return Generation(**(fields | overrides))


def test_sources_resolve_by_prompt_position_not_rank_or_input_order():
    first = _passage("a", rank=5)
    second = _passage("b", rank=5, ticker="BBB")
    prompt = build_prompt("Revenue?", [second, first])
    result = resolve_citations(
        "Revenue?", _generation(("It rose.", (2, 1))), iter(prompt.passages)
    )
    assert result.citations == (Citation(1, "a", True), Citation(2, "b", True))
    assert result.passages == prompt.passages
    assert result.cited_passages == (first, second)
    assert result.flagged_sentences == ()
    assert result.config is CONFIG
    assert result.latency_ms == 12.5


def test_resolver_does_not_resort_the_supplied_prompt_sources():
    shown = (_passage("b", 9), _passage("a", 1))
    result = resolve_citations("q", _generation(("Claim.", (1,))), shown)
    assert result.citations == (Citation(1, "b", True),)


def test_invented_marker_is_removed_but_retained_as_unresolved():
    result = resolve_citations("q", _generation(("Claim.", (9,))), [_passage()])
    assert result.text == "Claim."
    assert result.citations == (Citation(9, None, False),)
    assert result.unresolved_markers == (9,)
    assert result.cited_passages == ()
    assert result.flagged_sentences == result.sentences
    assert result.sentences[0].raw_text == "Claim. [9]"


def test_mixed_citations_flag_only_the_affected_sentence():
    result = resolve_citations("q", _generation(
        ("Supported reference.", (1,)), ("Mixed.", (1, 9)), ("Uncited.", ()),
    ), [_passage()])
    assert result.text == "Supported reference. [1] Mixed. [1] Uncited."
    assert [s.flagged for s in result.sentences] == [False, True, True]
    assert result.citations == (Citation(1, "p1", True), Citation(9, None, False))


def test_repeated_markers_have_one_audit_record_but_flag_each_occurrence():
    result = resolve_citations("q", _generation(
        ("First.", (1, 9, 9)), ("Second.", (9,)),
    ), [_passage()])
    assert len(result.citations) == 2
    assert len(result.flagged_sentences) == 2


def test_inline_inventions_cannot_hide_inside_structured_text():
    result = resolve_citations(
        "q", _generation(("It rose [9], then fell [1].", ())), [_passage()]
    )
    assert result.text == "It rose, then fell [1]."
    assert result.unresolved_markers == (9,)
    assert result.sentences[0].flagged


@pytest.mark.parametrize("marker", ["[0]", "[-1]"])
def test_nonpositive_markers_are_removed_and_audited(marker):
    result = resolve_citations("q", _generation((f"Claim {marker}.", (1,))), [_passage()])
    assert result.text == "Claim. [1]"
    assert result.sentences[0].invalid_markers == (marker,)
    assert result.sentences[0].flagged


def test_structured_boundaries_preserve_abbreviations_decimals_and_multiline_text():
    text = "Alpha Inc. reported $5.2 billion.\nGrowth was 3.5%."
    result = resolve_citations("q", _generation((text, (1,))), [_passage()])
    assert len(result.sentences) == 1
    assert result.text == text + " [1]"
    assert not result.sentences[0].flagged


def test_every_marker_is_unresolved_when_no_sources_were_shown():
    result = resolve_citations("q", _generation(("Claim.", (1, 3))), [])
    assert result.text == "Claim."
    assert result.unresolved_markers == (1, 3)
    assert result.sentences[0].flagged


@pytest.mark.parametrize("sentences", [(), (("Ignored claim.", (9,)),)])
def test_abstention_has_no_claims_or_citations(sentences):
    result = resolve_citations("q", _generation(*sentences, answerable=False), [_passage()])
    assert result.abstained
    assert result.text == ABSTAIN_PHRASE
    assert result.citations == result.sentences == result.flagged_sentences == ()


def test_blank_structured_sentences_are_skipped():
    result = resolve_citations(
        "q", _generation((" ", (9,)), ("Claim.", (1,))), [_passage()]
    )
    assert result.text == "Claim. [1]"
    assert result.unresolved_markers == ()


def test_range_validation_failure_recovers_sentence_boundaries_but_keeps_error():
    generation = _generation(
        ("First.", (1,)), ("Invented.", (9,)), answer=None,
        parse_error="source must be <= 1",
    )
    result = resolve_citations("q", generation, [_passage()])
    assert result.text == "First. [1] Invented."
    assert len(result.sentences) == 2
    assert result.unresolved_markers == (9,)
    assert result.parse_error == generation.parse_error
    assert [s.flagged for s in result.sentences] == [False, True]


def test_truncated_abstention_has_no_sentences_or_citations():
    generation = _generation(
        answer=None, raw='{"answerable": false', text=ABSTAIN_PHRASE,
        parse_error="invalid JSON", stop_reason="length",
    )
    result = resolve_citations("q", generation, [_passage()])
    assert result.abstained
    assert result.text == ABSTAIN_PHRASE
    assert result.citations == result.sentences == result.flagged_sentences == ()
    assert result.truncated and result.parse_error == "invalid JSON"


def test_truncated_output_remains_visible_and_flagged():
    generation = _generation(
        answer=None, raw='{"answerable":true,"sentences":[',
        text="Completed. [1] Invented. [9] Unfinished", parse_error="invalid JSON",
        stop_reason="length",
    )
    result = resolve_citations("q", generation, [_passage()])
    assert result.text == "Completed. [1] Invented. Unfinished"
    assert result.truncated and result.parse_error == "invalid JSON"
    assert not result.abstained
    assert result.unresolved_markers == (9,)
    assert result.sentences[0].incomplete and result.sentences[0].flagged


def test_empty_malformed_output_is_not_silently_counted_as_abstention():
    result = resolve_citations("q", _generation(
        answer=None, raw="{", text="", parse_error="invalid JSON",
    ), [_passage()])
    assert not result.abstained
    assert result.parse_error and result.text == ""


def test_full_citation_label_uses_stored_metadata_and_not_filing_year():
    assert render_citation(Citation(1, "p1", True), [_passage()]) == (
        "[1] Alpha Corp (AAA, CIK 0000000123), 10-K, fiscal year 2024, "
        "Part II / Item 7, Management's Discussion, filed 2025-02-01"
    )


def test_missing_metadata_is_explicit_and_never_guessed():
    passage = _passage(cik=None, form=None, fiscal_year=None, part=None,
                       item=None, title="", filing_date=None)
    assert render_citation(Citation(1, "p1", True), [passage]) == (
        "[1] Alpha Corp (AAA, CIK unknown), form unknown, fiscal year unknown, "
        "Part unknown / Item unknown, section title unknown, filed unknown"
    )


def test_unresolved_label_is_explicit_and_wrong_source_identity_is_refused():
    assert render_citation(Citation(9, None, False), [_passage()]) == "[9] Unresolved citation"
    for citation in (Citation(1, "other", True), Citation(1, None, False)):
        with pytest.raises(ValueError, match="numbered passages"):
            render_citation(citation, [_passage()])


def test_citation_metadata_survives_the_dense_index_projection():
    passage = _passage(form="10-Q", part="I", item="2")
    row = dataclasses.asdict(passage)
    stored = chunk_from_record(passage.chunk_id, passage.text, metadata_for(row))
    for source in (row, stored):
        retrieved = RetrievedPassage.from_chunk(source, score=1.0, rank=1, retriever="dense")
        assert (retrieved.cik, retrieved.form, retrieved.part, retrieved.filing_date) == (
            123, "10-Q", "I", "2025-02-01",
        )
        assert "Part I / Item 2" in render_citation(Citation(1, "p1", True), [retrieved])


def test_checks_are_frozen_and_json_serializable_for_the_app_and_metrics():
    result = resolve_citations("q", _generation(("Claim.", (9,))), [_passage()])
    data = json.loads(json.dumps(result.to_dict()))
    assert data["sentences"][0]["flagged"] is True
    assert data["sentences"][0]["citations"] == [
        {"marker": 9, "chunk_id": None, "resolved": False},
    ]
    assert data["passages"][0]["cik"] == 123
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.sentences[0].text = "changed"


def test_duplicate_source_ids_are_refused():
    with pytest.raises(ValueError, match="duplicate chunk_ids"):
        resolve_citations("q", _generation(("Claim.", (1,))), [_passage(), _passage()])
