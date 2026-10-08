"""Evidence is visible before generation starts, including cached answers (#42)."""

from dataclasses import replace

import pytest

from src.app.state import Remembered
from src.rag.answer import answer_question
from src.retrieval.records import Query
from tests.sample_answers import sample_answer
from tests.test_app_live import _ask


QUESTION = "What risks does Apple describe in FY2024?"


@pytest.mark.parametrize("route", ["retrieval", "facts", "empty", "refusal"])
def test_evidence_notification_precedes_generation_and_tokens(monkeypatch, route):
    import src.rag.answer as module

    answer = sample_answer(QUESTION)
    passage = answer.passages[0]
    events = []

    class Retriever:
        def search(self, query):
            events.append("search")
            # Include a rejected result and return valid results out of order.
            return [] if route == "empty" else [
                replace(passage, chunk_id="second", rank=2),
                replace(passage, chunk_id="blank", text=" "), passage,
            ]

    def generate(prompt, config, **kwargs):
        events.append("generate")
        kwargs["on_token"]("First ")
        kwargs["on_token"]("token")
        return object()

    monkeypatch.setattr(module, "generate", generate)
    monkeypatch.setattr(module, "resolve_citations", lambda *args: answer)
    monkeypatch.setattr(module, "answer_from_facts", lambda *args, **kwargs: answer)
    question = ("What was Apple's revenue in FY2024?" if route == "facts" else
                "Should I invest in Apple?" if route == "refusal" else QUESTION)
    result = answer_question(
        question, Retriever(), config=answer.config, use_decomposition=False,
        use_facts=route == "facts", query=Query(question, tickers=("AAPL",)),
        on_retrieved=lambda passages: events.append(tuple(p.chunk_id for p in passages)),
        on_token=lambda token: events.append(token),
    )
    if route == "retrieval":
        assert events == ["search", (passage.chunk_id, "second"), "generate", "First ", "token"]
    elif route == "facts":
        assert events == [(passage.chunk_id,), answer.text]
    else:
        assert events == (["search"] if route == "empty" else []) + [(), result.text]
        assert result.abstained


def test_cached_answer_notifies_new_display_without_repeating_search():
    calls = []

    class Stack:
        def answer(self, question, **kwargs):
            calls.append(question)
            answer = sample_answer(question)
            kwargs["on_retrieved"](answer.passages)
            return answer

    remembered = Remembered(Stack())
    first, second = [], []
    original = remembered.answer(QUESTION, on_retrieved=first.append)
    cached = remembered.answer(QUESTION, on_retrieved=second.append)
    assert cached is original
    assert calls == [QUESTION]
    assert first == second == [original.passages]


@pytest.mark.parametrize("streamed", [False, True])
def test_live_page_has_filters_and_evidence_while_answer_is_unfinished(monkeypatch, streamed):
    import streamlit as st

    def answer(question, *, on_retrieved, on_token, **kwargs):
        on_retrieved(sample_answer(question).passages)
        if streamed:
            on_token("First ")
            on_token("words")
        st.stop()

    ui = _ask(monkeypatch, answer)
    assert not ui.exception
    assert any("AAPL" in caption.value and "2024" in caption.value for caption in ui.caption)
    assert [heading.value for heading in ui.subheader] == ["Answer", "Retrieved evidence"]
    assert "AAPL" in ui.expander[0].label
    assert "Example passage" in ui.expander[0].text[0].value
    assert [text.value for text in ui.text] == (
        ["First words"] if streamed else []) + [sample_answer(QUESTION).passages[0].text]


def test_provider_error_retains_evidence_but_removes_unfinished_answer(monkeypatch):
    from src.rag.generate import ProviderUnavailable

    def answer(question, *, on_retrieved, on_token, **kwargs):
        on_retrieved(sample_answer(question).passages)
        on_token("Unfinished answer")
        raise ProviderUnavailable("Model unavailable")

    ui = _ask(monkeypatch, answer)
    assert not ui.exception
    assert "Model unavailable" in ui.error[0].value
    assert all(text.value != "Unfinished answer" for text in ui.text)
    assert "Example passage" in ui.expander[0].text[0].value


def test_completed_answer_keeps_one_evidence_panel_and_replaces_draft(monkeypatch):
    def answer(question, *, on_retrieved, on_token, **kwargs):
        completed = sample_answer(question)
        on_retrieved(completed.passages)
        on_token("Unfinished answer")
        return completed

    ui = _ask(monkeypatch, answer)
    assert not ui.exception
    assert [heading.value for heading in ui.subheader].count("Retrieved evidence") == 1
    assert len(ui.get("html")) == 1
    assert all(text.value != "Unfinished answer" for text in ui.text)
