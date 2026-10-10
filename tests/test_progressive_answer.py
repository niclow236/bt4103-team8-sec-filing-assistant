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
    captions = [caption.value for caption in ui.caption]
    assert ("Writing the answer…" in captions) is not streamed
    assert "Searching filings…" not in captions


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


def _in_order(node):
    """The page's elements top to bottom, as (type, value) pairs."""
    for child in getattr(node, "children", {}).values():
        yield child.type, getattr(child, "value", None)
        yield from _in_order(child)


@pytest.mark.parametrize("error_kind", ["provider", "file", "value"])
def test_errors_are_drawn_above_the_evidence(monkeypatch, error_kind):
    from src.rag.generate import ProviderUnavailable

    error = {"provider": ProviderUnavailable, "file": OSError, "value": ValueError}[error_kind]

    def answer(question, *, on_retrieved, **kwargs):
        on_retrieved(sample_answer(question).passages)
        raise error("Unavailable for this test")

    ui = _ask(monkeypatch, answer)
    assert not ui.exception
    order = [kind for kind, _ in _in_order(ui.main)]
    assert order.index("error") < order.index("expander")
    assert "Unavailable for this test" in ui.error[0].value


def test_a_refused_question_draws_no_evidence_panel(monkeypatch):
    from src.rag.constants import ABSTAIN_PHRASE

    def refused(question, *, on_retrieved, **kwargs):
        on_retrieved(())
        return replace(sample_answer(question), text=ABSTAIN_PHRASE, abstained=True,
                       abstention_reason="beyond_the_filings", sentences=(), citations=(),
                       passages=(), verification=None)

    ui = _ask(monkeypatch, refused, question="Should I buy Apple stock?")
    assert not ui.exception
    assert "gives no advice" in ui.warning[0].value
    assert "Retrieved evidence" not in [heading.value for heading in ui.subheader]
    assert "No supporting passages were retrieved." not in [c.value for c in ui.caption]


def test_search_status_is_visible_before_evidence_arrives(monkeypatch):
    import streamlit as st

    def answer(question, **kwargs):
        st.stop()

    ui = _ask(monkeypatch, answer)
    assert not ui.exception
    assert "Searching filings…" in [caption.value for caption in ui.caption]
    assert "Retrieved evidence" not in [heading.value for heading in ui.subheader]


@pytest.mark.parametrize("url", ["https://www.sec.gov/edgar/search/", "javascript:alert(1)", ""])
def test_evidence_reuses_the_corpus_panel_for_tables_and_safe_links(monkeypatch, url):
    import streamlit as st

    table = "Revenue\n\n| Year | Revenue |\n| --- | --- |\n| 2024 | $5 billion |"

    def answer(question, *, on_retrieved, **kwargs):
        passage = replace(sample_answer(question).passages[0], fiscal_year=None, item=None,
                          content_type="table", text=table, url=url)
        on_retrieved((passage,))
        st.stop()

    ui = _ask(monkeypatch, answer)
    assert not ui.exception
    assert "FYunknown" in ui.expander[0].label and "Item unknown" in ui.expander[0].label
    assert not ui.expander[0].proto.expanded
    assert any("Table passage" in badge.value for badge in ui.get("markdown"))
    assert "<table" in ui.get("html")[0].value
    assert ui.code[0].value == table
    links = ui.get("link_button")
    if url.startswith("https://"):
        assert len(links) == 1 and links[0].label == "Open filing on EDGAR"
    else:
        assert not links
        assert "Filing link unavailable" in [caption.value for caption in ui.caption]
