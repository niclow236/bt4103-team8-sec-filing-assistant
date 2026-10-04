"""The real app answers through a named configuration, the way the harness does.

The app no longer assembles a retriever and a generator of its own (#43): it
asks ``src.stack`` for a configuration by id and answers through it, so these
tests stand in a built ``Stack`` rather than patching the RAG entry point.
"""

from dataclasses import dataclass, replace
from typing import Any

from streamlit.testing.v1 import AppTest

import src.app.app as app_module
from src.rag.records import GenerationConfig
from src.stack import DEFAULT_STACK, SELECTABLE, stack_config
from tests.sample_answers import sample_answer


APP = "from src.app.app import main\nmain()"
GENERATION = GenerationConfig("ollama", "llama3.2:3b", "grounded_v4")


@dataclass
class FakeStack:
    """A built configuration that answers without an index or a model."""

    config: Any
    retriever: Any = object()
    generation: GenerationConfig = GENERATION
    llm: Any = None
    asked: list = None

    def __post_init__(self):
        self.asked = []

    def answer(self, question, **overrides):
        self.asked.append((question, overrides))
        return sample_answer(question)


def _standing_in(monkeypatch, config_id=DEFAULT_STACK, runs=()):
    """Stand a FakeStack in for whatever configuration the app selects."""
    built = {}

    def load_stack(selected):
        built.setdefault(selected, FakeStack(config=stack_config(selected)))
        return built[selected]

    monkeypatch.setattr(app_module, "load_stack", load_stack)
    monkeypatch.setattr(app_module, "measured", lambda: list(runs))
    monkeypatch.setattr(app_module, "verify_answer", lambda answer, *, parsed: answer)
    return built


def test_live_app_answers_through_the_selected_configuration(monkeypatch):
    built = _standing_in(monkeypatch)
    ui = AppTest.from_string(APP).run(timeout=30)
    assert not ui.exception
    ui.text_input[0].set_value("What was Apple's revenue in FY2024, Item 7?").run()
    ui.button[0].click().run()
    assert not ui.exception

    # Answered through the default configuration's stack, not a stack the app
    # assembled, and the question's own filters reached it.
    stack = built[DEFAULT_STACK]
    question, overrides = stack.asked[0]
    assert question == overrides["parsed"].question
    assert overrides["query"].filters == {
        "ticker": ["AAPL"], "fiscal_year": [2024], "item": ["7"],
    }
    assert stack.generation.provider in {"ollama", "mistral"}
    assert len(ui.get("html")) == 1

    ui.sidebar.multiselect[2].set_value(["8"]).run()
    assert not ui.exception
    assert len(ui.get("html")) == 0
    assert ui.info
    ui.button[0].click().run()
    assert stack.asked[-1][1]["query"].items == ("8",)


def test_live_app_offers_every_configuration_it_can_build(monkeypatch):
    # AppTest reports a selectbox's options already formatted, so each row is
    # named by its id and the configuration's own name. C0 is measured by the
    # ablation runner but cannot be built here, so it is not offered even when
    # a run has measured it.
    _standing_in(monkeypatch, runs=[("C0", "nightly-7 · C0 — naive BM25")])
    ui = AppTest.from_string(APP).run(timeout=30)
    assert not ui.exception
    options = list(ui.sidebar.selectbox[0].options)
    assert len(options) == len(SELECTABLE)
    assert all(
        label.startswith(f"{config_id} — ")
        for config_id, label in zip(SELECTABLE, options)
    )
    assert ui.sidebar.selectbox[0].value == DEFAULT_STACK


def test_live_app_says_when_a_row_does_not_apply_the_filters(monkeypatch):
    # The sidebar prints the filters it would search with, so a row that
    # ignores them has to say so.
    _standing_in(monkeypatch)
    ui = AppTest.from_string(APP).run(timeout=30)
    assert not any("filters above are not applied" in c.value for c in ui.sidebar.caption)
    ui.sidebar.selectbox[0].set_value("C3").run()
    assert not ui.exception
    assert any("filters above are not applied" in c.value for c in ui.sidebar.caption)


def test_live_app_labels_a_configuration_with_the_run_that_measured_it(monkeypatch):
    # Criterion: a configuration named in results/ can be selected in the app.
    _standing_in(monkeypatch, runs=[("C1", "nightly-7 · C1 — section-aware BM25")])
    ui = AppTest.from_string(APP).run(timeout=30)
    assert not ui.exception
    assert "nightly-7 · C1 — section-aware BM25" in list(ui.sidebar.selectbox[0].options)
    # The default row has no run, so the app says so rather than implying one.
    assert any("No run under results/" in caption.value for caption in ui.sidebar.caption)


def test_selecting_a_measured_configuration_builds_that_configuration(monkeypatch):
    built = _standing_in(monkeypatch, runs=[("C1", "nightly-7 · C1 — section-aware BM25")])
    ui = AppTest.from_string(APP).run(timeout=30)
    ui.text_input[0].set_value("What was Apple's revenue in FY2024?").run()
    ui.sidebar.selectbox[0].set_value("C1").run()
    ui.button[0].click().run()
    assert not ui.exception
    assert "C1" in built and DEFAULT_STACK not in built
    assert built["C1"].config.retriever == "bm25"


def test_live_app_reports_missing_index_without_model_call(monkeypatch):
    def no_index(config_id):
        raise FileNotFoundError("Build the local index first")

    monkeypatch.setattr(app_module, "load_stack", no_index)
    monkeypatch.setattr(app_module, "measured", lambda: [])
    ui = AppTest.from_string(APP).run(timeout=30)
    ui.text_input[0].set_value("Apple FY2024 revenue?").run()
    ui.button[0].click().run()
    assert not ui.exception
    assert "Build the local index first" in ui.error[0].value
    assert not ui.get("html")


def test_live_app_verifies_against_the_sidebar_scope(monkeypatch):
    verified = []
    _standing_in(monkeypatch)
    monkeypatch.setattr(app_module, "verify_answer",
                        lambda answer, *, parsed: verified.append(parsed) or answer)
    ui = AppTest.from_string(APP).run(timeout=30)
    ui.text_input[0].set_value("What was Apple's revenue in FY2024?").run()
    ui.sidebar.multiselect[0].set_value(["MSFT"]).run()
    ui.button[0].click().run()
    assert not ui.exception
    assert verified[0].tickers == ("MSFT",)
    assert verified[0].fiscal_years == (2024,)


def test_live_app_reads_the_question_with_the_facts_store_s_labels(monkeypatch):
    # answer_question and the evaluation harness read a question with the
    # store's labels as cues. The app read it without them, so a question
    # naming a line item by its label was searched as prose in the app and as
    # a figure question everywhere it was measured.
    read = []
    parse = app_module.parse_question
    monkeypatch.setattr(app_module, "parse_question",
                        lambda question, **options: read.append(options) or parse(
                            question, **options))
    _standing_in(monkeypatch)
    # The first question to need the labels reads them from the store, which
    # can take longer than AppTest's three seconds on a busy machine.
    ui = AppTest.from_string(APP, default_timeout=30).run()
    ui.text_input[0].set_value("What was Apple's gross profit in FY2024?").run()
    ui.button[0].click().run()
    assert not ui.exception
    assert read and all("facts_file" not in options for options in read)


def _ask(monkeypatch, answer, question="What was Apple's revenue in FY2024?"):
    """Ask the app one question, with ``answer`` standing in for the configuration's own."""
    _standing_in(monkeypatch)
    monkeypatch.setattr(FakeStack, "answer",
                        lambda self, question, **overrides: answer(question, **overrides))
    ui = AppTest.from_string(APP, default_timeout=30).run()
    ui.text_input[0].set_value(question).run()
    ui.button[0].click().run()
    assert not ui.exception
    return ui


def test_live_app_writes_the_answer_to_the_page_as_it_arrives(monkeypatch):
    import streamlit as st

    def answer(question, *args, on_token, **kwargs):
        on_token("Example revenue ")
        on_token("was $5 billion.")
        st.stop()  # The page as it stands part way through an answer.

    ui = _ask(monkeypatch, answer)
    assert [text.value for text in ui.text] == ["Example revenue was $5 billion."]


def test_live_app_replaces_the_streamed_prose_with_the_checked_answer(monkeypatch):
    # What is streamed is provisional. Once the answer is resolved and
    # checked, the card is the only copy of it on the page.
    def answer(question, *args, on_token, **kwargs):
        on_token("Example revenue was $5 billion.")
        return sample_answer(question)

    ui = _ask(monkeypatch, answer)
    assert not ui.text
    assert len(ui.get("html")) == 1


def test_live_app_leaves_no_half_answer_above_a_provider_error(monkeypatch):
    def answer(question, *args, on_token, **kwargs):
        on_token("Example revenue ")
        raise app_module.ProviderUnavailable("The model stopped answering")

    ui = _ask(monkeypatch, answer)
    assert "The model stopped answering" in ui.error[0].value
    assert not ui.text
    assert not ui.get("html")


def test_live_app_says_under_the_answer_how_the_question_was_read(monkeypatch):
    ui = _ask(monkeypatch, lambda question, *args, **kwargs: sample_answer(question))
    assert ui.main.caption[-1].value == (
        "Question type: numeric · Companies: AAPL · Fiscal years: FY2024")
    # In the scope the sidebar ends up with, which is what gets searched.
    ui.sidebar.multiselect[0].set_value(["MSFT"]).run()
    ui.button[0].click().run()
    assert not ui.exception
    assert ui.main.caption[-1].value == (
        "Question type: numeric · Companies: MSFT · Fiscal years: FY2024")


def test_live_app_names_the_filings_a_split_question_was_searched_in(monkeypatch):
    ui = _ask(monkeypatch, lambda question, *args, **kwargs: replace(
        sample_answer(question), sub_questions=("AAPL FY2024", "MSFT FY2024")),
        question="Compare Apple and Microsoft's revenue in FY2024")
    assert ui.main.caption[-1].value.endswith(
        " · Searched one filing at a time: AAPL FY2024, MSFT FY2024")


def test_live_app_searches_with_hybrid_unless_another_row_is_chosen(monkeypatch):
    # The row the app opens on is the one the 48 test questions were measured
    # best with (README, "How often the answers are right"), so a change of
    # DEFAULT_STACK to a BM25 row has to be made on purpose.
    built = _standing_in(monkeypatch)
    ui = AppTest.from_string(APP).run(timeout=30)
    ui.text_input[0].set_value("What was Apple's revenue in FY2024?").run()
    ui.button[0].click().run()
    ui.sidebar.selectbox[0].set_value("C1").run()
    ui.button[0].click().run()
    assert not ui.exception
    assert [stack.config.retriever for stack in built.values()] == ["hybrid", "bm25"]
