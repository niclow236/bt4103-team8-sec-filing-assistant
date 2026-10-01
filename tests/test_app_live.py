"""The real app passes sidebar choices to the existing RAG entry point."""

from streamlit.testing.v1 import AppTest

import src.app.app as app_module
from tests.sample_answers import sample_answer


APP = "from src.app.app import main\nmain()"


def test_live_app_passes_actual_query_and_shows_answer(monkeypatch):
    seen = []
    retriever = object()
    monkeypatch.setattr(app_module, "load_retriever", lambda method: retriever)

    def answer(question, passed_retriever, *, query, parsed, config):
        seen.append((question, passed_retriever, query, parsed, config))
        return sample_answer(question)

    monkeypatch.setattr(app_module, "answer_question", answer)
    monkeypatch.setattr(app_module, "verify_answer", lambda answer, *, parsed: answer)
    ui = AppTest.from_string(APP).run(timeout=30)
    assert not ui.exception
    ui.text_input[0].set_value("What was Apple's revenue in FY2024, Item 7?").run()
    ui.button[0].click().run()
    assert not ui.exception
    question, passed_retriever, query, parsed, config = seen[0]
    assert question == parsed.question
    assert passed_retriever is retriever
    assert query.filters == {"ticker": ["AAPL"], "fiscal_year": [2024], "item": ["7"]}
    assert config.provider in {"ollama", "mistral"}
    assert len(ui.get("html")) == 1
    ui.sidebar.multiselect[2].set_value(["8"]).run()
    assert not ui.exception
    assert len(ui.get("html")) == 0
    assert ui.info
    ui.button[0].click().run()
    assert seen[-1][2].items == ("8",)


def test_live_app_reports_missing_index_without_model_call(monkeypatch):
    def no_index(method):
        raise FileNotFoundError("Build the local index first")

    monkeypatch.setattr(app_module, "load_retriever", no_index)
    ui = AppTest.from_string(APP).run(timeout=30)
    ui.text_input[0].set_value("Apple FY2024 revenue?").run()
    ui.button[0].click().run()
    assert not ui.exception
    assert "Build the local index first" in ui.error[0].value
    assert not ui.get("html")


def test_live_app_verifies_against_the_sidebar_scope(monkeypatch):
    verified = []
    monkeypatch.setattr(app_module, "load_retriever", lambda method: object())
    monkeypatch.setattr(app_module, "answer_question",
                        lambda question, *args, **kwargs: sample_answer(question))
    monkeypatch.setattr(app_module, "verify_answer",
                        lambda answer, *, parsed: verified.append(parsed) or answer)
    ui = AppTest.from_string(APP).run(timeout=30)
    ui.text_input[0].set_value("What was Apple's revenue in FY2024?").run()
    ui.sidebar.multiselect[0].set_value(["MSFT"]).run()
    ui.button[0].click().run()
    assert not ui.exception
    assert verified[0].tickers == ("MSFT",)
    assert verified[0].fiscal_years == (2024,)
