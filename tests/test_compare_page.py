"""Compare registered stacks without hosted model calls or real indexes (#39)."""

from dataclasses import replace

import pytest
from streamlit.testing.v1 import AppTest

from src.app import state
from src.app.components import comparison_rows
from src.config import PROJECT_ROOT
from src.rag.generate import ProviderUnavailable
from src.stack import SELECTABLE, STACKS
from tests.sample_answers import sample_answer
from tests.test_app_live import FakeStack, _standing_in


PAGE = PROJECT_ROOT / "src/app/app_pages/compare.py"
QUESTION = "What risks does Apple describe in FY2024?"


def _page(monkeypatch, failure=None, empty=False):
    built = _standing_in(monkeypatch)

    def answer(self, question, **kwargs):
        self.asked.append((question, kwargs))
        if self.config.id == "C1" and failure is not None:
            raise failure
        result = sample_answer(question)
        shared = result.passages[0]
        own = replace(shared, chunk_id=f"only-{self.config.id}", rank=2,
                      text=f"Evidence unique to {self.config.id}.")
        if empty:
            return replace(result, passages=(), citations=(), abstained=True,
                           abstention_reason="no_evidence", text="I cannot answer.")
        return replace(result, passages=(shared, own))

    monkeypatch.setattr(FakeStack, "answer", answer)
    durations = iter((1.25, 2.5, 3.0, 4.0))

    class Stopwatch:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.seconds = next(durations)

    monkeypatch.setattr(state, "Stopwatch", Stopwatch)
    return AppTest.from_file(PAGE, default_timeout=30).run(), built


def _submit(ui):
    ui.text_input[0].set_value(QUESTION).run()
    return ui.button[0].click().run()


def test_compare_uses_registry_and_shows_answers_evidence_and_separate_timings(monkeypatch):
    ui, built = _page(monkeypatch)
    pickers = [box for box in ui.sidebar.selectbox if "configuration" in box.label.lower()]
    assert len(pickers) == 2
    assert all(box.options == [f"{key} — {STACKS[key].name}" for key in SELECTABLE] for box in pickers)
    _submit(ui)
    assert not ui.exception
    assert set(built) == {"C1", "C4"}
    for stack in built.values():
        question, options = stack.asked[0]
        assert question == QUESTION
        assert options["query"].tickers == ("AAPL",)
        assert options["parsed"].question == QUESTION
        assert stack.loaded_for == ["ollama"]
    assert len(ui.get("html")) == 2
    assert len(ui.dataframe) == 2
    assert ui.dataframe[0].value["Presence"].tolist() == ["Shared", "Only left"]
    assert ui.dataframe[1].value["Presence"].tolist() == ["Shared", "Only right"]
    captions = [caption.value for caption in ui.caption]
    assert "1 shared passages · 1 only left · 1 only right" in captions
    assert "Query completed in 1.25s" in captions
    assert "Query completed in 2.50s" in captions
    assert any("searches all filings" in caption for caption in captions)
    # Rerunning the page retains the pair without asking either configuration again.
    ui.run()
    assert len(ui.get("html")) == 2
    assert all(len(stack.asked) == 1 for stack in built.values())


@pytest.mark.parametrize("setting", ["question", "configuration", "filters", "provider"])
def test_changing_either_request_hides_both_old_answers(monkeypatch, setting):
    ui, built = _page(monkeypatch)
    _submit(ui)
    assert len(ui.get("html")) == 2
    if setting == "question":
        ui.text_input[0].set_value("What risks does Microsoft describe in FY2024?").run()
    elif setting == "configuration":
        ui.sidebar.selectbox(key="compare:right").set_value("C2").run()
    elif setting == "filters":
        ui.sidebar.multiselect[1].set_value([2023]).run()
    else:
        ui.sidebar.button_group[0].set_value("mistral").run()
    assert not ui.exception
    assert not ui.get("html")
    assert all(len(stack.asked) == 1 for stack in built.values())


@pytest.mark.parametrize("failure", [ProviderUnavailable("Model unavailable"), OSError("Missing index")])
def test_one_failure_does_not_hide_other_configuration(monkeypatch, failure):
    ui, built = _page(monkeypatch, failure=failure)
    _submit(ui)
    assert not ui.exception
    assert str(failure) in ui.error[0].value
    assert len(ui.get("html")) == 1
    assert ui.dataframe[0].value["Presence"].tolist() == ["Not compared", "Not compared"]
    assert set(built) == {"C1", "C4"}


def test_blank_or_identical_configuration_does_not_ask_stacks(monkeypatch):
    ui, built = _page(monkeypatch)
    ui.button[0].click().run()
    assert not built
    ui.sidebar.selectbox(key="compare:left").set_value("C4").run()
    _submit(ui)
    assert not ui.exception and not built
    assert any("different configurations" in info.value for info in ui.info)


def test_empty_evidence_and_abstentions_are_shown_on_both_sides(monkeypatch):
    ui, _ = _page(monkeypatch, empty=True)
    _submit(ui)
    assert not ui.exception
    assert not ui.dataframe
    assert sum("No passages" in caption.value for caption in ui.caption) == 2


def test_passage_overlap_uses_chunk_identity_not_rank_or_text():
    left = sample_answer(QUESTION)
    shared = left.passages[0]
    right = replace(left, passages=(replace(shared, rank=8, score=0.02),
                                   replace(shared, chunk_id="different-id", rank=9)))
    assert [row["Presence"] for row in comparison_rows(right, left, "right")] == ["Shared", "Only right"]


def test_compare_is_registered_without_removing_ask_or_browse(monkeypatch):
    _standing_in(monkeypatch)
    ui = AppTest.from_file(PROJECT_ROOT / "src/app/main.py", default_timeout=30).run()
    ui.switch_page("app_pages/compare.py").run()
    assert not ui.exception
    assert ui.title[0].value == "Compare configurations"
    ui.switch_page("app_pages/ask.py").run()
    assert not ui.exception
    assert ui.title[0].value == "SEC Filing Assistant"
