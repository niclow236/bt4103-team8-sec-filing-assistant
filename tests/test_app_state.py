"""Issue #36: a repeated demo question is answered once, and the app says how long one took.

The app's other tests stand a stack in for ``load_stack`` itself, so they never
reach the memory this is about. These stand one in for ``build_stack``, one
level down, so the app's own ``load_stack`` wraps it as it wraps a real one.
"""

import pytest
from streamlit.testing.v1 import AppTest

import src.app.app as app_module
import src.app.state as state_module
from src.app.state import Remembered, Stopwatch
from src.rag.generate import ProviderUnavailable
from src.retrieval.constants import FINAL_K
from src.retrieval.records import Query
from src.stack import DEFAULT_STACK, stack_config
from tests.sample_answers import sample_answer


APP = "from src.app.app import main\nmain()"
QUESTION = "What was Apple's revenue in FY2024?"


class CountingStack:
    """A built configuration that answers without an index or a model, and
    keeps what it was asked. ``failures`` are raised first, one per request."""

    def __init__(self, config=None, failures=()):
        self.config = config
        self.failures = list(failures)
        self.asked = []

    def answer(self, question, **overrides):
        self.asked.append((question, overrides))
        if self.failures:
            raise self.failures.pop(0)
        return sample_answer(question)


# --- the memory ------------------------------------------------------------------

def test_a_repeated_question_is_asked_of_the_stack_once():
    stack = CountingStack()
    remembered = Remembered(stack)
    first = remembered.answer(QUESTION, query=Query(QUESTION, tickers=("AAPL",)))
    again = remembered.answer(QUESTION, query=Query(QUESTION, tickers=("AAPL",)))
    assert again is first
    assert len(stack.asked) == 1


def test_another_question_filter_or_setting_is_asked():
    stack = CountingStack()
    remembered = Remembered(stack)
    remembered.answer(QUESTION, query=Query(QUESTION, tickers=("AAPL",)))
    # Another question, another company, another Item, another passage budget,
    # and a setting the caller overrides: each is a different answer.
    remembered.answer("What was Apple's net income in FY2024?",
                      query=Query(QUESTION, tickers=("AAPL",)))
    remembered.answer(QUESTION, query=Query(QUESTION, tickers=("MSFT",)))
    remembered.answer(QUESTION, query=Query(QUESTION, tickers=("AAPL",), items=("8",)))
    remembered.answer(QUESTION, query=Query(QUESTION, tickers=("AAPL",), top_k=4))
    remembered.answer(QUESTION, query=Query(QUESTION, tickers=("AAPL",)), min_score=0.5)
    assert len(stack.asked) == 6


def test_how_the_question_was_read_and_where_it_streams_do_not_make_it_another():
    stack = CountingStack()
    remembered = Remembered(stack)
    shown = []
    remembered.answer(QUESTION, query=Query(QUESTION), parsed="read once", on_token=shown.append)
    remembered.answer(QUESTION, query=Query(QUESTION), parsed="read again", on_token=print)
    assert len(stack.asked) == 1
    # The first request carries both to the stack, which is what streams.
    assert stack.asked[0][1]["parsed"] == "read once"
    assert stack.asked[0][1]["on_token"] == shown.append


def test_a_failure_is_not_remembered():
    # A busy provider now should not be the answer for the rest of the demo.
    stack = CountingStack(failures=[ProviderUnavailable("Mistral's rate limit was reached")])
    remembered = Remembered(stack)
    with pytest.raises(ProviderUnavailable):
        remembered.answer(QUESTION, query=Query(QUESTION))
    assert remembered.answer(QUESTION, query=Query(QUESTION)).question == QUESTION
    assert len(stack.asked) == 2


def test_each_configuration_remembers_its_own_answers():
    one, other = CountingStack(), CountingStack()
    Remembered(one).answer(QUESTION, query=Query(QUESTION))
    Remembered(other).answer(QUESTION, query=Query(QUESTION))
    assert (len(one.asked), len(other.asked)) == (1, 1)


def test_a_stopwatch_times_the_block_it_wraps(monkeypatch):
    clock = iter([10.0, 12.5])
    monkeypatch.setattr(state_module, "perf_counter", lambda: next(clock))
    watch = Stopwatch()
    assert watch.seconds is None
    with watch as running:
        assert running is watch and watch.seconds is None
    assert watch.seconds == 2.5


# --- in the app ------------------------------------------------------------------

@pytest.fixture
def built(monkeypatch):
    """The app's own ``load_stack`` over stacks that count, by configuration id.

    ``load_stack`` keeps what it builds for the process, so it is emptied
    before and after: an answer one test was given must not reach the next.
    """
    stacks = {}

    def build_stack(config_id, **kwargs):
        stacks[config_id] = CountingStack(stack_config(config_id))
        return stacks[config_id]

    monkeypatch.setattr(app_module, "build_stack", build_stack)
    monkeypatch.setattr(app_module, "measured", lambda: [])
    monkeypatch.setattr(app_module, "verify_answer", lambda answer, *, parsed: answer)
    app_module.load_stack.clear()
    yield stacks
    app_module.load_stack.clear()


def _timing_lines(ui):
    return [caption.value for caption in ui.main.caption
            if caption.value.startswith("Query completed in")]


def _ask(question=QUESTION):
    ui = AppTest.from_string(APP, default_timeout=30).run()
    ui.text_input[0].set_value(question).run()
    ui.button[0].click().run()
    assert not ui.exception
    return ui


def test_live_app_answers_a_repeated_question_from_memory(built):
    ui = _ask()
    ui.button[0].click().run()
    assert not ui.exception
    assert len(built[DEFAULT_STACK].asked) == 1
    assert len(ui.get("html")) == 1
    # In another browser session too: the memory is the process's.
    other = _ask()
    assert len(built[DEFAULT_STACK].asked) == 1
    assert len(other.get("html")) == 1


def test_live_app_asks_again_when_the_filters_or_the_configuration_change(built):
    ui = _ask()
    ui.sidebar.multiselect[2].set_value(["8"]).run()
    ui.button[0].click().run()
    assert [overrides["query"].items for _, overrides in built[DEFAULT_STACK].asked] == [
        (), ("8",)]
    ui.sidebar.selectbox[0].set_value("C1").run()
    ui.button[0].click().run()
    assert not ui.exception
    assert len(built["C1"].asked) == 1 and len(built[DEFAULT_STACK].asked) == 2


def test_live_app_asks_for_the_passages_the_generator_is_given(built):
    # A cached path once rebuilt the query and dropped it to the default 10.
    _ask()
    assert built[DEFAULT_STACK].asked[0][1]["query"].top_k == FINAL_K


def test_live_app_asks_again_after_a_failure(built, monkeypatch):
    build_stack = app_module.build_stack

    def failing_once(config_id, **kwargs):
        stack = build_stack(config_id, **kwargs)
        stack.failures.append(ProviderUnavailable("The model stopped answering"))
        return stack

    monkeypatch.setattr(app_module, "build_stack", failing_once)
    ui = _ask()
    assert "The model stopped answering" in ui.error[0].value
    assert not ui.get("html") and not _timing_lines(ui)
    ui.button[0].click().run()
    assert not ui.error and len(ui.get("html")) == 1
    assert len(built[DEFAULT_STACK].asked) == 2


def test_live_app_says_how_long_the_answer_took_for_as_long_as_it_shows_it(built):
    ui = _ask()
    assert len(_timing_lines(ui)) == 1
    # Drawn only in the run where Ask was pressed, the line went away on the
    # next rerun while the answer stayed.
    ui.sidebar.multiselect[2].set_value(["8"]).run()
    assert not ui.get("html") and not _timing_lines(ui)
    ui.sidebar.multiselect[2].set_value([]).run()
    assert not ui.exception
    assert len(ui.get("html")) == 1 and len(_timing_lines(ui)) == 1
