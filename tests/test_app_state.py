"""Issue #36: a repeated demo question is answered once, and the app says how long one took.

The app's other tests stand a stack in for ``load_stack`` itself, so they never
reach the memory this is about. These stand one in for ``build_stack``, one
level down, so the app's own ``load_stack`` wraps it as it wraps a real one.
"""

import threading
from dataclasses import replace

import pytest
from streamlit.testing.v1 import AppTest

import src.app.state as state_module
import src.rag.verify as verify_module
from src.config import PROJECT_ROOT
from src.app.state import Remembered, Stopwatch
from src.rag.generate import ProviderUnavailable
from src.retrieval.constants import FINAL_K
from src.retrieval.records import Query
from src.stack import DEFAULT_STACK, stack_config
from tests.sample_answers import sample_answer


APP = str(PROJECT_ROOT / "src" / "app" / "main.py")
QUESTION = "What was Apple's revenue in FY2024?"
# How long a test waits on a thread before calling it stuck.
WAIT = 10


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


@pytest.mark.parametrize("broken", [
    {"truncated": True},
    {"parse_error": "the output was cut off before the JSON closed"},
])
def test_an_answer_the_model_did_not_finish_is_not_remembered(broken):
    # Kept, a cut-off answer was every session's answer to that question until
    # the app restarted, and the healthy one behind it was never reached.
    class FirstBroken(CountingStack):
        def answer(self, question, **overrides):
            answer = super().answer(question, **overrides)
            return replace(answer, **broken) if len(self.asked) == 1 else answer

    stack = FirstBroken()
    remembered = Remembered(stack)
    first = remembered.answer(QUESTION, query=Query(QUESTION))
    second = remembered.answer(QUESTION, query=Query(QUESTION))
    assert first.truncated or first.parse_error      # shown for what it is
    assert not second.truncated and second.parse_error is None
    assert len(stack.asked) == 2
    assert remembered.answer(QUESTION, query=Query(QUESTION)) is second
    assert len(stack.asked) == 2


class SlowStack(CountingStack):
    """Answers only once ``release`` is set. ``arrived`` counts the requests
    that have reached it, and ``most_at_once`` how many were inside together."""

    def __init__(self, failures=()):
        super().__init__(failures=failures)
        self.arrived, self.release = threading.Semaphore(0), threading.Event()
        self.inside = self.most_at_once = 0
        self._count = threading.Lock()

    def answer(self, question, **overrides):
        with self._count:
            self.inside += 1
            self.most_at_once = max(self.most_at_once, self.inside)
        self.arrived.release()
        try:
            assert self.release.wait(WAIT), "the test never released the stack"
            return super().answer(question, **overrides)
        finally:
            with self._count:
                self.inside -= 1


def _in_threads(remembered, questions):
    """Ask each question on a thread of its own: the threads, and the list
    each puts what it got into, an answer or the error it met."""
    got = [None] * len(questions)

    def ask(index, question):
        try:
            got[index] = remembered.answer(question, query=Query(question))
        except ProviderUnavailable as error:
            got[index] = error

    threads = [threading.Thread(target=ask, args=(index, question), daemon=True)
               for index, question in enumerate(questions)]
    for thread in threads:
        thread.start()
    return threads, got


def _finish(threads):
    for thread in threads:
        thread.join(WAIT)
    assert not any(thread.is_alive() for thread in threads)


def test_two_sessions_asking_the_same_question_at_once_get_one_answer():
    stack = SlowStack()
    remembered = Remembered(stack)
    threads, got = _in_threads(remembered, [QUESTION, QUESTION])
    assert stack.arrived.acquire(timeout=WAIT)          # one is inside the stack
    # The other must not get in while the first is there. It is given time
    # to, and then the first is let finish.
    assert not stack.arrived.acquire(timeout=0.3)
    stack.release.set()
    _finish(threads)
    assert len(stack.asked) == 1 and stack.most_at_once == 1
    assert got[0] is got[1] and got[0].question == QUESTION


def test_different_questions_are_answered_at_the_same_time():
    stack = SlowStack()
    remembered = Remembered(stack)
    other = "What was Apple's net income in FY2024?"
    threads, got = _in_threads(remembered, [QUESTION, other])
    # Both are inside the stack before either is released.
    assert stack.arrived.acquire(timeout=WAIT) and stack.arrived.acquire(timeout=WAIT)
    assert stack.most_at_once == 2
    stack.release.set()
    _finish(threads)
    assert sorted(answer.question for answer in got) == sorted([QUESTION, other])


def test_a_session_waiting_on_a_failure_asks_in_its_turn(monkeypatch):
    # Only a session that found another asking the same question waits on its
    # event, so ``waiting`` is set once the second is waiting on the first.
    waiting = threading.Event()

    class Watched(threading.Event):
        def wait(self, timeout=None):
            waiting.set()
            return super().wait(timeout)

    monkeypatch.setattr(state_module, "Event", Watched)
    stack = SlowStack(failures=[ProviderUnavailable("Mistral's rate limit was reached")])
    remembered = Remembered(stack)
    threads, got = _in_threads(remembered, [QUESTION, QUESTION])
    assert stack.arrived.acquire(timeout=WAIT)
    # Released before the other got there, the first could fail and leave
    # nothing in flight, and the other would ask afresh without being woken.
    assert waiting.wait(WAIT), "the other session never waited on the first"
    stack.release.set()
    _finish(threads)
    # One met the failure, which was not kept. The other was woken, asked in
    # its turn and was answered. They were never inside together.
    errors = [item for item in got if isinstance(item, Exception)]
    answers = [item for item in got if not isinstance(item, Exception)]
    assert len(errors) == 1 and len(answers) == 1 and answers[0].question == QUESTION
    assert len(stack.asked) == 2 and stack.most_at_once == 1
    assert remembered.answer(QUESTION, query=Query(QUESTION)) is answers[0]
    assert remembered._being_answered == {}


def test_only_the_most_recently_asked_answers_are_kept():
    stack = CountingStack()
    remembered = Remembered(stack, limit=2)
    one, two, three = (f"What was Apple's revenue in FY202{n}?" for n in (2, 3, 4))
    remembered.answer(one, query=Query(one))
    remembered.answer(two, query=Query(two))
    remembered.answer(one, query=Query(one))          # one is now the more recent
    remembered.answer(three, query=Query(three))      # so two is the one dropped
    assert len(stack.asked) == 3
    remembered.answer(one, query=Query(one))
    assert len(stack.asked) == 3
    remembered.answer(two, query=Query(two))
    assert len(stack.asked) == 4
    with pytest.raises(ValueError, match="limit must be at least 1"):
        Remembered(stack, limit=0)


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
        stacks[config_id].built_with = kwargs
        return stacks[config_id]

    monkeypatch.setattr(state_module, "build_stack", build_stack)
    monkeypatch.setattr(state_module, "measured", lambda: [])
    monkeypatch.setattr(verify_module, "verify_answer", lambda answer, *, parsed: answer)
    state_module.load_stack.clear()
    yield stacks
    state_module.load_stack.clear()


def _timing_lines(ui):
    return [caption.value for caption in ui.main.caption
            if caption.value.startswith("Query completed in")]


def _ask(question=QUESTION):
    ui = AppTest.from_file(APP, default_timeout=30).run()
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
    build_stack = state_module.build_stack

    def failing_once(config_id, **kwargs):
        stack = build_stack(config_id, **kwargs)
        stack.failures.append(ProviderUnavailable("The model stopped answering"))
        return stack

    monkeypatch.setattr(state_module, "build_stack", failing_once)
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


def test_each_provider_is_built_once_over_the_same_indexes(built):
    # A page that offers both providers builds each the first time it is
    # asked, not on every Ask, and neither reads the indexes a second time.
    local = state_module.load_stack(DEFAULT_STACK, "ollama")
    local_stack = built[DEFAULT_STACK]
    hosted = state_module.load_stack(DEFAULT_STACK, "mistral")
    hosted_stack = built[DEFAULT_STACK]
    assert hosted is not local and hosted_stack is not local_stack
    assert (local_stack.built_with["provider"], hosted_stack.built_with["provider"]) == (
        "ollama", "mistral")
    # Neither builds its chat model with the stack: an answer that needs one does.
    assert local_stack.built_with["defer_llm"] and hosted_stack.built_with["defer_llm"]
    assert hosted_stack.built_with["parts"] is local_stack.built_with["parts"]
    assert state_module.load_stack(DEFAULT_STACK, "ollama") is local
    # Each remembers its own answers: one provider's is never given as the other's.
    local.answer(QUESTION, query=Query(QUESTION))
    hosted.answer(QUESTION, query=Query(QUESTION))
    assert (len(local_stack.asked), len(hosted_stack.asked)) == (1, 1)


def test_answer_models_reads_what_env_selects(monkeypatch):
    found = state_module.answer_models()
    assert (found.models, found.default, found.problem) == (
        {"ollama": "llama3.2:3b", "mistral": "ministral-8b-2512"}, "ollama", None)
    monkeypatch.setenv("LLM_PROVIDER", "mistral")
    monkeypatch.setenv("LLM_MODEL", "mistral-small-2506")
    found = state_module.answer_models()
    assert (found.models, found.default, found.problem) == (
        {"ollama": "llama3.2:3b", "mistral": "mistral-small-2506"}, "mistral", None)
    monkeypatch.setenv("LLM_PROVIDER", "mistrl")
    unknown = state_module.answer_models()
    assert unknown.default == "ollama" and "got 'mistrl'" in unknown.problem
    # The model was named for a provider that does not exist, so neither gets it.
    assert unknown.models == {"ollama": "llama3.2:3b", "mistral": "ministral-8b-2512"}


def test_answer_models_says_what_each_provider_lacks_and_builds_no_client(monkeypatch):
    # Read on every rerun of a page, so it asks each provider's settings and
    # nothing else: no client library imported, no client built.
    from src.rag.generate import _mistral_client

    # What is wrong and none of the advice, which is written for a command.
    assert state_module.answer_models().unready == {"mistral": "MISTRAL_API_KEY is not set"}
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    assert state_module.answer_models().unready == {}
    monkeypatch.setenv("LLM_NUM_GPU", "many")
    assert state_module.answer_models().unready == {
        "ollama": "LLM_NUM_GPU must be a whole number of layers, got 'many'"}
    assert _mistral_client.cache_info().currsize == 0


def test_a_request_is_everything_an_answer_was_asked_with():
    query = Query("What was revenue?", tickers=("AAPL",), fiscal_years=(2024,), items=("7",))
    request = state_module.Request.of(query, "C4", "ollama")
    assert request.question == "What was revenue?" and request.config_id == "C4"
    # Compared as the tuple it is, so one kept before Streamlit loaded the
    # module again after an edit still matches one built after.
    assert request == ("What was revenue?", ("AAPL",), (2024,), ("7",), "C4", "ollama")
    # Change any one of them and it is another request.
    others = [state_module.Request.of(changed, "C4", "ollama") for changed in (
        replace(query, text="What was net income?"), replace(query, tickers=()),
        replace(query, fiscal_years=(2023,)), replace(query, items=()))]
    others += [state_module.Request.of(query, "C3", "ollama"),
               state_module.Request.of(query, "C4", "mistral")]
    assert all(other != request for other in others) and len(set(others)) == len(others)
