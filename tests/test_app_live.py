"""The real app answers through a named configuration, the way the harness does.

The app no longer assembles a retriever and a generator of its own (#43): it
asks ``src.stack`` for a configuration by id and answers through it, so these
tests stand in a built ``Stack`` rather than patching the RAG entry point.
"""

from dataclasses import dataclass, replace
from typing import Any

from streamlit.testing.v1 import AppTest

import src.app.state as state_module
import src.rag.query as query_module
import src.rag.verify as verify_module
from src.config import PROJECT_ROOT
from src.rag.constants import ABSTAIN_PHRASE
from src.rag.generate import ProviderUnavailable
from src.rag.records import GenerationConfig
from src.stack import DEFAULT_STACK, SELECTABLE, stack_config
from tests.sample_answers import sample_answer


# The entry point as ``streamlit run`` is given it. The Ask page is a script
# that main.py runs, and it reads what it calls from these modules on every
# run, so a stand-in set on one of them is the one the page gets.
APP = str(PROJECT_ROOT / "src" / "app" / "main.py")
GENERATION = GenerationConfig("ollama", "llama3.2:3b", "grounded_v4")


@dataclass
class FakeStack:
    """A built configuration that answers without an index or a model."""

    config: Any
    retriever: Any = object()
    generation: GenerationConfig = GENERATION
    llm: Any = None
    asked: list = None
    loaded_for: list = None      # the provider of each Ask that loaded this stack

    def __post_init__(self):
        self.asked = []
        self.loaded_for = []

    def answer(self, question, **overrides):
        self.asked.append((question, overrides))
        return sample_answer(question)


def _standing_in(monkeypatch, config_id=DEFAULT_STACK, runs=()):
    """Stand a FakeStack in for whatever configuration the app selects."""
    built = {}

    def load_stack(selected, provider=None):
        built.setdefault(selected, FakeStack(config=stack_config(selected)))
        built[selected].loaded_for.append(provider)
        return built[selected]

    monkeypatch.setattr(state_module, "load_stack", load_stack)
    monkeypatch.setattr(state_module, "measured", lambda: list(runs))
    monkeypatch.setattr(verify_module, "verify_answer", lambda answer, *, parsed: answer)
    return built


def test_live_app_answers_through_the_selected_configuration(monkeypatch):
    built = _standing_in(monkeypatch)
    ui = AppTest.from_file(APP).run(timeout=30)
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
    ui = AppTest.from_file(APP).run(timeout=30)
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
    ui = AppTest.from_file(APP).run(timeout=30)
    assert not any("filters above are not applied" in c.value for c in ui.sidebar.caption)
    ui.sidebar.selectbox[0].set_value("C3").run()
    assert not ui.exception
    assert any("filters above are not applied" in c.value for c in ui.sidebar.caption)


def test_live_app_labels_a_configuration_with_the_run_that_measured_it(monkeypatch):
    # Criterion: a configuration named in results/ can be selected in the app.
    _standing_in(monkeypatch, runs=[("C1", "nightly-7 · C1 — section-aware BM25")])
    ui = AppTest.from_file(APP).run(timeout=30)
    assert not ui.exception
    assert "nightly-7 · C1 — section-aware BM25" in list(ui.sidebar.selectbox[0].options)
    # The default row has no run, so the app says so rather than implying one.
    assert any("No run under results/" in caption.value for caption in ui.sidebar.caption)


def test_selecting_a_measured_configuration_builds_that_configuration(monkeypatch):
    built = _standing_in(monkeypatch, runs=[("C1", "nightly-7 · C1 — section-aware BM25")])
    ui = AppTest.from_file(APP).run(timeout=30)
    ui.text_input[0].set_value("What was Apple's revenue in FY2024?").run()
    ui.sidebar.selectbox[0].set_value("C1").run()
    ui.button[0].click().run()
    assert not ui.exception
    assert "C1" in built and DEFAULT_STACK not in built
    assert built["C1"].config.retriever == "bm25"


def test_live_app_reports_missing_index_without_model_call(monkeypatch):
    def no_index(config_id, provider=None):
        raise FileNotFoundError("Build the local index first")

    monkeypatch.setattr(state_module, "load_stack", no_index)
    monkeypatch.setattr(state_module, "measured", lambda: [])
    ui = AppTest.from_file(APP).run(timeout=30)
    ui.text_input[0].set_value("Apple FY2024 revenue?").run()
    ui.button[0].click().run()
    assert not ui.exception
    assert "Build the local index first" in ui.error[0].value
    assert not ui.get("html")


def test_live_app_verifies_against_the_sidebar_scope(monkeypatch):
    verified = []
    _standing_in(monkeypatch)
    monkeypatch.setattr(verify_module, "verify_answer",
                        lambda answer, *, parsed: verified.append(parsed) or answer)
    ui = AppTest.from_file(APP).run(timeout=30)
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
    parse = query_module.parse_question
    monkeypatch.setattr(query_module, "parse_question",
                        lambda question, **options: read.append(options) or parse(
                            question, **options))
    _standing_in(monkeypatch)
    # The first question to need the labels reads them from the store, which
    # can take longer than AppTest's three seconds on a busy machine.
    ui = AppTest.from_file(APP, default_timeout=30).run()
    ui.text_input[0].set_value("What was Apple's gross profit in FY2024?").run()
    ui.button[0].click().run()
    assert not ui.exception
    assert read and all("facts_file" not in options for options in read)


def _ask(monkeypatch, answer, question="What was Apple's revenue in FY2024?"):
    """Ask the app one question, with ``answer`` standing in for the configuration's own."""
    _standing_in(monkeypatch)
    monkeypatch.setattr(FakeStack, "answer",
                        lambda self, question, **overrides: answer(question, **overrides))
    ui = AppTest.from_file(APP, default_timeout=30).run()
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
        raise ProviderUnavailable("The model stopped answering")

    ui = _ask(monkeypatch, answer)
    assert "The model stopped answering" in ui.error[0].value
    assert not ui.text
    assert not ui.get("html")


def _trace(ui):
    """The trace's steps, and every line written inside them.

    AppTest lists an expander drawn as a timeline step under ``status``.
    """
    return ([step.label for step in ui.main.status],
            [caption.value for step in ui.main.status for caption in step.caption])


def test_live_app_says_in_the_trace_how_the_question_was_read(monkeypatch):
    ui = _ask(monkeypatch, lambda question, *args, **kwargs: sample_answer(question))
    steps, lines = _trace(ui)
    assert steps[0] == "Read as: numeric question"
    assert lines[:3] == ["Question type: numeric", "Companies: AAPL", "Fiscal years: FY2024"]
    # In the scope the sidebar ends up with, which is what gets searched.
    ui.sidebar.multiselect[0].set_value(["MSFT"]).run()
    ui.button[0].click().run()
    assert not ui.exception
    assert _trace(ui)[1][:3] == [
        "Question type: numeric", "Companies: MSFT", "Fiscal years: FY2024"]


def test_live_app_names_the_filings_a_split_question_was_searched_in(monkeypatch):
    ui = _ask(monkeypatch, lambda question, *args, **kwargs: replace(
        sample_answer(question), sub_questions=("AAPL FY2024", "MSFT FY2024")),
        question="Compare Apple and Microsoft's revenue in FY2024")
    assert "Searched one filing at a time: AAPL FY2024, MSFT FY2024" in _trace(ui)[1]


def test_live_app_searches_with_hybrid_unless_another_row_is_chosen(monkeypatch):
    # The row the app opens on is the one the 48 test questions were measured
    # best with (README, "How often the answers are right"), so a change of
    # DEFAULT_STACK to a BM25 row has to be made on purpose.
    built = _standing_in(monkeypatch)
    ui = AppTest.from_file(APP).run(timeout=30)
    ui.text_input[0].set_value("What was Apple's revenue in FY2024?").run()
    ui.button[0].click().run()
    ui.sidebar.selectbox[0].set_value("C1").run()
    ui.button[0].click().run()
    assert not ui.exception
    assert [stack.config.retriever for stack in built.values()] == ["hybrid", "bm25"]


# --- the Ask page (#38) ----------------------------------------------------------

def _badges(ui):
    """The line of resolved filters under the question box, as it is written."""
    return [text.value for text in ui.main.markdown if "Searching" in text.value]


def _summary(ui):
    """The badges of the row above an answer. AppTest lists a badge as markdown."""
    return [text.value for text in ui.main.markdown if text.value.startswith(":")
            and "Searching" not in text.value]


def test_ask_page_shows_the_filters_a_question_resolved_to_before_it_is_asked(monkeypatch):
    built = _standing_in(monkeypatch)
    ui = AppTest.from_file(APP, default_timeout=30).run()
    assert not _badges(ui)                      # nothing to search for yet
    ui.text_input[0].set_value("What was Apple's revenue in FY2024, Item 7?").run()
    assert not ui.exception
    line = _badges(ui)[0]
    for badge in (":blue-badge[AAPL]", ":blue-badge[FY2024]", ":blue-badge[Item 7]",
                  ":violet-badge[numeric question]"):
        assert badge in line
    assert not built                            # shown without building or asking anything
    # A filter cleared by hand reads as searching everything, not as nothing.
    ui.sidebar.multiselect[0].set_value([]).run()
    assert ":gray-badge[every company]" in _badges(ui)[0]


def test_ask_page_shows_no_filters_for_a_row_that_applies_none(monkeypatch):
    _standing_in(monkeypatch)
    ui = AppTest.from_file(APP, default_timeout=30).run()
    ui.text_input[0].set_value("What was Apple's revenue in FY2024?").run()
    ui.sidebar.selectbox[0].set_value("C3").run()
    assert ":gray-badge[every company]" in _badges(ui)[0]
    assert ":blue-badge[AAPL]" not in _badges(ui)[0]


def test_ask_page_lists_the_passages_an_answer_was_written_from(monkeypatch):
    ui = _ask(monkeypatch, lambda question, *args, **kwargs: sample_answer(question))
    steps, _ = _trace(ui)
    assert steps == [
        "Read as: numeric question",
        "Found 1 passage in 1 filing with hybrid retrieval, 1 cited",
        "Written by test",
        "Checked: 1 supported, 1 mismatch",
    ]
    passages = ui.main.dataframe[0].value
    assert list(passages["Source"]) == ["[1]"]
    assert list(passages["Chunk"]) == ["test-passage"]
    assert bool(passages["Cited"][0])
    # Above the answer: the outcome, the configuration and what wrote it.
    assert _summary(ui) == [
        ":green-badge[:material/check_circle: Answered]",
        f":gray-badge[:material/tune: {DEFAULT_STACK} · {stack_config(DEFAULT_STACK).name}]",
        ":gray-badge[:material/edit_note: test (ollama)]",
    ]
    assert [heading.value for heading in ui.main.subheader] == ["Answer", "Retrieval trace"]


def test_ask_page_states_an_abstention_and_still_shows_its_trace(monkeypatch):
    def declined(question, *args, **kwargs):
        return replace(sample_answer(question), text=ABSTAIN_PHRASE, abstained=True,
                       abstention_reason="model_declined", sentences=(), citations=(),
                       verification=None)

    ui = _ask(monkeypatch, declined)
    assert _summary(ui)[0] == ":orange-badge[:material/do_not_disturb_on: No answer]"
    assert ui.warning[0].value.startswith("**No answer from the filings.**")
    assert "does not support an answer" in ui.warning[0].value
    assert not ui.get("html")                   # no empty card under it
    steps, _ = _trace(ui)
    assert steps[1].startswith("Found 1 passage")
    assert steps[2] == "test found no answer in the passages"


def test_ask_page_says_a_refused_question_was_not_searched(monkeypatch):
    def refused(question, *args, **kwargs):
        return replace(sample_answer(question), text=ABSTAIN_PHRASE, abstained=True,
                       abstention_reason="beyond_the_filings", sentences=(), citations=(),
                       passages=(), verification=None)

    ui = _ask(monkeypatch, refused, question="Should I buy Apple stock?")
    assert "gives no advice" in ui.warning[0].value
    steps, _ = _trace(ui)
    assert steps[1:3] == ["Not searched: no filing could answer this", "No model was asked"]
    assert not ui.main.dataframe
    # Nothing wrote it, so the row above names no model.
    assert len(_summary(ui)) == 2


def test_ask_with_no_question_asks_nothing(monkeypatch):
    built = _standing_in(monkeypatch)
    ui = AppTest.from_file(APP, default_timeout=30).run()
    ui.button[0].click().run()
    assert not ui.exception and not built
    assert any("Type a question first" in caption.value for caption in ui.main.caption)


def test_the_app_starts_from_its_file_without_the_project_on_the_path():
    # ``streamlit run src/app/main.py`` puts src/app on sys.path, not the
    # project root, and the app then failed on its first ``import src``.
    import subprocess
    import sys

    script = (
        "import runpy, sys;"
        f"sys.path[:] = [p for p in sys.path if p not in ('', {str(PROJECT_ROOT)!r})];"
        "import streamlit as st;"
        "st.navigation = lambda pages: type('P', (), {'run': lambda self: None})();"
        f"runpy.run_path({APP!r}, run_name='__main__');"
        "import src.app.state"
    )
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                          cwd=PROJECT_ROOT.parent)
    assert done.returncode == 0, done.stderr[-600:]


# --- what writes the answer is a choice on the page -----------------------------

def _provider(ui):
    """The sidebar's Answer model box, and the line under it."""
    box = ui.sidebar.button_group[0]
    line = next(caption.value for caption in ui.sidebar.caption
                if "API" in caption.value or "this computer" in caption.value)
    return box, line


def test_ask_page_opens_on_the_local_model_and_can_be_switched_to_mistral(monkeypatch):
    # The provider was .env's alone to choose. With a Mistral key in .env and
    # no LLM_PROVIDER line, every answer waited minutes for the local model
    # and nothing on the page could change it.
    built = _standing_in(monkeypatch)
    ui = AppTest.from_file(APP, default_timeout=30).run()
    box, line = _provider(ui)
    assert list(box.options) == ["Ollama", "Mistral"] and box.value == "ollama"
    assert line.startswith("llama3.2:3b, on this computer")
    ui.text_input[0].set_value("What was Apple's revenue in FY2024?").run()
    ui.button[0].click().run()
    assert built[DEFAULT_STACK].loaded_for == ["ollama"]
    assert len(ui.get("html")) == 1

    box.set_value("mistral").run()
    assert not ui.exception
    assert _provider(ui)[1].startswith("ministral-8b-2512, on Mistral's API")
    # An answer one provider wrote is not left under the other's name.
    assert not ui.get("html") and ui.info
    ui.button[0].click().run()
    assert built[DEFAULT_STACK].loaded_for == ["ollama", "mistral"]
    assert len(ui.get("html")) == 1


def test_ask_page_opens_on_the_provider_env_names(monkeypatch):
    # LLM_MODEL is the model of the provider .env selects, so the other one
    # is offered with its own default.
    monkeypatch.setenv("LLM_PROVIDER", "mistral")
    monkeypatch.setenv("LLM_MODEL", "mistral-small-2506")
    _standing_in(monkeypatch)
    ui = AppTest.from_file(APP, default_timeout=30).run()
    assert not ui.exception
    box, line = _provider(ui)
    assert box.value == "mistral" and line.startswith("mistral-small-2506, on Mistral's API")
    box.set_value("ollama").run()
    assert _provider(ui)[1].startswith("llama3.2:3b, on this computer")


def test_a_mistyped_provider_in_env_is_said_and_does_not_stop_the_page(monkeypatch):
    # Every Ask used to fail on it. The page now names what it will ask, so it
    # opens on the default provider, says what was wrong, and still answers.
    monkeypatch.setenv("LLM_PROVIDER", "mistrl")
    built = _standing_in(monkeypatch)
    ui = AppTest.from_file(APP, default_timeout=30).run()
    assert not ui.exception
    assert _provider(ui)[0].value == "ollama"
    said = ui.sidebar.warning[0].value
    assert "LLM_PROVIDER must be one of ollama, mistral, got 'mistrl'" in said
    ui.text_input[0].set_value("What was Apple's revenue in FY2024?").run()
    ui.button[0].click().run()
    assert not ui.exception and not ui.error
    assert built[DEFAULT_STACK].loaded_for == ["ollama"]


def test_picking_mistral_without_a_key_says_so_before_anything_is_loaded(monkeypatch):
    # Through the app's own load_stack: the key is checked when the stack is
    # built, ahead of the indexes, so this reads none.
    monkeypatch.setattr(state_module, "measured", lambda: [])
    state_module.load_stack.clear()
    ui = AppTest.from_file(APP, default_timeout=30).run()
    ui.sidebar.button_group[0].set_value("mistral").run()
    ui.text_input[0].set_value("What risks does Apple describe in its FY2024 10-K?").run()
    ui.button[0].click().run()
    state_module.load_stack.clear()
    assert not ui.exception
    assert "MISTRAL_API_KEY is not set" in ui.error[0].value
    assert not ui.get("html") and not ui.main.status


# --- Streamlit's file watcher and a library's lazy modules ----------------------

def _watcher(monkeypatch):
    """The watcher's own reading of a module's paths, and what it would have
    logged while reading, recorded in place of being logged."""
    import streamlit.watcher.local_sources_watcher as watcher

    warned = []
    monkeypatch.setattr(watcher._LOGGER, "warning", lambda *args, **kwargs: warned.append(args))
    return watcher.get_module_paths, warned


def _lazy_alias(name):
    """A module as transformers 5 makes an alias: no file, and any attribute it
    lacks looked up by importing a module that needs torchvision."""
    import types

    module = types.ModuleType(name)
    module.__file__ = None

    def missing(attribute):
        raise ModuleNotFoundError("No module named 'torchvision'")

    module.__getattr__ = missing
    return module


def test_the_file_watcher_reads_a_lazy_alias_without_setting_off_its_import(monkeypatch):
    # After each run Streamlit's watcher asks every loaded module for its
    # __path__. transformers answers for an alias by importing what the alias
    # stands for, and 102 of those imports need torchvision, so the terminal
    # filled with tracebacks after the first question.
    import sys
    import types

    paths_of, warned = _watcher(monkeypatch)
    name = "transformers.models.example.image_processing_example_fast"
    # As the watcher met one: nothing read, and a traceback logged.
    assert paths_of(_lazy_alias(name)) == set() and len(warned) == 1

    alias = _lazy_alias(name)
    plain = types.ModuleType("transformers.models.example.modeling_example")
    plain.__file__ = "modeling_example.py"
    monkeypatch.setitem(sys.modules, name, alias)
    monkeypatch.setitem(sys.modules, plain.__name__, plain)
    built = _standing_in(monkeypatch)
    ui = AppTest.from_file(APP, default_timeout=30).run()
    assert not ui.exception
    # After a run of the app the same reading logs nothing more.
    assert paths_of(alias) == set() and len(warned) == 1
    # Only an alias is touched.
    assert "__path__" not in vars(plain)

    # A run that stops part way is still followed by the watcher, and the
    # embedding model is first loaded inside an Ask.
    import streamlit as st

    later = _lazy_alias("transformers.models.later.image_processing_later_fast")

    def loads_and_stops(self, question, **overrides):
        sys.modules[later.__name__] = later
        st.stop()

    monkeypatch.setattr(FakeStack, "answer", loads_and_stops)
    ui.text_input[0].set_value("What was Apple's revenue in FY2024?").run()
    ui.button[0].click().run()
    sys.modules.pop(later.__name__, None)
    assert built and not ui.exception
    assert paths_of(later) == set() and len(warned) == 1


def test_the_file_watcher_reads_transformers_itself_quietly_after_a_run(monkeypatch):
    # The library as installed, whatever its version: the alias above is shaped
    # as transformers 5.17 shapes one, and this is what fails if a later
    # version shapes them some other way.
    import sys

    # Installed with sentence-transformers. It registers its aliases as it is imported.
    import transformers  # noqa: F401

    paths_of, warned = _watcher(monkeypatch)
    _standing_in(monkeypatch)
    ui = AppTest.from_file(APP, default_timeout=30).run()
    assert not ui.exception
    loaded = set(sys.modules)
    for name, module in list(sys.modules.items()):
        if name.startswith("transformers"):
            paths_of(module)
    assert warned == []
    assert set(sys.modules) == loaded        # and reading them imported nothing
