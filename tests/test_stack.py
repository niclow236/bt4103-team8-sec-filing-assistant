"""Issue #43: the app and the harness assemble from one configuration registry.

Nothing here loads an index or a model: ``build_stack`` takes an already-built
retriever and chat model, which is how the app's tests stand one in and how a
caller that loaded one once reuses it. What is under test is the assembly --
which configuration a id names, what a stack does with it, and that the three
callers no longer each know how to build a retriever.
"""

import json
from dataclasses import replace

import pytest

from src.rag.records import GenerationConfig
from src.retrieval.records import Query
from src.stack import (
    DEFAULT_STACK,
    RETRIEVERS,
    SELECTABLE,
    STACKS,
    MeasuredRun,
    Stack,
    StackConfig,
    build_retriever,
    build_stack,
    measured_runs,
    stack_config,
)

GENERATION = GenerationConfig("ollama", "llama3.2:3b", "grounded_v4")


class StubRetriever:
    name = "stub"

    def search(self, query, k=None):
        return []


def _stack(config_id=DEFAULT_STACK, **settings):
    return build_stack(config_id, retriever=StubRetriever(), llm=object(), **settings)


# --- the registry ----------------------------------------------------------------

def test_every_configuration_names_a_buildable_retriever():
    # "bm25-fixed-size" is the ablation runner's own baseline, built there.
    for config in STACKS.values():
        assert config.retriever in RETRIEVERS or config.retriever == "bm25-fixed-size"


def test_the_registry_holds_the_ablation_rows_the_runner_has_always_had():
    assert list(STACKS) == ["C0", "C1", "C2", "C3", "C4"]
    assert STACKS["C4"].metadata_filter is True
    assert STACKS["C3"].metadata_filter is False
    assert STACKS["C0"].chunking == "fixed-size"


def test_the_default_is_the_only_row_whose_retrieval_the_app_can_show():
    # The sidebar's company and year filters are the metadata filter, so a row
    # with it off would ignore what the user selected.
    assert STACKS[DEFAULT_STACK].metadata_filter is True


def test_only_rows_the_shared_path_can_build_are_selectable():
    # C0's baseline is built by the ablation runner alone, so offering it would
    # promise a stack build_stack refuses -- and the error blamed a retriever
    # the user never typed.
    assert SELECTABLE == ("C1", "C2", "C3", "C4")
    assert DEFAULT_STACK in SELECTABLE


def test_an_unknown_id_says_what_the_ids_are():
    with pytest.raises(ValueError, match="unknown configuration 'C9'.*C0, C1"):
        stack_config("C9")


def test_a_configuration_serialises_for_a_results_file():
    data = STACKS["C4"].to_dict()
    assert data["id"] == "C4" and data["retriever"] == "hybrid"
    assert set(data) == {"id", "name", "retriever", "chunking", "metadata_filter",
                         "top_k", "min_score", "use_facts", "use_decomposition",
                         "use_refusal"}
    json.dumps(data)


# --- building ---------------------------------------------------------------------

def test_building_a_configuration_carries_its_settings_onto_the_stack():
    stack = _stack("C4")
    assert isinstance(stack, Stack)
    assert stack.config.id == "C4"
    assert stack.config.use_facts is True and stack.config.use_decomposition is True
    assert stack.config.use_refusal is True
    assert stack.generation.provider in {"ollama", "mistral"}


def test_settings_given_at_build_time_override_the_configuration():
    stack = _stack("C4", top_k=3, min_score=0.5, use_facts=False, use_decomposition=False,
                   use_refusal=False)
    assert (stack.config.top_k, stack.config.min_score) == (3, 0.5)
    assert stack.config.use_facts is False and stack.config.use_decomposition is False
    assert stack.config.use_refusal is False
    # The registry itself is untouched by a build.
    assert STACKS["C4"].top_k != 3 and STACKS["C4"].use_facts is True


def test_an_unknown_setting_is_refused_rather_than_ignored():
    with pytest.raises(ValueError, match="unknown setting"):
        _stack("C4", retreiver="bm25")


def test_overriding_the_retriever_is_recorded_on_the_configuration():
    stack = _stack("C4", retriever_key="bm25")
    assert stack.config.retriever == "bm25"
    # The rest of the row is untouched, so a report still says it was C4 with
    # the retriever swapped rather than silently becoming another row.
    assert stack.config.id == "C4" and stack.config.metadata_filter is True
    assert stack.config.to_dict()["retriever"] == "bm25"


def test_the_generator_is_settled_before_an_index_is_read(monkeypatch):
    """A bad provider setting stops a caller before it has read an index.

    The evaluation command's own tests require this of it, and it is true of
    the command because it is true here: reading .env is cheap and loading an
    index is not, so a mistyped LLM_PROVIDER should not cost a corpus read.
    """
    import sys

    import src.stack as stack_module

    # src.rag re-exports the generate *function*, so the module attribute is
    # shadowed; sys.modules is how to reach the module to patch it.
    generate_module = sys.modules["src.rag.generate"]
    order = []
    monkeypatch.setattr(stack_module, "build_retriever",
                        lambda key, **kw: order.append("retriever"))
    monkeypatch.setattr(generate_module, "chat_model",
                        lambda config: order.append("generator"))
    build_stack("C4")
    assert order == ["generator", "retriever"]


def test_a_deferred_chat_model_is_not_built_with_the_stack(monkeypatch):
    # The app's stack. A figure looked up in the facts store asks no model, so
    # building the stack must not need the key of the provider picked. With
    # the model built here, Mistral picked and no key, every Ask was refused.
    import sys

    from src.rag.generate import ProviderUnavailable

    with pytest.raises(ProviderUnavailable, match="MISTRAL_API_KEY is not set"):
        build_stack("C4", retriever=StubRetriever(), provider="mistral")

    generate_module = sys.modules["src.rag.generate"]
    monkeypatch.setattr(generate_module, "chat_model",
                        lambda config: pytest.fail("the stack built a chat model"))
    stack = build_stack("C4", retriever=StubRetriever(), provider="mistral", defer_llm=True)
    assert stack.llm is None
    assert (stack.generation.provider, stack.generation.model) == ("mistral", "ministral-8b-2512")
    # A chat model handed in is still the one the stack uses.
    given = object()
    assert build_stack("C4", retriever=StubRetriever(), llm=given, defer_llm=True).llm is given
    # And a provider that does not exist is still refused where the stack is built.
    with pytest.raises(ValueError, match="the provider argument must be one of ollama, mistral"):
        build_stack("C4", retriever=StubRetriever(), provider="openai", defer_llm=True)


def test_overriding_the_retriever_does_not_load_the_one_it_replaced(monkeypatch):
    # --config C4 --retriever bm25 used to build the hybrid stack, reading the
    # dense index, and then throw it away.
    import src.stack as stack_module

    asked = []
    monkeypatch.setattr(stack_module, "build_retriever",
                        lambda key, **kw: asked.append(key) or f"<{key}>")
    # llm passed in, so no model is built either.
    stack = build_stack("C4", retriever_key="bm25", llm=object())
    assert asked == ["bm25"]
    assert stack.retriever == "<bm25>"


@pytest.mark.parametrize("key", ["", "bm52", "rerank", "bm25-fixed-size"])
def test_build_retriever_refuses_a_key_it_cannot_build(key):
    with pytest.raises(ValueError, match="unknown retriever"):
        build_retriever(key)


def test_build_retriever_knows_every_key_a_configuration_can_name(monkeypatch):
    # Each key reaches its own constructor, and asking for BM25 does not load
    # the dense index.
    import src.retrieval.bm25 as bm25_module
    import src.retrieval.dense as dense_module
    import src.retrieval.hybrid as hybrid_module

    loaded = []
    monkeypatch.setattr(bm25_module.BM25Retriever, "load",
                        classmethod(lambda cls, **kw: loaded.append("bm25") or "BM25"))
    monkeypatch.setattr(dense_module.DenseRetriever, "load",
                        classmethod(lambda cls, **kw: loaded.append("dense") or "DENSE"))
    monkeypatch.setattr(hybrid_module, "HybridRetriever",
                        lambda a, b: loaded.append("hybrid") or "HYBRID")

    assert build_retriever("bm25") == "BM25"
    assert loaded == ["bm25"]
    loaded.clear()
    assert build_retriever("dense") == "DENSE"
    assert loaded == ["dense"]
    loaded.clear()
    assert build_retriever("hybrid") == "HYBRID"
    assert loaded == ["bm25", "dense", "hybrid"]


def test_configurations_built_with_the_same_parts_share_their_indexes(monkeypatch):
    # The ablation runner builds bm25, dense and hybrid in one process; hybrid
    # must search the two already loaded rather than read each index again.
    import src.retrieval.bm25 as bm25_module
    import src.retrieval.dense as dense_module

    loaded = []
    monkeypatch.setattr(bm25_module.BM25Retriever, "load",
                        classmethod(lambda cls, **kw: loaded.append("bm25") or object()))
    monkeypatch.setattr(dense_module.DenseRetriever, "load",
                        classmethod(lambda cls, **kw: loaded.append("dense") or object()))
    parts = {}
    bm25 = build_retriever("bm25", parts=parts)
    dense = build_retriever("dense", parts=parts)
    hybrid = build_retriever("hybrid", parts=parts)
    assert loaded == ["bm25", "dense"]
    assert hybrid.bm25 is bm25 and hybrid.dense is dense
    # Without a shared dict every call still loads its own, so nothing is
    # cached between callers that did not ask for it.
    build_retriever("hybrid")
    assert loaded == ["bm25", "dense", "bm25", "dense"]


# --- answering through a stack -------------------------------------------------------

def test_a_stack_answers_with_its_own_settings(monkeypatch):
    seen = {}
    monkeypatch.setattr("src.rag.answer.answer_question",
                        lambda question, retriever, **kw: seen.update(
                            question=question, retriever=retriever, **kw))
    stack = _stack("C4", top_k=5, min_score=0.25, use_facts=False)
    stack.answer("What was Apple's revenue in FY2024?")
    assert seen["question"] == "What was Apple's revenue in FY2024?"
    assert seen["retriever"] is stack.retriever
    assert seen["config"] is stack.generation
    assert seen["llm"] is stack.llm
    assert seen["min_score"] == 0.25
    assert seen["use_facts"] is False
    assert seen["use_decomposition"] is True
    assert seen["use_refusal"] is True
    # A row that searches what it would otherwise refuse does so in the app too.
    _stack("C4", use_refusal=False).answer("Should I buy Apple stock?")
    assert seen["use_refusal"] is False


def test_a_caller_may_add_what_the_configuration_does_not_name(monkeypatch):
    seen = {}
    monkeypatch.setattr("src.rag.answer.answer_question",
                        lambda question, retriever, **kw: seen.update(kw))
    # Bound once: tokens.append is a new object on every attribute access.
    on_token = [].append
    _stack().answer("q", query="QUERY", parsed="PARSED", on_token=on_token)
    assert seen["query"] == "QUERY" and seen["parsed"] == "PARSED"
    assert seen["on_token"] is on_token


def test_a_caller_may_override_a_setting_the_configuration_names(monkeypatch):
    # A setting given at call time wins over the configuration's.
    seen = {}
    monkeypatch.setattr("src.rag.answer.answer_question",
                        lambda question, retriever, **kw: seen.update(kw))
    _stack("C4", min_score=0.25).answer("q", min_score=0.75)
    assert seen["min_score"] == 0.75


def test_a_row_without_the_metadata_filter_searches_the_whole_corpus(monkeypatch):
    # C3 was measured unfiltered (src/evaluation/run.py), so answering through
    # it must drop the filters its caller's Query carries. C4 keeps them.
    seen = {}
    monkeypatch.setattr("src.rag.answer.answer_question",
                        lambda question, retriever, **kw: seen.update(kw))
    asked = Query("What was Apple's revenue in FY2024?", top_k=7, tickers=("AAPL",),
                  fiscal_years=(2024,), items=("7",), keyword_text="revenue")
    _stack("C3").answer(asked.text, query=asked)
    assert seen["query"] == Query(asked.text, top_k=7)
    _stack("C3", top_k=5).answer(asked.text)
    assert seen["query"] == Query(asked.text, top_k=5)
    _stack("C4").answer(asked.text, query=asked)
    assert seen["query"] is asked


def test_the_harness_searches_the_way_the_row_was_measured():
    # The gap this closes: evaluate() recorded the row and then built its own
    # query, so C1 to C3 answered exactly as C4 did while the report said
    # otherwise.
    from src.evaluation import BenchmarkQuestion, evaluate

    searched = []

    class Recording(StubRetriever):
        def search(self, query, k=None):
            searched.append(query)
            return []

    question = BenchmarkQuestion(
        question_id="q1", question="What risks did Apple describe in FY2024?",
        expected_answer="", supporting_chunk_ids=("c1",), hard_negative_chunk_ids=(),
        ticker="AAPL", fiscal_year=2024, question_type="factual", difficulty="easy",
        source="manual",
    )
    for config_id in ("C3", "C4"):
        report = evaluate([question], Recording(), GENERATION, run_id=config_id,
                          stack=STACKS[config_id])
        assert report["stack"]["metadata_filter"] is STACKS[config_id].metadata_filter
    unfiltered, filtered = searched
    assert unfiltered.filters == {}
    assert filtered.filters == {"ticker": ["AAPL"], "fiscal_year": [2024]}


# --- the configurations results/ has measured ------------------------------------------

def _write_run(root, run_id, *config_ids, questions=48):
    run = root / run_id
    run.mkdir(parents=True)
    (run / "summary.json").write_text(json.dumps({
        "run_id": run_id,
        "top_k": 10,
        "configurations": [
            {"config": {"id": config_id, "name": STACKS[config_id].name},
             "questions": questions, "recall": 0.5, "ndcg": 0.4}
            for config_id in config_ids
        ],
    }), encoding="utf-8")
    return run


def test_every_configuration_a_run_measured_is_found(tmp_path):
    _write_run(tmp_path, "nightly-7", "C1", "C4")
    found = measured_runs(tmp_path)
    assert [(run.run_id, run.config_id) for run in found] == [
        ("nightly-7", "C1"), ("nightly-7", "C4"),
    ]
    assert found[0].name == "section-aware BM25"
    assert found[0].questions == 48
    assert found[0].metrics["recall"] == 0.5


def test_a_measured_run_is_labelled_by_the_run_and_the_row(tmp_path):
    _write_run(tmp_path, "nightly-7", "C1")
    assert measured_runs(tmp_path)[0].label == "nightly-7 · C1 — section-aware BM25"


def test_the_newest_run_comes_first(tmp_path):
    import os

    first = _write_run(tmp_path, "older", "C1")
    second = _write_run(tmp_path, "newer", "C4")
    os.utime(first / "..", None)
    os.utime(first, (1, 1))
    os.utime(second, (2, 2))
    assert [run.run_id for run in measured_runs(tmp_path)] == ["newer", "older"]


def test_no_results_directory_is_no_configurations(tmp_path):
    assert measured_runs(tmp_path / "absent") == []


def test_a_half_written_run_does_not_stop_the_app(tmp_path):
    _write_run(tmp_path, "good", "C1")
    (tmp_path / "no-summary").mkdir()
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "summary.json").write_text("{ not json", encoding="utf-8")
    assert [run.config_id for run in measured_runs(tmp_path)] == ["C1"]


def test_a_row_this_revision_no_longer_has_is_not_offered(tmp_path):
    # Offering it would promise a stack that cannot be built.
    run = tmp_path / "old"
    run.mkdir()
    (run / "summary.json").write_text(json.dumps({
        "run_id": "old",
        "configurations": [{"config": {"id": "C9", "name": "retired row"}, "questions": 1}],
    }), encoding="utf-8")
    assert measured_runs(tmp_path) == []


def test_a_run_without_an_id_is_not_offered(tmp_path):
    run = tmp_path / "nameless"
    run.mkdir()
    (run / "summary.json").write_text(json.dumps({
        "configurations": [{"config": {"id": "C1"}, "questions": 1}],
    }), encoding="utf-8")
    assert measured_runs(tmp_path) == []


# --- no duplicated wiring ------------------------------------------------------------

def test_neither_the_app_nor_the_commands_construct_a_retriever():
    """Criterion: no duplicated wiring between src/app/ and src/evaluation/.

    Each of the three used to name the retriever classes itself, and they had
    already drifted apart. Reading the source is the only way to assert that
    they stopped: a passing answer proves a stack was built, not where from.
    """
    from pathlib import Path

    from src.config import PROJECT_ROOT

    classes = ("BM25Retriever", "DenseRetriever", "HybridRetriever")
    # Every file of the app, so a page added later is held to it too.
    app = sorted(path.relative_to(PROJECT_ROOT).as_posix()
                 for path in (PROJECT_ROOT / "src" / "app").rglob("*.py"))
    assert "src/app/state.py" in app and "src/app/app_pages/ask.py" in app
    for relative in (*app, "src/evaluation/cli.py"):
        source = (PROJECT_ROOT / Path(relative)).read_text(encoding="utf-8")
        for name in classes:
            assert name not in source, f"{relative} still builds {name} itself"
    # The app builds a stack in one place, and its pages go through it.
    for relative in ("src/app/state.py", "src/evaluation/cli.py"):
        source = (PROJECT_ROOT / Path(relative)).read_text(encoding="utf-8")
        assert "build_stack" in source or "build_retriever" in source
    for relative in app:
        if relative != "src/app/state.py":
            source = (PROJECT_ROOT / Path(relative)).read_text(encoding="utf-8")
            assert "build_stack(" not in source, f"{relative} builds a stack of its own"


def test_the_ablation_runner_takes_its_rows_from_the_registry():
    from src.evaluation.run import CONFIGURATIONS

    assert set(CONFIGURATIONS) == set(STACKS)
    for config_id, row in CONFIGURATIONS.items():
        assert row["name"] == STACKS[config_id].name
        assert row["retriever"] == STACKS[config_id].retriever
        assert row["metadata_filter"] == STACKS[config_id].metadata_filter


def test_the_evaluation_command_leaves_a_rows_settings_alone_unless_told(monkeypatch, tmp_path):
    # A switch nobody typed is the row's. The command's own defaults used to
    # replace it, so a row with the facts route off was answered with it on
    # here and off in the app: one id, two systems.
    import src.evaluation.cli as cli

    monkeypatch.setitem(STACKS, "C5", replace(
        STACKS["C4"], id="C5", top_k=8, min_score=0.2, use_facts=False,
        use_decomposition=False, use_refusal=False))
    monkeypatch.setattr(cli, "SELECTABLE", (*SELECTABLE, "C5"))
    monkeypatch.setattr(cli, "build_stack", lambda *args, **kw: build_stack(
        *args, retriever=StubRetriever(), llm=object(), **kw))
    monkeypatch.setattr(cli, "load_questions", lambda *args, **kw: [])
    ran = {}

    def evaluate(questions, retriever, generation, **settings):
        ran.update(settings)
        return {"summary": {}, "by_answerability": {}, "results": []}

    monkeypatch.setattr(cli, "evaluate", evaluate)
    common = ["--run-id", "row", "--output", str(tmp_path / "report.json")]

    cli.main(["--config", "C5", *common])
    assert (ran["top_k"], ran["min_score"]) == (8, 0.2)
    assert ran["use_facts"] is False and ran["use_decomposition"] is False
    assert ran["use_refusal"] is False
    assert ran["stack"].to_dict()["use_facts"] is False

    # What is typed still wins, and only that.
    cli.main(["--config", "C5", "--top-k", "4", "--min-score", "0.5", *common])
    assert (ran["top_k"], ran["min_score"]) == (4, 0.5)
    assert ran["use_facts"] is False
    cli.main(["--config", "C4", "--no-facts", *common])
    assert ran["use_facts"] is False and ran["use_decomposition"] is True
    assert ran["use_refusal"] is True
    assert ran["top_k"] == STACKS["C4"].top_k
    cli.main(["--config", "C4", "--no-refusal", *common])
    assert ran["use_refusal"] is False and ran["use_facts"] is True
    assert ran["stack"].to_dict()["use_refusal"] is False


def test_dense_prefix_defaults_and_empty_prefix_reach_loader(monkeypatch, tmp_path):
    from src.retrieval.dense import DenseRetriever
    from src.retrieval.constants import QUERY_PREFIX

    loaded = []
    monkeypatch.setattr(DenseRetriever, "load", lambda **kw: loaded.append(kw) or object())
    build_retriever("dense", chroma_dir=tmp_path)
    build_retriever("dense", chroma_dir=tmp_path, embedding_query_prefix="")
    assert loaded[0]["embedding"].query_prefix == QUERY_PREFIX
    assert loaded[1]["embedding"].query_prefix == ""


def test_shared_parts_scope_dense_by_encoder_settings_and_corpus(monkeypatch, tmp_path):
    from src.retrieval.dense import DenseRetriever

    loaded = []
    monkeypatch.setattr(DenseRetriever, "load", lambda **kwargs: loaded.append(kwargs) or object())
    parts = {}
    default = build_retriever("dense", parts=parts)
    other = build_retriever("dense", chroma_dir=tmp_path / "E2", parts=parts)
    prefix = build_retriever("dense", chroma_dir=tmp_path / "E2", embedding_query_prefix="", parts=parts)
    corpus = build_retriever("dense", processed_dir=tmp_path / "processed", parts=parts)
    assert len({id(default), id(other), id(prefix), id(corpus)}) == 4
    assert build_retriever("dense", parts=parts) is default
    assert len(loaded) == 4
    with pytest.raises(ValueError, match="need chroma_dir"):
        build_retriever("dense", embedding_model="custom", parts=parts)


def test_e3_grouped_settings_reach_dense_load(monkeypatch, tmp_path):
    from src.evaluation.model_ablation import EMBEDDING_EXPERIMENTS, _default_embedding_stack
    from src.retrieval.bm25 import BM25Retriever
    from src.retrieval.dense import DenseRetriever

    seen = []
    monkeypatch.setattr(BM25Retriever, "load", lambda **kwargs: object())
    monkeypatch.setattr(DenseRetriever, "load", lambda **kwargs: seen.append(kwargs) or object())
    _default_embedding_stack(EMBEDDING_EXPERIMENTS[2], processed_dir=tmp_path)
    config = seen[0]["embedding"]
    assert (config.model, config.dimensions, config.query_prefix, config.passage_prefix, config.max_tokens) == (
        "intfloat/e5-base-v2", 768, "query: ", "passage: ", 512,
    )
    assert seen[0]["processed_dir"] == tmp_path
