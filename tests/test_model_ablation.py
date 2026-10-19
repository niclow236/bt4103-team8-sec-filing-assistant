"""Tests for E1-E3 and G1-G3 model ablations."""

import csv
import json
import re
from types import SimpleNamespace

import pytest

from src.evaluation.harness import RunInterrupted, RunStopped
from src.evaluation.model_ablation import (
    EMBEDDING_EXPERIMENTS,
    GENERATION_EXPERIMENTS,
    run_embedding_ablation,
    run_generation_ablation,
)
from src.evaluation.records import BenchmarkQuestion
from src.rag.records import GenerationConfig
from src.retrieval.records import RetrievedPassage
from test_full_ablation_runner import answer_run, runner  # noqa: F401  (fixtures)


def _question():
    return BenchmarkQuestion(
        question_id="q001",
        question="What was revenue?",
        expected_answer="$100",
        supporting_chunk_ids=("support",),
        hard_negative_chunk_ids=(),
        ticker="AAPL",
        fiscal_year=2024,
        question_type="numeric",
        difficulty="easy",
        source="handwritten",
    )


def _stack(experiment):
    from src.stack import stack_config
    class FakeRetriever:
        name = "hybrid"

        def search(self, query):
            return [RetrievedPassage(
                chunk_id="support", text="support", score=1.0, rank=1,
                retriever=self.name, ticker="AAPL", company="Apple",
                fiscal_year=2024, item="7", title="MD&A", url="https://example.com",
            )]

    return SimpleNamespace(
        config=stack_config("C4"),
        retriever=FakeRetriever(),
        generation=GenerationConfig(
            provider=getattr(experiment, "provider", "ollama"),
            model=getattr(experiment, "model", "test-model"),
            prompt_template_id="test",
        ),
        llm=object(),
    )


def _fake_evaluate(questions, retriever, config, **kwargs):
    return {
        "config": config.to_dict(),
        "results": [{"question_id": question.question_id} for question in questions],
        "summary": {"total": len(questions), "abstention_rate": 0.25},
    }


def test_embedding_matrix_has_three_configured_models_and_common_table(monkeypatch, tmp_path):
    import src.evaluation.model_ablation as module

    manifest = run_embedding_ablation(
        [_question()], _stack, run_id="embedding-run", results_root=tmp_path
    )

    assert [row["config"]["id"] for row in manifest["configurations"]] == ["E1", "E2", "E3"]
    assert {row["config"]["embedding_model"] for row in manifest["configurations"]} == {
        experiment.model for experiment in EMBEDDING_EXPERIMENTS
    }
    with (tmp_path / "embedding-run" / "summary.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 3
    assert "abstention_rate" in rows[0]
    assert rows[0]["recall"] == "1.0"


def test_generation_matrix_compares_the_local_and_both_hosted_providers(monkeypatch, tmp_path):
    import src.evaluation.model_ablation as module

    monkeypatch.setattr(module, "evaluate", _fake_evaluate)
    manifest = run_generation_ablation(
        [_question()], _stack, run_id="generation-run", results_root=tmp_path
    )

    assert [row["config"]["id"] for row in manifest["configurations"]] == ["G1", "G2", "G3"]
    assert {row["config"]["provider"] for row in manifest["configurations"]} == {
        experiment.provider for experiment in GENERATION_EXPERIMENTS
    }
    # G3 names SOCLAAS's model by its ID, not the "default" alias (#163).
    assert [(row["config"]["provider"], row["config"]["model"])
            for row in manifest["configurations"]][2] == ("soclaas", "qwen3.6:35b")
    assert json.loads(
        (tmp_path / "generation-run" / "G2" / "report.json").read_text()
    )["config"]["provider"] == "mistral"


def test_embedding_and_generation_writers_preserve_benchmark_metadata(monkeypatch, tmp_path):
    from dataclasses import replace
    from src.app.state import result_runs
    import src.evaluation.model_ablation as module

    questions = [_question(), replace(_question(), question_id="m001", source="xbrl",
                                     difficulty="mechanical", supporting_chunk_ids=("missed",))]
    monkeypatch.setattr(module, "evaluate", _fake_evaluate)
    for runner, run_id in ((run_embedding_ablation, "embedding"),
                           (run_generation_ablation, "generation")):
        manifest = runner(questions, _stack, run_id=run_id, results_root=tmp_path)
        for summary in manifest["configurations"]:
            rows = json.loads((tmp_path / run_id / summary["config"]["id"]
                               / "report.json").read_text())["results"]
            assert [{key: row[key] for key in ("question_id", "question_type", "source", "difficulty")}
                    for row in rows] == [
                {"question_id": "q001", "question_type": "numeric", "source": "handwritten", "difficulty": "easy"},
                {"question_id": "m001", "question_type": "numeric", "source": "xbrl", "difficulty": "mechanical"}]
    result_runs.clear()
    for run in result_runs(tmp_path):
        for config in run["configurations"]:
            groups = config["benchmark_metrics"]
            assert groups["handwritten"]["questions"] == groups["mechanical"]["questions"] == 1
            if run["run_id"] == "embedding":
                assert (groups["handwritten"]["recall"], groups["mechanical"]["recall"]) == (1.0, 0.0)


def test_generation_uses_c4_stack_settings(monkeypatch, tmp_path):
    import src.evaluation.model_ablation as module

    seen = []

    def record(questions, retriever, config, **kwargs):
        seen.append(kwargs)
        return _fake_evaluate(questions, retriever, config)

    monkeypatch.setattr(module, "evaluate", record)
    run_generation_ablation([_question()], _stack, run_id="settings", results_root=tmp_path)

    c4 = _stack(GENERATION_EXPERIMENTS[0]).config
    assert [kwargs["stack"] for kwargs in seen] == [c4, c4, c4]
    assert {kwargs["top_k"] for kwargs in seen} == {c4.top_k}
    assert {kwargs["use_facts"] for kwargs in seen} == {c4.use_facts}


def test_embedding_stack_passes_e3_query_and_passage_prefixes(monkeypatch):
    import src.evaluation.model_ablation as module

    seen = {}
    monkeypatch.setattr(
        module,
        "build_retriever",
        lambda key, **kwargs: seen.update(kwargs) or SimpleNamespace(name="hybrid"),
    )

    module._default_embedding_stack(EMBEDDING_EXPERIMENTS[2])

    assert seen["embedding"].model == "intfloat/e5-base-v2"
    assert seen["embedding"].query_prefix == "query: "
    assert seen["embedding"].passage_prefix == "passage: "


def test_embedding_index_preparation_uses_each_registry_configuration(monkeypatch, tmp_path):
    import src.evaluation.model_ablation as module

    calls = []
    monkeypatch.setattr(module, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(
        "src.retrieval.embed.build",
        lambda **kwargs: calls.append(kwargs),
    )

    module.prepare_embedding_indexes(tmp_path / "processed")

    assert [call["model_name"] for call in calls] == [
        experiment.model for experiment in EMBEDDING_EXPERIMENTS
    ]
    assert not any(call["rebuild"] for call in calls)


def test_embedding_metrics_and_saved_evidence_for_every_model(tmp_path):
    manifest = run_embedding_ablation([_question()], _stack, run_id="evidence", results_root=tmp_path)
    for summary in manifest["configurations"]:
        assert (summary["recall"], summary["ndcg"], summary["mrr"]) == (1.0, 1.0, 1.0)
        report = json.loads((tmp_path / "evidence" / summary["config"]["id"] / "report.json").read_text())
        row = report["results"][0]
        assert row["retrieved_chunk_ids"] == ["support"]
        assert row["retrieved_scores"] == [1.0]
        assert row["latency_ms"] >= 0


def test_embedding_uses_shared_c4_query_including_numeric_cues(tmp_path):
    from src.evaluation.run import _query

    searched = []
    def recording(experiment):
        stack = _stack(experiment)
        stack.retriever.search = lambda query: searched.append(query) or []
        return stack

    run_embedding_ablation([_question()], recording, run_id="filtered", results_root=tmp_path)
    assert searched == [_query(_question(), top_k=10, metadata_filter=True)] * 3
    assert searched[0].tickers == ("AAPL",)
    assert searched[0].fiscal_years == (2024,)


def test_both_matrices_build_all_stacks_before_search_or_output(monkeypatch, tmp_path):
    import pytest
    import src.evaluation.model_ablation as module
    from src.rag import ProviderUnavailable

    asked = []
    monkeypatch.setattr(module, "evaluate", lambda *a, **kw: asked.append("asked"))
    def no_second(experiment):
        if experiment.id in {"E2", "G2"}:
            raise ProviderUnavailable("missing index or key")
        stack = _stack(experiment)
        stack.retriever.search = lambda query: asked.append("searched") or []
        return stack

    for runner in (run_embedding_ablation, run_generation_ablation):
        with pytest.raises(ProviderUnavailable):
            runner([_question()], no_second, run_id="preflight", results_root=tmp_path)
        assert asked == []
        assert not (tmp_path / "preflight").exists()


def test_stopped_and_interrupted_generation_preserves_partial_rows(monkeypatch, tmp_path):
    import pytest
    import src.evaluation.model_ablation as module
    from src.evaluation.harness import RunInterrupted, RunStopped

    for error_type in (RunStopped, RunInterrupted):
        def stops_on_mistral(questions, retriever, config, **kwargs):
            report = _fake_evaluate(questions, retriever, config)
            if config.provider == "mistral":
                report["stopped"] = {"question_id": "q002", "error": "provider stopped"}
                raise error_type(report)
            return report
        monkeypatch.setattr(module, "evaluate", stops_on_mistral)
        with pytest.raises(error_type):
            run_generation_ablation([_question()], _stack, run_id=error_type.__name__, results_root=tmp_path)
        run_dir = tmp_path / error_type.__name__
        report = json.loads((run_dir / "G2" / "report.json").read_text())
        assert report["results"] == [{
            "question_id": "q001",
            "question_type": "numeric",
            "difficulty": "easy",
            "source": "handwritten",
        }]
        assert report["stopped"]["question_id"] == "q002"
        summary = json.loads((run_dir / "summary.json").read_text())
        assert summary["top_k"] == _stack(GENERATION_EXPERIMENTS[0]).config.top_k
        assert [row["config"]["id"] for row in summary["configurations"]] == ["G1", "G2"]
        assert [row["abstention_rate"] for row in summary["configurations"]] == [0.25, 0.25]
        with (run_dir / "summary.csv").open(newline="") as stream:
            assert [row["config"] for row in csv.DictReader(stream)] == ["G1", "G2"]


def test_default_generation_shares_parts_and_disables_facts(monkeypatch, tmp_path):
    import src.evaluation.model_ablation as module

    seen = []
    monkeypatch.setattr(module, "build_stack", lambda key, **kwargs: seen.append(kwargs))
    parts = {}
    for experiment in GENERATION_EXPERIMENTS:
        module._default_generation_stack(experiment, processed_dir=tmp_path, parts=parts)
    assert all(row["parts"] is parts and row["processed_dir"] == tmp_path for row in seen)
    assert all(row["use_facts"] is False for row in seen)
    assert [row["provider"] for row in seen] == ["ollama", "mistral", "soclaas"]


def test_generation_statistics_count_provider_routes_and_retries():
    import src.evaluation.model_ablation as module

    report = {"summary": {"total": 3, "abstention_rate": 1 / 3}, "results": [
        {"route": "facts", "attempts": 1, "answer": {"abstained": False, "latency_ms": 1}},
        {"route": "ollama", "attempts": 2, "answer": {"abstained": True, "latency_ms": 100}},
        {"route": "ollama", "attempts": 1, "answer": {"abstained": False, "latency_ms": 300}},
    ]}
    summary = module._generation_summary(GENERATION_EXPERIMENTS[0], report)
    assert summary["llm_questions"] == 2
    assert summary["llm_abstention_rate"] == 0.5
    assert summary["median_latency_ms"] == 100
    assert summary["retried_questions"] == 1


def test_dense_and_hybrid_views_report_independent_metrics(tmp_path):
    def two_views(experiment):
        stack = _stack(experiment)
        stack.dense_retriever = SimpleNamespace(name="dense", search=lambda query: [])
        return stack

    manifest = run_embedding_ablation([_question()], two_views, run_id="views", results_root=tmp_path)
    assert len(manifest["configurations"]) == 6
    for row in manifest["configurations"]:
        assert row["recall"] == (0.0 if row["config"]["id"].endswith("-dense") else 1.0)
    from src.stack import measured_runs
    assert measured_runs(tmp_path) == []


def test_truncation_is_in_embedding_table(monkeypatch, tmp_path):
    import src.evaluation.model_ablation as module

    def indexed(experiment):
        stack = _stack(experiment)
        stack.retriever.dense = SimpleNamespace(
            manifest=SimpleNamespace(max_tokens=256),
            collection=SimpleNamespace(count=lambda: 10, get=lambda **kw: {"ids": ["a", "b", "c"]}),
        )
        return stack
    manifest = run_embedding_ablation([_question()], indexed, run_id="truncated", results_root=tmp_path)
    assert all(row["truncation_rate"] == 0.3 and row["n_truncated"] == 3 for row in manifest["configurations"])
    with (tmp_path / "truncated" / "summary.csv").open(newline="") as stream:
        assert all(row["truncation_rate"] == "0.3" and row["max_tokens"] == "256" for row in csv.DictReader(stream))


def test_sampling_spans_full_file_and_has_exact_size():
    from src.evaluation.model_ablation import _sample
    from dataclasses import replace

    questions = [replace(_question(), question_id=str(i)) for i in range(27)]
    assert [q.question_id for q in _sample(questions, 14)] == [str(i) for i in range(0, 27, 2)]
    assert _sample(questions, 30) == questions
    assert _sample([], 14) == []
    assert _sample(questions, 1) == [questions[13]]


def test_cli_rejects_bad_options_before_loading_or_building(monkeypatch, tmp_path):
    import pytest
    import src.evaluation.model_ablation as module

    def unexpected(*args, **kwargs):
        pytest.fail("invalid options must fail before expensive work")
    monkeypatch.setattr(module, "load_questions", unexpected)
    monkeypatch.setattr(module, "prepare_embedding_indexes", unexpected)
    (tmp_path / "taken").mkdir()
    for options in (
        ["embedding", "--run-id", "taken", "--prepare-indexes"],
        ["embedding", "--run-id", "../escape", "--prepare-indexes"],
        ["embedding", "--run-id", "valid", "--limit", "0"],
        ["embedding", "--run-id", "valid", "--top-k", "0"],
        ["embedding", "--run-id", "valid", "--rebuild-indexes"],
        ["embedding", "--run-id", "valid", "--prepare-indexes",
         "--processed-dir", str(tmp_path / "other")],
        ["generation", "--run-id", "valid", "--top-k", "10"],
    ):
        with pytest.raises(SystemExit) as error:
            module.main([*options, "--results-root", str(tmp_path)])
        assert error.value.code == 2


def test_cli_passes_custom_corpus_and_prints_result(monkeypatch, tmp_path, capsys):
    import src.evaluation.model_ablation as module

    seen = []
    def load(path, **kwargs):
        seen.append(kwargs["processed_dir"])
        return [_question()]
    def stack(experiment, **kwargs):
        seen.append(kwargs["processed_dir"])
        return _stack(experiment)
    monkeypatch.setattr(module, "load_questions", load)
    monkeypatch.setattr(module, "_default_embedding_stack", stack)
    module.main(["embedding", "questions.jsonl", "--processed-dir", str(tmp_path / "corpus"),
                 "--results-root", str(tmp_path), "--run-id", "cli"])
    assert seen == [tmp_path / "corpus"] * 4
    assert "written:" in capsys.readouterr().out


def test_missing_provider_key_has_short_cli_error(monkeypatch, tmp_path, capsys):
    import pytest
    import src.evaluation.model_ablation as module
    from src.rag import ProviderUnavailable

    monkeypatch.setattr(module, "load_questions", lambda *args, **kwargs: [_question()])
    def no_key(*args, **kwargs):
        raise ProviderUnavailable("MISTRAL_API_KEY is not set")
    monkeypatch.setattr(module, "_default_generation_stack", no_key)
    with pytest.raises(SystemExit) as error:
        module.main(["generation", "--results-root", str(tmp_path), "--run-id", "missing"])
    assert error.value.code == 2
    assert "MISTRAL_API_KEY" in capsys.readouterr().err
    assert not (tmp_path / "missing").exists()


def test_real_embedding_matrix_keeps_encoders_prefixes_and_token_limits_separate(monkeypatch, corpus, tmp_path):
    """Exercise real Chroma, dense loading, hybrid fusion and result writing."""
    import numpy as np
    from conftest import vector_for
    import src.evaluation.model_ablation as module
    from src.pipeline.chunk import iter_chunks
    from src.retrieval import embed
    from src.retrieval.bm25 import BM25Retriever
    from dataclasses import replace

    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(module, "PROCESSED_DIR", corpus)
    encoded = []
    class Encoder:
        def __init__(self, model_name, dimensions):
            self.name = model_name
            self.dimensions = dimensions
            self.max_seq_length = 256 if "MiniLM" in model_name else 512
            self.tokenizer = lambda texts: {"input_ids": [[0] * 300 for text in texts]}
        def encode(self, texts, **kwargs):
            encoded.extend((self.name, text) for text in texts)
            vectors = [vector_for(text)[:self.dimensions] for text in texts]
            return np.stack([v / np.linalg.norm(v) for v in vectors])
    monkeypatch.setattr(embed, "_load_model", lambda threads=None, model_name=embed.EMBED_MODEL,
                        expected_dimensions=768: (Encoder(model_name, expected_dimensions), 1))
    chunks = list(iter_chunks(processed_dir=corpus))
    bm25 = BM25Retriever(chunks)
    monkeypatch.setattr(BM25Retriever, "load", lambda **kwargs: bm25)
    build_times = module.prepare_embedding_indexes(corpus)
    parts = {}
    question = replace(_question(), ticker="AAA", fiscal_year=2023,
                       question="What was revenue for AAA in FY2023?",
                       supporting_chunk_ids=(chunks[0]["chunk_id"],))
    manifest = run_embedding_ablation(
        [question], lambda experiment: module._default_embedding_stack(experiment, processed_dir=corpus, parts=parts),
        run_id="real", results_root=tmp_path / "results", build_times=build_times,
    )
    assert len(manifest["configurations"]) == 6
    for summary in manifest["configurations"]:
        assert summary["questions"] == 1
        assert summary["build_ms"] >= 0
        assert summary["truncation_rate"] == (1.0 if summary["config"]["id"].startswith("E2") else 0.0)
        assert 0 <= summary["recall"] <= 1
    for experiment in EMBEDDING_EXPERIMENTS:
        index_manifest = embed.read_manifest(embed.manifest_file_for(tmp_path / experiment.index_dir))
        assert (index_manifest.model, index_manifest.dimensions, index_manifest.max_tokens,
                index_manifest.passage_prefix) == (experiment.model, experiment.dimensions,
                                                  experiment.max_tokens, experiment.passage_prefix)
    e5_texts = [text for name, text in encoded if name == "intfloat/e5-base-v2"]
    assert all(text.startswith(("passage: ", "query: ")) for text in e5_texts)
    assert any(text.startswith("query: ") for text in e5_texts)
    assert all(not text.startswith("None") for _, text in encoded)


def test_custom_corpus_preparation_preserves_the_apps_real_index(monkeypatch, corpus, tmp_path, fake_model):
    import pytest
    import src.evaluation.model_ablation as module
    from src.retrieval import embed

    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(module, "PROCESSED_DIR", corpus)
    chroma = tmp_path / EMBEDDING_EXPERIMENTS[0].index_dir
    embed.build(chroma_dir=chroma, processed_dir=corpus)
    manifest_path = embed.manifest_file_for(chroma)
    before_manifest = manifest_path.read_bytes()
    collection = embed.open_collection(chroma)
    before_ids = set(collection.get(include=[])["ids"])
    before_encodes = fake_model.calls
    other_corpus = tmp_path / "one-filing"
    other_corpus.mkdir()
    filing = next(corpus.glob("*/*.json"))
    (other_corpus / filing.name).write_bytes(filing.read_bytes())

    for rebuild in (False, True):
        options = ["embedding", "questions.jsonl", "--prepare-indexes", "--processed-dir", str(other_corpus),
                   "--results-root", str(tmp_path / "results"), "--run-id", "other-corpus"]
        if rebuild:
            options.append("--rebuild-indexes")
        with pytest.raises(SystemExit) as error:
            module.main(options)
        assert error.value.code == 2
        with pytest.raises(ValueError, match="use the default corpus"):
            module.prepare_embedding_indexes(other_corpus, rebuild=rebuild)
        assert manifest_path.read_bytes() == before_manifest
        assert set(collection.get(include=[])["ids"]) == before_ids
        assert fake_model.calls == before_encodes
        assert not (tmp_path / "results").exists()
    assert embed.check_index(chroma, corpus) == []


def test_preparation_accepts_relative_path_to_default_corpus(monkeypatch, tmp_path):
    import src.evaluation.model_ablation as module

    default = tmp_path / "processed"
    default.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module, "PROCESSED_DIR", default)
    monkeypatch.setattr(module, "load_questions", lambda *args, **kwargs: [_question()])
    prepared = []
    monkeypatch.setattr(module, "prepare_embedding_indexes", lambda path, **kw: prepared.append(path) or {})
    monkeypatch.setattr(module, "_default_embedding_stack", lambda *args, **kwargs: _stack(args[0]))
    module.main(["embedding", "--prepare-indexes", "--processed-dir", "./processed",
                 "--results-root", str(tmp_path / "results"), "--run-id", "relative"])
    assert len(prepared) == 1
    assert prepared[0].resolve() == default


def test_auxiliary_embedding_indexes_are_siblings_of_app_index():
    from pathlib import Path
    from src.retrieval.embed import manifest_file_for, truncation_file_for

    app, *others = [Path(experiment.index_dir) for experiment in EMBEDDING_EXPERIMENTS]
    assert [path.name for path in others] == ["chroma-E2", "chroma-E3"]
    for index in others:
        assert index.parent == app.parent
        assert not index.is_relative_to(app)
        assert manifest_file_for(index).parent == app.parent
        assert truncation_file_for(index).parent == app.parent


def test_index_error_advice_distinguishes_auxiliary_dense_and_bm25(monkeypatch, tmp_path, capsys):
    import pytest
    import src.evaluation.model_ablation as module

    monkeypatch.setattr(module, "load_questions", lambda *args, **kwargs: [_question()])
    for detail in (
        "Dense index at data/index/chroma-E2 cannot be searched: no manifest. Run: python -m src.retrieval embed",
        "BM25 index does not match the corpus. Rebuild it: python -m src.retrieval bm25",
    ):
        def unavailable(*args, **kwargs):
            raise ValueError(detail)
        monkeypatch.setattr(module, "_default_embedding_stack", unavailable)
        with pytest.raises(SystemExit) as error:
            module.main(["embedding", "questions.jsonl", "--results-root", str(tmp_path), "--run-id", "missing"])
        assert error.value.code == 2
        message = capsys.readouterr().err
        assert message.index("src.evaluation.model_ablation embedding") < message.index(detail)
        assert "If it is E2 or E3" in message
        assert "default index and for BM25" in message
        assert detail in message
        assert not (tmp_path / "missing").exists()


def test_custom_corpus_error_recommends_preparing_only_default_corpus(monkeypatch, tmp_path, capsys):
    import pytest
    from shlex import join
    import src.evaluation.model_ablation as module

    default = tmp_path / "default corpus"
    monkeypatch.setattr(module, "PROCESSED_DIR", default)
    monkeypatch.setattr(module, "load_questions", lambda *args, **kwargs: [_question()])
    def unavailable(*args, **kwargs):
        raise ValueError("BM25 index does not match the corpus")
    monkeypatch.setattr(module, "_default_embedding_stack", unavailable)
    with pytest.raises(SystemExit):
        module.main(["embedding", "questions.jsonl", "--processed-dir", str(tmp_path / "other corpus"),
                     "--results-root", str(tmp_path / "results"), "--run-id", "missing"])
    assert join(["--processed-dir", str(default)]) in capsys.readouterr().err


# --- G rows over a full-matrix run's recorded evidence (#163) ------------------------

def _recorded_row(question_id, *, question_type="numeric", abstained=False, correctness=1.0,
                  parse_error=None, truncated=False, invalid=(), unresolved=(), generate_ms=1000.0,
                  input_tokens=500, output_tokens=50, attempts=1, method="llm_judge"):
    """A row as the harness writes one, with every field the replay summary reads."""
    return {
        "question_id": question_id, "question_type": question_type, "attempts": attempts,
        "route": "mistral",
        "answer": {"abstained": abstained, "parse_error": parse_error, "truncated": truncated},
        "quality": {"method": method, "correctness": correctness, "faithfulness": 1.0,
                    "answerability_correct": 1.0, "citation_precision": 1.0,
                    "citation_recall": 1.0, "invalid_markers": list(invalid),
                    "unresolved_markers": list(unresolved)},
        "latency_ms": {"retrieve": 0.0, "rerank": 0.0, "generate": generate_ms,
                       "retry_wait": 0.0, "other": 0.0, "total": generate_ms},
        "cost": {"usd": None, "input_tokens": input_tokens, "output_tokens": output_tokens},
        "judge_overhead": {"latency_ms": 0.0, "retry_wait_ms": 0.0, "total_ms": 0.0,
                           "cost": {"usd": None}},
    }


def _recorded(tmp_path, rows=None):
    from dataclasses import replace
    from src.evaluation.model_ablation import RecordedAnswers

    questions = (_question(), replace(_question(), question_id="q002", question_type="unanswerable"))
    return RecordedAnswers(
        run_dir=tmp_path / "full-matrix-20261017", questions=questions, retriever=object(),
        generation=GenerationConfig("mistral", "ministral-8b-2512", "grounded_v4"),
        rows=tuple(rows or (_recorded_row("q001"),
                            _recorded_row("q002", question_type="unanswerable", abstained=True))),
        facts=b"facts the queries were checked against",
    )


def _replaying(seen, rows=None):
    def evaluate(questions, retriever, config, **kwargs):
        seen.append((questions, retriever, config, kwargs))
        # As the harness's report does, it names the store it parsed with.
        return {"config": config.to_dict(), "facts_file": str(kwargs.get("facts_file")),
                "results": rows or [
            _recorded_row("q001", generate_ms=400.0),
            _recorded_row("q002", question_type="unanswerable", abstained=True,
                          generate_ms=400.0)]}
    return evaluate


def test_g3_replays_the_recorded_evidence_with_only_the_generator_changed(monkeypatch, tmp_path):
    import src.evaluation.model_ablation as module
    from src.evaluation.full_ablation import MATRIX
    from src.evaluation.model_ablation import run_generation_replay

    seen = []
    monkeypatch.setattr(module, "evaluate", _replaying(seen))
    recorded = _recorded(tmp_path)
    judge = SimpleNamespace(config=recorded.generation, to_dict=lambda: {"judge": "recorded"})
    g3 = GENERATION_EXPERIMENTS[2]
    manifest = run_generation_replay(recorded, [g3], run_id="replay", results_root=tmp_path,
                                     llm_for=lambda config: f"llm for {config.model}",
                                     judge=judge)

    [(questions, retriever, config, kwargs)] = seen
    assert list(questions) == list(recorded.questions) and retriever is recorded.retriever
    # The recorded answers' template and temperature, with G3's provider and model.
    assert config == GenerationConfig("soclaas", "qwen3.6:35b", "grounded_v4", 0.0)
    assert kwargs["llm"] == "llm for qwen3.6:35b" and kwargs["judge"] is judge
    assert kwargs["stack"] == MATRIX["C4"] and kwargs["top_k"] == MATRIX["C4"].top_k
    assert (kwargs["use_facts"], kwargs["use_refusal"], kwargs["use_decomposition"],
            kwargs["use_verification"]) == (False, False, False, False)
    # Parsed with a copy, kept with the run, of the facts store its queries
    # were checked against.
    assert kwargs["facts_file"] == tmp_path / "replay" / "facts.parquet"
    assert kwargs["facts_file"].read_bytes() == recorded.facts
    # The recorded answers are the G row whose provider and model wrote them.
    assert [row["config"]["id"] for row in manifest["configurations"]] == ["G2", "G3"]
    assert manifest["configurations"][0]["config"]["origin"] == "full-matrix-20261017/C4"
    report = json.loads((tmp_path / "replay" / "G3" / "report.json").read_text())
    assert report["config"]["provider"] == "soclaas"
    assert report["facts_file"] == str(tmp_path / "replay" / "facts.parquet")
    with (tmp_path / "replay" / "summary.csv").open(newline="") as stream:
        table = list(csv.DictReader(stream))
    assert [(row["config"], row["model"], row["median_generate_ms"]) for row in table] == [
        ("G2", "ministral-8b-2512", "1000.0"), ("G3", "qwen3.6:35b", "400.0")]


def test_the_replay_summary_counts_what_163_compares_generators_on(tmp_path):
    from src.evaluation.model_ablation import _replay_summary

    rows = [
        _recorded_row("a1", correctness=1.0, generate_ms=100.0, input_tokens=1000, output_tokens=10),
        _recorded_row("a2", correctness=0.5, generate_ms=200.0, invalid=[7], attempts=2),
        # Cut at the ceiling: truncated, and not also a schema failure.
        _recorded_row("a3", correctness=0.0, generate_ms=300.0, parse_error="cut", truncated=True),
        _recorded_row("a4", correctness=None, generate_ms=400.0, parse_error="bad schema",
                      method="judge_failed", input_tokens=None, output_tokens=None),
        _recorded_row("u1", question_type="unanswerable", abstained=True, generate_ms=5000.0),
        _recorded_row("u2", question_type="unanswerable", abstained=False, correctness=0.0,
                      generate_ms=600.0, unresolved=[3]),
    ]
    judge = GenerationConfig("mistral", "ministral-8b-2512", "answer-quality-v1")
    summary = _replay_summary(GENERATION_EXPERIMENTS[2], rows, judge=judge, origin="replayed")
    assert summary["questions"] == 6 and summary["judge"] == "mistral/ministral-8b-2512"
    assert (summary["correctness"], summary["correctness_scored"]) == (2.5 / 5, 5)
    assert (summary["unanswerable_abstention"], summary["answerable_abstention"]) == (0.5, 0.0)
    assert summary["invalid_citation_answers"] == 2
    assert (summary["schema_failures"], summary["truncated"], summary["judge_failures"]) == (1, 1, 1)
    assert summary["median_generate_ms"] == 350.0
    # Nearest rank: the 6th of 6 ordered values is the 95th percentile.
    assert summary["p95_generate_ms"] == 5000.0
    assert (summary["mean_input_tokens"], summary["mean_output_tokens"]) == (
        (1000 + 500 * 4) / 5, (10 + 50 * 4) / 5)
    assert summary["retried_questions"] == 1


def test_a_replay_that_cannot_build_its_models_writes_nothing(monkeypatch, tmp_path):
    import pytest
    import src.evaluation.model_ablation as module
    from src.evaluation.model_ablation import run_generation_replay
    from src.rag import ProviderUnavailable

    monkeypatch.setattr(module, "evaluate", lambda *a, **k: pytest.fail("answered"))

    def no_key(config):
        raise ProviderUnavailable("SOCLAAS_API_KEY is not set")

    recorded = _recorded(tmp_path)
    with pytest.raises(ProviderUnavailable):
        run_generation_replay(recorded, [GENERATION_EXPERIMENTS[2]], run_id="no-key",
                              results_root=tmp_path, llm_for=no_key,
                              judge=SimpleNamespace(config=recorded.generation, to_dict=dict))
    assert not (tmp_path / "no-key").exists()


@pytest.mark.parametrize("stopper", [RunStopped, RunInterrupted])
def test_a_stopped_replay_keeps_its_answers_and_its_table(monkeypatch, tmp_path, stopper):
    import src.evaluation.model_ablation as module
    from src.evaluation.model_ablation import run_generation_replay

    asked = []

    def stops(questions, retriever, config, **kwargs):
        asked.append(config.provider)
        raise stopper({"results": [_recorded_row("q001")],
                       "stopped": {"question_id": "q002", "error": "rate limit"}})

    monkeypatch.setattr(module, "evaluate", stops)
    recorded = _recorded(tmp_path)
    with pytest.raises(stopper):
        run_generation_replay(recorded, [GENERATION_EXPERIMENTS[2], GENERATION_EXPERIMENTS[0]],
                              run_id="stopped",
                              results_root=tmp_path, llm_for=lambda config: None,
                              judge=SimpleNamespace(config=recorded.generation, to_dict=dict))
    report = json.loads((tmp_path / "stopped" / "G3" / "report.json").read_text())
    assert [row["question_id"] for row in report["results"]] == ["q001"]
    summary = json.loads((tmp_path / "stopped" / "summary.json").read_text())
    assert [(row["config"]["id"], row["questions"]) for row in summary["configurations"]] == [
        ("G2", 2), ("G3", 1)]
    assert summary["configurations"][1]["stopped"]["question_id"] == "q002"
    # The rows after the one that stopped are not run.
    assert asked == ["soclaas"]


def test_recorded_answers_from_a_generator_outside_the_registry_are_their_own_row(
        monkeypatch, tmp_path):
    from dataclasses import replace
    import src.evaluation.model_ablation as module
    from src.evaluation.model_ablation import run_generation_replay

    monkeypatch.setattr(module, "evaluate", _replaying([]))
    recorded = replace(_recorded(tmp_path),
                       generation=GenerationConfig("mistral", "ministral-14b-2512", "grounded_v4"))
    manifest = run_generation_replay(
        recorded, [GENERATION_EXPERIMENTS[2]], run_id="outside", results_root=tmp_path,
        llm_for=lambda config: None,
        judge=SimpleNamespace(config=recorded.generation, to_dict=dict))
    first = manifest["configurations"][0]["config"]
    assert (first["id"], first["provider"], first["model"]) == (
        "recorded", "mistral", "ministral-14b-2512")


def test_replaying_the_model_that_wrote_the_recorded_answers_is_its_own_row(monkeypatch, tmp_path):
    # Decided on #163: any row can be replayed, so a baseline can be repeated,
    # but a repeat of G2 beside the recorded G2 is named apart from it.
    import src.evaluation.model_ablation as module
    from src.evaluation.model_ablation import run_generation_replay

    monkeypatch.setattr(module, "evaluate", _replaying([]))
    recorded = _recorded(tmp_path)
    manifest = run_generation_replay(
        recorded, [GENERATION_EXPERIMENTS[1], GENERATION_EXPERIMENTS[2]], run_id="repeat",
        results_root=tmp_path, llm_for=lambda config: None,
        judge=SimpleNamespace(config=recorded.generation, to_dict=dict))
    assert [row["config"]["id"] for row in manifest["configurations"]] == ["G2", "G2-replay", "G3"]
    assert (tmp_path / "repeat" / "G2-replay" / "report.json").exists()


def test_the_replay_is_judged_by_the_model_that_judged_the_recorded_answers(monkeypatch, tmp_path):
    import src.evaluation.model_ablation as module
    import src.evaluation.quality as quality
    from src.evaluation.model_ablation import run_generation_replay

    built = []

    class Judge:
        def __init__(self, config, **kwargs):
            built.append(config)
            self.config = config

        def to_dict(self):
            return {"config": self.config.to_dict()}

    seen = []
    monkeypatch.setattr(quality, "LLMJudge", Judge)
    monkeypatch.setattr(module, "evaluate", _replaying(seen))
    recorded = _recorded(tmp_path)
    run_generation_replay(recorded, [GENERATION_EXPERIMENTS[2]], run_id="judged",
                          results_root=tmp_path, llm_for=lambda config: None)
    assert built == [recorded.generation]
    assert seen[0][3]["judge"].config == recorded.generation


def test_recorded_answers_reads_a_complete_matrix_run_s_c4_answers(answer_run):
    import pytest
    from src.evaluation import full_ablation as matrix
    from src.evaluation.model_ablation import recorded_answers

    matrix.answers(answer_run.args)   # records answers-config.json, as the real run did
    directory, snapshot = answer_run.runner.directory, answer_run.runner.snapshot
    # C4's own gold passages, as the run recorded them, rather than the question's.
    transactions = matrix.read_rows(directory / "pairs.jsonl")
    transactions[0]["rows"]["C4"]["scoring_supporting_chunk_ids"] = ["recorded-gold"]
    matrix.write_rows(directory / "pairs.jsonl", transactions)
    recorded = recorded_answers(directory, chunks=snapshot)

    team = matrix.team_questions(answer_run.runner.questions)
    assert [q.question_id for q in recorded.questions] == [q.question_id for q in team]
    assert [row["question_id"] for row in recorded.rows] == [q.question_id for q in team]
    assert recorded.generation == GenerationConfig("mistral", "test", "grounded_v4")
    pairs = {t["question"]["question_id"]: t["rows"]["C4"]
             for t in matrix.read_rows(directory / "pairs.jsonl")}
    for question in recorded.questions:
        measured = pairs[question.question_id]
        # Scored against the gold passages the run recorded for C4.
        assert question.supporting_chunk_ids == tuple(measured["scoring_supporting_chunk_ids"])
        # And answered over the rankings it recorded for C4.
        query = matrix.MATRIX["C4"].scoped(matrix._query(question, top_k=16, metadata_filter=True))
        assert [p.chunk_id for p in recorded.retriever.search(query)] == (
            measured["retrieved_chunk_ids"])

    # A run recorded before the matrix named a setting (rerank, for
    # full-matrix-20261017) ran with that setting's default, and is read.
    config = json.loads((directory / "config.json").read_text())
    del config["matrix"]["C4"]["rerank"]
    (directory / "config.json").write_text(json.dumps(config))
    assert recorded_answers(directory, chunks=snapshot).rows == recorded.rows
    # A setting it recorded otherwise is refused.
    config["matrix"]["C4"]["use_facts"] = True
    (directory / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="C4's settings differ"):
        recorded_answers(directory, chunks=snapshot)
    config["matrix"]["C4"]["use_facts"] = False
    # As is a setting this code does not know.
    config["matrix"]["C4"]["unknown"] = True
    (directory / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="C4's settings differ"):
        recorded_answers(directory, chunks=snapshot)
    del config["matrix"]["C4"]["unknown"]
    (directory / "config.json").write_text(json.dumps(config))

    # Review of #163: a query this code would build differently from the one
    # the run recorded, as a changed facts label or company list makes it, is
    # refused before any model is asked. It failed part-way through the paid
    # run instead, losing every answer before it.
    pairs = matrix.read_rows(directory / "pairs.jsonl")
    original = json.loads(json.dumps(pairs))
    pairs[-1]["rows"]["C4"]["query"]["wants_figures"] = not pairs[-1]["rows"]["C4"]["query"][
        "wants_figures"]
    matrix.write_rows(directory / "pairs.jsonl", pairs)
    # It names the store to look at, not the deleted copy it parsed with,
    # which keeps the store's file name (follow-up review of #163).
    from src.retrieval.facts import FACTS_FILE
    with pytest.raises(ValueError, match=r"query .* differs from the one the run recorded; "
                                         + re.escape(f"the facts store ({FACTS_FILE})")):
        recorded_answers(directory, chunks=snapshot)
    matrix.write_rows(directory / "pairs.jsonl", original)

    # Anything but the run's own corpus, or C4 answers short of the team, is refused.
    with pytest.raises(ValueError, match="corpus"):
        recorded_answers(directory, chunks=snapshot[:1])
    matrix.write_rows(directory / "C4" / "answers.jsonl", list(recorded.rows)[:-1])
    with pytest.raises(ValueError, match="C4 answers"):
        recorded_answers(directory, chunks=snapshot)


def test_a_run_that_is_missing_or_unfinished_is_refused_by_name(answer_run, tmp_path):
    # QA of #163: these raised FileNotFoundError, a traceback rather than a
    # refusal, for a mistyped path or a run whose answers never started.
    import pytest
    from src.evaluation.model_ablation import recorded_answers

    # PR #177 review: a folder that is not on this machine, as a run folder
    # is git-ignored, is called missing rather than the wrong kind of run.
    with pytest.raises(ValueError, match=r"nope does not exist: .*copy it here"):
        recorded_answers(tmp_path / "nope", chunks=[])
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match=r"not a full-matrix run: .*config\.json"):
        recorded_answers(tmp_path / "empty", chunks=[])
    # Another kind of run, such as a C5 rerank reference, has a config.json
    # but no answers step to finish.
    other = tmp_path / "c5-reference"
    other.mkdir()
    (other / "config.json").write_text(json.dumps({"provenance": {}, "policy": {}}))
    with pytest.raises(ValueError, match=r"not a full-matrix run: .*full-matrix-v1"):
        recorded_answers(other, chunks=[])
    (other / "config.json").write_text("[]")
    with pytest.raises(ValueError, match=r"not a full-matrix run: .*full-matrix-v1"):
        recorded_answers(other, chunks=[])
    # Retrieval finished, answers not yet started.
    directory = answer_run.runner.directory
    with pytest.raises(ValueError, match=r"has no answers yet: .*answers-config\.json"):
        recorded_answers(directory, chunks=answer_run.runner.snapshot)


def test_the_replay_parses_with_a_snapshot_of_the_facts_store(answer_run, tmp_path, monkeypatch):
    import tempfile

    # Follow-up review of #163: the queries were checked against the facts
    # store once, but the harness parsed every question again against the
    # live file, so a facts rebuild during a long replay could still fail it
    # part-way. The check and the replay now read one snapshot of it.
    from src.evaluation import full_ablation as matrix
    from src.evaluation.model_ablation import recorded_answers

    matrix.answers(answer_run.args)
    # The check's own copy of the store is gone once the run is read: QA of
    # #163 found one left in the temporary directory by every load.
    scratch = tmp_path / "tmp"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    facts = tmp_path / "facts.parquet"
    facts.write_bytes(b"facts as of the check")
    # What the query check parses with, read while it runs: review of #163
    # found nothing held it to the snapshot rather than the live store.
    import src.evaluation.model_ablation as module
    checked = []
    check = module._check_queries

    def spy(scored, measured, facts_file, *args, **kwargs):
        checked.append((facts_file, facts_file.read_bytes() if facts_file.exists() else None))
        return check(scored, measured, facts_file, *args, **kwargs)

    monkeypatch.setattr(module, "_check_queries", spy)
    recorded = recorded_answers(answer_run.runner.directory, chunks=answer_run.runner.snapshot,
                                facts_file=facts)
    facts.write_bytes(b"rebuilt part-way through")
    assert recorded.facts == b"facts as of the check"
    [(path, read)] = checked
    assert path != facts and read == b"facts as of the check"
    checked.clear()
    assert list(scratch.iterdir()) == []

    # A fresh clone has no store: there is nothing to copy, and the replay
    # parses with it missing, as the run that recorded the queries did.
    absent = tmp_path / "absent.parquet"
    recorded = recorded_answers(answer_run.runner.directory, chunks=answer_run.runner.snapshot,
                                facts_file=absent)
    assert (recorded.facts_file, recorded.facts) == (absent, None)
    # Checked with it missing, not with whatever appears at its path later.
    [(path, read)] = checked
    assert path != absent and read is None


def test_a_replay_without_a_facts_store_parses_with_it_missing(monkeypatch, tmp_path):
    from dataclasses import replace
    import src.evaluation.model_ablation as module
    from src.evaluation.model_ablation import run_generation_replay

    # Review of #163: it parsed with the live path, so a store built while the
    # replay ran could still change a later query part-way through.
    seen = []
    monkeypatch.setattr(module, "evaluate", _replaying(seen))
    live = tmp_path / "absent.parquet"
    recorded = replace(_recorded(tmp_path), facts_file=live, facts=None)
    run_generation_replay(recorded, [GENERATION_EXPERIMENTS[2]], run_id="no-facts",
                          results_root=tmp_path, llm_for=lambda config: None,
                          judge=SimpleNamespace(config=recorded.generation, to_dict=dict))
    parsed_with = seen[0][3]["facts_file"]
    assert parsed_with != live and parsed_with.parent == tmp_path / "no-facts"
    assert not parsed_with.exists()
    # Its report says there was no store, rather than naming a file never
    # written (follow-up review of #163).
    report = json.loads((tmp_path / "no-facts" / "G3" / "report.json").read_text())
    assert report["facts_file"] is None


def test_a_run_judged_otherwise_or_short_of_its_team_retrieval_is_refused(answer_run):
    import pytest
    from src.evaluation import full_ablation as matrix
    from src.evaluation.model_ablation import recorded_answers

    matrix.answers(answer_run.args)
    directory, snapshot = answer_run.runner.directory, answer_run.runner.snapshot
    answered = json.loads((directory / "answers-config.json").read_text())
    (directory / "answers-config.json").write_text(
        json.dumps(answered | {"judge_rubric": "answer-quality-v0"}))
    with pytest.raises(ValueError, match="judged with 'answer-quality-v0'"):
        recorded_answers(directory, chunks=snapshot)
    (directory / "answers-config.json").write_text(json.dumps(answered))

    # A retrieval stopped before its last team question.
    team = {q.question_id for q in matrix.team_questions(answer_run.runner.questions)}
    pairs = matrix.read_rows(directory / "pairs.jsonl")
    last = max(i for i, t in enumerate(pairs) if t["question"]["question_id"] in team)
    matrix.write_rows(directory / "pairs.jsonl", pairs[:last])
    with pytest.raises(ValueError, match="team retrieval is incomplete"):
        recorded_answers(directory, chunks=snapshot)


def test_the_replay_command_runs_the_rows_it_names_over_the_run_it_names(monkeypatch, tmp_path,
                                                                          capsys):
    import pytest
    import src.evaluation.model_ablation as module
    from src.rag import ProviderUnavailable

    seen = {}
    monkeypatch.setattr(module, "recorded_answers",
                        lambda run_dir, **kwargs: seen.update(run_dir=run_dir, **kwargs) or "recorded")

    def replay(recorded, experiments, *, run_id, results_root, **kwargs):
        seen.update(recorded=recorded, rows=[e.id for e in experiments], run_id=run_id)
        (results_root / run_id).mkdir(parents=True)
        (results_root / run_id / "summary.csv").write_text("config,correctness\nG3,0.9\n")

    monkeypatch.setattr(module, "run_generation_replay", replay)
    module.main(["replay", "results/full-matrix-20261017", "--rows", "G3", "--run-id",
                 "generators-c4", "--results-root", str(tmp_path)])
    assert seen["run_dir"] == module.Path("results/full-matrix-20261017")
    assert (seen["recorded"], seen["rows"], seen["run_id"]) == ("recorded", ["G3"], "generators-c4")
    assert "G3,0.9" in capsys.readouterr().out

    # A missing key is a usage error before anything runs, as for the other matrices.
    def no_key(*args, **kwargs):
        raise ProviderUnavailable("SOCLAAS_API_KEY is not set")

    monkeypatch.setattr(module, "run_generation_replay", no_key)
    with pytest.raises(SystemExit) as raised:
        module.main(["replay", "run", "--rows", "G3", "--run-id", "again",
                     "--results-root", str(tmp_path)])
    assert raised.value.code == 2 and "SOCLAAS_API_KEY is not set" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        module.main(["replay", "run", "--rows", "G9", "--run-id", "x",
                     "--results-root", str(tmp_path)])

    def stopped(*args, **kwargs):
        raise RunStopped({"results": [], "stopped": {"question_id": "q2", "error": "busy"}})

    monkeypatch.setattr(module, "run_generation_replay", stopped)
    with pytest.raises(SystemExit) as raised:
        module.main(["replay", "run", "--rows", "G3", "--run-id", "halted",
                     "--results-root", str(tmp_path)])
    assert str(raised.value).endswith(f"partial results are under {tmp_path / 'halted'}")


def test_the_top_level_help_lists_the_replay_command(capsys):
    # QA of #163 (#180): main() sends `replay` to its own parser before
    # argparse runs, so the top-level help never showed it.
    import pytest
    import src.evaluation.model_ablation as module

    with pytest.raises(SystemExit) as raised:
        module.main(["--help"])
    assert raised.value.code == 0
    help_text = " ".join(capsys.readouterr().out.split())
    assert "python -m src.evaluation.model_ablation replay --help" in help_text


def test_a_replay_s_missing_key_advice_does_not_offer_llm_provider(monkeypatch, tmp_path, capsys):
    # QA of #163 (#180): a replay row fixes its provider, so setting
    # LLM_PROVIDER, which the provider's general advice offers, changes nothing.
    import pytest
    import src.evaluation.model_ablation as module

    monkeypatch.setattr(module, "recorded_answers", lambda run_dir, **kwargs: _recorded(tmp_path))
    monkeypatch.setenv("MISTRAL_API_KEY", "judge-key")   # the judge's, so G3's key is the one missing
    with pytest.raises(SystemExit) as raised:
        module.main(["replay", "run", "--rows", "G3", "--run-id", "no-key",
                     "--results-root", str(tmp_path)])
    error = capsys.readouterr().err
    assert raised.value.code == 2 and "SOCLAAS_API_KEY is not set: make a key" in error
    assert "LLM_PROVIDER" not in error
    assert error.rstrip().endswith("so the key is read")


@pytest.mark.parametrize("error, advice", [
    ("ProviderBusy: SOCLAAS sent nothing for 120s while running 'qwen3.6:35b': ask again, "
     "or set LLM_PROVIDER=ollama to answer locally",
     "ProviderBusy: SOCLAAS sent nothing for 120s while running 'qwen3.6:35b': ask again"),
    ("ProviderUnavailable: SOCLAAS stopped 'qwen3.6:35b' without finishing (finish_reason "
     "'content_filter'); set LLM_PROVIDER=ollama to answer locally",
     "ProviderUnavailable: SOCLAAS stopped 'qwen3.6:35b' without finishing (finish_reason "
     "'content_filter')"),
], ids=["ask-again", "finish-reason"])
def test_a_replay_stopped_part_way_does_not_offer_llm_provider(monkeypatch, tmp_path, error,
                                                               advice):
    import src.evaluation.model_ablation as module

    def stopped(*args, **kwargs):
        raise RunStopped({"results": [], "stopped": {"question_id": "q2", "error": error}})

    monkeypatch.setattr(module, "recorded_answers", lambda run_dir, **kwargs: "recorded")
    monkeypatch.setattr(module, "run_generation_replay", stopped)
    with pytest.raises(SystemExit) as raised:
        module.main(["replay", "run", "--rows", "G3", "--run-id", "halted",
                     "--results-root", str(tmp_path)])
    assert str(raised.value) == (
        f"stopped at q2: {advice}; partial results are under {tmp_path / 'halted'}")


def test_a_stopped_replay_s_table_says_which_row_stopped(monkeypatch, tmp_path):
    # QA of #163 (#180): the table showed a partial row's figures as if it were complete.
    import src.evaluation.model_ablation as module
    from src.evaluation.model_ablation import run_generation_replay

    def stops(questions, retriever, config, **kwargs):
        raise RunStopped({"results": [_recorded_row("q001")],
                          "stopped": {"question_id": "q002", "error": "rate limit"}})

    monkeypatch.setattr(module, "evaluate", stops)
    recorded = _recorded(tmp_path)
    with pytest.raises(RunStopped):
        run_generation_replay(recorded, [GENERATION_EXPERIMENTS[2]], run_id="stopped",
                              results_root=tmp_path, llm_for=lambda config: None,
                              judge=SimpleNamespace(config=recorded.generation, to_dict=dict))
    with (tmp_path / "stopped" / "summary.csv").open(newline="") as stream:
        table = {row["config"]: row.get("stopped") for row in csv.DictReader(stream)}
    # The recorded answers are complete; the replayed row names where it stopped.
    assert table["G2"] == "" and "q002" in table["G3"]


def test_an_answers_config_without_its_fields_is_refused_by_name(answer_run):
    # QA of #163 (#180): `{}` raised KeyError: 'judge_rubric', a traceback
    # rather than a refusal.
    from src.evaluation import full_ablation as matrix
    from src.evaluation.model_ablation import recorded_answers

    matrix.answers(answer_run.args)
    directory, snapshot = answer_run.runner.directory, answer_run.runner.snapshot
    (directory / "answers-config.json").write_text("{}")
    with pytest.raises(ValueError, match=r"answers-config\.json does not record 'judge_rubric'"):
        recorded_answers(directory, chunks=snapshot)
    (directory / "answers-config.json").write_text(json.dumps({"judge_rubric": "answer-quality-v1"}))
    with pytest.raises(ValueError, match=r"answers-config\.json does not record 'generation'"):
        recorded_answers(directory, chunks=snapshot)
    (directory / "answers-config.json").write_text("null")
    with pytest.raises(ValueError, match=r"answers-config\.json does not record 'judge_rubric'"):
        recorded_answers(directory, chunks=snapshot)


def test_a_stopped_replay_s_record_does_not_offer_llm_provider(monkeypatch, tmp_path):
    # QA of #180: the stop a replay records kept the provider's general advice.
    import src.evaluation.model_ablation as module
    from src.evaluation.model_ablation import run_generation_replay
    from src.rag.generate import ProviderBusy

    unreachable = ProviderBusy("cannot reach SOCLAAS")

    def stops(questions, retriever, config, **kwargs):
        raise RunStopped({"results": [_recorded_row("q001")], "stopped": {
            "question_id": "q002", "error": "ProviderBusy: cannot reach SOCLAAS: try another "
                                            "network, or set LLM_PROVIDER=ollama to answer locally"}}
                         ) from unreachable

    monkeypatch.setattr(module, "evaluate", stops)
    recorded = _recorded(tmp_path)
    with pytest.raises(RunStopped) as raised:
        run_generation_replay(recorded, [GENERATION_EXPERIMENTS[2]], run_id="stopped",
                              results_root=tmp_path, llm_for=lambda config: None,
                              judge=SimpleNamespace(config=recorded.generation, to_dict=dict))
    report = json.loads((tmp_path / "stopped" / "G3" / "report.json").read_text())
    summary = json.loads((tmp_path / "stopped" / "summary.json").read_text())
    advice = "ProviderBusy: cannot reach SOCLAAS: try another network"
    # PR review of #180: a caller other than the command, such as a notebook,
    # reads the error it is raised rather than the files.
    assert str(raised.value) == f"stopped at q002: {advice}"
    assert raised.value.__cause__ is unreachable
    assert report["stopped"]["error"] == advice
    assert summary["configurations"][1]["stopped"]["error"] == advice
    assert "LLM_PROVIDER" not in (tmp_path / "stopped" / "summary.csv").read_text()


def test_a_replay_s_retry_notices_do_not_offer_llm_provider(monkeypatch, tmp_path, caplog):
    # QA of #180: a busy provider's "asking again" notice during a replay
    # offered LLM_PROVIDER, which a replay row's fixed provider ignores.
    import src.evaluation.harness as harness
    import src.evaluation.model_ablation as module
    from src.evaluation.model_ablation import run_generation_replay
    from src.rag.generate import ProviderBusy

    busy = ("SOCLAAS failed while running 'qwen3.6:35b' (HTTP 503): ask again, or set "
            "LLM_PROVIDER=ollama to answer locally")

    def ask_twice():
        failures = iter([ProviderBusy(busy)])

        def ask():
            for failure in failures:
                raise failure
            return "answer"
        return harness.answer_with_retries("q001", ask)

    def evaluate(questions, retriever, config, **kwargs):
        ask_twice()
        return {"results": [_recorded_row("q001"), _recorded_row(
            "q002", question_type="unanswerable", abstained=True)]}

    monkeypatch.setattr(harness, "sleep", lambda seconds: None)
    monkeypatch.setattr(module, "evaluate", evaluate)
    recorded = _recorded(tmp_path)
    with caplog.at_level("WARNING", logger=harness.__name__):
        run_generation_replay(recorded, [GENERATION_EXPERIMENTS[2]], run_id="busy",
                              results_root=tmp_path, llm_for=lambda config: None,
                              judge=SimpleNamespace(config=recorded.generation, to_dict=dict))
        [notice] = caplog.messages
        assert notice == ("q001: SOCLAAS failed while running 'qwen3.6:35b' (HTTP 503): ask "
                          "again; asking again in 10s")
        # Outside a replay, where LLM_PROVIDER does choose the provider, the advice stays.
        caplog.clear()
        ask_twice()
        assert "LLM_PROVIDER=ollama" in caplog.messages[0]
