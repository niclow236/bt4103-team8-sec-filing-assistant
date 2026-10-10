"""Tests for E1-E3 and G1-G2 model ablations."""

import csv
import json
from types import SimpleNamespace

from src.evaluation.model_ablation import (
    EMBEDDING_EXPERIMENTS,
    GENERATION_EXPERIMENTS,
    run_embedding_ablation,
    run_generation_ablation,
)
from src.evaluation.records import BenchmarkQuestion
from src.rag.records import GenerationConfig
from src.retrieval.records import RetrievedPassage


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


def test_generation_matrix_compares_local_and_hosted_providers(monkeypatch, tmp_path):
    import src.evaluation.model_ablation as module

    monkeypatch.setattr(module, "evaluate", _fake_evaluate)
    manifest = run_generation_ablation(
        [_question()], _stack, run_id="generation-run", results_root=tmp_path
    )

    assert [row["config"]["id"] for row in manifest["configurations"]] == ["G1", "G2"]
    assert {row["config"]["provider"] for row in manifest["configurations"]} == {
        experiment.provider for experiment in GENERATION_EXPERIMENTS
    }
    assert json.loads(
        (tmp_path / "generation-run" / "G2" / "report.json").read_text()
    )["config"]["provider"] == "mistral"


def test_generation_uses_c4_stack_settings(monkeypatch, tmp_path):
    import src.evaluation.model_ablation as module

    seen = []

    def record(questions, retriever, config, **kwargs):
        seen.append(kwargs)
        return _fake_evaluate(questions, retriever, config)

    monkeypatch.setattr(module, "evaluate", record)
    run_generation_ablation([_question()], _stack, run_id="settings", results_root=tmp_path)

    c4 = _stack(GENERATION_EXPERIMENTS[0]).config
    assert [kwargs["stack"] for kwargs in seen] == [c4, c4]
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
    assert [row["provider"] for row in seen] == ["ollama", "mistral"]


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
