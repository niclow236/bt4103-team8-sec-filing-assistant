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

    assert seen["embedding_model"] == "intfloat/e5-base-v2"
    assert seen["embedding_query_prefix"] == "query: "
    assert seen["embedding_passage_prefix"] == "passage: "


def test_embedding_index_preparation_uses_each_registry_configuration(monkeypatch, tmp_path):
    import src.evaluation.model_ablation as module

    calls = []
    monkeypatch.setattr(
        "src.retrieval.embed.build",
        lambda **kwargs: calls.append(kwargs),
    )

    module.prepare_embedding_indexes(tmp_path / "processed")

    assert [call["model_name"] for call in calls] == [
        experiment.model for experiment in EMBEDDING_EXPERIMENTS
    ]
    assert not any(call["rebuild"] for call in calls)
