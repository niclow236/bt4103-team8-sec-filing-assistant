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
    class FakeRetriever:
        name = "hybrid"

        def search(self, query):
            return []

    return SimpleNamespace(
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

    monkeypatch.setattr(module, "evaluate", _fake_evaluate)
    manifest = run_embedding_ablation(
        [_question()], _stack, run_id="embedding-run", results_root=tmp_path
    )

    assert [row["config"] for row in manifest["rows"]] == ["E1", "E2", "E3"]
    assert {row["embedding_model"] for row in manifest["rows"]} == {
        experiment.model for experiment in EMBEDDING_EXPERIMENTS
    }
    with (tmp_path / "embedding-run" / "summary.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 3
    assert "abstention_rate" in rows[0]


def test_generation_matrix_compares_local_and_hosted_providers(monkeypatch, tmp_path):
    import src.evaluation.model_ablation as module

    monkeypatch.setattr(module, "evaluate", _fake_evaluate)
    manifest = run_generation_ablation(
        [_question()], _stack, run_id="generation-run", results_root=tmp_path
    )

    assert [row["config"] for row in manifest["rows"]] == ["G1", "G2"]
    assert {row["provider"] for row in manifest["rows"]} == {
        experiment.provider for experiment in GENERATION_EXPERIMENTS
    }
    assert json.loads(
        (tmp_path / "generation-run" / "G2" / "report.json").read_text()
    )["config"]["provider"] == "mistral"


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
    assert all(call["rebuild"] for call in calls)
