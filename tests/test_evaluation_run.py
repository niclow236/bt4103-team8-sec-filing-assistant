"""Tests for the C0-C4 retrieval ablation runner."""

import csv
import json

import pytest

from src.evaluation.records import BenchmarkQuestion
from src.evaluation.run import CONFIGURATIONS, run_ablation, run_configuration
from src.retrieval.records import Query, RetrievedPassage


def _question(question_id="q001", **overrides):
    values = {
        "question_id": question_id,
        "question": "What was revenue?",
        "expected_answer": "$100",
        "supporting_chunk_ids": ("support",),
        "hard_negative_chunk_ids": ("hard-negative",),
        "ticker": "AAPL",
        "fiscal_year": 2024,
        "question_type": "numeric",
        "difficulty": "easy",
        "source": "handwritten",
    }
    values.update(overrides)
    return BenchmarkQuestion(**values)


def _passage(chunk_id, rank, retriever="bm25"):
    return RetrievedPassage(
        chunk_id=chunk_id,
        text=chunk_id,
        score=float(10 - rank),
        rank=rank,
        retriever=retriever,
        ticker="AAPL",
        company="Apple",
        fiscal_year=2024,
        item="7",
        title="Management's Discussion and Analysis",
        url="https://example.com/filing",
    )


class FakeRetriever:
    name = "bm25"

    def __init__(self):
        self.queries = []

    def search(self, query: Query, k=None):
        self.queries.append(query)
        return [_passage("support", 1, self.name)]


def test_run_configuration_writes_per_question_and_summary_files(tmp_path):
    retriever = FakeRetriever()

    summary = run_configuration(
        [_question()], retriever, config_id="C1", output_dir=tmp_path / "C1", top_k=5
    )

    assert summary["questions"] == 1
    assert summary["recall"] == 1.0
    assert retriever.queries[0].top_k == 5
    row = json.loads((tmp_path / "C1" / "questions.jsonl").read_text())
    assert row["config"]["id"] == "C1"
    assert row["config"]["chunking"] == "section-aware"
    assert json.loads((tmp_path / "C1" / "summary.json").read_text())["mrr"] == 1.0


def test_c4_applies_question_metadata_filters(tmp_path):
    retriever = FakeRetriever()

    run_configuration(
        [_question(question="What was revenue for AAPL in FY2024?")],
        retriever,
        config_id="C4",
        output_dir=tmp_path / "C4",
    )

    assert retriever.queries[0].tickers == ("AAPL",)
    assert retriever.queries[0].fiscal_years == (2024,)
    assert retriever.queries[0].keyword_text == "What was revenue for?"


def test_run_ablation_writes_all_rows_and_summary_table(tmp_path):
    retrievers = {config_id: FakeRetriever() for config_id in CONFIGURATIONS}

    manifest = run_ablation(
        [_question()], retrievers, run_id="fixture-run", results_root=tmp_path
    )

    run_dir = tmp_path / "fixture-run"
    assert manifest["run_id"] == "fixture-run"
    assert {path.name for path in run_dir.iterdir()} == {
        "C0", "C1", "C2", "C3", "C4", "summary.csv", "summary.json"
    }
    with (run_dir / "summary.csv").open(newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 5


def test_rerunning_an_existing_run_id_is_refused(tmp_path):
    retrievers = {config_id: FakeRetriever() for config_id in CONFIGURATIONS}
    run_ablation([_question()], retrievers, run_id="run", results_root=tmp_path)

    with pytest.raises(FileExistsError, match="already holds a run"):
        run_ablation(
            [_question()],
            retrievers,
            run_id="run",
            results_root=tmp_path,
            configurations=("C2",),
        )

    with (tmp_path / "run" / "summary.csv").open(newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 5


def test_run_id_and_top_k_are_validated(tmp_path):
    with pytest.raises(ValueError, match="run_id"):
        run_ablation([], {}, run_id="bad/run", results_root=tmp_path)
    with pytest.raises(ValueError, match="top_k"):
        run_configuration([], FakeRetriever(), config_id="C1", output_dir=tmp_path, top_k=0)


def test_c0_is_not_run_on_the_section_aware_bm25_index(tmp_path):
    with pytest.raises(ValueError, match="missing retriever for C0"):
        run_ablation(
            [_question()],
            {"bm25": FakeRetriever()},
            run_id="run",
            results_root=tmp_path,
            configurations=("C0",),
        )


def test_c0_labels_every_window_a_passage_overlaps(monkeypatch):
    from pathlib import Path

    from src.evaluation.run import FixedSizeBM25Retriever

    opening = "Total net sales by category were as follows for fiscal 2024 and 2023, in million"
    first = "Sales. " + opening
    second = opening + ": iPhone 201,183; total 391,035."
    metadata = {
        "accession_no": "acc",
        "ticker": "AAPL",
        "company": "Apple",
        "fiscal_year": 2024,
        "item": "7",
        "title": "MD&A",
        "url": "https://example.com/filing",
        "content_type": "prose",
    }
    monkeypatch.setattr(
        "src.evaluation.run.iter_chunks",
        lambda processed_dir: [
            dict(metadata, chunk_id="a", text=first),
            dict(metadata, chunk_id="b", text=second),
        ],
    )

    c0 = FixedSizeBM25Retriever(Path("unused"), budget=100)
    question = _question(supporting_chunk_ids=("b",), hard_negative_chunk_ids=())

    assert c0.relevant_chunk_ids(question) == [
        "fixed-acc-00000000",
        "fixed-acc-00000100",
    ]
