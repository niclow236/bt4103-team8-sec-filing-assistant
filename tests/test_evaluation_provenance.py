"""Input changes must be visible even when filenames and chunk IDs survive."""

from src.evaluation.provenance import record_provenance
from src.evaluation.records import BenchmarkQuestion
from src.retrieval import constants
import subprocess

import pytest


@pytest.mark.parametrize("error", [FileNotFoundError("git"), subprocess.SubprocessError("git")])
def test_git_unavailable_records_unknown_commit(monkeypatch, error):
    from src.evaluation.provenance import _git

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(subprocess, "run", fail)
    assert _git("rev-parse", "HEAD") is None


def test_provenance_tracks_benchmark_text_and_corpus_metadata(monkeypatch, tmp_path):
    path = tmp_path / "questions.jsonl"
    path.write_text("original input")
    chunk = {"chunk_id": "chunk", "text": "$100", "accession_no": "filing",
             "ticker": "AAPL", "chunk_budget": 1800, "chunk_overlap": 200}
    monkeypatch.setattr("src.evaluation.provenance.iter_chunks", lambda **kwargs: [chunk])
    monkeypatch.setattr("src.evaluation.provenance._git", lambda *args: "commit")
    question = BenchmarkQuestion(
        question_id="q", question="Revenue?", expected_answer="$100",
        supporting_chunk_ids=("chunk",), hard_negative_chunk_ids=(),
        ticker="AAPL", fiscal_year=2024, question_type="numeric",
        difficulty="easy", source="handwritten",
    )
    first = record_provenance([path], [question], tmp_path)
    assert first["source_commit"] == "commit"
    assert first["by_benchmark"] == {"team": 1}
    assert first["corpus"]["n_passages"] == first["corpus"]["n_filings"] == 1
    assert first["retrieval_settings"]["BM25_K1"] == constants.BM25_K1
    assert first["environment"]["packages"]["sentence-transformers"]
    path.write_text("changed input")
    chunk["ticker"] = "MSFT"
    second = record_provenance([path], [question], tmp_path)
    assert first["inputs"][0]["sha256"] != second["inputs"][0]["sha256"]
    # Text-only BM25 identity stays the same; the full metadata digest catches
    # the changed scope that would alter a filtered search.
    assert first["corpus"]["fingerprint"] == second["corpus"]["fingerprint"]
    assert first["corpus"]["records_sha256"] != second["corpus"]["records_sha256"]


def test_provenance_counts_untagged_team_questions_beside_the_xbrl_set(monkeypatch, tmp_path):
    monkeypatch.setattr("src.evaluation.provenance.iter_chunks", lambda **kwargs: [])
    monkeypatch.setattr("src.evaluation.provenance._git", lambda *args: "commit")
    team = BenchmarkQuestion(
        question_id="q", question="Revenue?", expected_answer="$100",
        supporting_chunk_ids=(), hard_negative_chunk_ids=(), ticker="AAPL", fiscal_year=2024,
        question_type="numeric", difficulty="easy",
    )
    xbrl = BenchmarkQuestion(**{**team.to_dict(), "question_id": "x", "source": "xbrl",
                                "supporting_chunk_ids": (), "hard_negative_chunk_ids": ()})
    recorded = record_provenance([], [team, team, xbrl], tmp_path)
    assert recorded["by_benchmark"] == {"team": 2, "xbrl": 1}


def test_question_digest_keeps_the_form_recorded_runs_were_frozen_with():
    from src.evaluation.provenance import question_records_sha256
    team = BenchmarkQuestion(
        question_id="q", question="Revenue?", expected_answer="$100",
        supporting_chunk_ids=("chunk",), hard_negative_chunk_ids=(), ticker="AAPL", fiscal_year=2024,
        question_type="numeric", difficulty="easy",
    )
    # Recorded manifests hold this digest; changing the hashed form orphans them.
    assert question_records_sha256([team]) == "d8bdd1dd29762137914335744c3dd68915716d9adb058f40f8b7706e01ac0ca4"


def test_supplied_corpus_is_fingerprinted_without_reading_it_again(monkeypatch, tmp_path):
    def unexpected(**kwargs):
        pytest.fail("caller already loaded this corpus")
    monkeypatch.setattr("src.evaluation.provenance.iter_chunks", unexpected)
    chunks = [{"chunk_id": "a", "text": "value", "accession_no": "filing"}]
    result = record_provenance([], [], tmp_path, chunks=chunks)
    assert result["corpus"]["n_passages"] == result["corpus"]["n_filings"] == 1
