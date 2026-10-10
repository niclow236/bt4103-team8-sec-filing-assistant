"""Input changes must be visible even when filenames and chunk IDs survive."""

from src.evaluation.provenance import record_provenance
from src.evaluation.records import BenchmarkQuestion


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
    assert first["by_source"] == {"handwritten": 1}
    assert first["corpus"]["n_passages"] == first["corpus"]["n_filings"] == 1
    assert first["retrieval_settings"]["BM25_K1"] == 1.5
    assert first["environment"]["packages"]["sentence-transformers"]
    path.write_text("changed input")
    chunk["ticker"] = "MSFT"
    second = record_provenance([path], [question], tmp_path)
    assert first["inputs"][0]["sha256"] != second["inputs"][0]["sha256"]
    # Text-only BM25 identity stays the same; the full metadata digest catches
    # the changed scope that would alter a filtered search.
    assert first["corpus"]["fingerprint"] == second["corpus"]["fingerprint"]
    assert first["corpus"]["records_sha256"] != second["corpus"]["records_sha256"]
