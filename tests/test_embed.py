"""The dense index: resume, the manifest, truncation, and what #19 builds on.

Each test drives the real ``embed.build`` against a synthetic corpus and a
temporary index, with the encoder replaced by the deterministic fake in
conftest, so the whole file runs in seconds and needs no model.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from conftest import edit_filing, vector_for
from src.pipeline.chunk import iter_chunks
from src.retrieval import embed
from src.retrieval.base import matches
from src.retrieval.records import Query, RetrievedPassage, fingerprint_of


def build(corpus, chroma, **kwargs):
    return embed.build(chroma_dir=chroma, processed_dir=corpus, batch_size=8, **kwargs)


def stored(chroma):
    got = embed.open_collection(chroma).get(include=["documents", "metadatas", "embeddings"])
    return {chunk_id: (document, metadata, np.asarray(vector))
            for chunk_id, document, metadata, vector
            in zip(got["ids"], got["documents"], got["metadatas"], got["embeddings"])}


def corpus_digests(corpus):
    return {row["chunk_id"]: embed.digest_of(row) for row in iter_chunks(processed_dir=corpus)}


@pytest.fixture
def chroma(tmp_path):
    return tmp_path / "chroma"


def test_fresh_build_is_current(corpus, chroma, fake_model):
    manifest = build(corpus, chroma)
    index = stored(chroma)
    rows = list(iter_chunks(processed_dir=corpus))

    assert manifest.corpus_fingerprint == fingerprint_of(corpus_digests(corpus).values())
    assert embed.check_index(chroma, corpus) == []
    assert len(index) == len(rows)
    assert all(index[row["chunk_id"]][0] == row["text"] for row in rows)
    assert all(set(embed.INDEX_FIELDS) <= set(metadata) for _, metadata, _ in index.values())
    assert (manifest.chunk_budget, manifest.chunk_overlap) == (1800, 300)


def test_sidecar_files_live_beside_the_index(corpus, chroma, fake_model):
    build(corpus, chroma)
    assert embed.manifest_file_for(chroma).exists()
    assert embed.manifest_file_for(chroma).parent == chroma.parent


def test_batch_size_below_one_is_refused(corpus, chroma):
    with pytest.raises(ValueError):
        embed.build(chroma_dir=chroma, processed_dir=corpus, batch_size=0)


def test_killed_build_leaves_no_manifest_and_resume_replaces_rechunked(corpus, chroma, fake_model):
    """Finding 1: re-chunk between a kill and a resume, with no manifest on disk."""
    build(corpus, chroma)
    fake_model.fail_after, fake_model.calls = 2, 0
    with pytest.raises(KeyboardInterrupt):
        build(corpus, chroma, rebuild=True)
    fake_model.fail_after = None
    assert not embed.manifest_file_for(chroma).exists()
    assert any("no manifest" in problem for problem in embed.check_index(chroma, corpus))

    first = sorted(corpus.glob("*/*.json"))[0]
    written = set(stored(chroma))
    changed = [c["chunk_id"] for c in json.loads(first.read_text(encoding="utf-8"))["chunks"][:3]]
    assert set(changed) <= written, "the edited passages must have been embedded before the kill"
    edit_filing(first, lambda d: [c.update(text=c["text"] + " [re-chunked]") for c in d["chunks"][:3]])

    build(corpus, chroma)
    index = stored(chroma)
    for row in iter_chunks(processed_dir=corpus):
        if row["chunk_id"] in changed:
            assert index[row["chunk_id"]][1]["digest"] == embed.digest_of(row)
            assert np.allclose(index[row["chunk_id"]][2], vector_for(embed.embed_text(row)), atol=1e-6)
    assert embed.check_index(chroma, corpus) == []


def test_replacement_does_not_keep_metadata_the_passage_lost(corpus, chroma, fake_model):
    """Chroma merges metadata on upsert, so a replaced vector must be re-added."""
    build(corpus, chroma)
    first = sorted(corpus.glob("*/*.json"))[0]
    target = next(c["chunk_id"] for c in json.loads(first.read_text(encoding="utf-8"))["chunks"]
                  if c.get("heading"))

    def drop_heading(data):
        for chunk in data["chunks"]:
            if chunk["chunk_id"] == target:
                chunk["heading"] = None
                chunk["text"] += " [no heading now]"

    edit_filing(first, drop_heading)
    build(corpus, chroma)
    assert "heading" not in stored(chroma)[target][1]


def test_full_run_removes_passages_that_left_the_corpus(corpus, chroma, fake_model):
    build(corpus, chroma)
    gone = sorted(corpus.glob("*/*.json"))[-1]
    gone_ids = [c["chunk_id"] for c in json.loads(gone.read_text(encoding="utf-8"))["chunks"]]
    gone.unlink()
    build(corpus, chroma)
    assert not set(gone_ids) & set(stored(chroma))
    assert embed.check_index(chroma, corpus) == []


def test_narrowed_run_leaves_stale_outside_its_filter_and_says_so(corpus, chroma, fake_model):
    build(corpus, chroma)
    other = next(p for p in sorted(corpus.glob("*/*.json")) if p.parent.name != "AAA")
    edit_filing(other, lambda d: d["chunks"][0].update(text=d["chunks"][0]["text"] + " [moved]"))
    build(corpus, chroma, tickers=["AAA"])
    problems = embed.check_index(chroma, corpus)
    assert any("1 stale" in problem for problem in problems), problems
    build(corpus, chroma)
    assert embed.check_index(chroma, corpus) == []


def test_partial_rebuild_leaves_an_honest_manifest(corpus, chroma, fake_model):
    """Finding 2: --rebuild --tickers must not leave a full-corpus manifest."""
    build(corpus, chroma)
    manifest = build(corpus, chroma, rebuild=True, tickers=["AAA"])
    aaa = sum(1 for row in iter_chunks(processed_dir=corpus) if row["ticker"] == "AAA")
    assert manifest.n_passages == aaa
    assert any("missing" in problem for problem in embed.check_index(chroma, corpus))


def test_vectors_from_another_model_are_refused(corpus, chroma, fake_model, monkeypatch):
    build(corpus, chroma)
    monkeypatch.setattr(embed, "EMBED_MODEL", "some/other-model")
    with pytest.raises(RuntimeError, match="--rebuild"):
        build(corpus, chroma)
    assert any("not encoded by" in problem for problem in embed.check_index(chroma, corpus))


def test_legacy_vectors_without_a_model_record_are_refused(corpus, tmp_path):
    legacy = tmp_path / "legacy"
    embed.open_collection(legacy).add(
        ids=["a"], embeddings=[[0.0] * embed.EMBED_DIMENSIONS], documents=["t"],
        metadatas=[{"ticker": "A"}],
    )
    with pytest.raises(RuntimeError, match="None"):
        embed.build(chroma_dir=legacy, processed_dir=corpus)


def test_truncation_report_is_read_from_the_index(corpus, chroma, fake_model):
    report = embed.truncation_file_for(chroma)
    target = sorted(corpus.glob("*/*.json"))[0]
    edit_filing(target, lambda d: [c.update(text=c["text"] + " LONGLONG") for c in d["chunks"][:4]])
    build(corpus, chroma)
    assert json.loads(report.read_text(encoding="utf-8"))["n_truncated"] == 4

    # A later resume that encodes something else must not shrink the report.
    edit_filing(target, lambda d: d["chunks"][6].update(text=d["chunks"][6]["text"] + " x"))
    build(corpus, chroma)
    assert json.loads(report.read_text(encoding="utf-8"))["n_truncated"] == 4

    edit_filing(target, lambda d: [c.update(text=c["text"].replace(" LONGLONG", ""))
                                   for c in d["chunks"][:4]])
    build(corpus, chroma)
    assert not report.exists()


QUERIES = [
    Query("x"),
    Query("x", tickers=("AAA",)),
    Query("x", tickers=("aaa", "ccc"), fiscal_years=(2024,)),
    Query("x", items=("1a", "7")),
    Query("x", content_type="table"),
    Query("x", key_items_only=True),
    Query("x", tickers=("BBB",), fiscal_years=(2023,), items=("8",), content_type="table"),
]


@pytest.mark.parametrize("query", QUERIES)
def test_dense_filter_selects_what_bm25_selects(corpus, chroma, fake_model, query):
    """BM25 filters with base.matches; the dense index with where_for. They must agree."""
    build(corpus, chroma)
    collection = embed.open_collection(chroma)
    where = embed.where_for(query)
    dense = set(collection.get(where=where, include=[])["ids"]) if where else set(stored(chroma))
    sparse = {row["chunk_id"] for row in iter_chunks(processed_dir=corpus) if matches(row, query)}
    assert dense == sparse


def test_stored_records_project_like_the_rows_bm25_holds(corpus, chroma, fake_model):
    build(corpus, chroma)
    index = stored(chroma)
    for row in iter_chunks(processed_dir=corpus):
        document, metadata, _ = index[row["chunk_id"]]
        from_index = embed.chunk_from_record(row["chunk_id"], document, metadata)
        assert (RetrievedPassage.from_chunk(from_index, 0.0, 1, "x")
                == RetrievedPassage.from_chunk(row, 0.0, 1, "x"))


def test_a_record_with_empty_fields_still_projects():
    row = embed.chunk_from_record("x", "text", {"ticker": "A", "company": "A", "url": "u"})
    passage = RetrievedPassage.from_chunk(row, 0.0, 1, "dense")
    assert passage.item is None and passage.title == "" and passage.fiscal_year is None
