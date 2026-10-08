"""The fingerprint guard (#49): an index is refused by any corpus but the one it was built from.

An index is the corpus in another form, and the corpus moves: the chunk stage
takes ``--budget`` and ``--overlap``, and filings are added to the scope. An
index left behind still answers, from passages that no longer exist, and
nothing in a result says so. So each index writes an ``IndexManifest`` holding
a fingerprint of the passages that went in, and its retriever compares that
with the corpus on disk before it will load.

Both indexes are held to it here, the same way, because the guard is one idea
with two implementations: BM25 keeps its manifest inside ``bm25.pkl`` and
fingerprints the stored text, and the dense index keeps a file beside
``chroma/`` and fingerprints the text it encoded, context header included.
"""

from __future__ import annotations

import json
import pickle
import shutil
from dataclasses import replace

import pytest

from conftest import edit_filing, write_corpus
from src.pipeline.chunk import chunk_filing, iter_chunks, write_chunks
from src.pipeline.records import ParsedFiling, SectionRecord
from src.retrieval import embed
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.records import (
    IndexManifest,
    Query,
    corpus_fingerprint,
    fingerprint_of,
)

KINDS = pytest.mark.parametrize("kind", ["bm25", "dense"])


class Index:
    """One index of one kind, built in a temporary folder, and how to load it."""

    def __init__(self, kind: str, root) -> None:
        self.kind = kind
        self.bm25_file = root / "bm25.pkl"
        self.chroma_dir = root / "chroma"

    def build(self, corpus) -> IndexManifest:
        if self.kind == "bm25":
            return BM25Retriever.build(processed_dir=corpus, index_path=self.bm25_file).manifest
        return embed.build(chroma_dir=self.chroma_dir, processed_dir=corpus)

    def load(self, corpus, **settings):
        if self.kind == "bm25":
            return BM25Retriever.load(self.bm25_file, processed_dir=corpus, **settings)
        return DenseRetriever.load(chroma_dir=self.chroma_dir, processed_dir=corpus, **settings)

    def fingerprint_of(self, corpus) -> str:
        """The fingerprint this kind of index would record for a corpus."""
        rows = list(iter_chunks(processed_dir=corpus))
        if self.kind == "bm25":
            return corpus_fingerprint(rows)
        return fingerprint_of(embed.digest_of(row) for row in rows)


@pytest.fixture
def index(request, tmp_path, fake_model):
    return Index(request.getfixturevalue("kind"), tmp_path / "index")


def first_file(corpus):
    return sorted(corpus.glob("*/*.json"))[0]


# --- the guard --------------------------------------------------------------


@KINDS
def test_an_index_loads_against_the_corpus_it_was_built_from(kind, index, corpus):
    manifest = index.build(corpus)

    loaded = index.load(corpus)

    assert loaded.manifest == manifest
    assert (manifest.index_type, manifest.n_passages, manifest.n_filings) == (kind, 90, 6)
    assert manifest.corpus_fingerprint == index.fingerprint_of(corpus)
    assert loaded.search(Query("Risk Factors paragraph", top_k=3, tickers=("AAA",)))


@KINDS
def test_an_index_built_against_a_different_corpus_is_refused(kind, index, corpus, tmp_path):
    """The acceptance criterion as written: one corpus built the index, another is on disk."""
    index.build(corpus)
    other = write_corpus(tmp_path / "another" / "processed")
    for path in other.glob("*/*.json"):
        edit_filing(path, lambda data: [chunk.update(
            {"text": chunk["text"].replace("paragraph", "section")}) for chunk in data["chunks"]])
    assert index.fingerprint_of(other) != index.fingerprint_of(corpus)

    with pytest.raises(ValueError) as refusal:
        index.load(other)

    message = str(refusal.value)
    assert "does not match the corpus" in message
    assert f"python -m src.retrieval {'bm25' if kind == 'bm25' else 'embed'}" in message
    # The corpus it was built from still loads it: the index is not what changed.
    assert index.load(corpus).manifest.corpus_fingerprint == index.fingerprint_of(corpus)


def _edit_text(corpus):
    edit_filing(first_file(corpus), lambda data: data["chunks"][0].update(
        {"text": data["chunks"][0]["text"] + " Restated."}))


def _drop_passage(corpus):
    edit_filing(first_file(corpus), lambda data: data["chunks"].pop())


def _add_passage(corpus):
    edit_filing(first_file(corpus), lambda data: data["chunks"].append(
        {**data["chunks"][0], "chunk_id": "added-after-the-build", "text": "A new passage."}))


def _rename_passages(corpus):
    # What a re-chunk does to every passage, even where the text comes out the same.
    for path in corpus.glob("*/*.json"):
        edit_filing(path, lambda data: [chunk.update({"chunk_id": chunk["chunk_id"] + "-recut"})
                                        for chunk in data["chunks"]])


def _drop_filing(corpus):
    first_file(corpus).unlink()


def _add_filing(corpus):
    source = first_file(corpus)
    added = corpus / "DDD" / source.name
    added.parent.mkdir()
    shutil.copy(source, added)
    edit_filing(added, lambda data: (data.update(ticker="DDD", accession_no="added"), [
        chunk.update({"chunk_id": chunk["chunk_id"].replace(data["accession_no"], "added-")})
        for chunk in data["chunks"]]))


@KINDS
@pytest.mark.parametrize("move", [
    _edit_text, _drop_passage, _add_passage, _rename_passages, _drop_filing, _add_filing,
], ids=lambda move: move.__name__.strip("_"))
def test_every_way_a_corpus_moves_refuses_the_index_built_before_it(
        kind, index, corpus, move):
    index.build(corpus)
    before = index.fingerprint_of(corpus)

    move(corpus)

    assert index.fingerprint_of(corpus) != before
    with pytest.raises(ValueError, match="does not match the corpus"):
        index.load(corpus)


@KINDS
def test_the_refusal_says_which_side_holds_what(kind, index, corpus):
    built = index.build(corpus)
    _edit_text(corpus)
    _drop_passage(corpus)

    with pytest.raises(ValueError) as refusal:
        index.load(corpus)

    message = str(refusal.value)
    if kind == "bm25":
        # Both fingerprints, so the reader can tell which of the two moved.
        assert f"index built from {built.corpus_fingerprint}" in message
        assert f"corpus on disk is {index.fingerprint_of(corpus)}" in message
        assert str(index.bm25_file) in message
    else:
        # Counted exactly, from the digest each vector carries.
        assert "0 passages missing, 1 stale, 1 no longer in the corpus" in message
        assert str(index.chroma_dir) in message


@KINDS
def test_the_same_passages_in_another_order_are_the_same_corpus(kind, index, corpus):
    """A fingerprint is of the passages, not of one walk over them."""
    index.build(corpus)
    for path in corpus.glob("*/*.json"):
        edit_filing(path, lambda data: data["chunks"].reverse())
    shutil.move(corpus / "AAA", corpus / "ZZZ-renamed-folder")

    assert index.load(corpus).manifest.n_passages == 90


@KINDS
def test_rebuilding_after_the_corpus_moved_makes_the_index_loadable_again(kind, index, corpus):
    index.build(corpus)
    _rename_passages(corpus)
    with pytest.raises(ValueError):
        index.load(corpus)

    rebuilt = index.build(corpus)

    assert rebuilt.corpus_fingerprint == index.fingerprint_of(corpus)
    assert all(passage.chunk_id.endswith("-recut")
               for passage in index.load(corpus).search(Query("Business paragraph", top_k=5)))


@KINDS
def test_the_check_can_be_skipped_by_a_caller_that_says_so(kind, index, corpus):
    """For a test holding a hand-built index. A run that reports numbers leaves it on."""
    index.build(corpus)
    _edit_text(corpus)

    assert index.load(corpus, verify=False).search(Query("Business", top_k=3))


def test_each_kind_fingerprints_what_it_indexed_and_no_more(corpus, tmp_path, fake_model):
    """The dense index encodes a header built from the filing's identity, so a renamed
    company moves its vectors. BM25 indexes the stored text, which did not change."""
    sparse, dense = Index("bm25", tmp_path / "index"), Index("dense", tmp_path / "index")
    built = {sparse.build(corpus).corpus_fingerprint, dense.build(corpus).corpus_fingerprint}
    # Two strings for one corpus, by design: never compare a manifest across kinds.
    assert len(built) == 2

    edit_filing(first_file(corpus), lambda data: data.update(company="Alpha Renamed Corp"))

    assert sparse.load(corpus).manifest.n_passages == 90
    with pytest.raises(ValueError, match="15 stale"):
        dense.load(corpus)


# --- the failure the guard is for: a corpus cut again -----------------------


def _parsed() -> ParsedFiling:
    text = "\n\n".join(
        f"Paragraph {number} of the discussion covers one matter at length. " * 6
        for number in range(12))
    section = SectionRecord(
        section_id="part_ii_item_7", part="II", item="7", title="Management's Discussion",
        text=text, n_chars=len(text), n_tables=0, n_data_tables=0, is_key_section=True,
        is_stub=False, resolved_from=None, confidence=None, detection_method=None, validated=True,
    )
    return ParsedFiling(
        ticker="AAA", cik=1, company="Alpha Corp", form="10-K", filing_date="2025-02-01",
        accession_no="0000000001-25-000001", url="https://example.test/filing",
        source_path="raw.html", sections=[section], period_of_report="2024-12-31",
    )


def _chunk(processed, budget: int, overlap: int) -> int:
    chunked = chunk_filing(_parsed(), source_path="interim.json", budget=budget, overlap=overlap)
    write_chunks(chunked, processed / "AAA" / "filing.json")
    return len(chunked.chunks)


@KINDS
def test_cutting_the_corpus_at_another_budget_refuses_the_index_until_it_is_rebuilt(
        kind, index, tmp_path):
    """``python -m src.pipeline chunk --budget 1200 --force``, and then a search."""
    processed = tmp_path / "processed"
    shipped = _chunk(processed, budget=1800, overlap=300)
    manifest = index.build(processed)
    assert (manifest.chunk_budget, manifest.chunk_overlap, manifest.n_passages) == (
        1800, 300, shipped)

    smaller = _chunk(processed, budget=1200, overlap=200)
    assert smaller > shipped

    with pytest.raises(ValueError, match="does not match the corpus"):
        index.load(processed)

    rebuilt = index.build(processed)
    # The manifest measures the settings off the corpus, so it reads what was used.
    assert (rebuilt.chunk_budget, rebuilt.chunk_overlap, rebuilt.n_passages) == (
        1200, 200, smaller)
    assert index.load(processed).manifest.corpus_fingerprint != manifest.corpus_fingerprint


# --- an index that cannot be checked ----------------------------------------


def test_a_bm25_index_with_no_manifest_is_refused_since_it_cannot_be_checked(corpus, tmp_path):
    index_file = tmp_path / "bm25.pkl"
    with index_file.open("wb") as file:
        pickle.dump({"chunks": list(iter_chunks(processed_dir=corpus)), "manifest": None}, file)

    with pytest.raises(ValueError, match="carries no manifest"):
        BM25Retriever.load(index_file, processed_dir=corpus)
    assert BM25Retriever.load(index_file, processed_dir=corpus, verify=False).manifest is None


def test_a_bm25_index_is_not_built_from_an_empty_corpus(tmp_path):
    (tmp_path / "processed").mkdir()
    with pytest.raises(ValueError, match="no processed passages were found"):
        BM25Retriever.build(processed_dir=tmp_path / "processed", index_path=tmp_path / "b.pkl")
    assert not (tmp_path / "b.pkl").exists()


def test_the_dense_manifest_is_read_back_as_the_record_that_was_written(
        corpus, tmp_path, fake_model):
    written = embed.build(chroma_dir=tmp_path / "chroma", processed_dir=corpus)
    stored = json.loads((tmp_path / "chroma.manifest.json").read_text(encoding="utf-8"))

    assert embed.read_manifest(tmp_path / "chroma.manifest.json") == written
    assert stored["corpus_fingerprint"] == written.corpus_fingerprint
    assert (stored["index_type"], stored["model"], stored["dimensions"]) == (
        "dense", embed.EMBED_MODEL, embed.EMBED_DIMENSIONS)
    assert embed.read_manifest(tmp_path / "absent.manifest.json") is None


# --- what a manifest reports ------------------------------------------------


MANIFEST = IndexManifest(
    index_type="dense", path="data/index/chroma", corpus_fingerprint="built", n_passages=90,
    n_filings=6, built_at="2026-10-07T00:00:00+00:00", model="BAAI/bge-base-en-v1.5",
    dimensions=768, chunk_budget=1800, chunk_overlap=300, passage_prefix="", max_tokens=512,
)


def test_a_manifest_that_agrees_with_the_caller_reports_nothing():
    assert MANIFEST.mismatches() == []
    assert MANIFEST.mismatches(
        corpus_fingerprint="built", model="BAAI/bge-base-en-v1.5", dimensions=768,
        passage_prefix="", max_tokens=512) == []


@pytest.mark.parametrize("expected, said", [
    ({"corpus_fingerprint": "on-disk"},
     "corpus fingerprint: index built from built, corpus on disk is on-disk"),
    ({"model": "intfloat/e5-base-v2"},
     "model: index built with BAAI/bge-base-en-v1.5, asked for intfloat/e5-base-v2"),
    ({"dimensions": 384}, "dimensions: index holds 768, asked for 384"),
    ({"passage_prefix": "passage: "},
     "passage prefix: index built with '', asked for 'passage: '"),
    ({"max_tokens": 256}, "max_tokens: index built with 512, asked for 256"),
])
def test_each_disagreement_is_one_line_naming_both_sides(expected, said):
    assert MANIFEST.mismatches(**expected) == [said]


def test_every_disagreement_is_reported_not_only_the_first():
    found = MANIFEST.mismatches(corpus_fingerprint="on-disk", model="other", dimensions=384)
    assert [line.split(":")[0] for line in found] == ["corpus fingerprint", "model", "dimensions"]


def test_an_index_with_no_model_says_so_rather_than_printing_nothing():
    sparse = replace(MANIFEST, index_type="bm25", model="", dimensions=None)
    assert sparse.mismatches(model="BAAI/bge-base-en-v1.5") == [
        "model: index built with none, asked for BAAI/bge-base-en-v1.5"]
