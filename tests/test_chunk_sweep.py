"""The chunk-size sweep (#45): each size built, asked the same questions, and recorded.

The filings are synthetic and the encoder is the suite's stand-in, so nothing
here measures retrieval. What is checked is that a run compares like with
like: every size is cut from the same filings with the settings that follow
from its budget, is searched through its own indexes, answers the same
questions, and is written down with the fingerprint of the passages it was
measured on.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.evaluation.chunk_sweep as sweep
from src.evaluation.chunk_sweep import (
    BUDGETS,
    SWEEP_FIELDS,
    SweepBuild,
    build_indexes,
    check_slicing,
    chunk_corpus,
    curve,
    describe_corpus,
    run_chunk_sweep,
    sample_questions,
    seed_vectors,
    supported_by,
)
from src.evaluation.records import BenchmarkQuestion
from src.pipeline.chunk import interim_files, iter_chunks
from src.pipeline.constants import (
    CHUNK_CHAR_BUDGET,
    CHUNK_CHAR_OVERLAP,
    overlap_for,
    table_budget_for,
)
from src.pipeline.records import ParsedFiling, SectionRecord, TableRecord
from src.retrieval import embed
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.constants import CANDIDATE_K, FINAL_K, QUERY_PREFIX
from src.retrieval.records import Query, corpus_fingerprint
from conftest import FakeModel

COMPANIES = {"AAA": "Alpha Corp", "BBB": "Beta Inc."}
YEARS = (2023, 2024)
PARAGRAPHS = 12
TABLE_ROWS = 8
# Small enough that the synthetic Item is cut differently at each, and large
# enough that the smallest still asks for fewer passages than hybrid fuses.
SIZES = (600, 1200, 2400)
FILLER = ("The company reviews the related estimates each quarter and updates them as "
          "conditions change. ")


def _accession(number: int, year: int) -> str:
    return f"000000000{number}-{year % 100}-000001"


def _prose_figure(number: int, year: int, index: int) -> int:
    return 4000 + 100 * number + 20 * (year - YEARS[0]) + index


def _table_figure(number: int, year: int, index: int) -> int:
    return 7000 + 100 * number + 20 * (year - YEARS[0]) + index


def _section(section_id: str, item: str, title: str, text: str, tables=()) -> SectionRecord:
    return SectionRecord(
        section_id=section_id, part="II", item=item, title=title, text=text,
        n_chars=len(text), n_tables=len(tables), n_data_tables=len(tables),
        is_key_section=True, is_stub=False, resolved_from=None, confidence=None,
        detection_method=None, validated=True, tables=list(tables),
    )


def _filing(ticker: str, number: int, year: int, long_passage: bool = False) -> ParsedFiling:
    paragraphs = [
        f"{ticker} discussed topic {index} for fiscal {year}. Segment {index} revenue was "
        f"{_prose_figure(number, year, index):,} million. {FILLER * 2}".strip()
        for index in range(PARAGRAPHS)
    ]
    if long_passage:
        # The stand-in tokenizer counts 600 tokens for a text holding this word.
        paragraphs[0] += " LONGLONG"
    table = TableRecord(
        table_index=0, caption="ASSETS:", headers=["(In millions)", f"{year}"],
        rows=[[f"Line item {index}", f"{_table_figure(number, year, index):,}"]
              for index in range(TABLE_ROWS)],
        n_rows=TABLE_ROWS, n_cols=2, statement_title="CONSOLIDATED BALANCE SHEETS",
    )
    return ParsedFiling(
        ticker=ticker, cik=number, company=COMPANIES[ticker], form="10-K",
        filing_date=f"{year + 1}-02-01", accession_no=_accession(number, year),
        url=f"https://example.test/{_accession(number, year)}", source_path="raw.html",
        period_of_report=f"{year}-12-31",
        sections=[
            _section("part_ii_item_7", "7", "Management's Discussion and Analysis",
                     "\n\n".join(paragraphs)),
            _section("part_ii_item_8", "8", "Financial Statements",
                     "The consolidated financial statements follow and were audited.",
                     tables=[table]),
        ],
    )


def write_interim(interim_dir: Path, long_passage: bool = False) -> Path:
    for number, ticker in enumerate(COMPANIES, start=1):
        for year in YEARS:
            parsed = _filing(ticker, number, year, long_passage and (number, year) == (1, 2023))
            path = interim_dir / ticker / f"10-K_{parsed.filing_date}_{parsed.accession_no}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(asdict(parsed)), encoding="utf-8")
    return interim_dir


def write_facts(facts_file: Path) -> Path:
    """One fact for every figure the synthetic filings print, in base units."""
    rows = []
    for number, ticker in enumerate(COMPANIES, start=1):
        for year in YEARS:
            figures = [(f"Segment {index} revenue", _prose_figure(number, year, index))
                       for index in range(PARAGRAPHS)]
            figures += [(f"Line item {index}", _table_figure(number, year, index))
                        for index in range(TABLE_ROWS)]
            for label, figure in figures:
                rows.append({
                    "ticker": ticker, "cik": number, "company": COMPANIES[ticker],
                    "accession": _accession(number, year), "form": "10-K",
                    "filing_date": f"{year + 1}-02-01", "period_of_report": f"{year}-12-31",
                    "concept": f"us-gaap:{label.replace(' ', '')}", "label": label,
                    "value": float(figure * 1_000_000), "raw_value": str(figure * 1_000_000),
                    "unit": "USD", "scale": None, "fiscal_year": year, "fiscal_period": "FY",
                    "period_start": f"{year}-01-01", "period_end": f"{year}-12-31",
                    "period_type": "duration", "statement_type": "", "is_audited": True,
                    "is_current_year": True,
                })
    facts_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(facts_file, index=False)
    return facts_file


QUESTIONS = len(COMPANIES) * len(YEARS) * (PARAGRAPHS + TABLE_ROWS)


@pytest.fixture
def sources(tmp_path):
    """Parsed filings and a facts store that holds every figure they print."""
    return {
        "interim_dir": write_interim(tmp_path / "interim"),
        "facts_file": write_facts(tmp_path / "index" / "facts.parquet"),
        "sweep_dir": tmp_path / "sweep",
        "results_root": tmp_path / "results",
    }


@pytest.fixture(scope="module")
def finished(tmp_path_factory):
    """One complete run over three sizes, shared by the tests that only read it."""
    root = tmp_path_factory.mktemp("sweep")
    model = FakeModel()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(embed, "_load_model", lambda threads=None, **_: (model, 1))
        manifest = run_chunk_sweep(
            run_id="three-sizes", budgets=SIZES, per_filing=0,
            interim_dir=write_interim(root / "interim", long_passage=True),
            facts_file=write_facts(root / "index" / "facts.parquet"),
            sweep_dir=root / "sweep", results_root=root / "results",
        )
    return {"root": root, "manifest": manifest, "run_dir": root / "results" / "three-sizes",
            "builds": [SweepBuild(size, root / "sweep" / str(size)) for size in SIZES]}


def _rows(run_dir: Path, config_id: str) -> list[dict]:
    lines = (run_dir / config_id / "questions.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


# --- what follows from a budget ---------------------------------------------


def test_the_overlap_follows_the_budget_as_the_table_budget_does():
    assert overlap_for(CHUNK_CHAR_BUDGET) == CHUNK_CHAR_OVERLAP
    assert [overlap_for(budget) for budget in BUDGETS] == [200, 300, 400, 667]
    build = SweepBuild(2400, Path("sweep") / "2400")
    assert (build.overlap, build.table_budget) == (overlap_for(2400), table_budget_for(2400))


def test_the_sizes_are_the_four_the_issue_names_and_include_the_shipped_one():
    assert BUDGETS == (1200, 1800, 2400, 4000)
    assert CHUNK_CHAR_BUDGET in BUDGETS


def test_the_second_cutoff_fills_the_prompt_the_app_sends():
    cutoffs = [SweepBuild(budget, Path("sweep")).context_k for budget in BUDGETS]
    assert cutoffs == [24, 16, 12, 7]
    # At the shipped size it is the number of passages the app puts in a prompt.
    assert SweepBuild(CHUNK_CHAR_BUDGET, Path("sweep")).context_k == FINAL_K
    # Every size is handed about the same text, which is the point of it.
    assert {round(budget * cutoff, -3) for budget, cutoff in zip(BUDGETS, cutoffs)} <= {
        28_000, 29_000}


# --- building a size --------------------------------------------------------


def test_each_size_is_cut_into_its_own_corpus_with_its_settings_recorded(sources):
    paths = interim_files(sources["interim_dir"])
    small = SweepBuild(600, sources["sweep_dir"] / "600")
    large = SweepBuild(2400, sources["sweep_dir"] / "2400")
    assert chunk_corpus(small, paths) == chunk_corpus(large, paths) == 4

    cut_small = list(iter_chunks(processed_dir=small.processed_dir))
    cut_large = list(iter_chunks(processed_dir=large.processed_dir))
    assert len(cut_small) > len(cut_large)
    assert {(row["chunk_budget"], row["chunk_overlap"]) for row in cut_small} == {(600, 100)}
    assert {(row["chunk_budget"], row["chunk_overlap"]) for row in cut_large} == {(2400, 400)}
    # The same filings, and every figure of them, whatever the size.
    assert {row["accession_no"] for row in cut_small} == {row["accession_no"] for row in cut_large}
    for figure in (f"{_prose_figure(1, 2023, 5):,}", f"{_table_figure(2, 2024, 7):,}"):
        assert any(figure in row["text"] for row in cut_small)
        assert any(figure in row["text"] for row in cut_large)
    # The table budget moved with the prose one: the statement is split at the
    # small size and whole at the large.
    tables = lambda rows: [row for row in rows if row["content_type"] == "table"]  # noqa: E731
    assert len(tables(cut_small)) > len(tables(cut_large)) == 4


def test_cutting_again_leaves_out_filings_the_run_no_longer_covers(sources):
    paths = interim_files(sources["interim_dir"])
    build = SweepBuild(1200, sources["sweep_dir"] / "1200")
    chunk_corpus(build, paths)
    assert chunk_corpus(build, paths, fiscal_years=[2024]) == 2

    rows = list(iter_chunks(processed_dir=build.processed_dir))
    assert {row["fiscal_year"] for row in rows} == {2024}
    assert {row["ticker"] for row in rows} == set(COMPANIES)


def test_a_corpus_is_described_by_what_it_holds(sources):
    build = SweepBuild(1200, sources["sweep_dir"] / "1200")
    chunk_corpus(build, interim_files(sources["interim_dir"]))
    described, kinds = describe_corpus(build)

    rows = list(iter_chunks(processed_dir=build.processed_dir))
    assert described["n_passages"] == len(rows) == len(kinds)
    assert described["n_tables"] == 4 and described["n_filings"] == 4
    assert described["n_prose"] == len(rows) - 4
    assert set(kinds.values()) == {"prose", "table"}


# --- a whole run ------------------------------------------------------------


def test_every_size_is_measured_through_every_retriever(finished):
    rows = finished["manifest"]["configurations"]
    assert [row["config"]["id"] for row in rows] == [
        f"S{size}-{retriever}" for size in SIZES for retriever in ("bm25", "dense", "hybrid")
    ]
    for row in rows:
        config, size = row["config"], row["config"]["chunk_budget"]
        assert config["metadata_filter"] is True and config["chunking"] == "section-aware"
        assert config["chunk_overlap"] == overlap_for(size)
        assert config["table_budget"] == table_budget_for(size)
        assert config["context_k"] == SweepBuild(size, Path()).context_k
        assert row["questions"] == QUESTIONS
        assert row["n_filings"] == 4
        # What the benchmark generated for that size says supports each question.
        build = SweepBuild(size, finished["root"] / "sweep" / str(size))
        generated = [json.loads(line) for line in
                     build.benchmark.read_text(encoding="utf-8").splitlines()]
        assert row["supporting_per_question"] == pytest.approx(
            sum(len(q["supporting_chunk_ids"]) for q in generated) / len(generated))
        assert 1.0 <= row["supporting_per_question"] <= 3.0


def test_the_same_questions_are_asked_of_every_size(finished):
    asked = [
        [row["question_id"] for row in _rows(finished["run_dir"], config["config"]["id"])]
        for config in finished["manifest"]["configurations"]
    ]
    assert all(ids == asked[0] for ids in asked)
    assert len(asked[0]) == len(set(asked[0])) == QUESTIONS
    assert finished["manifest"]["questions"]["generated"] == {
        str(size): QUESTIONS for size in SIZES}


def test_a_question_is_scored_against_the_passages_of_the_size_it_was_asked_of(finished):
    """Re-chunking renames every passage, so one size's supporting ids score no other."""
    retrieved = {}
    for build in finished["builds"]:
        held = {row["chunk_id"] for row in iter_chunks(processed_dir=build.processed_dir)}
        rows = _rows(finished["run_dir"], f"{build.id}-bm25")
        retrieved[build.budget] = {
            chunk_id for row in rows for chunk_id in row["retrieved_chunk_ids"]}
        assert retrieved[build.budget] <= held
        assert all(row["recall"] is not None for row in rows)
    assert retrieved[600] - retrieved[2400]


def test_a_size_too_coarse_to_miss_finds_every_answer(finished):
    """At 2,400 a filing is four passages, fewer than the cutoff, so nothing can be missed."""
    for row in finished["manifest"]["configurations"]:
        if row["config"]["chunk_budget"] == 2400:
            assert row["recall"] == row["hit_rate"] == row["context_hit_rate"] == 1.0
        assert 0.0 <= row["hit_rate"] <= 1.0 and 0.0 < row["mrr"] <= 1.0


def test_both_cutoffs_are_scored_from_one_ranking(finished):
    for config in finished["manifest"]["configurations"]:
        settings = config["config"]
        depth = max(settings["top_k"], settings["context_k"])
        for row in _rows(finished["run_dir"], settings["id"]):
            assert len(row["retrieved_chunk_ids"]) <= depth
            assert len(set(row["retrieved_chunk_ids"])) == len(row["retrieved_chunk_ids"])
            # A deeper cutoff holds everything a shallower one does.
            if settings["context_k"] >= settings["top_k"]:
                assert row["context_hit"] >= row["hit"]
            assert row["context_chars"] > 0


def test_each_row_records_the_fingerprint_of_the_build_it_was_measured_on(finished):
    recorded = {build["chunk_budget"]: build for build in finished["manifest"]["builds"]}
    for build in finished["builds"]:
        corpus = corpus_fingerprint(iter_chunks(processed_dir=build.processed_dir))
        dense = embed.read_manifest(embed.manifest_file_for(build.chroma_dir))
        sparse = BM25Retriever.load(build.bm25_index, processed_dir=build.processed_dir).manifest
        assert recorded[build.budget]["corpus_fingerprint"] == corpus == sparse.corpus_fingerprint
        assert recorded[build.budget]["dense_fingerprint"] == dense.corpus_fingerprint
        # What #68 was for: the manifest reads the budget the corpus was cut with.
        assert (dense.chunk_budget, dense.chunk_overlap) == (build.budget, build.overlap)
        assert (sparse.chunk_budget, sparse.chunk_overlap) == (build.budget, build.overlap)
    for row in finished["manifest"]["configurations"]:
        size = row["config"]["chunk_budget"]
        assert row["corpus_fingerprint"] == recorded[size]["corpus_fingerprint"]
        assert row["dense_fingerprint"] == recorded[size]["dense_fingerprint"]
    assert len({build["corpus_fingerprint"] for build in recorded.values()}) == len(SIZES)


def test_the_run_is_written_in_the_table_the_other_runners_share(finished):
    run_dir = finished["run_dir"]
    with (run_dir / "summary.csv").open(newline="", encoding="utf-8") as stream:
        table = list(csv.DictReader(stream))
    assert [row["config"] for row in table] == [
        row["config"]["id"] for row in finished["manifest"]["configurations"]]
    assert set(SWEEP_FIELDS) <= set(table[0])
    assert {"config", "name", "questions", "recall", "ndcg", "mrr"} <= set(table[0])
    assert {row["chunk_budget"] for row in table} == {str(size) for size in SIZES}
    assert all(len(row["corpus_fingerprint"]) == 64 for row in table)

    saved = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert saved["run_id"] == "three-sizes" and saved["matrix"] == "chunk-size"
    assert saved["top_k"] == 10 and saved["context_chars"] == FINAL_K * CHUNK_CHAR_BUDGET
    assert saved["questions"]["asked"] == QUESTIONS and saved["questions"]["filings"] == 4
    for config in saved["configurations"]:
        row_summary = json.loads(
            (run_dir / config["config"]["id"] / "summary.json").read_text(encoding="utf-8"))
        assert row_summary["recall"] == config["recall"]


def test_what_the_encoder_cut_short_is_read_off_each_dense_index(finished):
    for row in finished["manifest"]["configurations"]:
        if row["config"]["retriever"] == "bm25":
            # BM25 reads a passage whole, so it has nothing to report.
            assert row["truncation_rate"] is None and row["n_truncated"] is None
            continue
        assert row["max_tokens"] == 512
        assert row["n_indexed"] == row["n_passages"]
        assert row["n_truncated"] >= 1
        assert row["truncation_rate"] == row["n_truncated"] / row["n_indexed"]


def test_questions_are_grouped_by_what_supports_them_at_one_size_for_every_column(finished):
    for row in finished["manifest"]["configurations"]:
        groups = row["by_support"]
        assert set(groups) == {"tables", "prose"}
        # 8 table rows and 12 paragraphs a filing, over 4 filings, at every size.
        assert groups["tables"]["questions"] == 4 * TABLE_ROWS
        assert groups["prose"]["questions"] == 4 * PARAGRAPHS
    assert finished["manifest"]["questions"]["grouped_by_support_at"] == 1200


def test_the_curve_has_a_column_for_each_size(finished):
    text = curve(finished["manifest"]["configurations"], finished["manifest"]["top_k"])
    lines = text.splitlines()
    assert lines[0].split()[-3:] == ["600", "1,200", "2,400"]
    assert lines[1].startswith("passages")
    assert any(line.startswith("cut short by the encoder at 512 tokens") for line in lines)
    for retriever in ("bm25", "dense", "hybrid"):
        assert retriever in lines
    assert sum(line.strip().startswith("Recall@10") for line in lines) == 3
    assert any("in the same prompt (top 48/24/12)" in line for line in lines)
    assert sum("characters in that prompt, mean" in line for line in lines) == 3


# --- running again ----------------------------------------------------------


def test_a_second_run_reuses_every_build_and_needs_its_own_run_id(sources, fake_model):
    encoded: list[str] = []
    encode = fake_model.encode
    fake_model.encode = lambda texts, **kwargs: encoded.extend(texts) or encode(texts, **kwargs)
    first = run_chunk_sweep(run_id="first", budgets=SIZES[:2], per_filing=2, **sources)
    passages = sum(build["n_passages"] for build in first["builds"])
    # Each filing's one line of Item 8 prose is a passage at both sizes, so the
    # second size copied those four from the first and encoded the rest.
    assert [build["seeded_vectors"] for build in first["builds"]] == [0, 4]
    assert sum(not text.startswith(QUERY_PREFIX) for text in encoded) == passages - 4

    with pytest.raises(FileExistsError, match="already holds a run"):
        run_chunk_sweep(run_id="first", budgets=SIZES[:2], per_filing=2, **sources)
    encoded.clear()
    again = run_chunk_sweep(run_id="second", budgets=SIZES[:2], per_filing=2, **sources)

    # The second time only questions were encoded, and no passage.
    assert encoded and all(text.startswith(QUERY_PREFIX) for text in encoded)
    assert [row["recall"] for row in again["configurations"]] == [
        row["recall"] for row in first["configurations"]]
    assert ([build["dense_fingerprint"] for build in again["builds"]]
            == [build["dense_fingerprint"] for build in first["builds"]])
    assert all(build["seeded_vectors"] == 0 for build in again["builds"])


def test_the_same_number_of_questions_is_drawn_from_each_filing(sources, fake_model):
    manifest = run_chunk_sweep(run_id="drawn", budgets=SIZES[:2], per_filing=3, **sources)
    assert manifest["questions"]["asked"] == 4 * 3
    rows = _rows(sources["results_root"] / "drawn", "S600-hybrid")
    drawn: dict[str, int] = {}
    for row in rows:
        filing = row["question_id"].rsplit("-usd-", 1)[1]
        drawn[filing] = drawn.get(filing, 0) + 1
    assert sorted(drawn.values()) == [3, 3, 3, 3]


def test_narrowing_the_corpus_narrows_the_questions(sources, fake_model):
    manifest = run_chunk_sweep(
        run_id="narrow", budgets=SIZES[:2], per_filing=0, tickers=["aaa"],
        fiscal_years=[2024], **sources,
    )
    assert manifest["corpus"] == {"tickers": ["AAA"], "fiscal_years": [2024]}
    assert manifest["questions"]["filings"] == 1
    assert manifest["questions"]["asked"] == PARAGRAPHS + TABLE_ROWS
    assert all(build["n_filings"] == 1 for build in manifest["builds"])


def test_bm25_alone_encodes_nothing(sources, fake_model):
    manifest = run_chunk_sweep(
        run_id="keywords", budgets=SIZES, retrievers=["bm25"], per_filing=0, **sources)

    assert fake_model.calls == 0
    assert [row["config"]["id"] for row in manifest["configurations"]] == [
        f"S{size}-bm25" for size in SIZES]
    assert manifest["embedding_model"] is None
    for size in SIZES:
        assert not (sources["sweep_dir"] / str(size) / "chroma").exists()
    assert all(build["dense_fingerprint"] is None for build in manifest["builds"])
    assert "cut short by the encoder" not in curve(manifest["configurations"], 10)


# --- vectors another index already holds ------------------------------------


def _vectors(chroma_dir: Path) -> dict[str, np.ndarray]:
    found = embed.open_collection(chroma_dir, create=False).get(include=["embeddings"])
    return {chunk_id: np.asarray(vector) for chunk_id, vector in
            zip(found["ids"], found["embeddings"])}


def _two_builds(sources) -> tuple[SweepBuild, SweepBuild]:
    paths = interim_files(sources["interim_dir"])
    donor = SweepBuild(1200, sources["sweep_dir"] / "1200")
    target = SweepBuild(2400, sources["sweep_dir"] / "2400")
    chunk_corpus(donor, paths)
    chunk_corpus(target, paths)
    return donor, target


def _shared(donor: SweepBuild, target: SweepBuild) -> set[str]:
    """The passages two builds hold under one id with one text."""
    held = {row["chunk_id"]: embed.digest_of(row)
            for row in iter_chunks(processed_dir=donor.processed_dir)}
    return {row["chunk_id"] for row in iter_chunks(processed_dir=target.processed_dir)
            if held.get(row["chunk_id"]) == embed.digest_of(row)}


def test_a_vector_another_size_holds_for_the_same_text_is_copied_not_encoded(
        sources, fake_model):
    donor, target = _two_builds(sources)
    embed.build(chroma_dir=donor.chroma_dir, processed_dir=donor.processed_dir)
    shared = _shared(donor, target)
    # The statement and the Item's one line of prose fit in a passage at both sizes.
    assert len(shared) == 8

    encoded = []
    encode = fake_model.encode
    fake_model.encode = lambda texts, **kwargs: encoded.extend(texts) or encode(texts, **kwargs)
    built = build_indexes(target, dense=True, donors=[donor.chroma_dir])

    assert built["seeded_vectors"] == len(shared)
    corpus = list(iter_chunks(processed_dir=target.processed_dir))
    assert len(encoded) == len(corpus) - len(shared)
    assert embed.check_index(target.chroma_dir, target.processed_dir) == []
    # Chroma normalises a vector as it stores it, so a copy is the original to
    # the last place a 32-bit float holds, not always bit for bit.
    copied, original = _vectors(target.chroma_dir), _vectors(donor.chroma_dir)
    for chunk_id in shared:
        assert np.allclose(copied[chunk_id], original[chunk_id], rtol=0, atol=1e-6)


def test_a_copied_vector_is_stored_as_an_encoded_one_would_be(sources, fake_model):
    donor, target = _two_builds(sources)
    embed.build(chroma_dir=donor.chroma_dir, processed_dir=donor.processed_dir)
    seeded = SweepBuild(2400, sources["sweep_dir"] / "seeded")
    chunk_corpus(seeded, interim_files(sources["interim_dir"]))
    build_indexes(seeded, dense=True, donors=[donor.chroma_dir])
    build_indexes(target, dense=True)

    def stored(build):
        found = embed.open_collection(build.chroma_dir, create=False).get(
            include=["documents", "metadatas"])
        return dict(zip(found["ids"], zip(found["documents"], found["metadatas"])))

    # The text, the citation fields, the digest, the token count and the model.
    assert stored(seeded) == stored(target)
    copied, encoded = _vectors(seeded.chroma_dir), _vectors(target.chroma_dir)
    assert copied.keys() == encoded.keys()
    for chunk_id, vector in encoded.items():
        assert np.allclose(copied[chunk_id], vector, rtol=0, atol=1e-6)
    assert (embed.read_manifest(embed.manifest_file_for(seeded.chroma_dir)).corpus_fingerprint
            == embed.read_manifest(embed.manifest_file_for(target.chroma_dir)).corpus_fingerprint)


def test_nothing_is_copied_from_an_index_that_cannot_vouch_for_its_vectors(sources, fake_model):
    donor, target = _two_builds(sources)
    embed.build(chroma_dir=donor.chroma_dir, processed_dir=donor.processed_dir)
    manifest_file = embed.manifest_file_for(donor.chroma_dir)
    written = json.loads(manifest_file.read_text(encoding="utf-8"))

    # Built under a shorter token limit: the same text, read less far.
    manifest_file.write_text(json.dumps({**written, "max_tokens": 256}), encoding="utf-8")
    assert seed_vectors(target.chroma_dir, target.processed_dir, [donor.chroma_dir]) == 0
    # Encoded by another model.
    manifest_file.write_text(json.dumps({**written, "model": "other/model"}), encoding="utf-8")
    assert seed_vectors(target.chroma_dir, target.processed_dir, [donor.chroma_dir]) == 0
    # A build that never finished leaves no manifest.
    manifest_file.unlink()
    assert seed_vectors(target.chroma_dir, target.processed_dir, [donor.chroma_dir]) == 0
    # No index there at all, and an index is no donor to itself.
    assert seed_vectors(target.chroma_dir, target.processed_dir,
                        [sources["sweep_dir"] / "absent", target.chroma_dir]) == 0
    assert embed.open_collection(target.chroma_dir, create=False).count() == 0


def test_a_vector_whose_text_has_changed_is_not_copied(sources, fake_model):
    donor, target = _two_builds(sources)
    embed.build(chroma_dir=donor.chroma_dir, processed_dir=donor.processed_dir)
    shared = _shared(donor, target)
    path = next(target.processed_dir.glob("AAA/*.json"))
    data = json.loads(path.read_text(encoding="utf-8"))
    changed = next(chunk for chunk in data["chunks"] if chunk["chunk_id"] in shared)
    changed["text"] += " Restated."
    changed["n_chars"] = len(changed["text"])
    path.write_text(json.dumps(data), encoding="utf-8")

    assert seed_vectors(target.chroma_dir, target.processed_dir,
                        [donor.chroma_dir]) == len(shared) - 1
    held = embed.open_collection(target.chroma_dir, create=False).get(include=[])["ids"]
    assert changed["chunk_id"] not in held
    # Copying leaves the index between two states, so it has no manifest yet.
    assert not embed.manifest_file_for(target.chroma_dir).exists()


# --- the questions ----------------------------------------------------------


def _question(question_id: str, ticker: str, year: int, supporting=("c1",)) -> BenchmarkQuestion:
    return BenchmarkQuestion(
        question_id=question_id, question="What was revenue?", expected_answer="1 USD",
        supporting_chunk_ids=tuple(supporting), hard_negative_chunk_ids=(), ticker=ticker,
        fiscal_year=year, question_type="numeric", difficulty="mechanical", source="xbrl",
    )


def test_only_a_question_every_size_can_be_asked_is_kept():
    one = [_question(f"q{n}", "AAA", 2024) for n in range(6)]
    other = [_question(f"q{n}", "AAA", 2024) for n in range(2, 9)]
    assert sample_questions({600: one, 1200: other}, per_filing=0) == ["q2", "q3", "q4", "q5"]


def test_the_draw_is_the_same_every_run_and_takes_all_of_a_small_filing():
    generated = {600: [_question(f"a{n:02d}", "AAA", 2024) for n in range(30)]
                 + [_question(f"b{n}", "BBB", 2024) for n in range(3)]}
    drawn = sample_questions(generated, per_filing=5, seed=1)
    assert drawn == sample_questions(generated, per_filing=5, seed=1)
    assert drawn != sample_questions(generated, per_filing=5, seed=2)
    assert [question_id for question_id in drawn if question_id.startswith("b")] == [
        "b0", "b1", "b2"]
    assert len(drawn) == 5 + 3


def test_a_question_is_grouped_by_the_kind_of_passage_that_supports_it():
    kinds = {"t1": "table", "t2": "table", "p1": "prose"}
    assert supported_by(_question("q", "AAA", 2024, ["t1", "t2"]), kinds) == "tables"
    assert supported_by(_question("q", "AAA", 2024, ["p1"]), kinds) == "prose"
    assert supported_by(_question("q", "AAA", 2024, ["t1", "p1"]), kinds) == "both"


class _Ranked:
    """A retriever whose order may depend on how many passages it is asked for."""

    name = "bm25"

    def __init__(self, stable: bool):
        self.stable = stable

    def search(self, query, k=None):
        wanted = query.top_k if k is None else k
        order = list(range(20)) if self.stable or wanted > 10 else list(reversed(range(20)))
        return [type("Passage", (), {"chunk_id": f"c{number}"})() for number in order[:wanted]]


def test_slicing_a_deep_search_is_refused_where_the_order_depends_on_the_cutoff():
    queries = [Query("What was revenue?", top_k=16)]
    check_slicing({"bm25": _Ranked(stable=True)}, queries, top_k=10)
    with pytest.raises(RuntimeError, match="cannot score both cutoffs"):
        check_slicing({"bm25": _Ranked(stable=False)}, queries, top_k=10)


# --- what a run refuses before it starts ------------------------------------


@pytest.mark.parametrize("settings, message", [
    ({"budgets": (1200, 1200)}, "distinct positive"),
    ({"budgets": ()}, "distinct positive"),
    ({"retrievers": ("bm25", "rerank")}, "retrievers must be among"),
    ({"top_k": 0}, "top_k must be positive"),
    ({"per_filing": -1}, "per_filing zero or more"),
    ({"run_id": "../elsewhere"}, "run_id must contain only"),
])
def test_a_bad_setting_is_refused_before_anything_is_built(sources, settings, message):
    with pytest.raises(ValueError, match=message):
        run_chunk_sweep(**{"run_id": "refused", "budgets": SIZES, **sources, **settings})
    assert not sources["sweep_dir"].exists() and not sources["results_root"].exists()


def test_a_size_that_needs_more_passages_than_hybrid_fuses_is_refused(sources):
    """500 characters would take 58 passages to fill the prompt, and hybrid fuses 50."""
    assert SweepBuild(500, Path()).context_k > CANDIDATE_K
    with pytest.raises(ValueError, match="CANDIDATE_K"):
        run_chunk_sweep(run_id="refused", budgets=(500, 1800), **sources)
    with pytest.raises(ValueError, match="CANDIDATE_K"):
        run_chunk_sweep(run_id="refused", budgets=(1800,), top_k=CANDIDATE_K + 1, **sources)
    assert not sources["sweep_dir"].exists()


def test_a_run_without_its_inputs_says_which_command_makes_them(sources):
    missing = {**sources, "facts_file": sources["facts_file"].parent / "absent.parquet"}
    with pytest.raises(FileNotFoundError, match="python -m src.retrieval facts"):
        run_chunk_sweep(run_id="refused", budgets=SIZES, **missing)
    empty = {**sources, "interim_dir": sources["interim_dir"].parent / "absent"}
    with pytest.raises(FileNotFoundError, match="python -m src.pipeline parse"):
        run_chunk_sweep(run_id="refused", budgets=SIZES, **empty)
    with pytest.raises(FileNotFoundError, match="is in scope"):
        run_chunk_sweep(run_id="refused", budgets=SIZES, tickers=["ZZZ"], **sources)
    assert not sources["sweep_dir"].exists() and not sources["results_root"].exists()


def test_a_fiscal_year_no_filing_reports_on_uses_no_run_id(sources):
    with pytest.raises(FileNotFoundError, match="fiscal years asked for"):
        run_chunk_sweep(run_id="refused", budgets=SIZES, fiscal_years=[1999], **sources)
    assert not sources["results_root"].exists()


# --- the command ------------------------------------------------------------


@pytest.fixture
def command(sources, fake_model, monkeypatch, tmp_path):
    """The command's arguments for the synthetic corpus, with no log file or donor index."""
    monkeypatch.setattr(sweep, "start_run_log", lambda name: tmp_path / f"{name}.log")
    monkeypatch.setattr(sweep, "CHROMA_DIR", tmp_path / "no-app-index")
    return [
        "--interim-dir", str(sources["interim_dir"]), "--facts-file", str(sources["facts_file"]),
        "--sweep-dir", str(sources["sweep_dir"]), "--results-root", str(sources["results_root"]),
    ]


def test_the_command_prints_the_curve_and_where_it_wrote(command, sources, capsys):
    sweep.main(["--run-id", "cli", "--budgets", "600", "1200", "--per-filing", "2", *command])

    printed = capsys.readouterr().out
    assert "S600: 600 characters a passage, 100 carried over, 240 a table passage" in printed
    assert "chunk budget, characters" in printed and "hybrid" in printed
    assert f"written: {sources['results_root'] / 'cli'}" in printed
    assert "Run log:" in printed
    assert (sources["results_root"] / "cli" / "summary.csv").is_file()


def test_the_command_passes_its_narrowing_and_its_choice_of_retrievers(
        command, sources, monkeypatch):
    seen = {}
    monkeypatch.setattr(sweep, "run_chunk_sweep", lambda **kwargs: seen.update(kwargs) or {
        "configurations": [], "top_k": kwargs["top_k"]})
    monkeypatch.setattr(sweep, "curve", lambda summaries, top_k: "")

    sweep.main(["--run-id", "cli", "--retrievers", "bm25", "--tickers", "AAA", "--fiscal-years",
                "2024", "--top-k", "5", "--threads", "2", *command])
    assert (seen["retrievers"], seen["tickers"]) == (["bm25"], ["AAA"])
    assert (seen["fiscal_years"], seen["top_k"], seen["threads"]) == ([2024], 5, 2)
    assert seen["budgets"] == list(BUDGETS)
    assert seen["donors"] == (sweep.CHROMA_DIR,)

    sweep.main(["--run-id", "cli", "--no-reuse", *command])
    assert seen["donors"] == ()


@pytest.mark.parametrize("arguments", [
    ["--run-id", "bad id"],
    ["--run-id", "ok", "--budgets", "0"],
    ["--run-id", "ok", "--per-filing", "-1"],
    ["--run-id", "ok", "--retrievers", "rerank"],
    [],
])
def test_the_command_refuses_a_bad_argument_as_a_usage_error(command, arguments, sources):
    with pytest.raises(SystemExit) as stopped:
        sweep.main([*arguments, *command])
    assert stopped.value.code == 2
    assert not sources["sweep_dir"].exists()


def test_the_command_refuses_a_run_id_already_used(command, sources, capsys):
    (sources["results_root"] / "taken").mkdir(parents=True)
    with pytest.raises(SystemExit) as stopped:
        sweep.main(["--run-id", "taken", *command])
    assert stopped.value.code == 2
    assert "already holds a run" in capsys.readouterr().err


def test_the_command_ends_a_run_it_cannot_start_with_what_to_do(command, sources, capsys):
    sources["facts_file"].unlink()
    with pytest.raises(SystemExit) as stopped:
        sweep.main(["--run-id", "cli", *command])
    assert "python -m src.retrieval facts" in str(stopped.value.code)
    assert "Run log:" in capsys.readouterr().out
