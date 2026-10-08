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
import importlib.util
import json
import re
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

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
    compared_with,
    curve,
    describe_corpus,
    filing_of,
    filings_in_scope,
    measure_build,
    run_chunk_sweep,
    sample_questions,
    seed_vectors,
    supported_by,
)
from src.config import PROJECT_ROOT
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
from src.stack import measured_runs
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


@pytest.mark.parametrize("budget, cutoff", [
    (1850, 15),     # 15.6 passages: a 16th would not fit, and rounding gave 16
    (1900, 15),
    (3700, 7),      # 7.8
    (19_000, 1),    # 1.5
    (28_800, 1),
    (60_000, 1),    # under one passage, and a prompt still holds one
])
def test_the_second_cutoff_never_allows_more_text_than_the_prompt_holds(budget, cutoff):
    build = SweepBuild(budget, Path("sweep"))
    assert build.context_k == cutoff
    assert cutoff == 1 or budget * cutoff <= sweep.CONTEXT_CHARS < budget * (cutoff + 1)


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


def test_the_filings_in_scope_are_found_before_anything_is_cut(sources):
    interim = sources["interim_dir"]
    assert filings_in_scope(interim) == interim_files(interim)
    assert {path.parent.name for path in filings_in_scope(interim, tickers=["bbb"])} == {"BBB"}
    # A filing dated February 2025 reports on fiscal 2024.
    narrowed = filings_in_scope(interim, tickers=["AAA", "BBB"], fiscal_years=[2024])
    assert len(narrowed) == 2 and all("_2025-02-01_" in path.name for path in narrowed)
    assert len(filings_in_scope(interim, fiscal_years=[2023, 2024])) == 4

    with pytest.raises(FileNotFoundError, match="is in scope"):
        filings_in_scope(interim, tickers=["ZZZ"])
    with pytest.raises(FileNotFoundError, match="fiscal years asked for"):
        filings_in_scope(interim, fiscal_years=[1999])
    with pytest.raises(FileNotFoundError, match="python -m src.pipeline parse"):
        filings_in_scope(interim.parent / "absent")


def test_cutting_again_leaves_out_filings_the_run_no_longer_covers(sources):
    paths = interim_files(sources["interim_dir"])
    build = SweepBuild(1200, sources["sweep_dir"] / "1200")
    assert chunk_corpus(build, paths) == 4
    assert chunk_corpus(
        build, filings_in_scope(sources["interim_dir"], fiscal_years=[2024])) == 2

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
            assert row["context_chars"] > 0 and row["top_chars"] > 0
            # And holds at least as much text.
            if settings["context_k"] >= settings["top_k"]:
                assert row["context_chars"] >= row["top_chars"]


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


def test_a_sweep_row_is_not_offered_as_a_configuration_the_app_can_build(finished):
    """The app labels a configuration with the run that measured it, read from
    the same ``results/``. A sweep row was measured on a build of its own under
    ``data/sweep/``, which the app does not search, so it must not be offered."""
    results_root = finished["run_dir"].parent
    saved = json.loads((finished["run_dir"] / "summary.json").read_text(encoding="utf-8"))
    assert len(saved["configurations"]) == len(SIZES) * 3
    assert measured_runs(results_root) == []


def test_what_the_encoder_cut_short_is_read_off_each_dense_index(finished):
    # The stand-in tokenizer counts 600 tokens, over the limit of 512, for a
    # passage that holds this word, and 10 for any other.
    long = {
        build.budget: sum(
            "LONGLONG" in row["text"] for row in iter_chunks(processed_dir=build.processed_dir))
        for build in finished["builds"]
    }
    assert all(count >= 1 for count in long.values())
    for row in finished["manifest"]["configurations"]:
        if row["config"]["retriever"] == "bm25":
            # BM25 reads a passage whole, so it has nothing to report.
            assert row["truncation_rate"] is None and row["n_truncated"] is None
            continue
        assert row["max_tokens"] == 512
        assert row["n_indexed"] == row["n_passages"]
        assert row["n_truncated"] == long[row["config"]["chunk_budget"]] < row["n_indexed"]
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
    assert sum("characters in the top 10, mean" in line for line in lines) == 3
    # One line a retriever at each cutoff, with nothing under the reference size itself.
    beside = [line for line in lines if "found only here / only at reference" in line]
    assert len(beside) == 6
    # Two sizes of the three have a count each way, as "found here/found there".
    assert all(len(re.findall("[0-9]+/[0-9]+", line)) == 2 for line in beside)


def _curve_row(budget: int, retriever: str, n: int, pair: tuple[int, ...] | None = None) -> dict:
    """A row whose every measure is a number no other measure or row has."""
    return {
        "config": {"id": f"S{budget}-{retriever}", "retriever": retriever,
                   "chunk_budget": budget, "context_k": {1800: 16, 4000: 7}[budget]},
        "n_passages": 1000 * n, "median_chars": 100.0 * n,
        # What the encoder cut short is a dense index's to say.
        "max_tokens": 512 if retriever == "dense" else None,
        "truncation_rate": n / 100 if retriever == "dense" else None,
        "recall": round(0.1 + n / 1000, 3), "ndcg": round(0.2 + n / 1000, 3),
        "mrr": round(0.3 + n / 1000, 3), "hit_rate": round(0.4 + n / 1000, 3),
        "top_chars": 1000.0 * n + 5, "context_hit_rate": round(0.6 + n / 1000, 3),
        "context_chars": 1000.0 * n + 7, "median_latency_ms": 10.0 * n + 8,
        "against_reference": pair and {
            "reference": f"S1800-{retriever}",
            "top_k": {"both": 0, "only_this": pair[0], "only_reference": pair[1], "neither": 0},
            "context_k": {"both": 0, "only_this": pair[2], "only_reference": pair[3], "neither": 0},
        },
    }


def test_the_curve_prints_each_measure_from_the_field_that_holds_it():
    """A line read from the wrong field, the wrong row or the wrong way round
    prints a number that is not the one expected here. In the shared run the
    two cutoffs find the same questions, so their lines print the same."""
    printed = curve([
        _curve_row(1800, "bm25", 1), _curve_row(4000, "bm25", 2, pair=(21, 22, 23, 24)),
        _curve_row(1800, "dense", 3), _curve_row(4000, "dense", 4, pair=(31, 32, 33, 34)),
    ], 10)

    # A label fills the first 52 columns, and an empty cell leaves no word.
    assert [(line[:52].rstrip(), line[52:].split()) for line in printed.splitlines()] == [
        ("chunk budget, characters", ["1,800", "4,000"]),
        ("passages", ["1,000", "2,000"]),
        ("median passage, characters", ["100", "200"]),
        ("cut short by the encoder at 512 tokens", ["3.0%", "4.0%"]),
        ("bm25", []),
        ("  Recall@10", ["0.101", "0.102"]),
        ("  nDCG@10", ["0.201", "0.202"]),
        ("  MRR@10", ["0.301", "0.302"]),
        ("  supporting passage in the top 10", ["40.1%", "40.2%"]),
        ("  characters in the top 10, mean", ["1,005", "2,005"]),
        ("  in the same prompt (top 16/7)", ["60.1%", "60.2%"]),
        ("  characters in that prompt, mean", ["1,007", "2,007"]),
        ("  median search, ms", ["18", "28"]),
        ("  found only here / only at reference, top 10", ["21/22"]),
        ("  found only here / only at reference, same prompt", ["23/24"]),
        ("dense", []),
        ("  Recall@10", ["0.103", "0.104"]),
        ("  nDCG@10", ["0.203", "0.204"]),
        ("  MRR@10", ["0.303", "0.304"]),
        ("  supporting passage in the top 10", ["40.3%", "40.4%"]),
        ("  characters in the top 10, mean", ["3,005", "4,005"]),
        ("  in the same prompt (top 16/7)", ["60.3%", "60.4%"]),
        ("  characters in that prompt, mean", ["3,007", "4,007"]),
        ("  median search, ms", ["38", "48"]),
        ("  found only here / only at reference, top 10", ["31/32"]),
        ("  found only here / only at reference, same prompt", ["33/34"]),
    ]


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


def test_a_build_already_indexed_takes_the_copy_in_place_of_its_stale_vector(
        sources, fake_model):
    """A build indexed before one of its passages changed.

    The donor holds the vector for what the passage says now. That vector
    replaces the stale one, stored with this corpus's own text and metadata,
    and the manifest is gone while the index is between the two.
    """
    donor, target = _two_builds(sources)
    embed.build(chroma_dir=donor.chroma_dir, processed_dir=donor.processed_dir)
    shared = _shared(donor, target)
    path = next(target.processed_dir.glob("AAA/*.json"))
    data = json.loads(path.read_text(encoding="utf-8"))
    passage = next(chunk for chunk in data["chunks"] if chunk["chunk_id"] in shared)
    chunk_id, current = passage["chunk_id"], passage["text"]

    # The target is indexed while that passage still reads differently.
    passage["text"] = current + " As first reported."
    passage["n_chars"] = len(passage["text"])
    path.write_text(json.dumps(data), encoding="utf-8")
    embed.build(chroma_dir=target.chroma_dir, processed_dir=target.processed_dir)
    stale = _vectors(target.chroma_dir)[chunk_id]
    assert embed.manifest_file_for(target.chroma_dir).exists()
    # Then it is cut again, and reads as the donor indexed it.
    passage["text"], passage["n_chars"] = current, len(current)
    path.write_text(json.dumps(data), encoding="utf-8")
    # What the donor stored beside the vector is not what this corpus says.
    theirs = embed.open_collection(donor.chroma_dir, create=False)
    stored = theirs.get(ids=[chunk_id], include=["metadatas"])["metadatas"][0]
    theirs.update(ids=[chunk_id], metadatas=[{**stored, "ticker": "ZZZ"}])

    assert seed_vectors(target.chroma_dir, target.processed_dir, [donor.chroma_dir]) == 1
    # Copying leaves the index between two states, so its manifest goes first.
    assert not embed.manifest_file_for(target.chroma_dir).exists()
    found = embed.open_collection(target.chroma_dir, create=False).get(
        ids=[chunk_id], include=["embeddings", "documents", "metadatas"])
    original = _vectors(donor.chroma_dir)[chunk_id]
    assert np.allclose(found["embeddings"][0], original, rtol=0, atol=1e-6)
    assert not np.allclose(found["embeddings"][0], stale, rtol=0, atol=1e-6)
    assert found["documents"] == [current]
    assert found["metadatas"][0]["ticker"] == "AAA"
    assert found["metadatas"][0][embed.DIGEST_FIELD] == stored[embed.DIGEST_FIELD]

    # Nothing is left for the encoder, and the index is whole again.
    calls = fake_model.calls
    embed.build(chroma_dir=target.chroma_dir, processed_dir=target.processed_dir)
    assert fake_model.calls == calls
    assert embed.check_index(target.chroma_dir, target.processed_dir) == []


def _indexed_another_way(sources, fake_model, **how) -> tuple[SweepBuild, SweepBuild, str]:
    """A donor a sweep built, and a target indexed some other way that has since
    lost one vector the donor holds."""
    donor, target = _two_builds(sources)
    embed.build(chroma_dir=donor.chroma_dir, processed_dir=donor.processed_dir)
    embed.build(chroma_dir=target.chroma_dir, processed_dir=target.processed_dir, **how)
    missing = sorted(_shared(donor, target))[0]
    embed.open_collection(target.chroma_dir, create=False).delete(ids=[missing])
    return donor, target, missing


def _held(build: SweepBuild) -> tuple[str, dict]:
    """A build's dense index as it stands: its manifest, and every vector with what
    is stored beside it."""
    found = embed.open_collection(build.chroma_dir, create=False).get(
        include=["embeddings", "documents", "metadatas"])
    rows = {chunk_id: (np.asarray(vector).tobytes(), document, metadata)
            for chunk_id, vector, document, metadata in zip(
                found["ids"], found["embeddings"], found["documents"], found["metadatas"])}
    return embed.manifest_file_for(build.chroma_dir).read_text(encoding="utf-8"), rows


@pytest.mark.parametrize("how, said", [
    # Read 128 tokens of each passage, where a sweep build and its donors read 512.
    ({"max_tokens": 128}, "token limit"),
    # Encoded by another model altogether.
    ({"model_name": "other/model"}, "not encoded by"),
])
def test_a_build_indexed_another_way_is_refused_before_a_vector_is_copied_into_it(
        sources, fake_model, how, said):
    """Seeding removes the manifest before it writes, and the manifest is where
    ``embed.build`` reads the token limit an index was encoded under. Copying first,
    a 128-token index took a 512-token vector and was then recorded as all 512."""
    donor, target, missing = _indexed_another_way(sources, fake_model, **how)
    before = _held(target)
    assert missing not in before[1]

    with pytest.raises(embed.MixedIndexError, match=said):
        seed_vectors(target.chroma_dir, target.processed_dir, [donor.chroma_dir])
    assert _held(target) == before

    # Through the sweep's own step, which says what rebuilds this index.
    calls = fake_model.calls
    with pytest.raises(RuntimeError, match=said) as refused:
        build_indexes(target, dense=True, donors=[donor.chroma_dir])
    assert "Delete that folder and run the same command again" in str(refused.value)
    assert str(target.chroma_dir) in str(refused.value)
    assert _held(target) == before and fake_model.calls == calls
    # And without a donor, where it is embed.build that refuses.
    with pytest.raises(RuntimeError, match="Delete that folder"):
        build_indexes(target, dense=True)
    assert _held(target) == before and fake_model.calls == calls


def test_a_refusal_to_mix_vectors_is_still_the_error_callers_catch():
    """``embed.build`` raised RuntimeError for both, and its callers catch that."""
    import copy
    import pickle

    refusal = embed.MixedIndexError("changing the encoder token limit requires rebuild=True")
    assert isinstance(refusal, RuntimeError)
    assert str(pickle.loads(pickle.dumps(refusal))) == str(copy.copy(refusal)) == str(refusal)


def test_a_build_left_unfinished_or_indexed_as_a_sweep_does_still_takes_copies(
        sources, fake_model):
    donor, target, missing = _indexed_another_way(sources, fake_model, max_tokens=512)
    # Finished, under the limit a sweep uses: the one missing vector is copied.
    assert seed_vectors(target.chroma_dir, target.processed_dir, [donor.chroma_dir]) == 1
    assert missing in _vectors(target.chroma_dir)
    # Interrupted: vectors and no manifest, which is how seeding itself leaves an
    # index until embed.build finishes it. Nothing recorded says they were encoded
    # another way, so the build goes on from where it stopped.
    assert not embed.manifest_file_for(target.chroma_dir).exists()
    embed.open_collection(target.chroma_dir, create=False).delete(ids=[missing])
    built = build_indexes(target, dense=True, donors=[donor.chroma_dir])
    assert built["seeded_vectors"] == 1
    assert embed.check_index(target.chroma_dir, target.processed_dir) == []
    assert embed.read_manifest(embed.manifest_file_for(target.chroma_dir)).max_tokens == 512


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


def test_a_filing_is_an_accession_so_an_amended_filing_is_drawn_from_on_its_own():
    """A company that files a 10-K and then a 10-K/A for one year has two filings.
    By company and year alone they were one, and shared one draw of per_filing."""
    original = [_question(f"o{n}", "AAA", 2024, supporting=(f"0001-25-000001_item_8_{n:03d}",))
                for n in range(6)]
    amended = [_question(f"a{n}", "AAA", 2024, supporting=(f"0001-25-000009_item_8_{n:03d}",))
               for n in range(6)]
    assert filing_of(original[0]) == ("AAA", 2024, "0001-25-000001")
    assert filing_of(amended[0]) == ("AAA", 2024, "0001-25-000009")

    drawn = sample_questions({600: original + amended}, per_filing=2, seed=1)

    assert len(drawn) == 4
    assert sum(question_id.startswith("o") for question_id in drawn) == 2
    assert sum(question_id.startswith("a") for question_id in drawn) == 2


def test_filings_are_drawn_from_in_the_order_of_company_and_year_as_before():
    """The accession only splits a company's year. It comes last in the key, so a
    corpus with one filing for each is drawn from in the order it always was, and
    a run over it asks the questions the recorded runs asked."""
    generated = {600: [
        _question(f"{ticker}{year}-{n}", ticker, year,
                  supporting=(f"{accession}_item_8_{n:03d}",))
        # Accessions that sort the other way round from company and year.
        for ticker, year, accession in [("AAA", 2023, "9999-23-1"), ("AAA", 2024, "5555-24-1"),
                                        ("BBB", 2023, "0001-23-1")]
        for n in range(8)
    ]}
    drawn = sample_questions(generated, per_filing=3, seed=4103)
    assert [question_id.split("-")[0] for question_id in drawn] == (
        ["AAA2023"] * 3 + ["AAA2024"] * 3 + ["BBB2023"] * 3)

    import random
    from collections import defaultdict

    by_company_and_year = defaultdict(list)
    for question in generated[600]:
        by_company_and_year[(question.ticker, question.fiscal_year)].append(question.question_id)
    rng = random.Random(4103)
    expected = []
    for key in sorted(by_company_and_year):
        expected += sorted(rng.sample(sorted(by_company_and_year[key]), 3))
    assert drawn == expected


def test_a_run_counts_its_filings_by_accession(finished):
    assert finished["manifest"]["questions"]["filings"] == len(COMPANIES) * len(YEARS)


def test_each_size_is_set_beside_the_reference_size_question_by_question(finished):
    """1,200 stands in for the shipped size here, as the nearest to it of the three."""
    assert finished["manifest"]["reference_budget"] == 1200
    rows = {row["config"]["id"]: row for row in finished["manifest"]["configurations"]}
    for config_id, row in rows.items():
        retriever = row["config"]["retriever"]
        pair = row["against_reference"]
        if row["config"]["chunk_budget"] == 1200:
            assert pair is None
            continue
        reference = rows[f"S1200-{retriever}"]
        assert pair["reference"] == reference["config"]["id"]
        for cutoff, rate in (("top_k", "hit_rate"), ("context_k", "context_hit_rate")):
            counts = pair[cutoff]
            assert sum(counts.values()) == QUESTIONS
            # The two rates, taken apart into the questions they are made of.
            assert counts["both"] + counts["only_this"] == round(row[rate] * QUESTIONS)
            assert counts["both"] + counts["only_reference"] == round(reference[rate] * QUESTIONS)
    saved = json.loads((finished["run_dir"] / "summary.json").read_text(encoding="utf-8"))
    assert [row["against_reference"] for row in saved["configurations"]] == [
        row["against_reference"] for row in rows.values()]


def test_a_row_is_set_beside_the_reference_row_of_its_retriever(sources, fake_model):
    """The counts a row records, worked out again from the questions of both rows.

    At a top 1 the three sizes find different questions. In the shared run
    every size finds the same ones, and a row set beside itself would read
    exactly as one set beside the reference.
    """
    manifest = run_chunk_sweep(
        run_id="top-one", budgets=SIZES, retrievers=["bm25"], per_filing=0, top_k=1, **sources)
    run_dir = sources["results_root"] / "top-one"
    found = {
        size: {row["question_id"]: (row["hit"], row["context_hit"])
               for row in _rows(run_dir, f"S{size}-bm25")}
        for size in SIZES
    }
    disagreed = 0
    for row in manifest["configurations"]:
        size = row["config"]["chunk_budget"]
        if size == 1200:
            assert row["against_reference"] is None
            continue
        assert row["against_reference"]["reference"] == "S1200-bm25"
        for position, cutoff in enumerate(("top_k", "context_k")):
            here = {name for name, hits in found[size].items() if hits[position]}
            there = {name for name, hits in found[1200].items() if hits[position]}
            assert row["against_reference"][cutoff] == {
                "both": len(here & there), "only_this": len(here - there),
                "only_reference": len(there - here), "neither": QUESTIONS - len(here | there),
            }
            disagreed += len(here ^ there)
    # With no question the sizes disagree on, the counts above would prove nothing.
    assert disagreed > 0


def test_two_rows_are_compared_on_the_questions_one_found_and_the_other_did_not():
    here = {"q1": (True, True), "q2": (True, False), "q3": (False, True), "q4": (False, False),
            "q5": (True, True)}
    there = {"q1": (True, True), "q2": (False, True), "q3": (True, True), "q4": (False, False),
             "q5": (False, False)}
    assert compared_with(here, there) == {
        "top_k": {"both": 1, "only_this": 2, "only_reference": 1, "neither": 1},
        "context_k": {"both": 2, "only_this": 1, "only_reference": 1, "neither": 1},
    }
    # A row beside itself differs on nothing.
    assert compared_with(here, here)["top_k"] == {
        "both": 3, "only_this": 0, "only_reference": 0, "neither": 2}


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


class _Fixed:
    """A retriever that returns the same twelve passages, best first, for every question."""

    name = "bm25"

    def __init__(self):
        self.depths: list[int] = []

    def search(self, query, k=None):
        wanted = query.top_k if k is None else k
        self.depths.append(wanted)
        # The passage ranked n holds n characters.
        return [
            SimpleNamespace(chunk_id=f"c{rank}", text="x" * rank, score=1.0 / rank, rank=rank,
                            retriever="bm25")
            for rank in range(1, 13)
        ][:wanted]


def test_each_cutoff_is_scored_on_its_own_share_of_one_ranking(tmp_path, monkeypatch):
    """Twelve passages of 2,400 characters fill the prompt, and ten are the top 10.

    A supporting passage ranked eleventh is missed by the top 10 and found in
    the prompt. One ranked third is found by both. The synthetic filings are
    too small to tell the two cutoffs apart, so the ranking is fixed here.
    """
    retriever = _Fixed()
    monkeypatch.setattr(sweep, "build_retriever", lambda key, **_: retriever)
    build = SweepBuild(2400, tmp_path / "2400")
    assert build.context_k == 12
    questions = [_question("near", "AAA", 2024, ["c3"]), _question("far", "AAA", 2024, ["c11"])]

    summaries, found = measure_build(
        build, questions, ("bm25",), run_dir=tmp_path / "run", top_k=10,
        described={}, support={"near": "prose", "far": "tables"},
    )

    near, far = _rows(tmp_path / "run", "S2400-bm25")
    # Each question is searched once, to the deeper cutoff.
    assert retriever.depths[-2:] == [12, 12]
    assert far["retrieved_chunk_ids"] == [f"c{rank}" for rank in range(1, 13)]
    # A length for every passage of that ranking, not only the top 10, and the
    # passages each question was scored against: what an equal-text rescoring reads.
    assert far["retrieved_chars"] == near["retrieved_chars"] == list(range(1, 13))
    assert (near["supporting_chunk_ids"], far["supporting_chunk_ids"]) == (["c3"], ["c11"])
    assert (near["hit"], near["context_hit"]) == (True, True)
    assert (far["hit"], far["context_hit"]) == (False, True)
    # The metrics are the top 10's, so the eleventh passage earns nothing.
    assert (far["recall"], far["ndcg"], far["mrr"]) == (0.0, 0.0, 0.0)
    assert near["recall"] == 1.0 and near["mrr"] == pytest.approx(1 / 3)
    assert (far["top_chars"], far["context_chars"]) == (sum(range(1, 11)), sum(range(1, 13)))
    assert found == {"S2400-bm25": {"near": (True, True), "far": (False, True)}}

    (summary,) = summaries
    assert (summary["hit_rate"], summary["context_hit_rate"]) == (0.5, 1.0)
    assert (summary["top_chars"], summary["context_chars"]) == (55, 78)
    assert summary["by_support"]["tables"] == {
        "questions": 1, "hit_rate": 0.0, "context_hit_rate": 1.0, "mrr": 0.0}
    # BM25 reads a passage whole, so it has nothing cut short to report.
    assert summary["n_truncated"] is None


# --- what a run refuses before it starts ------------------------------------


@pytest.mark.parametrize("settings, message", [
    ({"budgets": (1200, 1200)}, "distinct positive"),
    ({"budgets": ()}, "distinct positive"),
    ({"retrievers": ("bm25", "rerank")}, "retrievers must be among"),
    ({"top_k": 0}, "top_k must be positive"),
    ({"per_filing": -1}, "per_filing zero or more"),
    ({"run_id": "../elsewhere"}, "run_id must contain only"),
])
def test_a_bad_setting_is_refused_before_anything_is_built(
        sources, fake_model, settings, message):
    with pytest.raises(ValueError, match=message):
        run_chunk_sweep(**{"run_id": "refused", "budgets": SIZES, **sources, **settings})
    assert not sources["sweep_dir"].exists() and not sources["results_root"].exists()
    assert fake_model.calls == 0


def test_a_size_that_needs_more_passages_than_hybrid_fuses_is_refused(sources, fake_model):
    """500 characters would take 58 passages to fill the prompt, and hybrid fuses 50."""
    assert SweepBuild(500, Path()).context_k > CANDIDATE_K
    with pytest.raises(ValueError, match="CANDIDATE_K"):
        run_chunk_sweep(run_id="refused", budgets=(500, 1800), **sources)
    with pytest.raises(ValueError, match="CANDIDATE_K"):
        run_chunk_sweep(run_id="refused", budgets=(1800,), top_k=CANDIDATE_K + 1, **sources)
    assert not sources["sweep_dir"].exists()


def test_a_run_without_its_inputs_says_which_command_makes_them(sources, fake_model):
    missing = {**sources, "facts_file": sources["facts_file"].parent / "absent.parquet"}
    with pytest.raises(FileNotFoundError, match="python -m src.retrieval facts"):
        run_chunk_sweep(run_id="refused", budgets=SIZES, **missing)
    empty = {**sources, "interim_dir": sources["interim_dir"].parent / "absent"}
    with pytest.raises(FileNotFoundError, match="python -m src.pipeline parse"):
        run_chunk_sweep(run_id="refused", budgets=SIZES, **empty)
    with pytest.raises(FileNotFoundError, match="is in scope"):
        run_chunk_sweep(run_id="refused", budgets=SIZES, tickers=["ZZZ"], **sources)
    assert not sources["sweep_dir"].exists() and not sources["results_root"].exists()


def _indexes_under(sweep_dir: Path) -> list[str]:
    """The BM25 and dense indexes any size has under ``sweep_dir``."""
    return sorted(str(path.relative_to(sweep_dir)) for path in sweep_dir.rglob("*")
                  if path.name in ("bm25.pkl", "chroma"))


def test_a_fiscal_year_no_filing_reports_on_is_refused_before_a_build_is_touched(
        sources, fake_model):
    """The builds an earlier run made are as it left them, and no run id is used."""
    run_chunk_sweep(run_id="kept", budgets=SIZES[:1], retrievers=["bm25"], per_filing=1, **sources)
    build = SweepBuild(SIZES[0], sources["sweep_dir"] / str(SIZES[0]))
    held = sorted(path.name for path in build.processed_dir.glob("*/*.json"))
    assert len(held) == 4

    with pytest.raises(FileNotFoundError, match="fiscal years asked for"):
        run_chunk_sweep(run_id="refused", budgets=SIZES, fiscal_years=[1999], **sources)

    assert sorted(path.name for path in build.processed_dir.glob("*/*.json")) == held
    assert [path.name for path in sources["sweep_dir"].iterdir()] == [str(SIZES[0])]
    assert not (sources["results_root"] / "refused").exists()


def test_a_run_with_nothing_to_ask_is_refused_before_an_index_is_built(sources, fake_model):
    """The facts store holds figures for one company, and the run covers the other.

    Cutting each size and looking for its questions takes seconds. Building a
    dense index takes hours on a real corpus, so the refusal has to come first.
    """
    facts = pd.read_parquet(sources["facts_file"])
    facts[facts["ticker"] == "BBB"].to_parquet(sources["facts_file"], index=False)

    with pytest.raises(ValueError, match="nothing to measure") as refusal:
        run_chunk_sweep(run_id="refused", budgets=SIZES, tickers=["AAA"], **sources)

    # The size it stopped at, what to check, and the generator's own reason under it.
    message = str(refusal.value)
    assert message.startswith("S600: ") and "python -m src.retrieval facts\n" in message
    assert "No benchmark questions generated" in message.splitlines()[-1]
    assert _indexes_under(sources["sweep_dir"]) == [] and fake_model.calls == 0
    assert not sources["results_root"].exists()


def test_sizes_with_no_question_in_common_are_refused_before_an_index_is_built(
        sources, fake_model, monkeypatch):
    """Each size can be asked something, and no question can be asked of both."""
    asked = iter([[_question("small-only", "AAA", 2024)], [_question("large-only", "AAA", 2024)]])
    monkeypatch.setattr(sweep, "generate_xbrl_questions", lambda *_, **__: next(asked))

    with pytest.raises(ValueError, match="in every size, so there is nothing to measure"):
        run_chunk_sweep(run_id="refused", budgets=SIZES[:2], **sources)

    assert _indexes_under(sources["sweep_dir"]) == [] and fake_model.calls == 0
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


# --- the same rankings at an equal amount of text ---------------------------
#
# notebooks/retrieval/chunk_sweep_equal_text.py scores a finished run again.
# What it scores against has to be what the run was measured against.


@pytest.fixture(scope="module")
def equal_text():
    """The script, loaded from its file: ``notebooks/`` is not a package."""
    path = PROJECT_ROOT / "notebooks" / "retrieval" / "chunk_sweep_equal_text.py"
    spec = importlib.util.spec_from_file_location("chunk_sweep_equal_text", path)
    module = importlib.util.module_from_spec(spec)
    before = list(sys.path)
    spec.loader.exec_module(module)
    sys.path[:] = before      # the script puts the project root on the path to run alone
    return module


def test_a_row_stores_what_its_ranking_was_scored_against(finished):
    for build in finished["builds"]:
        lengths = {row["chunk_id"]: len(row["text"])
                   for row in iter_chunks(processed_dir=build.processed_dir)}
        supporting = {
            question["question_id"]: question["supporting_chunk_ids"]
            for question in map(json.loads,
                                build.benchmark.read_text(encoding="utf-8").splitlines())}
        for retriever in ("bm25", "dense", "hybrid"):
            for row in _rows(finished["run_dir"], f"{build.id}-{retriever}"):
                assert row["supporting_chunk_ids"] == supporting[row["question_id"]]
                assert row["retrieved_chars"] == [
                    lengths[chunk_id] for chunk_id in row["retrieved_chunk_ids"]]
                assert row["retrieved_chars"] and all(row["retrieved_chars"])


def _by_hand(run_dir: Path, config_id: str, characters: int) -> tuple[float, float]:
    """The share found, and the share that ran out, worked out one ranking at a time."""
    found = ran_out = 0
    rows = _rows(run_dir, config_id)
    for row in rows:
        taken, held = 0, 0
        for length in row["retrieved_chars"]:
            if taken and held + length > characters:
                break
            taken, held = taken + 1, held + length
        ran_out += taken == len(row["retrieved_chars"])
        found += bool(set(row["supporting_chunk_ids"]) & set(row["retrieved_chunk_ids"][:taken]))
    return found / len(rows), ran_out / len(rows)


def test_a_run_is_scored_again_from_its_own_files_alone(finished, equal_text, tmp_path):
    """Nothing under the sweep folder is read, so a later run cannot change the answer."""
    characters = (1, 700, 1500, 100_000)
    scored = equal_text.rescore(finished["run_dir"], characters, sweep_dir=tmp_path / "absent")

    assert len(scored) == len(SIZES) * 3 * len(characters)
    rates = set()
    for line in scored:
        config_id = f"S{line['chunk_budget']}-{line['retriever']}"
        found, ran_out = _by_hand(finished["run_dir"], config_id, line["characters"])
        assert (line["hit_rate"], line["ran_out"]) == (found, ran_out)
        assert line["questions"] == QUESTIONS and line["run_id"] == "three-sizes"
        rates.add(line["hit_rate"])
        if line["characters"] == 1:
            # The first passage always counts, however long it is.
            assert line["passages_taken"] == 1.0 and line["ran_out"] == 0.0
        if line["characters"] == 100_000:
            # More room than any stored ranking fills: every one of them ran out.
            assert line["ran_out"] == 1.0
    # The allowances tell the sizes apart, or this would pass on a constant.
    assert len(rates) > 2


def _copy_of(finished, tmp_path: Path, stored: bool) -> tuple[Path, Path]:
    """The finished run and its builds, copied so a test can change them. With
    ``stored`` false the rows are as a run before 8 October 2026 wrote them."""
    run_dir = shutil.copytree(finished["run_dir"], tmp_path / "results" / "three-sizes")
    sweep_dir = tmp_path / "sweep"
    for build in finished["builds"]:
        shutil.copytree(build.processed_dir, sweep_dir / str(build.budget) / "processed")
        shutil.copy(build.benchmark, sweep_dir / str(build.budget) / "benchmark.jsonl")
    if not stored:
        for path in run_dir.glob("*/questions.jsonl"):
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            for row in rows:
                del row["supporting_chunk_ids"], row["retrieved_chars"]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return run_dir, sweep_dir


def _relabel(sweep_dir: Path, size: int, change) -> None:
    path = sweep_dir / str(size) / "benchmark.jsonl"
    questions = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for question in questions:
        change(question)
    path.write_text("".join(json.dumps(question) + "\n" for question in questions),
                    encoding="utf-8")


def test_a_benchmark_written_again_does_not_change_what_a_run_scores_to(
        finished, equal_text, tmp_path):
    """A later sweep over the same filings writes each build's benchmark again. It
    passes the fingerprint check, and the run was then scored against its labels."""
    run_dir, sweep_dir = _copy_of(finished, tmp_path, stored=True)
    before = equal_text.rescore(run_dir, (700, 1500), sweep_dir)
    for size in SIZES:
        _relabel(sweep_dir, size, lambda question: question.update(
            supporting_chunk_ids=["a-passage-of-some-later-benchmark"]))
    assert equal_text.rescore(run_dir, (700, 1500), sweep_dir) == before
    assert any(line["hit_rate"] > 0 for line in before)


def test_a_run_from_before_rows_stored_their_labels_is_scored_from_its_build(
        finished, equal_text, tmp_path):
    run_dir, sweep_dir = _copy_of(finished, tmp_path, stored=False)
    assert equal_text.rescore(run_dir, (700, 1500), sweep_dir) == equal_text.rescore(
        finished["run_dir"], (700, 1500), sweep_dir=tmp_path / "absent")


@pytest.mark.parametrize("change", [
    # Another passage named as the support, so a question the run found is not.
    lambda question: question.update(supporting_chunk_ids=["some-other-passage"]),
    # One more supporting passage: found as before, and Recall is out of more.
    lambda question: question["supporting_chunk_ids"].append("one-more-passage"),
])
def test_such_a_run_is_refused_once_its_benchmark_has_been_written_again(
        finished, equal_text, tmp_path, change):
    run_dir, sweep_dir = _copy_of(finished, tmp_path, stored=False)
    _relabel(sweep_dir, SIZES[1], change)
    with pytest.raises(SystemExit) as refused:
        equal_text.rescore(run_dir, (700,), sweep_dir)
    assert "is no longer the benchmark three-sizes was scored against" in str(refused.value)
    assert f"S{SIZES[1]}-" in str(refused.value)


def test_such_a_run_is_refused_once_its_build_has_been_cut_again(
        finished, equal_text, tmp_path):
    run_dir, sweep_dir = _copy_of(finished, tmp_path, stored=False)
    path = next((sweep_dir / str(SIZES[0]) / "processed").glob("*/*.json"))
    data = json.loads(path.read_text(encoding="utf-8"))
    data["chunks"][0]["text"] += " Restated."
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(SystemExit) as refused:
        equal_text.rescore(run_dir, (700,), sweep_dir)
    assert "is no longer the build three-sizes measured" in str(refused.value)


def test_a_question_the_benchmark_no_longer_asks_refuses_such_a_run(
        finished, equal_text, tmp_path):
    run_dir, sweep_dir = _copy_of(finished, tmp_path, stored=False)
    path = sweep_dir / str(SIZES[0]) / "benchmark.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[1:]) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="is no longer the benchmark"):
        equal_text.rescore(run_dir, (700,), sweep_dir)


def test_a_label_moved_past_the_first_cutoff_is_caught_at_the_second(
        finished, equal_text, tmp_path):
    """A supporting passage ranked between the two cutoffs earns no Recall, so only
    the deeper cutoff's record of the question being found can say it has gone."""
    run_dir, sweep_dir = _copy_of(finished, tmp_path, stored=False)
    size = SIZES[0]
    row = _rows(run_dir, f"S{size}-bm25")[0]
    asked = (sweep_dir / str(size) / "benchmark.jsonl").read_text(encoding="utf-8").splitlines()
    supporting = next(question["supporting_chunk_ids"] for question in map(json.loads, asked)
                      if question["question_id"] == row["question_id"])
    others = [chunk["chunk_id"] for chunk in iter_chunks(
        processed_dir=sweep_dir / str(size) / "processed") if chunk["chunk_id"] not in supporting]
    # One row, cut off after the first passage and again after the second, with a
    # supporting passage ranked second: missed by the first cutoff, found by the other.
    row.update(retrieved_chunk_ids=[others[0], supporting[0]], retrieved_scores=[2.0, 1.0],
               hit=False, context_hit=True, recall=0.0)
    (run_dir / f"S{size}-bm25" / "questions.jsonl").write_text(
        json.dumps(row) + "\n", encoding="utf-8")
    manifest = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    (kept,) = [summary for summary in manifest["configurations"]
               if summary["config"]["id"] == f"S{size}-bm25"]
    kept["config"].update(top_k=1, context_k=2)
    manifest["configurations"] = [kept]
    (run_dir / "summary.json").write_text(json.dumps(manifest), encoding="utf-8")

    (scored,) = equal_text.rescore(run_dir, (100_000,), sweep_dir)
    assert scored["hit_rate"] == 1.0

    _relabel(sweep_dir, size, lambda question: question.update(
        supporting_chunk_ids=[others[1]]) if question["question_id"] == row["question_id"]
        else None)
    with pytest.raises(SystemExit, match="is no longer the benchmark"):
        equal_text.rescore(run_dir, (100_000,), sweep_dir)


def _written_run(root: Path, rankings: list[dict]) -> Path:
    """A run of one row, written as the sweep writes one."""
    run_dir = root / "by-hand"
    config = {"id": "S1000-bm25", "retriever": "bm25", "chunk_budget": 1000,
              "top_k": 10, "context_k": 28}
    (run_dir / "S1000-bm25").mkdir(parents=True)
    (run_dir / "S1000-bm25" / "questions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rankings), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps({
        "run_id": "by-hand", "configurations": [{"config": config}],
        "builds": [{"chunk_budget": 1000, "corpus_fingerprint": "f"}],
    }), encoding="utf-8")
    return run_dir


def _ranking(question_id: str, chars: list[int], supporting_at: int | None) -> dict:
    ids = [f"{question_id}-p{position}" for position in range(len(chars))]
    return {"question_id": question_id, "retrieved_chunk_ids": ids, "retrieved_chars": chars,
            "supporting_chunk_ids": ["elsewhere"] if supporting_at is None
            else [ids[supporting_at]]}


def test_a_ranking_is_read_to_the_last_passage_that_fits_and_no_further(equal_text, tmp_path):
    run_dir = _written_run(tmp_path, [
        _ranking("q1", [400, 400, 400], supporting_at=2),      # found only once 1,200 fit
        _ranking("q2", [900, 100], supporting_at=0),            # the first passage
        _ranking("q3", [300, 300], supporting_at=None),         # never found
        _ranking("q4", [2000, 50], supporting_at=1),            # behind a passage over the room
    ])
    scored = {line["characters"]: line
              for line in equal_text.rescore(run_dir, (800, 1000, 1200, 2050), tmp_path)}

    assert [scored[n]["hit_rate"] for n in (800, 1000, 1200, 2050)] == [1 / 4, 1 / 4, 2 / 4, 3 / 4]
    # q1 takes two of three at 800; q2 one of two (900 + 100 is over); q3 both; q4 its first.
    assert scored[800]["passages_taken"] == (2 + 1 + 2 + 1) / 4
    # A ranking whose every stored passage fitted may have had room for more: q3
    # at 800, q2 and q3 at 1,000, q1 as well at 1,200, and all four at 2,050.
    assert [scored[n]["ran_out"] for n in (800, 1000, 1200, 2050)] == [
        1 / 4, 2 / 4, 3 / 4, 4 / 4]


@pytest.mark.parametrize("characters", [(0,), (-4000,), (4000, 0), ()])
def test_an_amount_of_text_that_is_not_positive_is_refused(equal_text, tmp_path, characters):
    run_dir = _written_run(tmp_path, [_ranking("q1", [400], supporting_at=0)])
    with pytest.raises(ValueError, match="must be 1 or more"):
        equal_text.rescore(run_dir, characters, tmp_path)


@pytest.fixture
def equal_text_command(equal_text, finished, monkeypatch, tmp_path):
    monkeypatch.setattr(equal_text, "RESULTS_ROOT", finished["root"] / "results")
    monkeypatch.setattr(equal_text, "SWEEP_DIR", tmp_path / "absent")
    monkeypatch.setattr(equal_text, "ROOT", tmp_path)
    monkeypatch.setattr(equal_text, "RESULTS_CSV", tmp_path / "out" / "equal_text.csv")
    return equal_text


def test_the_equal_text_command_prints_both_tables_and_saves_its_rows(
        equal_text_command, tmp_path, capsys):
    equal_text_command.main(["three-sizes", "700", "1500"])

    printed = capsys.readouterr().out
    assert "three-sizes: supporting passage within the first N characters" in printed
    assert "hit_rate" in printed and "ran_out" in printed
    saved = pd.read_csv(tmp_path / "out" / "equal_text.csv", encoding="utf-8-sig")
    assert len(saved) == len(SIZES) * 3 * 2
    assert sorted(set(saved["characters"])) == [700, 1500]
    assert sorted(set(saved["chunk_budget"])) == list(SIZES)


@pytest.mark.parametrize("arguments, said", [
    ([], "chunk_sweep_equal_text.py chunk-sweep-fy2025-20261007"),
    (["three-sizes", "many"], "must be a whole number"),
    (["three-sizes", "4000.5"], "must be a whole number"),
    (["three-sizes", "0"], "must be 1 or more"),
    (["three-sizes", "-4000"], "must be 1 or more"),
    (["no-such-run"], "no finished run at"),
])
def test_the_equal_text_command_refuses_what_it_cannot_score(
        equal_text_command, tmp_path, arguments, said):
    with pytest.raises(SystemExit) as refused:
        equal_text_command.main(arguments)
    assert said in str(refused.value)
    assert not (tmp_path / "out").exists()


def test_the_equal_text_command_says_a_run_without_its_question_files_needs_them(
        equal_text_command, finished, monkeypatch, tmp_path):
    """Only a run's summaries are committed, so a clone has no rankings to score."""
    shutil.copytree(finished["run_dir"], tmp_path / "results" / "three-sizes")
    for path in (tmp_path / "results" / "three-sizes").glob("*/questions.jsonl"):
        path.unlink()
    monkeypatch.setattr(equal_text_command, "RESULTS_ROOT", tmp_path / "results")
    with pytest.raises(SystemExit) as refused:
        equal_text_command.main(["three-sizes"])
    assert "questions.jsonl is missing" in str(refused.value)
    assert "per-question files are not committed" in str(refused.value)
