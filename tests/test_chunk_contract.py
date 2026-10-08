"""The chunk contract (#49): what a passage promises to every stage after it.

Retrieval, citation and evaluation never read a filing. They read passages, and
each relies on something the chunker promised: an id that is unique and says
where the passage sits, text that belongs to one Item, a size that fits the
budget, the identity a citation is built from, and the settings the corpus was
cut with. ``test_chunk.py`` covers how the chunker decides a cut. This file
covers what holds of the result, whatever the decision, on one synthetic filing
that has each of the cases a real one has: headings, a paragraph longer than the
budget, an Item that answers with a cross-reference, a section with no Item
number, a table too long for one passage and one too wide.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, fields, replace

import pytest

import src.pipeline.chunk as chunk_stage
from src.pipeline.chunk import (
    chunk_all,
    chunk_filing,
    interim_files,
    iter_chunks,
    load_chunked,
    load_parsed,
    processed_path_for,
    report,
    write_chunks,
)
from src.pipeline.constants import (
    CHUNK_CHAR_BUDGET,
    CHUNK_CHAR_MINIMUM,
    CHUNK_CHAR_OVERLAP,
    table_budget_for,
)
from src.pipeline.records import (
    ChunkedFiling,
    ChunkRecord,
    ParsedFiling,
    SectionRecord,
    TableRecord,
)
from src.retrieval import embed
from src.retrieval.constants import PREFILTER_FIELDS
from src.retrieval.records import RetrievedPassage

ACCESSION = "0000000001-25-000001"


def sentences(subject: str, count: int, start: int = 0) -> list[str]:
    return [f"{subject} sentence {number} describes one particular matter in enough words "
            f"to read as prose." for number in range(start, start + count)]


def paragraph(subject: str, number: int) -> str:
    """About 290 characters of ordinary prose, distinct from every other paragraph."""
    return " ".join(sentences(f"{subject} paragraph {number}", 3))


def section(section_id: str, item: str | None, title: str, text: str, *, key: bool = True,
            tables=(), stub: bool = False, resolved_from: str | None = None) -> SectionRecord:
    return SectionRecord(
        section_id=section_id, part="I", item=item, title=title, text=text, n_chars=len(text),
        n_tables=len(tables), n_data_tables=len(tables), is_key_section=key, is_stub=stub,
        resolved_from=resolved_from, confidence=None, detection_method=None, validated=True,
        tables=list(tables),
    )


LONG_TABLE = TableRecord(
    table_index=0, caption="Maturities of long-term debt",
    headers=["(In millions)", "2025", "2024"],
    rows=[[f"Debt instrument {number}", f"{5000 + number:,}", f"{4000 + number:,}"]
          for number in range(70)],
    n_rows=70, n_cols=3,
)
WIDE_TABLE = TableRecord(
    table_index=1, caption="Quarterly results",
    headers=["(In millions)", *[f"Quarter {number} of fiscal year" for number in range(1, 13)]],
    rows=[[f"Measure {row}", *[f"{1000 * row + column:,}" for column in range(1, 13)]]
          for row in range(1, 4)],
    n_rows=3, n_cols=13,
)

BUSINESS = "\n\n".join([
    "Products",
    *[paragraph("Product", number) for number in range(6)],
    "Services",
    *[paragraph("Service", number) for number in range(4)],
])
RISK = " ".join(sentences("Risk", 60))          # one paragraph, about 5,500 characters
PROPERTIES = paragraph("Property", 0)
STATEMENTS = "\n\n".join(paragraph("Statement", number) for number in range(3))


def parsed_filing() -> ParsedFiling:
    return ParsedFiling(
        ticker="AAA", cik=1, company="Alpha Corp", form="10-K", filing_date="2025-02-01",
        accession_no=ACCESSION, url=f"https://example.test/{ACCESSION}", source_path="raw.html",
        # A December year end filed the next February: the fiscal year is 2024.
        period_of_report="2024-12-31",
        sections=[
            section("part_i_item_1", "1", "Business", BUSINESS),
            section("part_i_item_1a", "1A", "Risk Factors", RISK),
            section("part_i_item_2", "2", "Properties", PROPERTIES, key=False),
            section("part_ii_item_8", "8", "Financial Statements",
                    "See the financial statements under Item 15.", stub=True,
                    resolved_from="part_iv_item_15"),
            section("part_iv_item_15", "15", "Exhibits and Financial Statement Schedules",
                    STATEMENTS, key=False, tables=[LONG_TABLE, WIDE_TABLE]),
            section("signatures", None, "Signatures",
                    paragraph("Signature", 0) + " " + paragraph("Signature", 1)),
        ],
    )


def cut(budget: int = 600, overlap: int = 100, **settings) -> ChunkedFiling:
    return chunk_filing(parsed_filing(), source_path="data/interim/AAA/filing.json",
                        budget=budget, overlap=overlap, **settings)


def prose(chunked: ChunkedFiling, section_id: str | None = None) -> list[ChunkRecord]:
    return [chunk for chunk in chunked.chunks if chunk.content_type == "prose"
            and section_id in (None, chunk.section_id)]


def tables(chunked: ChunkedFiling, table_index: int) -> list[ChunkRecord]:
    return [chunk for chunk in chunked.chunks if chunk.table_index == table_index]


BUDGETS = pytest.mark.parametrize("budget, overlap", [(600, 100), (1200, 200), (1800, 300)])


# --- identity ---------------------------------------------------------------


@BUDGETS
def test_every_passage_has_a_unique_id_that_says_where_it_sits(budget, overlap):
    chunks = cut(budget, overlap).chunks
    ids = [chunk.chunk_id for chunk in chunks]
    assert len(ids) == len(set(ids))
    for chunk in chunks:
        if chunk.content_type == "table":
            expected = (f"{ACCESSION}_{chunk.section_id}_t{chunk.table_index:03d}"
                        f"_{chunk.chunk_index:02d}")
        else:
            expected = f"{ACCESSION}_{chunk.section_id}_{chunk.chunk_index:03d}"
        assert chunk.chunk_id == expected


@BUDGETS
def test_passages_are_numbered_from_zero_in_the_order_they_were_cut(budget, overlap):
    chunked = cut(budget, overlap)
    groups: dict[tuple, list[int]] = {}
    for chunk in chunked.chunks:
        groups.setdefault((chunk.section_id, chunk.table_index), []).append(chunk.chunk_index)
    assert len(groups) == 6   # the prose of four Items, and Item 15's two tables
    for numbers in groups.values():
        assert numbers == list(range(len(numbers)))


@BUDGETS
def test_a_passage_records_its_own_size_and_is_never_empty(budget, overlap):
    for chunk in cut(budget, overlap).chunks:
        assert chunk.n_chars == len(chunk.text) > 0
        assert chunk.text == chunk.text.strip()


@BUDGETS
def test_a_passage_never_spans_two_items(budget, overlap):
    """A citation names one Item, so a passage holding text of two could not be cited."""
    source = {record.section_id: record for record in parsed_filing().sections}
    subject = {"part_i_item_1": ("Product", "Service"), "part_i_item_1a": ("Risk",),
               "part_i_item_2": ("Property",), "part_iv_item_15": ("Statement",)}
    for chunk in prose(cut(budget, overlap)):
        own = source[chunk.section_id]
        assert (chunk.item, chunk.title, chunk.part) == (own.item, own.title, own.part)
        found = set(re.findall(r"(Product|Service|Risk|Property|Statement|Signature) ", chunk.text))
        assert found and found <= set(subject[chunk.section_id])


# --- the text ---------------------------------------------------------------


@BUDGETS
def test_no_paragraph_of_a_chunked_item_is_lost(budget, overlap):
    chunked = cut(budget, overlap)
    for section_id, text in (("part_i_item_1", BUSINESS), ("part_i_item_1a", RISK),
                             ("part_i_item_2", PROPERTIES), ("part_iv_item_15", STATEMENTS)):
        held = "".join("".join(chunk.text.split()) for chunk in prose(chunked, section_id))
        for block in text.split("\n\n"):
            for sentence in re.split(r"(?<=\.)\s+", block):
                assert "".join(sentence.split()) in held


@BUDGETS
def test_ordinary_prose_stays_within_the_budget_and_its_minimum(budget, overlap):
    """A passage is closed once it holds the minimum and one more paragraph would pass
    the budget, so the minimum is as far as one can run over."""
    for chunk in prose(cut(budget, overlap)):
        assert chunk.n_chars <= budget + CHUNK_CHAR_MINIMUM + len("\n\n")


@BUDGETS
def test_a_paragraph_longer_than_the_budget_is_cut_between_sentences(budget, overlap):
    risk = prose(cut(budget, overlap), "part_i_item_1a")
    assert len(risk) > 1 and len(RISK) > budget
    whole = sentences("Risk", 60)
    for chunk in risk:
        assert chunk.text.startswith("Risk sentence ") and chunk.text.endswith(".")
        assert all(part in whole for part in re.split(r"(?<=\.)\s+", chunk.text))
    # In order, and each sentence once: the cut loses and repeats nothing.
    rejoined = [part for chunk in risk for block in chunk.text.split("\n\n")
                for part in re.split(r"(?<=\.)\s+", block)]
    assert list(dict.fromkeys(rejoined)) == whole


def test_a_passage_after_the_first_opens_with_the_tail_of_the_one_before():
    """Whole paragraphs are carried over, as many as fit the overlap."""
    short = "\n\n".join(f"Short paragraph {number} is exactly one brief sentence of prose here."
                        for number in range(30))
    filing = replace(parsed_filing(), sections=[section("part_i_item_1", "1", "Business", short)])

    def tail_within(blocks: list[str], overlap: int) -> list[str]:
        """The last whole paragraphs of a passage that fit the overlap."""
        tail: list[str] = []
        for block in reversed(blocks):
            if sum(map(len, tail)) + len(block) > overlap:
                break
            tail.insert(0, block)
        return tail

    carried = chunk_filing(filing, "s", budget=600, overlap=200).chunks
    assert len(carried) > 2
    for before, after in zip(carried, carried[1:]):
        earlier, later = before.text.split("\n\n"), after.text.split("\n\n")
        tail = tail_within(earlier, 200)
        # Three paragraphs of 66 characters fit 200, and a fourth does not.
        assert len(tail) == 3
        assert later[:3] == tail
        assert later[3] not in earlier

    apart = chunk_filing(filing, "s", budget=600, overlap=0).chunks
    for before, after in zip(apart, apart[1:]):
        assert not set(before.text.split("\n\n")) & set(after.text.split("\n\n"))
    assert len(apart) < len(carried)


@BUDGETS
def test_a_passage_carries_the_heading_it_sits_under(budget, overlap):
    """The heading in force where the passage opens, which is the one a reader would
    find above its first paragraph in the filing."""
    business = prose(cut(budget, overlap), "part_i_item_1")
    for chunk in business:
        first = chunk.text.split("\n\n")[0]
        expected = "Products" if first == "Products" or first.startswith("Product") else "Services"
        assert chunk.heading == expected
    assert business[0].heading == "Products"
    # A paragraph that is one long block has nothing above it to carry.
    assert {chunk.heading for chunk in prose(cut(budget, overlap), "part_i_item_1a")} == {None}


def test_a_passage_opening_under_a_later_heading_carries_that_one():
    headings = [chunk.heading for chunk in prose(cut(600, 100), "part_i_item_1")]
    assert headings == sorted(headings) and set(headings) == {"Products", "Services"}


def test_page_furniture_is_dropped_and_a_sentence_split_by_a_page_is_rejoined():
    text = "\n\n".join([
        paragraph("Product", 0),
        "Table of Contents",
        "47",
        "The sentence that a page break interrupted carries on",
        "after the break and ends here, as one sentence should.",
        paragraph("Product", 1),
    ])
    filing = replace(parsed_filing(), sections=[section("part_i_item_1", "1", "Business", text)])
    blocks = "\n\n".join(chunk.text for chunk in chunk_filing(filing, "s").chunks).split("\n\n")

    assert "Table of Contents" not in blocks and "47" not in blocks
    assert ("The sentence that a page break interrupted carries on after the break and ends "
            "here, as one sentence should.") in blocks


# --- which Items are indexed ------------------------------------------------


@BUDGETS
def test_a_cross_reference_is_indexed_once_under_the_item_that_holds_the_text(budget, overlap):
    """Oracle answers Item 8 by pointing at Item 15. The text is stored once, and the
    passages that hold it say they answer for Item 8 too."""
    chunked = cut(budget, overlap)
    assert not [chunk for chunk in chunked.chunks if chunk.section_id == "part_ii_item_8"]
    for chunk in chunked.chunks:
        expected = ["8"] if chunk.section_id == "part_iv_item_15" else []
        assert chunk.incorporated_into == expected


@BUDGETS
def test_a_section_with_no_item_number_is_not_indexed(budget, overlap):
    chunks = cut(budget, overlap).chunks
    assert all(chunk.item for chunk in chunks)
    assert not any("Signature" in chunk.text for chunk in chunks)


def test_only_the_targeted_items_are_cut_when_asked():
    everything, narrowed = cut(), cut(key_items_only=True)
    assert {chunk.item for chunk in everything.chunks} == {"1", "1A", "2", "15"}
    assert {chunk.item for chunk in narrowed.chunks} == {"1", "1A"}
    assert all(chunk.is_key_section for chunk in narrowed.chunks)
    assert narrowed.key_items_only and not everything.key_items_only
    kept = [chunk for chunk in everything.chunks if chunk.is_key_section]
    assert narrowed.chunks == kept


# --- tables -----------------------------------------------------------------


@BUDGETS
def test_a_long_table_is_split_by_rows_with_its_header_on_every_part(budget, overlap):
    parts = tables(cut(budget, overlap), 0)
    assert len(parts) > 1
    body: list[str] = []
    for number, part in enumerate(parts, start=1):
        lines = part.text.split("\n")
        assert lines[0] == f"Maturities of long-term debt (part {number} of {len(parts)})"
        assert lines[1] == ""
        assert lines[2] == "| (In millions) | 2025 | 2024 |"
        assert lines[3] == "| --- | --- | --- |"
        assert part.content_type == "table" and part.table_caption == LONG_TABLE.caption
        assert part.heading == LONG_TABLE.caption
        assert part.n_chars <= table_budget_for(budget)
        body += lines[4:]
    # Every row once, in the filing's order, under no part's header but its own.
    assert body == ["| " + " | ".join(row) + " |" for row in LONG_TABLE.rows]


@BUDGETS
def test_a_wide_table_is_split_by_columns_with_its_row_labels_in_every_group(budget, overlap):
    """A row of thirteen columns is wider than any budget here, so no slicing by rows
    could fit it. Each group of columns repeats the row labels, since a figure with
    no label beside it belongs to no line item."""
    parts = tables(cut(budget, overlap), 1)
    assert len(parts) > 1
    cells: dict[tuple[str, str], str] = {}
    for part in parts:
        header, rule, *rows = part.text.split("\n")[2:]
        columns = [cell.strip() for cell in header.strip("|").split("|")]
        assert columns[0] == "(In millions)" and len(columns) > 1
        assert rule.count("---") == len(columns)
        # What splitting by column bounds is the width of a line. The part as a
        # whole is its header, its rule and at least one row, whatever they add up to.
        assert max(len(line) for line in part.text.split("\n")) <= table_budget_for(budget)
        for row in rows:
            label, *figures = [cell.strip() for cell in row.strip("|").split("|")]
            assert label in {"Measure 1", "Measure 2", "Measure 3"}
            for column, figure in zip(columns[1:], figures, strict=True):
                # Each figure once, under its own row and column, in one part only.
                assert (label, column) not in cells
                cells[(label, column)] = figure
    assert cells == {(row[0], column): figure for row in WIDE_TABLE.rows
                     for column, figure in zip(WIDE_TABLE.headers[1:], row[1:])}


def test_a_larger_budget_cuts_fewer_passages_of_both_kinds():
    small, large = cut(600, 100), cut(2400, 400)
    assert len(prose(small)) > len(prose(large))
    assert len(tables(small, 0)) > len(tables(large, 0))
    assert len(tables(small, 1)) > len(tables(large, 1))


# --- what is recorded, and what reads it ------------------------------------


def test_the_filing_records_the_settings_it_was_cut_with():
    assert (cut(1200, 150).chunk_budget, cut(1200, 150).chunk_overlap) == (1200, 150)
    default = chunk_filing(parsed_filing(), source_path="s")
    assert (default.chunk_budget, default.chunk_overlap) == (CHUNK_CHAR_BUDGET, CHUNK_CHAR_OVERLAP)
    assert default.source_path == "s"


def test_cutting_the_same_filing_twice_gives_the_same_passages(tmp_path):
    first, second = cut(), cut()
    assert first == second
    write_chunks(first, tmp_path / "one" / "filing.json")
    write_chunks(second, tmp_path / "two" / "filing.json")
    assert (tmp_path / "one" / "filing.json").read_bytes() == (
        tmp_path / "two" / "filing.json").read_bytes()


def test_what_is_written_reads_back_as_the_record_that_was_cut(tmp_path):
    chunked = cut()
    write_chunks(chunked, tmp_path / "AAA" / "filing.json")
    assert load_chunked(tmp_path / "AAA" / "filing.json") == chunked


# The fields a stored row has. Retrieval indexes them, a citation is built from
# them and the benchmark keys on them, so a field renamed or dropped here has to
# be a decision and not a side effect.
PASSAGE_FIELDS = {
    "chunk_id", "section_id", "part", "item", "title", "heading", "text", "n_chars",
    "chunk_index", "is_key_section", "incorporated_into", "content_type", "table_index",
    "table_caption",
}
FILING_FIELDS = {
    "ticker", "company", "cik", "form", "filing_date", "period_of_report", "fiscal_year",
    "accession_no", "url", "chunk_budget", "chunk_overlap", "key_items_only",
}


@pytest.fixture
def stored(tmp_path):
    """The synthetic filing as the chunk stage leaves it on disk."""
    write_chunks(cut(), tmp_path / "AAA" / "10-K_2025-02-01_filing.json")
    other = replace(cut(), ticker="BBB", company="Beta Inc.", accession_no="other",
                    period_of_report="2023-06-30", filing_date="2023-08-01",
                    chunks=[replace(chunk, chunk_id=chunk.chunk_id.replace(ACCESSION, "other"))
                            for chunk in cut().chunks])
    write_chunks(other, tmp_path / "BBB" / "10-K_2023-08-01_other.json")
    return tmp_path


def test_a_stored_row_has_exactly_the_fields_the_later_stages_read(stored):
    assert {field.name for field in fields(ChunkRecord)} == PASSAGE_FIELDS
    rows = list(iter_chunks(processed_dir=stored))
    assert len(rows) == 2 * len(cut().chunks)
    for row in rows:
        assert set(row) == PASSAGE_FIELDS | FILING_FIELDS


def test_a_row_carries_its_filings_identity_and_the_year_it_reports_on(stored):
    row = next(iter_chunks(processed_dir=stored, tickers=["AAA"]))
    assert (row["ticker"], row["company"], row["cik"], row["form"]) == (
        "AAA", "Alpha Corp", 1, "10-K")
    assert (row["accession_no"], row["url"]) == (ACCESSION, f"https://example.test/{ACCESSION}")
    # Filed in 2025, reporting on 2024: the fiscal year is the one a question names.
    assert (row["filing_date"], row["period_of_report"], row["fiscal_year"]) == (
        "2025-02-01", "2024-12-31", 2024)
    assert (row["chunk_budget"], row["chunk_overlap"], row["key_items_only"]) == (600, 100, False)


def test_rows_are_narrowed_by_company_fiscal_year_and_targeted_items(stored):
    count = len(cut().chunks)
    assert len(list(iter_chunks(processed_dir=stored, tickers=["bbb"]))) == count
    assert {row["ticker"] for row in iter_chunks(processed_dir=stored, fiscal_years=[2024])} == {
        "AAA"}
    assert {row["ticker"] for row in iter_chunks(processed_dir=stored,
                                                 fiscal_years=range(2023, 2025))} == {"AAA", "BBB"}
    assert not list(iter_chunks(processed_dir=stored, fiscal_years=[2025]))
    key = list(iter_chunks(processed_dir=stored, key_items_only=True))
    assert key and all(row["is_key_section"] for row in key)
    assert {row["item"] for row in key} == {"1", "1A"}


def test_a_corpus_cut_before_the_settings_were_recorded_still_loads(stored):
    """Its rows say the settings are unknown, which is true, and nothing fails on a missing key."""
    path = stored / "AAA" / "10-K_2025-02-01_filing.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    for field in ("chunk_budget", "chunk_overlap", "key_items_only"):
        del data[field]
    path.write_text(json.dumps(data), encoding="utf-8")

    older = load_chunked(path)
    assert (older.chunk_budget, older.chunk_overlap, older.key_items_only) == (None, None, False)
    row = next(iter_chunks(processed_dir=stored, tickers=["AAA"]))
    assert (row["chunk_budget"], row["chunk_overlap"]) == (None, None)


def test_every_row_the_chunker_writes_is_one_a_retriever_can_return(stored):
    """The seam between the pipeline and retrieval: a row becomes a ranked passage, and
    survives the dense index's storage, with what a citation needs intact."""
    for row in iter_chunks(processed_dir=stored):
        passage = RetrievedPassage.from_chunk(row, score=1.0, rank=1, retriever="bm25")
        for field in ("chunk_id", "text", "ticker", "company", "fiscal_year", "item", "title",
                      "url", "content_type", "cik", "form", "part", "filing_date"):
            assert getattr(passage, field) == row[field]

        stored_as = embed.metadata_for(row)
        assert all(isinstance(value, (str, int, float, bool)) for value in stored_as.values())
        assert {field: stored_as[field] for field in PREFILTER_FIELDS} == {
            field: row[field] for field in PREFILTER_FIELDS}
        back = embed.chunk_from_record(row["chunk_id"], row["text"], stored_as)
        assert RetrievedPassage.from_chunk(back, score=1.0, rank=1, retriever="bm25") == passage
        assert back["incorporated_into"] == row["incorporated_into"]


# --- the stage as a whole ---------------------------------------------------


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """An interim folder holding the synthetic filing, with the stage pointed at it."""
    monkeypatch.setattr(chunk_stage, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(chunk_stage, "ensure_data_dirs", lambda: None)
    interim = tmp_path / "data" / "interim"
    for ticker, form in (("AAA", "10-K"), ("BBB", "10-Q")):
        path = interim / ticker / f"{form}_2025-02-01_{ticker}.json"
        path.parent.mkdir(parents=True)
        filing = replace(parsed_filing(), ticker=ticker, form=form, accession_no=f"acc-{ticker}")
        path.write_text(json.dumps(asdict(filing)), encoding="utf-8")
    return {"interim": interim, "processed": tmp_path / "data" / "processed"}


def test_parsed_filings_are_read_back_as_the_records_the_parse_stage_wrote(workspace):
    paths = interim_files(workspace["interim"])
    assert [path.parent.name for path in paths] == ["AAA", "BBB"]
    loaded = load_parsed(paths[0])
    assert loaded == replace(parsed_filing(), accession_no="acc-AAA")
    assert isinstance(loaded.sections[4].tables[0], TableRecord)


def test_each_filing_is_written_beside_its_parsed_file_under_the_same_name(workspace):
    paths = interim_files(workspace["interim"])
    written = chunk_all(paths, budget=600, overlap=100, processed_dir=workspace["processed"])

    assert [filing.ticker for filing in written] == ["AAA", "BBB"]
    for path, filing in zip(paths, written):
        destination = processed_path_for(path, workspace["processed"])
        assert destination == workspace["processed"] / path.parent.name / path.name
        assert load_chunked(destination) == filing
        assert filing.source_path == f"data/interim/{path.parent.name}/{path.name}"
        assert (filing.chunk_budget, filing.chunk_overlap) == (600, 100)


def test_a_filing_already_cut_is_skipped_unless_forced(workspace, caplog):
    paths = interim_files(workspace["interim"])
    chunk_all(paths, budget=600, overlap=100, processed_dir=workspace["processed"])

    with caplog.at_level(logging.INFO, logger="src.pipeline.chunk"):
        assert chunk_all(paths, budget=1200, processed_dir=workspace["processed"]) == []
    assert caplog.text.count("already chunked, skipping") == 2
    kept = load_chunked(processed_path_for(paths[0], workspace["processed"]))
    assert kept.chunk_budget == 600

    forced = chunk_all(paths, force=True, budget=1200, processed_dir=workspace["processed"])
    assert [filing.chunk_budget for filing in forced] == [1200, 1200]
    assert load_chunked(processed_path_for(paths[0], workspace["processed"])).chunk_budget == 1200


def test_only_the_forms_asked_for_are_cut(workspace):
    written = chunk_all(interim_files(workspace["interim"]), forms=["10-Q"],
                        processed_dir=workspace["processed"])
    assert [filing.form for filing in written] == ["10-Q"]
    assert not (workspace["processed"] / "AAA").exists()


def test_one_unreadable_filing_does_not_cost_the_rest(workspace, caplog):
    broken = workspace["interim"] / "AAA" / "10-K_2025-02-01_AAA.json"
    broken.write_text(json.dumps({"ticker": "AAA", "sections": []}), encoding="utf-8")

    with caplog.at_level(logging.ERROR, logger="src.pipeline.chunk"):
        written = chunk_all(interim_files(workspace["interim"]),
                            processed_dir=workspace["processed"])
    assert [filing.ticker for filing in written] == ["BBB"]
    assert "could not be read, moving on" in caplog.text


def test_the_run_summary_says_what_was_cut_and_what_answers_for_another_item(capsys):
    report([cut()])
    printed = capsys.readouterr().out

    chunks = cut().chunks
    assert f"{len(chunks):,} passages" in printed
    key = sum(chunk.is_key_section for chunk in chunks)
    assert f"{key:,} from key Items, {len(chunks) - key:,} from the rest" in printed
    assert "1 Items hold text that another Item cross-refers to:" in printed
    assert "AAA 2025-02-01  Item 15 also answers Item 8" in printed


def test_the_run_summary_says_so_when_nothing_was_cut(capsys):
    report([])
    assert "Nothing new was chunked. Use --force" in capsys.readouterr().out
    report([replace(cut(), chunks=[])])
    assert "No section in those filings held text to chunk." in capsys.readouterr().out


def test_the_run_summary_names_passages_too_short_to_be_worth_a_vector(capsys):
    """The last slice of a long table is often a row or two, which is expected. A run
    of them in prose would mean the packer is closing passages early."""
    runt = replace(cut().chunks[0], chunk_id="a-stranded-heading", text="Short.", n_chars=6)
    chunks = [runt, *cut().chunks]
    report([replace(cut(), chunks=chunks)])
    printed = capsys.readouterr().out
    short = sum(chunk.n_chars < 200 for chunk in chunks)
    assert f"{short} passages are under 200 characters:" in printed
    assert "a-stranded-heading  6 chars" in printed
