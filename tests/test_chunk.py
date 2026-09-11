"""Chunker settings, dense text, flattened tables, and table passages."""

from __future__ import annotations

from src.pipeline.chunk import (
    _cost,
    _is_table_debris,
    chunk_section,
    chunk_tables,
    prose_blocks,
    resolve_chunk_settings,
    table_cells,
    table_figures,
)
from src.pipeline.constants import (
    CHUNK_CHAR_BUDGET,
    CHUNK_CHAR_OVERLAP,
    PROSE_CHARS_PER_TOKEN,
    TABLE_CHARS_PER_TOKEN,
    table_budget_for,
)
from src.pipeline.records import SectionRecord, TableRecord

PROSE = ("The Company designs, manufactures and markets smartphones, personal "
         "computers, tablets, wearables and accessories, and sells a variety of "
         "related services to consumers and businesses around the world.")
FIGURES = "$6.53 1.5 $34.2 8.3 $64.71 5.2 $170.9 (0.2) $304.29 0.1 $26.20 0.8 $27.4"


def test_ordinary_prose_costs_its_length():
    """So every passage of ordinary prose is cut exactly where it was before."""
    assert _cost(PROSE) == len(PROSE)


def test_dense_text_costs_up_to_the_table_rate():
    assert _cost(FIGURES) == round(len(FIGURES) * PROSE_CHARS_PER_TOKEN / TABLE_CHARS_PER_TOKEN)


def section(text="", tables=(), title="Financial Statements"):
    return SectionRecord(
        section_id="part_ii_item_8", part="II", item="8", title=title, text=text,
        n_chars=len(text), n_tables=len(tables), n_data_tables=len(tables),
        is_key_section=True, is_stub=False, resolved_from=None, confidence=None,
        detection_method=None, validated=True, tables=list(tables),
    )


def test_a_passage_of_dense_text_is_cut_shorter():
    dense = section("\n\n".join(f"Row {i} {FIGURES}" for i in range(60)))
    ordinary = section("\n\n".join(f"{PROSE} ({i})" for i in range(20)))
    longest_dense = max(len(p.text) for p in chunk_section(dense, "acc"))
    assert longest_dense <= table_budget_for(CHUNK_CHAR_BUDGET) + 200
    assert max(len(p.text) for p in chunk_section(ordinary, "acc")) > longest_dense


OPTIONS = TableRecord(
    table_index=0, caption="Stock options",
    headers=["", "Shares", "Weighted average price"],
    rows=[["Balance-July 31, 2020", "0.4", "$6.53"], ["Granted", "0.5", "$101.43"],
          ["Exercised", "(0.2)", "$4,127.82"]],
    n_rows=3, n_cols=3,
)


def debris(block):
    return _is_table_debris(block, table_figures([OPTIONS]), table_cells([OPTIONS]))


def test_a_flattened_table_with_cells_run_together_is_debris():
    """"July 31, 2020" and "0.4" run together as "20200.4", a figure no table holds."""
    assert debris("Balance-July 31, 20200.4\xa0$6.53\xa0Granted0.5\xa0$101.43\xa0Exercised(0.2)$4,127.82")


def test_a_block_holding_a_figure_the_table_lacks_is_kept():
    assert not debris("Balance-July 31, 20200.4 $6.53 Granted0.5 $101.43 Forfeited 9,999")


def test_prose_is_never_debris():
    assert not debris(PROSE)
    assert not debris("Options granted during 2020 had a weighted average price of $6.53.")


def test_prose_blocks_reports_what_it_dropped():
    flattened = "Balance-July 31, 20200.4\xa0$6.53\xa0Granted0.5\xa0$101.43\xa0Exercised(0.2)$4,127.82"
    kept, dropped = prose_blocks(section(f"{PROSE}\n\n{flattened}", tables=[OPTIONS]))
    assert kept == [PROSE] and dropped == [flattened]


def test_one_recorded_pair_is_the_answer():
    assert resolve_chunk_settings({(1200, 100)}) == (1200, 100, None)


def test_an_unrecorded_corpus_assumes_the_constants_and_says_so():
    budget, overlap, note = resolve_chunk_settings({(None, None)})
    assert (budget, overlap) == (CHUNK_CHAR_BUDGET, CHUNK_CHAR_OVERLAP)
    assert "re-chunk" in note.lower()


def test_a_mixed_corpus_records_neither():
    assert resolve_chunk_settings({(1800, 300), (1200, 100)})[:2] == (None, None)
    # Half recorded, half not, is mixed too: silence is not agreement.
    assert resolve_chunk_settings({(1800, 300), (None, None)})[:2] == (None, None)


def section_with(table: TableRecord, title: str = "Financial Statements") -> SectionRecord:
    return section(tables=[table], title=title)


def long_table(caption: str) -> TableRecord:
    headers = ["(In millions)", "2024", "2023", "2022"]
    rows = [[f"Line item number {i}", f"{1000 + i:,}", f"{900 + i:,}", f"{800 + i:,}"]
            for i in range(60)]
    return TableRecord(table_index=0, caption=caption, headers=headers, rows=rows,
                       n_rows=len(rows), n_cols=len(headers))


def test_every_multi_row_piece_fits_the_budget_with_its_caption():
    caption = "Consolidated statements of operations, a caption long enough to matter"
    passages = chunk_tables(section_with(long_table(caption)), accession_no="acc")
    budget = table_budget_for(CHUNK_CHAR_BUDGET)
    assert len(passages) > 1
    for passage in passages:
        body_rows = passage.text.count("\n| ") - 1   # header and rule are two lines
        if body_rows > 1:
            assert passage.n_chars <= budget, (passage.n_chars, budget)


def test_the_part_suffix_appears_once_per_piece():
    passages = chunk_tables(section_with(long_table("Operations")), accession_no="acc")
    first_lines = [passage.text.split("\n", 1)[0] for passage in passages]
    assert all(line.count("(part ") == 1 for line in first_lines)
    assert first_lines[0] == f"Operations (part 1 of {len(passages)})"


def test_an_empty_caption_falls_back_to_the_item_title():
    passages = chunk_tables(section_with(long_table(""), title="Selected Data"), accession_no="acc")
    assert passages[0].text.startswith("Selected Data")


def test_an_enormous_caption_does_not_shatter_its_table():
    passages = chunk_tables(section_with(long_table("x" * 2000)), accession_no="acc")
    assert len(passages) < 60, "one passage per row means the caption ate the budget"
