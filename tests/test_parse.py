"""Rebuilding tables from the grid pandas hands back (#62, #64).

``_extract_tables`` is fed a stand-in for an edgartools section whose tables
return prepared DataFrames, so the rebuild logic is tested without a filing.
The first frame is the exact shape of the table #62 was diagnosed on: Oracle's
FY2023 working capital, whose figures pandas read into the third header row.
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from src.pipeline.parse import _extract_tables


def section_of(*frames, name="part_ii_item_7"):
    tables = [
        SimpleNamespace(to_dataframe=lambda frame=frame: frame, caption=None,
                        html=lambda: "<table></table>", row_count=0, col_count=0)
        for frame in frames
    ]
    return SimpleNamespace(name=name, tables=lambda: tables)


def working_capital() -> pd.DataFrame:
    """As ``to_dataframe`` returns ORCL 0000950170-23-028914, Item 7, table 15."""
    columns = pd.MultiIndex.from_tuples([
        ("As of May 31,", "2023", "$(2,086"), ("As of May 31,", "", ")"),
        ("As of May 31,", "", ""), ("As of May 31,", "Change", "*"),
        ("As of May 31,", "", ""), ("As of May 31,", "2022", "$12,122"),
    ])
    index = pd.Index(["Cash, cash equivalents and marketable securities"],
                     name=("", "(Dollars in millions)", "Working capital"))
    return pd.DataFrame([["$10,187", "", "", -53.0, "", "$21,902"]], index=index, columns=columns)


def test_a_table_whose_data_sits_in_its_header_is_rebuilt():
    records, data_tables, failures = _extract_tables(section_of(working_capital()))
    assert failures == [] and data_tables == 1 and len(records) == 1
    table = records[0]
    # The figures are in a row, under the label that says what they are.
    assert table.rows[0][0] == "Working capital"
    assert "$(2,086" in table.rows[0] and "$12,122" in table.rows[0]
    assert table.rows[1][0] == "Cash, cash equivalents and marketable securities"
    # "As of May 31," spans every column, so it is stated once, above the labels.
    assert table.headers == ["(Dollars in millions) As of May 31,", "2023", "", "Change", "2022"]


def test_the_caption_is_never_a_tuple():
    records, _, _ = _extract_tables(section_of(working_capital()))
    assert records[0].caption == "(Dollars in millions)"


def test_an_all_blank_index_name_gives_no_caption():
    """So the chunker falls back to the Item title rather than "('', '')"."""
    frame = pd.DataFrame(
        [["100", "200"], ["300", "400"]],
        index=pd.Index(["Revenue", "Cost"], name=("", "")),
        columns=pd.MultiIndex.from_tuples([("Year", "2024"), ("Year", "2023")]),
    )
    records, _, _ = _extract_tables(section_of(frame))
    assert records[0].caption == ""
    assert records[0].headers == ["Year", "2024", "2023"]
    assert records[0].rows == [["Revenue", "100", "200"], ["Cost", "300", "400"]]


@pytest.mark.parametrize("number", [0, 4])
def test_a_numbered_index_is_no_label(number):
    """pandas numbers the column it made the index; that labels nothing."""
    frame = pd.DataFrame(
        [["1,000", "900"], ["250", "200"]],
        index=pd.Index(["Revenue", "Net income"], name=number),
        columns=["Fiscal 2024", "Fiscal 2023"],
    )
    records, _, _ = _extract_tables(section_of(frame))
    assert records[0].caption == ""
    assert records[0].headers == ["", "Fiscal 2024", "Fiscal 2023"]


def test_a_caption_without_letters_is_no_caption():
    """Salesforce's FY2021 filing puts a stray "4" in the corner cell of its tables."""
    frame = pd.DataFrame(
        [["1,000", "900"], ["250", "200"]],
        index=pd.Index(["Revenues", "Net income"], name="4"),
        columns=["Fiscal 2021", "Fiscal 2020"],
    )
    records, _, _ = _extract_tables(section_of(frame))
    assert records[0].caption == ""


def test_pandas_numbering_labels_nothing():
    """No header row and no label column: pandas numbers both, and neither is text."""
    frame = pd.DataFrame([["2022", "9,583"], ["2023", "10,000"]])   # columns 0, 1; RangeIndex
    records, _, _ = _extract_tables(section_of(frame))
    assert records[0].headers == ["", ""]
    assert records[0].rows == [["2022", "9,583"], ["2023", "10,000"]]


def test_a_table_with_an_ordinary_header_is_unchanged():
    frame = pd.DataFrame(
        [[1000.0, 900.0], [250.0, 200.0]],
        index=pd.Index(["Revenue", "Net income"], name="(In millions)"),
        columns=["2024", "2023"],
    )
    records, _, _ = _extract_tables(section_of(frame))
    table = records[0]
    assert table.caption == "(In millions)"
    assert table.headers == ["(In millions)", "2024", "2023"]
    assert table.rows == [["Revenue", "1,000", "900"], ["Net income", "250", "200"]]


def test_a_header_spanning_every_column_is_stated_once():
    frame = pd.DataFrame(
        [["$1,000", "$400", "$600"], ["$200", "$200", "-"]],
        index=pd.Index(["Money market funds", "Treasuries"], name="(in millions)"),
        columns=pd.MultiIndex.from_tuples([
            ("Fair Value Measurements Using", "Total"),
            ("Fair Value Measurements Using", "Level 1"),
            ("Fair Value Measurements Using", "Level 2"),
        ]),
    )
    records, _, _ = _extract_tables(section_of(frame))
    assert records[0].headers == ["(in millions) Fair Value Measurements Using",
                                  "Total", "Level 1", "Level 2"]
    assert records[0].caption == "(in millions)"


def test_a_spanning_header_the_row_labels_already_state_is_not_repeated():
    """Microsoft's income statement: "(In millions)" is on both axes.

    pandas puts the same header levels on the index name and on the columns, so
    the phrase factored out of the column labels is often already in the corner
    cell. Appending it would print it twice in the cell that opens every piece
    of a split table.
    """
    frame = pd.DataFrame(
        [["211,915", "198,270"], ["88,136", "83,383"]],
        index=pd.Index(["Revenue", "Cost of revenue"],
                       name=("(In millions)", "Year Ended June 30,")),
        columns=pd.MultiIndex.from_tuples([
            ("(In millions)", "2024"), ("(In millions)", "2023"),
        ]),
    )
    records, _, _ = _extract_tables(section_of(frame))
    assert records[0].headers == ["(In millions) Year Ended June 30,", "2024", "2023"]


def test_a_header_over_only_some_columns_is_left_in_each_label():
    """Two year groups: neither spans every column, so neither is factored out."""
    frame = pd.DataFrame(
        [["10", "20", "30", "40"], ["50", "60", "70", "80"]],
        index=pd.Index(["Revenue", "Cost"], name="(in millions)"),
        columns=pd.MultiIndex.from_tuples([
            ("2024", "First quarter"), ("2024", "Second quarter"),
            ("2023", "First quarter"), ("2023", "Second quarter"),
        ]),
    )
    records, _, _ = _extract_tables(section_of(frame))
    assert records[0].headers == ["(in millions)", "2024 First quarter", "2024 Second quarter",
                                  "2023 First quarter", "2023 Second quarter"]


def test_a_header_level_repeated_by_the_filer_is_stated_once():
    """ServiceNow's exhibit index marks its header row twice.

    The same text arrives at two levels of the same column, and joining the
    levels printed it twice on every piece of a split table. A level that only
    repeats the one above it says nothing the first did not.
    """
    frame = pd.DataFrame(
        [["3.1", "Restated Certificate of Incorporation", "8-K"],
         ["4.1", "Form of Common Stock Certificate", "S-1/A"]],
        columns=pd.MultiIndex.from_tuples([
            ("Exhibit Number", "Exhibit Number"),
            ("Description of Document", "Description of Document"),
            ("Incorporated by Reference", "Form"),
        ]),
    )
    records, _, _ = _extract_tables(section_of(frame, name="part_iv_item_15"))
    assert records[0].headers == ["Exhibit Number", "Description of Document",
                                  "Incorporated by Reference Form"]


def test_a_single_header_level_is_never_factored_out():
    """Nothing would be left to tell the columns apart."""
    frame = pd.DataFrame(
        [["10", "20"], ["30", "40"]],
        index=pd.Index(["Revenue", "Cost"], name="(in millions)"),
        columns=pd.MultiIndex.from_tuples([("Amount", ""), ("Amount", "")]),
    )
    records, _, _ = _extract_tables(section_of(frame))
    assert records[0].headers == ["(in millions)", "Amount", "Amount"]


def test_a_one_line_schedule_of_figures_is_kept():
    """Salesforce's "Employee stock awards | 7 | 13 | 39": three cells by the count.

    Laid out as filers do, with spacer columns between the years, which is what
    gets it past the wrapper check on arrival and down to three cells after.
    """
    frame = pd.DataFrame(
        [["7", "", "13", "", "39"]],
        index=pd.Index(["Employee stock awards"], name="Fiscal Year Ended January 31,"),
        columns=["2025", "", "2024", "", "2023"],
    )
    records, _, failures = _extract_tables(section_of(frame))
    assert failures == []
    assert records[0].rows == [["Employee stock awards", "7", "13", "39"]]


def test_a_one_row_exhibit_fragment_without_labels_is_kept():
    """Intuit's exhibit 104, split into a table of its own: no label column, one row."""
    # Spacer columns, as filed: five cells on arrival, three once they are dropped.
    frame = pd.DataFrame(
        [["104", "", "Cover Page Interactive Data File (embedded within Inline XBRL)", "", "X"]],
        columns=["Exhibit Number", "", "Exhibit Description", "", "Filed Herewith"],
    )
    records, _, failures = _extract_tables(section_of(frame, name="part_iv_item_15"))
    assert failures == []
    assert records[0].headers == ["Exhibit Number", "Exhibit Description", "Filed Herewith"]
    assert records[0].rows[0][0] == "104"


def test_a_signature_block_is_still_rejected():
    frame = pd.DataFrame(
        [["CISCO SYSTEMS, INC.", ""], ["/S/ CHARLES H. ROBBINS", ""],
         ["Chair and Chief Executive Officer", ""]],
        columns=["September 3, 2025", ""],
    )
    records, _, failures = _extract_tables(section_of(frame, name="signatures"))
    assert records == [] and len(failures) == 1


def test_a_layout_table_is_still_not_a_table():
    frame = pd.DataFrame([["", ""], ["", "x"]], columns=["a", "b"])
    records, _, failures = _extract_tables(section_of(frame))
    assert records == []
    assert [failure.kind for failure in failures] == ["too few cells after dropping empty columns"]


# --- statement titles (#88) -----------------------------------------------------

def statement_section(markdown: str, *tables, name="part_ii_item_8"):
    """A section whose tables carry their rows, as the real parser's do."""
    nodes = [
        SimpleNamespace(
            to_dataframe=lambda rows=rows: pd.DataFrame(
                [row[1:] for row in rows],
                index=pd.Index([row[0] for row in rows], name="Caption"),
                columns=["2024", "2023"],
            ),
            caption=None, rows=rows, html=lambda: "<table></table>",
            row_count=len(rows), col_count=3,
        )
        for rows in tables
    ]
    return SimpleNamespace(name=name, tables=lambda: nodes, markdown=markdown)


BALANCE_SHEET = [["Total current assets", "152,987", "143,566"],
                 ["Total assets", "364,980", "352,583"],
                 ["Total liabilities", "308,030", "290,437"]]
SEGMENTS = [["Americas", "167,045", "162,560"],
            ["Europe", "101,328", "94,294"],
            ["Greater China", "66,952", "72,559"]]


def markdown_for(*blocks: str) -> str:
    return NEWLINE.join(blocks)


NEWLINE = chr(10)


def rows_markdown(rows) -> str:
    return NEWLINE.join("| " + " | ".join(row) + " |" for row in rows)


def test_a_statement_table_takes_the_filing_own_heading():
    markdown = markdown_for(
        "**Apple Inc.**", "", "**CONSOLIDATED BALANCE SHEETS**", "",
        "(In millions, except par value)", "", rows_markdown(BALANCE_SHEET),
    )
    records, _, _ = _extract_tables(statement_section(markdown, BALANCE_SHEET))
    assert records[0].statement_title == "CONSOLIDATED BALANCE SHEETS"


def test_a_note_below_a_statement_does_not_inherit_its_title():
    markdown = markdown_for(
        "**CONSOLIDATED BALANCE SHEETS**", "", rows_markdown(BALANCE_SHEET), "",
        "Segment information is reported below.", "", "Products and services", "",
        "The Company reports segments as follows.", "", "Reportable segments", "",
        "Revenue by segment follows.", "", "Segment detail", "", "More prose here.", "",
        rows_markdown(SEGMENTS),
    )
    records, _, _ = _extract_tables(statement_section(markdown, BALANCE_SHEET, SEGMENTS))
    assert [record.statement_title for record in records] == ["CONSOLIDATED BALANCE SHEETS", ""]


def test_a_section_without_markdown_leaves_every_title_empty():
    records, _, _ = _extract_tables(section_of(working_capital()))
    assert records[0].statement_title == ""


def test_a_heading_written_the_other_way_round_is_read():
    """Microsoft heads the same statements "INCOME STATEMENTS", "BALANCE SHEETS"."""
    markdown = markdown_for("**BALANCE SHEETS**", "", "(In millions)", "",
                            rows_markdown(BALANCE_SHEET))
    records, _, _ = _extract_tables(statement_section(markdown, BALANCE_SHEET))
    assert records[0].statement_title == "BALANCE SHEETS"


def test_a_heading_set_as_the_table_own_first_row_is_read():
    """Intuit's heading is a row of the table: "... OPERATIONS | | | | |"."""
    markdown = markdown_for("| CONSOLIDATED BALANCE SHEETS |  |  |", rows_markdown(BALANCE_SHEET))
    records, _, _ = _extract_tables(statement_section(markdown, BALANCE_SHEET))
    assert records[0].statement_title == "CONSOLIDATED BALANCE SHEETS"


def test_a_row_that_wraps_does_not_hide_the_heading():
    """A row too long for one line ends the run of table lines (Adobe)."""
    head, tail = BALANCE_SHEET[:1], BALANCE_SHEET[1:]
    markdown = markdown_for(
        "**CONSOLIDATED BALANCE SHEETS**", "", "(In millions)", "",
        rows_markdown(head), "",
        "Adjustments to reconcile net income to net cash provided by operating", "",
        rows_markdown(tail),
    )
    records, _, _ = _extract_tables(statement_section(markdown, BALANCE_SHEET))
    assert records[0].statement_title == "CONSOLIDATED BALANCE SHEETS"


def test_a_heading_below_a_long_gap_does_not_reach_the_next_table():
    prose = ["Some prose about the segment results."] * 10
    markdown = markdown_for("**CONSOLIDATED BALANCE SHEETS**", "", *prose, "",
                            rows_markdown(SEGMENTS))
    records, _, _ = _extract_tables(statement_section(markdown, SEGMENTS))
    assert records[0].statement_title == ""


def test_a_heading_set_with_letter_spacing_is_read_and_repaired():
    """Microsoft: "CASH FLOWS S TATEMENTS" is one heading, spaced out."""
    markdown = markdown_for("**CASH FLOWS S TATEMENTS**", "", rows_markdown(BALANCE_SHEET))
    records, _, _ = _extract_tables(statement_section(markdown, BALANCE_SHEET))
    assert records[0].statement_title == "CASH FLOWS STATEMENTS"


def test_the_index_of_statements_is_not_itself_a_statement():
    """The Item opens with a table listing every statement and its page."""
    index = [["Consolidated Balance Sheets", "66", ""],
             ["Consolidated Statements of Operations", "67", ""],
             ["Consolidated Statements of Cash Flows", "68", ""]]
    markdown = markdown_for(rows_markdown(index), "", "**CONSOLIDATED BALANCE SHEETS**", "",
                            rows_markdown(BALANCE_SHEET))
    records, _, _ = _extract_tables(statement_section(markdown, index, BALANCE_SHEET))
    assert [record.statement_title for record in records] == ["", "CONSOLIDATED BALANCE SHEETS"]


def test_a_heading_two_rows_into_the_table_is_read():
    """Palo Alto opens the table with spacer rows, then the heading."""
    markdown = markdown_for("|  |  |  |", "| CONSOLIDATED BALANCE SHEETS |  |  |",
                            rows_markdown(BALANCE_SHEET))
    records, _, _ = _extract_tables(statement_section(markdown, BALANCE_SHEET))
    assert records[0].statement_title == "CONSOLIDATED BALANCE SHEETS"


def test_a_company_prefixed_heading_in_a_gap_still_ends_the_table():
    """The gap between two runs of rows is a new table when a heading sits in it.

    ServiceNow heads each statement with its own name in front, and the gap
    check used to test the bare pattern, which does not know that form. The
    two runs were rejoined and the cash flow rows took the balance sheet's
    title.
    """
    markdown = markdown_for(
        "**CONSOLIDATED BALANCE SHEETS**", "", rows_markdown(BALANCE_SHEET), "",
        "ServiceNow, Inc. Consolidated Statements of Cash Flows", "",
        rows_markdown(SEGMENTS),
    )
    records, _, _ = _extract_tables(statement_section(markdown, BALANCE_SHEET, SEGMENTS))
    assert [record.statement_title for record in records] == [
        "CONSOLIDATED BALANCE SHEETS", "Consolidated Statements of Cash Flows",
    ]


def test_a_heading_combining_two_statements_is_read():
    """A good number of filers head the income statement with both names."""
    markdown = markdown_for(
        "**Consolidated Statements of Operations and Comprehensive Income (Loss)**", "",
        "(In millions)", "", rows_markdown(BALANCE_SHEET),
    )
    records, _, _ = _extract_tables(statement_section(markdown, BALANCE_SHEET))
    assert records[0].statement_title == (
        "Consolidated Statements of Operations and Comprehensive Income (Loss)"
    )


def test_a_short_block_does_not_read_the_heading_below_it():
    """A block shorter than STATEMENT_TITLE_ROWS must not scan past its own end."""
    note = [["Deferred tax assets", "1,200", "1,100"],
            ["Deferred tax liabilities", "300", "250"]]
    markdown = markdown_for(
        rows_markdown(note), "**CONSOLIDATED BALANCE SHEETS**", "",
        rows_markdown(BALANCE_SHEET),
    )
    records, _, _ = _extract_tables(statement_section(markdown, note, BALANCE_SHEET))
    assert [record.statement_title for record in records] == [
        "", "CONSOLIDATED BALANCE SHEETS",
    ]


def test_a_running_header_repeated_at_each_page_break_is_not_the_index():
    """ServiceNow repeats the heading in the header opening every page."""
    header = "| Table of Contents | Part II | Consolidated Balance Sheets | (in millions) |"
    head, tail = BALANCE_SHEET[:2], BALANCE_SHEET[2:]
    markdown = markdown_for(header, rows_markdown(head), header, rows_markdown(tail))
    records, _, _ = _extract_tables(statement_section(markdown, BALANCE_SHEET))
    assert records[0].statement_title == "Consolidated Balance Sheets"


def test_a_statement_broken_over_three_page_blocks_keeps_its_title():
    """Each block holds a third of the rows, so pairing must not divide by the union."""
    rows = [[f"Line item {i}", f"{i}00", f"{i}50"] for i in range(9)]
    blocks = [rows_markdown(rows[i:i + 3]) for i in (0, 3, 6)]
    markdown = markdown_for(
        "**CONSOLIDATED BALANCE SHEETS**", "", blocks[0], "",
        "**CONSOLIDATED BALANCE SHEETS**", "", blocks[1], "",
        "**CONSOLIDATED BALANCE SHEETS**", "", blocks[2],
    )
    records, _, _ = _extract_tables(statement_section(markdown, rows))
    assert records[0].statement_title == "CONSOLIDATED BALANCE SHEETS"


def test_a_single_bulleted_heading_is_still_read():
    """A filer may bullet the heading, and the bullet arrives as a lost glyph."""
    markdown = markdown_for(
        "\ufffd Balance sheets as of December 31, 2025 and 2024.", "",
        rows_markdown(BALANCE_SHEET),
    )
    records, _, _ = _extract_tables(statement_section(markdown, BALANCE_SHEET))
    assert records[0].statement_title == "Balance sheets as of December 31, 2025 and 2024"


def test_a_run_of_bulleted_statements_is_an_index_not_a_heading():
    """Texas Instruments bullets all six statements under a list heading."""
    markdown = markdown_for(
        "List of financial statements:", "",
        "\ufffd Income for each of the three years ended December 31, 2025.", "",
        "\ufffd Comprehensive income for each of the three years ended December 31, 2025.", "",
        "\ufffd Balance sheets as of December 31, 2025 and 2024.", "",
        "\ufffd Cash flows for each of the three years ended December 31, 2025.", "",
        rows_markdown(BALANCE_SHEET),
    )
    records, _, _ = _extract_tables(statement_section(markdown, BALANCE_SHEET))
    assert records[0].statement_title == ""
