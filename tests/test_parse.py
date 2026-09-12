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
