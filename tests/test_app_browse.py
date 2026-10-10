"""Issue #40: the Browse page exposes every stored passage without a model."""

from streamlit.testing.v1 import AppTest

from src.app import components, state
from src.app.components import _corpus_table_html
from src.config import PROJECT_ROOT
from src.pipeline.chunk import iter_chunks


BROWSE = str(PROJECT_ROOT / "src" / "app" / "app_pages" / "browse.py")
MAIN = str(PROJECT_ROOT / "src" / "app" / "main.py")


def passage(chunk_id, *, ticker="AAPL", company="Apple Inc.", year=2024, item="7",
            content_type="prose", text="A prose passage.", url=None):
    """A processed passage with the filing metadata ``iter_chunks`` attaches."""
    return {
        "chunk_id": chunk_id,
        "ticker": ticker,
        "company": company,
        "fiscal_year": year,
        "item": item,
        "content_type": content_type,
        "text": text,
        "url": url or f"https://www.sec.gov/Archives/{ticker}/{year}",
        "accession_no": f"{ticker}-{year}",
        "heading": "Selected heading",
        "table_caption": "",
        "title": "Selected Item",
    }


def browse(monkeypatch, rows):
    monkeypatch.setattr(state, "corpus_passages", lambda: tuple(rows))
    app = AppTest.from_file(BROWSE, default_timeout=30).run()
    assert not app.exception
    return app


def test_browse_is_registered_in_the_app_navigation(monkeypatch):
    monkeypatch.setattr(state, "corpus_passages", lambda: (passage("registered"),))
    app = AppTest.from_file(MAIN, default_timeout=30).run()
    app.switch_page("app_pages/browse.py").run()
    assert not app.exception
    assert app.title[0].value == "Browse Filing Corpus"
    assert app.main.status[0].label == "Selected heading"
    assert any(caption.value == "Chunk ID: registered" for caption in app.caption)


def test_picker_only_offers_existing_company_year_item_combinations(monkeypatch):
    rows = [
        passage("a-2024-7"),
        passage("a-2024-8", item="8"),
        passage("a-2023-1a", year=2023, item="1A"),
        passage("m-2022-8", ticker="MSFT", company="Microsoft Corp.", year=2022, item="8"),
    ]
    app = browse(monkeypatch, rows)
    assert list(app.selectbox[0].options) == ["AAPL — Apple Inc.", "MSFT — Microsoft Corp."]
    assert list(app.selectbox[1].options) == ["FY2024", "FY2023"]
    assert list(app.selectbox[2].options) == ["Item 7", "Item 8"]
    assert app.main.status[0].label == "Selected heading"

    # An upstream change replaces stale downstream values, so the three menus
    # can never describe an empty combination.
    app.selectbox[0].set_value("MSFT").run()
    assert not app.exception
    assert list(app.selectbox[1].options) == ["FY2022"]
    assert list(app.selectbox[2].options) == ["Item 8"]
    assert [panel.label for panel in app.main.status] == ["Selected heading"]
    assert any(caption.value == "Chunk ID: m-2022-8" for caption in app.caption)

    app.selectbox[0].set_value("AAPL").run()
    app.selectbox[1].set_value(2023).run()
    assert not app.exception
    assert list(app.selectbox[2].options) == ["Item 1A"]
    assert app.main.status[0].label == "Selected heading"


def test_every_passage_has_chunk_id_edgar_link_and_distinct_type_rendering(monkeypatch):
    prose = passage("prose-chunk", text="Narrative text stays readable as prose.")
    table = passage(
        "table-chunk", content_type="table",
        text=("Consolidated statements\n\n"
              "| Year | 2024 |\n| --- | --- |\n| Revenue | $10 |"),
        url="https://www.sec.gov/Archives/table")
    table["table_caption"] = "Consolidated statements"
    app = browse(monkeypatch, [prose, table])

    panels = app.main.status
    assert [(panel.label, panel.icon) for panel in panels] == [
        ("Selected heading", ":material/article:"),
        ("Consolidated statements", ":material/table_chart:"),
        ("Stored text", ":material/code:"),
    ]
    assert {caption.value for caption in app.caption} >= {
        "Chunk ID: prose-chunk", "Chunk ID: table-chunk"}
    links = app.get("link_button")
    assert [link.url for link in links] == [prose["url"], table["url"]]
    assert all(link.label == "Open filing on EDGAR" for link in links)
    assert [text.value for text in app.text] == [prose["text"]]
    # The rendered table does not replace what retrieval sees: the exact
    # stored text stays on the page inside the table's own panel.
    assert [code.value for code in panels[1].code] == [table["text"]]
    rendered_tables = app.get("html")
    assert len(rendered_tables) == 1
    assert "<table>" in rendered_tables[0].value
    assert "Consolidated statements" in rendered_tables[0].value
    assert "Revenue" in rendered_tables[0].value
    assert [badge.value for badge in app.get("markdown")] == [
        ":blue-badge[:material/article: Prose passage]",
        ":orange-badge[:material/table_chart: Table passage]",
    ]


def test_table_renderer_preserves_multiple_headers_and_escapes_filing_text():
    html = _corpus_table_html(
        "Operating expenses (part 1 of 2)\n\n"
        "|  | 2025 | 2025 | Change |\n"
        "| Category | Amount | Share | Percent |\n"
        "| --- | --- | --- | --- |\n"
        "| Research & development | $34,550 | 10 | <8% |"
    )
    assert "<caption>Operating expenses (part 1 of 2)</caption>" in html
    assert html.count("<thead><tr>") == 1
    assert html.count("<th>") == 8
    assert "Research &amp; development" in html
    assert "&lt;8%" in html


def test_table_renderer_combines_repeated_years_with_financial_value_fragments():
    html = _corpus_table_html(
        "Management's Discussion and Analysis\n\n"
        "|  | 2025 | 2025 | 2025 | Change | Change | 2024 | 2024 | 2024 |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
        "| Research and development | $ | 34,550 |  | 10 | % | $ | 31,370 |  |\n"
        "| Percentage of total net sales | 8 |  | % |  |  | 8 |  | % |"
    )
    assert html.count("<th>2025</th>") == 1
    assert html.count("<th>Change</th>") == 1
    assert html.count("<th>2024</th>") == 1
    assert "<td>$34,550</td>" in html
    assert "<td>10%</td>" in html
    assert "<td>$31,370</td>" in html
    assert html.count("<td>8%</td>") == 2


def test_table_renderer_keeps_distinct_columns_under_a_repeated_header():
    # Amazon FY2021 10-K, Item 7 (0001018724-22-000005_part_ii_item_7_t007_01),
    # as stored: one year spans three measures, each split from its "$", and
    # the second row puts a figure in the first row's "$" column.
    year = "Year Ended December 31, 2020"
    html = _corpus_table_html(
        "Management's Discussion and Analysis (MD&A) (part 2 of 3)\n\n"
        f"|  | {year} | {year} | {year} | {year} | {year} | {year} |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| Net sales | $ | 386,064 | $ | -1,438 | $ | 384,626 |\n"
        "| Operating expenses |  | 363,165 | -989 |  |  | 362,176 |"
    )
    assert html.count(f"<th>{year}</th>") == 3
    assert "<td>$386,064</td><td>$-1,438</td><td>$384,626</td>" in html
    assert "<td>363,165</td><td>-989</td><td>362,176</td>" in html


def test_table_renderer_joins_a_percent_split_with_its_closing_bracket():
    # Intuit FY2021 10-K, Item 7: "(2%)" is stored as "-2" and "%)".
    html = _corpus_table_html(
        "(Dollars in millions)\n\n"
        "|  | 2020-2019 % Change | 2020-2019 % Change |\n"
        "| --- | --- | --- |\n"
        "| QuickBooks Online Accounting | 38 | % |\n"
        "| Desktop Services and Supplies | -2 | %) |"
    )
    assert html.count("<th>2020-2019 % Change</th>") == 1
    assert "<td>38%</td>" in html
    assert "<td>-2%)</td>" in html


def test_table_renderer_joins_a_value_split_under_a_blank_header():
    # Apple FY2021 10-K, Item 8 (0000320193-21-000105_part_ii_item_8_t002_00):
    # the year sits over the "$" and the amount is under a blank header.
    html = _corpus_table_html(
        "CONSOLIDATED STATEMENTS OF COMPREHENSIVE INCOME\n\n"
        "| Years ended | September 25, 2021 |  | September 26, 2020 |  |\n"
        "| --- | --- | --- | --- | --- |\n"
        "| Net income | $ | 94,680 | $ | 57,411 |\n"
        "| Other comprehensive income |  | 569 |  | 42 |"
    )
    assert html.count("<th>") == 3
    assert "<td>$94,680</td><td>$57,411</td>" in html
    assert "<td>569</td><td>42</td>" in html


def test_sticky_column_colours_follow_the_viewers_theme(monkeypatch):
    table = passage("themed", content_type="table",
                    text="Label\n\n| Year | 2024 |\n| --- | --- |\n| Revenue | $10 |")
    assert "--sec-table-base: #ffffff; --sec-table-ink: #31333f" in browse(
        monkeypatch, [table]).get("html")[0].value
    monkeypatch.setattr(components, "_dark_theme", lambda: True)
    assert "--sec-table-base: #0e1117; --sec-table-ink: #fafafa" in browse(
        monkeypatch, [table]).get("html")[0].value


def test_unrecognised_table_falls_back_to_the_stored_text(monkeypatch):
    table = passage("loose-table", content_type="table", text="Year | 2024\nRevenue | $10")
    app = browse(monkeypatch, [table])
    assert not app.get("html")
    assert [code.value for code in app.code] == [table["text"]]


def test_pagination_makes_a_large_item_fully_reachable(monkeypatch):
    rows = [passage(f"chunk-{number:03d}") for number in range(27)]
    app = browse(monkeypatch, rows)
    assert len(app.main.status) == 25
    assert list(app.selectbox[3].options) == ["Page 1 of 2", "Page 2 of 2"]
    assert any("Showing passages 1–25 of 27" in caption.value for caption in app.caption)

    app.selectbox[3].set_value(2).run()
    assert not app.exception
    assert [panel.label for panel in app.main.status] == [
        "Selected heading (Part 26 of 27)", "Selected heading (Part 27 of 27)"]
    assert {caption.value for caption in app.caption} >= {
        "Chunk ID: chunk-025", "Chunk ID: chunk-026"}
    assert any("Showing passages 26–27 of 27" in caption.value for caption in app.caption)


def test_empty_or_unreadable_corpus_is_explained_on_the_page(monkeypatch):
    app = browse(monkeypatch, [])
    assert "No processed passages were found" in app.warning[0].value
    assert not app.selectbox

    def unreadable():
        raise OSError("broken filing")

    monkeypatch.setattr(state, "corpus_passages", unreadable)
    app = AppTest.from_file(BROWSE, default_timeout=30).run()
    assert not app.exception
    assert app.error[0].value == "Could not read the processed filing corpus: broken filing"


def test_corpus_is_loaded_from_processed_passages_once(monkeypatch):
    calls = []
    expected = passage("only")
    monkeypatch.setattr(state, "iter_chunks", lambda: calls.append(True) or iter([expected]))
    state.corpus_passages.clear()
    try:
        assert state.corpus_passages() == (expected,)
        assert state.corpus_passages() == (expected,)
        assert calls == [True]
    finally:
        state.corpus_passages.clear()


def test_a_new_selection_returns_to_the_first_page(monkeypatch):
    rows = [passage(f"seven-{number:03d}") for number in range(27)]
    rows += [passage(f"eight-{number:03d}", item="8") for number in range(27)]
    app = browse(monkeypatch, rows)
    app.selectbox[3].set_value(2).run()
    app.selectbox[2].set_value("8").run()
    assert not app.exception
    assert app.selectbox[3].value == 1
    assert any(caption.value == "Chunk ID: eight-000" for caption in app.caption)


def test_a_passage_without_heading_or_safe_link_says_so(monkeypatch):
    row = passage("bare")
    row.update(heading="", url="javascript:alert(1)")
    app = browse(monkeypatch, [row])
    assert app.main.status[0].label == "Selected Item"
    assert not app.get("link_button")
    assert any(caption.value == "Filing link unavailable" for caption in app.caption)


def test_dollar_signs_in_a_heading_are_not_read_as_math(monkeypatch):
    row = passage("dollars")
    row["heading"] = "Losses on strategic investments, net$(277)$(239)"
    app = browse(monkeypatch, [row])
    assert app.main.status[0].label == (
        r"Losses on strategic investments, net\$(277)\$(239)")


def test_passages_without_a_scope_are_counted_not_hidden(monkeypatch):
    app = browse(monkeypatch, [passage("dated"), passage("undated", year=None)])
    assert any(caption.value == "1 of 2 passages have no company, fiscal year or Item "
               "and are not listed." for caption in app.caption)


def test_a_malformed_filing_is_explained_on_the_page(monkeypatch, tmp_path):
    (tmp_path / "AAPL").mkdir()
    (tmp_path / "AAPL" / "10-K.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(state, "corpus_passages", lambda: tuple(iter_chunks(tmp_path)))
    app = AppTest.from_file(BROWSE, default_timeout=30).run()
    assert not app.exception
    assert app.error[0].value.startswith("Could not read the processed filing corpus")


def test_an_empty_corpus_is_read_again_on_the_next_rerun(monkeypatch):
    reads = iter([(), (passage("arrived"),)])
    monkeypatch.setattr(state, "iter_chunks", lambda: next(reads))
    state.corpus_passages.clear()
    try:
        assert state.corpus_passages() == ()
        assert state.corpus_passages() == (passage("arrived"),)
    finally:
        state.corpus_passages.clear()


# --- #127: telling a page of panels apart without opening each one -----------

def table(chunk_id, text, *, caption="", heading="", **changes):
    """A table passage as the chunker writes one: a label line, then the grid."""
    row = passage(chunk_id, content_type="table", text=text, **changes)
    row.update(table_caption=caption, heading=heading)
    return row


STATEMENT = ("Financial Statements (part 2 of 3)\n\n"
             "| | 2025 | 2024 |\n| Total net sales | 391,035 | 383,285 |")


def test_an_uncaptioned_table_is_titled_from_its_own_label_line(monkeypatch):
    # 7,232 of the corpus's 10,615 tables have no caption and no heading, so
    # they all fell back to the Item title and read as the same entry.
    app = browse(monkeypatch, [table("t000", STATEMENT, item="8")])
    assert app.main.status[0].label == "Financial Statements (part 2 of 3)"


def test_same_named_fragmented_tables_use_one_unambiguous_part_count(monkeypatch):
    rows = [
        table("first-1", "Financial Statements (part 1 of 2)\n\n| First | 1 |"),
        table("first-2", "Financial Statements (part 2 of 2)\n\n| First | 2 |"),
        table("second-1", "Financial Statements (part 1 of 2)\n\n| Second | 1 |"),
        table("second-2", "Financial Statements (part 2 of 2)\n\n| Second | 2 |"),
    ]
    app = browse(monkeypatch, rows)
    assert [panel.label for panel in app.main.status] == [
        "Financial Statements (Part 1 of 4)",
        "Financial Statements (Part 2 of 4)",
        "Financial Statements (Part 3 of 4)",
        "Financial Statements (Part 4 of 4)",
    ]


def test_a_stored_caption_still_wins_over_the_label_line(monkeypatch):
    row = table("t001", STATEMENT, caption="Consolidated statements of operations", item="8")
    app = browse(monkeypatch, [row])
    assert app.main.status[0].label.startswith("Consolidated statements of operations")


def test_a_table_that_opens_with_its_grid_keeps_the_item_title(monkeypatch):
    app = browse(monkeypatch, [table("t002", "| | 2025 |\n| Net sales | 391,035 |", item="8")])
    assert app.main.status[0].label == "Selected Item"


def test_the_label_line_is_not_used_for_prose(monkeypatch):
    # A prose passage's first line is its text, not a label.
    row = passage("p000", text="The Company is subject to various legal proceedings.")
    row["heading"] = ""
    app = browse(monkeypatch, [row])
    assert app.main.status[0].label == "Selected Item"


def test_no_two_panels_in_one_selection_share_a_label(monkeypatch):
    # Apple's FY2025 Item 8 holds 64 tables, 61 of them titled the same way.
    rows = [table(f"t{number:03d}", STATEMENT, item="8") for number in range(64)]
    app = browse(monkeypatch, rows)
    labels = [panel.label for panel in app.main.status]
    assert len(labels) == 25 and len(set(labels)) == 25
    app.selectbox[3].set_value(3).run()
    assert not app.exception
    rest = [panel.label for panel in app.main.status]
    assert len(set(rest)) == len(rest)
    assert not set(rest) & set(labels)


def test_repeated_headers_are_numbered_without_chunk_ids(monkeypatch):
    rows = [passage("0000320193-25-000073_part_ii_item_8_026", item="8"),
            passage("0000320193-25-000073_part_ii_item_8_027", item="8")]
    for row in rows:
        row["accession_no"] = "0000320193-25-000073"
    app = browse(monkeypatch, rows)
    assert [panel.label for panel in app.main.status] == [
        "Selected heading (Part 1 of 2)",
        "Selected heading (Part 2 of 2)",
    ]
    assert {caption.value for caption in app.caption} >= {
        "Chunk ID: 0000320193-25-000073_part_ii_item_8_026",
        "Chunk ID: 0000320193-25-000073_part_ii_item_8_027",
    }


def test_panel_heading_does_not_repeat_the_selected_item(monkeypatch):
    rows = [passage("first"), passage("second")]
    heading = ("Item 7.\u00a0\u00a0\u00a0\u00a0Management's Discussion and Analysis of "
               "Financial Condition and Results of Operations")
    for row in rows:
        row["heading"] = heading
    app = browse(monkeypatch, rows)
    assert [panel.label for panel in app.main.status] == [
        "Management's Discussion and Analysis of Financial Condition and Results of "
        "Operations (Part 1 of 2)",
        "Management's Discussion and Analysis of Financial Condition and Results of "
        "Operations (Part 2 of 2)",
    ]


def test_item_prefix_is_removed_without_a_space_and_never_from_a_longer_item(monkeypatch):
    rows = [passage("glued", item="1"), passage("longer", item="1"),
            passage("bare", item="1"), passage("colon", item="1"),
            passage("dash", item="1")]
    rows[0]["heading"] = "Item\u00a01.Business"
    rows[1]["heading"] = "Item 1A. Risk Factors"
    rows[2]["heading"] = "Item 1."
    rows[3]["heading"] = "Item 1: Overview"
    rows[4]["heading"] = "ITEM 1 - BUSINESS"
    app = browse(monkeypatch, rows)
    assert [panel.label for panel in app.main.status] == [
        "Business", "Item 1A. Risk Factors", "Item 1.", "Overview", "BUSINESS"]


def test_each_repeated_header_has_its_own_part_count(monkeypatch):
    rows = [passage("a"), passage("b"), passage("c"), passage("d")]
    rows[2]["heading"] = "Another heading"
    rows[3]["heading"] = "Another heading"
    app = browse(monkeypatch, rows)
    assert [panel.label for panel in app.main.status] == [
        "Selected heading (Part 1 of 2)",
        "Selected heading (Part 2 of 2)",
        "Another heading (Part 1 of 2)",
        "Another heading (Part 2 of 2)",
    ]


def test_a_passage_with_no_chunk_id_is_still_labelled(monkeypatch):
    row = passage("", item="8")
    row["heading"] = ""
    row["title"] = ""
    app = browse(monkeypatch, [row])
    assert app.main.status[0].label == "Prose passage"
