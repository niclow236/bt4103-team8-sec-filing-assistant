"""Issue #40: the Browse page exposes every stored passage without a model."""

from streamlit.testing.v1 import AppTest

from src.app import state
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
        text="Year | 2024\nRevenue | $10", url="https://www.sec.gov/Archives/table")
    table["table_caption"] = "Consolidated statements"
    app = browse(monkeypatch, [prose, table])

    panels = app.main.status
    assert [(panel.label, panel.icon) for panel in panels] == [
        ("Selected heading", ":material/article:"),
        ("Consolidated statements", ":material/table_chart:"),
    ]
    assert {caption.value for caption in app.caption} >= {
        "Chunk ID: prose-chunk", "Chunk ID: table-chunk"}
    links = app.get("link_button")
    assert [link.url for link in links] == [prose["url"], table["url"]]
    assert all(link.label == "Open filing on EDGAR" for link in links)
    assert [text.value for text in app.text] == [prose["text"]]
    assert [code.value for code in app.code] == [table["text"]]
    assert [badge.value for badge in app.get("markdown")] == [
        ":blue-badge[:material/article: Prose passage]",
        ":orange-badge[:material/table_chart: Table passage]",
    ]


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
