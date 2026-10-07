"""Issue #37: rendered evidence and real Streamlit widget reruns, offline."""

from dataclasses import replace
from html import escape

from bs4 import BeautifulSoup
import pytest
from streamlit.testing.v1 import AppTest

from src.app.components import answer_card_html
from src.rag.citations import render_citation
from src.rag.constants import ABSTAIN_PHRASE
from src.rag.records import Citation, VerificationCheck, VerificationResult
from tests.sample_answers import sample_answer


def soup(answer=None, key="answer"):
    return BeautifulSoup(answer_card_html(answer or sample_answer("Example?"), key=key), "html.parser")


def test_inline_links_target_the_correct_expandable_passage():
    answer = sample_answer("Example?")
    page = soup(answer)
    links = page.select("a[data-citation]")
    assert len(links) == 2  # repeated marker, one evidence panel
    for link in links:
        panel = page.select_one(link["href"])
        assert panel.name == "details"
        assert panel.blockquote.text == answer.passages[0].text
        assert panel.summary.text == render_citation(answer.citations[0], answer.passages)
        assert panel.select_one('a[target="_blank"]')["href"] == answer.passages[0].url
    assert len(page.select("details")) == 1
    assert "target.open = true" in page.script.text


def test_unresolved_markers_are_visible_but_never_fake_links():
    page = soup()
    unresolved = page.select_one(".sec-unresolved")
    assert unresolved.text == "[9] (unresolved)"
    assert unresolved.name != "a"
    assert "Unresolved citation: no source passage available" in page.text


def test_per_claim_status_distinguishes_support_mismatch_and_uncited_claims():
    rows = soup().select(".sec-claim")
    assert "sec-supported" in rows[0]["class"]
    assert "sec-warning" in rows[1]["class"]
    assert "sec-mismatch" in rows[2]["class"]
    assert "sec-warning" in rows[3]["class"]


def test_resolved_citations_do_not_claim_fact_verification():
    page = soup(replace(sample_answer("q"), verification=None))
    assert "Factual verification has not been run" in page.text
    assert not page.select(".sec-supported")


def test_unverified_paraphrase_is_not_an_ask_page_warning_but_global_checks_remain_visible():
    answer = replace(sample_answer("q"), verification=VerificationResult("factual", (
        VerificationCheck("groundedness", "unverified", 0, "Claim", "No semantic check."),
        VerificationCheck("output", "unverified", None, "Whole answer", "Output needs review."),
    )))
    page = soup(answer)
    claim = page.select(".sec-claim")[0]
    assert "sec-warning" not in claim["class"]
    assert claim.select_one(".sec-label") is None
    assert "No semantic check." not in page.text
    assert "Output needs review." in page.text


def test_supported_evidence_outranks_unverified_groundedness_but_not_citation_defects():
    answer = replace(sample_answer("q"), verification=VerificationResult("numeric", (
        VerificationCheck("fact", "supported", 0, "Revenue", "The annual fact agrees."),
        VerificationCheck("groundedness", "unverified", 0, "Revenue",
                          "No semantic entailment check was run."),
        VerificationCheck("fact", "supported", 1, "Unavailable claim",
                          "A structured fact agrees."),
    )))
    page = soup(answer)
    rows = page.select(".sec-claim")
    assert "sec-supported" in rows[0]["class"]
    assert "Checked against cited evidence" in rows[0].text
    assert "No semantic entailment check was run." not in page.text
    assert "sec-warning" in rows[1]["class"]
    assert "sec-supported" not in rows[1]["class"]


def test_word_for_word_sentence_with_only_a_groundedness_check_is_supported():
    answer = replace(sample_answer("q"), verification=VerificationResult("factual", (
        VerificationCheck("groundedness", "supported", 0, "Revenue",
                          "Extractive wording found in a cited passage."),
    )))
    row = soup(answer).select(".sec-claim")[0]
    assert "sec-supported" in row["class"]
    assert "Checked against cited evidence" in row.text


@pytest.mark.parametrize("url", ["javascript:alert(1)", "data:text/html,bad", "https://", "https://[bad"])
def test_unsafe_or_missing_filing_links_are_not_clickable(url):
    answer = sample_answer("q")
    answer = replace(answer, passages=(replace(answer.passages[0], url=url),))
    page = soup(answer)
    assert not page.select('a[target="_blank"]')
    assert "Filing link unavailable" in page.text


def test_answer_metadata_and_passage_are_escaped_as_text():
    attack = '<script>alert("bad")</script><img src=x onerror=alert(1)>'
    answer = sample_answer(attack)
    answer = replace(answer, passages=(replace(answer.passages[0], text=attack, company=attack),),
                     sentences=(replace(answer.sentences[0], raw_text=attack + " [1]"),))
    page = soup(answer, key=attack)
    assert not page.select("img")
    assert len(page.select("script")) == 1  # only the fixed click handler
    assert page.blockquote.text == attack
    assert page.h3.text == attack
    assert escape(attack) in answer_card_html(answer)


def test_distinct_card_keys_do_not_collide():
    first, second = soup(key="first"), soup(key="second")
    assert first.details["id"] != second.details["id"]
    assert first.select_one("a[data-citation]")["href"] != second.select_one("a[data-citation]")["href"]


def test_citation_uses_prompt_position_not_retrieval_rank():
    answer = sample_answer("q")
    first = replace(answer.passages[0], rank=8, chunk_id="first", text="First source")
    second = replace(first, rank=1, chunk_id="second", text="Second source")
    citation = Citation(2, "second", True)
    sentence = replace(answer.sentences[0], raw_text="Claim. [2]", text="Claim. [2]",
                       citations=(citation,))
    answer = replace(answer, citations=(citation,), passages=(first, second), sentences=(sentence,))
    page = soup(answer)
    assert page.details.blockquote.text == "Second source"
    assert len(page.select("details")) == 1


@pytest.mark.parametrize("marker", ["[0]", "[-1]"])
def test_invalid_markers_cannot_become_links(marker):
    answer = sample_answer("q")
    sentence = replace(answer.sentences[0], raw_text="Claim. " + marker, citations=(),
                       invalid_markers=(marker,))
    page = soup(replace(answer, sentences=(sentence,), citations=()))
    assert not page.select("a[data-citation]")
    assert page.select_one(".sec-unresolved").text == marker + " (unresolved)"


def test_missing_metadata_is_explicit_and_passage_text_is_not_trimmed():
    answer = sample_answer("q")
    passage = replace(answer.passages[0], company="", ticker="", fiscal_year=None,
                      part=None, item=None, cik=None, filing_date=None, title="", form=None,
                      text="  Column A | Column B\n  1 | 2\n")
    page = soup(replace(answer, passages=(passage,)))
    assert "fiscal year unknown" in page.summary.text
    assert "CIK unknown" in page.summary.text
    assert page.blockquote.text == passage.text


def test_truncated_and_legacy_answers_are_conservatively_flagged():
    answer = replace(sample_answer("q"), sentences=(), truncated=True, parse_error="cut off")
    page = soup(answer)
    assert "Incomplete or malformed answer" in page.text
    assert all("sec-warning" in row["class"] for row in page.select(".sec-claim"))


def test_an_abstention_is_never_laid_out_as_a_card_of_claims():
    # Its text is the abstain sentence, which the card would otherwise show as
    # one uncited claim that needs review. ``answer_card`` states it instead;
    # see test_an_abstention_is_stated_with_its_reason_and_what_to_try.
    answer = replace(sample_answer("q"), text=ABSTAIN_PHRASE, abstained=True,
                     abstention_reason="filters_excluded_all", sentences=(), citations=())
    with pytest.raises(ValueError, match="abstention_notice"):
        answer_card_html(answer)


SIDEBAR_APP = '''
import streamlit as st
from src.app.components import filter_sidebar
q = st.text_input("Question", "Compare Apple and Microsoft in FY2023 and FY2024, Items 7 and 8")
st.session_state["result"] = filter_sidebar(q)
'''


def sidebar():
    app = AppTest.from_string(SIDEBAR_APP).run(timeout=30)
    assert not app.exception
    return app


def test_sidebar_says_what_it_searches_in_words():
    # It used to print the Query's own field names: "ticker: AAPL; fiscal_year: 2024".
    app = sidebar()
    assert app.sidebar.caption[-1].value == (
        "Searching: AAPL, MSFT; FY2023, FY2024; Item 7, Item 8")
    for control in app.sidebar.multiselect:
        control.set_value([])
    app.run()
    assert app.sidebar.caption[-1].value == (
        "Searching: every company; every fiscal year; every Item")


def test_sidebar_extracts_filters_and_returns_a_retrieval_query():
    app = sidebar()
    query = app.session_state["result"]
    assert query.tickers == ("AAPL", "MSFT")
    assert query.fiscal_years == (2023, 2024)
    assert query.items == ("7", "8")
    assert app.sidebar.multiselect[0].value == ["AAPL", "MSFT"]
    assert app.sidebar.multiselect[1].value == [2023, 2024]
    assert app.sidebar.multiselect[2].value == ["7", "8"]


def test_manual_filters_and_clearing_survive_reruns_and_can_be_reset():
    app = sidebar()
    app.sidebar.multiselect[0].set_value(["META"]).run()
    app.sidebar.multiselect[1].set_value([2025]).run()
    app.sidebar.multiselect[2].set_value(["1A"]).run()
    assert app.session_state["result"].filters == {
        "ticker": ["META"], "fiscal_year": [2025], "item": ["1A"],
    }
    for control in app.sidebar.multiselect:
        control.set_value([])
    app.run()
    assert app.session_state["result"].filters == {}
    app.run()
    assert app.session_state["result"].filters == {}
    app.sidebar.button[0].click().run()
    assert app.session_state["result"].items == ("7", "8")
    assert app.session_state["result"].tickers == ("AAPL", "MSFT")
    assert not app.exception


def test_new_question_replaces_old_filters_and_blank_clears_them():
    app = sidebar()
    app.text_input[0].set_value("Meta FY2025 Item 1a risks?").run()
    assert app.session_state["result"].filters == {
        "ticker": ["META"], "fiscal_year": [2025], "item": ["1A"],
    }
    app.text_input[0].set_value("").run()
    assert app.session_state["result"].filters == {}
    assert not app.exception


def test_out_of_scope_mentions_warn_and_unknown_item_is_not_silently_dropped():
    app = sidebar()
    app.text_input[0].set_value("NVIDIA FY2015 Item 99").run()
    assert len(app.sidebar.warning) == 2
    assert app.session_state["result"].items == ("99",)
    assert "99" in app.sidebar.multiselect[2].value
    assert not app.exception


def test_sidebar_reads_an_unparsed_question_with_the_facts_store_s_labels(monkeypatch):
    # The Query it returns is searched as it stands. Read without the labels,
    # a question naming a line item by its label got no lean toward tables
    # from the sidebar, while answer_question read it as asking for a figure.
    import src.app.components as components

    read = []
    parse = components.parse_question
    monkeypatch.setattr(components, "parse_question",
                        lambda question, **options: read.append(options) or parse(
                            question, **options))
    sidebar()
    assert read and all("facts_file" not in options for options in read)


def test_sidebar_item_reaches_answer_retrieval_without_facts_bypass(monkeypatch):
    from src.rag.answer import answer_question
    from src.rag.query import parse_question

    app = sidebar()
    app.text_input[0].set_value("What was Apple's revenue in FY2024?").run()
    app.sidebar.multiselect[2].set_value(["7"]).run()
    query = app.session_state["result"]

    def forbidden(*args, **kwargs):
        pytest.fail("Facts shortcut must not bypass the selected Item")

    monkeypatch.setattr("src.rag.answer.answer_from_facts", forbidden)
    monkeypatch.setattr("src.rag.answer.generate", forbidden)

    class EmptyRetriever:
        def search(self, requested):
            assert requested.filters == query.filters
            return []

    answer = answer_question(query.text, EmptyRetriever(), sample_answer("q").config,
                             query=query, parsed=parse_question(query.text, facts_file=None))
    assert answer.abstained


# --- the Ask page's pieces (#38) -------------------------------------------------

def test_card_can_leave_the_question_to_the_page_above_it():
    assert soup().select("h3")
    page = BeautifulSoup(
        answer_card_html(sample_answer("Example?"), show_question=False), "html.parser")
    assert not page.select("h3")
    assert page.select(".sec-claim")


def test_trace_rows_follow_prompt_order_and_mark_what_was_cited():
    from src.app.components import trace_rows
    from src.retrieval.records import RetrievedPassage

    first = sample_answer("q").passages[0]
    second = RetrievedPassage(
        chunk_id="other", text="Other text.", score=0.5, rank=1, retriever="hybrid",
        ticker="MSFT", company="Microsoft", fiscal_year=2023, item="1A", title="Risk Factors",
        url="javascript:alert(1)", content_type="table", sources=("bm25", "dense"))
    answer = replace(sample_answer("q"), passages=(first, second))
    rows = trace_rows(answer)
    assert [row["Source"] for row in rows] == ["[1]", "[2]"]
    assert [row["Cited"] for row in rows] == [True, False]
    assert rows[1]["Found by"] == "bm25, dense" and rows[0]["Found by"] == "test"
    assert rows[1]["Type"] == "table" and rows[1]["Passage"] == "Other text."
    # Only an http(s) filing link is offered as one, as on the card.
    assert rows[0]["Filing"] == first.url and rows[1]["Filing"] is None


ABSTAINED_APP = '''
from dataclasses import replace
from src.app.components import answer_card
from src.rag.constants import ABSTAIN_PHRASE
from tests.sample_answers import sample_answer
answer_card(replace(sample_answer("q"), text=ABSTAIN_PHRASE, abstained=True,
                    abstention_reason="filters_excluded_all", sentences=(), citations=(),
                    passages=()))
'''


def test_an_abstention_is_stated_with_its_reason_and_what_to_try():
    app = AppTest.from_string(ABSTAINED_APP).run(timeout=30)
    assert not app.exception
    assert app.warning[0].value == (
        "**No answer from the filings.** The selected filters excluded every indexed passage.")
    assert "Clear a company, fiscal year or Item" in app.caption[0].value
    assert not app.get("html")
    # One built with no reason, which the pipeline never makes, says so.
    unexplained = ABSTAINED_APP.replace('abstention_reason="filters_excluded_all", ', "")
    app = AppTest.from_string(unexplained).run(timeout=30)
    assert not app.exception
    assert app.warning[0].value == (
        "**No answer from the filings.** No reason was recorded for this abstention.")
    assert not app.caption
