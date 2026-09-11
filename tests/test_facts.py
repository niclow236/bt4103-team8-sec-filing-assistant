"""The facts store: which rows describe their filing's own year, and refresh.

EDGAR is replaced by fakes throughout, so these run offline. The fakes carry
only the attributes ``facts._rows_for`` reads.
"""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest

import edgar.entity
from src.retrieval import facts


def frame(*rows):
    return pd.DataFrame(
        [{"period_start": start, "period_end": end, "period_of_report": report}
         for start, end, report in rows]
    )


def test_annual_and_instant_facts_are_the_current_year():
    marked = facts.mark_current_year(frame(
        (date(2024, 1, 1), date(2024, 12, 31), "2024-12-31"),   # calendar year
        (date(2023, 9, 30), date(2024, 9, 28), "2024-09-28"),   # 52-week year, 364 days
        (date(2022, 9, 25), date(2023, 9, 30), "2023-09-30"),   # 53-week year, 371 days
        (None, date(2024, 12, 31), "2024-12-31"),               # a balance at year end
    ))
    assert marked["is_current_year"].tolist() == [True, True, True, True]


def test_a_quarter_ending_on_the_year_end_is_not_the_year():
    marked = facts.mark_current_year(frame(
        (date(2022, 10, 1), date(2022, 12, 31), "2022-12-31"),  # Amazon's Q4 impairment
        (date(2022, 1, 1), date(2022, 12, 31), "2022-12-31"),   # the annual figure
    ))
    assert marked["is_current_year"].tolist() == [False, True]


def test_comparatives_are_not_the_current_year():
    marked = facts.mark_current_year(frame(
        (date(2022, 1, 1), date(2022, 12, 31), "2024-12-31"),
        (None, date(2023, 12, 31), "2024-12-31"),
    ))
    assert marked["is_current_year"].tolist() == [False, False]


# --- refresh ----------------------------------------------------------------

FILINGS = {
    "acc-a": SimpleNamespace(accession_no="acc-a", ticker="AAA", cik=1, company="Alpha",
                             form="10-K", filing_date="2025-02-01", period_of_report="2024-12-31"),
    "acc-b": SimpleNamespace(accession_no="acc-b", ticker="BBB", cik=2, company="Beta",
                             form="10-K", filing_date="2025-02-01", period_of_report="2024-12-31"),
}


def fact(accession, concept, value, start, end):
    return SimpleNamespace(
        taxonomy="us-gaap", accession=accession, concept=concept, label=concept,
        numeric_value=value, value=value, unit="USD", scale=None, fiscal_year=2024,
        fiscal_period="FY", period_start=start, period_end=end,
        period_type="duration" if start else "instant", statement_type="", is_audited=True,
    )


def entity(*items):
    return SimpleNamespace(get_all_facts=lambda: iter(items))


ANNUAL = (date(2024, 1, 1), date(2024, 12, 31))
Q4 = (date(2024, 10, 1), date(2024, 12, 31))


@pytest.fixture
def offline(monkeypatch):
    """No identity, no manifest on disk, and EDGAR answered by a dict."""
    answers = {
        1: entity(fact("acc-a", "Revenue", 100.0, *ANNUAL), fact("acc-a", "Revenue", 30.0, *Q4)),
        2: entity(fact("acc-b", "Revenue", 200.0, *ANNUAL)),
    }
    monkeypatch.setattr(facts, "configure_edgar", lambda: None)
    monkeypatch.setattr(facts, "corpus_filings", lambda: dict(FILINGS))

    def fetch(cik, *args, **kwargs):
        answer = answers[cik]
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(edgar.entity, "get_company_facts", fetch)
    return answers


def rows(path, ticker):
    stored = pd.read_parquet(path)
    return stored[stored["ticker"] == ticker].reset_index(drop=True)


def test_build_marks_the_annual_row_and_keeps_the_q4_row(tmp_path, offline):
    store = tmp_path / "facts.parquet"
    facts.build(facts_file=store)
    alpha = rows(store, "AAA").sort_values("value")
    assert alpha["value"].tolist() == [30.0, 100.0]
    assert alpha["is_current_year"].tolist() == [False, True]


def test_an_annual_and_q4_figure_of_equal_value_are_both_kept(tmp_path, offline):
    """The duplicate key includes period_start, or one of the two is dropped."""
    offline[1] = entity(fact("acc-a", "Impairment", 101.0, *ANNUAL),
                        fact("acc-a", "Impairment", 101.0, *Q4))
    store = tmp_path / "facts.parquet"
    facts.build(facts_file=store)
    assert len(rows(store, "AAA")) == 2


def test_refreshing_one_company_keeps_the_others(tmp_path, offline):
    """Finding 3: --tickers AAA --refresh must not delete BBB."""
    store = tmp_path / "facts.parquet"
    facts.build(facts_file=store)
    beta = rows(store, "BBB")
    offline[1] = entity(fact("acc-a", "Revenue", 111.0, *ANNUAL))
    facts.build(tickers=["AAA"], refresh=True, facts_file=store)
    assert rows(store, "AAA")["value"].tolist() == [111.0]
    pd.testing.assert_frame_equal(rows(store, "BBB"), beta)


def test_a_failed_request_keeps_what_was_stored(tmp_path, offline):
    store = tmp_path / "facts.parquet"
    facts.build(facts_file=store)
    beta = rows(store, "BBB")
    offline[2] = ConnectionError("simulated outage")
    facts.build(refresh=True, facts_file=store)
    pd.testing.assert_frame_equal(rows(store, "BBB"), beta)


def test_an_empty_answer_keeps_what_was_stored(tmp_path, offline):
    store = tmp_path / "facts.parquet"
    facts.build(facts_file=store)
    beta = rows(store, "BBB")
    offline[2] = entity()
    facts.build(refresh=True, facts_file=store)
    pd.testing.assert_frame_equal(rows(store, "BBB"), beta)


def test_facts_from_a_filing_that_left_the_manifest_are_dropped(tmp_path, offline, monkeypatch):
    store = tmp_path / "facts.parquet"
    facts.build(facts_file=store)
    monkeypatch.setattr(facts, "corpus_filings", lambda: {"acc-a": FILINGS["acc-a"]})
    facts.build(facts_file=store)
    assert set(pd.read_parquet(store)["accession"]) == {"acc-a"}


def test_load_recomputes_the_flag_for_an_older_store(tmp_path, offline):
    store = tmp_path / "facts.parquet"
    facts.build(facts_file=store)
    stale = pd.read_parquet(store)
    stale["is_current_year"] = True        # what the old rule wrote for the Q4 row
    stale.to_parquet(store, index=False)
    alpha = facts.load_facts(store).query("ticker == 'AAA'").sort_values("value")
    assert alpha["is_current_year"].tolist() == [False, True]
