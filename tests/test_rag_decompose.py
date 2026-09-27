"""Issue #35: a question about several filings is searched once per filing.

The retriever here is a stub over a small corpus of passages, but it honours
the Query's ticker and fiscal-year filters and ranks the way a real one does --
by how well the passage's own text matches the question's words. That matters:
the failure decomposition exists to fix only appears when one filing can
out-rank another for the whole budget, so a stub that returned a fixed list
would pass whatever this module did.
"""

import pytest

from src.rag.answer import answer_question
from src.rag.decompose import (
    Decomposition,
    SubQuestion,
    decompose,
    label_for,
    merge,
    search_decomposed,
)
from src.retrieval.records import Query, RetrievedPassage


def _passage(ticker, year, index, text=None, score=1.0):
    """One passage of one filing. chunk_id carries the filing, as the chunker's does."""
    return RetrievedPassage(
        chunk_id=f"{ticker}-{year}-{index}",
        text=text if text is not None else f"{ticker} {year} risk disclosure paragraph {index}.",
        score=score, rank=1, retriever="hybrid", ticker=ticker, company=f"{ticker} Inc.",
        fiscal_year=year, item="1A", title="Risk Factors",
        url=f"https://example.test/{ticker}/{year}",
    )


class FilterRespectingRetriever:
    """Applies the Query's ticker and year filters, then ranks by word overlap.

    Enough of a retriever to show the problem: given one query admitting two
    filings, the filing whose text repeats the question's words takes the top
    of the list, and with a budget of k the other can be shut out entirely.
    """

    name = "stub"

    def __init__(self, *passages):
        self.corpus = list(passages)
        self.queries = []

    def search(self, query, k=None):
        self.queries.append(query)
        wanted = k if k is not None else query.top_k
        words = set(query.text.lower().split())
        admitted = [
            p for p in self.corpus
            if (not query.tickers or p.ticker in query.tickers)
            and (not query.fiscal_years or p.fiscal_year in query.fiscal_years)
        ]
        scored = sorted(
            admitted,
            key=lambda p: (-len(words & set(p.text.lower().split())), p.chunk_id),
        )
        from dataclasses import replace
        return [replace(p, rank=rank, score=1.0 / rank)
                for rank, p in enumerate(scored[:wanted], start=1)]


def _query(text="How did AI risk change for AAPL and MSFT?", **changes):
    fields = dict(text=text, tickers=("AAPL", "MSFT"), fiscal_years=(2024,), top_k=4)
    return Query(**(fields | changes))


# --- what splits, and what does not ---------------------------------------------

def test_a_question_about_one_filing_does_not_split():
    assert decompose(_query(tickers=("AAPL",), fiscal_years=(2024,))) == ()
    assert decompose(_query(tickers=(), fiscal_years=())) == ()


def test_two_companies_split_by_company_keeping_the_years():
    subs = decompose(_query(tickers=("AAPL", "MSFT"), fiscal_years=(2024,)))
    assert [sub.label for sub in subs] == ["AAPL FY2024", "MSFT FY2024"]
    assert [sub.tickers for sub in subs] == [("AAPL",), ("MSFT",)]


def test_two_years_of_one_company_split_by_year():
    subs = decompose(_query(tickers=("AAPL",), fiscal_years=(2023, 2024)))
    assert [sub.label for sub in subs] == ["AAPL FY2023", "AAPL FY2024"]
    assert [sub.fiscal_years for sub in subs] == [(2023,), (2024,)]


def test_a_change_over_time_between_two_companies_splits_on_the_pair():
    # The question in the issue: two companies, two years, four filings.
    subs = decompose(_query(tickers=("AAPL", "MSFT"), fiscal_years=(2023, 2024)))
    assert [sub.label for sub in subs] == [
        "AAPL FY2023", "AAPL FY2024", "MSFT FY2023", "MSFT FY2024",
    ]


def test_a_split_too_wide_for_the_budget_falls_back_to_companies():
    # Two companies over five years is ten filings; the passage budget cannot
    # say anything with a tenth each, so the years stay whole.
    subs = decompose(_query(tickers=("AAPL", "MSFT"),
                            fiscal_years=(2021, 2022, 2023, 2024, 2025)))
    assert [sub.label for sub in subs] == [
        "AAPL FY2021 FY2022 FY2023 FY2024 FY2025",
        "MSFT FY2021 FY2022 FY2023 FY2024 FY2025",
    ]
    assert all(sub.fiscal_years == (2021, 2022, 2023, 2024, 2025) for sub in subs)


def test_many_years_of_one_company_still_split_by_year():
    # One company over five years is five sub-questions, which is more than the
    # limit on pairs; the limit is on the cross product, not on one axis.
    subs = decompose(_query(tickers=("AAPL",), fiscal_years=(2021, 2022, 2023)))
    assert [sub.label for sub in subs] == ["AAPL FY2021", "AAPL FY2022", "AAPL FY2023"]


def test_the_limit_on_pairs_is_configurable():
    # top_k has to cover the pairs too, since no sub-question may end up with
    # no passage; six pairs need a budget of six.
    six = _query(tickers=("AAPL", "MSFT"), fiscal_years=(2022, 2023, 2024), top_k=6)
    assert len(decompose(six, limit=6)) == 6
    assert len(decompose(six, limit=4)) == 2


def test_a_sub_question_keeps_every_other_setting_of_the_query():
    base = _query(items=("1A",), content_type="prose", table_boost=2.0,
                  keyword_text="ai risk", wants_figures=True, top_k=6)
    for sub in decompose(base):
        assert sub.query.text == base.text
        assert sub.query.keyword_text == base.keyword_text
        assert sub.query.items == base.items
        assert sub.query.content_type == base.content_type
        assert sub.query.table_boost == base.table_boost
        assert sub.query.wants_figures is True
        assert sub.query.top_k == base.top_k


@pytest.mark.parametrize("tickers, years, expected", [
    (("AAPL",), (2024,), "AAPL FY2024"),
    (("AAPL",), (), "AAPL"),
    ((), (2024,), "FY2024"),
    (("AAPL",), (2023, 2024), "AAPL FY2023 FY2024"),
])
def test_a_label_reads_as_the_filters_it_stands_for(tickers, years, expected):
    assert label_for(tickers, years) == expected


# --- merging ---------------------------------------------------------------------

def _sub(label, ticker, year):
    return SubQuestion(label=label, query=_query(tickers=(ticker,), fiscal_years=(year,)))


def test_the_budget_is_shared_round_by_round_not_won():
    a, m = _sub("AAPL FY2024", "AAPL", 2024), _sub("MSFT FY2024", "MSFT", 2024)
    passages, provenance = merge(
        [(a, [_passage("AAPL", 2024, i) for i in range(4)]),
         (m, [_passage("MSFT", 2024, i) for i in range(4)])],
        wanted=4,
    )
    assert [p.chunk_id for p in passages] == [
        "AAPL-2024-0", "MSFT-2024-0", "AAPL-2024-1", "MSFT-2024-1",
    ]
    assert set(provenance.values()) == {"AAPL FY2024", "MSFT FY2024"}


def test_the_merged_passages_are_reranked_from_one():
    a, m = _sub("AAPL FY2024", "AAPL", 2024), _sub("MSFT FY2024", "MSFT", 2024)
    passages, _ = merge(
        [(a, [_passage("AAPL", 2024, i) for i in range(3)]),
         (m, [_passage("MSFT", 2024, i) for i in range(3)])],
        wanted=4,
    )
    assert [p.rank for p in passages] == [1, 2, 3, 4]


def test_a_sub_question_that_found_less_does_not_shrink_the_others():
    a, m = _sub("AAPL FY2024", "AAPL", 2024), _sub("MSFT FY2024", "MSFT", 2024)
    passages, provenance = merge(
        [(a, [_passage("AAPL", 2024, i) for i in range(4)]),
         (m, [_passage("MSFT", 2024, 0)])],
        wanted=4,
    )
    assert len(passages) == 4
    assert sum(1 for label in provenance.values() if label == "AAPL FY2024") == 3


def test_merging_stops_at_the_budget():
    a, m = _sub("AAPL FY2024", "AAPL", 2024), _sub("MSFT FY2024", "MSFT", 2024)
    passages, _ = merge(
        [(a, [_passage("AAPL", 2024, i) for i in range(8)]),
         (m, [_passage("MSFT", 2024, i) for i in range(8)])],
        wanted=3,
    )
    assert len(passages) == 3


def test_nothing_found_merges_to_nothing():
    a = _sub("AAPL FY2024", "AAPL", 2024)
    assert merge([(a, [])], wanted=4) == ((), {})


# --- searching ---------------------------------------------------------------------

def test_a_single_filing_question_is_not_decomposed():
    retriever = FilterRespectingRetriever(_passage("AAPL", 2024, 0))
    assert search_decomposed(_query(tickers=("AAPL",), fiscal_years=(2024,)), retriever) is None
    assert retriever.queries == []


def test_every_filing_is_searched_once_under_its_own_filters():
    retriever = FilterRespectingRetriever(
        *[_passage("AAPL", 2024, i) for i in range(3)],
        *[_passage("MSFT", 2024, i) for i in range(3)],
    )
    result = search_decomposed(_query(), retriever)
    assert isinstance(result, Decomposition)
    assert len(retriever.queries) == 2
    assert [q.tickers for q in retriever.queries] == [("AAPL",), ("MSFT",)]
    assert all(len(q.fiscal_years) == 1 for q in retriever.queries)


def test_decomposition_rescues_the_filing_a_single_search_shuts_out():
    # MSFT's passages happen to repeat the question's words; AAPL's do not. One
    # search over both filings hands MSFT the whole budget, which is a
    # comparison with nothing to compare.
    corpus = (
        *[_passage("AAPL", 2024, i, text=f"Artificial intelligence paragraph {i}.")
          for i in range(4)],
        *[_passage("MSFT", 2024, i, text="How did AI risk change for AAPL and MSFT? " + str(i))
          for i in range(4)],
    )
    retriever = FilterRespectingRetriever(*corpus)
    single = retriever.search(_query())
    assert {p.ticker for p in single} == {"MSFT"}

    decomposed = search_decomposed(_query(), FilterRespectingRetriever(*corpus))
    assert {p.ticker for p in decomposed.passages} == {"AAPL", "MSFT"}
    assert len(decomposed.passages) == 4


def test_provenance_names_the_sub_question_that_found_each_passage():
    retriever = FilterRespectingRetriever(
        *[_passage("AAPL", 2023, i) for i in range(2)],
        *[_passage("AAPL", 2024, i) for i in range(2)],
    )
    result = search_decomposed(_query(tickers=("AAPL",), fiscal_years=(2023, 2024)), retriever)
    for passage in result.passages:
        label = result.provenance[passage.chunk_id]
        assert label == f"AAPL FY{passage.fiscal_year}"


def test_the_evidence_can_be_grouped_back_under_each_filing():
    retriever = FilterRespectingRetriever(
        *[_passage("AAPL", 2024, i) for i in range(2)],
        *[_passage("MSFT", 2024, i) for i in range(2)],
    )
    grouped = search_decomposed(_query(), retriever).by_sub_question()
    assert set(grouped) == {"AAPL FY2024", "MSFT FY2024"}
    assert all(p.ticker == "AAPL" for p in grouped["AAPL FY2024"])
    assert all(p.ticker == "MSFT" for p in grouped["MSFT FY2024"])


def test_a_filing_the_corpus_has_nothing_for_is_still_named():
    # A comparison resting on one side only is something the reader has to see.
    retriever = FilterRespectingRetriever(*[_passage("AAPL", 2024, i) for i in range(2)])
    grouped = search_decomposed(_query(), retriever).by_sub_question()
    assert grouped["MSFT FY2024"] == ()
    assert len(grouped["AAPL FY2024"]) == 2


def test_each_sub_search_asks_for_the_whole_budget():
    # So the merge chooses from a full ranking per filing, and a sub-question
    # whose neighbours found little can contribute more than its share.
    retriever = FilterRespectingRetriever(*[_passage("AAPL", 2024, i) for i in range(6)])
    search_decomposed(_query(top_k=4), retriever)
    assert all(q.top_k == 4 for q in retriever.queries)


# --- through answer_question -------------------------------------------------------

def _answering(monkeypatch):
    """Stop at generation, so the passages the prompt was built from are visible."""
    seen = {}

    def fake_generate(prompt, config, **kwargs):
        seen["passages"] = prompt.passages
        raise RuntimeError("reached generation")

    monkeypatch.setattr("src.rag.answer.generate", fake_generate)
    return seen


def test_answer_question_decomposes_and_records_what_it_split_into(monkeypatch):
    seen = _answering(monkeypatch)
    retriever = FilterRespectingRetriever(
        *[_passage("AAPL", 2024, i) for i in range(4)],
        *[_passage("MSFT", 2024, i) for i in range(4)],
    )
    with pytest.raises(RuntimeError, match="reached generation"):
        answer_question("Compare Apple and Microsoft's AI risk in FY2024",
                        retriever, query=_query(), use_facts=False)
    assert {p.ticker for p in seen["passages"]} == {"AAPL", "MSFT"}


def test_the_labels_reach_the_answer(monkeypatch):
    from src.rag.records import GenerationConfig

    retriever = FilterRespectingRetriever()   # nothing found, so it abstains
    answer = answer_question(
        "Compare Apple and Microsoft's AI risk in FY2024", retriever,
        GenerationConfig("ollama", "m", "grounded_v4"), query=_query(), use_facts=False,
    )
    assert answer.abstained is True
    assert answer.sub_questions == ("AAPL FY2024", "MSFT FY2024")
    assert answer.to_dict()["sub_questions"] == ["AAPL FY2024", "MSFT FY2024"]


def test_use_decomposition_false_searches_once(monkeypatch):
    seen = _answering(monkeypatch)
    retriever = FilterRespectingRetriever(
        *[_passage("AAPL", 2024, i) for i in range(4)],
        *[_passage("MSFT", 2024, i) for i in range(4)],
    )
    with pytest.raises(RuntimeError, match="reached generation"):
        answer_question("Compare Apple and Microsoft's AI risk in FY2024",
                        retriever, query=_query(), use_facts=False,
                        use_decomposition=False)
    assert len(retriever.queries) == 1
    assert retriever.queries[0].tickers == ("AAPL", "MSFT")


def test_a_single_filing_question_is_unaffected_either_way(monkeypatch):
    seen = _answering(monkeypatch)
    single = _query(tickers=("AAPL",), fiscal_years=(2024,))
    for use_decomposition in (True, False):
        retriever = FilterRespectingRetriever(*[_passage("AAPL", 2024, i) for i in range(4)])
        with pytest.raises(RuntimeError, match="reached generation"):
            answer_question("What was Apple's AI risk in FY2024?", retriever, query=single,
                            use_facts=False, use_decomposition=use_decomposition)
        assert len(retriever.queries) == 1
        assert len(seen["passages"]) == 4


def test_an_undecomposed_answer_records_no_sub_questions(monkeypatch):
    from src.rag.records import GenerationConfig

    retriever = FilterRespectingRetriever()
    answer = answer_question(
        "What was Apple's AI risk in FY2024?", retriever,
        GenerationConfig("ollama", "m", "grounded_v4"),
        query=_query(tickers=("AAPL",), fiscal_years=(2024,)), use_facts=False,
    )
    assert answer.sub_questions == ()


# --- the harness and the command line ------------------------------------------------

def test_the_harness_counts_the_rows_it_split(monkeypatch):
    from src.evaluation.harness import evaluate
    from src.evaluation.records import BenchmarkQuestion
    from src.rag.records import Answer, GenerationConfig

    config = GenerationConfig("ollama", "m", "grounded_v4")
    answers = iter([
        Answer(question="q1", text="a", citations=(), passages=(), abstained=True,
               config=config, abstention_reason="no_evidence",
               sub_questions=("AAPL FY2024", "MSFT FY2024")),
        Answer(question="q2", text="a", citations=(), passages=(), abstained=True,
               config=config, abstention_reason="no_evidence"),
    ])
    monkeypatch.setattr("src.evaluation.harness.answer_question", lambda *a, **k: next(answers))
    questions = [
        BenchmarkQuestion.from_mapping({
            "question_id": f"q{n}", "question": "Compare Apple and Microsoft in FY2024",
            "expected_answer": "x", "supporting_chunk_ids": ["c1"],
            "hard_negative_chunk_ids": [], "ticker": None, "fiscal_year": None,
            "question_type": "comparative", "difficulty": "hard", "source": "manual",
        })
        for n in (1, 2)
    ]
    report = evaluate(questions, FilterRespectingRetriever(), config, run_id="r")
    assert report["decomposed"] == 1
    assert report["use_decomposition"] is True
    assert report["results"][0]["sub_questions"] == ["AAPL FY2024", "MSFT FY2024"]
    assert report["results"][1]["sub_questions"] == []


@pytest.mark.parametrize("argv, expected", [
    ([], True),
    (["--no-decompose"], False),
])
def test_the_evaluation_command_can_turn_decomposition_off(monkeypatch, tmp_path, argv, expected):
    import src.evaluation.cli as cli
    from src.retrieval.bm25 import BM25Retriever

    seen = {}
    monkeypatch.setattr(cli, "load_questions", lambda *a, **k: [])
    monkeypatch.setattr(BM25Retriever, "load", lambda **k: object())
    monkeypatch.setattr(cli, "evaluate", lambda *a, **k: seen.update(k) or
                        {"summary": {}, "by_answerability": {}, "results": []})
    cli.main(["q.jsonl", "--retriever", "bm25", "--run-id", "r",
              "--output", str(tmp_path / "report.json"), *argv])
    assert seen["use_decomposition"] is expected


# --- interactions ---------------------------------------------------------------------

def test_a_question_the_facts_route_answers_is_never_decomposable():
    # The two features cannot both fire, and it is the classifier that makes
    # that true rather than an ordering in answer_question: a question is
    # numeric only when it names at most one company and at most one year, and
    # that is exactly the question decompose() returns nothing for.
    from src.rag.query import parse_question

    for question in ["What was Apple's total revenue in FY2024?",
                     "How much net income did Microsoft report in FY2023?",
                     "What were Apple's cash and cash equivalents in FY2024?"]:
        parsed = parse_question(question)
        assert parsed.question_type == "numeric"
        assert decompose(parsed.to_query(top_k=8)) == ()


@pytest.mark.parametrize("question, expect_split", [
    ("Compare Apple and Microsoft's AI risk in FY2024", True),
    ("How did Apple's risk disclosure change between FY2023 and FY2024?", True),
    ("What does Apple say about AI risk in FY2024?", False),
])
def test_the_questions_the_issue_is_about_are_the_ones_that_split(question, expect_split):
    from src.rag.query import parse_question

    subs = decompose(parse_question(question).to_query(top_k=8))
    assert bool(subs) is expect_split


def test_a_passage_returned_by_two_sub_questions_is_kept_once():
    # Disjoint filters mean this cannot happen with a retriever that honours
    # them, but the merged set feeds Answer.passages, which refuses duplicates.
    a, m = _sub("AAPL FY2024", "AAPL", 2024), _sub("MSFT FY2024", "MSFT", 2024)
    shared = _passage("AAPL", 2024, 0)
    passages, provenance = merge([(a, [shared]), (m, [shared])], wanted=4)
    assert [p.chunk_id for p in passages] == ["AAPL-2024-0"]
    assert provenance == {"AAPL-2024-0": "AAPL FY2024"}


def test_a_split_that_would_leave_a_filing_with_nothing_is_not_made():
    # Eight companies and a budget of four: the last four would be labelled as
    # searched and have no passage, which is a comparison missing a side while
    # saying it has one. The single search it would have had is more honest.
    many = _query(tickers=("AAPL", "MSFT", "ORCL", "CRM", "ADBE"), fiscal_years=(2024,), top_k=4)
    assert decompose(many) == ()
    assert len(decompose(replace_top_k(many, 5))) == 5


def replace_top_k(query, top_k):
    from dataclasses import replace
    return replace(query, top_k=top_k)


def test_five_years_of_one_company_split_when_the_budget_covers_them():
    # The corpus's whole span, and a passage from each year answers "how did it
    # evolve" better than eight from whichever two years match the wording.
    five = _query(tickers=("AAPL",), fiscal_years=(2021, 2022, 2023, 2024, 2025), top_k=8)
    assert [sub.label for sub in decompose(five)] == [
        "AAPL FY2021", "AAPL FY2022", "AAPL FY2023", "AAPL FY2024", "AAPL FY2025",
    ]
    # With a budget of four, the same question takes one search.
    assert decompose(replace_top_k(five, 4)) == ()
