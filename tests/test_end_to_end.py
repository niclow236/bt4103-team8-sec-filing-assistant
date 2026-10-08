"""One corpus through every stage (#49): pipeline, retrieval, rag and evaluation.

Every other test file checks one package against stand-ins for its neighbours.
Nothing here is a stand-in but the two models: the encoder, which is the
suite's deterministic one, and the chat model, which is scripted. The passages
are cut by the chunker from parsed-filing records, both indexes are built from
those passages and loaded through their own guards, the questions are generated
from a facts store against the same passages, and the answers are the ones
``answer_question`` returns.

So what is checked is what holds between the stages: that the id the chunker
gives a passage is the id it is retrieved, cited and scored under, that a
question about one filing is answered from that filing, that a citation is
rendered from what the corpus stored, that a marker the model made up is
flagged and never shown as a source, and that the numbers a run reports are
the ones those answers give.
"""

from __future__ import annotations

import json
from dataclasses import asdict

import pandas as pd
import pytest
from bs4 import BeautifulSoup
from langchain_core.messages import AIMessageChunk

from conftest import FakeModel
from src.app.components import answer_card_html
from src.evaluation import evaluate, generate_xbrl_questions, load_questions, score_question
from src.evaluation.records import BenchmarkValidationError, RunResult
from src.evaluation.run import FixedSizeBM25Retriever, run_ablation
from src.pipeline.chunk import chunk_filing, iter_chunks, write_chunks
from src.pipeline.records import ParsedFiling, SectionRecord, TableRecord
from src.rag import answer_question, config_from_env, render_citation
from src.rag.constants import ABSTAIN_PHRASE
from src.retrieval import embed
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.constants import CONTEXT_HEADER
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.records import Query

COMPANIES = {"AAA": "Alpha Corp", "BBB": "Beta Inc."}
YEARS = (2023, 2024)
# One paragraph of each filing's discussion is about each of these, and no
# other passage of the filing uses the word, so a question naming one has
# exactly one passage that answers it.
TOPICS = ("cloud", "hardware", "advertising", "licensing", "consulting", "payments")
CONFIG = config_from_env(model="test-model", environ={})
BUDGET, OVERLAP = 500, 0


def _accession(number: int, year: int) -> str:
    return f"000000000{number}-{year % 100}-000001"


def _figure(number: int, year: int, index: int) -> int:
    return 4000 + 100 * number + 20 * (year - YEARS[0]) + index


def _parsed(ticker: str, number: int, year: int) -> ParsedFiling:
    paragraphs = [
        f"{topic.capitalize()} revenue was {_figure(number, year, index):,} million in fiscal "
        f"{year}. The company describes what moved the {topic} business during the year, and "
        "what management expects of it, in the paragraphs that follow this one in the filing. "
        "It reviews those estimates each quarter and updates them as conditions change."
        for index, topic in enumerate(TOPICS)
    ]
    table = TableRecord(
        table_index=0, caption="ASSETS:", headers=["(In millions)", f"{year}"],
        rows=[["Total assets", f"{9000 + 100 * number + year % 100:,}"],
              ["Total liabilities", f"{6000 + 100 * number + year % 100:,}"]],
        n_rows=2, n_cols=2, statement_title="CONSOLIDATED BALANCE SHEETS",
    )

    def section(section_id, item, title, text, tables=()):
        return SectionRecord(
            section_id=section_id, part="II", item=item, title=title, text=text,
            n_chars=len(text), n_tables=len(tables), n_data_tables=len(tables),
            is_key_section=True, is_stub=False, resolved_from=None, confidence=None,
            detection_method=None, validated=True, tables=list(tables),
        )

    return ParsedFiling(
        ticker=ticker, cik=number, company=COMPANIES[ticker], form="10-K",
        filing_date=f"{year + 1}-02-01", accession_no=_accession(number, year),
        url=f"https://example.test/{_accession(number, year)}", source_path="raw.html",
        period_of_report=f"{year}-12-31",
        sections=[
            section("part_ii_item_7", "7", "Management's Discussion and Analysis",
                    "\n\n".join(paragraphs)),
            section("part_ii_item_8", "8", "Financial Statements",
                    "The consolidated financial statements follow and were audited.", [table]),
        ],
    )


def _cut(processed, budget: int, overlap: int) -> None:
    for number, ticker in enumerate(COMPANIES, start=1):
        for year in YEARS:
            parsed = _parsed(ticker, number, year)
            write_chunks(
                chunk_filing(parsed, source_path="interim.json", budget=budget, overlap=overlap),
                processed / ticker / f"10-K_{parsed.filing_date}_{parsed.accession_no}.json",
            )


def _facts(facts_file) -> None:
    rows = []
    for number, ticker in enumerate(COMPANIES, start=1):
        for year in YEARS:
            for index, topic in enumerate(TOPICS):
                figure = _figure(number, year, index)
                rows.append({
                    "ticker": ticker, "cik": number, "company": COMPANIES[ticker],
                    "accession": _accession(number, year), "form": "10-K",
                    "filing_date": f"{year + 1}-02-01", "period_of_report": f"{year}-12-31",
                    "concept": f"us-gaap:{topic.capitalize()}Revenue",
                    "label": f"{topic} revenue", "value": float(figure * 1_000_000),
                    "raw_value": str(figure * 1_000_000), "unit": "USD", "scale": None,
                    "fiscal_year": year, "fiscal_period": "FY",
                    "period_start": f"{year}-01-01", "period_end": f"{year}-12-31",
                    "period_type": "duration", "statement_type": "", "is_audited": True,
                    "is_current_year": True,
                })
    pd.DataFrame(rows).to_parquet(facts_file, index=False)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """The corpus, both indexes and the benchmark, built once by the stages themselves."""
    root = tmp_path_factory.mktemp("end-to-end")
    processed = root / "processed"
    _cut(processed, BUDGET, OVERLAP)
    BM25Retriever.build(processed_dir=processed, index_path=root / "bm25.pkl")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(embed, "_load_model", lambda threads=None, **_: (FakeModel(), 1))
        embed.build(chroma_dir=root / "chroma", processed_dir=processed)
    _facts(root / "facts.parquet")
    generate_xbrl_questions(root / "facts.parquet", processed_dir=processed,
                            output_path=root / "benchmark.jsonl")
    return root


@pytest.fixture
def corpus_ids(world):
    return {row["chunk_id"] for row in iter_chunks(processed_dir=world / "processed")}


@pytest.fixture
def questions(world):
    """The benchmark as a run loads it: validated against the passages on disk."""
    return load_questions(world / "benchmark.jsonl", processed_dir=world / "processed")


@pytest.fixture
def retrievers(world, fake_model):
    """Each index loaded through its own guard against the corpus it was built from."""
    bm25 = BM25Retriever.load(world / "bm25.pkl", processed_dir=world / "processed")
    dense = DenseRetriever.load(chroma_dir=world / "chroma", processed_dir=world / "processed")
    return {"bm25": bm25, "dense": dense, "hybrid": HybridRetriever(bm25, dense)}


class Scripted:
    """A chat model that answers with the sentences it was given, whatever it is asked."""

    def __init__(self, *sentences: tuple[str, list[int]], answerable: bool = True) -> None:
        self.answer = {"answerable": answerable,
                       "sentences": [{"text": text, "sources": sources}
                                     for text, sources in sentences]}
        self.prompts: list[str] = []

    def stream(self, messages, **kwargs):
        self.prompts.append("\n".join(str(getattr(m, "content", m)) for m in messages))
        yield AIMessageChunk(content=json.dumps(self.answer))


class NeverAsked:
    def stream(self, *args, **kwargs):
        pytest.fail("the model must not be asked when there is no evidence to answer from")


def ask(question: str, retriever, llm, **filters):
    return answer_question(
        question, retriever, CONFIG, query=Query(question, top_k=8, **filters), llm=llm,
        use_facts=False, use_refusal=False,
    )


# --- one id, from the chunker to the score ----------------------------------


def test_the_corpus_is_what_the_chunker_cut_from_the_parsed_filings(world, corpus_ids):
    rows = list(iter_chunks(processed_dir=world / "processed"))
    # Six paragraphs, the Item 8 line and its statement, in each of four filings.
    assert len(rows) == len(corpus_ids) == 4 * (len(TOPICS) + 2)
    assert {(row["ticker"], row["fiscal_year"]) for row in rows} == {
        (ticker, year) for ticker in COMPANIES for year in YEARS}
    assert {(row["chunk_budget"], row["chunk_overlap"]) for row in rows} == {(BUDGET, OVERLAP)}
    for topic in TOPICS:
        assert sum(topic in row["text"].lower() for row in rows) == 4


def test_both_indexes_hold_exactly_the_passages_the_chunker_wrote(world, corpus_ids, retrievers):
    assert {chunk["chunk_id"] for chunk in retrievers["bm25"].chunks} == corpus_ids
    assert set(retrievers["dense"].collection.get(include=[])["ids"]) == corpus_ids
    for retriever in (retrievers["bm25"], retrievers["dense"]):
        assert retriever.manifest.n_passages == len(corpus_ids)
        assert (retriever.manifest.chunk_budget, retriever.manifest.chunk_overlap) == (
            BUDGET, OVERLAP)


def test_the_benchmark_is_generated_against_the_same_passages(questions, corpus_ids, world):
    assert len(questions) == 4 * len(TOPICS)
    rows = {row["chunk_id"]: row for row in iter_chunks(processed_dir=world / "processed")}
    for question in questions:
        assert set(question.supporting_chunk_ids) <= corpus_ids
        # The supporting passage is in the filing the question is about, and prints the figure.
        for chunk_id in question.supporting_chunk_ids:
            row = rows[chunk_id]
            assert (row["ticker"], row["fiscal_year"]) == (question.ticker, question.fiscal_year)
            millions = int(question.expected_answer.split()[0]) // 1_000_000
            assert f"{millions:,}" in row["text"]


@pytest.mark.parametrize("method", ["bm25", "dense", "hybrid"])
def test_a_question_about_one_filing_is_searched_in_that_filing_alone(
        retrievers, corpus_ids, method):
    """Four filings print a cloud revenue figure, and three of them would be the wrong answer."""
    retriever = retrievers[method]
    found = retriever.search(Query("cloud revenue", top_k=8, tickers=("BBB",),
                                   fiscal_years=(2023,)))

    assert len(found) == len(TOPICS) + 2          # all of one filing, and nothing else
    assert {(passage.ticker, passage.fiscal_year) for passage in found} == {("BBB", 2023)}
    assert {passage.chunk_id for passage in found} <= corpus_ids
    assert [passage.rank for passage in found] == list(range(1, len(found) + 1))
    assert {passage.retriever for passage in found} == {method}


def test_keyword_search_ranks_the_one_passage_that_answers_first(retrievers, questions):
    """End to end for BM25: the chunker's text, the tokenizer, the filter and the ranking
    put the supporting passage at rank 1 for every question, so every metric is 1."""
    for question in questions:
        query = Query(question.question, top_k=5, tickers=(question.ticker,),
                      fiscal_years=(question.fiscal_year,))
        found = retrievers["bm25"].search(query)
        scored = score_question(
            question, RunResult.from_passages(question.question_id, found, retriever="bm25"), k=5)

        assert found[0].chunk_id == question.supporting_chunk_ids[0]
        assert (scored["recall"], scored["mrr"]) == (1.0, 1.0)
        assert scored["ndcg"] == pytest.approx(1.0)


# --- a cited answer ---------------------------------------------------------


def test_a_citation_is_traced_to_the_passage_and_the_filing_it_came_from(
        retrievers, corpus_ids, world):
    model = Scripted(("Cloud revenue was 4,220 million.", [1]))

    answer = ask("What was cloud revenue?", retrievers["bm25"], model,
                 tickers=("BBB",), fiscal_years=(2024,))

    assert answer.text == "Cloud revenue was 4,220 million. [1]"
    assert not answer.abstained and answer.flagged_sentences == ()
    (citation,) = answer.citations
    cited = answer.passages[0]
    # The marker is a position in the prompt, and resolves to the chunker's id.
    assert (citation.marker, citation.resolved, citation.chunk_id) == (1, True, cited.chunk_id)
    assert cited.chunk_id in corpus_ids
    assert cited.chunk_id == f"{_accession(2, 2024)}_part_ii_item_7_000"
    # The label is built from what the corpus stored. The model wrote an integer.
    assert render_citation(citation, answer.passages) == (
        "[1] Beta Inc. (BBB, CIK 0000000002), 10-K, fiscal year 2024, Part II / Item 7, "
        "Management's Discussion and Analysis, filed 2025-02-01")
    # And the prompt numbered the passage the model was shown as [1].
    assert "4,220 million" in model.prompts[0] and cited.text in model.prompts[0]


def test_the_text_a_citation_quotes_is_the_filings_and_not_the_header_it_was_embedded_with(
        retrievers, world):
    """The dense index encodes a context header before each passage and stores none of it."""
    stored = {row["chunk_id"]: row for row in iter_chunks(processed_dir=world / "processed")}
    header = CONTEXT_HEADER.format(company="Alpha Corp", ticker="AAA", fiscal_year=2024,
                                   form="10-K", item="7",
                                   title="Management's Discussion and Analysis")

    answer = ask("What was hardware revenue?", retrievers["dense"],
                 Scripted(("Hardware revenue is stated.", [1])),
                 tickers=("AAA",), fiscal_years=(2024,))

    assert header.strip() == (
        "Alpha Corp (AAA) FY2024 10-K, Item 7: Management's Discussion and Analysis")
    for passage in answer.passages:
        assert passage.text == stored[passage.chunk_id]["text"]
        assert "Alpha Corp (AAA) FY2024" not in passage.text


def test_an_invented_citation_is_flagged_and_never_rendered_as_a_source(retrievers):
    """The acceptance criterion: a marker naming no passage the model was shown."""
    model = Scripted(("Cloud revenue was 4,220 million.", [1]),
                     ("Margins improved sharply.", [99]))

    answer = ask("What was cloud revenue?", retrievers["hybrid"], model,
                 tickers=("BBB",), fiscal_years=(2024,))

    # Flagged: recorded as unresolved, and its sentence marked for review.
    assert answer.unresolved_markers == (99,)
    invented = next(citation for citation in answer.citations if citation.marker == 99)
    assert (invented.resolved, invented.chunk_id) == (False, None)
    assert [sentence.flagged for sentence in answer.sentences] == [False, True]
    assert answer.flagged_sentences == (answer.sentences[1],)
    # Not rendered: the answer's text drops the marker, and its label names no filing.
    assert answer.text == "Cloud revenue was 4,220 million. [1] Margins improved sharply."
    assert render_citation(invented, answer.passages) == "[99] Unresolved citation"
    # The claim it was attached to cites nothing that was retrieved.
    assert len(answer.passages) < 99 and len(answer.cited_passages) == 1


def test_the_app_shows_an_invented_citation_as_a_warning_and_not_as_a_link(retrievers):
    answer = ask("What was cloud revenue?", retrievers["hybrid"],
                 Scripted(("Cloud revenue was 4,220 million.", [1]),
                          ("Margins improved sharply.", [99])),
                 tickers=("BBB",), fiscal_years=(2024,))

    page = BeautifulSoup(answer_card_html(answer), "html.parser")
    supported, invented = page.select(".sec-claim")

    # The real citation opens its passage. The invented one opens nothing.
    assert [link["data-citation"] for link in page.select("a[data-citation]")] == ["1"]
    assert not invented.select("a")
    assert invented.select_one(".sec-unresolved").text == "[99] (unresolved)"
    assert "sec-warning" in invented["class"]
    assert "sec-unresolved" not in str(supported)
    # Only the passage that exists has a panel to open. Where the invented
    # one's would be there is a warning that names no filing.
    (panel,) = page.select("details")
    assert panel["id"].endswith("-citation-1")
    (warning,) = [tag for tag in page.select("[id]") if tag["id"].endswith("-citation-99")]
    assert warning.name == "p" and "sec-warning" in warning["class"]
    assert warning.text.startswith("[99] Unresolved citation")


def test_a_filter_no_filing_satisfies_is_abstained_from_without_asking_a_model(retrievers):
    for method, retriever in retrievers.items():
        answer = ask("What was cloud revenue?", retriever, NeverAsked(), tickers=("ZZZ",))

        assert answer.abstained and answer.text == ABSTAIN_PHRASE, method
        assert answer.abstention_reason == "filters_excluded_all"
        assert answer.citations == answer.passages == ()


# --- a run ------------------------------------------------------------------


def test_the_harness_reports_the_answers_the_app_would_show(retrievers, questions, corpus_ids):
    asked = questions[:6]
    model = Scripted(("The figure is stated in the first source.", [1]),
                     ("A second claim cites a source that was never shown.", [99]))

    report = evaluate(asked, retrievers["bm25"], CONFIG, run_id="end-to-end", llm=model,
                      use_facts=False, top_k=5)

    assert report["summary"]["total"] == len(asked) == len(report["results"])
    assert report["summary"]["abstained"] == 0 and report["stopped"] is None
    assert report["routes"] == {CONFIG.provider: len(asked)}
    assert len(model.prompts) == len(asked)
    for question, row in zip(asked, report["results"]):
        assert row["question_id"] == question.question_id
        assert row["supporting_chunk_ids"] == list(question.supporting_chunk_ids)
        answer = row["answer"]
        cited = [citation["chunk_id"] for citation in answer["citations"] if citation["resolved"]]
        # BM25 put the supporting passage first, and [1] is the first passage of the prompt.
        assert cited == [question.supporting_chunk_ids[0]] and set(cited) <= corpus_ids
        assert row["quality"]["citation_precision"] == 1.0
        assert row["quality"]["citation_recall"] == 1.0
        # The invented marker is counted apart, so it cannot pass for evidence.
        assert row["quality"]["unresolved_markers"] == [99]
        assert row["quality"]["resolved_citations"] == 1
    assert report["quality"]["citation_precision"] == {
        "mean": 1.0, "scored": len(asked), "total": len(asked)}


def test_the_ablation_runner_scores_every_row_on_the_same_corpus(
        retrievers, questions, world, tmp_path):
    fixed = FixedSizeBM25Retriever(world / "processed", budget=BUDGET)

    manifest = run_ablation(questions, {**retrievers, "bm25-fixed-size": fixed},
                            run_id="end-to-end", results_root=tmp_path, top_k=10)

    rows = {row["config"]["id"]: row for row in manifest["configurations"]}
    assert list(rows) == ["C0", "C1", "C2", "C3", "C4"]
    assert all(row["questions"] == len(questions) for row in rows.values())
    # With the question's own filters a filing is eight passages, so the top ten holds
    # the answer whatever the method. Without them four filings print one each.
    assert rows["C4"]["recall"] == 1.0
    assert rows["C1"]["recall"] == 1.0 and 0.25 <= rows["C1"]["mrr"] <= 1.0
    saved = json.loads((tmp_path / "end-to-end" / "summary.json").read_text(encoding="utf-8"))
    assert [row["config"]["id"] for row in saved["configurations"]] == list(rows)
    for config_id in rows:
        lines = (tmp_path / "end-to-end" / config_id / "questions.jsonl").read_text(
            encoding="utf-8").splitlines()
        assert len(lines) == len(questions)
        first = json.loads(lines[0])
        assert first["question_id"] == questions[0].question_id
        assert first["config"]["id"] == config_id and len(first["retrieved_chunk_ids"]) <= 10


# --- the corpus is cut again ------------------------------------------------


def test_cutting_the_corpus_again_is_refused_by_everything_built_from_the_old_cut(
        world, tmp_path, fake_model):
    """Three guards on one event. After ``chunk --budget 1500 --force`` the old ids are
    gone, so both indexes and the benchmark have to be refused, not searched and scored."""
    recut = tmp_path / "processed"
    _cut(recut, budget=1500, overlap=250)
    old = {row["chunk_id"] for row in iter_chunks(processed_dir=world / "processed")}
    new = {row["chunk_id"] for row in iter_chunks(processed_dir=recut)}
    assert len(new) < len(old) and not old <= new

    with pytest.raises(ValueError, match="does not match the corpus"):
        BM25Retriever.load(world / "bm25.pkl", processed_dir=recut)
    with pytest.raises(ValueError, match="does not match the corpus"):
        DenseRetriever.load(chroma_dir=world / "chroma", processed_dir=recut)
    with pytest.raises(BenchmarkValidationError, match="unknown chunk IDs"):
        load_questions(world / "benchmark.jsonl", processed_dir=recut)

    # Built again from the new cut, all three agree with it.
    rebuilt = BM25Retriever.build(processed_dir=recut, index_path=tmp_path / "bm25.pkl")
    embed.build(chroma_dir=tmp_path / "chroma", processed_dir=recut)
    dense = DenseRetriever.load(chroma_dir=tmp_path / "chroma", processed_dir=recut)
    regenerated = generate_xbrl_questions(world / "facts.parquet", processed_dir=recut,
                                          output_path=tmp_path / "benchmark.jsonl")
    assert rebuilt.manifest.chunk_budget == dense.manifest.chunk_budget == 1500
    assert {question.question_id for question in regenerated} == {
        question.question_id
        for question in load_questions(world / "benchmark.jsonl",
                                       processed_dir=world / "processed")}
    assert all(set(question.supporting_chunk_ids) <= new for question in regenerated)


def test_the_parsed_filing_survives_the_trip_to_disk_the_stages_make(tmp_path):
    """What the parse stage writes is what the chunk stage reads back."""
    from src.pipeline.chunk import load_parsed

    parsed = _parsed("AAA", 1, 2024)
    path = tmp_path / "AAA" / "filing.json"
    path.parent.mkdir()
    path.write_text(json.dumps(asdict(parsed)), encoding="utf-8")

    assert load_parsed(path) == parsed
    assert chunk_filing(load_parsed(path), "s", budget=BUDGET, overlap=OVERLAP) == chunk_filing(
        parsed, "s", budget=BUDGET, overlap=OVERLAP)
