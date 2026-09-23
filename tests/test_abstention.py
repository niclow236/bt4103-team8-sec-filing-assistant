"""Issue #33: real indexed retrieval, generation bypass, UI and evaluation."""

import json
from dataclasses import replace

import pytest
from langchain_core.messages import AIMessageChunk

from src.app.answers import render_answer, write_answer_page
from src.evaluation import BenchmarkQuestion, evaluate
from src.evaluation.cli import main
from src.pipeline.chunk import iter_chunks
from src.rag import answer_question, config_from_env, verify_answer
from src.rag.constants import ABSTAIN_PHRASE
from src.rag.records import ABSTENTION_MESSAGES
from src.retrieval import embed
from src.retrieval.base import WrappingRetriever
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.records import Query, RetrievedPassage


CONFIG = config_from_env(model="test-model", environ={})


class NeverGenerate:
    def stream(self, *args, **kwargs):
        pytest.fail("No-evidence paths must not call the model")


class Model:
    def __init__(self, answerable=True):
        self.calls = []
        self.answerable = answerable

    def stream(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        yield AIMessageChunk(content=json.dumps({
            "answerable": self.answerable,
            "sentences": [{"text": "Revenue increased.", "sources": [1]}],
        }))


def passage(**changes):
    fields = dict(chunk_id="p1", text="Revenue increased.", score=0.8, rank=1,
                  retriever="dense", ticker="AAA", company="Alpha", fiscal_year=2024,
                  item="7", title="MD&A", url="https://example.test")
    return RetrievedPassage(**(fields | changes))


class StaticRetriever:
    name = "dense"

    def __init__(self, passages):
        self.passages = passages

    def search(self, query, k=None):
        return self.passages


@pytest.fixture(params=["bm25", "dense", "hybrid"])
def indexed(request, corpus, tmp_path, fake_model):
    bm25 = BM25Retriever(list(iter_chunks(processed_dir=corpus)))
    if request.param == "bm25":
        return bm25
    chroma = tmp_path / "chroma"
    embed.build(chroma_dir=chroma, processed_dir=corpus, batch_size=8, sort_window=8)
    dense = DenseRetriever.load(chroma_dir=chroma, processed_dir=corpus)
    return dense if request.param == "dense" else HybridRetriever(bm25, dense)


def assert_abstention(result, reason):
    assert result.abstained
    assert result.abstention_reason == reason
    assert result.to_dict()["abstention_reason"] == reason
    assert result.text == ABSTAIN_PHRASE
    assert result.citations == result.sentences == ()
    html = render_answer(result)
    assert 'data-status="abstained"' in html
    assert ABSTENTION_MESSAGES[reason] in html
    assert "has not been verified" not in html


def test_all_metadata_filters_trigger_abstention(indexed):
    for filters in ({"tickers": ("MISSING",)}, {"fiscal_years": (1900,)},
                    {"items": ("99",)}, {"content_type": "missing"},
                    {"items": ("1A",), "content_type": "table"}):
        result = answer_question("What happened?", indexed, CONFIG,
                                 query=Query("What happened?", **filters), llm=NeverGenerate())
        assert_abstention(result, "filters_excluded_all")
        assert result.latency_ms == 0


def test_score_floor_rejects_real_results_without_generation(indexed):
    result = answer_question("Revenue?", indexed, CONFIG, min_score=1e9,
                             query=Query("Revenue?", tickers=("AAA",)), llm=NeverGenerate())
    assert_abstention(result, "below_threshold")


def test_native_threshold_is_not_mislabelled_as_an_empty_filter(indexed, monkeypatch):
    if indexed.name == "bm25":
        monkeypatch.setattr("src.retrieval.bm25.MIN_BM25_SCORE", 1e9)
    elif indexed.name == "dense":
        indexed.min_score = 1e9
    else:
        monkeypatch.setattr("src.retrieval.hybrid.MIN_FUSED_SCORE", 1e9)
    result = answer_question("Revenue?", indexed, CONFIG,
                             query=Query("Revenue?", tickers=("AAA",)), llm=NeverGenerate())
    assert_abstention(result, "below_threshold")


def test_wrapper_propagates_candidate_availability(corpus):
    class Reranker(WrappingRetriever):
        name = "rerank"

        def score(self, query, passages):
            return [-1.0] * len(passages)

    retriever = Reranker(BM25Retriever(list(iter_chunks(processed_dir=corpus))), min_score=0)
    for ticker, reason in [("AAA", "below_threshold"), ("MISSING", "filters_excluded_all")]:
        assert_abstention(answer_question("Revenue?", retriever, CONFIG,
                          query=Query("Revenue?", tickers=(ticker,)), llm=NeverGenerate()), reason)


def test_empty_index_and_unknown_custom_retriever_do_not_invent_a_cause():
    for retriever in (BM25Retriever([]), StaticRetriever([])):
        result = answer_question("Revenue?", retriever, CONFIG,
                                 query=Query("Revenue?", tickers=("AAA",)), llm=NeverGenerate())
        assert_abstention(result, "no_evidence")


def test_key_section_filter_is_checked_against_index_metadata():
    row = dict(chunk_id="p", text="Revenue increased", is_key_section=False)
    result = answer_question("Revenue?", BM25Retriever([row]), CONFIG,
                             query=Query("Revenue?", key_items_only=True), llm=NeverGenerate())
    assert_abstention(result, "filters_excluded_all")


def test_surviving_evidence_generates_and_cites_only_kept_passages():
    model = Model()
    retriever = StaticRetriever([passage(chunk_id="weak", score=0.1, text="EXCLUDED"),
                                 passage(rank=2)])
    tokens = []
    result = answer_question("Revenue?", retriever, CONFIG, min_score=0.8,
                             llm=model, on_token=tokens.append)
    assert not result.abstained
    assert result.abstention_reason is None
    assert result.citations[0].chunk_id == "p1"
    assert len(result.passages) == 1
    assert "EXCLUDED" not in model.calls[0][0][1]["content"]
    assert "".join(tokens) == result.text


def test_model_can_decline_even_with_high_scoring_evidence():
    result = answer_question("Revenue?", StaticRetriever([passage()]), CONFIG, llm=Model(False))
    assert_abstention(result, "model_declined")


def test_retrieval_abstention_streams_and_survives_verification(tmp_path):
    tokens = []
    result = answer_question("Revenue?", StaticRetriever([]), CONFIG,
                             llm=NeverGenerate(), on_token=tokens.append)
    assert tokens == [ABSTAIN_PHRASE]
    result = verify_answer(result, facts_file=tmp_path / "absent")
    assert_abstention(result, "no_evidence")
    assert result.verification.numeric_support_rate is None
    assert result.verification_warnings == ()


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -float("inf")])
def test_invalid_scores_cannot_be_used_as_evidence(score):
    result = answer_question("Revenue?", StaticRetriever([passage(score=score)]), CONFIG,
                             llm=NeverGenerate())
    assert_abstention(result, "no_evidence")
    with pytest.raises(ValueError, match="finite"):
        answer_question("Revenue?", StaticRetriever([]), CONFIG, min_score=score)


def test_blank_evidence_and_bad_request_do_not_generate():
    assert_abstention(answer_question("Revenue?", StaticRetriever([passage(text=" ")]),
                      CONFIG, llm=NeverGenerate()), "no_evidence")
    with pytest.raises(ValueError, match="top_k"):
        answer_question("Revenue?", StaticRetriever([]), CONFIG, query=Query("Revenue?", top_k=0))
    with pytest.raises(ValueError, match="blank"):
        answer_question(" ", StaticRetriever([]), CONFIG)


def test_errors_are_not_abstentions():
    class Broken(StaticRetriever):
        def search(self, query, k=None):
            raise RuntimeError("index unavailable")

    with pytest.raises(RuntimeError, match="index unavailable"):
        answer_question("Revenue?", Broken([]), CONFIG)


def question(number, question_type="factual", ticker="AAA"):
    return BenchmarkQuestion(str(number), "What happened?", "Revenue increased.",
                             () if question_type == "unanswerable" else ("p1",), (),
                             ticker, 2024, question_type, "easy", "test")


def test_harness_reports_denominators_subsets_and_serialized_reasons(tmp_path):
    class ByTicker(StaticRetriever):
        def search(self, query, k=None):
            return [] if query.tickers == ("MISSING",) else self.passages

        def has_candidates(self, query):
            return query.tickers != ("MISSING",)

    report = evaluate([question(1), question(2, "unanswerable", "MISSING"),
                       question(3, ticker="MISSING")], ByTicker([passage()]), CONFIG,
                      run_id="run", llm=Model())
    assert report["summary"] == {"total": 3, "abstained": 2, "abstention_rate": 2/3,
                                  "reasons": {"filters_excluded_all": 2}}
    assert report["by_answerability"]["unanswerable"]["abstention_rate"] == 1
    assert report["by_answerability"]["answerable"]["abstention_rate"] == 0.5
    decoded = json.loads(json.dumps(report))
    page = write_answer_page(decoded["results"], tmp_path / "answers.html")
    assert page.read_text(encoding="utf-8").count(ABSTENTION_MESSAGES["filters_excluded_all"]) == 2


def test_model_abstention_counts_in_harness_and_empty_subsets_are_null():
    report = evaluate([question(1, "unanswerable")], StaticRetriever([passage()]), CONFIG,
                      run_id="run", llm=Model(False))
    assert report["summary"]["reasons"] == {"model_declined": 1}
    assert report["by_answerability"]["answerable"]["abstention_rate"] is None
    empty = evaluate([], StaticRetriever([]), CONFIG, run_id="empty")
    assert empty["summary"]["abstention_rate"] is None
    with pytest.raises(ValueError, match="unique"):
        evaluate([question(1), question(1)], StaticRetriever([]), CONFIG, run_id="run")


def test_evaluation_cli_writes_report_and_viewer_rows(corpus, tmp_path, monkeypatch, capsys):
    benchmark = tmp_path / "questions.jsonl"
    benchmark.write_text(json.dumps(question(1, "unanswerable", "MISSING").to_dict()) + "\n")
    retriever = BM25Retriever(list(iter_chunks(processed_dir=corpus)))
    monkeypatch.setattr(BM25Retriever, "load", lambda **kwargs: retriever)
    output, answers = tmp_path / "report.json", tmp_path / "answers.jsonl"
    main([str(benchmark), "--processed-dir", str(corpus), "--retriever", "bm25",
          "--run-id", "cli", "--output", str(output), "--answers", str(answers)])
    assert json.loads(output.read_text())["summary"]["abstention_rate"] == 1
    assert json.loads(answers.read_text())["answer"]["abstention_reason"] == "filters_excluded_all"
    assert '"abstention_rate": 1.0' in capsys.readouterr().out


def test_reason_requires_an_abstention():
    result = answer_question("Revenue?", StaticRetriever([passage()]), CONFIG, llm=Model())
    with pytest.raises(ValueError, match="abstained=True"):
        replace(result, abstention_reason="below_threshold")


@pytest.mark.parametrize("settings", [{"top_k": 0}, {"min_score": float("nan")},
                                      {"run_id": " "}])
def test_empty_evaluation_still_validates_configuration(settings):
    with pytest.raises(ValueError):
        evaluate([], StaticRetriever([]), CONFIG, **({"run_id": "empty"} | settings))


@pytest.mark.parametrize("args", [["--top-k", "0"], ["--min-score", "nan"],
                                 ["--run-id", " "], ["--answers", "report.json"]])
def test_cli_rejects_bad_configuration_before_loading_indexes(args):
    with pytest.raises(SystemExit) as error:
        main(["--run-id", "test", "--output", "report.json", *args])
    assert error.value.code == 2
