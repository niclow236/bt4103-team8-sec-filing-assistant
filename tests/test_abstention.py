"""Issue #33: real indexed retrieval, generation bypass, UI and evaluation."""

import copy
import json
import pickle
from dataclasses import replace

import pytest
from langchain_core.messages import AIMessageChunk

from src.app.answers import main as view
from src.app.answers import render_answer, write_answer_page
from src.evaluation import BenchmarkQuestion, evaluate
from src.evaluation.cli import main
from src.evaluation.harness import (
    MAX_RETRY_WAIT_S,
    RETRY_WAITS_S,
    RunInterrupted,
    RunStopped,
    answer_with_retries,
)
from src.pipeline.chunk import iter_chunks
from src.rag import (
    ProviderBusy,
    ProviderUnavailable,
    answer_question,
    config_from_env,
    parse_question,
    verify_answer,
)
from src.rag.constants import ABSTAIN_PHRASE
from src.rag.records import ABSTENTION_MESSAGES, NO_ABSTENTION_REASON
from src.retrieval import embed
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.records import Query, RetrievedPassage
from src.stack import STACKS


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
    # Saved with no reason, as a run from before reasons were recorded is,
    # the page says so and gives it no reason of its own.
    unexplained = render_answer({**result.to_dict(), "abstention_reason": None})
    assert NO_ABSTENTION_REASON in unexplained and ABSTENTION_MESSAGES[reason] not in unexplained


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


def test_empty_index_and_unknown_custom_retriever_do_not_invent_a_cause():
    for retriever in (BM25Retriever([]), StaticRetriever([])):
        result = answer_question("Revenue?", retriever, CONFIG,
                                 query=Query("Revenue?", tickers=("AAA",)), llm=NeverGenerate())
        assert_abstention(result, "no_evidence")


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


class Stating(Model):
    """A chat model that answers each question with the next of ``sentences``."""

    def __init__(self, *sentences):
        super().__init__()
        self.sentences = iter(sentences)

    def stream(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        yield AIMessageChunk(content=json.dumps({
            "answerable": True,
            "sentences": [{"text": next(self.sentences), "sources": [1]}],
        }))


def test_harness_checks_each_answer_as_the_app_does(tmp_path):
    class ByTicker(StaticRetriever):
        def search(self, query, k=None):
            return [] if query.tickers == ("MISSING",) else self.passages

    asked = [question(1), question(2), question(3), question(4, ticker="MISSING")]
    report = evaluate(asked, ByTicker([passage(text="Revenue was $5.2 billion.")]), CONFIG,
                      run_id="run", llm=Stating("Revenue was $5.2 billion.",
                                                "Revenue was $6 billion.", "Revenue increased."))
    figure_checks = [[check["status"] for check in row["answer"]["verification"]["checks"]
                      if check["kind"] == "passage"] for row in report["results"]]
    assert figure_checks == [["supported"], ["mismatch"], [], []]
    # By the worst check of each answer, as the answer card colours a claim.
    # The abstention states nothing and is not counted.
    assert report["checks"] == {"mismatch": 1, "supported": 1, "unchecked": 1}

    # The rows are what --answers writes, and the viewer shows their checks.
    answers = tmp_path / "answers.jsonl"
    answers.write_text("".join(json.dumps(row) + "\n" for row in report["results"]),
                       encoding="utf-8")
    view([str(answers), "--output", str(tmp_path / "answers.html")])
    page = (tmp_path / "answers.html").read_text(encoding="utf-8")
    assert page.count("<article>") == 4
    assert page.count('data-status="mismatch"') == 1
    assert "This answer has not been verified" not in page


def test_harness_checks_an_answer_in_the_scope_the_benchmark_names():
    # The benchmark's company replaces the question's own. A passage from
    # another company cannot then support the figure, however well it matches.
    cited = StaticRetriever([passage(text="Revenue was $5.2 billion.")])
    own = evaluate([question(1)], cited, CONFIG, run_id="run",
                   llm=Stating("Revenue was $5.2 billion."))
    assert own["checks"] == {"supported": 1}
    other = evaluate([question(1, ticker="BBB")], cited, CONFIG, run_id="run",
                     llm=Stating("Revenue was $5.2 billion."))
    assert other["checks"] == {"mismatch": 1}


@pytest.mark.parametrize("config_id", ["C3", "C4"])
def test_harness_checks_an_answer_against_the_benchmark_s_company_under_any_row(config_id):
    # C3 was measured without the metadata filter, so its search names no
    # company. The answer is still about the benchmark's, and the app checks
    # one against the sidebar's company under such a row too. Checked against
    # the query the row searched with, another company's passage supported it.
    cited = StaticRetriever([passage(text="Revenue was $5.2 billion.")])
    other = evaluate([question(1, ticker="BBB")], cited, CONFIG, run_id="run",
                     llm=Stating("Revenue was $5.2 billion."), stack=STACKS[config_id])
    assert other["checks"] == {"mismatch": 1}


class StatingAll(Model):
    """A chat model that answers every question with all of ``sentences``."""

    def __init__(self, *sentences):
        super().__init__()
        self.sentences = sentences

    def stream(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        yield AIMessageChunk(content=json.dumps({
            "answerable": True,
            "sentences": [{"text": text, "sources": [1]} for text in self.sentences],
        }))


def test_harness_does_not_count_a_partly_unverified_answer_as_supported():
    # One sentence the cited passage supports, and a second the checker cannot
    # check because it names two years. Taken over the whole answer at once,
    # the first sentence's support made the answer "supported".
    cited = StaticRetriever([passage(text="Revenue was $5.2 billion.")])
    report = evaluate([question(1)], cited, CONFIG, run_id="run",
                      llm=StatingAll("Revenue was $5.2 billion.",
                                     "Revenue was $4.0 billion in 2023 and 2024."))
    checks = report["results"][0]["answer"]["verification"]["checks"]
    assert [(check["sentence_index"], check["status"]) for check in checks
            if check["kind"] == "passage"] == [(0, "supported"), (1, "unverified")]
    assert report["checks"] == {"unverified": 1}
    # With both sentences supported, the answer is.
    both = evaluate([question(1)], cited, CONFIG, run_id="run",
                    llm=StatingAll("Revenue was $5.2 billion.", "Revenue was $5.2 billion."))
    assert both["checks"] == {"supported": 1}


class Failing(Model):
    """A chat model whose requests after the first ``after`` raise ``error``
    ``times`` times, then answer again."""

    def __init__(self, error, times, after=0):
        super().__init__()
        self.error, self.times, self.after, self.requests = error, times, after, 0

    def stream(self, messages, **kwargs):
        self.requests += 1
        if self.after < self.requests <= self.after + self.times:
            raise self.error
        return super().stream(messages, **kwargs)


def test_a_busy_provider_is_asked_again_before_it_stops_the_run(monkeypatch, caplog):
    waits = []
    monkeypatch.setattr("src.evaluation.harness.sleep", waits.append)
    busy = ProviderBusy("Mistral's rate limit was reached")

    model = Failing(busy, times=len(RETRY_WAITS_S))
    report = evaluate([question(1)], StaticRetriever([passage()]), CONFIG, run_id="run", llm=model)
    assert (report["summary"]["total"], report["summary"]["abstained"]) == (1, 0)
    assert (model.requests, waits) == (len(RETRY_WAITS_S) + 1, list(RETRY_WAITS_S))
    assert report["results"][0]["attempts"] == len(RETRY_WAITS_S) + 1
    assert report["stopped"] is None
    assert f"1: {busy}; asking again in {RETRY_WAITS_S[0]}s" in caplog.text


def test_a_busy_provider_is_asked_again_without_searching_again(monkeypatch):
    # Only generation failed, so only generation is repeated (#108): the
    # passages and prompt of the first attempt are the ones asked again.
    monkeypatch.setattr("src.evaluation.harness.sleep", lambda seconds: None)

    class Counting(StaticRetriever):
        searches = 0

        def search(self, query, k=None):
            self.searches += 1
            return super().search(query, k)

    retriever = Counting([passage()])
    model = Failing(ProviderBusy("rate limit"), times=1)
    report = evaluate([question(1)], retriever, CONFIG, run_id="run", llm=model)
    assert (retriever.searches, model.requests) == (1, 2)
    assert report["results"][0]["attempts"] == 2
    assert report["results"][0]["answer"]["abstained"] is False


def test_the_retry_is_one_function_a_script_can_ask_through_too(monkeypatch, caplog):
    # The measurement scripts under notebooks/answers/ each had this loop, and
    # their own ask() returns more than an Answer.
    waits = []
    monkeypatch.setattr("src.evaluation.harness.sleep", waits.append)
    busy = ProviderBusy("Mistral's rate limit was reached", retry_after=90)
    asked = []

    def ask():
        asked.append(1)
        if len(asked) == 1:
            raise busy
        return "an answer", 1.5

    assert answer_with_retries("What happened?", ask) == (("an answer", 1.5), 2)
    assert waits == [90]
    assert f"What happened?: {busy}; asking again in 90s" in caplog.text
    # The last attempt's error is raised, not waited out.
    with pytest.raises(ProviderBusy):
        answer_with_retries("q", lambda: (_ for _ in ()).throw(ProviderBusy("still busy")))
    assert waits == [90, *RETRY_WAITS_S]


def test_a_run_the_provider_stops_keeps_the_answers_before_it(monkeypatch):
    waits = []
    monkeypatch.setattr("src.evaluation.harness.sleep", waits.append)
    busy = ProviderBusy("Mistral's rate limit was reached")
    # Question 1 answers; question 2 stays busy past every wait.
    model = Failing(busy, times=len(RETRY_WAITS_S) + 1, after=1)
    with pytest.raises(RunStopped,
                       match="stopped at 2: ProviderBusy: Mistral's rate limit") as raised:
        evaluate([question(1), question(2), question(3)], StaticRetriever([passage()]), CONFIG,
                 run_id="run", llm=model)
    assert raised.value.__cause__ is busy
    assert waits == list(RETRY_WAITS_S)
    report = raised.value.report
    assert report["stopped"] == {"question_id": "2",
                                 "error": "ProviderBusy: Mistral's rate limit was reached"}
    assert [row["question_id"] for row in report["results"]] == ["1"]
    assert report["summary"]["total"] == 1


def test_a_failure_asking_again_cannot_fix_stops_the_run_at_once(monkeypatch):
    monkeypatch.setattr("src.evaluation.harness.sleep",
                        lambda seconds: pytest.fail("a refused key is not asked again"))
    model = Failing(ProviderUnavailable("Mistral refused the API key"), times=1)
    # Caught as a ProviderUnavailable, as a notebook's handler written for
    # evaluate before RunStopped existed would catch it.
    with pytest.raises(ProviderUnavailable, match="stopped at 1: ProviderUnavailable: Mistral "
                                                  "refused the API key") as raised:
        evaluate([question(1)], StaticRetriever([passage()]), CONFIG, run_id="run", llm=model)
    assert type(raised.value) is RunStopped
    assert model.requests == 1
    assert raised.value.report["results"] == []


def _interrupted(seconds):
    raise KeyboardInterrupt


@pytest.mark.parametrize("while_", ["answering", "waiting"])
def test_ctrl_c_keeps_the_answers_before_it(monkeypatch, while_):
    # Most likely pressed during a retry's wait, when a watched run sits idle.
    if while_ == "waiting":
        monkeypatch.setattr("src.evaluation.harness.sleep", _interrupted)
        model = Failing(ProviderBusy("rate limit"), times=1, after=1)
    else:
        model = Failing(KeyboardInterrupt(), times=1, after=1)
    with pytest.raises(RunInterrupted,
                       match="stopped at 2: KeyboardInterrupt: interrupted") as raised:
        evaluate([question(1), question(2), question(3)], StaticRetriever([passage()]), CONFIG,
                 run_id="run", llm=model)
    assert [row["question_id"] for row in raised.value.report["results"]] == ["1"]
    # Still an interrupt, so it stops a notebook cell, and not a provider
    # failure that a handler for one would take it for.
    assert isinstance(raised.value, KeyboardInterrupt)
    assert not isinstance(raised.value, ProviderUnavailable)


@pytest.mark.parametrize("stopper", [RunStopped, RunInterrupted])
def test_a_stopped_run_survives_a_copy_and_a_process_boundary(stopper):
    report = {"stopped": {"question_id": "2", "error": "ProviderBusy: rate limit"},
              "results": [{"question_id": "1"}]}
    for rebuilt in (copy.copy(stopper(report)), pickle.loads(pickle.dumps(stopper(report)))):
        assert type(rebuilt) is stopper
        assert rebuilt.report == report
        assert str(rebuilt) == "stopped at 2: ProviderBusy: rate limit"


def test_a_wait_the_provider_asks_for_is_given(monkeypatch):
    waits = []
    monkeypatch.setattr("src.evaluation.harness.sleep", waits.append)
    model = Failing(ProviderBusy("rate limit", retry_after=90), times=len(RETRY_WAITS_S))
    evaluate([question(1)], StaticRetriever([passage()]), CONFIG, run_id="run", llm=model)
    # Each wait is the longer of the harness's own and the 90 s asked for.
    assert waits == [max(wait, 90) for wait in RETRY_WAITS_S]


def test_a_wait_longer_than_a_run_can_sit_through_stops_it_at_once(monkeypatch):
    monkeypatch.setattr("src.evaluation.harness.sleep",
                        lambda seconds: pytest.fail("an hour is not waited out"))
    model = Failing(ProviderBusy("rate limit", retry_after=MAX_RETRY_WAIT_S + 1), times=1)
    with pytest.raises(RunStopped, match="stopped at 1: ProviderBusy"):
        evaluate([question(1)], StaticRetriever([passage()]), CONFIG, run_id="run", llm=model)
    assert model.requests == 1


def test_an_error_that_is_not_the_provider_s_comes_through_as_itself():
    # A bug is not a stopped run, and must not look like one.
    with pytest.raises(KeyError):
        evaluate([question(1)], StaticRetriever([passage()]), CONFIG, run_id="run",
                 llm=Failing(KeyError("a bug"), times=1))


def test_evaluation_cli_writes_report_and_viewer_rows(corpus, tmp_path, monkeypatch, capsys):
    benchmark = tmp_path / "questions.jsonl"
    benchmark.write_text(json.dumps(question(1, "unanswerable", "MISSING").to_dict()) + "\n")
    retriever = BM25Retriever(list(iter_chunks(processed_dir=corpus)))
    monkeypatch.setattr(BM25Retriever, "load", lambda **kwargs: retriever)
    output, answers = tmp_path / "report.json", tmp_path / "answers.jsonl"
    # Ollama, so a .env that picks Mistral without a key does not stop the run
    # before it starts: the command builds the chat model first.
    main([str(benchmark), "--processed-dir", str(corpus), "--retriever", "bm25",
          "--run-id", "cli", "--output", str(output), "--answers", str(answers),
          "--provider", "ollama"])
    assert json.loads(output.read_text())["summary"]["abstention_rate"] == 1
    row = json.loads(answers.read_text())
    assert row["answer"]["abstention_reason"] == "filters_excluded_all"
    assert row["attempts"] == 1
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


# --- questions the corpus cannot answer ---------------------------------------

class NeverSearch:
    name = "dense"

    def search(self, query, k=None):
        pytest.fail("A refused question must not be searched")


@pytest.mark.parametrize("question, reason", [
    ("Should I buy Amazon stock?", "beyond_the_filings"),
    # Asked of a model with sixteen passages about Meta, this one was answered
    # with a description of its spending plans.
    ("Is Meta a good investment?", "beyond_the_filings"),
    ("Based on what Apple disclosed, predict next quarter's revenue", "beyond_the_filings"),
    # Searched, these get passages from the companies the corpus does hold.
    ("What was Intel's total revenue in FY2024?", "company_not_in_corpus"),
    ("How many employees did NVIDIA have in FY2024?", "company_not_in_corpus"),
    ("What was the total revenue of Intel in FY2024?", "company_not_in_corpus"),
    # The suffix's full stop does not excuse a request made after it.
    ("Based on Apple Inc. filings, predict next year revenue", "beyond_the_filings"),
])
def test_a_question_the_corpus_cannot_answer_is_refused_without_a_search(question, reason,
                                                                         tmp_path):
    said = []
    result = answer_question(question, NeverSearch(), CONFIG, llm=NeverGenerate(),
                             facts_file=tmp_path / "missing.parquet", on_token=said.append)
    assert_abstention(result, reason)
    assert result.latency_ms == 0
    # The app streams, and a refusal reaches the page as its one sentence, once.
    assert said == [ABSTAIN_PHRASE]


@pytest.mark.parametrize("question", [
    # The company is in the corpus and only the year is not. A filing prints
    # the two years before its own, so the FY2021 statements may hold it.
    "What was Apple's total revenue in FY2020?",
    # One company in the corpus and one outside it: the first can be answered.
    "How does Apple describe competition with NVIDIA?",
    # About what a filing says of the future, which a filing does say.
    "What did Apple say it expects next year?",
    # "next year" and a share price are in questions a filing answers: the
    # contractual obligations table, remaining performance obligations, Item
    # 5's repurchases and the cover page. Refused on those words, each of
    # these got no answer where main gave one.
    "What were Microsoft's purchase obligations due next year in FY2024?",
    "How much of Oracle's remaining performance obligations will be recognized in the "
    "next fiscal year, as of FY2024?",
    "What average share price did Apple pay for repurchases in FY2024?",
    "What was the market capitalization of Apple's stock held by non-affiliates in FY2024?",
    # So the two that no filing answers are searched too, and left to the model.
    "What will Microsoft's revenue be next year?",
    "What is Apple's current stock price?",
    # A company outside the corpus, asked about in the filings inside it.
    "Which companies named NVIDIA as a competitor in FY2024?",
    "What did the filings disclose about supply agreements with Intel in FY2023?",
    "Which companies reported revenue from Intel as a customer in FY2024?",
    # Named, with no figure of its own asked for: the model has to abstain.
    "What did NVIDIA report in 2023?",
    # The full stop of "Inc." or "Corp." is not a clause break, so the verb
    # after it is the company's and not an instruction to the engine.
    "What did Apple Inc. estimate as its effective tax rate for fiscal 2024?",
    "What did Microsoft Corp. project for capital expenditures in FY2024?",
    # An outside company as some filing's customer, supplier or product: the
    # figure asked for is the filing's, not the outside company's own.
    "Did NVIDIA account for more than 10% of any company's revenue in FY2024?",
    "Did Samsung account for more than 10% of net sales in FY2024?",
    "How much were purchases of NVIDIA chips in FY2024?",
    "How much revenue came from customers of Intel in FY2024?",
])
def test_a_question_some_filing_may_answer_is_still_searched(question, tmp_path):
    model = Model()
    result = answer_question(question, StaticRetriever([passage(ticker="AAPL")]), CONFIG,
                             llm=model, facts_file=tmp_path / "missing.parquet")
    assert model.calls and not result.abstained


def test_a_company_chosen_by_hand_is_searched_whatever_the_question_named(tmp_path):
    # The app's sidebar lets a user pick the company to search, and Apple's
    # filings may well say something about Intel, even about its revenue.
    asked = "What was Intel's total revenue in FY2024?"
    assert parse_question(asked, facts_file=None).refused == "company"
    model = Model()
    result = answer_question(asked, StaticRetriever([passage(ticker="AAPL")]), CONFIG, llm=model,
                             query=Query(asked, tickers=("AAPL",)),
                             facts_file=tmp_path / "missing.parquet")
    assert model.calls and not result.abstained
    # Advice is refused whichever company is searched.
    advice = "Should I buy Intel stock?"
    refused = answer_question(advice, NeverSearch(), CONFIG, llm=NeverGenerate(),
                              query=Query(advice, tickers=("AAPL",)),
                              facts_file=tmp_path / "missing.parquet")
    assert_abstention(refused, "beyond_the_filings")


def test_a_reading_of_another_question_is_not_answered_with(tmp_path):
    # The reading decides the refusal and the route. One left over from another
    # question would refuse this one without a search.
    stale = parse_question("Should I buy Amazon stock?", facts_file=None)
    asked = "What does Apple say about its supply chain?"
    with pytest.raises(ValueError, match="parsed question must match the question"):
        answer_question(asked, NeverSearch(), CONFIG, llm=NeverGenerate(), parsed=stale,
                        facts_file=tmp_path / "missing.parquet")
    # Space around the question is not another question: parse_question strips.
    spaced = f"  {asked}  "
    model = Model()
    result = answer_question(spaced, StaticRetriever([passage(ticker="AAPL")]), CONFIG,
                             llm=model, parsed=parse_question(spaced, facts_file=None),
                             facts_file=tmp_path / "missing.parquet")
    assert model.calls and not result.abstained


def test_refusal_is_off_for_the_without_half_of_the_comparison(tmp_path):
    model = Model()
    result = answer_question("Should I buy Amazon stock?", StaticRetriever([passage()]), CONFIG,
                             llm=model, facts_file=tmp_path / "missing.parquet",
                             use_refusal=False)
    assert model.calls and not result.abstained


@pytest.mark.parametrize("argv, expected", [
    ([], True),
    (["--no-refusal"], False),
])
def test_the_evaluation_command_can_turn_the_refusal_off(monkeypatch, tmp_path, argv, expected):
    import src.evaluation.cli as cli

    seen = {}
    monkeypatch.setattr(cli, "load_questions", lambda *a, **k: [])
    monkeypatch.setattr(BM25Retriever, "load", lambda **k: object())
    monkeypatch.setattr(cli, "evaluate", lambda *a, **k: seen.update(k) or
                        {"summary": {}, "by_answerability": {}, "results": []})
    cli.main(["q.jsonl", "--retriever", "bm25", "--run-id", "r",
              "--output", str(tmp_path / "report.json"), "--provider", "ollama", *argv])
    assert seen["use_refusal"] is expected
    # The report's ``stack`` is built from the same value, so it says so too.
    assert seen["stack"].use_refusal is expected


def test_evaluation_counts_the_questions_it_refused():
    asked = [replace(question(1), question="What does Apple say about its supply chain?"),
             replace(question(2, "unanswerable", None), question="Should I buy Amazon stock?")]
    report = evaluate(asked, StaticRetriever([passage()]), CONFIG, run_id="refusal",
                      llm=Model())
    assert (report["use_refusal"], report["refused"]) == (True, 1)
    assert report["results"][1]["answer"]["abstention_reason"] == "beyond_the_filings"
    without = evaluate(asked, StaticRetriever([passage()]), CONFIG, run_id="no-refusal",
                       llm=Model(), use_refusal=False)
    assert (without["use_refusal"], without["refused"]) == (False, 0)
