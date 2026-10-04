"""Gold answer evaluation, citation denominators, resource accounting and CLI."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
import httpx
from langchain_core.messages import AIMessage, AIMessageChunk
from pydantic import ValidationError

from src.evaluation.cli import main
from src.evaluation.harness import RunStopped, evaluate
from src.evaluation.quality import (
    Judgement, LLMJudge, RUBRIC_ID, TokenPrices, citation_scores, query_cost,
)
from src.evaluation.records import BenchmarkQuestion
from src.rag.citations import resolve_citations
from src.rag.constants import PROMPT_TEMPLATE_ID
from src.rag.generate import ProviderBusy, ProviderUnavailable
from src.rag.records import Generation, GenerationConfig, GroundedAnswer
from src.retrieval.records import RetrievedPassage

CONFIG = GenerationConfig("ollama", "test", PROMPT_TEMPLATE_ID)


def question(identifier="q1", *, unanswerable=False):
    return BenchmarkQuestion(identifier, "What happened?", "Revenue increased.",
                             () if unanswerable else ("p1", "p2"), (), "AAA", 2024,
                             "unanswerable" if unanswerable else "factual", "easy", "test")


def passage(identifier="p1", rank=1):
    return RetrievedPassage(chunk_id=identifier, text="Revenue increased.", score=1,
                            rank=rank, retriever="dense", ticker="AAA", company="Alpha",
                            fiscal_year=2024, item="7", title="MD&A", url="https://example.test")


def answer(sources=(1, 3, 9)):
    structured = GroundedAnswer.model_validate({"answerable": True, "sentences": [
        {"text": "Revenue increased.", "sources": list(sources)},
        {"text": "Revenue increased again.", "sources": [1]},
    ]})
    generation = Generation(structured.render(), structured, structured.model_dump_json(),
                            CONFIG, 10, 100, 20, "stop")
    return resolve_citations("What happened?", generation,
                             (passage(), passage("p2", 2), passage("p3", 3)))


class Retriever:
    name = "dense"

    def __init__(self, passages=None):
        self.passages = [passage()] if passages is None else passages
        self.calls = 0

    def search(self, query, k=None):
        self.calls += 1
        return self.passages


class Model:
    def __init__(self, *, abstain=False, clock=None):
        self.abstain = abstain
        self.clock = clock

    def stream(self, messages, **kwargs):
        if self.clock:
            self.clock[0] += .07
        yield AIMessageChunk(content=json.dumps({"answerable": not self.abstain,
                                                "sentences": [] if self.abstain else [
            {"text": "Revenue increased.", "sources": [1]},
        ]}), usage_metadata={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
                             response_metadata={"finish_reason": "stop", "done_reason": "stop"})


class JudgeModel:
    model = "judge-test"

    def __init__(self, *, parsed=None, error=None, clock=None):
        self.parsed = parsed if parsed is not None else {
            "faithfulness": .75, "correctness": .5,
            "faithfulness_reason": "One claim lacks support.",
            "correctness_reason": "A required part is missing.",
        }
        self.error = error
        self.clock = clock
        self.calls = []

    def with_structured_output(self, schema, **options):
        self.schema, self.options = schema, options
        return self

    def model_copy(self, *, update):
        self.request_options = update
        return self

    def invoke(self, messages, **kwargs):
        self.calls.append(messages)
        if self.clock:
            self.clock[0] += .4
        return {"parsed": self.parsed, "parsing_error": self.error,
                "raw": AIMessage(content="", usage_metadata={
                    "input_tokens": 200, "output_tokens": 30, "total_tokens": 230})}


def judge(model=None, provider="ollama"):
    return LLMJudge(GenerationConfig(provider, "judge-test", RUBRIC_ID),
                    llm=model or JudgeModel())


def test_citations_count_resolved_distinct_ids_not_passages_or_repeats():
    scores = citation_scores(question(), answer())
    assert scores == {"citation_precision": .5, "citation_recall": .5,
                      "correct_citations": 1, "resolved_citations": 2, "gold_citations": 2,
                      "unresolved_markers": [9], "invalid_markers": []}
    assert citation_scores(question(), answer((3,)))["citation_precision"] == .5  # second sentence cites 1


def test_unanswerable_and_missing_citation_denominators():
    scores = citation_scores(question(unanswerable=True), answer())
    assert scores["citation_recall"] is None
    assert scores["citation_precision"] == 0
    uncited = replace(answer(), citations=(), sentences=(), text="Revenue increased.")
    assert citation_scores(question(), uncited)["citation_recall"] == 0
    assert citation_scores(question(), uncited)["citation_precision"] is None


@pytest.mark.parametrize("score", [-.1, 1.1, float("nan"), float("inf")])
def test_judge_schema_rejects_invalid_scores(score):
    with pytest.raises(ValidationError):
        Judgement(faithfulness=score, correctness=1, faithfulness_reason="x", correctness_reason="x")


def test_judge_compares_gold_and_only_cited_evidence_and_records_schema():
    model = JudgeModel()
    result = judge(model)(question(), answer())
    data = json.loads(model.calls[0][1][1])
    assert data["gold_answer"] == question().expected_answer
    assert [p["chunk_id"] for p in data["cited_evidence"]] == ["p1", "p3"]
    assert len(data["sentences"]) == 2
    assert model.schema is Judgement
    assert model.options == {"method": "json_schema", "include_raw": True}
    assert result["faithfulness"] == .75 and result["input_tokens"] == 200
    assert model.request_options["temperature"] == 0
    assert model.request_options["num_predict"] > 0


def test_judge_invalid_response_and_identity_fail_explicitly():
    with pytest.raises(ProviderUnavailable, match="invalid structured"):
        judge(JudgeModel(error=ValueError("bad JSON")))(question(), answer())
    model = JudgeModel()
    model.model = "wrong-model"
    with pytest.raises(ValueError, match="match recorded"):
        judge(model)(question(), answer())


@pytest.mark.parametrize("provider", ["ollama", "mistral"])
def test_judge_real_clients_send_schema_options_and_return_usage(provider):
    from langchain_ollama import ChatOllama
    from langchain_mistralai import ChatMistralAI
    seen = []
    scores = {"faithfulness": .8, "correctness": .6,
              "faithfulness_reason": "Evidence supports most claims.",
              "correctness_reason": "Some required detail is missing."}

    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        if provider == "ollama":
            response = {"model": "test", "message": {"role": "assistant", "content": json.dumps(scores)},
                        "done": True, "done_reason": "stop", "prompt_eval_count": 200, "eval_count": 30}
            return httpx.Response(200, text=json.dumps(response) + "\n")
        return httpx.Response(200, json={"id": "test", "object": "chat.completion", "model": "test",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": json.dumps(scores)},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 200, "completion_tokens": 30, "total_tokens": 230}})

    transport = httpx.MockTransport(handler)
    if provider == "ollama":
        model = ChatOllama(model="test", client_kwargs={"transport": transport})
    else:
        model = ChatMistralAI(model="test", api_key="test-key", max_retries=0,
                             client=httpx.Client(base_url="https://example.test", transport=transport))
    result = LLMJudge(replace(CONFIG, provider=provider, prompt_template_id=RUBRIC_ID), llm=model)(
        question(), answer())
    assert result["faithfulness"] == .8 and result["input_tokens"] == 200
    request = seen[0]
    if provider == "ollama":
        assert request["options"]["temperature"] == 0
        assert "faithfulness" in request["format"]["properties"]
        assert request["options"]["num_ctx"] > 0
    else:
        assert request["temperature"] == 0 and request["max_tokens"] > 0
        assert "faithfulness" in request["response_format"]["json_schema"]["schema"]["properties"]


@pytest.mark.parametrize("input_rate,output_rate", [(-1, 0), (0, -1), (float("nan"), 0),
                                                    (0, float("inf")), (True, 0)])
def test_cost_prices_reject_bad_values(input_rate, output_rate):
    with pytest.raises(ValueError):
        TokenPrices(input_rate, output_rate)


def test_hosted_cost_uses_actual_tokens_and_explicit_rates():
    cost = query_cost("mistral", [{"input_tokens": 1_000_000, "output_tokens": 500_000}],
                      attempts=1, prices=TokenPrices(2, 6))
    assert cost["usd"] == 5 and cost["complete"]
    assert cost["model_calls"] == cost["successful_calls"] == 1


@pytest.mark.parametrize("usage,attempts,prices,reason", [
    ({"input_tokens": 100, "output_tokens": 20}, 1, None, "prices_not_supplied"),
    ({"input_tokens": None, "output_tokens": 20}, 1, TokenPrices(1, 1), "token_usage_unknown"),
    ({"input_tokens": 100, "output_tokens": True}, 1, TokenPrices(1, 1), "token_usage_unknown"),
    ({"input_tokens": 100, "output_tokens": 20}, 2, TokenPrices(1, 1), "failed_attempt_usage_unknown"),
])
def test_unknown_cost_never_becomes_zero(usage, attempts, prices, reason):
    cost = query_cost("mistral", [usage], attempts=attempts, prices=prices)
    assert cost["usd"] is None and not cost["complete"] and cost["reason"] == reason


def test_no_model_and_local_routes_have_zero_hosted_cost():
    assert query_cost("mistral", [], attempts=0, prices=None)["usd"] == 0
    assert query_cost("ollama", [{}], attempts=1, prices=None)["usd"] == 0


def test_harness_semantic_scores_do_not_replace_numeric_proxy():
    report = evaluate([question()], Retriever(), CONFIG, run_id="quality", llm=Model(),
                      use_facts=False, judge=judge())
    row = report["results"][0]
    assert row["quality"]["faithfulness"] == .75
    assert row["quality"]["correctness"] == .5
    assert row["quality"]["citation_recall"] == .5
    assert report["quality"]["correctness"] == {"mean": .5, "scored": 1, "total": 1}
    assert report["checks"] == {"unchecked": 1}
    assert report["judge"]["rubric"] == RUBRIC_ID
    assert row["cost"]["input_tokens"] == 100
    assert row["judge_overhead"]["cost"]["input_tokens"] == 200
    assert row["expected_answer"] == question().expected_answer
    json.dumps(report, allow_nan=False)


def test_no_judge_preserves_null_semantic_scores_with_explicit_coverage():
    report = evaluate([question()], Retriever(), CONFIG, run_id="quality", llm=Model())
    assert report["quality"]["faithfulness"] == {"mean": None, "scored": 0, "total": 1}
    assert report["results"][0]["quality"]["method"] == "not_scored"
    empty = evaluate([], Retriever(), CONFIG, run_id="empty")
    assert empty["resources"]["latency_ms"]["total"] == {"mean": None, "queries": 0}
    assert empty["resources"]["api_cost_usd"]["mean"] is None


def test_abstention_correctness_answerability_and_faithfulness_denominators():
    model = JudgeModel()
    report = evaluate([question(), question("q2", unanswerable=True)], Retriever([]),
                      CONFIG, run_id="quality", judge=judge(model))
    assert [r["quality"]["correctness"] for r in report["results"]] == [0, 1]
    assert all(r["quality"]["faithfulness"] is None for r in report["results"])
    assert [r["quality"]["citation_recall"] for r in report["results"]] == [0, None]
    assert not model.calls
    assert report["by_answerability"]["unanswerable"]["abstention_rate"] == 1


def test_answering_unanswerable_is_incorrect_even_if_judge_disagrees():
    model = JudgeModel(parsed={"faithfulness": 1, "correctness": 1,
                              "faithfulness_reason": "supported", "correctness_reason": "correct"})
    report = evaluate([question(unanswerable=True)], Retriever(), CONFIG, run_id="quality",
                      llm=Model(), judge=judge(model))
    assert report["results"][0]["quality"]["correctness"] == 0


def test_judge_failure_keeps_answer_and_does_not_zero_fill_scores():
    report = evaluate([question()], Retriever(), CONFIG, run_id="quality", llm=Model(),
                      judge=judge(JudgeModel(error=ValueError("bad JSON")), "mistral"))
    row = report["results"][0]
    assert row["quality"]["method"] == "judge_failed"
    assert row["quality"]["correctness"] is None
    assert not row["answer"]["abstained"] and report["stopped"] is None
    assert row["judge_overhead"]["cost"]["usd"] is None


def test_stage_timing_and_cost_exclude_judge_and_include_retries(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("src.evaluation.harness.perf_counter", lambda: clock[0])
    monkeypatch.setattr("src.evaluation.harness.sleep", lambda duration: clock.__setitem__(0, clock[0] + duration))

    class TimedRetriever(Retriever):
        def search(self, query, k=None):
            clock[0] += .02
            return super().search(query, k)

    class BusyModel(Model):
        calls = 0

        def stream(self, messages, **kwargs):
            self.calls += 1
            if self.calls == 1:
                clock[0] += .01
                raise ProviderBusy("busy")
            yield from super().stream(messages, **kwargs)

    import src.evaluation.harness as harness
    verify = harness.verify_answer

    def timed_verify(*args, **kwargs):
        clock[0] += .03
        return verify(*args, **kwargs)

    monkeypatch.setattr(harness, "verify_answer", timed_verify)
    retriever = TimedRetriever()
    report = evaluate([question()], retriever, replace(CONFIG, provider="mistral"),
                      run_id="quality", llm=BusyModel(clock=clock), judge=judge(JudgeModel(clock=clock)),
                      token_prices=TokenPrices(1, 1), use_facts=False)
    row = report["results"][0]
    assert row["latency_ms"] == pytest.approx({"retrieve": 20, "rerank": 0,
                                               "generate": 10080, "total": 10130, "other": 30})
    assert row["judge_overhead"]["latency_ms"] == pytest.approx(400)
    assert row["attempts"] == 2 and retriever.calls == 1
    assert row["cost"]["reason"] == "failed_attempt_usage_unknown"
    assert report["resources"]["api_cost_usd"]["unknown_queries"] == 1


def test_partial_report_keeps_completed_metrics(monkeypatch):
    class Failing(Retriever):
        def search(self, query, k=None):
            if self.calls:
                raise ProviderUnavailable("offline")
            return super().search(query, k)

    with pytest.raises(RunStopped) as error:
        evaluate([question(), question("q2")], Failing(), CONFIG, run_id="quality", llm=Model())
    report = error.value.report
    assert report["stopped"]["question_id"] == "q2"
    assert report["quality"]["citation_recall"]["scored"] == 1
    assert report["resources"]["latency_ms"]["retrieve"]["queries"] == 1


@pytest.mark.parametrize("flags", [["--input-usd-per-million", "1"],
                                    ["--input-usd-per-million", "nan", "--output-usd-per-million", "1"],
                                    ["--judge-model", "test"],
                                    ["--judge-input-usd-per-million", "1", "--judge-output-usd-per-million", "1"]])
def test_cli_rejects_invalid_evaluation_options_before_loading(flags, tmp_path, monkeypatch):
    monkeypatch.setattr("src.evaluation.cli.build_stack", lambda *a, **k: pytest.fail("loaded index"))
    with pytest.raises(SystemExit) as error:
        main(["--run-id", "test", "--output", str(tmp_path / "report.json"), *flags])
    assert error.value.code == 2


def test_cli_writes_quality_and_separate_judge_cost(tmp_path, monkeypatch):
    # Exercise CLI wiring and the real harness, with no corpus or provider calls.
    settings = SimpleNamespace(min_score=None, top_k=8, use_facts=False,
                               use_decomposition=False, use_refusal=False)
    settings.to_dict = lambda: {"key": "test"}
    settings.scoped = lambda query: query
    stack = SimpleNamespace(config=settings, generation=replace(CONFIG, provider="mistral"),
                            retriever=Retriever(), llm=Model())
    monkeypatch.setattr("src.evaluation.cli.build_stack", lambda *a, **k: stack)
    monkeypatch.setattr("src.evaluation.cli.load_questions", lambda *a, **k: [question()])
    monkeypatch.setattr("src.evaluation.cli.LLMJudge", lambda *a, **k: judge(provider="mistral"))
    output, answers = tmp_path / "report.json", tmp_path / "answers.jsonl"
    main(["--run-id", "quality", "--output", str(output), "--answers", str(answers), "--judge",
          "--input-usd-per-million", "2", "--output-usd-per-million", "6",
          "--judge-input-usd-per-million", "1", "--judge-output-usd-per-million", "3"])
    report = json.loads(output.read_text())
    row = json.loads(answers.read_text())
    assert row == report["results"][0]
    assert row["cost"]["usd"] == pytest.approx(.00032)
    assert row["judge_overhead"]["cost"]["usd"] == pytest.approx(.00029)
    assert row["quality"]["method"] == "llm_judge"
