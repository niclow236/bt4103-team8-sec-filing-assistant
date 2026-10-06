"""Gold answer evaluation, citation denominators, resource accounting and CLI."""

import json
from importlib import import_module
from dataclasses import replace
from types import SimpleNamespace

import pytest
import httpx
from langchain_core.messages import AIMessage, AIMessageChunk
from pydantic import ValidationError

from src.evaluation.cli import main
from src.evaluation.harness import RunStopped, evaluate
from src.evaluation.quality import (
    JudgeInvalid, Judgement, LLMJudge, RUBRIC_ID, TokenPrices, citation_scores, query_cost,
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
    with pytest.raises(JudgeInvalid, match="invalid structured"):
        judge(JudgeModel(error=ValueError("bad JSON")))(question(), answer())
    model = JudgeModel()
    model.model = "wrong-model"
    with pytest.raises(ValueError, match="config records"):
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
                                               "generate": 80, "retry_wait": 10000,
                                               "total": 10130, "other": 30})
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


def test_cli_writes_quality_and_separate_judge_cost(tmp_path, monkeypatch, capsys):
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
    assert json.loads(capsys.readouterr().out)["quality"] == report["quality"]


@pytest.mark.parametrize("abstain", [False, True])
@pytest.mark.parametrize("judged", [False, True])
def test_correctness_never_averages_unjudged_abstentions(abstain, judged):
    questions = [question(), question("q2"), question("q3", unanswerable=True)]
    report = evaluate(questions, Retriever(), CONFIG, run_id="quality",
                      llm=Model(abstain=abstain), judge=judge() if judged else None)
    decisions = [0, 0, 1] if abstain else [1, 1, 0]
    assert [r["quality"]["answerability_correct"] for r in report["results"]] == decisions
    assert report["quality"]["answerability_correct"] == {
        "mean": sum(decisions)/3, "scored": 3, "total": 3}
    correctness = report["quality"]["correctness"]
    if judged:
        assert correctness == {"mean": 1/3, "scored": 3, "total": 3}
    else:
        assert correctness == {"mean": None, "scored": 0, "total": 3}
        assert all(r["quality"]["correctness"] is None for r in report["results"])


@pytest.mark.parametrize("bad_scores", [{"faithfulness": .5}, {
    "faithfulness": 2, "correctness": 1, "faithfulness_reason": "x", "correctness_reason": "x"}])
def test_judge_validates_missing_and_invalid_scores_at_its_boundary(bad_scores):
    with pytest.raises(JudgeInvalid, match="invalid structured"):
        judge(JudgeModel(parsed=bad_scores))(question(), answer())


@pytest.mark.parametrize("provider,metadata", [
    ("ollama", {"done_reason": "length"}),
    ("ollama", {"done_reason": "model_length"}),
    ("mistral", {"finish_reason": "length"}),
    ("mistral", {"finish_reason": "model_length"}),
])
def test_truncated_judge_response_leaves_answer_unscored(provider, metadata):
    class CutJudge(JudgeModel):
        def invoke(self, messages, **kwargs):
            result = super().invoke(messages, **kwargs)
            result["raw"].response_metadata = metadata
            return result

    report = evaluate([question()], Retriever(), CONFIG, run_id="quality",
                      llm=Model(), judge=judge(CutJudge(), provider))
    row = report["results"][0]
    assert row["quality"]["method"] == "judge_failed"
    assert "truncated" in row["quality"]["error"]
    assert row["quality"]["correctness"] is None
    assert report["stopped"] is None and report["quality"]["correctness"]["scored"] == 0


def test_unreachable_judge_stops_and_keeps_completed_answer_and_cost():
    class DownJudge(JudgeModel):
        def invoke(self, messages, **kwargs):
            self.calls.append(messages)
            raise ProviderUnavailable("Ollama has no model 'judge-test'")

    model, retriever = DownJudge(), Retriever()
    with pytest.raises(RunStopped) as stopped:
        evaluate([question(), question("q2")], retriever, replace(CONFIG, provider="mistral"),
                 run_id="quality", llm=Model(), judge=judge(model),
                 token_prices=TokenPrices(1, 1))
    report = stopped.value.report
    assert report["stopped"]["question_id"] == "q1"
    assert len(model.calls) == retriever.calls == len(report["results"]) == 1
    row = report["results"][0]
    assert not row["answer"]["abstained"] and row["quality"]["method"] == "judge_failed"
    assert row["cost"]["usd"] == pytest.approx(.00012)
    assert report["quality"]["correctness"] == {"mean": None, "scored": 0, "total": 1}


def test_candidate_diagnostics_count_as_retrieval(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("src.evaluation.harness.perf_counter", lambda: clock[0])

    class Empty(Retriever):
        def has_candidates(self, query):
            clock[0] += .05
            return True

    row = evaluate([question()], Empty([]), CONFIG, run_id="quality",
                   use_decomposition=False)["results"][0]
    assert row["answer"]["abstention_reason"] == "below_threshold"
    assert row["latency_ms"]["retrieve"] == pytest.approx(50)


@pytest.mark.parametrize("flags,model,shares_llm", [([], "test", True),
    (["--judge-model", "other"], "other", False)])
def test_cli_builds_the_judge_it_records(flags, model, shares_llm, tmp_path, monkeypatch):
    settings = SimpleNamespace(min_score=None, top_k=8, use_facts=False,
                               use_decomposition=False, use_refusal=False)
    settings.to_dict = lambda: {"key": "test"}
    settings.scoped = lambda query: query
    stack = SimpleNamespace(config=settings, generation=CONFIG, retriever=Retriever(), llm=Model())
    built = []

    def spy(config, *, llm=None):
        built.append((config, llm))
        judge_model = JudgeModel()
        judge_model.model = config.model
        return LLMJudge(config, llm=judge_model)

    monkeypatch.setattr("src.evaluation.cli.build_stack", lambda *a, **k: stack)
    monkeypatch.setattr("src.evaluation.cli.load_questions", lambda *a, **k: [question()])
    monkeypatch.setattr("src.evaluation.cli.LLMJudge", spy)
    main(["--run-id", "quality", "--output", str(tmp_path/"report.json"), "--judge", *flags])
    (config, llm), = built
    assert (config.model, config.prompt_template_id) == (model, CONFIG.prompt_template_id)
    assert (llm is stack.llm) is shares_llm
    recorded = json.loads((tmp_path/"report.json").read_text())["judge"]["config"]
    assert (recorded["model"], recorded["prompt_template_id"]) == (model, RUBRIC_ID)


@pytest.mark.parametrize("failure", ["busy", "http429", "finish_error"])
def test_busy_judge_retries_with_wait_time_and_unknown_failed_usage(monkeypatch, failure):
    clock = [0.0]
    monkeypatch.setattr("src.evaluation.harness.perf_counter", lambda: clock[0])
    monkeypatch.setattr("src.evaluation.harness.sleep", lambda duration: clock.__setitem__(0, clock[0]+duration))

    class BusyJudge(JudgeModel):
        def invoke(self, messages, **kwargs):
            if not self.calls and failure != "finish_error":
                self.calls.append(messages)
                if failure == "busy":
                    raise ProviderBusy("busy")
                request = httpx.Request("POST", "https://api.mistral.ai/v1/chat/completions")
                response = httpx.Response(429, request=request, json={"message": "rate limit"})
                raise httpx.HTTPStatusError("rate limit", request=request, response=response)
            result = super().invoke(messages, **kwargs)
            result["raw"].response_metadata = {"finish_reason": "error" if len(self.calls)==1 else "stop"}
            return result

    model = BusyJudge(clock=clock)
    retriever = Retriever()
    row = evaluate([question()], retriever, CONFIG, run_id="quality", llm=Model(),
                   judge=judge(model, "mistral"), judge_token_prices=TokenPrices(1, 1))["results"][0]
    assert row["quality"]["method"] == "llm_judge" and len(model.calls) == 2
    assert retriever.calls == row["cost"]["model_calls"] == 1
    assert row["judge_overhead"]["cost"]["model_calls"] == 2
    cost = row["judge_overhead"]["cost"]
    assert cost["successful_calls"] == 1
    if failure == "finish_error":
        assert cost["usage_calls"] == 2 and cost["input_tokens"] == 400
        assert cost["usd"] == pytest.approx(.00046)
    else:
        assert cost["reason"] == "failed_attempt_usage_unknown" and cost["usage_calls"] == 1
    assert row["judge_overhead"]["latency_ms"] == pytest.approx(800 if failure=="finish_error" else 400)
    assert row["judge_overhead"]["retry_wait_ms"] == pytest.approx(10000)
    assert row["judge_overhead"]["total_ms"] == pytest.approx(10800 if failure=="finish_error" else 10400)


def test_rejected_judge_key_maps_to_provider_failure_and_stops():
    class Rejected(JudgeModel):
        def invoke(self, messages, **kwargs):
            request = httpx.Request("POST", "https://api.mistral.ai/v1/chat/completions")
            response = httpx.Response(401, request=request, json={"message": "invalid API key"})
            raise httpx.HTTPStatusError("refused", request=request, response=response)

    with pytest.raises(RunStopped, match="Mistral refused the API key") as error:
        evaluate([question(), question("q2")], Retriever(), CONFIG, run_id="quality",
                 llm=Model(), judge=judge(Rejected(), "mistral"))
    assert len(error.value.report["results"]) == 1


def test_interrupted_judge_retry_keeps_paid_answer_and_partial_cli_report(tmp_path, monkeypatch, capsys):
    class BusyJudge(JudgeModel):
        def invoke(self, messages, **kwargs):
            raise ProviderBusy("busy")

    def interrupt(duration):
        raise KeyboardInterrupt

    settings = SimpleNamespace(min_score=None, top_k=8, use_facts=False,
                               use_decomposition=False, use_refusal=False)
    settings.to_dict = lambda: {"key": "test"}
    settings.scoped = lambda query: query
    stack = SimpleNamespace(config=settings, generation=replace(CONFIG, provider="mistral"),
                            retriever=Retriever(), llm=Model())
    monkeypatch.setattr("src.evaluation.harness.sleep", interrupt)
    monkeypatch.setattr("src.evaluation.cli.build_stack", lambda *a, **k: stack)
    monkeypatch.setattr("src.evaluation.cli.load_questions", lambda *a, **k: [question(), question("q2")])
    monkeypatch.setattr("src.evaluation.cli.LLMJudge", lambda *a, **k: judge(BusyJudge(), "mistral"))
    output = tmp_path/"partial.json"
    with pytest.raises(SystemExit, match="KeyboardInterrupt"):
        main(["--run-id", "quality", "--output", str(output), "--judge",
              "--input-usd-per-million", "1", "--output-usd-per-million", "1"])
    report = json.loads(output.read_text())
    assert report["stopped"]["question_id"] == "q1" and len(report["results"]) == 1
    assert report["stopped"]["stage"] == "judge"
    row = report["results"][0]
    assert row["quality"]["error"] == "interrupted" and row["quality"]["method"] == "judge_failed"
    assert row["cost"]["usd"] == pytest.approx(.00012)
    assert row["judge_overhead"]["cost"]["reason"] == "failed_attempt_usage_unknown"
    assert json.loads(capsys.readouterr().out)["quality"]["correctness"]["scored"] == 0


def test_judge_outage_cli_exits_unsuccessfully_with_zero_score_coverage(tmp_path, monkeypatch, capsys):
    class DownJudge(JudgeModel):
        def invoke(self, messages, **kwargs):
            raise ProviderUnavailable("judge offline")

    settings = SimpleNamespace(min_score=None, top_k=8, use_facts=False,
                               use_decomposition=False, use_refusal=False)
    settings.to_dict = lambda: {"key": "test"}
    settings.scoped = lambda query: query
    stack = SimpleNamespace(config=settings, generation=CONFIG, retriever=Retriever(), llm=Model())
    monkeypatch.setattr("src.evaluation.cli.build_stack", lambda *a, **k: stack)
    monkeypatch.setattr("src.evaluation.cli.load_questions", lambda *a, **k: [question(), question("q2")])
    monkeypatch.setattr("src.evaluation.cli.LLMJudge", lambda *a, **k: judge(DownJudge()))
    output = tmp_path/"partial.json"
    with pytest.raises(SystemExit, match="judge offline"):
        main(["--run-id", "quality", "--output", str(output), "--judge"])
    report = json.loads(output.read_text())
    assert report["stopped"]["question_id"] == "q1"
    assert [r["question_id"] for r in report["results"]] == ["q1"]
    assert json.loads(capsys.readouterr().out)["quality"]["faithfulness"]["scored"] == 0


@pytest.fixture
def cli_stack(monkeypatch):
    settings = SimpleNamespace(min_score=None, top_k=8, use_facts=False,
                               use_decomposition=False, use_refusal=False)
    settings.to_dict = lambda: {"key": "test"}
    settings.scoped = lambda query: query
    stack = SimpleNamespace(config=settings, generation=replace(CONFIG, provider="mistral"),
                            retriever=Retriever(), llm=Model())
    monkeypatch.setattr("src.evaluation.cli.build_stack", lambda *a, **k: stack)
    monkeypatch.setattr("src.evaluation.cli.load_questions", lambda *a, **k:
                        [question(f"q{i}") for i in range(1, 7)])
    return stack


@pytest.mark.parametrize("failure", [RuntimeError, NotImplementedError, httpx.DecodingError,
                                     ProviderUnavailable])
def test_judge_stop_on_question_five_preserves_partial_files_and_resume_stage(
        failure, cli_stack, tmp_path, monkeypatch):
    class FailingJudge(JudgeModel):
        def invoke(self, messages, **kwargs):
            if len(self.calls) == 4:
                self.calls.append(messages)
                raise failure("judge broke")
            return super().invoke(messages, **kwargs)

    model = FailingJudge()
    monkeypatch.setattr("src.evaluation.cli.LLMJudge", lambda *a, **k: judge(model, "mistral"))
    output, answers = tmp_path/"partial.json", tmp_path/"answers.jsonl"
    with pytest.raises(SystemExit, match="4 answered before it"):
        main(["--run-id", "review", "--output", str(output), "--answers", str(answers),
              "--judge", "--input-usd-per-million", "1", "--output-usd-per-million", "1",
              "--judge-input-usd-per-million", "1", "--judge-output-usd-per-million", "1"])
    report = json.loads(output.read_text())
    assert report["stopped"] == {"question_id": "q5", "stage": "judge",
                                 "error": f"{failure.__name__}: judge broke"}
    assert [r["question_id"] for r in report["results"]] == [f"q{i}" for i in range(1, 6)]
    assert [json.loads(line) for line in answers.read_text().splitlines()] == report["results"]
    assert cli_stack.retriever.calls == len(model.calls) == 5
    last = report["results"][-1]
    assert last["quality"]["method"] == "judge_failed" and last["quality"]["correctness"] is None
    assert not last["answer"]["abstained"] and last["cost"]["model_calls"] == 1
    assert report["quality"]["correctness"] == {"mean": .5, "scored": 4, "total": 5}
    assert report["resources"]["api_cost_usd"]["known_total"] == pytest.approx(.0006)
    assert report["resources"]["judge_overhead"]["api_cost_usd"] == {
        "known_total": pytest.approx(.00092), "mean": pytest.approx(.00023),
        "known_queries": 4, "unknown_queries": 1}


@pytest.mark.parametrize("failure", ["length", "model_length", "content_filter", "bad_json", "bad_score"])
@pytest.mark.parametrize("unanswerable", [False, True])
def test_failed_judge_response_keeps_known_spend_and_remains_unscored(failure, unanswerable):
    class InvalidJudge(JudgeModel):
        def invoke(self, messages, **kwargs):
            result = super().invoke(messages, **kwargs)
            result["raw"].usage_metadata = {"input_tokens": 200, "output_tokens": 4096,
                                           "total_tokens": 4296}
            result["raw"].response_metadata = {"finish_reason":
                                               failure if failure not in {"bad_json", "bad_score"} else "stop"}
            if failure == "bad_json":
                result["parsed"], result["parsing_error"] = None, ValueError("bad JSON")
            elif failure == "bad_score":
                result["parsed"] = {"faithfulness": 2}
            return result

    report = evaluate([question(unanswerable=unanswerable), question("q2")], Retriever(), CONFIG,
                      run_id="review", llm=Model(), judge=judge(InvalidJudge(), "mistral"),
                      judge_token_prices=TokenPrices(2, 3))
    assert report["stopped"] is None and len(report["results"]) == 2
    row = report["results"][0]
    assert row["quality"]["method"] == "judge_failed"
    assert row["quality"]["correctness"] is row["quality"]["faithfulness"] is None
    assert row["quality"]["answerability_correct"] == float(not unanswerable)
    assert report["quality"]["correctness"]["scored"] == 0
    cost = row["judge_overhead"]["cost"]
    assert (cost["input_tokens"], cost["output_tokens"]) == (200, 4096)
    assert cost["usd"] == pytest.approx(.012688) and cost["complete"]
    assert cost["model_calls"] == cost["usage_calls"] == 1 and cost["successful_calls"] == 0
    assert report["resources"]["judge_overhead"]["api_cost_usd"]["known_total"] == pytest.approx(.025376)


@pytest.mark.parametrize("judge_provider", ["ollama", "mistral"])
def test_judge_construction_rejects_wrong_client_package(judge_provider):
    from langchain_ollama import ChatOllama
    from langchain_mistralai import ChatMistralAI
    model = (ChatMistralAI(model="judge-test", api_key="test-key") if judge_provider == "ollama"
             else ChatOllama(model="judge-test"))
    with pytest.raises(ValueError, match="config records the provider"):
        judge(model, judge_provider)


def test_judge_builds_and_checks_its_client_once(monkeypatch):
    built = []
    model = JudgeModel()

    def build(config):
        built.append(config)
        return model

    monkeypatch.setattr(import_module("src.rag.generate"), "chat_model", build)
    evaluator = LLMJudge(GenerationConfig("ollama", "judge-test", CONFIG.prompt_template_id))
    for _ in range(3):
        evaluator(question(), answer())
    assert len(built) == 1 and built[0].prompt_template_id == RUBRIC_ID
    assert evaluator.model is model and len(model.calls) == 3


def test_cli_judge_configuration_fails_before_benchmark_or_paid_answers(cli_stack, tmp_path, monkeypatch):
    model = JudgeModel()
    model.model = "wrong"
    cli_stack.llm = model
    monkeypatch.setattr("src.evaluation.cli.load_questions", lambda *a, **k: pytest.fail("loaded benchmark"))
    output = tmp_path/"report.json"
    with pytest.raises(SystemExit) as error:
        main(["--run-id", "review", "--output", str(output), "--judge"])
    assert error.value.code == 2 and not output.exists() and not model.calls


def test_unknown_empty_and_known_zero_cost_totals_are_distinct():
    unknown = evaluate([question()], Retriever(), replace(CONFIG, provider="mistral"),
                       run_id="review", llm=Model(), judge=judge(provider="mistral"))
    for summary in (unknown["resources"]["api_cost_usd"],
                    unknown["resources"]["judge_overhead"]["api_cost_usd"]):
        assert summary == {"known_total": None, "mean": None, "known_queries": 0, "unknown_queries": 1}
    empty = evaluate([], Retriever(), CONFIG, run_id="empty")
    assert empty["resources"]["api_cost_usd"]["known_total"] is None
    assert empty["resources"]["judge_overhead"]["api_cost_usd"]["known_total"] is None
    local = evaluate([question()], Retriever(), CONFIG, run_id="local", llm=Model())
    assert local["resources"]["api_cost_usd"]["known_total"] == 0
    assert local["resources"]["judge_overhead"]["api_cost_usd"]["known_total"] == 0


def test_answer_and_judge_retry_waits_have_separate_aggregate_stages(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("src.evaluation.harness.perf_counter", lambda: clock[0])
    monkeypatch.setattr("src.evaluation.harness.sleep", lambda duration: clock.__setitem__(0, clock[0]+duration))

    class BusyAnswer(Model):
        calls = 0

        def stream(self, messages, **kwargs):
            self.calls += 1
            if self.calls == 1:
                clock[0] += .01
                raise ProviderBusy("busy")
            yield from super().stream(messages, **kwargs)

    class BusyJudge(JudgeModel):
        def invoke(self, messages, **kwargs):
            if not self.calls:
                self.calls.append(messages)
                clock[0] += .02
                raise ProviderBusy("busy")
            return super().invoke(messages, **kwargs)

    report = evaluate([question()], Retriever(), CONFIG, run_id="review",
                      llm=BusyAnswer(clock=clock), judge=judge(BusyJudge(clock=clock)))
    row = report["results"][0]
    stages = row["latency_ms"]
    assert stages["generate"] == pytest.approx(80)
    assert stages["retry_wait"] == pytest.approx(10000)
    assert stages["total"] == pytest.approx(sum(stages[s] for s in ("retrieve", "rerank", "generate", "retry_wait", "other")))
    overhead = row["judge_overhead"]
    assert overhead["latency_ms"] == pytest.approx(420)
    assert overhead["retry_wait_ms"] == pytest.approx(10000)
    assert overhead["total_ms"] == pytest.approx(10420)
    assert report["resources"]["latency_ms"]["generate"]["mean"] == pytest.approx(80)
    assert report["resources"]["latency_ms"]["retry_wait"]["mean"] == pytest.approx(10000)
    assert report["resources"]["judge_overhead"]["latency_ms"]["generate"]["mean"] == pytest.approx(420)
    assert report["resources"]["judge_overhead"]["latency_ms"]["retry_wait"]["mean"] == pytest.approx(10000)
