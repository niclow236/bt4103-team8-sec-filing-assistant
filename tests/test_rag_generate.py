"""Generation: a local model through LangChain's ChatOllama, held to a Pydantic schema.

Nothing here touches a real server. The chat model is a real ``ChatOllama`` whose
HTTP client runs on ``httpx.MockTransport``, answering ``/api/chat`` with the
newline-delimited JSON Ollama streams, so the LangChain and Ollama client code
that parses a response runs exactly as it does against a live server. The
model's output is fed one character at a time where streaming is under test,
which is the harshest split of the JSON the prose renderer has to survive.
"""

import dataclasses
import json
import os

import httpx
import pytest
from langchain_ollama import ChatOllama

from src.rag.constants import (
    ABSTAIN_PHRASE,
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_URL,
    GENERATION_TIMEOUT_S,
    MAX_OUTPUT_TOKENS,
    NUM_CTX,
    PROMPT_TEMPLATE_ID,
)
from src.rag.citations import resolve_citations
from src.rag.generate import (
    ProviderUnavailable,
    _prose,
    chat_model,
    config_from_env,
    generate,
    stream,
)
from src.rag.prompt import build_prompt
from src.rag.records import CitedSentence, Generation, GenerationConfig, GroundedAnswer
from src.retrieval.records import RetrievedPassage


def _passage(n: int) -> RetrievedPassage:
    return RetrievedPassage(
        chunk_id=f"p{n}", text=f"Revenue was ${n}00.", score=1.0 / n, rank=n, retriever="hybrid",
        ticker="AAA", company="Alpha Corp", fiscal_year=2024, item="7", title="MD&A",
        url=f"https://example.test/p{n}",
    )


PROMPT = build_prompt("What was revenue?", [_passage(1), _passage(2), _passage(3)])

ANSWER = {
    "answerable": True,
    "sentences": [
        {"text": "Revenue was $100.", "sources": [1]},
        {"text": "It rose from $90 a year earlier.", "sources": [1, 3]},
    ],
}
ANSWER_TEXT = "Revenue was $100. [1] It rose from $90 a year earlier. [1][3]"


def _config(**overrides) -> GenerationConfig:
    fields = dict(provider="ollama", model="m", prompt_template_id=PROMPT_TEMPLATE_ID)
    fields.update(overrides)
    return GenerationConfig(**fields)


def _ndjson(output: str, *, size: int = 1, done_reason: str = "stop",
            prompt_tokens: int = 50, eval_tokens: int = 20) -> str:
    """Ollama's streamed /api/chat body for a model that wrote ``output``, ``size`` characters a line."""
    lines = [
        json.dumps({"model": "m", "message": {"role": "assistant", "content": output[i:i + size]},
                    "done": False})
        for i in range(0, len(output), size)
    ]
    lines.append(json.dumps({
        "model": "m", "message": {"role": "assistant", "content": ""}, "done": True,
        "done_reason": done_reason, "prompt_eval_count": prompt_tokens, "eval_count": eval_tokens,
    }))
    return "\n".join(lines) + "\n"


def _ollama(handler, model: str = "m") -> ChatOllama:
    return ChatOllama(
        model=model, base_url=DEFAULT_OLLAMA_URL,
        client_kwargs={"transport": httpx.MockTransport(handler)},
    )


def _serving(output, requests=None, **ndjson) -> ChatOllama:
    """A ChatOllama whose server streams ``output`` back, recording each request."""
    body = output if isinstance(output, str) else json.dumps(output)

    def handler(request):
        if requests is not None:
            requests.append(request)
        return httpx.Response(200, text=_ndjson(body, **ndjson))

    return _ollama(handler)


@pytest.mark.parametrize("marker", [1, 99])
def test_generation_resolves_real_and_invented_citations_end_to_end(marker):
    output = {"answerable": True, "sentences": [{"text": "Revenue rose.", "sources": [marker]}]}
    generation = generate(PROMPT, _config(), llm=_serving(output))
    answer = resolve_citations("What was revenue?", generation, PROMPT.passages)
    citation, = answer.citations
    assert citation.marker == marker
    assert citation.resolved is (marker == 1)
    assert citation.chunk_id == ("p1" if marker == 1 else None)
    assert answer.text == ("Revenue rose. [1]" if marker == 1 else "Revenue rose.")
    assert bool(answer.flagged_sentences) is (marker == 99)
    assert answer.parse_error == generation.parse_error


# --- the answer ----------------------------------------------------------------------

def test_generate_parses_the_answer_and_renders_it_with_markers():
    result = generate(PROMPT, _config(), llm=_serving(ANSWER))
    assert isinstance(result, Generation)
    assert result.answer == GroundedAnswer(
        answerable=True,
        sentences=(CitedSentence(text="Revenue was $100.", sources=(1,)),
                   CitedSentence(text="It rose from $90 a year earlier.", sources=(1, 3))),
    )
    assert result.text == ANSWER_TEXT
    assert json.loads(result.raw) == ANSWER
    assert result.parse_error is None


def test_generate_reports_tokens_stop_reason_and_latency():
    result = generate(PROMPT, _config(), llm=_serving(ANSWER, prompt_tokens=1234, eval_tokens=56))
    assert (result.input_tokens, result.output_tokens) == (1234, 56)
    assert result.stop_reason == "stop"
    assert result.truncated is False
    assert isinstance(result.latency_ms, float) and result.latency_ms >= 0
    assert result.config == _config()


def test_the_request_carries_the_schema_and_the_generation_options():
    requests = []
    generate(PROMPT, _config(temperature=0.3), llm=_serving(ANSWER, requests), max_tokens=99)
    [request] = requests
    assert request.method == "POST"
    assert str(request.url) == f"{DEFAULT_OLLAMA_URL}/api/chat"
    body = json.loads(request.content)
    assert body["model"] == "m"
    assert body["stream"] is True
    assert body["messages"] == PROMPT.to_messages()
    assert body["format"] == GroundedAnswer.for_sources(3).model_json_schema()
    assert body["options"] == {"temperature": 0.3, "num_ctx": NUM_CTX, "num_predict": 99}


def test_the_ceiling_defaults_to_the_constant():
    requests = []
    generate(PROMPT, _config(), llm=_serving(ANSWER, requests))
    assert json.loads(requests[0].content)["options"]["num_predict"] == MAX_OUTPUT_TOKENS


def test_the_schema_only_admits_the_sources_the_prompt_showed():
    schema = GroundedAnswer.for_sources(3).model_json_schema()
    numbers = schema["$defs"]["CitedSentence"]["properties"]["sources"]["items"]
    assert (numbers["minimum"], numbers["maximum"]) == (1, 3)
    assert set(schema["required"]) == {"answerable", "sentences"}
    with pytest.raises(ValueError):
        GroundedAnswer.for_sources(0)


# --- streaming -----------------------------------------------------------------------

def _drain(deltas):
    seen = []
    while True:
        try:
            seen.append(next(deltas))
        except StopIteration as done:
            return seen, done.value


def test_stream_yields_prose_that_joins_to_the_text():
    seen, result = _drain(stream(PROMPT, _config(), llm=_serving(ANSWER, size=1)))
    assert "".join(seen) == result.text == ANSWER_TEXT
    assert len(seen) > 5
    assert not any("{" in delta or '"' in delta for delta in seen)


def test_on_token_sees_the_same_prose_in_order():
    seen = []
    result = generate(PROMPT, _config(), llm=_serving(ANSWER, size=1), on_token=seen.append)
    assert "".join(seen) == result.text


def test_markers_wait_until_their_sentence_is_closed():
    # "[1" could still become "[12]", so nothing is shown for it yet.
    cut = '{"answerable": true, "sentences": [{"text": "Revenue was $100.", "sources": [1'
    assert _prose(cut, complete=False) == "Revenue was $100."
    closed = cut + ']}, {"text": "It'
    assert _prose(closed, complete=False) == "Revenue was $100. [1] It"


def test_wherever_the_output_is_cut_the_stream_joins_to_the_text():
    # Quotes, an escape, a two-digit source and an empty source list: every
    # place a partial read of JSON can go wrong, cut at every character.
    output = json.dumps({"answerable": True, "sentences": [
        {"text": 'Revenue was "$12.5 billion".', "sources": [1, 3]},
        {"text": "Café costs fell.", "sources": []},
    ]})
    for cut in range(1, len(output) + 1):
        seen, result = _drain(stream(
            PROMPT, _config(), llm=_serving(output[:cut], size=1, done_reason="length"),
        ))
        assert "".join(seen) == result.text, (cut, output[:cut])


def test_a_sentence_cut_inside_an_escape_does_not_rewrite_what_was_shown():
    before = '{"answerable": true, "sentences": [{"text": "A.", "sources": [1]}, {"text": "B'
    mid_escape = before + "\\u00"
    assert _prose(mid_escape, complete=False) == "A. [1]"
    assert _prose(before + "\\u00e9 grew", complete=False).startswith(_prose(before, complete=False))


# --- abstention ----------------------------------------------------------------------

def test_an_abstention_reads_as_the_abstain_sentence():
    seen, result = _drain(stream(PROMPT, _config(),
                                 llm=_serving({"answerable": False, "sentences": []}, size=1)))
    assert result.text == ABSTAIN_PHRASE == "".join(seen)
    assert result.answer.abstained is True


def test_answerable_with_no_sentences_is_still_an_abstention():
    result = generate(PROMPT, _config(), llm=_serving({"answerable": True, "sentences": []}))
    assert result.answer.abstained is True
    assert result.text == ABSTAIN_PHRASE


def test_sentences_written_after_saying_unanswerable_are_not_shown():
    output = {"answerable": False, "sentences": [{"text": "From memory, $5.", "sources": [2]}]}
    seen, result = _drain(stream(PROMPT, _config(), llm=_serving(output, size=1)))
    assert result.text == ABSTAIN_PHRASE == "".join(seen)
    assert "memory" not in result.text


# --- output that does not parse --------------------------------------------------------

def test_a_truncated_answer_keeps_what_arrived_and_says_why():
    cut = json.dumps(ANSWER)[:95]   # inside the second sentence's text
    seen, result = _drain(stream(PROMPT, _config(), llm=_serving(cut, size=1, done_reason="length")))
    assert result.truncated is True
    assert result.answer is None
    assert "json" in result.parse_error.lower() or "eof" in result.parse_error.lower()
    assert result.text.startswith("Revenue was $100. [1] It")
    assert "[3]" not in result.text
    assert "".join(seen) == result.text


def test_a_source_number_outside_the_prompt_is_refused_but_still_shown():
    # A runtime that ignored the schema: the answer is refused, and the text
    # keeps the marker so the resolver (#31) can flag it as unresolved.
    output = {"answerable": True, "sentences": [{"text": "Revenue was $100.", "sources": [7]}]}
    result = generate(PROMPT, _config(), llm=_serving(output))
    assert result.answer is None
    assert "less than or equal to 3" in result.parse_error
    assert result.text == "Revenue was $100. [7]"


# --- guards ----------------------------------------------------------------------------

def test_a_prompt_from_another_template_is_refused():
    with pytest.raises(ValueError, match="rendered from template"):
        generate(PROMPT, _config(prompt_template_id="grounded_v1"), llm=_serving(ANSWER))


def test_a_model_other_than_the_one_the_config_records_is_refused():
    with pytest.raises(ValueError, match="serves 'other'"):
        generate(PROMPT, _config(), llm=_ollama(lambda request: None, model="other"))


# --- the Generation record ---------------------------------------------------------------

def test_generation_is_frozen_and_serialises():
    result = generate(PROMPT, _config(), llm=_serving(ANSWER))
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.text = "changed"
    data = result.to_dict()
    assert set(data) == {"text", "answer", "raw", "config", "latency_ms", "input_tokens",
                         "output_tokens", "stop_reason", "truncated", "parse_error"}
    assert data["answer"] == ANSWER
    assert data["config"] == _config().to_dict()
    json.dumps(data)


def test_generation_has_an_answer_or_a_parse_error_and_not_both():
    answer = GroundedAnswer(answerable=False, sentences=())
    with pytest.raises(ValueError, match="either"):
        Generation("t", None, "", _config(), 1.0, None, None, None)
    with pytest.raises(ValueError, match="either"):
        Generation("t", answer, "", _config(), 1.0, None, None, None, parse_error="bad")


def test_generation_refuses_negative_latency():
    answer = GroundedAnswer(answerable=False, sentences=())
    with pytest.raises(ValueError, match="latency_ms"):
        Generation("t", answer, "{}", _config(), -1.0, None, None, None)


# --- configuration -----------------------------------------------------------------------

_LLM_VARS = ("LLM_MODEL", "LLM_BASE_URL", "LLM_NUM_GPU")


@pytest.fixture
def clean_env(monkeypatch):
    """No LLM variables before the test, and none left behind by it.

    load_dotenv writes into the real os.environ, which monkeypatch does not
    see, so the variables a .env test loads are removed again afterwards.
    """
    for name in _LLM_VARS:
        monkeypatch.delenv(name, raising=False)
    yield
    for name in _LLM_VARS:
        os.environ.pop(name, None)


def _dotenv(tmp_path, **values):
    path = tmp_path / ".env"
    path.write_text("".join(f"{k}={v}\n" for k, v in values.items()))
    return path


def test_config_defaults_to_the_local_model(tmp_path, clean_env):
    config = config_from_env(dotenv=tmp_path / "absent.env")
    assert (config.provider, config.model) == ("ollama", DEFAULT_MODEL)
    assert config.prompt_template_id == PROMPT_TEMPLATE_ID
    assert config.temperature == 0.0


def test_config_reads_the_model_from_the_dotenv_file(tmp_path, clean_env):
    assert config_from_env(dotenv=_dotenv(tmp_path, LLM_MODEL="qwen2.5:7b")).model == "qwen2.5:7b"


def test_a_shell_variable_wins_over_the_dotenv_file(tmp_path, clean_env, monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "from-shell")
    assert config_from_env(dotenv=_dotenv(tmp_path, LLM_MODEL="from-file")).model == "from-shell"


def test_an_argument_wins_over_the_environment():
    assert config_from_env(model="explicit", environ={"LLM_MODEL": "env"}).model == "explicit"
    assert config_from_env(environ={"LLM_MODEL": "  "}).model == DEFAULT_MODEL


def test_chat_model_is_chat_ollama_on_the_configured_server(tmp_path, clean_env):
    llm = chat_model(_config(model="llama3.2:3b"), dotenv=tmp_path / "absent.env")
    assert isinstance(llm, ChatOllama)
    assert llm.model == "llama3.2:3b"
    assert llm.base_url == DEFAULT_OLLAMA_URL
    assert llm.num_gpu is None
    assert llm.client_kwargs == {"timeout": GENERATION_TIMEOUT_S}


def test_chat_model_reads_the_gpu_layers_from_the_dotenv_file(tmp_path, clean_env):
    assert chat_model(_config(), dotenv=_dotenv(tmp_path, LLM_NUM_GPU="0")).num_gpu == 0


def test_gpu_layers_must_be_a_whole_number(tmp_path, clean_env):
    with pytest.raises(ValueError, match="LLM_NUM_GPU"):
        chat_model(_config(), dotenv=_dotenv(tmp_path, LLM_NUM_GPU="none"))


def test_the_request_carries_the_gpu_layers_only_when_the_model_sets_them():
    requests = []
    llm = _serving(ANSWER, requests)
    generate(PROMPT, _config(), llm=llm)
    assert "num_gpu" not in json.loads(requests[0].content)["options"]
    generate(PROMPT, _config(), llm=llm.model_copy(update={"num_gpu": 0}))
    assert json.loads(requests[1].content)["options"]["num_gpu"] == 0


def test_chat_model_reads_the_server_from_the_dotenv_file(tmp_path, clean_env):
    path = _dotenv(tmp_path, LLM_BASE_URL="http://gpu-box:11434/")
    assert chat_model(_config(), dotenv=path).base_url == "http://gpu-box:11434"
    assert chat_model(_config(), base_url="http://other:1/", dotenv=path).base_url == "http://other:1"


def test_chat_model_refuses_a_hosted_provider():
    with pytest.raises(ProviderUnavailable, match="no budget"):
        chat_model(_config(provider="anthropic"))


# --- when Ollama cannot answer --------------------------------------------------------------

def _failing(exception_type):
    def handler(request):
        raise exception_type("boom", request=request)
    return _ollama(handler)


def test_no_server_says_how_to_start_one():
    with pytest.raises(ProviderUnavailable, match="ollama serve"):
        generate(PROMPT, _config(), llm=_failing(httpx.ConnectError))


@pytest.mark.parametrize("timeout", [httpx.ConnectTimeout, httpx.ReadTimeout])
def test_a_timeout_says_the_model_is_too_slow_for_the_machine(timeout):
    with pytest.raises(ProviderUnavailable, match="smaller model"):
        generate(PROMPT, _config(), llm=_failing(timeout))


def test_a_model_that_is_not_pulled_says_how_to_pull_it():
    handler = lambda request: httpx.Response(404, json={"error": "model 'm' not found"})
    with pytest.raises(ProviderUnavailable, match="ollama pull m"):
        generate(PROMPT, _config(), llm=_ollama(handler))


def test_a_server_error_is_reported_with_its_message():
    handler = lambda request: httpx.Response(500, json={"error": "model requires more system memory"})
    with pytest.raises(ProviderUnavailable, match="more system memory"):
        generate(PROMPT, _config(), llm=_ollama(handler))


def test_an_error_streamed_mid_response_is_reported():
    handler = lambda request: httpx.Response(200, text=json.dumps({"error": "out of memory"}) + "\n")
    with pytest.raises(ProviderUnavailable, match="out of memory"):
        generate(PROMPT, _config(), llm=_ollama(handler))


def test_an_unrelated_error_is_not_disguised():
    class Broken(ChatOllama):
        def stream(self, *args, **kwargs):
            raise KeyError("a bug, not an outage")

    with pytest.raises(KeyError):
        generate(PROMPT, _config(), llm=Broken(model="m"))
