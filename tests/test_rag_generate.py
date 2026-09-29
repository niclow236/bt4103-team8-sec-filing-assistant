"""Generation: Ollama or Mistral through LangChain, held to a Pydantic schema.

Nothing here touches a real server. The chat model is a real ``ChatOllama`` whose
HTTP client runs on ``httpx.MockTransport``, answering ``/api/chat`` with the
newline-delimited JSON Ollama streams, so the LangChain and Ollama client code
that parses a response runs exactly as it does against a live server. Mistral
is tested the same way: a real ``ChatMistralAI`` on a mock transport, answering
``/chat/completions`` with the server-sent events Mistral's API streams. The
model's output is fed one character at a time where streaming is under test,
which is the harshest split of the JSON the prose renderer has to survive.
"""

import dataclasses
import json
import os

import httpx
import pytest
from dotenv import dotenv_values
from langchain_mistralai import ChatMistralAI
from langchain_ollama import ChatOllama

from src.config import PROJECT_ROOT
from src.rag.constants import (
    ABSTAIN_PHRASE,
    DEFAULT_MISTRAL_MODEL,
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_URL,
    GENERATION_TIMEOUT_S,
    HOSTED_TIMEOUT_S,
    MAX_OUTPUT_TOKENS,
    NUM_CTX,
    PROMPT_TEMPLATE_ID,
    PROVIDERS,
)
from src.rag.citations import resolve_citations
from src.rag.generate import (
    _PROVIDERS,
    ProviderBusy,
    ProviderUnavailable,
    _mistral_client,
    _prose,
    _text,
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


def _mistral_config(**overrides) -> GenerationConfig:
    return _config(**{"provider": "mistral", "model": DEFAULT_MISTRAL_MODEL, **overrides})


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

_LLM_VARS = ("LLM_PROVIDER", "LLM_MODEL", "LLM_BASE_URL", "LLM_NUM_GPU", "MISTRAL_API_KEY",
             "MISTRAL_BASE_URL")


@pytest.fixture
def clean_env(monkeypatch):
    """No LLM variables before the test, and none left behind by it.

    load_dotenv writes into the real os.environ, which monkeypatch does not
    see, so the variables a .env test loads are removed again afterwards. The
    Mistral clients the test builds are dropped too, so no later test is
    handed one back.
    """
    for name in _LLM_VARS:
        monkeypatch.delenv(name, raising=False)
    _mistral_client.cache_clear()
    yield
    for name in _LLM_VARS:
        os.environ.pop(name, None)
    _mistral_client.cache_clear()


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


def test_config_reads_the_provider_and_uses_its_default_model(tmp_path, clean_env):
    config = config_from_env(dotenv=_dotenv(tmp_path, LLM_PROVIDER="Mistral"))
    assert (config.provider, config.model) == ("mistral", DEFAULT_MISTRAL_MODEL)


def test_the_model_setting_applies_to_either_provider():
    config = config_from_env(environ={"LLM_PROVIDER": "mistral", "LLM_MODEL": "ministral-14b-2512"})
    assert (config.provider, config.model) == ("mistral", "ministral-14b-2512")


@pytest.mark.parametrize("provider, environ, expected", [
    # A teammate's .env set up for Ollama, running the evaluation with --provider mistral.
    ("mistral", {"LLM_MODEL": "llama3.1:8b"}, DEFAULT_MISTRAL_MODEL),
    ("ollama", {"LLM_PROVIDER": "mistral", "LLM_MODEL": "ministral-14b-2512"}, DEFAULT_MODEL),
    # The provider .env already names: its model still applies.
    ("mistral", {"LLM_PROVIDER": "mistral", "LLM_MODEL": "ministral-14b-2512"}, "ministral-14b-2512"),
    ("Ollama ", {"LLM_MODEL": "llama3.1:8b"}, "llama3.1:8b"),
])
def test_a_provider_chosen_over_the_environment_uses_its_own_default_model(provider, environ, expected):
    assert config_from_env(provider=provider, environ=environ).model == expected


def test_an_argument_chooses_the_provider_over_the_environment():
    assert config_from_env(provider="ollama", environ={"LLM_PROVIDER": "mistral"}).provider == "ollama"
    assert config_from_env(environ={"LLM_PROVIDER": "  "}).provider == "ollama"
    # A typo in .env does not block choosing a provider with the argument.
    config = config_from_env(provider="ollama", environ={"LLM_PROVIDER": "mistrl", "LLM_MODEL": "x"})
    assert (config.provider, config.model) == ("ollama", DEFAULT_MODEL)


def test_an_unknown_provider_setting_is_refused():
    with pytest.raises(ValueError, match="LLM_PROVIDER must be one of ollama, mistral"):
        config_from_env(environ={"LLM_PROVIDER": "openai"})


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


def test_chat_model_refuses_an_unknown_provider():
    with pytest.raises(ProviderUnavailable, match="LLM_PROVIDER to one of ollama, mistral"):
        chat_model(_config(provider="anthropic"))


def test_chat_model_is_chat_mistral_ai_with_the_key_from_the_dotenv_file(tmp_path, clean_env):
    llm = chat_model(_mistral_config(), dotenv=_dotenv(tmp_path, MISTRAL_API_KEY="test-key"))
    assert isinstance(llm, ChatMistralAI)
    assert llm.model == DEFAULT_MISTRAL_MODEL
    assert llm.mistral_api_key.get_secret_value() == "test-key"
    assert llm.timeout == HOSTED_TIMEOUT_S


def test_a_temperature_the_api_accepts_is_not_refused_when_the_model_is_built(tmp_path, clean_env):
    # ChatMistralAI's own check stops at 1; Mistral's API took 1.2 on 28 September
    # 2026 and refuses above 1.5 with a 422, which the error handling reads.
    chat_model(_mistral_config(temperature=1.2), dotenv=_dotenv(tmp_path, MISTRAL_API_KEY="test-key"))


def test_an_ollama_address_is_not_taken_for_mistral(tmp_path, clean_env):
    with pytest.raises(ValueError, match="MISTRAL_BASE_URL"):
        chat_model(_mistral_config(), base_url="http://proxy:8080/v1",
                   dotenv=_dotenv(tmp_path, MISTRAL_API_KEY="test-key"))


def test_every_provider_the_command_offers_can_answer():
    assert set(PROVIDERS) == set(_PROVIDERS)


def test_one_mistral_client_serves_every_question_with_the_same_key(tmp_path, clean_env):
    config = _mistral_config()
    first = chat_model(config, dotenv=_dotenv(tmp_path, MISTRAL_API_KEY="test-key"))
    assert chat_model(config, dotenv=_dotenv(tmp_path, MISTRAL_API_KEY="test-key")) is first
    os.environ.pop("MISTRAL_API_KEY")
    other = chat_model(config, dotenv=_dotenv(tmp_path, MISTRAL_API_KEY="another-key"))
    assert other is not first
    assert other.mistral_api_key.get_secret_value() == "another-key"


@pytest.mark.parametrize("values", [{}, {"MISTRAL_API_KEY": ""}])
def test_a_missing_key_says_where_to_make_one(tmp_path, clean_env, values):
    # No key line is what .env.example leaves, and an empty value what removing
    # its # without pasting a key leaves.
    with pytest.raises(ProviderUnavailable, match="console.mistral.ai"):
        chat_model(_mistral_config(), dotenv=_dotenv(tmp_path, **values))


def test_a_key_added_to_the_dotenv_file_after_the_error_is_read(tmp_path, clean_env):
    # What the missing-key message asks for, in one notebook: refused, the key
    # pasted into .env, the question asked again, with no restart.
    path = _dotenv(tmp_path, LLM_PROVIDER="mistral")
    with pytest.raises(ProviderUnavailable, match="MISTRAL_API_KEY is not set"):
        chat_model(_mistral_config(), dotenv=path)
    path.write_text("LLM_PROVIDER=mistral\nMISTRAL_API_KEY=test-key\n")
    assert chat_model(_mistral_config(), dotenv=path).mistral_api_key.get_secret_value() == "test-key"


def test_the_env_template_leaves_the_key_line_commented():
    # An uncommented MISTRAL_API_KEY= is loaded as "", and a variable that is set
    # is never replaced from .env, so a key pasted in later would not be read.
    assert "MISTRAL_API_KEY" not in dotenv_values(PROJECT_ROOT / ".env.example")


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
    with pytest.raises(ProviderUnavailable, match="more system memory") as raised:
        generate(PROMPT, _config(), llm=_ollama(handler))
    # The same model on the same machine fails the same way again.
    assert not isinstance(raised.value, ProviderBusy)


def test_an_ollama_server_with_a_full_queue_is_asked_again():
    # Ollama's own 503 when more requests are waiting than OLLAMA_MAX_QUEUE allows.
    busy = {"error": "server busy, please try again.  maximum pending requests exceeded"}
    with pytest.raises(ProviderBusy, match="Ollama at .* is busy running 'm'.*: ask again"):
        generate(PROMPT, _config(), llm=_ollama(lambda request: httpx.Response(503, json=busy)))


def test_an_error_streamed_mid_response_is_reported():
    handler = lambda request: httpx.Response(200, text=json.dumps({"error": "out of memory"}) + "\n")
    with pytest.raises(ProviderUnavailable, match="out of memory"):
        generate(PROMPT, _config(), llm=_ollama(handler))


def test_a_connection_ollama_closes_part_way_says_what_to_do():
    class Dropped(httpx.SyncByteStream):
        def __iter__(self):
            yield _ndjson(json.dumps(ANSWER))[:200].encode()
            raise httpx.RemoteProtocolError(
                "peer closed connection without sending complete message body")

    with pytest.raises(ProviderBusy, match=r"connection to Ollama at .* broke off .*"
                                           r"\(RemoteProtocolError.*still running"):
        generate(PROMPT, _config(), llm=_ollama(lambda request: httpx.Response(200, stream=Dropped())))


def test_an_unrelated_error_is_not_disguised():
    class Broken(ChatOllama):
        def stream(self, *args, **kwargs):
            raise KeyError("a bug, not an outage")

    with pytest.raises(KeyError):
        generate(PROMPT, _config(), llm=Broken(model="m"))


# --- Mistral --------------------------------------------------------------------------------

MISTRAL_URL = "https://api.mistral.ai/v1"


def _sse(output: str, *, size: int = 1, finish_reason: str = "stop",
         prompt_tokens: int = 50, completion_tokens: int = 20, thinking: str | None = None) -> str:
    """Mistral's streamed /chat/completions body for a model that wrote ``output``.

    With ``thinking``, the model reasons before it answers and every piece comes
    as typed blocks, which is how Mistral's reasoning models stream.
    """
    def piece(content):
        return {"id": "c", "object": "chat.completion.chunk", "model": DEFAULT_MISTRAL_MODEL,
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": content},
                             "finish_reason": None}]}

    texts = [output[i:i + size] for i in range(0, len(output), size)]
    if thinking is None:
        chunks = [piece(text) for text in texts]
    else:
        chunks = [piece([{"type": "thinking", "thinking": [{"type": "text", "text": thinking}]}])]
        chunks += [piece([{"type": "text", "text": text}]) for text in texts]
    chunks.append({
        "id": "c", "object": "chat.completion.chunk", "model": DEFAULT_MISTRAL_MODEL,
        "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                  "total_tokens": prompt_tokens + completion_tokens},
    })
    return "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"


def _mistral(handler) -> ChatMistralAI:
    return ChatMistralAI(
        model=DEFAULT_MISTRAL_MODEL, api_key="test-key", max_retries=1,
        client=httpx.Client(base_url=MISTRAL_URL, transport=httpx.MockTransport(handler)),
    )


def _mistral_serving(output, requests=None, **sse) -> ChatMistralAI:
    """A ChatMistralAI whose API streams ``output`` back, recording each request."""
    body = output if isinstance(output, str) else json.dumps(output)

    def handler(request):
        if requests is not None:
            requests.append(request)
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              text=_sse(body, **sse))

    return _mistral(handler)


def test_a_mistral_answer_streams_and_parses_like_an_ollama_one():
    seen, result = _drain(stream(PROMPT, _mistral_config(), llm=_mistral_serving(
        ANSWER, prompt_tokens=1234, completion_tokens=56)))
    assert "".join(seen) == result.text == ANSWER_TEXT
    assert json.loads(result.raw) == ANSWER
    assert result.parse_error is None
    assert (result.input_tokens, result.output_tokens) == (1234, 56)
    assert (result.stop_reason, result.truncated) == ("stop", False)


def test_the_mistral_request_carries_the_schema_strictly_and_the_options():
    requests = []
    generate(PROMPT, _mistral_config(temperature=0.3), llm=_mistral_serving(ANSWER, requests),
             max_tokens=99)
    [request] = requests
    assert request.method == "POST"
    assert str(request.url) == f"{MISTRAL_URL}/chat/completions"
    body = json.loads(request.content)
    assert (body["model"], body["stream"]) == (DEFAULT_MISTRAL_MODEL, True)
    assert [(m["role"], m["content"]) for m in body["messages"]] == [
        (m["role"], m["content"]) for m in PROMPT.to_messages()
    ]
    assert body["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "GroundedAnswer", "strict": True,
                        "schema": GroundedAnswer.for_sources(3).model_json_schema()},
    }
    assert (body["temperature"], body["max_tokens"]) == (0.3, 99)
    # Ollama's fields are not sent: they mean nothing to Mistral's API.
    assert not {"format", "options"} & set(body)


@pytest.mark.parametrize("reason", ["length", "model_length"])
def test_a_mistral_answer_cut_off_at_a_limit_is_truncated(reason):
    cut = json.dumps(ANSWER)[:95]   # inside the second sentence's text
    result = generate(PROMPT, _mistral_config(), llm=_mistral_serving(cut, finish_reason=reason))
    assert result.truncated is True
    assert result.answer is None
    assert result.text.startswith("Revenue was $100. [1] It")


def test_a_reasoning_model_s_thinking_is_not_part_of_the_answer():
    seen, result = _drain(stream(PROMPT, _mistral_config(), llm=_mistral_serving(
        ANSWER, thinking="The first passage gives the revenue.")))
    assert "".join(seen) == result.text == ANSWER_TEXT
    assert result.raw == json.dumps(ANSWER)


def test_a_piece_sent_as_blocks_reads_as_its_text():
    assert _text("plain") == "plain"
    blocks = [{"type": "text", "text": "Rev"}, "enue",
              {"type": "thinking", "thinking": [{"type": "text", "text": "hidden"}]}]
    assert _text(blocks) == "Revenue"
    assert _text([]) == ""


def test_a_text_block_with_no_text_adds_nothing():
    assert _text([{"type": "text", "text": None}, {"type": "text", "text": "Rev"}]) == "Rev"


# --- when Mistral cannot answer -------------------------------------------------------------

def _refusing(status, body=None):
    return _mistral(lambda request: httpx.Response(status, json=body) if body is not None
                    else httpx.Response(status))


@pytest.mark.parametrize("status, body, expected", [
    # The body Mistral's API sent for an expired key, on 27 September 2026.
    (401, {"detail": "Your API key expired on 2026-09-26."},
     r"refused the API key \(HTTP 401: Your API key expired on 2026-09-26\); check MISTRAL_API_KEY"),
    (401, {"message": "Unauthorized"}, r"refused the API key \(HTTP 401: Unauthorized\)"),
    (403, {"message": "Model not available on your tier"}, "plan does not include"),
    (429, {"message": "Requests rate limit exceeded"}, "rate limit"),
    # The body Mistral's API sent for an Ollama model name, on 28 September 2026.
    (400, {"object": "error", "message": "Invalid model: llama3.1:8b", "type": "invalid_model",
           "param": None, "code": "1500"},
     "set LLM_MODEL to a Mistral model"),
    # A refusal that mentions a model without the model being wrong is not read
    # as one: the advice would name the model already in use.
    (400, {"object": "error", "type": "invalid_request_error",
           "message": "Prompt contains 140000 tokens, too large for model with 131072 maximum "
                      "context length"},
     r"^Mistral's API refused the request \(HTTP 400: Prompt contains 140000 tokens"),
    # An outage rather than a refusal: ask again, or answer locally.
    (500, {"message": "Internal server error"},
     r"^Mistral's API failed while running 'ministral-8b-2512' \(HTTP 500: Internal server "
     r"error\): ask again"),
    (502, None, r"failed while running .*\(HTTP 502\): ask again"),
    # A problem with a null location is in the request as a whole, rather than a
    # TypeError from joining None.
    (422, {"message": {"detail": [{"loc": None, "msg": "Field required"}]}},
     r"\(HTTP 422: request: Field required\)$"),
    # The body Mistral's API sent for temperature 2, on 28 September 2026: the
    # message is a list of problems, each read as where it is and what is wrong.
    (422, {"object": "error", "type": "invalid_request_error", "param": None, "code": None,
           "message": {"detail": [{"type": "less_than_equal", "loc": ["body", "temperature"],
                                   "msg": "Input should be less than or equal to 1.5",
                                   "input": 2, "ctx": {"le": 1.5}}]}},
     r"^Mistral's API refused the request \(HTTP 422: body\.temperature: Input should be less "
     r"than or equal to 1\.5\)$"),
    # The same problems straight under detail, as a proxy built on FastAPI sends them.
    (422, {"detail": [{"loc": ["body", "temperature"], "msg": "Input should be less than or "
                                                              "equal to 1.5"}]},
     r"\(HTTP 422: body\.temperature: Input should be less than or equal to 1\.5\)$"),
])
def test_a_refused_request_says_what_to_do(status, body, expected):
    with pytest.raises(ProviderUnavailable, match=expected):
        generate(PROMPT, _mistral_config(), llm=_refusing(status, body))


def test_a_refusal_whose_body_was_never_read_still_says_what_to_do():
    request = httpx.Request("POST", f"{MISTRAL_URL}/chat/completions")
    unread = httpx.Response(401, stream=httpx.ByteStream(b'{"detail": "expired"}'), request=request)

    class Refusing(ChatMistralAI):
        def stream(self, *args, **kwargs):
            raise httpx.HTTPStatusError("refused", request=request, response=unread)

    llm = Refusing(model=DEFAULT_MISTRAL_MODEL, api_key="test-key")
    with pytest.raises(ProviderUnavailable, match=r"refused the API key \(HTTP 401\)"):
        generate(PROMPT, _mistral_config(), llm=llm)


def test_an_unrelated_error_from_mistral_is_not_disguised():
    class Broken(ChatMistralAI):
        def stream(self, *args, **kwargs):
            raise KeyError("a bug, not an outage")

    with pytest.raises(KeyError):
        generate(PROMPT, _mistral_config(), llm=Broken(model=DEFAULT_MISTRAL_MODEL, api_key="test-key"))


@pytest.mark.parametrize("exception_type, expected", [
    (httpx.ConnectError, "cannot reach Mistral"),
    (httpx.ReadTimeout, "sent nothing"),
])
def test_no_response_from_mistral_says_what_to_do(exception_type, expected):
    def handler(request):
        raise exception_type("boom", request=request)

    with pytest.raises(ProviderUnavailable, match=expected):
        generate(PROMPT, _mistral_config(), llm=_mistral(handler))


@pytest.mark.parametrize("address, expected", [
    ("https://api.mistrl.ai/v1", r"cannot reach .*\(boom\): check the internet connection, "
                                 r"unset MISTRAL_BASE_URL, since nobody needs to set it, or set"),
    (None, r"cannot reach .*\(boom\): check the internet connection, or set LLM_PROVIDER"),
])
def test_a_mistral_address_that_cannot_be_reached_names_the_setting(
        monkeypatch, clean_env, address, expected):
    # A mistyped host fails on every attempt, like being offline, so the
    # setting is named whenever someone has set it, with httpx's reason.
    if address is not None:
        monkeypatch.setenv("MISTRAL_BASE_URL", address)
    with pytest.raises(ProviderBusy, match=expected):
        generate(PROMPT, _mistral_config(), llm=_mistral(_raises(httpx.ConnectError)))


def test_a_connection_dropped_part_way_through_an_answer_says_what_to_do():
    class Dropped(httpx.SyncByteStream):
        def __iter__(self):
            yield _sse(json.dumps(ANSWER)).split("data: [DONE]")[0][:200].encode()
            raise httpx.RemoteProtocolError(
                "peer closed connection without sending complete message body")

    def handler(request):
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=Dropped())

    with pytest.raises(ProviderUnavailable, match="connection to Mistral's API at .* broke off "
                                                  r"\(RemoteProtocolError"):
        generate(PROMPT, _mistral_config(), llm=_mistral(handler))


@pytest.mark.parametrize("reason, expected, busy", [
    ("error", "part-way with an error: ask again", True),
    # A reason nothing expects comes back the same for the same prompt.
    ("tool_calls", r"without finishing \(finish_reason 'tool_calls'\); set LLM_PROVIDER", False),
])
def test_an_answer_mistral_did_not_finish_is_not_handed_on_as_one(reason, expected, busy):
    half = json.dumps(ANSWER)[:95]
    with pytest.raises(ProviderUnavailable, match=f"stopped 'ministral-8b-2512' {expected}") as raised:
        generate(PROMPT, _mistral_config(), llm=_mistral_serving(half, finish_reason=reason))
    assert isinstance(raised.value, ProviderBusy) is busy


@pytest.mark.parametrize("finish_reason", ["error", "tool_calls"])
def test_a_whole_answer_that_ended_badly_is_still_refused(finish_reason):
    # Only a missing reason is taken on the output's word; one the API gave is not.
    with pytest.raises(ProviderUnavailable, match="without finishing|with an error"):
        generate(PROMPT, _mistral_config(),
                 llm=_mistral_serving(ANSWER, finish_reason=finish_reason))


def test_a_finish_reason_sent_on_two_chunks_is_still_a_finished_answer():
    # Merged chunk metadata joins strings end to end, so "stop" twice would read
    # "stopstop", and adds the repeated token counts to each other.
    events = _sse(json.dumps(ANSWER)).split("\n\n")
    twice = events[:-2] + events[-3:-2] + events[-2:]

    def handler(request):
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              text="\n\n".join(twice))

    result = generate(PROMPT, _mistral_config(), llm=_mistral(handler))
    assert (result.text, result.stop_reason) == (ANSWER_TEXT, "stop")
    assert (result.input_tokens, result.output_tokens) == (50, 20)


def test_a_whole_answer_whose_last_event_lost_its_reason_is_handed_on():
    # ChatMistralAI records finish_reason only when the same event names the
    # model, which a proxy may leave out. The answer itself validated.
    events = _sse(json.dumps(ANSWER)).split("\n\n")
    last = json.loads(events[-3].removeprefix("data: "))
    del last["model"]
    events[-3] = f"data: {json.dumps(last)}"

    def handler(request):
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              text="\n\n".join(events))

    result = generate(PROMPT, _mistral_config(), llm=_mistral(handler))
    assert (result.text, result.stop_reason, result.parse_error) == (ANSWER_TEXT, None, None)


@pytest.mark.parametrize("tail", [
    "",   # the stream closed before its last event
    # An error object in place of the last event: it has no choices, so
    # ChatMistralAI skips it.
    'data: {"object": "error", "message": "Service unavailable"}\n\n',
], ids=["closed", "error-object"])
def test_a_mistral_stream_that_ends_without_a_finish_reason_is_not_an_answer(tail):
    events = _sse(json.dumps(ANSWER)).split("\n\n")[:60]

    def handler(request):
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              text="\n\n".join(events) + "\n\n" + tail)

    with pytest.raises(ProviderBusy, match="part-way without finishing: ask again"):
        generate(PROMPT, _mistral_config(), llm=_mistral(handler))


def test_a_reply_that_is_not_an_event_stream_names_the_address_setting():
    # A proxy, a Wi-Fi login page or a wrong MISTRAL_BASE_URL: asking again cannot help.
    expected = "did not answer as Mistral's API does .*MISTRAL_BASE_URL"
    with pytest.raises(ProviderUnavailable, match=expected) as raised:
        generate(PROMPT, _mistral_config(),
                 llm=_mistral(lambda request: httpx.Response(200, json={"choices": []})))
    assert not isinstance(raised.value, ProviderBusy)


def test_an_address_without_its_scheme_names_the_address_setting():
    llm = ChatMistralAI(model=DEFAULT_MISTRAL_MODEL, api_key="test-key", base_url="api.mistral.ai/v1")
    with pytest.raises(ProviderUnavailable, match=r"UnsupportedProtocol.*MISTRAL_BASE_URL"):
        generate(PROMPT, _mistral_config(), llm=llm)


def _raises(exception_type):
    def handler(request):
        raise exception_type("boom", request=request)
    return handler


@pytest.mark.parametrize("handler, busy", [
    (lambda request: httpx.Response(429, json={"message": "Requests rate limit exceeded"}), True),
    (lambda request: httpx.Response(503, json={"message": "Service unavailable"}), True),
    (_raises(httpx.ConnectError), True),
    (_raises(httpx.ReadTimeout), True),
    (_raises(httpx.RemoteProtocolError), True),
    (lambda request: httpx.Response(401, json={"detail": "Unauthorized"}), False),
    (lambda request: httpx.Response(400, json={"message": "Invalid model: x",
                                               "type": "invalid_model"}), False),
    (lambda request: httpx.Response(422, json={"message": "Invalid request"}), False),
], ids=["429", "503", "connect", "timeout", "dropped", "401", "invalid-model", "422"])
def test_only_a_failure_asking_again_may_fix_is_busy(handler, busy):
    # ProviderBusy is what the evaluation harness asks again on.
    with pytest.raises(ProviderUnavailable) as raised:
        generate(PROMPT, _mistral_config(), llm=_mistral(handler))
    assert isinstance(raised.value, ProviderBusy) is busy


def test_a_failed_mistral_request_is_sent_once():
    # ChatMistralAI's retry wraps building the lazy event-stream iterator, not
    # sending the request, so even its default of five retries sends it once.
    requests = []

    def handler(request):
        requests.append(request)
        raise httpx.ConnectError("boom", request=request)

    llm = ChatMistralAI(model=DEFAULT_MISTRAL_MODEL, api_key="test-key", max_retries=5,
                        client=httpx.Client(base_url=MISTRAL_URL,
                                            transport=httpx.MockTransport(handler)))
    with pytest.raises(ProviderBusy):
        generate(PROMPT, _mistral_config(), llm=llm)
    assert len(requests) == 1


def test_a_chat_model_for_the_other_provider_is_refused():
    def never(request):
        raise AssertionError("no request should be sent")

    with pytest.raises(ValueError, match="ChatOllama but the config records the provider 'mistral'"):
        generate(PROMPT, _mistral_config(), llm=_ollama(never, model=DEFAULT_MISTRAL_MODEL))
    with pytest.raises(ValueError, match="ChatMistralAI but the config records the provider 'ollama'"):
        generate(PROMPT, _config(model=DEFAULT_MISTRAL_MODEL), llm=_mistral(never))


# --- the evaluation command -----------------------------------------------------------------

@pytest.mark.parametrize("provider, client", [("ollama", ChatOllama), ("mistral", ChatMistralAI)])
def test_the_evaluation_command_chooses_the_provider(monkeypatch, tmp_path, clean_env, provider,
                                                     client):
    import src.evaluation.cli as cli
    from src.retrieval.bm25 import BM25Retriever

    seen = {}
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    monkeypatch.setattr(cli, "load_questions", lambda *a, **k: [])
    monkeypatch.setattr(BM25Retriever, "load", lambda **k: object())
    monkeypatch.setattr(cli, "evaluate", lambda questions, retriever, config, llm, **k:
                        seen.update(config=config, llm=llm) or
                        {"summary": {}, "by_answerability": {}, "results": []})
    cli.main(["q.jsonl", "--retriever", "bm25", "--run-id", "r", "--output",
              str(tmp_path / "report.json"), "--provider", provider, "--model", "chosen"])
    assert (seen["config"].provider, seen["config"].model) == (provider, "chosen")
    # The one chat model built before anything loaded answers every question.
    assert isinstance(seen["llm"], client) and seen["llm"].model == "chosen"


@pytest.mark.parametrize("values, expected", [
    ({"LLM_PROVIDER": "mistrl"}, "LLM_PROVIDER must be one of ollama, mistral"),
    ({"LLM_PROVIDER": "mistral", "MISTRAL_API_KEY": ""}, "MISTRAL_API_KEY is not set"),
    ({"LLM_PROVIDER": "ollama", "LLM_NUM_GPU": "all"}, "LLM_NUM_GPU must be a whole number"),
], ids=["provider", "key", "gpu-layers"])
def test_the_evaluation_command_refuses_its_settings_before_loading_anything(
        monkeypatch, tmp_path, clean_env, capsys, values, expected):
    import src.evaluation.cli as cli

    for name, value in values.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(cli, "load_questions", lambda *a, **k: pytest.fail("loaded questions"))
    with pytest.raises(SystemExit) as raised:
        cli.main(["q.jsonl", "--run-id", "r", "--output", str(tmp_path / "report.json")])
    assert raised.value.code == 2
    assert expected in capsys.readouterr().err
