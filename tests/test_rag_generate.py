"""Generation: one function, two providers, streaming, and the numbers for the metrics track.

Nothing here touches the network. The entry points are tested through a fake
provider; the hosted provider through a fake SDK client that returns the
shapes the SDK's streaming helper does; the local provider through
``httpx.MockTransport``, so its real request and line parsing run against a
canned server.
"""

import dataclasses
import json
import os
from collections.abc import Generator

import httpx
import pytest

from src.rag.constants import (
    ANTHROPIC_EFFORT,
    ANTHROPIC_THINKING_HEADROOM,
    DEFAULT_MODELS,
    DEFAULT_OLLAMA_URL,
    MAX_OUTPUT_TOKENS,
    PROMPT_TEMPLATE_ID,
)
from src.rag.generate import (
    AnthropicProvider,
    OllamaProvider,
    ProviderUnavailable,
    Usage,
    config_from_env,
    generate,
    provider_for,
    stream,
)
from src.rag.prompt import build_prompt
from src.rag.records import Generation, GenerationConfig
from src.retrieval.records import RetrievedPassage

PASSAGE = RetrievedPassage(
    chunk_id="p1", text="Revenue was $100.", score=1.0, rank=1, retriever="hybrid",
    ticker="AAA", company="Alpha Corp", fiscal_year=2024, item="7", title="MD&A",
    url="https://example.test/p1",
)
PROMPT = build_prompt("What was revenue?", [PASSAGE])


def _config(provider: str = "fake", **overrides) -> GenerationConfig:
    fields = dict(provider=provider, model="m", prompt_template_id=PROMPT_TEMPLATE_ID)
    fields.update(overrides)
    return GenerationConfig(**fields)


class FakeProvider:
    """Yields a fixed answer in pieces and reports fixed usage."""

    name = "fake"

    def __init__(self, pieces=("Revenue ", "was $100 ", "[1]."), usage=Usage(10, 3, "end_turn")):
        self.pieces, self.usage, self.calls = pieces, usage, []

    def stream(self, prompt, config, max_tokens) -> Generator[str, None, Usage]:
        self.calls.append((prompt, config, max_tokens))
        for piece in self.pieces:
            yield piece
        return self.usage


# --- generate and stream --------------------------------------------------------

def test_generate_returns_the_text_and_the_usage():
    result = generate(PROMPT, _config(), provider=FakeProvider())
    assert isinstance(result, Generation)
    assert result.text == "Revenue was $100 [1]."
    assert (result.input_tokens, result.output_tokens) == (10, 3)
    assert result.stop_reason == "end_turn"
    assert result.truncated is False
    assert result.config == _config()


def test_generate_measures_latency_and_carries_the_config():
    result = generate(PROMPT, _config(), provider=FakeProvider())
    assert result.latency_ms >= 0
    assert isinstance(result.latency_ms, float)


def test_generate_hands_the_prompt_config_and_ceiling_to_the_provider():
    provider = FakeProvider()
    generate(PROMPT, _config(), provider=provider, max_tokens=99)
    assert provider.calls == [(PROMPT, _config(), 99)]


def test_generate_defaults_the_ceiling_to_the_constant():
    provider = FakeProvider()
    generate(PROMPT, _config(), provider=provider)
    assert provider.calls[0][2] == MAX_OUTPUT_TOKENS


def test_on_token_sees_every_delta_in_order():
    seen = []
    generate(PROMPT, _config(), provider=FakeProvider(), on_token=seen.append)
    assert seen == ["Revenue ", "was $100 ", "[1]."]


def test_stream_yields_deltas_and_returns_the_generation():
    deltas = stream(PROMPT, _config(), provider=FakeProvider())
    seen = []
    while True:
        try:
            seen.append(next(deltas))
        except StopIteration as done:
            result = done.value
            break
    assert seen == ["Revenue ", "was $100 ", "[1]."]
    assert result.text == "".join(seen)


def test_a_provider_that_reports_no_usage_gives_none_not_zero():
    class Silent:
        name = "fake"

        def stream(self, prompt, config, max_tokens):
            yield "x"
            return None

    result = generate(PROMPT, _config(), provider=Silent())
    assert (result.input_tokens, result.output_tokens, result.stop_reason) == (None, None, None)


def test_a_truncated_stop_reason_is_flagged():
    result = generate(PROMPT, _config(), provider=FakeProvider(usage=Usage(1, 2, "max_tokens")))
    assert result.truncated is True
    assert generate(PROMPT, _config(), provider=FakeProvider(usage=Usage(1, 2, "length"))).truncated


def test_a_provider_that_does_not_match_the_config_is_refused():
    with pytest.raises(ValueError, match="does not match config.provider"):
        generate(PROMPT, _config(provider="ollama"), provider=FakeProvider())


def test_a_config_for_another_template_is_refused():
    with pytest.raises(ValueError, match="rendered from template"):
        generate(PROMPT, _config(prompt_template_id="other_v9"), provider=FakeProvider())


# --- the Generation record ---------------------------------------------------------

def test_generation_is_frozen_and_serialises():
    result = generate(PROMPT, _config(), provider=FakeProvider())
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.text = "changed"
    data = result.to_dict()
    assert data["text"] == result.text
    assert data["config"] == _config().to_dict()
    assert data["truncated"] is False
    assert data["refused"] is False
    assert set(data) == {"text", "config", "latency_ms", "input_tokens", "output_tokens",
                         "stop_reason", "truncated", "refused"}
    json.dumps(data)


def test_a_refusal_is_read_apart_from_a_truncation_and_an_abstention():
    refused = generate(PROMPT, _config(), provider=FakeProvider(pieces=(), usage=Usage(5, 0, "refusal")))
    assert refused.refused is True
    assert refused.truncated is False
    assert refused.text == ""
    assert generate(PROMPT, _config(), provider=FakeProvider()).refused is False


def test_generation_refuses_negative_latency():
    with pytest.raises(ValueError, match="latency_ms"):
        Generation("t", _config(), -1.0, None, None, None)


# --- configuration -------------------------------------------------------------------

def test_config_from_env_prefers_the_hosted_api_when_a_key_is_present():
    config = config_from_env(environ={"ANTHROPIC_API_KEY": "sk-test"})
    assert config.provider == "anthropic"
    assert config.model == DEFAULT_MODELS["anthropic"]
    assert config.prompt_template_id == PROMPT_TEMPLATE_ID


def test_config_from_env_falls_back_to_the_local_server_without_a_key():
    config = config_from_env(environ={})
    assert config.provider == "ollama"
    assert config.model == DEFAULT_MODELS["ollama"]


def test_config_from_env_honours_the_variables_and_the_arguments():
    env = {"LLM_PROVIDER": "Ollama", "LLM_MODEL": "mistral:7b", "ANTHROPIC_API_KEY": "sk"}
    assert config_from_env(environ=env).provider == "ollama"
    assert config_from_env(environ=env).model == "mistral:7b"
    explicit = config_from_env(provider="anthropic", model="claude-sonnet-5", environ=env)
    assert (explicit.provider, explicit.model) == ("anthropic", "claude-sonnet-5")


def test_config_from_env_refuses_an_unknown_provider():
    with pytest.raises(ProviderUnavailable, match="unknown provider 'gpt'"):
        config_from_env(environ={"LLM_PROVIDER": "gpt"})


def test_provider_for_refuses_an_unknown_name():
    with pytest.raises(ProviderUnavailable, match="unknown provider"):
        provider_for("gpt")


# --- .env ------------------------------------------------------------------------------
# The path every teammate hits first: a .env written from .env.example has to
# be read on the generation path, not only where configure_edgar runs.

_LLM_VARS = ("ANTHROPIC_API_KEY", "LLM_PROVIDER", "LLM_MODEL", "LLM_BASE_URL")


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


def test_config_from_env_reads_the_dotenv_file(tmp_path, clean_env):
    path = _dotenv(tmp_path, ANTHROPIC_API_KEY="sk-test", LLM_PROVIDER="anthropic",
                   LLM_MODEL="claude-opus-5")
    config = config_from_env(dotenv=path)
    assert (config.provider, config.model) == ("anthropic", "claude-opus-5")


def test_a_key_in_the_dotenv_file_alone_selects_the_hosted_api(tmp_path, clean_env):
    config = config_from_env(dotenv=_dotenv(tmp_path, ANTHROPIC_API_KEY="sk-test"))
    assert config.provider == "anthropic"


def test_a_missing_dotenv_file_falls_back_to_the_local_server(tmp_path, clean_env):
    config = config_from_env(dotenv=tmp_path / "absent.env")
    assert config.provider == "ollama"


def test_a_shell_variable_wins_over_the_dotenv_file(tmp_path, clean_env, monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "from-shell")
    config = config_from_env(dotenv=_dotenv(tmp_path, LLM_MODEL="from-file"))
    assert config.model == "from-shell"


def test_anthropic_provider_finds_the_key_in_the_dotenv_file(tmp_path, clean_env, monkeypatch):
    # The key is only in the file. The provider has to see it, and must not
    # send the teammate back to the .env they already filled in.
    path = _dotenv(tmp_path, ANTHROPIC_API_KEY="sk-test")
    built = {}
    monkeypatch.setattr("anthropic.Anthropic", lambda **kw: built.setdefault("client", object()))
    provider = AnthropicProvider(dotenv=path)
    assert provider._client is built["client"]


def test_anthropic_provider_without_a_key_anywhere_says_what_to_set(tmp_path, clean_env):
    with pytest.raises(ProviderUnavailable, match="ANTHROPIC_API_KEY is not set"):
        AnthropicProvider(dotenv=tmp_path / "absent.env")


def test_ollama_provider_reads_the_base_url_from_the_dotenv_file(tmp_path, clean_env):
    path = _dotenv(tmp_path, LLM_BASE_URL="http://gpu-box:11434/")
    provider = OllamaProvider(client=object(), dotenv=path)
    assert provider._base_url == "http://gpu-box:11434"


# --- the hosted API -----------------------------------------------------------------

class FakeStreamResponse:
    """What the SDK's ``messages.stream`` context manager exposes."""

    def __init__(self, pieces, usage, stop_reason):
        self.text_stream = iter(pieces)
        self._final = type("Final", (), {
            "usage": type("Usage", (), {"input_tokens": usage[0], "output_tokens": usage[1]})(),
            "stop_reason": stop_reason,
        })()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self._final


class FakeAnthropicClient:
    def __init__(self, pieces=("A", "B"), usage=(120, 7), stop_reason="end_turn"):
        self.pieces, self.usage, self.stop_reason, self.requests = pieces, usage, stop_reason, []
        self.messages = self

    def stream(self, **request):
        self.requests.append(request)
        return FakeStreamResponse(self.pieces, self.usage, self.stop_reason)


def test_anthropic_provider_sends_system_and_user_turns_and_reads_usage():
    client = FakeAnthropicClient()
    provider = AnthropicProvider(client=client)
    result = generate(PROMPT, _config("anthropic", model="claude-opus-5"), provider=provider)
    assert client.requests == [{
        "model": "claude-opus-5",
        "max_tokens": MAX_OUTPUT_TOKENS + ANTHROPIC_THINKING_HEADROOM,
        "output_config": {"effort": ANTHROPIC_EFFORT},
        "system": PROMPT.system,
        "messages": [{"role": "user", "content": PROMPT.user}],
    }]
    assert result.text == "AB"
    assert (result.input_tokens, result.output_tokens) == (120, 7)
    assert result.stop_reason == "end_turn"


def test_anthropic_provider_turns_the_effort_down_and_gives_thinking_its_own_room():
    # The model thinks out of the same ceiling as the answer. Left at the
    # default effort, the ceiling meant for the answer can go on thinking.
    client = FakeAnthropicClient()
    generate(PROMPT, _config("anthropic"), provider=AnthropicProvider(client=client), max_tokens=100)
    [request] = client.requests
    assert request["output_config"]["effort"] == "low"
    assert request["max_tokens"] == 100 + ANTHROPIC_THINKING_HEADROOM
    assert "thinking" not in request   # never disabled: that leaks tags into the text


def test_anthropic_provider_reads_a_refusal():
    provider = AnthropicProvider(client=FakeAnthropicClient(pieces=(), stop_reason="refusal"))
    result = generate(PROMPT, _config("anthropic"), provider=provider)
    assert result.refused is True and result.text == ""


def test_anthropic_provider_flags_a_max_tokens_stop():
    provider = AnthropicProvider(client=FakeAnthropicClient(stop_reason="max_tokens"))
    assert generate(PROMPT, _config("anthropic"), provider=provider).truncated


def test_anthropic_provider_refuses_a_temperature():
    provider = AnthropicProvider(client=FakeAnthropicClient())
    with pytest.raises(ValueError, match="does not take a temperature"):
        generate(PROMPT, _config("anthropic", temperature=0.7), provider=provider)


def test_anthropic_provider_streams_deltas_as_they_arrive():
    provider = AnthropicProvider(client=FakeAnthropicClient(pieces=("x", "y", "z")))
    seen = []
    generate(PROMPT, _config("anthropic"), provider=provider, on_token=seen.append)
    assert seen == ["x", "y", "z"]


# --- the local server ------------------------------------------------------------------

def _ollama_lines(pieces, *, prompt_tokens=50, eval_tokens=4, done_reason="stop"):
    """The NDJSON body Ollama streams back from /api/chat."""
    lines = [json.dumps({"message": {"role": "assistant", "content": p}, "done": False}) for p in pieces]
    lines.append(json.dumps({
        "message": {"role": "assistant", "content": ""}, "done": True,
        "done_reason": done_reason, "prompt_eval_count": prompt_tokens, "eval_count": eval_tokens,
    }))
    return "\n".join(lines) + "\n"


def _ollama_provider(handler, base_url=None):
    return OllamaProvider(base_url=base_url, client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_ollama_provider_posts_the_chat_and_parses_the_stream():
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, text=_ollama_lines(["Rev", "enue [1]."]))

    result = generate(
        PROMPT, _config("ollama", model="llama3.1:8b"), provider=_ollama_provider(handler), max_tokens=64
    )
    assert result.text == "Revenue [1]."
    assert (result.input_tokens, result.output_tokens) == (50, 4)
    assert result.stop_reason == "stop"
    assert result.truncated is False

    [request] = requests
    assert request.method == "POST"
    assert str(request.url) == f"{DEFAULT_OLLAMA_URL}/api/chat"
    body = json.loads(request.content)
    assert body["model"] == "llama3.1:8b"
    assert body["stream"] is True
    assert body["messages"] == PROMPT.to_messages()
    assert body["options"] == {"temperature": 0.0, "num_predict": 64}


def test_ollama_provider_passes_the_temperature_through():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content)["options"])
        return httpx.Response(200, text=_ollama_lines(["x"]))

    generate(PROMPT, _config("ollama", temperature=0.4), provider=_ollama_provider(handler))
    assert seen["temperature"] == 0.4


def test_ollama_provider_flags_a_length_stop():
    handler = lambda request: httpx.Response(200, text=_ollama_lines(["x"], done_reason="length"))
    assert generate(PROMPT, _config("ollama"), provider=_ollama_provider(handler)).truncated


def test_ollama_provider_streams_deltas_and_skips_empty_ones():
    handler = lambda request: httpx.Response(200, text=_ollama_lines(["a", "", "b"]))
    seen = []
    generate(PROMPT, _config("ollama"), provider=_ollama_provider(handler), on_token=seen.append)
    assert seen == ["a", "b"]


def test_ollama_provider_reads_the_base_url_from_the_environment(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://gpu-box:11434/")
    seen = []
    handler = lambda request: (seen.append(str(request.url)), httpx.Response(200, text=_ollama_lines(["x"])))[1]
    generate(PROMPT, _config("ollama"), provider=_ollama_provider(handler))
    assert seen == ["http://gpu-box:11434/api/chat"]


def test_ollama_provider_says_when_the_model_is_not_pulled():
    handler = lambda request: httpx.Response(404, json={"error": "model 'nope' not found"})
    with pytest.raises(ProviderUnavailable, match="has no model 'nope'.*ollama pull nope"):
        generate(PROMPT, _config("ollama", model="nope"), provider=_ollama_provider(handler))


@pytest.mark.parametrize("error", [
    httpx.ConnectError("connection refused"),
    httpx.ConnectTimeout("timed out"),      # a host that swallows packets
    httpx.ReadTimeout("timed out"),         # a server that is listening but wedged
])
def test_ollama_provider_says_when_the_server_is_down_or_wedged(error):
    def handler(request):
        error.request = request
        raise error

    with pytest.raises(ProviderUnavailable, match="no usable Ollama server at .*ollama serve"):
        generate(PROMPT, _config("ollama"), provider=_ollama_provider(handler))


def test_ollama_provider_surfaces_a_streamed_error():
    handler = lambda request: httpx.Response(200, text=json.dumps({"error": "out of memory"}) + "\n")
    with pytest.raises(ProviderUnavailable, match="Ollama error: out of memory"):
        generate(PROMPT, _config("ollama"), provider=_ollama_provider(handler))


def test_ollama_provider_raises_on_a_server_error():
    handler = lambda request: httpx.Response(500, text="boom")
    with pytest.raises(httpx.HTTPStatusError):
        generate(PROMPT, _config("ollama"), provider=_ollama_provider(handler))
