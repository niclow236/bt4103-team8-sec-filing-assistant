"""Generate an answer from a grounded prompt, with a local model or Mistral's API.

Two providers answer, chosen with ``LLM_PROVIDER`` in .env (#106). Ollama, the
default, runs a model on the machine running the code, with no account, key or
network. Mistral's hosted API answers in seconds rather than minutes, with each
teammate's own key. One function, :func:`generate`, runs a ``GroundedPrompt``
through the configured model and returns a ``Generation``: the answer, how long
it took, and how many tokens it used.

Two libraries do the work, each where it earns its place.

LangChain is the client: ``ChatOllama`` for Ollama, ``ChatMistralAI`` for
Mistral. Each streams and reports the token counts, and both implement the
chat-model interface every LangChain model shares. That interface is the
provider interface here: :func:`generate` takes any chat model, so a test passes
a fake one, and the two providers differ only in how a request is sent and how a
failure reads, both kept in this module.

Pydantic fixes the shape of the answer. The JSON schema of
``GroundedAnswer.for_sources(n)`` is sent with every request, as Ollama's output
format or as Mistral's strict JSON-schema response format, and both constrain
decoding to it: the model cannot leave out the source numbers, and cannot cite a
source it was not shown. The same Pydantic model then validates what came back.

The model writes JSON, and nobody wants to watch JSON arrive. :func:`stream`
reads the partial JSON as it grows and yields the answer as prose, adding a
sentence's markers once that sentence is closed, so the app can show a
readable answer while it is written, and the text it ends with is
``Generation.text``.

Every request carries its own options (temperature, output ceiling, and for
Ollama the context window) from the ``GenerationConfig`` and ``constants.py``,
so an answer is produced by the settings it records, whichever chat model was
passed in. Nothing is retried. A server that is not running, a model that is not
pulled, a missing or refused key, a rate limit or a response that does not
arrive in time raises :class:`ProviderUnavailable` with what to do about it,
since the person who reads it is a teammate on a fresh clone.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Generator, Mapping
from pathlib import Path
from time import perf_counter
from typing import Any

from langchain_core.utils.json import parse_partial_json
from pydantic import ValidationError

from ..config import ENV_FILE, load_env
from .constants import (
    ABSTAIN_PHRASE,
    DEFAULT_MISTRAL_MODEL,
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_URL,
    DEFAULT_PROVIDER,
    GENERATION_TIMEOUT_S,
    HOSTED_TIMEOUT_S,
    LLM_BASE_URL_ENV,
    LLM_MODEL_ENV,
    LLM_NUM_GPU_ENV,
    LLM_PROVIDER_ENV,
    MAX_OUTPUT_TOKENS,
    MISTRAL,
    MISTRAL_API_KEY_ENV,
    NUM_CTX,
    OLLAMA,
    PROMPT_TEMPLATE_ID,
    PROVIDERS,
)
from .prompt import GroundedPrompt
from .records import Generation, GenerationConfig, GroundedAnswer, render_sentence

# Where a teammate makes their own Mistral key, named in every message that asks
# for one.
MISTRAL_CONSOLE = "https://console.mistral.ai"
MISTRAL_API_URL = "https://api.mistral.ai/v1"


class ProviderUnavailable(RuntimeError):
    """The model cannot answer: no server or connection, no such model, a missing
    or refused key, a rate limit, or no response in time."""


# --- the entry points ------------------------------------------------------------

def generate(
    prompt: GroundedPrompt,
    config: GenerationConfig,
    *,
    llm: Any | None = None,
    max_tokens: int = MAX_OUTPUT_TOKENS,
    on_token: Callable[[str], None] | None = None,
) -> Generation:
    """Run the prompt through the configured model and return what it wrote.

    ``llm`` defaults to :func:`chat_model` for the config; pass a LangChain
    chat model to use one you built, or a fake in a test. ``on_token`` is
    called with each piece of prose as it arrives, for a caller that wants
    to show progress without driving :func:`stream` itself.

    The latency is the whole call, from the request going out to the last
    token arriving, which is what a user waits for. On a model's first call
    that includes Ollama loading it into memory.
    """
    deltas = stream(prompt, config, llm=llm, max_tokens=max_tokens)
    while True:
        try:
            delta = next(deltas)
        except StopIteration as done:
            return done.value
        if on_token is not None:
            on_token(delta)


def stream(
    prompt: GroundedPrompt,
    config: GenerationConfig,
    *,
    llm: Any | None = None,
    max_tokens: int = MAX_OUTPUT_TOKENS,
) -> Generator[str, None, Generation]:
    """Yield the answer as prose while it is written, and return the ``Generation``.

    For the app: ``for delta in stream(...)`` writes the answer into the UI,
    and the generator's return value carries the finished record, read
    through ``StopIteration.value`` or by calling :func:`generate` instead.
    The deltas only ever extend what was shown, and joined they are
    ``Generation.text``.
    """
    if prompt.template_id != config.prompt_template_id:
        raise ValueError(
            f"prompt was rendered from template {prompt.template_id!r} but the config "
            f"records {config.prompt_template_id!r}; the results row would lie"
        )
    model = llm if llm is not None else chat_model(config)
    served = getattr(model, "model", None)
    if served is not None and served != config.model:
        raise ValueError(
            f"the chat model serves {served!r} but the config records {config.model!r}"
        )

    schema = GroundedAnswer.for_sources(prompt.n_sources)
    request = _request(config, model, schema, max_tokens)

    started = perf_counter()
    raw = ""
    shown = ""
    merged = None
    try:
        for chunk in model.stream(prompt.to_messages(), **request):
            merged = chunk if merged is None else merged + chunk
            piece = _text(chunk.content)
            if not piece:
                continue
            raw += piece
            prose = _prose(raw, complete=False)
            if len(prose) > len(shown) and prose.startswith(shown):
                yield prose[len(shown):]
                shown = prose
    except Exception as error:
        unavailable = _unavailable(error, model, config)
        if unavailable is None:
            raise
        raise unavailable from error
    latency_ms = (perf_counter() - started) * 1000.0

    answer, parse_error = _validate(raw, schema)
    if answer is not None:
        text = answer.render()
    else:
        text = _prose(raw, complete=_is_json(raw))
        if not text.startswith(shown):
            # Cut inside an escape, the last sentence cannot be read at all,
            # so what was already shown is the most of the answer there is.
            text = shown
    if len(text) > len(shown) and text.startswith(shown):
        yield text[len(shown):]

    usage = getattr(merged, "usage_metadata", None) or {}
    metadata = getattr(merged, "response_metadata", None) or {}
    return Generation(
        text=text,
        answer=answer,
        raw=raw,
        config=config,
        latency_ms=latency_ms,
        input_tokens=usage.get("input_tokens"),
        output_tokens=usage.get("output_tokens"),
        # Ollama calls it done_reason and Mistral finish_reason; both say
        # "length" for an answer cut off at the output ceiling.
        stop_reason=metadata.get("done_reason") or metadata.get("finish_reason"),
        parse_error=parse_error,
    )


def _request(
    config: GenerationConfig, model: Any, schema: type[GroundedAnswer], max_tokens: int
) -> dict[str, Any]:
    """What goes to the chat model with the messages: the schema and the settings.

    The same schema and settings for both providers, in the form each API takes.
    Ollama constrains decoding to the schema given as its output ``format``, and
    takes the settings as ``options``, the context window among them, since its
    own default window is smaller than the prompt. Mistral takes the schema as a
    strict JSON-schema response format and the settings as named fields; its
    window is the model's own.
    """
    if config.provider == MISTRAL:
        return {
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "GroundedAnswer",
                    "schema": schema.model_json_schema(),
                    "strict": True,
                },
            },
            "temperature": config.temperature,
            "max_tokens": max_tokens,
        }
    options = {"temperature": config.temperature, "num_ctx": NUM_CTX, "num_predict": max_tokens}
    # Per-request options replace the chat model's own, so the one setting that
    # belongs to the machine rather than to the config is carried across.
    num_gpu = getattr(model, "num_gpu", None)
    if num_gpu is not None:
        options["num_gpu"] = num_gpu
    return {"format": schema.model_json_schema(), "options": options}


def _text(content: Any) -> str:
    """The text of one streamed piece, whether it came as a string or as blocks.

    Ollama's pieces are strings. Mistral's can be a list of typed blocks, and
    only the text blocks are the answer.
    """
    if isinstance(content, str):
        return content
    return "".join(
        block if isinstance(block, str) else str(block.get("text", ""))
        for block in content or ()
        if isinstance(block, str)
        or (isinstance(block, dict) and block.get("type", "text") == "text")
    )


# --- configuration ------------------------------------------------------------------

def _environment(environ: Mapping[str, str] | None, dotenv: Path) -> Mapping[str, str]:
    """The settings to read: the mapping given, else the process environment
    with the local .env loaded into it first.

    Loading here rather than at import is what makes .env.example true: every
    variable it documents is read on the path that uses it. A variable already
    set in the shell wins over the file.
    """
    if environ is not None:
        return environ
    load_env(dotenv)
    return os.environ


def config_from_env(
    model: str | None = None,
    temperature: float = 0.0,
    environ: Mapping[str, str] | None = None,
    dotenv: Path = ENV_FILE,
    provider: str | None = None,
) -> GenerationConfig:
    """The ``GenerationConfig`` the environment asks for.

    Provider: the argument, else ``LLM_PROVIDER``, else ``DEFAULT_PROVIDER``
    (Ollama). Model: the argument, else ``LLM_MODEL``, else that provider's
    default, ``DEFAULT_MODEL`` for Ollama and ``DEFAULT_MISTRAL_MODEL`` for
    Mistral. The environment is the process's, with ``dotenv`` (the project's
    .env) loaded into it first; ``environ`` replaces both, for a test.
    """
    env = _environment(environ, dotenv)
    chosen_provider = (provider or env.get(LLM_PROVIDER_ENV) or "").strip().lower() or DEFAULT_PROVIDER
    if chosen_provider not in PROVIDERS:
        raise ValueError(
            f"{LLM_PROVIDER_ENV} must be one of {', '.join(PROVIDERS)}, got {chosen_provider!r}"
        )
    default = DEFAULT_MISTRAL_MODEL if chosen_provider == MISTRAL else DEFAULT_MODEL
    chosen = (model or env.get(LLM_MODEL_ENV) or "").strip() or default
    return GenerationConfig(
        provider=chosen_provider,
        model=chosen,
        prompt_template_id=PROMPT_TEMPLATE_ID,
        temperature=temperature,
    )


def chat_model(
    config: GenerationConfig,
    *,
    base_url: str | None = None,
    dotenv: Path = ENV_FILE,
) -> Any:
    """The LangChain chat model for a config: ``ChatOllama`` or ``ChatMistralAI``.

    For Ollama, the server is ``base_url``, else ``LLM_BASE_URL`` from the
    environment or .env, else Ollama's default address. ``LLM_NUM_GPU``, when
    set, says how many layers go on the GPU; see ``constants.LLM_NUM_GPU_ENV``
    for when to set it. For Mistral, see :func:`_mistral_model`; ``base_url``
    does not apply. The generation options are not set here: :func:`stream`
    sends them with every request.

    Each client library is imported here rather than at the top of the module
    because it takes seconds to import, and code that only parses a question
    or renders a prompt should not pay for it.
    """
    if config.provider == MISTRAL:
        return _mistral_model(config, dotenv)
    if config.provider != OLLAMA:
        raise ProviderUnavailable(
            f"unknown provider {config.provider!r}: set {LLM_PROVIDER_ENV} to one of "
            f"{', '.join(PROVIDERS)}"
        )
    from langchain_ollama import ChatOllama

    load_env(dotenv)
    if base_url is None:
        base_url = os.environ.get(LLM_BASE_URL_ENV) or DEFAULT_OLLAMA_URL
    layers = os.environ.get(LLM_NUM_GPU_ENV, "").strip()
    if layers and not layers.isdigit():
        raise ValueError(f"{LLM_NUM_GPU_ENV} must be a whole number of layers, got {layers!r}")
    return ChatOllama(
        model=config.model,
        base_url=base_url.rstrip("/"),
        num_gpu=int(layers) if layers else None,
        client_kwargs={"timeout": GENERATION_TIMEOUT_S},
    )


def _mistral_model(config: GenerationConfig, dotenv: Path) -> Any:
    """``ChatMistralAI`` for the config, with the key from the environment or .env.

    The key is each teammate's own, from their own Mistral account, so a missing
    one is refused here, with where to make one, rather than on the first request,
    where the API would only answer 401. One attempt per request, as for Ollama:
    the client's own retries would hide an outage behind minutes of waiting.
    """
    from langchain_mistralai import ChatMistralAI

    load_env(dotenv)
    key = (os.environ.get(MISTRAL_API_KEY_ENV) or "").strip()
    if not key:
        raise ProviderUnavailable(
            f"the provider is {MISTRAL!r} but {MISTRAL_API_KEY_ENV} is not set: make a key with "
            f"your own account at {MISTRAL_CONSOLE} (API Keys) and put it in your .env, as "
            f"'Setting up Mistral' in the README says, or set {LLM_PROVIDER_ENV}={OLLAMA} "
            f"to answer locally"
        )
    return ChatMistralAI(
        model=config.model,
        api_key=key,
        temperature=config.temperature,
        max_tokens=MAX_OUTPUT_TOKENS,
        max_retries=1,
        timeout=HOSTED_TIMEOUT_S,
    )


# --- reading the output --------------------------------------------------------------

def _prose(raw: str, *, complete: bool) -> str:
    """The answer as prose, as far as the JSON written so far says it.

    ``complete`` is whether ``raw`` is the model's whole output. Until it is,
    the last sentence may still be being written: its text is shown as it
    grows, but its markers wait until the sentence is closed, since a
    half-written "[1" could yet become "[12]". A sentence is closed once the
    next one has begun. So what this returns only ever extends what it
    returned for a shorter prefix of the same output, which is what lets
    :func:`stream` show it as it comes.
    """
    try:
        data = parse_partial_json(raw)
    except ValueError:
        return ""
    if not isinstance(data, dict):
        return ""
    if data.get("answerable") is False:
        return ABSTAIN_PHRASE
    items = data.get("sentences")
    if not isinstance(items, list):
        items = []
    parts = []
    for position, item in enumerate(items):
        text = item.get("text") if isinstance(item, dict) else None
        if not isinstance(text, str):
            break
        closed = complete or position < len(items) - 1
        sources = item.get("sources") if closed else None
        numbers = (
            [n for n in sources if isinstance(n, int) and not isinstance(n, bool)]
            if isinstance(sources, list)
            else []
        )
        if text.strip():
            parts.append(render_sentence(text, numbers))
    prose = " ".join(parts)
    if complete and not prose:
        return ABSTAIN_PHRASE
    return prose


def _is_json(raw: str) -> bool:
    """Whether the output is a whole JSON document, as opposed to one cut off."""
    try:
        json.loads(raw)
    except ValueError:
        return False
    return True


def _validate(
    raw: str, schema: type[GroundedAnswer]
) -> tuple[GroundedAnswer | None, str | None]:
    """The output validated against the schema it was decoded under, or why not.

    A valid answer is handed on as a plain ``GroundedAnswer``, so that two
    answers compare equal whatever source count they were validated against.
    """
    try:
        parsed = schema.model_validate_json(raw)
    except ValidationError as error:
        problems = [
            f"{'.'.join(str(part) for part in problem['loc']) or 'output'}: {problem['msg']}"
            for problem in error.errors()[:3]
        ]
        return None, "; ".join(problems)
    return GroundedAnswer.model_validate(parsed.model_dump()), None


def _unavailable(error: Exception, model: Any, config: GenerationConfig) -> ProviderUnavailable | None:
    """The error to raise in place of one from the client, or None to let it through.

    The Ollama client does not wrap every failure the same way on a streamed
    request: a refused connection arrives as httpx's ``ConnectError``, not
    the client's own ``ConnectionError``, and a model that has not been pulled
    as a ``ResponseError`` with status 404. Each becomes a message that says
    what to run.
    """
    if config.provider == MISTRAL:
        return _mistral_unavailable(error, model, config)
    import httpx
    from ollama import ResponseError

    url = getattr(model, "base_url", None) or DEFAULT_OLLAMA_URL
    if isinstance(error, (httpx.ConnectError, ConnectionError)):
        return ProviderUnavailable(
            f"no Ollama server at {url}: start the Ollama app or run `ollama serve` on "
            f"this computer, or set {LLM_BASE_URL_ENV} in .env if you changed its port"
        )
    if isinstance(error, httpx.TimeoutException):
        return ProviderUnavailable(
            f"Ollama at {url} sent nothing for {GENERATION_TIMEOUT_S:.0f}s while running "
            f"{config.model!r}; on this machine it needs a smaller model (set {LLM_MODEL_ENV}) "
            f"or a longer GENERATION_TIMEOUT_S"
        )
    if isinstance(error, ResponseError):
        if error.status_code == 404:
            return ProviderUnavailable(
                f"Ollama at {url} has no model {config.model!r}: run `ollama pull {config.model}`"
            )
        return ProviderUnavailable(f"Ollama could not run {config.model!r}: {error.error}")
    return None


def _mistral_unavailable(
    error: Exception, model: Any, config: GenerationConfig
) -> ProviderUnavailable | None:
    """The error to raise in place of one from Mistral's client, or None.

    The client raises httpx's own errors: ``HTTPStatusError`` for a response the
    API refused, with its body already read, and a timeout or network error when
    no response came. Each becomes a message that says what to do.
    """
    import httpx

    if isinstance(error, httpx.HTTPStatusError):
        status = error.response.status_code
        detail = _error_detail(error.response)
        # What the API said, where it said anything: "Your API key expired on
        # 2026-09-26" tells a teammate more than any status code.
        said = f"HTTP {status}: {detail}" if detail else f"HTTP {status}"
        if status == 403 and "tier" in detail.lower():
            return ProviderUnavailable(
                f"your Mistral plan does not include {config.model!r} ({said}); set "
                f"{LLM_MODEL_ENV} to a model it does, such as {DEFAULT_MISTRAL_MODEL}, or unset it"
            )
        if status in (401, 403):
            return ProviderUnavailable(
                f"Mistral refused the API key ({said}); check {MISTRAL_API_KEY_ENV} in your .env, "
                f"or make a new key with your own account at {MISTRAL_CONSOLE} (API Keys)"
            )
        if status == 429:
            return ProviderUnavailable(
                f"Mistral's rate limit for {config.model!r} was reached ({said}); wait a minute "
                f"and ask again, since the limits are per account"
            )
        if status in (400, 404, 422) and "model" in detail.lower():
            return ProviderUnavailable(
                f"Mistral could not run {config.model!r} ({said}); set {LLM_MODEL_ENV} to a "
                f"Mistral model, such as {DEFAULT_MISTRAL_MODEL}, or unset it"
            )
        return ProviderUnavailable(f"Mistral's API refused the request ({said})")
    url = getattr(model, "endpoint", None) or MISTRAL_API_URL
    if isinstance(error, httpx.TimeoutException):
        return ProviderUnavailable(
            f"Mistral's API at {url} sent nothing for {HOSTED_TIMEOUT_S}s while running "
            f"{config.model!r}: ask again, or set {LLM_PROVIDER_ENV}={OLLAMA} to answer locally"
        )
    if isinstance(error, (httpx.NetworkError, ConnectionError)):
        return ProviderUnavailable(
            f"cannot reach Mistral's API at {url}: check the internet connection, or set "
            f"{LLM_PROVIDER_ENV}={OLLAMA} to answer locally"
        )
    return None


def _error_detail(response: Any) -> str:
    """What an error response said, shortly: its message field where it has one.

    Mistral's API puts it in ``message`` for most errors and in ``detail`` for a
    refused key. The closing full stop is dropped, since the text is quoted
    inside a longer message.
    """
    try:
        text = response.text
    except Exception:
        return ""
    try:
        body = json.loads(text)
    except ValueError:
        return text.strip()[:300].rstrip(".")
    if isinstance(body, dict):
        for field in ("message", "detail", "error"):
            value = body.get(field)
            if isinstance(value, str) and value.strip():
                return value.strip()[:300].rstrip(".")
    return text.strip()[:300].rstrip(".")
