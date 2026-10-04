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
a fake one. What differs between the two providers (how the client is built, how
a request is sent, how a failure and a stop read) is kept in one ``_Provider``
each, at the end of this module.

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
passed in. Nothing is retried here. A server that is not running, a model that
is not pulled, a missing or refused key, a rate limit or a response that does
not arrive in time raises :class:`ProviderUnavailable` with what to do about
it, since the person who reads it is a teammate on a fresh clone. The failures
that asking again may fix raise :class:`ProviderBusy`, a kind of
``ProviderUnavailable``, which the evaluation harness asks again on.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Generator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from functools import lru_cache
from math import ceil, isfinite
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
    MISTRAL_API_URL,
    MISTRAL_BASE_URL_ENV,
    MISTRAL_CONSOLE,
    NUM_CTX,
    OLLAMA,
    PROMPT_TEMPLATE_ID,
    PROVIDERS,
)
from .prompt import GroundedPrompt
from .records import Generation, GenerationConfig, GroundedAnswer, render_sentence


class ProviderUnavailable(RuntimeError):
    """The model cannot answer: no server or connection, no such model, a missing
    or refused key, a rate limit or a used-up quota, or no response in time.

    ``reason`` is what is wrong on its own, in a few words, without the advice
    the message goes on to give. It is for a caller with advice of its own:
    the app says it beside the provider picked, where "restart the notebook
    or command" is not what there is to do. The whole message, where a
    failure names none.
    """

    def __init__(self, message: str, *, reason: str | None = None) -> None:
        super().__init__(message)
        self.reason = reason if reason is not None else message


class ProviderBusy(ProviderUnavailable):
    """A failure that asking again may fix: a rate limit, a server error, a
    connection that broke off or never came, or an answer that did not finish.

    ``retry_after`` is how many seconds the provider asked to be left before
    the next request, from the ``Retry-After`` of a 429 or a 5xx, or None where
    it did not say.
    """

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


# What every Mistral failure offers instead.
_ANSWER_LOCALLY = f"set {LLM_PROVIDER_ENV}={OLLAMA} to answer locally"


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

    With ``llm`` given, ``config.provider`` must still be one of ``PROVIDERS``,
    since the request's shape and the reading of its failures depend on it. A
    fake chat model has to end its stream as the provider's own does, with a
    ``finish_reason`` (Mistral) or ``done_reason`` (Ollama), or else with output
    that is a whole JSON document, or with a token count that reached
    ``max_tokens``, which reads as an answer cut off at the ceiling.
    """
    if prompt.template_id != config.prompt_template_id:
        raise ValueError(
            f"prompt was rendered from template {prompt.template_id!r} but the config "
            f"records {config.prompt_template_id!r}; the results row would lie"
        )
    provider = _provider(config)
    model = llm if llm is not None else chat_model(config)
    served = getattr(model, "model", None)
    if served is not None and served != config.model:
        raise ValueError(
            f"the chat model serves {served!r} but the config records {config.model!r}"
        )
    client = _client_package(model)
    if client is not None and client != provider.package:
        raise ValueError(
            f"the chat model is a {type(model).__name__} but the config records the "
            f"provider {config.provider!r}"
        )

    schema = GroundedAnswer.for_sources(prompt.n_sources)
    request = provider.request(config, model, schema, max_tokens)

    started = perf_counter()
    raw = ""
    shown = ""
    # Why the answer ended and what it used, each from the last chunk that
    # says so. Merging the chunks would join a "stop" sent twice into
    # "stopstop" and add a repeated token count to itself.
    ended: Mapping[str, Any] = {}
    usage: Mapping[str, Any] = {}
    try:
        for chunk in model.stream(prompt.to_messages(), **request):
            metadata = getattr(chunk, "response_metadata", None) or {}
            if metadata.get("finish_reason") or metadata.get("done_reason"):
                ended = metadata
            usage = getattr(chunk, "usage_metadata", None) or usage
            piece = _text(chunk.content)
            if not piece:
                continue
            raw += piece
            prose = _prose(raw, complete=False)
            if len(prose) > len(shown) and prose.startswith(shown):
                yield prose[len(shown):]
                shown = prose
    except Exception as error:
        unavailable = provider.unavailable(error, model, config)
        if unavailable is None:
            raise
        # A connection that drops after the chunk saying why the answer ended,
        # while the client reads on for the stream's end (Mistral's "data:
        # [DONE]"), has lost nothing, so the answer is kept. Before that chunk,
        # what arrived is part of an answer, and asking again may finish it.
        if not (ended and isinstance(unavailable, ProviderBusy)):
            raise unavailable from error
    latency_ms = (perf_counter() - started) * 1000.0
    whole = _is_json(raw)
    if not ended and not whole and (usage.get("output_tokens") or 0) >= max_tokens:
        # Cut at the output ceiling rather than dropped: a proxy can lose the
        # reason, but the count still says every allowed token was used.
        ended = {"finish_reason": "length", "done_reason": "length"}
    stop_reason = provider.stop_reason(ended, config, whole)

    answer, parse_error = _validate(raw, schema)
    if answer is not None:
        text = answer.render()
    else:
        text = _prose(raw, complete=whole)
        if not text.startswith(shown):
            # Cut inside an escape, the last sentence cannot be read at all,
            # so what was already shown is the most of the answer there is.
            text = shown
    if len(text) > len(shown) and text.startswith(shown):
        yield text[len(shown):]

    return Generation(
        text=text,
        answer=answer,
        raw=raw,
        config=config,
        latency_ms=latency_ms,
        input_tokens=usage.get("input_tokens"),
        output_tokens=usage.get("output_tokens"),
        stop_reason=stop_reason,
        parse_error=parse_error,
    )


def _client_package(model: Any) -> str | None:
    """The LangChain package a chat model's client comes from, or None for a fake.

    Read from the class and its bases, so a subclass of ``ChatOllama`` counts as
    Ollama's. :func:`stream` refuses a model whose package is not the config's
    provider's, since the request it would send is the other API's.
    """
    for cls in type(model).__mro__:
        package = cls.__module__.split(".")[0]
        if package in _PACKAGES:
            return package
    return None


def _text(content: Any) -> str:
    """The text of one streamed piece, whether it came as a string or as blocks.

    Ollama's pieces are strings. Mistral's can be a list of typed blocks, and
    only the text blocks are the answer.
    """
    if isinstance(content, str):
        return content
    return "".join(
        block if isinstance(block, str) else str(block.get("text") or "")
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
    (Ollama). Model: the argument, else ``LLM_MODEL`` when the provider is the
    one the environment names, else that provider's default, ``DEFAULT_MODEL``
    for Ollama and ``DEFAULT_MISTRAL_MODEL`` for Mistral. ``LLM_MODEL`` is set
    for the provider in .env, so ``--provider mistral`` beside an Ollama
    ``LLM_MODEL`` asks Mistral for its default rather than for a model it does
    not serve. The environment is the process's, with ``dotenv`` (the project's
    .env) loaded into it first; ``environ`` replaces both, for a test.
    """
    env = _environment(environ, dotenv)
    # Checked only when it is the one used, so a typo in .env does not block
    # choosing a provider with the argument, and refused under the name of
    # whichever of the two it came from.
    configured = _provider_setting(env.get(LLM_PROVIDER_ENV))
    if (provider or "").strip():
        chosen_provider = _provider_name(provider, "the provider argument")
    else:
        chosen_provider = _provider_name(configured, LLM_PROVIDER_ENV)
    default = DEFAULT_MISTRAL_MODEL if chosen_provider == MISTRAL else DEFAULT_MODEL
    configured_model = env.get(LLM_MODEL_ENV) if chosen_provider == configured else None
    chosen = (model or configured_model or "").strip() or default
    return GenerationConfig(
        provider=chosen_provider,
        model=chosen,
        prompt_template_id=PROMPT_TEMPLATE_ID,
        temperature=temperature,
    )


def _provider_setting(value: str | None) -> str:
    """A provider setting as written: trimmed, lower case, blank for the default."""
    return (value or "").strip().lower() or DEFAULT_PROVIDER


def _provider_name(value: str | None, setting: str) -> str:
    """A provider setting as one of ``PROVIDERS``, refused where the config is
    built if it names none of them, under the name of the ``setting`` it came
    from."""
    name = _provider_setting(value)
    if name not in PROVIDERS:
        raise ValueError(f"{setting} must be one of {', '.join(PROVIDERS)}, got {name!r}")
    return name


def chat_model(
    config: GenerationConfig,
    *,
    base_url: str | None = None,
    environ: Mapping[str, str] | None = None,
    dotenv: Path = ENV_FILE,
) -> Any:
    """The LangChain chat model for a config: ``ChatOllama`` or ``ChatMistralAI``.

    For Ollama, the server is ``base_url``, else ``LLM_BASE_URL`` from the
    environment or .env, else Ollama's default address. ``LLM_NUM_GPU``, when
    set, says how many layers go on the GPU; see ``constants.LLM_NUM_GPU_ENV``
    for when to set it. For Mistral, see :func:`_mistral_settings`, which refuses
    ``base_url``, since Mistral's address comes from ``MISTRAL_BASE_URL``. The
    settings are read as :func:`config_from_env` reads them: the process
    environment with ``dotenv`` loaded into it, or ``environ`` in place of both.
    The generation options are not set here: :func:`stream` sends them with
    every request.

    Each client library is imported here rather than at the top of the module
    because it takes seconds to import, and code that only parses a question
    or renders a prompt should not pay for it.
    """
    return _provider(config).build(config, base_url, _environment(environ, dotenv))


def check_provider(
    config: GenerationConfig,
    *,
    base_url: str | None = None,
    environ: Mapping[str, str] | None = None,
    dotenv: Path = ENV_FILE,
) -> None:
    """Refuse the settings :func:`chat_model` would refuse, and build nothing.

    For a caller that has to say early that a provider cannot answer, and may
    never need its client. The app offers both providers and builds a chat
    model only when an answer needs one, since a figure looked up in the facts
    store needs none; it says beside the provider picked that its key is
    missing, before a question is asked. That check runs on every rerun of the
    page, and building ``ChatOllama`` to make it took 0.8 s to import its
    library and 16 ms and two HTTP clients each time after.

    Each provider checks its settings in one function, which this and its
    ``build`` both call, so what is refused here is what :func:`chat_model`
    refuses, with the same words. What only a request can show is not checked:
    a server that is not running, a key the API refuses.
    """
    _provider(config).settings(config, base_url, _environment(environ, dotenv))


def _ollama_settings(
    config: GenerationConfig, base_url: str | None, env: Mapping[str, str],
) -> dict[str, Any]:
    """What ``ChatOllama`` is built with, on the server :func:`chat_model` describes."""
    if base_url is None:
        base_url = (env.get(LLM_BASE_URL_ENV) or "").strip() or DEFAULT_OLLAMA_URL
    layers = (env.get(LLM_NUM_GPU_ENV) or "").strip()
    if layers and not layers.isdigit():
        raise ValueError(f"{LLM_NUM_GPU_ENV} must be a whole number of layers, got {layers!r}")
    return {
        "model": config.model,
        "base_url": base_url.rstrip("/"),
        "num_gpu": int(layers) if layers else None,
        "client_kwargs": {"timeout": GENERATION_TIMEOUT_S},
    }


def _ollama_model(config: GenerationConfig, base_url: str | None, env: Mapping[str, str]) -> Any:
    """``ChatOllama`` for the config, with the settings :func:`_ollama_settings` read."""
    from langchain_ollama import ChatOllama

    return ChatOllama(**_ollama_settings(config, base_url, env))


def _mistral_settings(
    config: GenerationConfig, base_url: str | None, env: Mapping[str, str],
) -> tuple[str, str, str]:
    """What ``ChatMistralAI`` is built with: the model, the key and the address.

    The key is each teammate's own, from their own Mistral account, so a missing
    one is refused here, with where to make one, rather than on the first request,
    where the API would only answer 401.

    ``base_url`` is refused rather than ignored: it is the Ollama server's
    address, and a caller that passes one expects its requests to go there.
    """
    if base_url is not None:
        raise ValueError(
            f"base_url is the Ollama server's address; Mistral's comes from {MISTRAL_BASE_URL_ENV}"
        )
    key = (env.get(MISTRAL_API_KEY_ENV) or "").strip()
    if not key:
        # A key pasted into .env after a blank MISTRAL_API_KEY= line was read
        # is not seen, since a variable that is set wins over .env, even "".
        lacks = f"{MISTRAL_API_KEY_ENV} is not set"
        raise ProviderUnavailable(
            f"the provider is {MISTRAL!r} but {lacks}: make a key with "
            f"your own account at {MISTRAL_CONSOLE} (API Keys) and put it in your .env, as "
            f"'Setting up Mistral' in the README says, then restart the notebook or command so "
            f"the key is read, or {_ANSWER_LOCALLY}",
            reason=lacks,
        )
    # The address given to the client in every case, so a setting read from
    # ``environ`` is not passed over for the process's own MISTRAL_BASE_URL,
    # which ChatMistralAI reads when it is given none.
    address = (env.get(MISTRAL_BASE_URL_ENV) or "").strip() or MISTRAL_API_URL
    return config.model, key, address


def _mistral_model(config: GenerationConfig, base_url: str | None, env: Mapping[str, str]) -> Any:
    """``ChatMistralAI`` for the config, with the key from the environment or .env.

    The same model, key and address get the same client back, so a run of
    questions reuses one connection to the API rather than opening two new HTTP
    clients, and a new TLS handshake inside the measured latency, for every
    question.
    """
    return _mistral_client(*_mistral_settings(config, base_url, env))


@lru_cache(maxsize=8)
def _mistral_client(model: str, key: str, base_url: str) -> Any:
    """One ``ChatMistralAI`` per model, key and address, built on first use.

    The temperature and the output ceiling are not set here: :func:`stream`
    sends them with every request, as it does for Ollama. Set here as well,
    they would only add ChatMistralAI's own check that the temperature is
    within 0 to 1, stricter than the API's 1.5 and raised outside the error
    handling. ``max_retries`` only affects ``invoke``: ChatMistralAI's retry
    wraps building the lazy event-stream iterator, so a streamed request is
    sent once whatever it is set to, as for Ollama.
    """
    from langchain_mistralai import ChatMistralAI

    return ChatMistralAI(
        model=model,
        api_key=key,
        base_url=base_url,
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
        return None, _problems(error.errors(), whole="output")
    return GroundedAnswer.model_validate(parsed.model_dump()), None


def _problems(problems: Sequence[Mapping[str, Any]], *, whole: str) -> str:
    """The first three validation problems, each as where it is and what is wrong.

    Pydantic's errors and a Mistral 422's are the same shape, a ``loc`` and a
    ``msg``: "sentences.0.sources.0: Input should be less than or equal to 3".
    A problem with no location, or a null one, is in ``whole``, and a location
    given as one value rather than a list, as a proxy may send it, is one part.
    Nothing here may raise: it runs while a provider's error is being read, and
    would hide that error.
    """
    def where(loc: Any) -> str:
        parts = loc if isinstance(loc, (list, tuple)) else () if loc is None else [loc]
        return ".".join(str(part) for part in parts) or whole

    return "; ".join(f"{where(problem.get('loc'))}: {problem['msg']}" for problem in problems[:3])


def _ollama_unavailable(
    error: Exception, model: Any, config: GenerationConfig
) -> ProviderUnavailable | None:
    """The error to raise in place of one from Ollama's client, or None to let it through.

    The Ollama client does not wrap every failure the same way on a streamed
    request: a refused connection arrives as httpx's ``ConnectError``, not
    the client's own ``ConnectionError``, a model that has not been pulled
    as a ``ResponseError`` with status 404, a full queue as one with status
    503, a proxy in ``HTTP_PROXY`` that refuses as httpx's ``ProxyError``, and
    a connection Ollama closes part-way through an answer as httpx's
    ``RemoteProtocolError`` or a ``NetworkError``. Each becomes a message that
    says what to run, and the ones asking again may fix are
    :class:`ProviderBusy`. Any other transport error, such as a malformed
    request, fails the same way every time and comes through as itself.
    """
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
        # 503 is Ollama's queue being full, more requests waiting than
        # OLLAMA_MAX_QUEUE allows, which asking again may fix, as for Mistral's
        # 5xx. Its 500s, such as a model too big for memory, are not.
        if error.status_code == 503:
            return ProviderBusy(
                f"Ollama at {url} is busy running {config.model!r} ({error.error}): ask again"
            )
        return ProviderUnavailable(f"Ollama could not run {config.model!r}: {error.error}")
    # A proxy set for the whole machine that also catches this computer's own
    # address: it refuses the same way on every request.
    if isinstance(error, httpx.ProxyError):
        return ProviderUnavailable(
            f"a proxy refused the connection to Ollama at {url} ({error}): add 127.0.0.1 to "
            f"NO_PROXY, or unset HTTP_PROXY, so requests to this computer skip the proxy"
        )
    # Ollama closing the connection part-way, as it does when the process
    # running the model stops, or the connection failing mid-answer.
    if isinstance(error, (httpx.RemoteProtocolError, httpx.NetworkError)):
        return ProviderBusy(
            f"the connection to Ollama at {url} broke off while running {config.model!r} "
            f"({type(error).__name__}: {error}): check that Ollama is still running, and "
            f"ask again"
        )
    return None


def _mistral_unavailable(
    error: Exception, model: Any, config: GenerationConfig
) -> ProviderUnavailable | None:
    """The error to raise in place of one from Mistral's client, or None.

    The client raises httpx's own errors: ``HTTPStatusError`` for a response the
    API refused, with its body already read, and a transport error when no
    response came or the connection broke off part-way. Each becomes a message
    that says what to do, and the ones asking again may fix are
    :class:`ProviderBusy`. Any other transport error, such as a malformed
    request, fails the same way every time and comes through as itself.
    """
    import httpx
    from httpx_sse import SSEError

    # ChatMistralAI always sets its endpoint; a fake chat model has none.
    url = getattr(model, "endpoint", None)
    api = f"Mistral's API at {url}" if url else "Mistral's API"
    if isinstance(error, httpx.HTTPStatusError):
        status = error.response.status_code
        detail, kind = _error_detail(error.response)
        # What the API said, where it said anything: "Your API key expired on
        # 2026-09-26" tells a teammate more than any status code.
        said = f"HTTP {status}: {detail}" if detail else f"HTTP {status}"
        # Mistral refuses in JSON. A 401 or 403 page in anything else, such as
        # a school firewall's or Cloudflare's HTML, never reached Mistral, so
        # the key was not what it refused.
        content_type = error.response.headers.get("content-type", "")
        if status in (401, 403) and content_type and "json" not in content_type:
            return ProviderUnavailable(
                f"something other than {api} refused the request ({said}), such as a firewall "
                f"or a proxy on this network, so the key was not checked: try another network, "
                f"or unset {MISTRAL_BASE_URL_ENV} if it is set, since nobody needs to set it"
            )
        if status == 403 and "tier" in detail.lower():
            return ProviderUnavailable(
                f"your Mistral plan does not include {config.model!r} ({said}); set "
                f"{LLM_MODEL_ENV} to a model it does, such as {DEFAULT_MISTRAL_MODEL}, or unset it"
            )
        if status in (401, 403):
            # A key already read stays in the environment, since a variable
            # that is set wins over .env, so a replaced key needs a new process.
            return ProviderUnavailable(
                f"Mistral refused the API key ({said}); check {MISTRAL_API_KEY_ENV} in your .env, "
                f"or make a new key with your own account at {MISTRAL_CONSOLE} (API Keys), then "
                f"restart the notebook or command so the new key is read"
            )
        if status == 429:
            # Every 429 is taken as a limit a wait may lift, the per-minute rate
            # limit among them. A used-up quota comes back the same way, but
            # its wording is not known here, and guessing it could stop a run
            # that a wait would have saved. The harness asks again at most
            # twice, and a stopped run keeps its answers, so that costs at most
            # one round of waiting.
            retry_after = _retry_after(error.response)
            wait = f"wait {ceil(retry_after)}s" if retry_after else "wait a minute"
            return ProviderBusy(
                f"Mistral's rate limit for {config.model!r} was reached ({said}); {wait} and "
                f"ask again, since the limits are per account",
                retry_after=retry_after,
            )
        # Mistral's own name for the error, since a message can mention a model
        # for other reasons: "too large for model with 131072 maximum context".
        if kind == "invalid_model":
            return ProviderUnavailable(
                f"Mistral could not run {config.model!r} ({said}); set {LLM_MODEL_ENV} to a "
                f"Mistral model, such as {DEFAULT_MISTRAL_MODEL}, or unset it"
            )
        # An outage rather than anything wrong with the request. A 503 may say
        # how long it will last, as a 429 does.
        if status >= 500:
            return ProviderBusy(
                f"Mistral's API failed while running {config.model!r} ({said}): ask again, "
                f"or {_ANSWER_LOCALLY}",
                retry_after=_retry_after(error.response),
            )
        return ProviderUnavailable(f"Mistral's API refused the request ({said})")
    if isinstance(error, httpx.TimeoutException):
        return ProviderBusy(
            f"{api} sent nothing for {HOSTED_TIMEOUT_S}s while running "
            f"{config.model!r}: ask again, or {_ANSWER_LOCALLY}"
        )
    if isinstance(error, (httpx.ConnectError, ConnectionError)):
        why = f" ({error})" if str(error) else ""
        # A network whose proxy re-signs HTTPS fails every request this way.
        if "CERTIFICATE_VERIFY_FAILED" in str(error):
            return ProviderUnavailable(
                f"cannot reach {api}{why}: its certificate did not verify, as happens on a "
                f"network whose proxy re-signs HTTPS, so asking again will not help: try "
                f"another network, or {_ANSWER_LOCALLY}"
            )
        # Anywhere but Mistral's own address, the address is the likelier cause
        # than being offline, and a mistyped one fails on every attempt.
        if url and url.rstrip("/") != MISTRAL_API_URL:
            return ProviderUnavailable(
                f"cannot reach {api}{why}, the address {MISTRAL_BASE_URL_ENV} sets: correct it, "
                f"or unset it, since nobody needs to set it"
            )
        # Offline, or a name that did not resolve, which looks the same.
        return ProviderBusy(
            f"cannot reach {api}{why}: check the internet connection, or {_ANSWER_LOCALLY}"
        )
    # A proxy set in HTTPS_PROXY that refuses the connection does so every time.
    if isinstance(error, httpx.ProxyError):
        return ProviderUnavailable(
            f"a proxy refused the connection to {api} ({error}): check HTTPS_PROXY in the "
            f"environment, or unset it"
        )
    # A reply that is not an event stream (SSEError, itself a TransportError),
    # or an address without http(s)://: something other than Mistral's API
    # answered, so asking again cannot help. A proxy or a Wi-Fi login page can
    # do that as well as a wrong MISTRAL_BASE_URL.
    if isinstance(error, (SSEError, httpx.UnsupportedProtocol)):
        return ProviderUnavailable(
            f"{api} did not answer as Mistral's API does ({type(error).__name__}: {error}): "
            f"unset {MISTRAL_BASE_URL_ENV} if it is set, since nobody needs to set it, or sign "
            f"in to the network if it has a login page"
        )
    # A connection dropped part-way through an answer: RemoteProtocolError, or
    # a NetworkError such as a failed read.
    if isinstance(error, (httpx.RemoteProtocolError, httpx.NetworkError)):
        return ProviderBusy(
            f"the connection to {api} broke off ({type(error).__name__}: {error}): ask "
            f"again, or {_ANSWER_LOCALLY}"
        )
    return None


def _ollama_stop_reason(
    metadata: Mapping[str, Any], config: GenerationConfig, whole: bool
) -> str | None:
    """Why Ollama stopped, as its ``done_reason`` says: "length" at the output ceiling.

    Ollama sends ``done_reason`` on its last chunk only. Without it, and without
    a whole JSON document (see :func:`_mistral_stop_reason`), the stream ended
    part-way, as when a proxy or a restarted server closes it cleanly, and it
    raises ``ProviderBusy`` rather than handing on the half that arrived. An
    answer whose ``eval_count`` reached the ceiling has already been read as
    "length" by :func:`stream`, since a server that sends no reason, such as an
    older one, still sends the count.
    """
    reason = metadata.get("done_reason")
    if reason is None and not whole:
        raise ProviderBusy(
            f"Ollama stopped {config.model!r} part-way without finishing: check that Ollama "
            f"is still running, and ask again"
        )
    return reason


def _mistral_stop_reason(
    metadata: Mapping[str, Any], config: GenerationConfig, whole: bool
) -> str | None:
    """Why Mistral stopped, as its ``finish_reason`` says, where the answer finished.

    "stop" is a finished answer, and "length" and "model_length" one cut off at
    a limit, which the ``Generation`` records as truncated. Anything else is an
    answer that did not finish, and raises instead of handing on the half that
    arrived. "error" is the API failing part-way, and no reason at all is a
    stream that ended before its last event, or whose last event was an error
    object with no ``choices``, which ChatMistralAI skips: asking again may fix
    either, so they are ``ProviderBusy``. Any other reason comes back the same
    for the same prompt, so it is a plain ``ProviderUnavailable``.

    ``whole`` is whether the output is a whole JSON document. With no reason,
    that is an answer whose last event lost its reason rather than a cut
    stream, whose JSON never closes: ChatMistralAI records ``finish_reason``
    only when the same event names the model, which a proxy set through
    ``MISTRAL_BASE_URL`` may leave out. It is handed on with no reason, and
    with a ``parse_error`` where it fails the schema, as it would be with one.
    An answer cut at the ceiling that lost its reason the same way has already
    been read as "length" by :func:`stream`, from its token count, which
    ChatMistralAI records whether or not the event names the model.
    """
    reason = metadata.get("finish_reason")
    if reason in ("stop", "length", "model_length"):
        return reason
    if reason is None and whole:
        return None
    if reason in (None, "error"):
        why = "with an error" if reason == "error" else "without finishing"
        raise ProviderBusy(
            f"Mistral's API stopped {config.model!r} part-way {why}: ask again, or "
            f"{_ANSWER_LOCALLY}"
        )
    raise ProviderUnavailable(
        f"Mistral's API stopped {config.model!r} without finishing (finish_reason "
        f"{reason!r}); {_ANSWER_LOCALLY}"
    )


def _error_detail(response: Any) -> tuple[str, str]:
    """What an error response said, shortly, and Mistral's ``type`` for the error.

    Mistral's API puts the text in ``message`` for most errors and in ``detail``
    for a refused key. A request that fails validation (422) has a ``message``
    holding a list of problems under ``detail``, each read as where it is and
    what is wrong; the same list straight under ``detail``, as a proxy built on
    FastAPI sends it, is read the same way. An OpenAI-style proxy, such as
    LiteLLM, nests the whole error under ``error``, which is read in its place.
    An HTML page, such as a firewall's, is named as one rather than quoted. The
    closing full stop is dropped, since the text is quoted inside a longer
    message. The type is "" where the body gives none.
    """
    def short(value: str) -> str:
        return value.strip()[:300].rstrip(".")

    try:
        text = response.text
    except Exception:
        return "", ""
    try:
        body = json.loads(text)
    except (ValueError, RecursionError):
        if "html" in response.headers.get("content-type", ""):
            return "an HTML page", ""
        return short(text), ""
    if not isinstance(body, dict):
        return short(text), ""
    # An OpenAI-style proxy nests the error: {"error": {"message": ..., "type": ...}}.
    if isinstance(body.get("error"), dict):
        body = body["error"]
    kind = body["type"] if isinstance(body.get("type"), str) else ""
    for field in ("message", "detail", "error"):
        value = body.get(field)
        if isinstance(value, str) and value.strip():
            return short(value), kind
        # A 422's problems: under message.detail as Mistral sends them, or
        # straight under detail as a FastAPI proxy would.
        listed = value.get("detail") if isinstance(value, dict) else value
        if isinstance(listed, list):
            problems = [
                problem for problem in listed
                if isinstance(problem, dict) and isinstance(problem.get("msg"), str)
            ]
            if problems:
                return short(_problems(problems, whole="request")), kind
    return short(text), kind


def _retry_after(response: Any) -> float | None:
    """The seconds a ``Retry-After`` header asks for, given as a number of seconds
    or as an HTTP date, or None where there is none or it cannot be read."""
    value = (response.headers.get("retry-after") or "").strip()
    if not value:
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            when = parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        seconds = (when - datetime.now(timezone.utc)).total_seconds()
    return max(seconds, 0.0) if isfinite(seconds) else None


# --- the providers ---------------------------------------------------------------

def _ollama_request(
    config: GenerationConfig, model: Any, schema: type[GroundedAnswer], max_tokens: int
) -> dict[str, Any]:
    """What goes to Ollama with the messages: the schema and the settings.

    Ollama constrains decoding to the schema given as its output ``format``, and
    takes the settings as ``options``, the context window among them, since its
    own default window is smaller than the prompt.
    """
    options = {"temperature": config.temperature, "num_ctx": NUM_CTX, "num_predict": max_tokens}
    # Per-request options replace the chat model's own, so the one setting that
    # belongs to the machine rather than to the config is carried across.
    num_gpu = getattr(model, "num_gpu", None)
    if num_gpu is not None:
        options["num_gpu"] = num_gpu
    return {"format": schema.model_json_schema(), "options": options}


def _mistral_request(
    config: GenerationConfig, model: Any, schema: type[GroundedAnswer], max_tokens: int
) -> dict[str, Any]:
    """What goes to Mistral with the messages: the same schema and settings as
    for Ollama, the schema as a strict JSON-schema response format and the
    settings as named fields. Its context window is the model's own."""
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


@dataclass(frozen=True)
class _Provider:
    """Everything that differs between the providers, for one of them.

    ``package`` is the LangChain package its chat model comes from; ``build``
    makes that model for a config, from the settings :func:`chat_model` read;
    ``settings`` checks those settings and returns what ``build`` builds with,
    building nothing itself, which is all :func:`check_provider` runs;
    ``request`` is what goes with the messages;
    ``unavailable`` turns the client's error into a ``ProviderUnavailable``, or
    None to let it through; ``stop_reason`` reads why the answer ended, from the
    last chunk's metadata and whether the output is a whole JSON document, and
    raises where the answer did not finish.
    """

    package: str
    build: Callable[[GenerationConfig, str | None, Mapping[str, str]], Any]
    settings: Callable[[GenerationConfig, str | None, Mapping[str, str]], Any]
    request: Callable[[GenerationConfig, Any, type[GroundedAnswer], int], dict[str, Any]]
    unavailable: Callable[[Exception, Any, GenerationConfig], ProviderUnavailable | None]
    stop_reason: Callable[[Mapping[str, Any], GenerationConfig, bool], str | None]


_PROVIDERS = {
    OLLAMA: _Provider("langchain_ollama", _ollama_model, _ollama_settings, _ollama_request,
                      _ollama_unavailable, _ollama_stop_reason),
    MISTRAL: _Provider("langchain_mistralai", _mistral_model, _mistral_settings,
                       _mistral_request, _mistral_unavailable, _mistral_stop_reason),
}
_PACKAGES = frozenset(provider.package for provider in _PROVIDERS.values())


def _provider(config: GenerationConfig) -> _Provider:
    """The provider a config records, refused with what to set if it is unknown."""
    try:
        return _PROVIDERS[config.provider]
    except KeyError:
        raise ProviderUnavailable(
            f"unknown provider {config.provider!r}: set {LLM_PROVIDER_ENV} to one of "
            f"{', '.join(PROVIDERS)}"
        ) from None
