"""Generate an answer from a grounded prompt, through whichever provider is configured.

One function, :func:`generate`, behind one small interface, :class:`Provider`.
The hosted API and the local server are two implementations of it, chosen by
``GenerationConfig.provider``, so the question of which one ships is a
configuration change and not a code change: the app, the harness and the
citation resolver call ``generate`` and never see a client.

Every provider streams. The interface is a generator that yields the text as
it arrives and returns the token counts when it is done, so the app can write
tokens into the UI as they come and the metrics track still gets its numbers
at the end. :func:`generate` drives that generator to completion and hands
back a ``Generation``; :func:`stream` exposes the deltas for a caller that
wants them.

Nothing here is retried or cached beyond what the provider's own client does.
A provider that is not reachable -- no key, no server -- raises
:class:`ProviderUnavailable` with a message that says what to set, since the
person who sees it is a teammate on a fresh clone, not the model.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Generator
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Protocol

from .constants import (
    ANTHROPIC,
    ANTHROPIC_KEY_ENV,
    DEFAULT_MODELS,
    DEFAULT_OLLAMA_URL,
    GENERATION_TIMEOUT_S,
    LLM_BASE_URL_ENV,
    LLM_MODEL_ENV,
    LLM_PROVIDER_ENV,
    MAX_OUTPUT_TOKENS,
    OLLAMA,
    PROMPT_TEMPLATE_ID,
)
from .prompt import GroundedPrompt
from .records import Generation, GenerationConfig


class ProviderUnavailable(RuntimeError):
    """The configured provider cannot be reached: no key, no server, or no such name."""


@dataclass(frozen=True)
class Usage:
    """What a provider reports when it finishes: the token counts and why it stopped.

    Each field is None when the provider did not say. The counts are in the
    provider's own tokens, so they compare within a provider and not across.
    """

    input_tokens: int | None = None
    output_tokens: int | None = None
    stop_reason: str | None = None


class Provider(Protocol):
    """One way of turning a prompt into text.

    A protocol rather than a base class, for the reason ``retrieval.base``
    gives: nothing has to inherit from anything, and a test can pass a plain
    class that yields three strings. ``name`` is what ``GenerationConfig``
    records; ``stream`` yields the answer as it arrives and returns the
    :class:`Usage` when it is done, which is what lets one call serve both
    the UI and the metrics track.
    """

    name: str

    def stream(
        self, prompt: GroundedPrompt, config: GenerationConfig, max_tokens: int
    ) -> Generator[str, None, Usage]: ...


# --- the entry points ------------------------------------------------------------

def generate(
    prompt: GroundedPrompt,
    config: GenerationConfig,
    *,
    provider: Provider | None = None,
    max_tokens: int = MAX_OUTPUT_TOKENS,
    on_token: Callable[[str], None] | None = None,
) -> Generation:
    """Run the prompt through the configured provider and return what it wrote.

    ``provider`` defaults to the one ``config.provider`` names; pass one to
    use a client you built, or a fake in a test. ``on_token`` is called with
    each delta as it arrives, for a caller that wants to show progress but
    does not want to drive :func:`stream` itself.

    The latency is the whole call, from the request going out to the last
    token arriving, which is what a user waits for and what the metrics
    track reports.
    """
    deltas = stream(prompt, config, provider=provider, max_tokens=max_tokens)
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
    provider: Provider | None = None,
    max_tokens: int = MAX_OUTPUT_TOKENS,
) -> Generator[str, None, Generation]:
    """Yield the answer as it arrives, and return the ``Generation`` at the end.

    For the app: ``for delta in stream(...)`` writes tokens into the UI, and
    the generator's return value -- read through ``StopIteration.value``, or
    by wrapping in :func:`generate` -- carries the finished record.
    """
    if prompt.template_id != config.prompt_template_id:
        raise ValueError(
            f"prompt was rendered from template {prompt.template_id!r} but the config "
            f"records {config.prompt_template_id!r}; the results row would lie"
        )
    chosen = provider if provider is not None else provider_for(config.provider)
    if chosen.name != config.provider:
        raise ValueError(
            f"provider {chosen.name!r} does not match config.provider {config.provider!r}"
        )

    started = perf_counter()
    parts: list[str] = []
    source = chosen.stream(prompt, config, max_tokens)
    while True:
        try:
            delta = next(source)
        except StopIteration as done:
            usage: Usage = done.value if isinstance(done.value, Usage) else Usage()
            break
        parts.append(delta)
        yield delta
    latency_ms = (perf_counter() - started) * 1000.0
    return Generation(
        text="".join(parts),
        config=config,
        latency_ms=latency_ms,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        stop_reason=usage.stop_reason,
    )


# --- configuration ------------------------------------------------------------------

def config_from_env(
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 0.0,
    environ: dict[str, str] | None = None,
) -> GenerationConfig:
    """The ``GenerationConfig`` the environment asks for.

    Provider: the argument, else ``LLM_PROVIDER``, else "anthropic" when a
    key is present and "ollama" when not, so a fresh clone with no key still
    answers. Model: the argument, else ``LLM_MODEL``, else the provider's
    default. ``environ`` is for tests; it defaults to ``os.environ``.
    """
    env = os.environ if environ is None else environ
    chosen = (provider or env.get(LLM_PROVIDER_ENV) or "").strip().lower()
    if not chosen:
        chosen = ANTHROPIC if env.get(ANTHROPIC_KEY_ENV, "").strip() else OLLAMA
    if chosen not in DEFAULT_MODELS:
        raise ProviderUnavailable(
            f"unknown provider {chosen!r}; set {LLM_PROVIDER_ENV} to one of "
            + ", ".join(sorted(DEFAULT_MODELS))
        )
    return GenerationConfig(
        provider=chosen,
        model=(model or env.get(LLM_MODEL_ENV) or DEFAULT_MODELS[chosen]).strip(),
        prompt_template_id=PROMPT_TEMPLATE_ID,
        temperature=temperature,
    )


def provider_for(name: str, **options: Any) -> Provider:
    """The provider a config names, built from the environment.

    ``options`` go to the provider's constructor: a client for a test, a
    base URL for a server on another machine.
    """
    if name == ANTHROPIC:
        return AnthropicProvider(**options)
    if name == OLLAMA:
        return OllamaProvider(**options)
    raise ProviderUnavailable(
        f"unknown provider {name!r}; expected one of " + ", ".join(sorted(DEFAULT_MODELS))
    )


# --- the hosted API ----------------------------------------------------------------

class AnthropicProvider:
    """The hosted API, through the official SDK.

    The prompt's system text goes in ``system`` and its user text as the one
    user turn, which is the shape the API takes. The SDK's streaming helper
    yields the text deltas and accumulates the final message, whose ``usage``
    carries the token counts.

    Temperature is not sent. The current Claude models do not take a
    sampling temperature -- the request is rejected -- and there is no
    substitute knob worth pretending is one, so a config asking for anything
    but 0.0 is refused here rather than silently ignored: the config is a
    record of what was actually used.
    """

    name = ANTHROPIC

    def __init__(self, client: Any | None = None, timeout: float = GENERATION_TIMEOUT_S):
        # The SDK is imported here rather than at module level so that the
        # local provider works on a machine without it installed, and so a
        # missing key is reported as a ProviderUnavailable with the variable
        # to set rather than as the SDK's own exception at first call.
        if client is None:
            import anthropic

            if not os.environ.get(ANTHROPIC_KEY_ENV, "").strip():
                raise ProviderUnavailable(
                    f"{ANTHROPIC_KEY_ENV} is not set; put it in .env, or set "
                    f"{LLM_PROVIDER_ENV}={OLLAMA} to use the local server"
                )
            client = anthropic.Anthropic(timeout=timeout)
        self._client = client

    def stream(
        self, prompt: GroundedPrompt, config: GenerationConfig, max_tokens: int
    ) -> Generator[str, None, Usage]:
        if config.temperature != 0.0:
            raise ValueError(
                f"the {ANTHROPIC} provider does not take a temperature; "
                f"got {config.temperature}, set 0.0"
            )
        with self._client.messages.stream(
            model=config.model,
            max_tokens=max_tokens,
            system=prompt.system,
            messages=[{"role": "user", "content": prompt.user}],
        ) as response:
            for delta in response.text_stream:
                yield delta
            final = response.get_final_message()
        usage = getattr(final, "usage", None)
        return Usage(
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
            stop_reason=getattr(final, "stop_reason", None),
        )


# --- the local server ----------------------------------------------------------------

class OllamaProvider:
    """A local Ollama server, through its HTTP API.

    ``POST /api/chat`` with ``stream: true`` answers with one JSON object per
    line: each carries the next piece of the message, and the last one, with
    ``done`` true, carries the token counts and the reason it stopped. No
    SDK: the API is three fields, and ``httpx`` is already a dependency.

    The prompt goes over as the two chat messages it renders to, so the
    system rules and the sources reach the model the same way they reach the
    hosted one. Temperature is passed through, and ``num_predict`` is
    Ollama's name for the output ceiling.
    """

    name = OLLAMA

    def __init__(
        self,
        base_url: str | None = None,
        client: Any | None = None,
        timeout: float = GENERATION_TIMEOUT_S,
    ):
        import httpx

        self._base_url = (base_url or os.environ.get(LLM_BASE_URL_ENV) or DEFAULT_OLLAMA_URL).rstrip("/")
        self._client = client if client is not None else httpx.Client(timeout=timeout)

    def stream(
        self, prompt: GroundedPrompt, config: GenerationConfig, max_tokens: int
    ) -> Generator[str, None, Usage]:
        import httpx

        body = {
            "model": config.model,
            "messages": prompt.to_messages(),
            "stream": True,
            "options": {"temperature": config.temperature, "num_predict": max_tokens},
        }
        try:
            with self._client.stream("POST", f"{self._base_url}/api/chat", json=body) as response:
                if response.status_code == 404:
                    # Ollama answers 404 for a model it has not pulled, with
                    # the reason in the body. Read it so the message says so.
                    response.read()
                    raise ProviderUnavailable(
                        f"Ollama at {self._base_url} has no model {config.model!r}: "
                        f"{_ollama_error(response)}. Run: ollama pull {config.model}"
                    )
                response.raise_for_status()
                usage = Usage()
                for line in response.iter_lines():
                    if not line.strip():
                        continue
                    event = json.loads(line)
                    if "error" in event:
                        raise ProviderUnavailable(f"Ollama error: {event['error']}")
                    delta = event.get("message", {}).get("content", "")
                    if delta:
                        yield delta
                    if event.get("done"):
                        usage = Usage(
                            input_tokens=event.get("prompt_eval_count"),
                            output_tokens=event.get("eval_count"),
                            stop_reason=event.get("done_reason"),
                        )
                return usage
        except httpx.ConnectError as error:
            raise ProviderUnavailable(
                f"no Ollama server at {self._base_url} ({error}); start it with "
                f"`ollama serve`, or set {LLM_BASE_URL_ENV}, or set "
                f"{LLM_PROVIDER_ENV}={ANTHROPIC} with an {ANTHROPIC_KEY_ENV}"
            ) from error


def _ollama_error(response: Any) -> str:
    """The error string in an Ollama error body, or the raw text if it is not JSON."""
    try:
        return response.json().get("error", response.text)
    except ValueError:
        return response.text
