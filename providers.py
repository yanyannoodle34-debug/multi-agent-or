"""LLM provider backends: Anthropic (native) and OpenAI-compatible (NVIDIA NIM, etc.).

Both backends expose the same pair of calls the orchestrator uses, so the rest of the
code stays provider-agnostic:

    complete(prompt, *, model, system, max_tokens, temperature) -> str          (sync)
    acomplete(prompt, *, model, system, max_tokens, temperature) -> str          (async)

Select the active backend once at startup with `configure(name, provider_cfg)`.
`llm_utils.call_llm` / `call_llm_async` delegate here.

Providers:
  - "anthropic"          native Anthropic SDK.       Auth: ANTHROPIC_API_KEY
  - anything else         OpenAI-compatible Chat API. Auth: <api_key_env> (default NVIDIA_API_KEY)
    (e.g. "nvidia")       NVIDIA NIM base_url: https://integrate.api.nvidia.com/v1
                          Get a FREE key at https://build.nvidia.com

Clients are built lazily on the first real call, so `configure()` (and resolving model
IDs) is safe to run offline / in tests without an API key or network.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import time

log = logging.getLogger("orchestration.providers")

# Retry policy for transient errors (rate limits, connection blips, 5xx).
_MAX_ATTEMPTS = 4
_BASE_DELAY = 1.0
_MAX_DELAY = 30.0


def _backoff(attempt: int) -> float:
    return min(_MAX_DELAY, _BASE_DELAY * (2 ** (attempt - 1))) * (0.5 + random.random())


class _Backend:
    """A provider backend. Subclasses build their SDK client and implement the two calls."""

    name = "base"

    def is_retryable(self, exc: Exception) -> bool:  # noqa: ARG002
        return False

    def complete(self, prompt, *, model, system, max_tokens, temperature) -> str:
        raise NotImplementedError

    async def acomplete(self, prompt, *, model, system, max_tokens, temperature) -> str:
        raise NotImplementedError


class _AnthropicBackend(_Backend):
    name = "anthropic"

    def __init__(self, cfg: dict):
        from anthropic import (
            Anthropic,
            APIConnectionError,
            APIStatusError,
            AsyncAnthropic,
            RateLimitError,
        )

        # max_retries=0: this module owns retry-with-backoff (see `complete`/`acomplete`),
        # so disable the SDK's own retry layer to avoid compounding (up to 4 x the SDK's
        # default per call).
        self._sync = Anthropic(max_retries=0)
        self._async = AsyncAnthropic(max_retries=0)
        self._always = (RateLimitError, APIConnectionError)
        self._status = APIStatusError

    def is_retryable(self, exc: Exception) -> bool:
        if isinstance(exc, self._always):
            return True
        if isinstance(exc, self._status):
            return exc.status_code >= 500  # retry 5xx; surface 4xx (auth, bad request)
        return False

    def _kwargs(self, prompt, system, max_tokens, temperature, model):
        kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        return kwargs

    def complete(self, prompt, *, model, system, max_tokens, temperature) -> str:
        resp = self._sync.messages.create(**self._kwargs(prompt, system, max_tokens, temperature, model))
        return "".join(b.text for b in resp.content if b.type == "text")

    async def acomplete(self, prompt, *, model, system, max_tokens, temperature) -> str:
        resp = await self._async.messages.create(**self._kwargs(prompt, system, max_tokens, temperature, model))
        return "".join(b.text for b in resp.content if b.type == "text")


class _OpenAICompatBackend(_Backend):
    """OpenAI-compatible Chat Completions backend — covers NVIDIA NIM and similar."""

    name = "openai-compatible"

    def __init__(self, cfg: dict):
        from openai import (
            APIConnectionError,
            APIStatusError,
            AsyncOpenAI,
            OpenAI,
            RateLimitError,
        )

        base_url = cfg.get("base_url")
        key_env = cfg.get("api_key_env", "NVIDIA_API_KEY")
        api_key = os.environ.get(key_env)
        if not api_key:
            raise RuntimeError(
                f"Set the {key_env} environment variable for the "
                f"'{cfg.get('_name', 'openai')}' provider "
                f"(free keys: NVIDIA https://build.nvidia.com , "
                f"OpenRouter https://openrouter.ai/keys)."
            )
        # Optional per-provider headers — e.g. OpenRouter's HTTP-Referer / X-Title
        # used for app attribution and its model-usage rankings.
        headers = cfg.get("headers") or None
        # max_retries=0: this module owns retry-with-backoff (see `complete`/`acomplete`),
        # so disable the SDK's own retry layer to avoid compounding retries.
        self._sync = OpenAI(base_url=base_url, api_key=api_key, default_headers=headers, max_retries=0)
        self._async = AsyncOpenAI(base_url=base_url, api_key=api_key, default_headers=headers, max_retries=0)
        self._always = (RateLimitError, APIConnectionError)
        self._status = APIStatusError

    def is_retryable(self, exc: Exception) -> bool:
        if isinstance(exc, self._always):
            return True
        if isinstance(exc, self._status):
            return exc.status_code >= 500
        return False

    def _messages(self, prompt, system):
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        return msgs

    def complete(self, prompt, *, model, system, max_tokens, temperature) -> str:
        resp = self._sync.chat.completions.create(
            model=model,
            messages=self._messages(prompt, system),
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""

    async def acomplete(self, prompt, *, model, system, max_tokens, temperature) -> str:
        resp = await self._async.chat.completions.create(
            model=model,
            messages=self._messages(prompt, system),
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""


# --- active-provider state (client built lazily on first call) ---------------------------

_active_name: str | None = None
_active_cfg: dict | None = None
_backend: _Backend | None = None


def configure(name: str, provider_cfg: dict | None = None) -> None:
    """Select the active provider. The SDK client is built lazily on the first call."""
    global _active_name, _active_cfg, _backend
    _active_name = name
    _active_cfg = dict(provider_cfg or {})
    _active_cfg["_name"] = name
    _backend = None
    log.info("Provider set to %r (client builds on first call).", name)


def active_name() -> str | None:
    return _active_name


def _get_backend() -> _Backend:
    global _backend
    if _backend is None:
        if _active_name is None:
            raise RuntimeError("No provider configured; call providers.configure(...) first.")
        if _active_name == "anthropic":
            _backend = _AnthropicBackend(_active_cfg or {})
        else:  # any other name is treated as OpenAI-compatible (NVIDIA NIM, etc.)
            _backend = _OpenAICompatBackend(_active_cfg or {})
        log.info("Built %s client for provider %r.", _backend.name, _active_name)
    return _backend


def complete(prompt: str, **kw) -> str:
    """Synchronous completion via the active provider, with retry on transient errors."""
    backend = _get_backend()
    attempt = 0
    while True:
        try:
            return backend.complete(prompt, **kw)
        except Exception as exc:  # noqa: BLE001 - re-raised below unless retryable
            if not backend.is_retryable(exc):
                raise
            attempt += 1
            if attempt >= _MAX_ATTEMPTS:
                log.error("Giving up after %d attempts: %s", attempt, exc)
                raise
            delay = _backoff(attempt)
            log.warning("Transient error (attempt %d/%d): %s — retrying in %.1fs",
                        attempt, _MAX_ATTEMPTS, exc, delay)
            time.sleep(delay)


async def acomplete(prompt: str, **kw) -> str:
    """Async completion via the active provider, with retry on transient errors."""
    backend = _get_backend()
    attempt = 0
    while True:
        try:
            return await backend.acomplete(prompt, **kw)
        except Exception as exc:  # noqa: BLE001
            if not backend.is_retryable(exc):
                raise
            attempt += 1
            if attempt >= _MAX_ATTEMPTS:
                log.error("Giving up after %d attempts: %s", attempt, exc)
                raise
            await asyncio.sleep(_backoff(attempt))
