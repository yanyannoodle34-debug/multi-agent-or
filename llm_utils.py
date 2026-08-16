"""Shared foundation for orchestration workflows.

Copy this file into any generated orchestration project. It provides the four
primitives every pattern needs:

    call_llm        - a single, retried, synchronous LLM call
    call_llm_async  - the async version, for fan-out
    parallel_map    - run an async coroutine over many inputs with bounded concurrency
    extract_tag     - pull structured fields out of an XML-tagged response
    with_retry      - decorator: exponential backoff on transient API errors

Design goals: no framework lock-in, model IDs supplied by the caller (from config),
every network call retried, and structured hand-offs via XML tags.

Requires: `pip install anthropic`
Auth:     set the ANTHROPIC_API_KEY environment variable.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import random
import re
import time
from typing import Awaitable, Callable, Iterable, Sequence, TypeVar

from anthropic import Anthropic, AsyncAnthropic
from anthropic import APIStatusError, APIConnectionError, RateLimitError

log = logging.getLogger("orchestration")

T = TypeVar("T")
R = TypeVar("R")

# Reuse clients across calls (connection pooling). Auth comes from ANTHROPIC_API_KEY.
_sync_client = Anthropic()
_async_client = AsyncAnthropic()

# Errors worth retrying: rate limits, connection blips, and 5xx from the API.
_RETRYABLE = (RateLimitError, APIConnectionError)


def with_retry(
    max_attempts: int = 4,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
) -> Callable:
    """Decorator: retry a function on transient API errors with exponential backoff + jitter."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except _RETRYABLE as exc:
                    attempt = _handle_retry(exc, attempt, max_attempts, base_delay, max_delay)
                except APIStatusError as exc:
                    # Retry 5xx only; re-raise 4xx (bad request, auth, etc.).
                    if exc.status_code >= 500:
                        attempt = _handle_retry(exc, attempt, max_attempts, base_delay, max_delay)
                    else:
                        raise

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            attempt = 0
            while True:
                try:
                    return await func(*args, **kwargs)
                except _RETRYABLE as exc:
                    attempt = _handle_retry(exc, attempt, max_attempts, base_delay, max_delay, sleep=False)
                    await asyncio.sleep(_backoff(attempt, base_delay, max_delay))
                except APIStatusError as exc:
                    if exc.status_code >= 500:
                        attempt = _handle_retry(exc, attempt, max_attempts, base_delay, max_delay, sleep=False)
                        await asyncio.sleep(_backoff(attempt, base_delay, max_delay))
                    else:
                        raise

        return async_wrapper if asyncio.iscoroutinefunction(func) else wrapper

    return decorator


def _backoff(attempt: int, base_delay: float, max_delay: float) -> float:
    return min(max_delay, base_delay * (2 ** (attempt - 1))) * (0.5 + random.random())


def _handle_retry(exc, attempt, max_attempts, base_delay, max_delay, sleep=True):
    attempt += 1
    if attempt >= max_attempts:
        log.error("Giving up after %d attempts: %s", attempt, exc)
        raise
    delay = _backoff(attempt, base_delay, max_delay)
    log.warning("Transient error (attempt %d/%d): %s — retrying in %.1fs",
                attempt, max_attempts, exc, delay)
    if sleep:
        time.sleep(delay)
    return attempt


@with_retry()
def call_llm(
    prompt: str,
    *,
    model: str,
    system: str | None = None,
    max_tokens: int = 2048,
    temperature: float = 1.0,
) -> str:
    """One synchronous LLM call. Returns the concatenated text of the response."""
    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        kwargs["system"] = system
    resp = _sync_client.messages.create(**kwargs)
    text = "".join(block.text for block in resp.content if block.type == "text")
    log.debug("call_llm model=%s in=%dch out=%dch", model, len(prompt), len(text))
    return text


@with_retry()
async def call_llm_async(
    prompt: str,
    *,
    model: str,
    system: str | None = None,
    max_tokens: int = 2048,
    temperature: float = 1.0,
) -> str:
    """Async LLM call — use inside parallel_map for fan-out."""
    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        kwargs["system"] = system
    resp = await _async_client.messages.create(**kwargs)
    return "".join(block.text for block in resp.content if block.type == "text")


async def parallel_map(
    func: Callable[[T], Awaitable[R]],
    items: Sequence[T],
    *,
    max_concurrency: int = 5,
) -> list[R]:
    """Run an async function over items with bounded concurrency, preserving order.

    Exceptions are returned in-place (as the item's result) rather than cancelling the
    whole batch — the caller decides how to handle partial failure.
    """
    sem = asyncio.Semaphore(max_concurrency)

    async def _guarded(index: int, item: T):
        async with sem:
            try:
                return index, await func(item)
            except Exception as exc:  # noqa: BLE001 - deliberate: report, don't crash the batch
                log.error("Worker failed on item %d: %s", index, exc)
                return index, exc

    tasks = [_guarded(i, item) for i, item in enumerate(items)]
    results: list[R] = [None] * len(items)  # type: ignore[list-item]
    for coro in asyncio.as_completed(tasks):
        index, value = await coro
        results[index] = value
    return results


def extract_tag(text: str, tag: str, *, default: str | None = None) -> str | None:
    """Extract the inner text of the first <tag>...</tag> in an LLM response.

    Prompt models to wrap structured fields in XML tags, then parse them here — far more
    robust than positional splitting or brittle JSON when the model adds prose around it.
    """
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else default


def extract_all_tags(text: str, tag: str) -> list[str]:
    """Extract every <tag>...</tag> occurrence — handy for lists of subtasks."""
    return [m.strip() for m in re.findall(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL | re.IGNORECASE)]


def run(coro: Awaitable[R]) -> R:
    """Convenience wrapper so entrypoints can call async orchestrators without asyncio boilerplate."""
    return asyncio.run(coro)  # type: ignore[arg-type]
