"""Shared foundation for the orchestration workflow.

Provides the primitives every step needs, independent of which LLM provider is active:

    call_llm        - a single, retried, synchronous LLM call
    call_llm_async  - the async version, for fan-out
    parallel_map    - run an async coroutine over many inputs with bounded concurrency
    extract_tag     - pull structured fields out of an XML-tagged response
    extract_all_tags- pull every occurrence of a tag (lists of assignments/subtasks)
    run             - asyncio.run wrapper for entrypoints

`call_llm` / `call_llm_async` delegate to `providers`, which owns the actual SDK client
and retry-with-backoff and can target either the native Anthropic SDK or an
OpenAI-compatible endpoint (NVIDIA NIM, etc.). Select the backend once at startup with
`providers.configure(...)` (see `orchestrator.activate`).
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Awaitable, Callable, Sequence, TypeVar

import providers

log = logging.getLogger("orchestration")

T = TypeVar("T")
R = TypeVar("R")


def call_llm(
    prompt: str,
    *,
    model: str,
    system: str | None = None,
    max_tokens: int = 2048,
    temperature: float = 1.0,
) -> str:
    """One synchronous LLM call via the active provider. Returns the response text."""
    text = providers.complete(
        prompt, model=model, system=system, max_tokens=max_tokens, temperature=temperature
    )
    log.debug("call_llm model=%s in=%dch out=%dch", model, len(prompt), len(text))
    return text


async def call_llm_async(
    prompt: str,
    *,
    model: str,
    system: str | None = None,
    max_tokens: int = 2048,
    temperature: float = 1.0,
) -> str:
    """Async LLM call via the active provider — use inside parallel_map for fan-out."""
    return await providers.acomplete(
        prompt, model=model, system=system, max_tokens=max_tokens, temperature=temperature
    )


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
    """Extract every <tag>...</tag> occurrence — handy for lists of assignments/subtasks."""
    return [m.strip() for m in re.findall(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL | re.IGNORECASE)]


def run(coro: Awaitable[R]) -> R:
    """Convenience wrapper so entrypoints can call async orchestrators without asyncio boilerplate."""
    return asyncio.run(coro)  # type: ignore[arg-type]
