"""Partial- vs total-failure handling in run_task (the 'graceful degradation' contract)."""

from __future__ import annotations

import asyncio

import pytest

import orchestrator as o
from llm_utils import parallel_map


def _router_two() -> str:
    return ("<analysis>x</analysis>"
            "<assignment><agent>sales</agent><instructions>a</instructions></assignment>"
            "<assignment><agent>engineering</agent><instructions>b</instructions></assignment>")


def test_partial_failure_still_synthesizes(cfg, monkeypatch):
    monkeypatch.setattr(o, "call_llm", lambda p, **k: _router_two() if "<task>" in p else "[deliverable]")

    async def one_fails(prompt, **kw):
        if "engineering" in prompt.lower():
            raise RuntimeError("engineering boom")
        return "[sales ok]"

    monkeypatch.setattr(o, "call_llm_async", one_fails)
    result = asyncio.run(o.run_task("task", cfg, memory=None))
    assert result.deliverable == "[deliverable]"
    failed = [isinstance(x, Exception) for x in result.outputs]
    assert failed.count(True) == 1 and failed.count(False) == 1


def test_all_failures_raise(cfg, monkeypatch):
    monkeypatch.setattr(o, "call_llm", lambda p, **k: _router_two() if "<task>" in p else "[deliverable]")

    async def all_fail(prompt, **kw):
        raise RuntimeError("boom")

    monkeypatch.setattr(o, "call_llm_async", all_fail)
    with pytest.raises(RuntimeError, match="All specialists failed"):
        asyncio.run(o.run_task("task", cfg, memory=None))


def test_parallel_map_returns_exceptions_in_place():
    async def f(x):
        if x == 2:
            raise ValueError("bad two")
        return x * 10

    out = asyncio.run(parallel_map(f, [1, 2, 3], max_concurrency=2))
    assert out[0] == 10 and out[2] == 30
    assert isinstance(out[1], ValueError)


def test_parallel_map_preserves_order():
    async def f(x):
        await asyncio.sleep((5 - x) * 0.01)  # later items finish first
        return x

    out = asyncio.run(parallel_map(f, [1, 2, 3, 4], max_concurrency=4))
    assert out == [1, 2, 3, 4]
