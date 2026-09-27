"""Shared pytest fixtures for the offline orchestration tests.

Every test runs fully offline: `orchestrator.activate()` resolves the provider and model
IDs with a lazy client (no key, no network), and the two LLM entry points
(`call_llm` / `call_llm_async`) are stubbed so routing, memory, and synthesis run without
the API. See CLAUDE.md → "Testing orchestration logic offline".
"""

from __future__ import annotations

import os
import sys

import pytest

# Make the project modules importable when pytest is run from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import orchestrator as o  # noqa: E402


@pytest.fixture
def cfg(tmp_path):
    """A live config with memory + roster pointed at an isolated temp dir per test."""
    c = o.activate(o.load_config())
    c["memory"] = {"enabled": True, "dir": str(tmp_path / "data"), "recall": 3}
    return c


@pytest.fixture
def stub_llm(monkeypatch):
    """Stub the coordinator/specialist LLM calls with caller-supplied canned responses.

    Usage:
        stub_llm(router="<analysis>…</analysis><assignment>…</assignment>",
                 specialist="[output]",
                 synth="[deliverable]")

    `router` is returned by the first sync `call_llm` (routing) and `synth` by the second
    (synthesis); if `synth` is omitted the router text is reused. `specialist` is returned
    by every async `call_llm_async`. Returns a dict recording the calls made.
    """
    def _install(router: str, specialist: str = "[specialist output]", synth: str | None = None):
        calls = {"sync": [], "async": []}
        sync_returns = [router, synth if synth is not None else router]

        def fake_sync(prompt, **kw):
            calls["sync"].append({"prompt": prompt, "kw": kw})
            # First sync call = route, second = synthesize; reuse the last thereafter.
            idx = min(len(calls["sync"]) - 1, len(sync_returns) - 1)
            return sync_returns[idx]

        async def fake_async(prompt, **kw):
            calls["async"].append({"prompt": prompt, "kw": kw})
            return specialist

        # Patch the names the orchestrator module actually calls.
        monkeypatch.setattr(o, "call_llm", fake_sync)
        monkeypatch.setattr(o, "call_llm_async", fake_async)
        return calls

    return _install
