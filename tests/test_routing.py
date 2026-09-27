"""Routing + graceful-degradation contract (orchestrator.route / run_task)."""

from __future__ import annotations

import asyncio

import orchestrator as o


def _assignment(agent: str, instr: str = "do the thing") -> str:
    return f"<assignment><agent>{agent}</agent><instructions>{instr}</instructions></assignment>"


def test_route_picks_named_subset(cfg, stub_llm):
    stub_llm(router="<analysis>two teams</analysis>" + _assignment("sales") + _assignment("engineering"))
    analysis, assignments = o.route("task", cfg)
    assert analysis == "two teams"
    assert [a.key for a in assignments] == ["sales", "engineering"]


def test_route_skips_unknown_specialist(cfg, stub_llm):
    stub_llm(router="<analysis>x</analysis>" + _assignment("sales") + _assignment("bogus"))
    _, assignments = o.route("task", cfg)
    assert [a.key for a in assignments] == ["sales"]


def test_route_dedupes_double_assignment(cfg, stub_llm):
    stub_llm(router="<analysis>x</analysis>" + _assignment("sales", "first") + _assignment("sales", "second"))
    _, assignments = o.route("task", cfg)
    assert [a.key for a in assignments] == ["sales"]
    assert assignments[0].instructions == "first"  # keeps the first


def test_empty_route_falls_back_to_whole_roster(cfg, stub_llm):
    stub_llm(router="<analysis>no assignments here</analysis>")
    _, assignments = o.route("task", cfg)
    assert {a.key for a in assignments} == set(cfg["specialists"].keys())


def test_missing_instructions_default_to_task(cfg, stub_llm):
    stub_llm(router="<analysis>x</analysis><assignment><agent>sales</agent></assignment>")
    _, assignments = o.route("the original task", cfg)
    assert assignments[0].instructions == "the original task"


def test_run_task_end_to_end(cfg, stub_llm):
    stub_llm(
        router="<analysis>x</analysis>" + _assignment("sales"),
        specialist="[sales output]",
        synth="[final deliverable]",
    )
    result = asyncio.run(o.run_task("task", cfg, memory=None))
    assert result.deliverable == "[final deliverable]"
    assert [a.key for a in result.assignments] == ["sales"]
    assert result.outputs == ["[sales output]"]
