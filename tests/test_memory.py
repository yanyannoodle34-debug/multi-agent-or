"""Per-agent CSV memory store round-trip (memory.MemoryStore)."""

from __future__ import annotations

from memory import FIELDS, MemoryStore, new_run_id


def test_record_and_recall_round_trip(tmp_path):
    m = MemoryStore(base_dir=str(tmp_path), recall=3)
    rid = new_run_id()
    m.record("sales", "task one", "instr one", "output one", run_id=rid)
    m.record("sales", "task two", "instr two", "output two", run_id=rid)
    rows = m.recall("sales")
    assert [r["task"] for r in rows] == ["task one", "task two"]  # newest last
    assert rows[0]["run_id"] == rid
    assert list(rows[0].keys()) == FIELDS


def test_recall_limit_returns_most_recent(tmp_path):
    m = MemoryStore(base_dir=str(tmp_path), recall=2)
    for i in range(5):
        m.record("ops", f"task {i}", "i", f"out {i}")
    rows = m.recall("ops")  # default limit == recall == 2
    assert [r["task"] for r in rows] == ["task 3", "task 4"]


def test_record_stores_exception_as_failure(tmp_path):
    m = MemoryStore(base_dir=str(tmp_path))
    m.record("eng", "t", "i", RuntimeError("kaboom"))
    assert m.recall("eng")[0]["output"].startswith("(failed:")


def test_clear_and_exists(tmp_path):
    m = MemoryStore(base_dir=str(tmp_path))
    assert not m.exists("sales")
    m.record("sales", "t", "i", "o")
    assert m.exists("sales")
    assert m.clear("sales") is True
    assert m.clear("sales") is False  # already gone
    assert not m.exists("sales")


def test_recall_context_empty_when_no_history(tmp_path):
    m = MemoryStore(base_dir=str(tmp_path))
    assert m.recall_context("sales") == ""


def test_recall_context_renders_recent_rows(tmp_path):
    m = MemoryStore(base_dir=str(tmp_path))
    m.record("sales", "plan the launch", "i", "launch plan details")
    block = m.recall_context("sales")
    assert "plan the launch" in block
    assert "launch plan details" in block


def test_path_sanitizes_agent_name(tmp_path):
    m = MemoryStore(base_dir=str(tmp_path))
    # Traversal / odd chars must not escape the base dir.
    p = m.path("../../etc/passwd")
    assert p.startswith(str(tmp_path))
    assert ".." not in p.split("/")[-1]
