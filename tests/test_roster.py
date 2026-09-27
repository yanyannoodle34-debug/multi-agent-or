"""Runtime roster overlay: add/remove specialists persist and survive re-activation."""

from __future__ import annotations

import pytest

import orchestrator as o


def test_add_specialist_persists_via_overlay(cfg):
    key = o.add_specialist(cfg, "Finance", "Finance", "You are the Finance specialist.")
    assert key == "finance"
    assert cfg["specialists"]["finance"]["label"] == "Finance"
    # A fresh apply_roster over a clean base roster should re-add it from the overlay.
    fresh = {"specialists": dict(o.load_config()["specialists"]), "memory": cfg["memory"]}
    o.apply_roster(fresh)
    assert "finance" in fresh["specialists"]


def test_remove_specialist_persists_via_overlay(cfg):
    o.remove_specialist(cfg, "sales")
    assert "sales" not in cfg["specialists"]
    fresh = {"specialists": dict(o.load_config()["specialists"]), "memory": cfg["memory"]}
    o.apply_roster(fresh)
    assert "sales" not in fresh["specialists"]


def test_add_then_remove_nets_out(cfg):
    o.add_specialist(cfg, "legal", "Legal", "You are Legal.")
    o.remove_specialist(cfg, "legal")
    fresh = {"specialists": dict(o.load_config()["specialists"]), "memory": cfg["memory"]}
    o.apply_roster(fresh)
    assert "legal" not in fresh["specialists"]


def test_add_specialist_normalizes_key(cfg):
    key = o.add_specialist(cfg, "  Data Science!! ", "Data Sci", "You are Data Science.")
    assert key == "datascience"


def test_add_specialist_rejects_empty_system(cfg):
    with pytest.raises(ValueError):
        o.add_specialist(cfg, "x", "X", "   ")


def test_add_specialist_rejects_bad_key(cfg):
    with pytest.raises(ValueError):
        o.add_specialist(cfg, "!!!", "X", "valid system prompt")


def test_cannot_remove_last_agent(cfg):
    for key in list(cfg["specialists"])[:-1]:
        o.remove_specialist(cfg, key)
    last = next(iter(cfg["specialists"]))
    with pytest.raises(ValueError, match="last remaining"):
        o.remove_specialist(cfg, last)


def test_remove_unknown_agent_raises(cfg):
    with pytest.raises(ValueError, match="No agent"):
        o.remove_specialist(cfg, "nonexistent")
