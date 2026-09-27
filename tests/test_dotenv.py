"""The zero-dependency .env loader (orchestrator.load_dotenv)."""

from __future__ import annotations

import os

import orchestrator as o


def test_loads_key_value_pairs(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("FOO_KEY=bar\nBAZ_KEY=qux\n")
    monkeypatch.delenv("FOO_KEY", raising=False)
    monkeypatch.delenv("BAZ_KEY", raising=False)
    o.load_dotenv(str(env))
    assert os.environ["FOO_KEY"] == "bar"
    assert os.environ["BAZ_KEY"] == "qux"


def test_existing_env_wins(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("ALREADY_SET=from_file\n")
    monkeypatch.setenv("ALREADY_SET", "from_shell")
    o.load_dotenv(str(env))
    assert os.environ["ALREADY_SET"] == "from_shell"


def test_ignores_comments_blanks_and_malformed(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# a comment\n\n   \nNO_EQUALS_SIGN\nGOOD_KEY=value\n")
    monkeypatch.delenv("GOOD_KEY", raising=False)
    monkeypatch.delenv("NO_EQUALS_SIGN", raising=False)
    o.load_dotenv(str(env))
    assert os.environ["GOOD_KEY"] == "value"
    assert "NO_EQUALS_SIGN" not in os.environ


def test_strips_export_prefix_and_quotes(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('export QUOTED_KEY="spaced value"\nSINGLE_KEY=\'single\'\n')
    monkeypatch.delenv("QUOTED_KEY", raising=False)
    monkeypatch.delenv("SINGLE_KEY", raising=False)
    o.load_dotenv(str(env))
    assert os.environ["QUOTED_KEY"] == "spaced value"
    assert os.environ["SINGLE_KEY"] == "single"


def test_missing_file_is_noop(tmp_path):
    # Should not raise.
    o.load_dotenv(str(tmp_path / "does_not_exist.env"))
