"""Entrypoint for the multi-agent orchestration system.

Usage:
    export ANTHROPIC_API_KEY=...
    python run.py "Plan the launch of our new analytics dashboard"

    # or pipe the task in:
    echo "Draft a Q3 pipeline recovery plan" | python run.py

Reads config.yaml, runs the coordinator over the task, and prints the deliverable.
"""

from __future__ import annotations

import logging
import sys

import yaml

from llm_utils import run
from orchestrator import orchestrate

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def read_task() -> str:
    if len(sys.argv) > 1:
        return " ".join(sys.argv[1:]).strip()
    if not sys.stdin.isatty():
        return sys.stdin.read().strip()
    return ""


def main() -> None:
    task = read_task()
    if not task:
        print('Usage: python run.py "<your task>"', file=sys.stderr)
        raise SystemExit(2)

    cfg = load_config()
    result = run(orchestrate(task, cfg))
    print("\n" + "=" * 60 + "\nDELIVERABLE\n" + "=" * 60 + "\n")
    print(result)


if __name__ == "__main__":
    main()
