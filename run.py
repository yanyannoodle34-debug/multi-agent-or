"""CLI entrypoint for the multi-agent orchestration system.

Usage:
    # pick your provider + key (see config.yaml). For NVIDIA's free NIM API:
    export NVIDIA_API_KEY=nvapi-...        # get one at https://build.nvidia.com
    # or, for Anthropic:  export ANTHROPIC_API_KEY=...

    python run.py "Plan the launch of our new analytics dashboard"
    echo "Draft a Q3 pipeline recovery plan" | python run.py

Reads config.yaml, activates the configured provider, runs the coordinator over the task
(with per-agent memory if enabled), and prints the deliverable.
"""

from __future__ import annotations

import logging
import sys

from llm_utils import run
from orchestrator import activate, build_memory, load_config, run_task

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


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

    cfg = activate(load_config())
    memory = build_memory(cfg)
    result = run(run_task(task, cfg, memory))

    engaged = ", ".join(a.label for a in result.assignments)
    print(f"\nProvider: {cfg.get('provider')}   Specialists engaged: {engaged}")
    print("\n" + "=" * 60 + "\nDELIVERABLE\n" + "=" * 60 + "\n")
    print(result.deliverable)


if __name__ == "__main__":
    main()
