"""Primary-agent orchestration over a fixed roster of specialist bots.

A coordinator (the primary agent) reads a task, routes it to the relevant subset of a
*fixed* specialist roster — Sales, Marketing, Operations, Engineering — writes tailored
instructions for each engaged specialist, runs them in parallel, then synthesizes their
outputs into a single deliverable.

This is a routing + orchestrator-workers hybrid: the workers are a known team (not
invented at runtime), but which of them run — and what each is told — is decided per task.
Each specialist can carry memory across runs (see `memory.py`), and the whole flow can be
driven from the CLI (`run.py`) or the local web dashboard (`dashboard.py`).

Config lives in config.yaml (provider + models, per-specialist personas, memory, limits).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from typing import Awaitable, Callable

import yaml

import providers
from llm_utils import (
    call_llm,
    call_llm_async,
    extract_all_tags,
    extract_tag,
    parallel_map,
)
from memory import MemoryStore, new_run_id

log = logging.getLogger("orchestration.coordinator")


@dataclass
class Assignment:
    """One specialist engaged for a task, with instructions tailored by the coordinator."""

    key: str          # roster key, e.g. "sales"
    label: str        # display name, e.g. "Sales"
    system: str       # the specialist's persona/system prompt (from config)
    instructions: str # what the coordinator is asking this specialist to do


@dataclass
class RunResult:
    """The full result of one orchestration run — used by the CLI and the dashboard."""

    task: str
    analysis: str
    assignments: list[Assignment]
    outputs: list        # each entry is a str, or an Exception if that specialist failed
    deliverable: str

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "analysis": self.analysis,
            "specialists": [
                {
                    "key": a.key,
                    "label": a.label,
                    "instructions": a.instructions,
                    "output": str(o) if isinstance(o, Exception) else o,
                    "failed": isinstance(o, Exception),
                }
                for a, o in zip(self.assignments, self.outputs)
            ],
            "deliverable": self.deliverable,
        }


ROUTER_SYSTEM_TEMPLATE = """You are the coordinator (primary agent) of a specialist team.
Your job is to read a task and decide which specialists should work on it, and what each
should do. You do NOT do the work yourself — you delegate.

Available specialists:
{roster}

Engage only the specialists the task actually needs — usually 1 to 3, occasionally all
four. For each engaged specialist, write self-contained instructions scoped to their remit
(don't ask Sales to do Engineering's job). If the task is trivial or clearly outside every
remit, you may engage just one.

Respond in exactly this format, and nothing else:
<analysis>one or two sentences on how you're splitting this across specialists</analysis>
<assignment><agent>roster_key</agent><instructions>self-contained instructions for that specialist</instructions></assignment>
(repeat the <assignment> block for each engaged specialist; use the exact roster_key)"""

SYNTH_SYSTEM = """You are the coordinator synthesizing your specialists' work into one
deliverable for the requester. You are given the original task and each specialist's output.
Combine them into a single, coherent, well-structured response. Resolve overlaps and
conflicts, honor cross-team hand-offs the specialists flagged, and keep the structure the
task implies. Lead with the answer; keep it actionable. Output only the final deliverable."""


def _roster_block(specialists: dict) -> str:
    lines = []
    for key, spec in specialists.items():
        # First ~200 chars of the persona is enough for the router to choose.
        summary = " ".join(spec["system"].split())
        summary = summary[:200] + ("…" if len(summary) > 200 else "")
        lines.append(f"- {key} ({spec.get('label', key)}): {summary}")
    return "\n".join(lines)


def route(task: str, cfg: dict) -> tuple[str, list[Assignment]]:
    """Coordinator step: decide which specialists to engage and what to ask each."""
    specialists = cfg["specialists"]
    system = (cfg.get("prompts") or {}).get("router_system") or ROUTER_SYSTEM_TEMPLATE.format(
        roster=_roster_block(specialists)
    )
    raw = call_llm(
        f"<task>\n{task}\n</task>",
        model=cfg["models"]["coordinator"],
        system=system,
        max_tokens=cfg.get("router_max_tokens", 1200),
        temperature=cfg.get("router_temperature", 0.4),
    )
    analysis = extract_tag(raw, "analysis", default="(no analysis)") or ""

    assignments: list[Assignment] = []
    seen: set[str] = set()
    for block in extract_all_tags(raw, "assignment"):
        key = (extract_tag(block, "agent", default="") or "").strip().lower()
        instr = extract_tag(block, "instructions", default="") or ""
        if key not in specialists:
            log.warning("Router named unknown specialist %r; skipping.", key)
            continue
        if key in seen:  # coordinator double-assigned; keep the first.
            continue
        seen.add(key)
        spec = specialists[key]
        assignments.append(
            Assignment(
                key=key,
                label=spec.get("label", key),
                system=spec["system"],
                instructions=instr or task,
            )
        )

    if not assignments:
        # Degrade gracefully: engage the whole roster rather than crash on a bad route.
        log.warning("Router produced no valid assignments; engaging all specialists.")
        for key, spec in specialists.items():
            assignments.append(
                Assignment(key=key, label=spec.get("label", key),
                           system=spec["system"], instructions=task)
            )

    log.info("Coordinator engaged %d specialist(s): %s",
             len(assignments), [a.label for a in assignments])
    log.info("Coordinator analysis: %s", analysis)
    return analysis, assignments


async def _run_specialist(assignment: Assignment, task: str, cfg: dict, history: str = "") -> str:
    prompt = (
        f"Overall task (for context only):\n{task}\n\n"
        + (f"{history}\n\n" if history else "")
        + f"Your assignment as the {assignment.label} specialist:\n{assignment.instructions}\n\n"
        + "Complete only your part. Be concrete and self-contained, and flag any hand-offs "
        + "to other specialists at the end under a 'Hand-offs:' line."
    )
    return await call_llm_async(
        prompt,
        model=cfg["models"]["specialist"],
        system=assignment.system,
        max_tokens=cfg.get("specialist_max_tokens", 2048),
        temperature=cfg.get("specialist_temperature", 0.8),
    )


def synthesize(task: str, assignments: list[Assignment], outputs: list, cfg: dict) -> str:
    """Coordinator step: merge specialist outputs into the final deliverable."""
    system = (cfg.get("prompts") or {}).get("synthesizer_system") or SYNTH_SYSTEM
    parts = []
    for a, out in zip(assignments, outputs):
        body = f"(specialist failed: {out})" if isinstance(out, Exception) else out
        parts.append(f'<specialist_output name="{a.label}">\n{body}\n</specialist_output>')
    prompt = f"<original_task>\n{task}\n</original_task>\n\n" + "\n\n".join(parts)
    return call_llm(
        prompt,
        model=cfg["models"]["synthesizer"],
        system=system,
        max_tokens=cfg.get("synth_max_tokens", 3000),
        temperature=cfg.get("synth_temperature", 0.6),
    )


async def run_task(
    task: str,
    cfg: dict,
    memory: MemoryStore | None = None,
    on_phase: Callable[[str], Awaitable[None]] | None = None,
) -> RunResult:
    """Full coordinator run: route → (recall memory) → dispatch specialists → synthesize.

    When `memory` is provided, each specialist is primed with its recent history and its
    result is recorded back to its per-agent CSV. Pass `on_phase` (an async callback) to
    receive human-readable progress updates — used by the Telegram bot for live status.

    The synchronous coordinator calls (route, synthesize) run in a worker thread so this
    coroutine never blocks the caller's event loop — the bot stays responsive (and
    cancellable) while a task runs.
    """

    async def phase(msg: str) -> None:
        log.info("phase: %s", msg)
        if on_phase is not None:
            try:
                await on_phase(msg)
            except Exception:  # noqa: BLE001 - progress reporting must never break a run
                log.warning("on_phase callback failed", exc_info=True)

    await phase("Routing task to specialists…")
    analysis, assignments = await asyncio.to_thread(route, task, cfg)

    await phase("Engaged: " + ", ".join(a.label for a in assignments) + ". Running specialists…")
    histories: dict[str, str] = {}
    if memory is not None:
        for a in assignments:
            histories[a.key] = memory.recall_context(a.key)

    outputs = await parallel_map(
        lambda a: _run_specialist(a, task, cfg, histories.get(a.key, "")),
        assignments,
        max_concurrency=cfg.get("max_concurrency", 4),
    )

    if memory is not None:
        run_id = new_run_id()
        for a, o in zip(assignments, outputs):
            memory.record(a.key, task, a.instructions, o, run_id=run_id)

    failed = sum(1 for o in outputs if isinstance(o, Exception))
    if failed == len(outputs):
        # Nothing to synthesize — surface the failure rather than an empty deliverable.
        raise RuntimeError("All specialists failed: " + "; ".join(str(o) for o in outputs))
    if failed:
        log.warning("%d/%d specialists failed; synthesizing from partial results.",
                    failed, len(outputs))

    await phase("Synthesizing final deliverable…")
    deliverable = await asyncio.to_thread(synthesize, task, assignments, outputs, cfg)
    return RunResult(task=task, analysis=analysis, assignments=assignments,
                     outputs=outputs, deliverable=deliverable)


async def orchestrate(task: str, cfg: dict) -> str:
    """Back-compat thin wrapper: run the task and return just the final deliverable."""
    return (await run_task(task, cfg)).deliverable


# --- config + wiring helpers (shared by run.py and dashboard.py) -------------------------

def load_dotenv(path: str = ".env") -> None:
    """Load KEY=VALUE lines from a local .env into os.environ (existing vars win).

    Zero-dependency and best-effort: a missing or unreadable file is a no-op. Lets users
    keep provider keys (OPENROUTER_API_KEY, TELEGRAM_BOT_TOKEN, …) in a .env instead of
    exporting them each shell. An `export ` prefix is tolerated; surrounding quotes are
    stripped. Existing environment variables are never overridden.
    """
    if not os.path.exists(path):
        return
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_config(path: str = "config.yaml") -> dict:
    load_dotenv()  # pick up a local .env before anything reads provider keys
    with open(path) as f:
        return yaml.safe_load(f)


def activate(cfg: dict) -> dict:
    """Select the configured provider and resolve its model IDs into cfg["models"].

    Also applies any runtime roster overlay (agents added/removed via the bot). Safe to
    call offline: the provider client is only built on the first real LLM call, so this
    resolves models without needing an API key or network.
    """
    name = cfg.get("provider", "anthropic")
    pcfg = (cfg.get("providers") or {}).get(name)
    if not pcfg:
        raise SystemExit(f"Provider {name!r} is not defined under `providers:` in config.yaml")
    models = pcfg.get("models")
    if not models:
        raise SystemExit(f"Provider {name!r} has no `models:` block in config.yaml")
    providers.configure(name, pcfg)
    cfg["models"] = models
    apply_roster(cfg)
    return cfg


# --- runtime roster management (add/remove specialists, persisted as an overlay) ---------
# The specialist roster lives in config.yaml, but admins can add/remove agents at runtime.
# Rather than rewrite the (commented) YAML, changes are stored as a small JSON overlay next
# to the memory CSVs and merged over the config roster on load — so they survive restarts.

def _roster_path(cfg: dict) -> str:
    base = (cfg.get("memory") or {}).get("dir", "data")
    return os.path.join(base, "roster.json")


def _norm_key(key: str) -> str:
    return "".join(c for c in (key or "").strip().lower() if c.isalnum() or c in ("-", "_"))


def load_roster_overlay(cfg: dict) -> dict:
    path = _roster_path(cfg)
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            return {"add": dict(data.get("add") or {}), "remove": list(data.get("remove") or [])}
        except (OSError, ValueError):
            log.warning("Could not read roster overlay %s; ignoring.", path)
    return {"add": {}, "remove": []}


def save_roster_overlay(cfg: dict, overlay: dict) -> None:
    path = _roster_path(cfg)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"add": overlay.get("add", {}), "remove": overlay.get("remove", [])}, f, indent=2)


def apply_roster(cfg: dict) -> dict:
    """Merge the runtime overlay (removals then additions) over the config roster."""
    base = dict(cfg.get("specialists") or {})
    overlay = load_roster_overlay(cfg)
    for key in overlay["remove"]:
        base.pop(key, None)
    base.update(overlay["add"])
    cfg["specialists"] = base
    return cfg


def add_specialist(cfg: dict, key: str, label: str, system: str) -> str:
    """Add or update a specialist and persist it to the overlay. Returns the normalized key."""
    key = _norm_key(key)
    if not key:
        raise ValueError("Agent key must contain letters, digits, '-' or '_'.")
    system = (system or "").strip()
    if not system:
        raise ValueError("The system prompt cannot be empty.")
    spec = {"label": (label or "").strip() or key.title(), "system": system}
    cfg.setdefault("specialists", {})[key] = spec
    overlay = load_roster_overlay(cfg)
    overlay["add"][key] = spec
    if key in overlay["remove"]:
        overlay["remove"].remove(key)
    save_roster_overlay(cfg, overlay)
    log.info("Added specialist %r (%s).", key, spec["label"])
    return key


def remove_specialist(cfg: dict, key: str) -> None:
    """Remove a specialist and persist the removal to the overlay."""
    key = _norm_key(key)
    specs = cfg.get("specialists") or {}
    if key not in specs:
        raise ValueError(f"No agent named {key!r}.")
    if len(specs) <= 1:
        raise ValueError("Can't remove the last remaining agent.")
    specs.pop(key, None)
    overlay = load_roster_overlay(cfg)
    overlay["add"].pop(key, None)
    if key not in overlay["remove"]:
        overlay["remove"].append(key)
    save_roster_overlay(cfg, overlay)
    log.info("Removed specialist %r.", key)


def build_memory(cfg: dict) -> MemoryStore | None:
    """Construct the per-agent memory store if enabled in config, else None."""
    mc = cfg.get("memory") or {}
    if not mc.get("enabled"):
        return None
    return MemoryStore(
        base_dir=mc.get("dir", "data"),
        recall=mc.get("recall", 3),
        export_xlsx=mc.get("export_xlsx", False),
    )
