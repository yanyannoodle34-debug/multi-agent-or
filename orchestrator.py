"""Primary-agent orchestration over a fixed roster of specialist bots.

A coordinator (the primary agent) reads a task, routes it to the relevant subset of a
*fixed* specialist roster — Sales, Marketing, Operations, Engineering — writes tailored
instructions for each engaged specialist, runs them in parallel, then synthesizes their
outputs into a single deliverable.

This is a routing + orchestrator-workers hybrid: the workers are a known team (not
invented at runtime), but which of them run — and what each is told — is decided per task.

Run:
    export ANTHROPIC_API_KEY=...
    python run.py "Plan the launch of our new analytics dashboard"

Config lives in config.yaml (models, per-specialist personas, limits).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from llm_utils import (
    call_llm,
    call_llm_async,
    extract_all_tags,
    extract_tag,
    parallel_map,
)

log = logging.getLogger("orchestration.coordinator")


@dataclass
class Assignment:
    """One specialist engaged for a task, with instructions tailored by the coordinator."""

    key: str          # roster key, e.g. "sales"
    label: str        # display name, e.g. "Sales"
    system: str       # the specialist's persona/system prompt (from config)
    instructions: str # what the coordinator is asking this specialist to do


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
        # First sentence of the persona is enough for the router to choose.
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


async def _run_specialist(assignment: Assignment, task: str, cfg: dict) -> str:
    prompt = (
        f"Overall task (for context only):\n{task}\n\n"
        f"Your assignment as the {assignment.label} specialist:\n{assignment.instructions}\n\n"
        "Complete only your part. Be concrete and self-contained, and flag any hand-offs "
        "to other specialists at the end under a 'Hand-offs:' line."
    )
    return await call_llm_async(
        prompt,
        model=cfg["models"]["specialist"],
        system=assignment.system,
        max_tokens=cfg.get("specialist_max_tokens", 2048),
        temperature=cfg.get("specialist_temperature", 0.8),
    )


def synthesize(task: str, assignments: list[Assignment], outputs: list[str], cfg: dict) -> str:
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


async def orchestrate(task: str, cfg: dict) -> str:
    """Full coordinator run: route → dispatch specialists in parallel → synthesize."""
    _, assignments = route(task, cfg)
    outputs = await parallel_map(
        lambda a: _run_specialist(a, task, cfg),
        assignments,
        max_concurrency=cfg.get("max_concurrency", 4),
    )
    failed = sum(1 for o in outputs if isinstance(o, Exception))
    if failed:
        log.warning("%d/%d specialists failed; synthesizing from partial results.",
                    failed, len(outputs))
    if failed == len(outputs):
        # Nothing to synthesize — surface the failure rather than an empty deliverable.
        errs = "; ".join(str(o) for o in outputs)
        raise RuntimeError(f"All specialists failed: {errs}")
    return synthesize(task, assignments, outputs, cfg)
