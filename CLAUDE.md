# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`multi-agent-or` is a multi-agent orchestration system where a **primary (coordinator) agent** delegates tasks to **specialized bots** covering four domains:

- **Sales** — pipeline, qualification, outreach, deal structure, forecasting
- **Marketing** — positioning, campaigns, channels, demand gen, funnel metrics
- **Operations** — process, tooling, vendors, capacity, compliance, unit economics
- **Engineering** — feasibility, architecture, build-vs-buy, effort/risk estimates

Built in Python on the Anthropic SDK. No heavy framework — the coordinator control flow is
plain Python over `client.messages.create`.

## Commands

```bash
pip install -r requirements.txt          # anthropic, pyyaml
export ANTHROPIC_API_KEY=...             # auth; never hardcode the key

python run.py "<task>"                    # run the coordinator over a task
echo "<task>" | python run.py            # or pipe the task in

python -m py_compile llm_utils.py orchestrator.py run.py   # syntax check
```

There is no test suite yet. The orchestration logic is verified offline by stubbing
`orchestrator.call_llm` / `call_llm_async` (see the offline-test approach below) so routing,
fallback, and synthesis can be exercised without live API calls.

## Architecture

Routing + orchestrator–workers **hybrid**: the specialist roster is *fixed and known*, but
which specialists run — and what each is instructed to do — is decided per task at runtime.

```
run.py ──▶ orchestrator.orchestrate(task, cfg)
             │
             ├─ route()        coordinator picks the relevant specialists (a subset of the
             │                  roster) and writes tailored <assignment> instructions for each
             ├─ parallel_map()  engaged specialists run concurrently (bounded by max_concurrency)
             └─ synthesize()    coordinator merges specialist outputs into one deliverable
```

- **`config.yaml`** is the tunable surface — model IDs, the `specialists:` roster (each with
  a `label` and a `system` persona), concurrency, token budgets, temperatures. The roster is
  data: add/remove a specialist by editing this block and the coordinator routes over it. The
  coordinator uses a strong model; specialists use a faster one (configurable per role).
- **`llm_utils.py`** is the shared foundation copied verbatim across projects: `call_llm` /
  `call_llm_async` (each wrapped in `with_retry` for transient 429/5xx), `parallel_map`
  (bounded concurrency, returns exceptions in-place rather than crashing the batch), and
  `extract_tag` / `extract_all_tags` (XML-tag parsing for structured hand-offs).
- **`orchestrator.py`** holds the coordinator control flow. Hand-offs between steps are
  structured: the router emits `<assignment><agent>key</agent><instructions>…</instructions>`
  blocks parsed into `Assignment` dataclasses, never re-parsed free text.

Conventions worth preserving:
- **Model IDs live in `config.yaml`, never hardcoded in logic.** Check current IDs at
  <https://docs.claude.com/en/docs/about-claude/models>.
- **Graceful degradation**: an unknown specialist key is skipped; an empty route falls back
  to the whole roster; partial specialist failures still synthesize (only an all-fail errors).
- **Every LLM call is retried**; every fan-out uses real concurrency.
- **Config keys can be `None`, not just missing.** A commented-out YAML block (e.g. `prompts:`)
  loads as `None`, so read optional maps as `(cfg.get("x") or {})`, not `cfg.get("x", {})`.

### Testing orchestration logic offline

Stub the two LLM entry points to exercise control flow without the API:

```python
import orchestrator as o
o.call_llm = lambda prompt, **k: "<analysis>…</analysis><assignment>…</assignment>"
async def fake(prompt, **k): return "[specialist output]"
o.call_llm_async = fake
```

## License

MIT — see `LICENSE`.
