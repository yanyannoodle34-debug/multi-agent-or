# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`multi-agent-or` is a multi-agent orchestration system where a **primary (coordinator) agent** delegates tasks to **specialized bots** covering four domains:

- **Sales** — pipeline, qualification, outreach, deal structure, forecasting
- **Marketing** — positioning, campaigns, channels, demand gen, funnel metrics
- **Operations** — process, tooling, vendors, capacity, compliance, unit economics
- **Engineering** — feasibility, architecture, build-vs-buy, effort/risk estimates

Built in Python. No heavy framework — the coordinator control flow is plain Python. The LLM
backend is pluggable: **OpenRouter** (OpenAI-compatible gateway — the default), **NVIDIA NIM**
(OpenAI-compatible, free API key), **DeepSeek** (OpenAI-compatible), or the **native Anthropic
SDK**, selected by one line in `config.yaml`.

## Commands

```bash
pip install -r requirements.txt          # pyyaml, openai, anthropic, flask

# Provider auth (pick per config.yaml `provider:`):
export OPENROUTER_API_KEY=sk-or-...      # default provider; key: https://openrouter.ai/keys
export NVIDIA_API_KEY=nvapi-...          # if provider: nvidia (free: https://build.nvidia.com)
export ANTHROPIC_API_KEY=...             # if provider: anthropic

python run.py "<task>"                    # run the coordinator over a task (CLI)
echo "<task>" | python run.py            # or pipe the task in
python dashboard.py                       # local web UI at http://127.0.0.1:5000
python telegram_bot.py                     # Telegram control bot (needs TELEGRAM_BOT_TOKEN)

python -m py_compile providers.py memory.py llm_utils.py orchestrator.py run.py dashboard.py telegram_bot.py
```

There is no formal test suite. The orchestration logic and the dashboard endpoints are
verified offline by stubbing `orchestrator.call_llm` / `call_llm_async` (see below) so
routing, fallback, memory, and synthesis run without live API calls.

## Architecture

Routing + orchestrator–workers **hybrid**: the specialist roster is *fixed and known*, but
which specialists run — and what each is instructed to do — is decided per task at runtime.

```
run.py / dashboard.py / telegram_bot.py ──▶ orchestrator.activate(cfg)   # provider + models
                          orchestrator.run_task(task, cfg, memory, on_phase) -> RunResult
                            │
                            ├─ route()        coordinator picks specialists (subset of roster)
                            │                  and writes tailored <assignment> instructions
                            ├─ recall()        each engaged agent is primed with its CSV memory
                            ├─ parallel_map()  specialists run concurrently (max_concurrency)
                            ├─ record()        each result is appended to data/<agent>.csv
                            └─ synthesize()    coordinator merges outputs into one deliverable
```

`run_task` offloads the synchronous coordinator calls (`route`, `synthesize`) to a worker
thread via `asyncio.to_thread` and reports progress through the optional async `on_phase`
callback — so an async caller's event loop (the Telegram bot) stays responsive and the run
stays cancellable while it executes.

- **`config.yaml`** is the tunable surface — `provider:` + per-provider `models:`, the
  `specialists:` roster (each with a `label` and a `system` persona), the `memory:` block,
  concurrency, token budgets, temperatures. The roster is data: add/remove a specialist by
  editing this block and the coordinator routes over it. The coordinator uses a strong model;
  specialists use a faster one (configurable per role).
- **`providers.py`** owns the LLM backends and retry-with-backoff. `configure(name, cfg)`
  selects the active provider; the SDK client is built **lazily on the first call**, so
  `activate()` / model resolution work offline without a key. `"anthropic"` → native SDK;
  any other name → OpenAI-compatible backend (OpenRouter, NVIDIA NIM, etc., via `base_url` +
  `api_key_env`, with optional per-provider `headers` — e.g. OpenRouter's `HTTP-Referer`/`X-Title`).
  Adding an OpenAI-compatible provider is config-only; no code change needed.
- **`llm_utils.py`** holds provider-agnostic primitives: `call_llm` / `call_llm_async`
  (thin delegators to `providers`), `parallel_map` (bounded concurrency, returns exceptions
  in-place rather than crashing the batch), and `extract_tag` / `extract_all_tags` (XML parse).
- **`memory.py`** is the per-agent CSV store (`data/<agent>.csv`, Excel-compatible):
  `record`, `recall`, `recall_context` (renders recent rows into a prompt block), plus
  `path`/`exists`/`clear` for the bot's download/upload/clear controls. `FIELDS` is the
  canonical header (upload validation checks against it). Optional `.xlsx` mirror via openpyxl.
- **`orchestrator.py`** holds the coordinator control flow plus wiring helpers
  (`load_config`, `activate`, `build_memory`). Hand-offs are structured: the router emits
  `<assignment><agent>key</agent><instructions>…</instructions>` blocks parsed into
  `Assignment` dataclasses; a full run returns a `RunResult` (`.to_dict()` feeds the dashboard).
  Runtime roster edits (`add_specialist`/`remove_specialist`, used by the bot) persist as a
  JSON overlay (`data/roster.json`) that `apply_roster` merges over the config roster in
  `activate()` — so added/removed agents survive restarts without rewriting `config.yaml`.
- **`dashboard.py`** is a local Flask UI. All async orchestration runs on a single
  long-lived event loop in a background thread (so the async SDK clients stay bound to one
  loop across requests); handlers submit coroutines via `asyncio.run_coroutine_threadsafe`.
- **`telegram_bot.py`** is a `python-telegram-bot` control bot (admin-gated via
  `telegram.admin_ids`, overridable by the `TELEGRAM_ADMIN_IDS` env for the dashboard-launched
  subprocess). Inline-button menus route on short `callback_data` prefixes (`menu:`, `run:`,
  `agent:`, `key:`, `admin:`); free-text/file replies are captured via a per-user
  `context.user_data["await"]` pending-action flag (cleared at the top of every callback, then
  re-armed by handlers that expect input — including the 3-step add-agent flow). Running tasks
  are tracked as `asyncio.Task` in `chat_data["run_task"]` so **Stop** can cancel them; agents
  can be added/removed live via `orchestrator.add_specialist`/`remove_specialist`. A global
  `add_error_handler` plus per-handler try/except keep it alive; `_ensure_event_loop()` sets a
  loop before `run_polling()` for Python 3.12+/3.14. Module-level `CFG`/`MEMORY` are mutated in
  place when switching provider / toggling memory / setting keys / editing the roster.

Conventions worth preserving:
- **Model IDs live in `config.yaml`, never hardcoded in logic** — under each provider block.
  OpenRouter: <https://openrouter.ai/models> (namespaced `vendor/model`, `:free` variants);
  NVIDIA: <https://build.nvidia.com/models>; DeepSeek: `deepseek-chat`/`deepseek-reasoner`;
  Anthropic: <https://docs.claude.com/en/docs/about-claude/models>.
- **Graceful degradation**: an unknown specialist key is skipped; an empty route falls back
  to the whole roster; partial specialist failures still synthesize (only an all-fail errors).
- **Provider clients build lazily** — never construct an SDK client at import time; keep
  `activate()` and offline tests key-free.
- **Every LLM call is retried**; every fan-out uses real concurrency.
- **Config keys can be `None`, not just missing.** A commented-out YAML block (e.g. `prompts:`)
  loads as `None`, so read optional maps as `(cfg.get("x") or {})`, not `cfg.get("x", {})`.

### Testing orchestration logic offline

`activate()` resolves the provider/models without a key (lazy client). Stub the two LLM
entry points to exercise control flow — routing, memory, synthesis — without the API:

```python
import orchestrator as o
cfg = o.activate(o.load_config())          # resolves cfg["models"]; no network/client yet
o.call_llm = lambda prompt, **k: "<analysis>…</analysis><assignment><agent>sales</agent><instructions>…</instructions></assignment>"
async def fake(prompt, **k): return "[specialist output]"
o.call_llm_async = fake
# o.run_task(task, cfg, memory) now runs fully offline
```

## License

MIT — see `LICENSE`.
