# multi-agent-or

Multi-agent orchestration where a **primary (coordinator) agent** coordinates specialized
bots across **Sales, Marketing, Operations, and Engineering**.

The coordinator reads a task, routes it to the relevant subset of the specialist roster,
writes tailored instructions for each engaged specialist, runs them in parallel, and
synthesizes their outputs into a single deliverable. This is a routing + orchestrator–workers
hybrid: the roster is fixed and known, but *which* specialists run — and what each is told —
is decided per task.

Runs on **OpenRouter** (default), **NVIDIA NIM's free API**, **DeepSeek**, or the
**Anthropic API** — a one-line switch in `config.yaml`. Each agent keeps **persistent
memory** in CSV (Excel-compatible), and there's a **local web dashboard** to drive the
whole thing.

```
                    ┌─────────────────────────┐
   task ──────────▶ │  Coordinator (route)    │  picks specialists + writes their briefs
                    └────────────┬────────────┘
              ┌──────────────────┼──────────────────┐   (parallel fan-out)
              ▼                  ▼                  ▼
          ┌───────┐         ┌─────────┐        ┌───────────┐   ...
          │ Sales │         │Marketing│        │Operations │   each primed with its
          └───┬───┘         └────┬────┘        └─────┬─────┘   own CSV memory, then
              │ ▲                │ ▲                 │ ▲       records its result back
              └─┴────────────────┴─┴─────────────────┴─┘  data/<agent>.csv (Excel)
                    ┌────────────▼────────────┐
   deliverable ◀─── │ Coordinator (synthesize)│  merges outputs, resolves hand-offs
                    └─────────────────────────┘
```

## Setup

```bash
pip install -r requirements.txt

# Default provider is OpenRouter — get a key at https://openrouter.ai/keys
export OPENROUTER_API_KEY=sk-or-...
# ...or switch config.yaml's `provider:` and set the matching key:
# export NVIDIA_API_KEY=nvapi-...      # provider: nvidia  (free: https://build.nvidia.com)
# export ANTHROPIC_API_KEY=...         # provider: anthropic
```

## Run

**CLI:**

```bash
python run.py "Plan the launch of our new analytics dashboard"
echo "Draft a Q3 pipeline recovery plan" | python run.py
```

**Local web dashboard:**

```bash
python dashboard.py          # serves http://127.0.0.1:5000
```

Submit a task, watch which specialists the coordinator engages, read each bot's output and
the final deliverable, and browse each agent's memory — all locally. The dashboard also has
a **Telegram bot panel**: set the bot token and admin IDs, **Start/Stop** the bot, and watch
live status — so you can run the Telegram bot without touching the terminal (it runs as a
managed subprocess; the token and admin IDs are passed via its environment, never saved to disk).

**Telegram control bot:**

```bash
export TELEGRAM_BOT_TOKEN=...             # from @BotFather
python telegram_bot.py
```

Control everything from Telegram with inline buttons (see below).

## Providers (OpenRouter, free NVIDIA key, or Anthropic)

`config.yaml` picks the backend with one line:

```yaml
provider: openrouter   # or: nvidia, anthropic
```

- **openrouter** — the [OpenRouter](https://openrouter.ai) gateway: one key, many models
  behind an **OpenAI-compatible** API. Key at <https://openrouter.ai/keys>; namespaced model
  IDs (`vendor/model`) at <https://openrouter.ai/models> — many have a free `:free` variant.
  The provider block can set optional `headers` (`HTTP-Referer` / `X-Title`) for OpenRouter's
  app attribution and rankings.
- **nvidia** — NVIDIA NIM, also **OpenAI-compatible**. Free API key from
  <https://build.nvidia.com>; model IDs at <https://build.nvidia.com/models>.
- **deepseek** — DeepSeek's API, also **OpenAI-compatible**. Key at
  <https://platform.deepseek.com/api_keys>; models `deepseek-chat` (V3) and
  `deepseek-reasoner` (R1).
- **anthropic** — the native Anthropic SDK; model IDs at
  <https://docs.claude.com/en/docs/about-claude/models>.

Each provider block carries its own model IDs, so switching is just changing `provider:`.
Any other OpenAI-compatible endpoint works too — add a block with a `base_url`, an
`api_key_env`, and optional `headers`, and the OpenAI-compatible backend handles it.

## Agent memory (CSV / Excel data store)

With `memory.enabled: true` (the default), every specialist run is appended to
`data/<agent>.csv` — one file per agent, columns `timestamp, run_id, task, instructions,
output`. Before each run, an agent is primed with its most recent `memory.recall` entries so
it builds on past work. The CSVs open directly in Excel; set `memory.export_xlsx: true` (and
`pip install openpyxl`) to also mirror each history to `data/<agent>.xlsx`.

```yaml
memory:
  enabled: true
  dir: "data"
  recall: 3            # past entries fed back into each agent's next prompt
  export_xlsx: false
```

## Telegram control bot

Drive the whole system from Telegram with friendly inline buttons — no terminal needed.

```bash
pip install python-telegram-bot>=21
export TELEGRAM_BOT_TOKEN=...             # from @BotFather
# set your Telegram user id(s) under telegram.admin_ids in config.yaml (find yours via @userinfobot)
python telegram_bot.py
```

Send `/start`, then use the menu:

- **▶️ Run task / ⏹ Stop** — run an orchestration; live progress updates as it routes,
  runs specialists, and synthesizes. Stop cancels a run mid-flight.
- **📊 Status** — a live dashboard: active provider, model IDs, masked API key, memory
  state, whether a task is running, and the last run summary.
- **🧠 Agents** — **➕ add** a new specialist (guided key → label → system-prompt flow) or,
  per agent, view its persona, **📥 download** / **⬆️ upload** (header-validated) / **🗑 clear**
  its memory CSV, or **❌ delete** it (with confirmation). Added/removed agents persist across
  restarts (stored as a small `data/roster.json` overlay) and the coordinator routes to the
  updated roster immediately.
- **🔑 API keys** — set any provider's key (session-scoped; your message with the secret is
  deleted best-effort) and switch the active provider.
- **⚙️ Admin** — toggle memory; see admin IDs.

Every screen has a **⬅️ Back** button. Access is gated to `telegram.admin_ids`; a global
error handler keeps the bot alive and replies with a friendly message on any failure. The
bot stays responsive during a run (the synchronous coordinator calls are offloaded to a
worker thread), so **Stop** and the menu always work.

```yaml
telegram:
  enabled: true
  token_env: "TELEGRAM_BOT_TOKEN"
  admin_ids: [123456789]     # empty = OPEN mode (anyone can control it) — not recommended
```

## Layout

| File             | Role                                                                       |
|------------------|----------------------------------------------------------------------------|
| `config.yaml`    | Tunable surface: provider + models, specialist personas, memory, telegram  |
| `providers.py`   | LLM backends: native Anthropic and OpenAI-compatible (OpenRouter/NVIDIA)    |
| `llm_utils.py`   | Provider-agnostic primitives: `call_llm(_async)`, `parallel_map`, XML parse |
| `memory.py`      | Per-agent CSV/Excel memory store: `record`, `recall`, download/upload/clear |
| `orchestrator.py`| Coordinator flow: `route` → parallel specialists (+memory) → `synthesize`   |
| `run.py`         | CLI entrypoint                                                              |
| `dashboard.py`   | Local Flask web dashboard                                                   |
| `telegram_bot.py`| Telegram control bot (inline buttons, admin-gated)                          |

## Notes

- Every API call is retried with backoff (transient 429/5xx are normal).
- Specialists run concurrently with bounded concurrency (`max_concurrency`).
- Partial specialist failures still synthesize (logged); only an all-fail errors out.
- Model IDs live in `config.yaml`, never in the Python.
- The orchestration logic (routing, fallback, memory, synthesis) and the dashboard endpoints
  are tested offline by stubbing the LLM calls; the code has not been run against a live API
  here — set your key and try the commands above.

## License

MIT — see `LICENSE`.
