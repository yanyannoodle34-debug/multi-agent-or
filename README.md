# multi-agent-or

Multi-agent orchestration where a **primary (coordinator) agent** coordinates specialized
bots across **Sales, Marketing, Operations, and Engineering**.

The coordinator reads a task, routes it to the relevant subset of the specialist roster,
writes tailored instructions for each engaged specialist, runs them in parallel, and
synthesizes their outputs into a single deliverable. This is a routing + orchestrator–workers
hybrid: the roster is fixed and known, but *which* specialists run — and what each is told —
is decided per task.

Runs on **NVIDIA NIM's free API** (default) or the **Anthropic API** — a one-line switch in
`config.yaml`. Each agent keeps **persistent memory** in CSV (Excel-compatible), and there's
a **local web dashboard** to drive the whole thing.

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

# Default provider is NVIDIA NIM — grab a FREE key at https://build.nvidia.com
export NVIDIA_API_KEY=nvapi-...
# ...or switch config.yaml to the anthropic provider and set:
# export ANTHROPIC_API_KEY=...
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
the final deliverable, and browse each agent's memory — all locally.

## Providers (free NVIDIA key or Anthropic)

`config.yaml` picks the backend with one line:

```yaml
provider: nvidia   # or: anthropic
```

- **nvidia** — NVIDIA NIM, which is **OpenAI-compatible**. Free API key from
  <https://build.nvidia.com>; browse model IDs at <https://build.nvidia.com/models>.
- **anthropic** — the native Anthropic SDK; model IDs at
  <https://docs.claude.com/en/docs/about-claude/models>.

Each provider block carries its own model IDs, so switching is just changing `provider:`.
Any other OpenAI-compatible endpoint works too — add a block with a `base_url` and an
`api_key_env`, and the OpenAI-compatible backend handles it.

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

## Layout

| File             | Role                                                                       |
|------------------|----------------------------------------------------------------------------|
| `config.yaml`    | Tunable surface: provider + models, specialist personas, memory, limits    |
| `providers.py`   | LLM backends: native Anthropic and OpenAI-compatible (NVIDIA NIM); retry    |
| `llm_utils.py`   | Provider-agnostic primitives: `call_llm(_async)`, `parallel_map`, XML parse |
| `memory.py`      | Per-agent CSV/Excel memory store: `record`, `recall`, `recall_context`     |
| `orchestrator.py`| Coordinator flow: `route` → parallel specialists (+memory) → `synthesize`   |
| `run.py`         | CLI entrypoint                                                              |
| `dashboard.py`   | Local Flask web dashboard                                                   |

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
