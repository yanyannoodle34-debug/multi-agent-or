# multi-agent-or

Multi-agent orchestration where a **primary (coordinator) agent** coordinates specialized
bots across **Sales, Marketing, Operations, and Engineering**.

The coordinator reads a task, routes it to the relevant subset of the specialist roster,
writes tailored instructions for each engaged specialist, runs them in parallel, and
synthesizes their outputs into a single deliverable. This is a routing + orchestrator–workers
hybrid: the roster is fixed and known, but *which* specialists run — and what each is told —
is decided per task.

```
                    ┌─────────────────────────┐
   task ──────────▶ │  Coordinator (route)    │  picks specialists + writes their briefs
                    └────────────┬────────────┘
              ┌──────────────────┼──────────────────┐   (parallel fan-out)
              ▼                  ▼                  ▼
          ┌───────┐         ┌─────────┐        ┌───────────┐   ...
          │ Sales │         │Marketing│        │Operations │
          └───┬───┘         └────┬────┘        └─────┬─────┘
              └──────────────────┼───────────────────┘
                    ┌────────────▼────────────┐
   deliverable ◀─── │ Coordinator (synthesize)│  merges outputs, resolves hand-offs
                    └─────────────────────────┘
```

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...   # never hardcode the key
```

## Run

```bash
python run.py "Plan the launch of our new analytics dashboard"

# or pipe the task in:
echo "Draft a Q3 pipeline recovery plan" | python run.py
```

## Layout

| File             | Role                                                                    |
|------------------|-------------------------------------------------------------------------|
| `config.yaml`    | Tunable surface: model IDs, per-specialist personas, concurrency, limits |
| `llm_utils.py`   | Shared foundation: retried sync/async LLM calls, parallel map, XML parsing |
| `orchestrator.py`| Coordinator control flow: `route` → parallel specialists → `synthesize` |
| `run.py`         | Thin entrypoint: load config, take the task, print the deliverable       |

## Configuration

Everything tunable lives in `config.yaml` — model IDs, the specialist roster and each bot's
system prompt, concurrency, token budgets, and temperatures. Model IDs are **not** hardcoded
in the Python; check current IDs at
<https://docs.claude.com/en/docs/about-claude/models>. Add or remove a specialist by editing
the `specialists:` block — the coordinator routes over whatever roster it finds there.

## Notes

- Every API call is wrapped in retry-with-backoff (transient 429/5xx are normal).
- Specialists run concurrently with bounded concurrency (`max_concurrency`).
- If some specialists fail, the coordinator synthesizes from partial results and logs the
  failures; it only errors out if *all* specialists fail.
- The code is structurally tested offline (routing, fallback, synthesis) but has not been
  run against the live API here — set your key and try the run command above.

## License

MIT — see `LICENSE`.
