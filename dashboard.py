"""Local web dashboard to control the multi-agent orchestrator.

    pip install flask
    export NVIDIA_API_KEY=nvapi-...     # or ANTHROPIC_API_KEY, per config.yaml
    python dashboard.py                 # serves http://127.0.0.1:5000

Submit a task, watch which specialists the coordinator engages, read each bot's output
and the final deliverable, and browse each agent's CSV memory — all locally.

The dashboard also manages the Telegram control bot: set its token and admin IDs, start
and stop it, and see live status. The bot runs as a managed subprocess; its token and
admin IDs are passed via the child's environment (never written to disk).

Async note: all orchestration runs on a single long-lived event loop in a background
thread (so the async SDK clients stay bound to one loop across requests); Flask request
handlers submit coroutines to it and block for the result.
"""

from __future__ import annotations

import asyncio
import atexit
import logging
import os
import subprocess
import sys
import threading

from flask import Flask, jsonify, render_template_string, request

from orchestrator import activate, build_memory, load_config, run_task

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("orchestration.dashboard")

CFG = activate(load_config())
MEMORY = build_memory(CFG)

# One persistent event loop for all async orchestration work (see module docstring).
_loop = asyncio.new_event_loop()
threading.Thread(target=lambda: (_loop.run_forever()), daemon=True).start()


def _submit(coro):
    return asyncio.run_coroutine_threadsafe(coro, _loop).result()


# --- Telegram bot management -------------------------------------------------------------
# The bot runs as a managed subprocess (isolated event loop, clean start/stop). Its token
# and admin IDs are set here and passed via the child's environment — never written to disk.

_TG = CFG.get("telegram") or {}
TOKEN_ENV = _TG.get("token_env", "TELEGRAM_BOT_TOKEN")
BOT_LOG = "telegram_bot.log"

_bot_lock = threading.Lock()
_bot_proc: subprocess.Popen | None = None
_bot_settings = {
    "token": os.environ.get(TOKEN_ENV, ""),
    "admin_ids": [int(x) for x in (_TG.get("admin_ids") or [])],
}


def _mask(value: str) -> str:
    if not value:
        return ""
    return f"{value[:4]}…{value[-4:]}" if len(value) > 12 else "•" * len(value)


def _bot_running() -> bool:
    return _bot_proc is not None and _bot_proc.poll() is None


def _bot_status() -> dict:
    proc = _bot_proc
    exit_code = None if (proc is None or proc.poll() is None) else proc.poll()
    tail = ""
    if not _bot_running() and os.path.exists(BOT_LOG):
        try:
            with open(BOT_LOG, encoding="utf-8", errors="replace") as f:
                tail = "".join(f.readlines()[-6:]).strip()
        except OSError:
            tail = ""
    return {
        "running": _bot_running(),
        "pid": proc.pid if _bot_running() else None,
        "exit_code": exit_code,
        "token_set": bool(_bot_settings["token"]),
        "token_masked": _mask(_bot_settings["token"]),
        "token_env": TOKEN_ENV,
        "admin_ids": _bot_settings["admin_ids"],
        "recent_log": tail,
    }


def _start_bot() -> None:
    with _bot_lock:
        if _bot_running():
            raise RuntimeError("Bot is already running.")
        if not _bot_settings["token"]:
            raise RuntimeError("Set the Telegram bot token first.")
        env = dict(os.environ)
        env[TOKEN_ENV] = _bot_settings["token"]
        env["TELEGRAM_ADMIN_IDS"] = ",".join(str(i) for i in _bot_settings["admin_ids"])
        logfile = open(BOT_LOG, "w", encoding="utf-8")  # noqa: SIM115 - handle owned by child
        global _bot_proc
        _bot_proc = subprocess.Popen(
            [sys.executable, "telegram_bot.py"],
            env=env, stdout=logfile, stderr=subprocess.STDOUT,
            cwd=os.path.dirname(os.path.abspath(__file__)) or None,
        )
        log.info("Started Telegram bot subprocess (pid=%s).", _bot_proc.pid)


def _stop_bot() -> None:
    with _bot_lock:
        global _bot_proc
        if _bot_proc is None or _bot_proc.poll() is not None:
            _bot_proc = None
            return
        _bot_proc.terminate()
        try:
            _bot_proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            _bot_proc.kill()
        log.info("Stopped Telegram bot subprocess.")
        _bot_proc = None


@atexit.register
def _cleanup_bot() -> None:
    if _bot_running():
        _stop_bot()


app = Flask(__name__)

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Multi-Agent Orchestrator</title>
<style>
  :root { color-scheme: light dark; }
  * { box-sizing: border-box; }
  body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
         margin: 0; padding: 0 0 3rem; line-height: 1.5;
         background: #0f1220; color: #e6e8ef; }
  header { padding: 1.1rem 1.4rem; border-bottom: 1px solid #262b40;
           display: flex; align-items: baseline; gap: .8rem; flex-wrap: wrap; }
  h1 { font-size: 1.15rem; margin: 0; }
  .badge { font-size: .75rem; padding: .15rem .55rem; border-radius: 999px;
           background: #1d2440; border: 1px solid #33406b; color: #9fb0e0; }
  main { max-width: 900px; margin: 0 auto; padding: 1.4rem; }
  textarea { width: 100%; min-height: 90px; padding: .7rem .8rem; border-radius: 10px;
             border: 1px solid #333a57; background: #151a2e; color: inherit;
             font: inherit; resize: vertical; }
  .row { display: flex; gap: .6rem; align-items: center; margin-top: .6rem; flex-wrap: wrap; }
  button { background: #4666d6; color: #fff; border: 0; padding: .55rem 1.1rem;
           border-radius: 8px; font: inherit; cursor: pointer; }
  button.secondary { background: #232a44; color: #cdd5f0; border: 1px solid #39426a; }
  button:disabled { opacity: .55; cursor: default; }
  .chip { font-size: .78rem; padding: .2rem .6rem; border-radius: 999px;
          background: #1a2138; border: 1px solid #333d63; color: #b9c4e8; }
  .card { background: #151a2e; border: 1px solid #262b40; border-radius: 12px;
          padding: 1rem 1.1rem; margin-top: 1rem; }
  .card h3 { margin: 0 0 .4rem; font-size: .95rem; }
  .muted { color: #8b93ab; font-size: .85rem; }
  .fail { color: #ff8f8f; }
  pre { white-space: pre-wrap; word-wrap: break-word; margin: .3rem 0 0; font: inherit; }
  .deliverable { border-color: #3a56b8; background: #141b33; }
  .spinner { display: none; }
  .spinner.on { display: inline; }
  .memtable { width: 100%; border-collapse: collapse; font-size: .82rem; margin-top: .5rem; }
  .memtable th, .memtable td { text-align: left; border-bottom: 1px solid #262b40;
                               padding: .35rem .5rem; vertical-align: top; }
  a { color: #8fa6ef; }
  .fld { display: block; margin-top: .6rem; font-size: .85rem; color: #cdd5f0; }
  .fld input { display: block; width: 100%; margin-top: .25rem; padding: .5rem .6rem;
               border-radius: 8px; border: 1px solid #333a57; background: #10152a;
               color: inherit; font: inherit; }
  .dot { display: inline-block; width: .6rem; height: .6rem; border-radius: 50%;
         vertical-align: middle; margin: 0 .1rem; }
  .dot-on { background: #46d67a; box-shadow: 0 0 6px #46d67a; }
  .dot-off { background: #6b7280; }
  .tglog { color: #9aa3bd; font-size: .78rem; margin-top: .5rem; max-height: 8rem;
           overflow: auto; }
</style>
</head>
<body>
<header>
  <h1>Multi-Agent Orchestrator</h1>
  <span class="badge">provider: {{ provider }}</span>
  <span class="badge">memory: {{ memory_state }}</span>
</header>
<main>
  <textarea id="task" placeholder="Describe a task — e.g. 'Plan the launch of our new analytics dashboard'"></textarea>
  <div class="row">
    <button id="runBtn" onclick="runTask()">Run</button>
    <span class="chip" id="roster"></span>
    <span class="spinner muted" id="spin">running the coordinator…</span>
  </div>

  <div id="results"></div>

  <div class="card">
    <h3>Agent memory</h3>
    <div class="muted">Each agent stores its runs in <code>data/&lt;agent&gt;.csv</code> (opens in Excel).</div>
    <div class="row" id="memBtns"></div>
    <div id="memView"></div>
  </div>

  <div class="card">
    <h3>Telegram bot <span id="tgDot" class="dot dot-off" title="stopped"></span>
        <span class="muted" id="tgState">checking…</span></h3>
    <div class="muted" id="tgInfo"></div>

    <label class="fld">Bot token <span class="muted">(from @BotFather)</span>
      <input id="tgToken" type="password" placeholder="123456:ABC-DEF…" autocomplete="off">
    </label>
    <label class="fld">Admin IDs <span class="muted">(comma-separated Telegram user IDs; empty = open mode)</span>
      <input id="tgAdmins" type="text" placeholder="123456789, 987654321" autocomplete="off">
    </label>

    <div class="row">
      <button class="secondary" onclick="tgSave()">Save settings</button>
      <button id="tgStart" onclick="tgStart()">▶️ Start</button>
      <button id="tgStop" class="secondary" onclick="tgStop()">⏹ Stop</button>
    </div>
    <div class="muted" id="tgMsg"></div>
    <pre id="tgLog" class="tglog"></pre>
  </div>
</main>

<script>
const PROVIDER = {{ provider|tojson }};
const AGENTS = {{ agents|tojson }};
const MEMORY_ON = {{ memory_on|tojson }};

document.getElementById("roster").textContent = "specialists: " + AGENTS.join(", ");
const memBtns = document.getElementById("memBtns");
AGENTS.forEach(a => {
  const b = document.createElement("button");
  b.className = "secondary";
  b.textContent = a;
  b.onclick = () => loadMemory(a);
  memBtns.appendChild(b);
});

function esc(s){ const d = document.createElement("div"); d.textContent = s == null ? "" : s; return d.innerHTML; }

async function runTask(){
  const task = document.getElementById("task").value.trim();
  if(!task) return;
  const btn = document.getElementById("runBtn");
  const spin = document.getElementById("spin");
  const results = document.getElementById("results");
  btn.disabled = true; spin.classList.add("on"); results.innerHTML = "";
  try {
    const resp = await fetch("/run", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({task})
    });
    const data = await resp.json();
    if(!resp.ok){ results.innerHTML = `<div class="card fail">Error: ${esc(data.error || resp.statusText)}</div>`; return; }
    renderResult(data);
  } catch(e){
    results.innerHTML = `<div class="card fail">Request failed: ${esc(e.message)}</div>`;
  } finally {
    btn.disabled = false; spin.classList.remove("on");
  }
}

function renderResult(data){
  const r = document.getElementById("results");
  let html = "";
  html += `<div class="card"><h3>Coordinator routing</h3><div class="muted">${esc(data.analysis)}</div></div>`;
  (data.specialists || []).forEach(s => {
    html += `<div class="card"><h3>${esc(s.label)} ${s.failed ? '<span class="fail">(failed)</span>' : ''}</h3>`;
    html += `<div class="muted">brief: ${esc(s.instructions)}</div>`;
    html += `<pre class="${s.failed ? 'fail' : ''}">${esc(s.output)}</pre></div>`;
  });
  html += `<div class="card deliverable"><h3>Final deliverable</h3><pre>${esc(data.deliverable)}</pre></div>`;
  r.innerHTML = html;
}

async function loadMemory(agent){
  const view = document.getElementById("memView");
  view.innerHTML = `<div class="muted">loading ${esc(agent)}…</div>`;
  const resp = await fetch("/memory/" + encodeURIComponent(agent));
  const data = await resp.json();
  const rows = data.rows || [];
  if(!rows.length){ view.innerHTML = `<div class="muted">No memory yet for ${esc(agent)}.</div>`; return; }
  let html = `<table class="memtable"><tr><th>When</th><th>Task</th><th>Output (truncated)</th></tr>`;
  rows.slice().reverse().forEach(row => {
    const out = (row.output || "").slice(0, 300);
    html += `<tr><td>${esc(row.timestamp)}</td><td>${esc((row.task||"").slice(0,80))}</td><td>${esc(out)}</td></tr>`;
  });
  html += `</table>`;
  view.innerHTML = html;
}

// ---- Telegram bot control ----
let tgAdminsTouched = false, tgTokenTouched = false;
document.getElementById("tgAdmins").addEventListener("input", () => tgAdminsTouched = true);
document.getElementById("tgToken").addEventListener("input", () => tgTokenTouched = true);

function tgMsg(text, fail){
  const el = document.getElementById("tgMsg");
  el.textContent = text || "";
  el.className = "muted" + (fail ? " fail" : "");
}

function renderTgStatus(s){
  const dot = document.getElementById("tgDot");
  dot.className = "dot " + (s.running ? "dot-on" : "dot-off");
  dot.title = s.running ? "running" : "stopped";
  document.getElementById("tgState").textContent =
    s.running ? `running (pid ${s.pid})` : (s.exit_code != null ? `stopped (exit ${s.exit_code})` : "stopped");
  const admins = (s.admin_ids && s.admin_ids.length) ? s.admin_ids.join(", ") : "— none (OPEN mode) —";
  document.getElementById("tgInfo").innerHTML =
    `token (${esc(s.token_env)}): <code>${s.token_set ? esc(s.token_masked) : "not set"}</code>` +
    ` &nbsp;•&nbsp; admins: <code>${esc(admins)}</code>`;
  document.getElementById("tgStart").disabled = s.running || !s.token_set;
  document.getElementById("tgStop").disabled = !s.running;
  // Don't clobber fields the user is editing.
  if(!tgAdminsTouched) document.getElementById("tgAdmins").value = (s.admin_ids||[]).join(", ");
  const logEl = document.getElementById("tgLog");
  logEl.textContent = (!s.running && s.recent_log) ? s.recent_log : "";
}

async function tgStatus(){
  try {
    const r = await fetch("/telegram/status");
    renderTgStatus(await r.json());
  } catch(e){ /* dashboard still loading */ }
}

async function tgSave(){
  const body = {
    admin_ids: document.getElementById("tgAdmins").value,
  };
  const tok = document.getElementById("tgToken").value.trim();
  if(tgTokenTouched && tok) body.token = tok;
  const r = await fetch("/telegram/settings", {
    method: "POST", headers: {"Content-Type":"application/json"}, body: JSON.stringify(body)
  });
  const d = await r.json();
  if(!r.ok){ tgMsg(d.error || "save failed", true); return; }
  tgTokenTouched = false; document.getElementById("tgToken").value = "";
  tgAdminsTouched = false;
  tgMsg(d.note || "Saved.");
  renderTgStatus(d);
}

async function tgStart(){
  tgMsg("starting…");
  const r = await fetch("/telegram/start", {method: "POST"});
  const d = await r.json();
  if(!r.ok){ tgMsg(d.error || "start failed", true); renderTgStatus(d); return; }
  tgMsg("started."); renderTgStatus(d);
}

async function tgStop(){
  tgMsg("stopping…");
  const r = await fetch("/telegram/stop", {method: "POST"});
  const d = await r.json();
  tgMsg("stopped."); renderTgStatus(d);
}

tgStatus();
setInterval(tgStatus, 3000);   // live status refresh
</script>
</body>
</html>"""


@app.route("/")
def index():
    return render_template_string(
        PAGE,
        provider=CFG.get("provider"),
        agents=list(CFG["specialists"].keys()),
        memory_on=MEMORY is not None,
        memory_state="on" if MEMORY is not None else "off",
    )


@app.route("/run", methods=["POST"])
def run_endpoint():
    payload = request.get_json(silent=True) or {}
    task = (payload.get("task") or "").strip()
    if not task:
        return jsonify({"error": "empty task"}), 400
    try:
        result = _submit(run_task(task, CFG, MEMORY))
    except Exception as exc:  # noqa: BLE001 - surface any run failure to the UI
        log.exception("run failed")
        return jsonify({"error": str(exc)}), 500
    return jsonify(result.to_dict())


@app.route("/memory/<agent>")
def memory_endpoint(agent):
    if MEMORY is None:
        return jsonify({"agent": agent, "rows": []})
    return jsonify({"agent": agent, "rows": MEMORY.recall(agent, limit=20)})


# --- Telegram bot control endpoints ------------------------------------------------------

@app.route("/telegram/status")
def telegram_status():
    return jsonify(_bot_status())


@app.route("/telegram/settings", methods=["POST"])
def telegram_settings():
    payload = request.get_json(silent=True) or {}
    if "token" in payload:
        _bot_settings["token"] = (payload.get("token") or "").strip()
    if "admin_ids" in payload:
        raw = payload.get("admin_ids") or ""
        ids = []
        for part in str(raw).replace(",", " ").split():
            try:
                ids.append(int(part))
            except ValueError:
                return jsonify({"error": f"invalid admin id: {part!r}"}), 400
        _bot_settings["admin_ids"] = ids
    note = "Saved. Restart the bot to apply." if _bot_running() else "Saved."
    return jsonify({"ok": True, "note": note, **_bot_status()})


@app.route("/telegram/start", methods=["POST"])
def telegram_start():
    try:
        _start_bot()
    except RuntimeError as exc:
        return jsonify({"error": str(exc), **_bot_status()}), 400
    return jsonify({"ok": True, **_bot_status()})


@app.route("/telegram/stop", methods=["POST"])
def telegram_stop():
    _stop_bot()
    return jsonify({"ok": True, **_bot_status()})


if __name__ == "__main__":
    print("Dashboard: http://127.0.0.1:5000  (provider: %s, memory: %s)"
          % (CFG.get("provider"), "on" if MEMORY else "off"))
    app.run(host="127.0.0.1", port=5000, threaded=True)
