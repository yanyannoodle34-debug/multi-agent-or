"""Telegram control bot for the multi-agent orchestrator.

Drive the whole system from Telegram with inline buttons — no terminal needed:

  • Run / stop orchestration tasks, with live progress ("live dashboard")
  • Status view: active provider, models, memory, running task
  • Per-agent memory: download the CSV, upload a replacement CSV, or clear it
  • API keys & provider: set a provider's key (session-scoped) and switch provider
  • Admin-gated: only configured Telegram user IDs may control the bot
  • Friendly navigation: every screen has a ⬅️ Back button; robust error handling

Setup:
    pip install python-telegram-bot>=21
    export TELEGRAM_BOT_TOKEN=...        # from @BotFather
    # plus your LLM provider key, e.g.  export OPENROUTER_API_KEY=sk-or-...
    # set telegram.admin_ids in config.yaml to your Telegram user id(s)
    python telegram_bot.py

The token and admin list come from config.yaml (`telegram:` block); the token itself is
read from the environment variable named there — never hardcode it.
"""

from __future__ import annotations

import asyncio
import csv
import html
import io
import logging
import os

from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Update,
)
from telegram.constants import ParseMode
from telegram.error import BadRequest
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

import orchestrator
import providers
from memory import FIELDS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("orchestration.telegram")

# Shared config/state, loaded once at startup.
CFG = orchestrator.activate(orchestrator.load_config())
MEMORY = orchestrator.build_memory(CFG)
TG = CFG.get("telegram") or {}
TELEGRAM_MAX = 4096


def _load_admin_ids() -> set[int]:
    """Admin IDs from the TELEGRAM_ADMIN_IDS env override (comma-separated) if present,
    else from config.yaml's telegram.admin_ids. An explicit empty env value = OPEN mode.
    The env path lets the web dashboard launch this bot with admins set, no config edit."""
    raw = os.environ.get("TELEGRAM_ADMIN_IDS")
    if raw is not None:
        ids: set[int] = set()
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                ids.add(int(part))
            except ValueError:
                log.warning("Ignoring invalid admin id %r", part)
        return ids
    return {int(x) for x in (TG.get("admin_ids") or [])}


ADMIN_IDS = _load_admin_ids()


# --------------------------------------------------------------------------- helpers

def is_admin(user_id: int | None) -> bool:
    """Admins-only when admin_ids is set; open mode (with a warning) when it's empty."""
    if not ADMIN_IDS:
        return True
    return user_id in ADMIN_IDS


def mask_secret(value: str | None) -> str:
    if not value:
        return "— not set —"
    if len(value) <= 8:
        return "•" * len(value)
    return f"{value[:4]}…{value[-3:]} ({len(value)} chars)"


def chunk_text(text: str, size: int = TELEGRAM_MAX - 96) -> list[str]:
    """Split long text into Telegram-sized chunks, preferring newline boundaries."""
    text = text or "(empty)"
    chunks: list[str] = []
    while len(text) > size:
        cut = text.rfind("\n", 0, size)
        if cut < size // 2:  # no good newline; hard-cut
            cut = size
        chunks.append(text[:cut])
        text = text[cut:].lstrip("\n")
    chunks.append(text)
    return chunks


async def send_long(chat, text: str) -> None:
    for part in chunk_text(text):
        await chat.send_message(part)


def _providers() -> dict:
    return CFG.get("providers") or {}


def _active_key_env() -> str:
    name = CFG.get("provider")
    pcfg = _providers().get(name, {})
    return pcfg.get("api_key_env", "")


# --------------------------------------------------------------------------- keyboards

def kb_main() -> InlineKeyboardMarkup:
    running = "⏹ Stop task"
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("▶️ Run task", callback_data="menu:run"),
         InlineKeyboardButton(running, callback_data="run:stop")],
        [InlineKeyboardButton("📊 Status", callback_data="menu:status"),
         InlineKeyboardButton("🧠 Agents", callback_data="menu:agents")],
        [InlineKeyboardButton("🔑 API keys", callback_data="menu:keys"),
         InlineKeyboardButton("⚙️ Admin", callback_data="menu:admin")],
    ])


def kb_back(target: str = "menu:main") -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Back", callback_data=target)]])


def kb_agents() -> InlineKeyboardMarkup:
    rows = [[InlineKeyboardButton(spec.get("label", key), callback_data=f"agent:{key}")]
            for key, spec in CFG["specialists"].items()]
    rows.append([InlineKeyboardButton("⬅️ Back", callback_data="menu:main")])
    return InlineKeyboardMarkup(rows)


def kb_agent(agent: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📥 Download CSV", callback_data=f"agent:dl:{agent}"),
         InlineKeyboardButton("⬆️ Update CSV", callback_data=f"agent:up:{agent}")],
        [InlineKeyboardButton("🗑 Clear memory", callback_data=f"agent:clr:{agent}")],
        [InlineKeyboardButton("⬅️ Back", callback_data="menu:agents")],
    ])


def kb_keys() -> InlineKeyboardMarkup:
    rows = []
    active = CFG.get("provider")
    for name in _providers():
        tick = "✅ " if name == active else ""
        rows.append([
            InlineKeyboardButton(f"{tick}Use {name}", callback_data=f"key:use:{name}"),
            InlineKeyboardButton(f"Set {name} key", callback_data=f"key:set:{name}"),
        ])
    rows.append([InlineKeyboardButton("⬅️ Back", callback_data="menu:main")])
    return InlineKeyboardMarkup(rows)


def kb_admin() -> InlineKeyboardMarkup:
    mem = "🧠 Memory: ON — turn off" if MEMORY else "🧠 Memory: OFF — turn on"
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(mem, callback_data="admin:togglemem")],
        [InlineKeyboardButton("⬅️ Back", callback_data="menu:main")],
    ])


# --------------------------------------------------------------------------- screens

def text_main() -> str:
    return ("<b>Multi-Agent Orchestrator</b>\n"
            f"Provider: <code>{html.escape(str(CFG.get('provider')))}</code>   "
            f"Memory: <code>{'on' if MEMORY else 'off'}</code>\n\n"
            "Pick an action:")


def text_status(context: ContextTypes.DEFAULT_TYPE) -> str:
    m = CFG.get("models", {})
    running = _running_task(context)
    last = context.chat_data.get("last_run")
    lines = [
        "<b>📊 Status</b>",
        f"Provider: <code>{html.escape(str(CFG.get('provider')))}</code>",
        f"  coordinator: <code>{html.escape(str(m.get('coordinator')))}</code>",
        f"  specialist:  <code>{html.escape(str(m.get('specialist')))}</code>",
        f"  synthesizer: <code>{html.escape(str(m.get('synthesizer')))}</code>",
        f"API key ({_active_key_env()}): <code>{html.escape(mask_secret(os.environ.get(_active_key_env())))}</code>",
        f"Memory: <code>{'on' if MEMORY else 'off'}</code>",
        f"Specialists: <code>{', '.join(CFG['specialists'])}</code>",
        f"Task running: <code>{'yes' if running else 'no'}</code>",
    ]
    if last:
        lines.append(f"Last run: <code>{html.escape(last)}</code>")
    if not ADMIN_IDS:
        lines.append("\n⚠️ <i>No admin_ids set — bot is in OPEN mode.</i>")
    return "\n".join(lines)


def text_keys() -> str:
    lines = ["<b>🔑 API keys &amp; provider</b>",
             f"Active provider: <code>{html.escape(str(CFG.get('provider')))}</code>\n"]
    for name, pcfg in _providers().items():
        env = pcfg.get("api_key_env", "?")
        lines.append(f"• <b>{html.escape(name)}</b> ({env}): "
                     f"<code>{html.escape(mask_secret(os.environ.get(env)))}</code>")
    lines.append("\n“Use” switches the active provider. “Set … key” updates that key for "
                 "this bot session.")
    return "\n".join(lines)


def text_admin() -> str:
    admins = ", ".join(str(a) for a in ADMIN_IDS) if ADMIN_IDS else "— none (OPEN mode) —"
    return ("<b>⚙️ Admin</b>\n"
            f"Admin IDs: <code>{html.escape(admins)}</code>\n"
            f"Memory: <code>{'on' if MEMORY else 'off'}</code>\n"
            f"Provider: <code>{html.escape(str(CFG.get('provider')))}</code>")


# --------------------------------------------------------------------------- task run/stop

def _running_task(context: ContextTypes.DEFAULT_TYPE) -> asyncio.Task | None:
    task = context.chat_data.get("run_task")
    return task if (task and not task.done()) else None


async def _do_run(task_text: str, status_msg, context: ContextTypes.DEFAULT_TYPE) -> None:
    async def on_phase(msg: str) -> None:
        try:
            await status_msg.edit_text(
                f"⏳ <b>Running</b>\n<code>{html.escape(task_text[:120])}</code>\n\n{html.escape(msg)}",
                parse_mode=ParseMode.HTML,
                reply_markup=InlineKeyboardMarkup(
                    [[InlineKeyboardButton("⏹ Stop", callback_data="run:stop")]]),
            )
        except BadRequest:
            pass  # "message is not modified" or similar — safe to ignore

    try:
        result = await orchestrator.run_task(task_text, CFG, MEMORY, on_phase=on_phase)
        engaged = ", ".join(a.label for a in result.assignments)
        context.chat_data["last_run"] = f"{engaged} — deliverable {len(result.deliverable)} chars"
        await status_msg.edit_text(
            f"✅ <b>Done</b> — engaged: {html.escape(engaged)}",
            parse_mode=ParseMode.HTML,
            reply_markup=kb_back(),
        )
        await send_long(status_msg.chat, "📦 Deliverable:\n\n" + result.deliverable)
    except asyncio.CancelledError:
        try:
            await status_msg.edit_text("⏹ <b>Stopped.</b>", parse_mode=ParseMode.HTML,
                                       reply_markup=kb_back())
        except BadRequest:
            pass
        raise
    except Exception as exc:  # noqa: BLE001 - surface any failure to the user, keep bot alive
        log.exception("run failed")
        try:
            await status_msg.edit_text(f"❌ <b>Error:</b> {html.escape(str(exc))}",
                                       parse_mode=ParseMode.HTML, reply_markup=kb_back())
        except BadRequest:
            pass
    finally:
        context.chat_data.pop("run_task", None)


async def start_run(task_text: str, chat, context: ContextTypes.DEFAULT_TYPE) -> None:
    if _running_task(context):
        await chat.send_message("⚠️ A task is already running. Stop it first.",
                                reply_markup=kb_back())
        return
    status_msg = await chat.send_message("⏳ Starting…")
    task = asyncio.create_task(_do_run(task_text, status_msg, context))
    context.chat_data["run_task"] = task


# --------------------------------------------------------------------------- command handlers

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_admin(update.effective_user.id):
        await update.message.reply_text("⛔ Not authorized. Your ID: "
                                        f"{update.effective_user.id}")
        return
    context.user_data.pop("await", None)
    await update.message.reply_text(text_main(), parse_mode=ParseMode.HTML,
                                    reply_markup=kb_main())


# --------------------------------------------------------------------------- callback router

async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    q = update.callback_query
    if not is_admin(q.from_user.id):
        await q.answer("Not authorized", show_alert=True)
        return
    await q.answer()
    data = q.data or ""

    async def edit(text: str, kb: InlineKeyboardMarkup) -> None:
        try:
            await q.edit_message_text(text, parse_mode=ParseMode.HTML, reply_markup=kb)
        except BadRequest as e:
            if "not modified" not in str(e).lower():
                raise

    # --- top-level menus ---
    if data == "menu:main":
        context.user_data.pop("await", None)
        return await edit(text_main(), kb_main())
    if data == "menu:status":
        return await edit(text_status(context), kb_back())
    if data == "menu:agents":
        return await edit("🧠 <b>Agent memory</b>\nPick an agent:", kb_agents())
    if data == "menu:keys":
        return await edit(text_keys(), kb_keys())
    if data == "menu:admin":
        return await edit(text_admin(), kb_admin())

    # --- run / stop ---
    if data == "menu:run":
        context.user_data["await"] = {"kind": "task"}
        return await edit("▶️ Send me the <b>task</b> as a message and I'll run it.",
                          kb_back())
    if data == "run:stop":
        task = _running_task(context)
        if task:
            task.cancel()
            await q.answer("Stopping…")
        else:
            await q.answer("No task running", show_alert=True)
        return

    # --- agents ---
    if data.startswith("agent:dl:"):
        return await _send_agent_csv(q, data.split(":", 2)[2])
    if data.startswith("agent:up:"):
        agent = data.split(":", 2)[2]
        context.user_data["await"] = {"kind": "csv", "agent": agent}
        return await edit(f"⬆️ Send me a <b>.csv</b> file to replace <code>{html.escape(agent)}</code>"
                          f"'s memory.\nHeader must be: <code>{html.escape(','.join(FIELDS))}</code>",
                          kb_back("menu:agents"))
    if data.startswith("agent:clr:"):
        agent = data.split(":", 2)[2]
        removed = MEMORY.clear(agent) if MEMORY else False
        await q.answer("Cleared" if removed else "Nothing to clear")
        return await edit(_text_agent(agent), kb_agent(agent))
    if data.startswith("agent:"):
        agent = data.split(":", 1)[1]
        return await edit(_text_agent(agent), kb_agent(agent))

    # --- keys / provider ---
    if data.startswith("key:use:"):
        name = data.split(":", 2)[2]
        try:
            _switch_provider(name)
            await q.answer(f"Now using {name}")
        except SystemExit as exc:
            await q.answer(str(exc), show_alert=True)
        return await edit(text_keys(), kb_keys())
    if data.startswith("key:set:"):
        name = data.split(":", 2)[2]
        context.user_data["await"] = {"kind": "apikey", "provider": name}
        env = _providers().get(name, {}).get("api_key_env", "?")
        return await edit(f"🔑 Send the API key for <b>{html.escape(name)}</b> "
                          f"(stored in <code>{html.escape(env)}</code> for this session).\n"
                          "I'll delete your message if I can.", kb_back("menu:keys"))

    # --- admin ---
    if data == "admin:togglemem":
        _toggle_memory()
        await q.answer("Memory " + ("on" if MEMORY else "off"))
        return await edit(text_admin(), kb_admin())

    await q.answer("Unknown action", show_alert=True)


def _text_agent(agent: str) -> str:
    label = CFG["specialists"].get(agent, {}).get("label", agent)
    rows = MEMORY.recall(agent, limit=3) if MEMORY else []
    exists = MEMORY.exists(agent) if MEMORY else False
    body = [f"🧠 <b>{html.escape(label)}</b> memory",
            f"CSV: <code>{'present' if exists else 'empty'}</code>"]
    if rows:
        body.append("\nRecent:")
        for r in rows:
            body.append(f"• <i>{html.escape(r.get('timestamp', ''))}</i>: "
                        f"{html.escape((r.get('task') or '')[:60])}")
    return "\n".join(body)


async def _send_agent_csv(q, agent: str) -> None:
    if not MEMORY or not MEMORY.exists(agent):
        await q.answer("No CSV yet for this agent", show_alert=True)
        return
    try:
        with open(MEMORY.path(agent), "rb") as f:
            await q.message.chat.send_document(document=f, filename=f"{agent}.csv",
                                               caption=f"{agent} memory")
    except Exception as exc:  # noqa: BLE001
        log.exception("csv download failed")
        await q.message.chat.send_message(f"❌ Could not send CSV: {html.escape(str(exc))}",
                                          parse_mode=ParseMode.HTML)


# --------------------------------------------------------------------------- input handlers

async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_admin(update.effective_user.id):
        return
    pending = context.user_data.get("await")
    if not pending:
        await update.message.reply_text("Use the menu:", reply_markup=kb_main())
        return

    kind = pending.get("kind")
    if kind == "task":
        context.user_data.pop("await", None)
        await start_run(update.message.text.strip(), update.message.chat, context)
    elif kind == "apikey":
        context.user_data.pop("await", None)
        await _set_api_key(pending["provider"], update, context)
    else:
        await update.message.reply_text("Send a CSV file, or go Back.",
                                        reply_markup=kb_back("menu:agents"))


async def on_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_admin(update.effective_user.id):
        return
    pending = context.user_data.get("await")
    if not pending or pending.get("kind") != "csv":
        await update.message.reply_text("I wasn't expecting a file. Use the menu:",
                                        reply_markup=kb_main())
        return
    agent = pending["agent"]
    context.user_data.pop("await", None)
    await _save_agent_csv(agent, update, context)


async def _set_api_key(provider: str, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    key = update.message.text.strip()
    env = _providers().get(provider, {}).get("api_key_env")
    if not env:
        await update.message.reply_text("❌ Unknown provider.", reply_markup=kb_back("menu:keys"))
        return
    os.environ[env] = key
    providers.configure(provider, _providers().get(provider))  # rebuild client lazily w/ new key
    # If it's the active provider, reset so the next call uses the new key.
    if provider == CFG.get("provider"):
        providers.configure(provider, _providers().get(provider))
    try:
        await update.message.delete()  # best-effort: strip the secret from chat
    except Exception:  # noqa: BLE001
        pass
    await update.message.chat.send_message(
        f"✅ Key for <b>{html.escape(provider)}</b> set: "
        f"<code>{html.escape(mask_secret(key))}</code>",
        parse_mode=ParseMode.HTML, reply_markup=kb_back("menu:keys"))


async def _save_agent_csv(agent: str, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    doc = update.message.document
    if not doc.file_name.lower().endswith(".csv"):
        await update.message.reply_text("❌ That's not a .csv file.",
                                        reply_markup=kb_back("menu:agents"))
        return
    if not MEMORY:
        await update.message.reply_text("❌ Memory is disabled in config.",
                                        reply_markup=kb_back("menu:agents"))
        return
    try:
        tg_file = await doc.get_file()
        data = bytes(await tg_file.download_as_bytearray())
        # Validate it parses as CSV with the expected header.
        reader = csv.reader(io.StringIO(data.decode("utf-8", errors="replace")))
        header = next(reader, [])
        if [h.strip() for h in header] != FIELDS:
            await update.message.reply_text(
                "❌ Header must be exactly:\n<code>" + html.escape(",".join(FIELDS)) + "</code>",
                parse_mode=ParseMode.HTML, reply_markup=kb_back("menu:agents"))
            return
        with open(MEMORY.path(agent), "wb") as f:
            f.write(data)
        rows = max(0, len(data.decode("utf-8", errors="replace").splitlines()) - 1)
        await update.message.reply_text(
            f"✅ Updated <b>{html.escape(agent)}</b> memory ({rows} rows).",
            parse_mode=ParseMode.HTML, reply_markup=kb_back("menu:agents"))
    except Exception as exc:  # noqa: BLE001
        log.exception("csv upload failed")
        await update.message.reply_text(f"❌ Upload failed: {html.escape(str(exc))}",
                                        parse_mode=ParseMode.HTML, reply_markup=kb_back("menu:agents"))


# --------------------------------------------------------------------------- runtime config ops

def _switch_provider(name: str) -> None:
    if name not in _providers():
        raise SystemExit(f"unknown provider {name}")
    CFG["provider"] = name
    orchestrator.activate(CFG)  # re-resolves models + configures providers (lazy client)


def _toggle_memory() -> None:
    global MEMORY
    mc = CFG.setdefault("memory", {})
    mc["enabled"] = not bool(mc.get("enabled"))
    MEMORY = orchestrator.build_memory(CFG)


# --------------------------------------------------------------------------- error handler

async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    log.error("Unhandled error", exc_info=context.error)
    try:
        if isinstance(update, Update) and update.effective_chat:
            await context.bot.send_message(update.effective_chat.id,
                                           "⚠️ Something went wrong. Please try again.")
    except Exception:  # noqa: BLE001
        pass


# --------------------------------------------------------------------------- entrypoint

def build_application(token: str) -> Application:
    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler(["start", "menu", "help"], cmd_start))
    app.add_handler(CallbackQueryHandler(on_callback))
    app.add_handler(MessageHandler(filters.Document.ALL, on_document))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    app.add_error_handler(on_error)
    return app


def main() -> None:
    if not TG.get("enabled", True):
        raise SystemExit("Telegram bot disabled in config.yaml (telegram.enabled: false).")
    token_env = TG.get("token_env", "TELEGRAM_BOT_TOKEN")
    token = os.environ.get(token_env)
    if not token:
        raise SystemExit(f"Set {token_env} (bot token from @BotFather).")
    if not ADMIN_IDS:
        log.warning("No telegram.admin_ids set — bot runs in OPEN mode (anyone can control it).")
    log.info("Starting Telegram bot (provider=%s, memory=%s, admins=%s)",
             CFG.get("provider"), "on" if MEMORY else "off", ADMIN_IDS or "open")
    app = build_application(token)
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
