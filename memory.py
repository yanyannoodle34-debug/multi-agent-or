"""Per-agent persistent memory, backed by CSV (Excel-compatible), one file per agent.

Each specialist's runs are appended to `<dir>/<agent>.csv` with columns:

    timestamp, run_id, task, instructions, output

`recall` returns the most recent rows and `recall_context` renders them into a short
block that gets fed back into the agent's next prompt — so each bot carries continuity
across runs. CSV files open directly in Excel; set `export_xlsx` to also mirror each
agent's history into `<dir>/<agent>.xlsx` (requires `openpyxl`).
"""

from __future__ import annotations

import csv
import datetime
import logging
import os
import threading

log = logging.getLogger("orchestration.memory")

FIELDS = ["timestamp", "run_id", "task", "instructions", "output"]


def new_run_id() -> str:
    """A short, sortable id shared by all agents in one orchestration run."""
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


class MemoryStore:
    def __init__(self, base_dir: str = "data", recall: int = 3, export_xlsx: bool = False):
        self.base_dir = base_dir
        self.recall_n = recall
        self.export_xlsx = export_xlsx
        os.makedirs(base_dir, exist_ok=True)
        self._lock = threading.Lock()  # CSV writes may come from concurrent dashboard requests

    def _path(self, agent: str) -> str:
        safe = "".join(c for c in agent if c.isalnum() or c in ("-", "_")) or "agent"
        return os.path.join(self.base_dir, f"{safe}.csv")

    def path(self, agent: str) -> str:
        """Public path to an agent's CSV (may not exist yet)."""
        return self._path(agent)

    def exists(self, agent: str) -> bool:
        return os.path.exists(self._path(agent))

    def clear(self, agent: str) -> bool:
        """Delete an agent's CSV memory. Returns True if a file was removed."""
        p = self._path(agent)
        with self._lock:
            if os.path.exists(p):
                os.remove(p)
                log.info("memory: cleared %s (%s)", agent, p)
                return True
        return False

    def record(self, agent: str, task: str, instructions: str, output, run_id: str = "") -> None:
        """Append one run's result for `agent`. `output` may be a string or an Exception."""
        body = output if isinstance(output, str) else f"(failed: {output})"
        row = {
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "run_id": run_id,
            "task": task,
            "instructions": instructions,
            "output": body,
        }
        path = self._path(agent)
        with self._lock:
            is_new = not os.path.exists(path)
            with open(path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDS)
                if is_new:
                    writer.writeheader()
                writer.writerow(row)
            if self.export_xlsx:
                self._export_xlsx(agent, path)
        log.info("memory: recorded %s run (run_id=%s) -> %s", agent, run_id, path)

    def recall(self, agent: str, limit: int | None = None) -> list[dict]:
        """Return up to `limit` most-recent rows for `agent` (newest last)."""
        limit = self.recall_n if limit is None else limit
        path = self._path(agent)
        if limit <= 0 or not os.path.exists(path):
            return []
        with self._lock, open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        return rows[-limit:]

    def recall_context(self, agent: str, limit: int | None = None) -> str:
        """Render recent memory into a prompt block; empty string if there's no history."""
        rows = self.recall(agent, limit)
        if not rows:
            return ""
        parts = []
        for r in rows:
            out = r.get("output", "")
            out = out[:600] + ("…" if len(out) > 600 else "")
            parts.append(f"- [{r.get('timestamp', '')}] task: {r.get('task', '')[:120]}\n"
                         f"  you produced: {out}")
        return ("Your recent related work (for continuity — build on it, don't repeat "
                "it verbatim):\n" + "\n".join(parts))

    def _export_xlsx(self, agent: str, csv_path: str) -> None:
        try:
            from openpyxl import Workbook
        except ImportError:
            log.warning("export_xlsx is on but openpyxl isn't installed; "
                        "run `pip install openpyxl`. Skipping xlsx export.")
            return
        wb = Workbook()
        ws = wb.active
        ws.title = agent[:31] or "agent"
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.reader(f):
                ws.append(row)
        wb.save(os.path.splitext(csv_path)[0] + ".xlsx")
