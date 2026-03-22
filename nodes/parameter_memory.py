"""
Parameter Memory v2 — Compact persistent parameter history for ComfyUI.

Two components:
  1. ParameterHistoryMEC node: displays stored parameter history in the workflow
  2. Server route /mec/param_history: receives delta snapshots from JS, stores in SQLite

The JS extension (js/parameter_memory.js) handles:
  - Real-time widget change interception on ALL nodes
  - Pre-execution diff snapshots (only non-default values)
  - Hover tooltips (Alt+hover → see previous value & default)
  - Right-click context menus (history, diff, reset, presets)

v2 changes:
  - DB stores one row per node per run (JSON diff blob) instead of one row per param
  - Accepts delta-only payloads from JS (only changed-from-default params)
  - Auto-prunes old runs beyond MAX_RUNS
  - Cleaner grouped output formatting
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime

logger = logging.getLogger("MEC")

# ── Config ────────────────────────────────────────────────────────────
MAX_RUNS = 30  # auto-prune: keep only this many distinct runs

# ── DB path lives next to this file ──────────────────────────────────
_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "param_history.db")
_DB_PATH = os.path.normpath(_DB_PATH)


def _get_db():
    """Get a SQLite connection with WAL mode and v2 schema."""
    conn = sqlite3.connect(_DB_PATH, timeout=5.0)
    conn.execute("PRAGMA journal_mode=WAL")
    # v2 schema: one row per node per run, delta JSON blob
    conn.execute("""
        CREATE TABLE IF NOT EXISTS param_diffs (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            ts         TEXT    NOT NULL,
            run_id     INTEGER NOT NULL,
            node_id    TEXT    NOT NULL,
            node_title TEXT    NOT NULL,
            node_class TEXT    NOT NULL,
            delta_json TEXT    NOT NULL
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_diffs_run
        ON param_diffs(run_id)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_diffs_class
        ON param_diffs(node_class)
    """)
    conn.commit()
    return conn


def _prune_old_runs(conn):
    """Keep only the last MAX_RUNS distinct runs."""
    try:
        rows = conn.execute(
            "SELECT DISTINCT run_id FROM param_diffs ORDER BY run_id DESC"
        ).fetchall()
        if len(rows) > MAX_RUNS:
            cutoff = rows[MAX_RUNS - 1][0]
            conn.execute("DELETE FROM param_diffs WHERE run_id < ?", (cutoff,))
            conn.commit()
    except Exception as e:
        logger.warning(f"[MEC] prune error: {e}")


def _store_snapshot(data: dict):
    """Store a run snapshot (delta format) into the DB.

    data format from JS v2:
    {
        "node_id": {
            "title": "KSampler",
            "class": "KSampler",
            "run_id": 5,
            "ts": "2026-03-19 10:00:00",
            "delta": {"steps": 20, "cfg": 7.0}   // only non-default params
        },
        ...
    }
    """
    try:
        conn = _get_db()
        rows = []
        for node_id, info in data.items():
            ts = info.get("ts", datetime.now().isoformat(sep=" ", timespec="seconds"))
            run_id = info.get("run_id", 0)
            title = info.get("title", "")
            cls = info.get("class", "")
            # v2: accept "delta" field; fall back to "values" for backward compat
            delta = info.get("delta", info.get("values", {}))
            if not delta:
                continue  # skip nodes with no changes from defaults
            rows.append((ts, run_id, str(node_id), title, cls, json.dumps(delta)))

        if rows:
            conn.executemany(
                "INSERT INTO param_diffs (ts, run_id, node_id, node_title, node_class, delta_json) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                rows,
            )
            conn.commit()
            _prune_old_runs(conn)
        conn.close()
    except Exception as e:
        logger.warning(f"[MEC] param_history DB write error: {e}")


def _query_history(node_class: str = "", last_n_runs: int = 10) -> list[dict]:
    """Query parameter history from the DB."""
    try:
        conn = _get_db()
        if node_class:
            rows = conn.execute(
                "SELECT ts, run_id, node_id, node_title, node_class, delta_json "
                "FROM param_diffs WHERE node_class = ? "
                "ORDER BY run_id DESC, id DESC LIMIT ?",
                (node_class, last_n_runs * 20),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT ts, run_id, node_id, node_title, node_class, delta_json "
                "FROM param_diffs ORDER BY run_id DESC, id DESC LIMIT ?",
                (last_n_runs * 20,),
            ).fetchall()
        conn.close()

        results = []
        for r in rows:
            results.append({
                "ts": r[0], "run_id": r[1], "node_id": r[2],
                "node_title": r[3], "node_class": r[4],
                "delta": json.loads(r[5]),
            })
        return results
    except Exception as e:
        logger.warning(f"[MEC] param_history DB query error: {e}")
        return []


def _diff_runs(run_a: int, run_b: int) -> list[dict]:
    """Compare two runs and return parameter differences."""
    try:
        conn = _get_db()

        def _get_run_deltas(rid):
            rows = conn.execute(
                "SELECT node_id, node_title, node_class, delta_json "
                "FROM param_diffs WHERE run_id = ?",
                (rid,),
            ).fetchall()
            nodes = {}
            for r in rows:
                nodes[r[0]] = {
                    "node_id": r[0], "node_title": r[1],
                    "node_class": r[2], "delta": json.loads(r[3]),
                }
            return nodes

        na = _get_run_deltas(run_a)
        nb = _get_run_deltas(run_b)
        conn.close()

        diffs = []
        all_node_ids = set(na.keys()) | set(nb.keys())
        for nid in sorted(all_node_ids):
            da = na.get(nid, {}).get("delta", {})
            db_ = nb.get(nid, {}).get("delta", {})
            info = na.get(nid) or nb.get(nid)
            all_params = set(da.keys()) | set(db_.keys())
            for param in sorted(all_params):
                va = da.get(param)
                vb = db_.get(param)
                if va != vb:
                    diffs.append({
                        "node_id":    info["node_id"],
                        "node_title": info["node_title"],
                        "node_class": info["node_class"],
                        "param_name": param,
                        "run_a":      va,
                        "run_b":      vb,
                    })
        return diffs
    except Exception as e:
        logger.warning(f"[MEC] param_history diff error: {e}")
        return []


# ══════════════════════════════════════════════════════════════════════
#  ComfyUI Node: Parameter History Viewer
# ══════════════════════════════════════════════════════════════════════

class ParameterHistoryMEC:
    """View and query the parameter history database.

    Connects to the SQLite DB written by the JS extension and provides
    structured output of parameter change history.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["all_history", "last_run_diff", "node_class_filter"], {
                    "default": "all_history",
                    "tooltip": (
                        "all_history: show recent parameter changes across all nodes\n"
                        "last_run_diff: show what changed between the last two runs\n"
                        "node_class_filter: filter history to a specific node class"
                    ),
                }),
                "last_n_runs": ("INT", {
                    "default": 5, "min": 1, "max": 100, "step": 1,
                    "tooltip": "How many recent runs to include",
                }),
            },
            "optional": {
                "node_class_filter": ("STRING", {
                    "default": "",
                    "tooltip": "Node class name to filter by (e.g. 'KSampler')",
                }),
                "run_a": ("INT", {
                    "default": 0, "min": 0,
                    "tooltip": "First run number for diff mode (0 = auto-detect last two)",
                }),
                "run_b": ("INT", {
                    "default": 0, "min": 0,
                    "tooltip": "Second run number for diff mode (0 = auto-detect last two)",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("history_report",)
    OUTPUT_NODE = True
    FUNCTION = "query"
    CATEGORY = "MaskEditControl/Utils"
    DESCRIPTION = (
        "Query the parameter history database. Shows what parameters were changed, "
        "when, and what the previous values were. Supports run-to-run diffs."
    )

    def query(self, mode, last_n_runs, node_class_filter="", run_a=0, run_b=0):
        if mode == "last_run_diff":
            return self._diff_mode(run_a, run_b)
        elif mode == "node_class_filter":
            return self._filter_mode(node_class_filter, last_n_runs)
        else:
            return self._all_mode(last_n_runs)

    def _all_mode(self, last_n_runs):
        rows = _query_history(last_n_runs=last_n_runs)
        if not rows:
            return ("No parameter history recorded yet. Run the workflow at least once.",)

        # Group by run_id
        runs: dict[int, list] = {}
        for r in rows:
            rid = r["run_id"]
            if rid not in runs:
                runs[rid] = []
            runs[rid].append(r)

        lines = ["═══ Parameter History (deltas from defaults) ═══", ""]
        for rid in sorted(runs.keys(), reverse=True):
            entries = runs[rid]
            ts = entries[0]["ts"] if entries else "?"
            lines.append(f"── Run #{rid} ({ts}) ──")
            for e in entries:
                delta = e["delta"]
                params = ", ".join(f"{k}={json.dumps(v)}" for k, v in delta.items())
                lines.append(f"  {e['node_title']} ({e['node_class']})")
                lines.append(f"    {params}")
            lines.append("")

        return ("\n".join(lines),)

    def _diff_mode(self, run_a, run_b):
        if run_a == 0 or run_b == 0:
            try:
                conn = _get_db()
                rids = conn.execute(
                    "SELECT DISTINCT run_id FROM param_diffs ORDER BY run_id DESC LIMIT 2"
                ).fetchall()
                conn.close()
                if len(rids) < 2:
                    return ("Need at least 2 runs for diff. Run the workflow more.",)
                run_b = rids[0][0]
                run_a = rids[1][0]
            except Exception:
                return ("Could not read run history from DB.",)

        diffs = _diff_runs(run_a, run_b)
        if not diffs:
            return (f"No parameter differences between Run #{run_a} and Run #{run_b}.",)

        lines = [f"═══ Diff: Run #{run_a} → Run #{run_b} ═══", ""]
        for d in diffs:
            lines.append(
                f"  {d['node_title']} ({d['node_class']}) → {d['param_name']}: "
                f"{d['run_a']} → {d['run_b']}"
            )
        return ("\n".join(lines),)

    def _filter_mode(self, node_class, last_n_runs):
        if not node_class:
            return ("Provide a node_class_filter value to filter history.",)

        rows = _query_history(node_class=node_class, last_n_runs=last_n_runs)
        if not rows:
            return (f"No history found for node class '{node_class}'.",)

        lines = [f"═══ History for {node_class} (deltas) ═══", ""]
        for r in rows:
            delta = r["delta"]
            params = ", ".join(f"{k}={json.dumps(v)}" for k, v in delta.items())
            lines.append(f"  Run #{r['run_id']} ({r['ts']})")
            lines.append(f"    {params}")
        return ("\n".join(lines),)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Always re-execute to get fresh data
        return float("NaN")


# ══════════════════════════════════════════════════════════════════════
#  Server route registration (called from __init__.py)
# ══════════════════════════════════════════════════════════════════════

def register_routes(server):
    """Register /mec/param_history POST endpoint on the ComfyUI server."""
    from aiohttp import web

    @server.routes.post("/mec/param_history")
    async def _handle_param_history(request):
        try:
            data = await request.json()
            _store_snapshot(data)
            return web.json_response({"status": "ok"})
        except Exception as e:
            logger.warning(f"[MEC] /mec/param_history error: {e}")
            return web.json_response({"status": "error", "msg": str(e)}, status=400)
