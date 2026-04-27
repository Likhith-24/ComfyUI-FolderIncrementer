"""
BatchVersionManagerMEC – Shot/task hierarchy with atomic version reservation.

Layout:
    <root>/<show>/<shot>/<task>/v<NNN>/

Behaviour:
  - Discovers the next free version by scanning existing ``v###`` dirs.
  - When ``reserve=True``, atomically reserves the version by creating
    a `.lock` file using ``open(..., 'x')``. If the lock already
    exists (another process won the race), retries up to ``max_retries``
    with the next version number.
  - Always returns paths with forward slashes (vfx pipelines, render
    farms, and OS-agnostic asset DBs all expect this).

Outputs ``next_version_path`` (string), ``version_int``, and ``info_json``.
No pixel I/O; safe to call inside a workflow.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger("MEC.BatchVersionManager")

_VERSION_DIR_RE = re.compile(r"^v(\d{1,6})$")
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_\-.]")


def _safe(token: str, fallback: str) -> str:
    """Sanitize a hierarchy token (show/shot/task)."""
    if not token or not token.strip():
        return fallback
    cleaned = _SAFE_NAME_RE.sub("_", token.strip())
    cleaned = cleaned.strip("._")
    return cleaned or fallback


def _scan_max_version(task_dir: Path) -> int:
    if not task_dir.is_dir():
        return 0
    max_v = 0
    for entry in task_dir.iterdir():
        if not entry.is_dir():
            continue
        m = _VERSION_DIR_RE.match(entry.name)
        if m:
            v = int(m.group(1))
            if v > max_v:
                max_v = v
    return max_v


class BatchVersionManagerMEC:
    """Allocate the next free ``v###`` directory under <root>/<show>/<shot>/<task>/."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "root": ("STRING", {
                    "default": "",
                    "tooltip": "Absolute output root (e.g. D:/projects/renders).",
                }),
                "show": ("STRING", {"default": "show"}),
                "shot": ("STRING", {"default": "sh010"}),
                "task": ("STRING", {"default": "comp"}),
            },
            "optional": {
                "reserve": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Atomically reserve the version with a .lock file. "
                        "When False, only computes the path — no disk writes."
                    ),
                }),
                "padding": ("INT", {
                    "default": 3, "min": 1, "max": 6,
                    "tooltip": "Zero-pad width for v### (3 → v001, 4 → v0001).",
                }),
                "max_retries": ("INT", {
                    "default": 5, "min": 1, "max": 50,
                    "tooltip": "On lock-race contention, advance version and retry this many times.",
                }),
                "min_version": ("INT", {
                    "default": 1, "min": 1, "max": 999999,
                    "tooltip": "Floor for the first version when no v### exists yet.",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "INT", "STRING", "STRING")
    RETURN_NAMES = ("version_path", "version_int", "version_label", "info_json")
    FUNCTION = "allocate"
    CATEGORY = "MaskEditControl/IO"
    DESCRIPTION = (
        "Compute (and optionally atomically reserve) the next v### directory under "
        "<root>/<show>/<shot>/<task>/. Forward-slash output paths."
    )

    def allocate(
        self,
        root: str,
        show: str,
        shot: str,
        task: str,
        reserve: bool = False,
        padding: int = 3,
        max_retries: int = 5,
        min_version: int = 1,
    ):
        if not root or not root.strip():
            raise ValueError("root path is required.")
        root_p = Path(root.strip()).expanduser()

        show_s = _safe(show, "show")
        shot_s = _safe(shot, "shot")
        task_s = _safe(task, "task")

        task_dir = root_p / show_s / shot_s / task_s
        next_v = max(_scan_max_version(task_dir) + 1, min_version)

        reserved = False
        attempts = 0
        target: Path = task_dir / f"v{next_v:0{padding}d}"
        if reserve:
            for attempts in range(max_retries):
                target = task_dir / f"v{next_v:0{padding}d}"
                try:
                    target.mkdir(parents=True, exist_ok=False)
                    lock = target / ".lock"
                    # 'x' = exclusive create; raises FileExistsError on race.
                    with open(lock, "x", encoding="utf-8") as fh:
                        fh.write(json.dumps({
                            "show": show_s, "shot": shot_s, "task": task_s,
                            "version": next_v,
                        }))
                    reserved = True
                    break
                except FileExistsError:
                    logger.info(
                        "[MEC] Version v%d already taken; advancing.", next_v,
                    )
                    next_v += 1
                except OSError as exc:
                    raise RuntimeError(
                        f"Failed to reserve version under {task_dir}: {exc}"
                    ) from exc
            else:
                raise RuntimeError(
                    f"Could not reserve a version after {max_retries} retries; "
                    f"last attempted v{next_v}."
                )

        label = f"v{next_v:0{padding}d}"
        info = {
            "root": str(root_p).replace("\\", "/"),
            "show": show_s,
            "shot": shot_s,
            "task": task_s,
            "version": next_v,
            "label": label,
            "reserved": reserved,
            "attempts": attempts + 1 if reserve else 0,
            "path": target.as_posix(),
        }
        logger.info(
            "[MEC] BatchVersionManager %s/%s/%s → %s (reserved=%s)",
            show_s, shot_s, task_s, label, reserved,
        )
        return (target.as_posix(), next_v, label, json.dumps(info, indent=2))


NODE_CLASS_MAPPINGS = {"BatchVersionManagerMEC": BatchVersionManagerMEC}
NODE_DISPLAY_NAME_MAPPINGS = {"BatchVersionManagerMEC": "Batch Version Manager (MEC)"}
