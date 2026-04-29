"""Pre-install dependency conflict checker for ComfyUI custom nodes.

Why this exists
---------------
ComfyUI's custom-node ecosystem shares a single Python environment.
When pack A pins ``torch==2.1.0`` and pack B installs ``torch>=2.4``,
pip silently upgrades torch and pack A breaks at runtime — the
infamous "ComfyUI dependency hell".

This module classifies a candidate ``requirements.txt`` against the
*currently installed* environment **before** anything is touched:

    breaking : will definitely change the version of a CRITICAL package
               (torch, numpy, transformers, …)
    risky    : version range is unpinned / unbounded / very wide
    safe     : current install already satisfies the requirement

It also exposes ``simulate_install`` which calls
``pip install --dry-run --report -`` (the *supported public API*; the
``pip._internal`` resolver is explicitly unstable and is NOT used).

Public API
----------
* :data:`CRITICAL_PACKAGES`
* :func:`check_conflicts`
* :func:`format_warning_message`
* :func:`simulate_install`
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, distributions
from importlib.metadata import version as _installed_version
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# ── packages whose version change is almost always destabilising ────
CRITICAL_PACKAGES: frozenset = frozenset({
    "torch",
    "torchvision",
    "torchaudio",
    "numpy",
    "transformers",
    "diffusers",
    "xformers",
    "accelerate",
    "timm",
    "safetensors",
})

_REQ_LINE_RE = re.compile(
    r"""^\s*
        (?P<name>[A-Za-z0-9][A-Za-z0-9._\-]*)
        (?P<extras>\[[^\]]+\])?
        (?P<spec>[^;#]*)?
        (?:;.*)?$
    """,
    re.VERBOSE,
)

_OP_RE = re.compile(r"(>=|<=|==|!=|>|<|~=|===)\s*([0-9][0-9A-Za-z.\-+]*)")


# ── version helpers (no external deps) ──────────────────────────────
def _ver_tuple(v: str) -> Tuple[int, ...]:
    """Best-effort numeric tuple. Stops at the first non-numeric segment."""
    out: List[int] = []
    for chunk in v.split("."):
        m = re.match(r"\d+", chunk)
        if not m:
            break
        out.append(int(m.group(0)))
    return tuple(out) or (0,)


def _cmp(a: Tuple[int, ...], b: Tuple[int, ...]) -> int:
    n = max(len(a), len(b))
    a_ = a + (0,) * (n - len(a))
    b_ = b + (0,) * (n - len(b))
    return (a_ > b_) - (a_ < b_)


def _parse_specifier(spec: str) -> List[Tuple[str, Tuple[int, ...], str]]:
    """``'>=1.0,<2.0'`` → [('>=',(1,0),'1.0'), ('<',(2,0),'2.0')]."""
    out: List[Tuple[str, Tuple[int, ...], str]] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        m = _OP_RE.match(part)
        if not m:
            continue
        op, raw = m.group(1), m.group(2)
        out.append((op, _ver_tuple(raw), raw))
    return out


def _satisfies(installed: Tuple[int, ...],
               spec: List[Tuple[str, Tuple[int, ...], str]]) -> bool:
    if not spec:
        return True
    for op, ver, _raw in spec:
        c = _cmp(installed, ver)
        ok = (
            (op == ">=" and c >= 0)
            or (op == "<=" and c <= 0)
            or (op == ">"  and c >  0)
            or (op == "<"  and c <  0)
            or (op in ("==", "===") and c == 0)
            or (op == "!=" and c != 0)
            or (op == "~=" and c >= 0
                and len(ver) >= 2
                and installed[: len(ver) - 1] == ver[: len(ver) - 1])
        )
        if not ok:
            return False
    return True


def _has_upper_bound(spec: List[Tuple[str, Tuple[int, ...], str]]) -> bool:
    return any(op in ("<", "<=", "==", "===", "~=") for op, _, _ in spec)


# ── parsing ─────────────────────────────────────────────────────────
def _parse_requirements_file(path: Path) -> List[Tuple[str, str]]:
    """Read a requirements.txt and return ``[(canonical_name, spec_str), …]``."""
    out: List[Tuple[str, str]] = []
    text = Path(path).read_text(encoding="utf-8")
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            # Skip blanks, comments, and pip flags (-r, --index-url, …)
            continue
        m = _REQ_LINE_RE.match(line)
        if not m:
            continue
        name = m.group("name").lower().replace("_", "-")
        spec = (m.group("spec") or "").strip()
        out.append((name, spec))
    return out


def _installed_index() -> Dict[str, str]:
    """Map canonical package name → installed version."""
    out: Dict[str, str] = {}
    for dist in distributions():
        name = (dist.metadata["Name"] or "").lower().replace("_", "-")
        if name:
            out[name] = dist.version
    return out


# ── data classes ────────────────────────────────────────────────────
@dataclass
class ConflictEntry:
    package: str
    required: str
    installed: Optional[str]
    reason: str

    def to_dict(self) -> dict:
        return {
            "package": self.package,
            "required": self.required or "(any)",
            "installed": self.installed,
            "reason": self.reason,
        }


@dataclass
class ConflictReport:
    breaking: List[ConflictEntry] = field(default_factory=list)
    risky:    List[ConflictEntry] = field(default_factory=list)
    safe:     List[ConflictEntry] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "breaking": [e.to_dict() for e in self.breaking],
            "risky":    [e.to_dict() for e in self.risky],
            "safe":     [e.to_dict() for e in self.safe],
        }

    @property
    def has_breaking(self) -> bool:
        return bool(self.breaking)


# ── core ────────────────────────────────────────────────────────────
def check_conflicts(
    new_node_requirements_path: str | Path,
    *,
    installed: Optional[Dict[str, str]] = None,
    critical: Iterable[str] = CRITICAL_PACKAGES,
) -> ConflictReport:
    """Classify every requirement in ``new_node_requirements_path``.

    ``installed`` is injected by tests; production callers leave it ``None``
    and the function probes the live interpreter via ``importlib.metadata``.
    """
    reqs = _parse_requirements_file(Path(new_node_requirements_path))
    inst = installed if installed is not None else _installed_index()
    crit = {c.lower() for c in critical}
    report = ConflictReport()

    for pkg, spec_str in reqs:
        spec = _parse_specifier(spec_str)
        installed_ver = inst.get(pkg)

        # Not installed yet → safe (pip will resolve it freely)
        if installed_ver is None:
            report.safe.append(ConflictEntry(
                pkg, spec_str, None,
                "package not currently installed; will be added cleanly",
            ))
            continue

        satisfies = _satisfies(_ver_tuple(installed_ver), spec)

        if not satisfies:
            severity = "breaking" if pkg in crit else "risky"
            entry = ConflictEntry(
                pkg, spec_str, installed_ver,
                f"installed {installed_ver} does not satisfy {spec_str or '(any)'}; "
                f"pip would change a "
                f"{'CRITICAL shared' if pkg in crit else 'shared'} package",
            )
            getattr(report, severity).append(entry)
            continue

        # satisfied — but unbounded specs on critical packages are still risky
        if pkg in crit and (not spec or not _has_upper_bound(spec)):
            report.risky.append(ConflictEntry(
                pkg, spec_str, installed_ver,
                "specifier has no upper bound on a CRITICAL package; "
                "pip MAY pull a breaking major during transitive resolution",
            ))
        else:
            report.safe.append(ConflictEntry(
                pkg, spec_str, installed_ver,
                f"installed {installed_ver} satisfies {spec_str or '(any)'}",
            ))

    return report


def format_warning_message(report: ConflictReport) -> str:
    """Human-readable summary suitable for the ComfyUI Manager UI."""
    lines: List[str] = []
    if report.breaking:
        lines.append("⛔ BREAKING — installing this node WILL change critical packages:")
        for e in report.breaking:
            lines.append(
                f"   • {e.package}: installed {e.installed} → required {e.required or '(any)'}"
            )
    if report.risky:
        lines.append("⚠️  RISKY — possible silent regressions:")
        for e in report.risky:
            lines.append(
                f"   • {e.package}: installed {e.installed}, required {e.required or '(any)'} — {e.reason}"
            )
    if report.safe:
        lines.append(f"✅ {len(report.safe)} requirement(s) already satisfied.")
    if not lines:
        lines.append("✅ No requirements to evaluate.")
    if report.breaking:
        lines.append("")
        lines.append("Recommendation: do NOT auto-install. Pin versions or use a venv.")
    elif report.risky:
        lines.append("")
        lines.append("Recommendation: review the risky entries before continuing.")
    return "\n".join(lines)


def simulate_install(
    requirements_path: str | Path,
    *,
    timeout: int = 120,
    extra_args: Optional[List[str]] = None,
) -> dict:
    """Dry-run pip resolution.

    Uses pip's *public* JSON report flag (``--dry-run --report -``), introduced
    in pip 22.2 and stable since. Does NOT touch ``pip._internal`` (unstable).

    Returns a dict::

        {
          "ok"        : bool,           # exit code 0?
          "stdout"    : str,
          "stderr"    : str,
          "report"    : dict | None,    # parsed JSON report if available
          "would_install": [{"name": str, "version": str}, ...]
        }
    """
    cmd = [
        sys.executable, "-m", "pip", "install",
        "--dry-run", "--quiet",
        "--report", "-",
        "-r", str(requirements_path),
    ]
    if extra_args:
        cmd.extend(extra_args)

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False,
        )
    except FileNotFoundError as exc:
        return {"ok": False, "stdout": "", "stderr": str(exc),
                "report": None, "would_install": []}
    except subprocess.TimeoutExpired as exc:
        return {"ok": False, "stdout": exc.stdout or "",
                "stderr": f"pip timed out after {timeout}s",
                "report": None, "would_install": []}

    report: Optional[dict] = None
    would: List[dict] = []
    if proc.stdout:
        try:
            report = json.loads(proc.stdout)
        except json.JSONDecodeError:
            report = None
    if isinstance(report, dict):
        for inst in report.get("install", []):
            meta = (inst.get("metadata") or {})
            name = meta.get("name") or inst.get("requested", "")
            ver  = meta.get("version", "")
            if name:
                would.append({"name": name.lower(), "version": ver})

    return {
        "ok": proc.returncode == 0,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "report": report,
        "would_install": would,
    }


__all__ = [
    "CRITICAL_PACKAGES",
    "ConflictEntry",
    "ConflictReport",
    "check_conflicts",
    "format_warning_message",
    "simulate_install",
]
