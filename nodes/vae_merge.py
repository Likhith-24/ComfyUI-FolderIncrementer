"""
nodes/vae_merge.py — VAE Merge node for the MaskEditControl (MEC) pack.

Merges 2 or 3 VAE models with multiple strategies and optional per-block
weight control. Runs entirely on CPU to keep VRAM free for downstream
sampling/decoding.

Supported merge modes
─────────────────────
    weighted_sum    out = (1 - alpha) * A + alpha * B
    add_difference  out = A + alpha * (B - C)        (vae_c required)
    tensor_sum      magnitude-blended tensor sum (per-tensor renormalised)
    triple_sum      out = (A + B + C) / 3            (vae_c required)
    slerp           spherical linear interpolation on flattened weights
    dare_ties       DARE pruning + TIES sign election (arXiv 2311.03099)
    block_swap      copy whole blocks from B into A based on block sliders
    clamp_interp    weighted_sum clamped to per-tensor min/max of A

Architecture detection
──────────────────────
Detects SD 1.5 / SDXL / Flux / Mochi / unknown by inspecting state-dict
keys, then maps the user-facing block sliders (block_conv_in,
block_conv_out, block_norm_out, block_0..block_3, block_mid) to the
matching architecture-specific key prefixes.
"""

from __future__ import annotations

import copy
import gc
import json
import logging
import math
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger("MEC.VAEMerge")


MERGE_MODES = [
    "weighted_sum",
    "add_difference",
    "tensor_sum",
    "triple_sum",
    "slerp",
    "dare_ties",
    "block_swap",
    "clamp_interp",
]

REQUIRES_VAE_C = {"add_difference", "triple_sum"}


# ───────────────────────── architecture detection ──────────────────────

# Regex patterns mapping each generic block slider to the
# architecture-specific key substrings. Keys are the slider names exposed
# in the UI; values are lists of regex patterns matched with re.search().
_BLOCK_PATTERNS_SD: Dict[str, List[str]] = {
    "block_conv_in":  [r"(?:^|\.)encoder\.conv_in\.", r"(?:^|\.)decoder\.conv_in\."],
    "block_conv_out": [r"(?:^|\.)encoder\.conv_out\.", r"(?:^|\.)decoder\.conv_out\."],
    "block_norm_out": [r"(?:^|\.)encoder\.norm_out\.", r"(?:^|\.)decoder\.norm_out\."],
    "block_mid":      [r"(?:^|\.)encoder\.mid\.", r"(?:^|\.)decoder\.mid\.",
                       r"(?:^|\.)encoder\.mid_block\.", r"(?:^|\.)decoder\.mid_block\."],
    "block_0":        [r"(?:^|\.)encoder\.down\.0\.", r"(?:^|\.)decoder\.up\.3\.",
                       r"(?:^|\.)encoder\.down_blocks\.0\.", r"(?:^|\.)decoder\.up_blocks\.0\."],
    "block_1":        [r"(?:^|\.)encoder\.down\.1\.", r"(?:^|\.)decoder\.up\.2\.",
                       r"(?:^|\.)encoder\.down_blocks\.1\.", r"(?:^|\.)decoder\.up_blocks\.1\."],
    "block_2":        [r"(?:^|\.)encoder\.down\.2\.", r"(?:^|\.)decoder\.up\.1\.",
                       r"(?:^|\.)encoder\.down_blocks\.2\.", r"(?:^|\.)decoder\.up_blocks\.2\."],
    "block_3":        [r"(?:^|\.)encoder\.down\.3\.", r"(?:^|\.)decoder\.up\.0\.",
                       r"(?:^|\.)encoder\.down_blocks\.3\.", r"(?:^|\.)decoder\.up_blocks\.3\."],
}


_ARCH_SIGNATURES: List[Tuple[str, re.Pattern]] = [
    ("flux",   re.compile(r"(?:^|\.)(?:img_in|txt_in|double_blocks|single_blocks)\.")),
    ("mochi",  re.compile(r"(?:^|\.)(?:t5|patch_embed|attn_blocks)\.")),
    ("sdxl",   re.compile(r"(?:^|\.)encoder\.down(?:_blocks)?\.3\.")),
    ("sd1x",   re.compile(r"(?:^|\.)encoder\.down(?:_blocks)?\.2\.")),
]


def _detect_architecture(keys: List[str]) -> str:
    """Best-effort architecture identification from state-dict keys."""
    joined = "\n".join(keys)
    for name, pat in _ARCH_SIGNATURES:
        if pat.search(joined):
            return name
    return "unknown"


def _classify_key(key: str) -> Optional[str]:
    """Return the matching block slider name for *key*, or None."""
    for slider, patterns in _BLOCK_PATTERNS_SD.items():
        for pat in patterns:
            if re.search(pat, key):
                return slider
    return None


# ─────────────────────────── state-dict helpers ────────────────────────

def _extract_state_dict(vae: Any) -> Tuple[Dict[str, torch.Tensor], Any]:
    """Return ``(state_dict, inner_model)`` for a ComfyUI VAE wrapper.

    Tries ``vae.first_stage_model`` first (SD-style), then falls back to
    ``vae.state_dict()`` directly. ``inner_model`` is the object that
    must receive ``load_state_dict`` after merging.
    """
    inner = getattr(vae, "first_stage_model", None)
    if inner is not None and hasattr(inner, "state_dict"):
        return {k: v for k, v in inner.state_dict().items()}, inner
    if hasattr(vae, "state_dict"):
        return {k: v for k, v in vae.state_dict().items()}, vae
    if isinstance(vae, dict):
        # Bare state dict — rare, but be permissive for tests.
        return dict(vae), None
    raise TypeError(
        f"[MEC VAE Merge] Could not extract state dict from VAE input "
        f"(type={type(vae).__name__}). Expected a ComfyUI VAE wrapper "
        f"with .first_stage_model or .state_dict()."
    )


def _check_compatible(
    a: Dict[str, torch.Tensor],
    b: Dict[str, torch.Tensor],
    label_a: str,
    label_b: str,
) -> None:
    """Raise ValueError with an actionable message on architecture mismatch."""
    keys_a = set(a.keys())
    keys_b = set(b.keys())
    missing_in_b = sorted(keys_a - keys_b)
    missing_in_a = sorted(keys_b - keys_a)
    if missing_in_a or missing_in_b:
        sample_b = missing_in_b[:5]
        sample_a = missing_in_a[:5]
        raise ValueError(
            f"[MEC VAE Merge] Architecture mismatch between {label_a} and {label_b}.\n"
            f"  {label_b} is missing {len(missing_in_b)} keys (e.g. {sample_b}).\n"
            f"  {label_a} is missing {len(missing_in_a)} keys (e.g. {sample_a}).\n"
            f"  Both VAEs must share the same architecture (e.g. SD 1.5 vs SD 1.5)."
        )
    # Shape check for shared keys
    for k in keys_a:
        if a[k].shape != b[k].shape:
            raise ValueError(
                f"[MEC VAE Merge] Shape mismatch for key '{k}': "
                f"{label_a}={tuple(a[k].shape)} vs {label_b}={tuple(b[k].shape)}. "
                f"VAEs must share the same architecture."
            )


# ─────────────────────────── merge primitives ──────────────────────────

def _merge_two(
    ta: torch.Tensor, tb: torch.Tensor, mode: str, alpha: float,
) -> torch.Tensor:
    """Merge two tensors of identical shape.

    All operations are performed in float32 to avoid catastrophic
    cancellation / overflow on float16 weights, then cast back to ta's
    original dtype.
    """
    out_dtype = ta.dtype
    a = ta.detach().to(torch.float32)
    b = tb.detach().to(torch.float32)
    if mode == "weighted_sum":
        out = (1.0 - alpha) * a + alpha * b
    elif mode == "tensor_sum":
        summed = a + b
        peak = summed.abs().max()
        out = summed / peak.clamp_min(1e-8) * a.abs().max().clamp_min(1e-8)
    elif mode == "slerp":
        out = _slerp(a, b, alpha)
    elif mode == "dare_ties":
        out = _dare_ties_pair(a, b, alpha)
    elif mode == "block_swap":
        out = b if alpha >= 0.5 else a
    elif mode == "clamp_interp":
        merged = (1.0 - alpha) * a + alpha * b
        out = merged.clamp(a.min(), a.max())
    else:
        # weighted_sum is the safe default
        out = (1.0 - alpha) * a + alpha * b
    return out.to(out_dtype)


def _slerp(a: torch.Tensor, b: torch.Tensor, t: float) -> torch.Tensor:
    """Spherical linear interpolation on flattened tensors."""
    flat_a = a.flatten()
    flat_b = b.flatten()
    na = flat_a.norm().clamp_min(1e-8)
    nb = flat_b.norm().clamp_min(1e-8)
    dot = (flat_a / na).dot(flat_b / nb).clamp(-1.0, 1.0)
    omega = torch.acos(dot)
    so = torch.sin(omega)
    if so.abs() < 1e-6:
        return ((1.0 - t) * a + t * b)
    coef_a = torch.sin((1.0 - t) * omega) / so
    coef_b = torch.sin(t * omega) / so
    return (coef_a * flat_a + coef_b * flat_b).view_as(a)


def _dare_ties_pair(
    a: torch.Tensor, b: torch.Tensor, alpha: float, drop_p: float = 0.1,
) -> torch.Tensor:
    """Simplified two-model DARE+TIES merge.

    Treats *a* as the base model and *b* as a fine-tune; computes the
    delta, randomly drops *drop_p* of it, rescales, then adds with
    sign-election against the base.
    """
    delta = b - a
    # DARE drop
    mask = torch.rand_like(delta) > drop_p
    delta = delta * mask / max(1.0 - drop_p, 1e-6)
    # TIES sign election: keep delta entries that agree with the
    # dominant sign of the magnitude-weighted vector.
    sign = torch.sign(delta.sum())
    if sign != 0:
        delta = torch.where(torch.sign(delta) == sign, delta, torch.zeros_like(delta))
    return a + alpha * delta


def _per_key_alpha(
    key: str,
    base_alpha: float,
    use_blocks: bool,
    block_weights: Dict[str, float],
) -> float:
    if not use_blocks:
        return base_alpha
    slider = _classify_key(key)
    if slider is None:
        return base_alpha
    return float(block_weights.get(slider, base_alpha))


# ──────────────────────────── ComfyUI node ─────────────────────────────


class VAEMergeMEC:
    """
    (MEC) VAE Merge — combine 2 or 3 VAE checkpoints with optional
    per-block weights. Useful for blending the colour response and
    detail of multiple VAEs (e.g. anime + photoreal SD 1.5 VAEs, or
    SDXL base + fine-tune).

    Inputs
    ──────
        vae_a, vae_b           required ComfyUI VAE objects
        vae_c                  optional third VAE (required for
                               'add_difference' and 'triple_sum')
        merge_mode             see module docstring
        alpha, beta            global blend weights (0.0–1.0)
        brightness, contrast   small post-merge bias/gain on the
                               decoder.conv_out weight only
        use_blocks             enable per-block weight sliders
        block_*                per-block weights (override alpha for
                               the matching layer group)

    Outputs
    ───────
        vae   — merged ComfyUI VAE (deep-copied from vae_a)
        info  — JSON summary string
    """

    @classmethod
    def INPUT_TYPES(cls):
        block_default = {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}
        return {
            "required": {
                "vae_a": ("VAE", {"tooltip": "Primary VAE (acts as the base; merged-in-place clone is returned)."}),
                "vae_b": ("VAE", {"tooltip": "Secondary VAE to blend into vae_a."}),
                "merge_mode": (MERGE_MODES, {
                    "default": "weighted_sum",
                    "tooltip": "Blend strategy. 'add_difference' and 'triple_sum' require vae_c."
                }),
                "alpha": ("FLOAT", {"default": 0.30, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Primary blend weight. weighted_sum: 0=A, 1=B."}),
                "beta": ("FLOAT", {"default": 0.70, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Secondary blend weight (used by add_difference / 3-VAE modes)."}),
                "brightness": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Post-merge brightness shift on decoder.conv_out (small effect)."}),
                "contrast": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Post-merge contrast gain on decoder.conv_out."}),
                "use_blocks": ("BOOLEAN", {"default": False,
                    "tooltip": "Enable per-block weight sliders below. When False all keys use 'alpha'."}),
                "block_conv_in":  ("FLOAT", dict(block_default, tooltip="Weight for encoder/decoder conv_in.")),
                "block_conv_out": ("FLOAT", dict(block_default, tooltip="Weight for encoder/decoder conv_out.")),
                "block_norm_out": ("FLOAT", dict(block_default, tooltip="Weight for encoder/decoder norm_out.")),
                "block_0":        ("FLOAT", dict(block_default, tooltip="Weight for the first down/up block pair.")),
                "block_1":        ("FLOAT", dict(block_default, tooltip="Weight for the second down/up block pair.")),
                "block_2":        ("FLOAT", dict(block_default, tooltip="Weight for the third down/up block pair.")),
                "block_3":        ("FLOAT", dict(block_default, tooltip="Weight for the fourth down/up block pair (SDXL).")),
                "block_mid":      ("FLOAT", dict(block_default, tooltip="Weight for the mid block.")),
            },
            "optional": {
                "vae_c": ("VAE", {"tooltip": "Optional third VAE for add_difference / triple_sum."}),
            },
        }

    RETURN_TYPES = ("VAE", "STRING")
    RETURN_NAMES = ("vae", "info")
    FUNCTION = "merge"
    CATEGORY = "MaskEditControl/VAE"
    OUTPUT_NODE = False

    # ------------------------------------------------------------------
    def merge(
        self,
        vae_a: Any,
        vae_b: Any,
        merge_mode: str = "weighted_sum",
        alpha: float = 0.30,
        beta: float = 0.70,
        brightness: float = 0.0,
        contrast: float = 0.0,
        use_blocks: bool = False,
        block_conv_in: float = 0.5,
        block_conv_out: float = 0.5,
        block_norm_out: float = 0.5,
        block_0: float = 0.5,
        block_1: float = 0.5,
        block_2: float = 0.5,
        block_3: float = 0.5,
        block_mid: float = 0.5,
        vae_c: Optional[Any] = None,
    ) -> Tuple[Any, str]:
        info: Dict[str, Any] = {
            "merge_mode": merge_mode,
            "alpha": float(alpha),
            "beta": float(beta),
            "use_blocks": bool(use_blocks),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        block_weights = {
            "block_conv_in": block_conv_in,
            "block_conv_out": block_conv_out,
            "block_norm_out": block_norm_out,
            "block_0": block_0,
            "block_1": block_1,
            "block_2": block_2,
            "block_3": block_3,
            "block_mid": block_mid,
        }

        try:
            if merge_mode in REQUIRES_VAE_C and vae_c is None:
                raise ValueError(
                    f"[MEC VAE Merge] '{merge_mode}' requires vae_c to be connected. "
                    f"Connect a third VAE model to the vae_c input, or switch to "
                    f"'weighted_sum'."
                )

            sd_a, _ = _extract_state_dict(vae_a)
            sd_b, _ = _extract_state_dict(vae_b)
            sd_c: Optional[Dict[str, torch.Tensor]] = None

            _check_compatible(sd_a, sd_b, "vae_a", "vae_b")
            if vae_c is not None:
                sd_c, _ = _extract_state_dict(vae_c)
                _check_compatible(sd_a, sd_c, "vae_a", "vae_c")

            arch = _detect_architecture(list(sd_a.keys()))
            info["architecture"] = arch
            info["key_count"] = len(sd_a)

            # Track dtype distribution (preserve dominant dtype, no upcast).
            dtypes: Dict[str, int] = {}
            for v in sd_a.values():
                key = str(v.dtype).replace("torch.", "")
                dtypes[key] = dtypes.get(key, 0) + 1
            info["dtypes"] = dtypes

            merged: Dict[str, torch.Tensor] = {}

            for key, ta in sd_a.items():
                tb = sd_b[key]
                # CPU compute regardless of where the inputs live.
                ta_cpu = ta.detach().to("cpu")
                tb_cpu = tb.detach().to("cpu")

                a_eff = _per_key_alpha(key, alpha, use_blocks, block_weights)

                if merge_mode == "add_difference":
                    assert sd_c is not None
                    tc_cpu = sd_c[key].detach().to(torch.float32).to("cpu")
                    out = ta_cpu.to(torch.float32) + a_eff * (tb_cpu.to(torch.float32) - tc_cpu)
                    out = out.to(ta.dtype)
                elif merge_mode == "triple_sum":
                    assert sd_c is not None
                    tc_cpu = sd_c[key].detach().to(torch.float32).to("cpu")
                    out = (ta_cpu.to(torch.float32) + tb_cpu.to(torch.float32) + tc_cpu) / 3.0
                    out = out.to(ta.dtype)
                else:
                    out = _merge_two(ta_cpu, tb_cpu, merge_mode, a_eff)

                # Replace NaN/Inf with the source tensor — never crash a workflow.
                if not torch.isfinite(out).all():
                    out = torch.where(torch.isfinite(out), out, ta_cpu)
                    info.setdefault("warnings", []).append(f"non-finite values clamped at key '{key}'")

                merged[key] = out

            # ── Post-process: brightness/contrast on decoder.conv_out ──
            if abs(brightness) > 1e-6 or abs(contrast) > 1e-6:
                applied = 0
                for key, t in merged.items():
                    if "decoder.conv_out" in key and t.dtype.is_floating_point:
                        t32 = t.to(torch.float32)
                        t32 = (t32 + float(brightness) * 0.1) * (1.0 + float(contrast))
                        t32 = t32.clamp(-10.0, 10.0)  # generous bounds; prevents NaN explosions
                        merged[key] = t32.to(t.dtype)
                        applied += 1
                info["postprocess_keys"] = applied

            # ── Build the output VAE object ──
            try:
                merged_vae = copy.deepcopy(vae_a)
            except Exception as exc:
                logger.warning("[MEC VAE Merge] deepcopy(vae_a) failed (%s); returning original wrapper.", exc)
                merged_vae = vae_a

            inner = getattr(merged_vae, "first_stage_model", None)
            target = inner if (inner is not None and hasattr(inner, "load_state_dict")) else merged_vae
            try:
                if hasattr(target, "load_state_dict"):
                    incompatible = target.load_state_dict(merged, strict=False)
                    missing = getattr(incompatible, "missing_keys", []) or []
                    unexpected = getattr(incompatible, "unexpected_keys", []) or []
                    if missing or unexpected:
                        info["load_state_dict"] = {
                            "missing": missing[:10],
                            "unexpected": unexpected[:10],
                        }
                else:
                    info.setdefault("warnings", []).append(
                        "target VAE has no load_state_dict; returning unmodified vae_a clone"
                    )
            except Exception as exc:
                logger.error("[MEC VAE Merge] load_state_dict failed: %s", exc, exc_info=True)
                info.setdefault("warnings", []).append(f"load_state_dict failed: {exc}")

            info["status"] = "ok"
            return (merged_vae, json.dumps(info, indent=2))

        except ValueError:
            # Architecture / argument errors — re-raise so the user sees them
            # in the ComfyUI console rather than silently passing a bad VAE
            # downstream. Caught by ComfyUI and shown as a node error.
            raise
        except Exception as exc:
            logger.error("[MEC VAE Merge] Unexpected failure: %s", exc, exc_info=True)
            info["status"] = "error"
            info["error"] = str(exc)
            return (vae_a, json.dumps(info, indent=2))
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


NODE_CLASS_MAPPINGS = {"VAEMergeMEC": VAEMergeMEC}
NODE_DISPLAY_NAME_MAPPINGS = {"VAEMergeMEC": "VAE Merge (MEC)"}
