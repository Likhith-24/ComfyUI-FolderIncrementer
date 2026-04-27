"""
Model-analysis nodes (MEC):
  - VAESimilarityAnalyserMEC: Tensor-by-tensor cosine similarity between
    two VAEs (or any two state-dict-bearing model objects).
  - VAEBlockInspectorMEC: Per-block weight statistics for a VAE.

These are diagnostic nodes; they never modify the input models.
Both work on whatever ``state_dict()`` the wrapped object exposes; if the
input is already a plain ``dict`` of tensors, they accept that too.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

import torch

logger = logging.getLogger("MEC.ModelAnalysis")


_BLOCK_RE = re.compile(
    r"^(encoder|decoder|quant_conv|post_quant_conv)"
    r"(?:\.(?:down_blocks|up_blocks|mid_block|conv_in|conv_out|norm_out))?",
)


def _to_state_dict(obj: Any) -> dict[str, torch.Tensor]:
    if obj is None:
        raise ValueError("VAE input is None.")
    if isinstance(obj, dict):
        return {k: v for k, v in obj.items() if isinstance(v, torch.Tensor)}
    sd_fn = getattr(obj, "state_dict", None)
    if callable(sd_fn):
        sd = sd_fn()
        return {k: v for k, v in sd.items() if isinstance(v, torch.Tensor)}
    inner = getattr(obj, "model", None) or getattr(obj, "first_stage_model", None)
    if inner is not None and hasattr(inner, "state_dict"):
        return {k: v for k, v in inner.state_dict().items() if isinstance(v, torch.Tensor)}
    raise ValueError(f"Cannot extract state_dict from {type(obj).__name__}")


def _block_of(key: str) -> str:
    m = _BLOCK_RE.match(key)
    if not m:
        return "other"
    parts = key.split(".")
    if len(parts) >= 2:
        return ".".join(parts[:2])
    return parts[0]


class VAESimilarityAnalyserMEC:
    """Cosine similarity between two VAE state dicts, per tensor & per block."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae_a": ("VAE",),
                "vae_b": ("VAE",),
            },
            "optional": {
                "include_per_tensor": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING", "FLOAT")
    RETURN_NAMES = ("report_json", "global_cosine")
    FUNCTION = "analyse"
    CATEGORY = "MaskEditControl/ModelAnalysis"
    DESCRIPTION = "Cosine similarity between two VAEs (per tensor + per block)."

    def analyse(self, vae_a, vae_b, include_per_tensor: bool = False):
        sd_a = _to_state_dict(vae_a)
        sd_b = _to_state_dict(vae_b)
        common = sorted(set(sd_a) & set(sd_b))
        only_a = sorted(set(sd_a) - set(sd_b))
        only_b = sorted(set(sd_b) - set(sd_a))
        per_tensor: list[dict] = []
        block_dot: dict[str, float] = {}
        block_norm_a: dict[str, float] = {}
        block_norm_b: dict[str, float] = {}
        global_dot = 0.0
        global_na = 0.0
        global_nb = 0.0
        for k in common:
            ta = sd_a[k].detach().to(torch.float32).flatten()
            tb = sd_b[k].detach().to(torch.float32).flatten()
            if ta.shape != tb.shape:
                continue
            dot = float((ta * tb).sum().item())
            na = float((ta * ta).sum().item())
            nb = float((tb * tb).sum().item())
            global_dot += dot
            global_na += na
            global_nb += nb
            blk = _block_of(k)
            block_dot[blk] = block_dot.get(blk, 0.0) + dot
            block_norm_a[blk] = block_norm_a.get(blk, 0.0) + na
            block_norm_b[blk] = block_norm_b.get(blk, 0.0) + nb
            if include_per_tensor:
                cos = dot / max((na ** 0.5) * (nb ** 0.5), 1e-12)
                per_tensor.append({"key": k, "cos": cos, "shape": list(ta.shape)})
        block_cos = {
            blk: block_dot[blk] / max((block_norm_a[blk] ** 0.5) * (block_norm_b[blk] ** 0.5), 1e-12)
            for blk in block_dot
        }
        global_cos = global_dot / max((global_na ** 0.5) * (global_nb ** 0.5), 1e-12)
        report = {
            "global_cosine": global_cos,
            "common_tensors": len(common),
            "only_in_a": only_a[:50],
            "only_in_b": only_b[:50],
            "per_block_cosine": block_cos,
        }
        if include_per_tensor:
            report["per_tensor"] = per_tensor
        return (json.dumps(report, indent=2), float(global_cos))


class VAEBlockInspectorMEC:
    """Per-block parameter statistics (mean / std / abs_mean / count)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"vae": ("VAE",)}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report_json",)
    FUNCTION = "inspect"
    CATEGORY = "MaskEditControl/ModelAnalysis"
    DESCRIPTION = "Per-block weight stats for a VAE (mean/std/abs_mean/count)."

    def inspect(self, vae):
        sd = _to_state_dict(vae)
        agg: dict[str, dict[str, float]] = {}
        for k, v in sd.items():
            blk = _block_of(k)
            t = v.detach().to(torch.float32).flatten()
            n = t.numel()
            if n == 0:
                continue
            stat = agg.setdefault(blk, {"sum": 0.0, "sumsq": 0.0, "abssum": 0.0, "count": 0})
            stat["sum"] += float(t.sum().item())
            stat["sumsq"] += float((t * t).sum().item())
            stat["abssum"] += float(t.abs().sum().item())
            stat["count"] += int(n)
        out = {}
        for blk, s in agg.items():
            n = max(s["count"], 1)
            mean = s["sum"] / n
            var = max(s["sumsq"] / n - mean * mean, 0.0)
            out[blk] = {
                "mean": mean,
                "std": var ** 0.5,
                "abs_mean": s["abssum"] / n,
                "count": s["count"],
            }
        return (json.dumps({"blocks": out, "total_tensors": len(sd)}, indent=2),)


NODE_CLASS_MAPPINGS = {
    "VAESimilarityAnalyserMEC": VAESimilarityAnalyserMEC,
    "VAEBlockInspectorMEC": VAEBlockInspectorMEC,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "VAESimilarityAnalyserMEC": "VAE Similarity Analyser (MEC)",
    "VAEBlockInspectorMEC": "VAE Block Inspector (MEC)",
}
