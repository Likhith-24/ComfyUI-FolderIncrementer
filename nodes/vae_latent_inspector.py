"""
VAELatentInspectorMEC – Inspect a LATENT tensor for sanity and stats.

Reports min/max/mean/std/abs-mean per channel, NaN/Inf counts, dynamic
range, and an actionable verdict (``healthy`` / ``low_contrast`` /
``saturated`` / ``corrupt``). Pure tensor; no model loads.

Use to diagnose VAE merge regressions, broken preprocessing, or to
spot blown-out / quantized latents before running an expensive
sampler pass.
"""
from __future__ import annotations

import json
import logging
import math

import torch

logger = logging.getLogger("MEC.VAELatentInspector")


def _channel_stats(t: torch.Tensor) -> list[dict]:
    """t: [B, C, H, W] float — return per-channel stats list."""
    out = []
    if t.dim() != 4:
        return out
    for c in range(t.shape[1]):
        ch = t[:, c].float()
        out.append({
            "channel": c,
            "min": float(ch.min().item()),
            "max": float(ch.max().item()),
            "mean": float(ch.mean().item()),
            "std": float(ch.std().item()) if ch.numel() > 1 else 0.0,
            "abs_mean": float(ch.abs().mean().item()),
        })
    return out


def _verdict(min_v: float, max_v: float, std: float, nan_count: int, inf_count: int) -> str:
    if nan_count > 0 or inf_count > 0:
        return "corrupt"
    rng = max_v - min_v
    if not math.isfinite(rng):
        return "corrupt"
    # Check saturated first: extreme magnitude beats low-contrast diagnosis,
    # because a constant tensor at value 100.0 has std=0 but is clearly
    # blown out, not "low contrast".
    if max_v > 50.0 or min_v < -50.0:
        return "saturated"
    if rng < 0.05:
        return "low_contrast"
    if std < 0.01:
        return "low_contrast"
    return "healthy"


class VAELatentInspectorMEC:
    """Inspect a LATENT for NaNs, Infs, range, and per-channel stats."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "ComfyUI LATENT dict (must contain 'samples')."}),
            },
            "optional": {
                "fail_on_corrupt": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "If True, raise ValueError when NaN/Inf detected.",
                }),
            },
        }

    RETURN_TYPES = ("LATENT", "STRING", "STRING", "INT", "INT")
    RETURN_NAMES = ("latent_passthrough", "info_json", "verdict", "nan_count", "inf_count")
    FUNCTION = "inspect"
    CATEGORY = "MaskEditControl/Diagnostics"
    DESCRIPTION = (
        "Inspect a LATENT tensor: per-channel min/max/mean/std, NaN & Inf counts, "
        "and a one-word verdict (healthy/low_contrast/saturated/corrupt). "
        "Latent is passed through unchanged."
    )

    def inspect(self, latent, fail_on_corrupt: bool = False):
        if not isinstance(latent, dict) or "samples" not in latent:
            raise ValueError("LATENT input must be a dict with key 'samples'.")
        t = latent["samples"]
        if not torch.is_tensor(t):
            raise ValueError("latent['samples'] must be a torch.Tensor.")

        nan_count = int(torch.isnan(t).sum().item())
        inf_count = int(torch.isinf(t).sum().item())
        finite = t[torch.isfinite(t)] if (nan_count + inf_count) > 0 else t
        if finite.numel() == 0:
            min_v = max_v = mean_v = std_v = float("nan")
        else:
            min_v = float(finite.min().item())
            max_v = float(finite.max().item())
            mean_v = float(finite.mean().item())
            std_v = float(finite.std().item()) if finite.numel() > 1 else 0.0

        info = {
            "shape": list(t.shape),
            "dtype": str(t.dtype),
            "device": str(t.device),
            "numel": int(t.numel()),
            "nan_count": nan_count,
            "inf_count": inf_count,
            "min": min_v,
            "max": max_v,
            "mean": mean_v,
            "std": std_v,
            "dynamic_range": (max_v - min_v) if math.isfinite(max_v - min_v) else None,
            "channels": _channel_stats(t),
        }
        verdict = _verdict(min_v, max_v, std_v, nan_count, inf_count)
        info["verdict"] = verdict

        if fail_on_corrupt and verdict == "corrupt":
            raise ValueError(
                f"LATENT contains NaN={nan_count}, Inf={inf_count}; refusing to pass through."
            )

        logger.info(
            "[MEC] LatentInspector: shape=%s verdict=%s NaN=%d Inf=%d range=[%.4g, %.4g]",
            list(t.shape), verdict, nan_count, inf_count, min_v, max_v,
        )
        return (latent, json.dumps(info, indent=2), verdict, nan_count, inf_count)


NODE_CLASS_MAPPINGS = {"VAELatentInspectorMEC": VAELatentInspectorMEC}
NODE_DISPLAY_NAME_MAPPINGS = {"VAELatentInspectorMEC": "VAE Latent Inspector (MEC)"}
