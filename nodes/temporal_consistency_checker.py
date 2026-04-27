"""
TemporalConsistencyCheckerMEC – Score frame-to-frame stability of a mask
or image sequence.

Metrics:
  - ``mask_iou``: Per-pair IoU of binary masks. Detects flicker /
    label switches.
  - ``pixel_diff``: L1 mean diff in [0, 1] across consecutive frames.
    No external deps. Pure-torch fallback.
  - ``flow_warp``: Compute Farneback optical flow (cv2) between
    frames i and i+1, warp frame i forward, then measure L1 diff
    against frame i+1. Requires OpenCV; otherwise transparently
    degrades to ``pixel_diff``.

Outputs a JSON of per-pair scores plus aggregate min/max/mean and a
``flicker_score`` (1 - mean of normalized stability).

This is read-only / diagnostic — input is passed through unchanged.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import torch

logger = logging.getLogger("MEC.TemporalConsistencyChecker")

try:
    import cv2  # type: ignore[import-not-found]
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


def _to_uint8_gray(arr_hw3: np.ndarray) -> np.ndarray:
    rgb = (np.clip(arr_hw3, 0.0, 1.0) * 255.0).astype(np.uint8)
    if rgb.ndim == 3 and rgb.shape[-1] == 3:
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if _HAS_CV2 else rgb.mean(axis=-1).astype(np.uint8)
    return rgb


def _flow_warp_diff(prev_rgb: np.ndarray, curr_rgb: np.ndarray) -> float:
    """Forward-warp prev → curr via Farneback flow, return mean L1 diff."""
    prev_g = _to_uint8_gray(prev_rgb)
    curr_g = _to_uint8_gray(curr_rgb)
    flow = cv2.calcOpticalFlowFarneback(
        prev_g, curr_g, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
    )
    h, w = prev_g.shape
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x + flow[..., 0]).astype(np.float32)
    map_y = (grid_y + flow[..., 1]).astype(np.float32)
    warped = cv2.remap(prev_rgb.astype(np.float32), map_x, map_y,
                       interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return float(np.mean(np.abs(warped - curr_rgb.astype(np.float32))))


def _mask_iou(prev: torch.Tensor, curr: torch.Tensor, thr: float = 0.5) -> float:
    a = (prev > thr)
    b = (curr > thr)
    inter = (a & b).sum().item()
    union = (a | b).sum().item()
    if union == 0:
        return 1.0  # both empty → consider them perfectly consistent
    return float(inter) / float(union)


class TemporalConsistencyCheckerMEC:
    """Diagnose flicker / drift in mask or image sequences."""

    METRICS = ["mask_iou", "pixel_diff", "flow_warp"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "metric": (cls.METRICS, {
                    "default": "pixel_diff",
                    "tooltip": (
                        "mask_iou: requires MASK input; per-frame IoU.\n"
                        "pixel_diff: L1 diff between consecutive frames (image or mask).\n"
                        "flow_warp: Farneback flow + warp + L1 diff (requires OpenCV)."
                    ),
                }),
            },
            "optional": {
                "image": ("IMAGE", {
                    "tooltip": "Image batch (B,H,W,3) for pixel_diff / flow_warp.",
                }),
                "mask": ("MASK", {
                    "tooltip": "Mask batch (B,H,W) for mask_iou or pixel_diff.",
                }),
                "binarize_threshold": ("FLOAT", {
                    "default": 0.5, "min": 0.01, "max": 0.99, "step": 0.01,
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "FLOAT", "STRING")
    RETURN_NAMES = ("image_passthrough", "mask_passthrough", "flicker_score", "report_json")
    FUNCTION = "check"
    CATEGORY = "MaskEditControl/Diagnostics"
    DESCRIPTION = (
        "Score frame-to-frame stability of an IMAGE or MASK batch. "
        "Returns flicker_score in [0, 1] (lower = more stable) and a per-pair JSON report."
    )

    def check(
        self,
        metric: str,
        image: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        binarize_threshold: float = 0.5,
    ):
        if metric == "mask_iou" and mask is None:
            raise ValueError("mask_iou requires a MASK input.")
        if metric in ("pixel_diff", "flow_warp") and image is None and mask is None:
            raise ValueError(f"{metric} requires either an IMAGE or a MASK input.")

        # Decide which tensor we score over
        actual_metric = metric
        if metric == "flow_warp" and not _HAS_CV2:
            logger.warning("[MEC] flow_warp unavailable (no OpenCV); using pixel_diff.")
            actual_metric = "pixel_diff"

        if actual_metric == "mask_iou":
            seq = mask
        elif image is not None:
            seq = image
        else:
            seq = mask

        if seq is None or seq.shape[0] < 2:
            report = {
                "metric": actual_metric,
                "n_frames": 0 if seq is None else int(seq.shape[0]),
                "pairs": [],
                "flicker_score": 0.0,
                "note": "Sequence too short to score; need >=2 frames.",
            }
            return (image, mask, 0.0, json.dumps(report, indent=2))

        B = int(seq.shape[0])
        scores: list[float] = []
        seq_cpu = seq.detach().cpu()
        for i in range(B - 1):
            a = seq_cpu[i]
            b = seq_cpu[i + 1]
            if actual_metric == "mask_iou":
                s = _mask_iou(a, b, binarize_threshold)
                # Convert IoU (1 = stable) to instability in [0, 1]
                scores.append(1.0 - s)
            elif actual_metric == "pixel_diff":
                scores.append(float((a.float() - b.float()).abs().mean().item()))
            elif actual_metric == "flow_warp":
                a_np = a.numpy()
                b_np = b.numpy()
                if a_np.ndim == 2:
                    a_np = np.stack([a_np] * 3, axis=-1)
                    b_np = np.stack([b_np] * 3, axis=-1)
                # Already H,W,3 for IMAGE input
                scores.append(_flow_warp_diff(a_np, b_np) / 255.0)

        flicker = float(np.mean(scores)) if scores else 0.0
        report = {
            "metric": actual_metric,
            "requested_metric": metric,
            "n_frames": B,
            "n_pairs": len(scores),
            "min": float(np.min(scores)) if scores else 0.0,
            "max": float(np.max(scores)) if scores else 0.0,
            "mean": flicker,
            "std": float(np.std(scores)) if scores else 0.0,
            "pairs": [{"i": i, "score": float(s)} for i, s in enumerate(scores)],
            "flicker_score": flicker,
        }
        return (image, mask, flicker, json.dumps(report, indent=2))


NODE_CLASS_MAPPINGS = {"TemporalConsistencyCheckerMEC": TemporalConsistencyCheckerMEC}
NODE_DISPLAY_NAME_MAPPINGS = {
    "TemporalConsistencyCheckerMEC": "Temporal Consistency Checker (MEC)"
}
