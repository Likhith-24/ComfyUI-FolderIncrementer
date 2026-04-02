"""
MotionMaskTrackerMEC — Per-frame motion detection mask generator.

Generates masks showing WHAT MOVED between consecutive frames using
four independently toggleable detection methods:
  1. Pixel difference (absolute per-pixel change)
  2. Optical flow magnitude (Farneback cv2 / phase correlation torch fallback)
  3. Background subtraction (static BG model from first N frames)
  4. Histogram difference (per-region color histogram change)

Combined result = union or intersection of enabled methods.
Frame 0 always outputs zeros (no previous frame to compare).

Use cases:
  - Find moving objects in static-camera footage
  - Generate masks for targeted inpainting of changed regions
  - QA flagging of high-motion frames
  - Feed into Conditional Mask Router to skip still frames

VRAM Tier: 1 (pure tensor ops, optional cv2 for flow)

Files CREATED: nodes/motion_mask_tracker.py
Files MODIFIED: __init__.py (import + mapping)
Files UNTOUCHED: All existing node files
"""

from __future__ import annotations

import math
import logging
from typing import Tuple

import torch
import torch.nn.functional as F
import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from .stabilization_utils import (
    compensate_camera_motion,
    compute_motion_magnitudes,
    motion_adaptive_temporal_smooth,
)

logger = logging.getLogger("MEC")


# ══════════════════════════════════════════════════════════════════════
#  Helper: morphological grow (dilation via max_pool)
# ══════════════════════════════════════════════════════════════════════

def _grow_mask(mask: torch.Tensor, pixels: float) -> torch.Tensor:
    """Morphological dilation of (B, H, W) mask by 'pixels' px."""
    if pixels <= 0:
        return mask
    k = int(pixels) * 2 + 1
    pad = int(pixels)
    m4 = mask.unsqueeze(1)  # (B, 1, H, W)
    dilated = F.max_pool2d(m4, kernel_size=k, stride=1, padding=pad)
    return dilated.squeeze(1).clamp(0.0, 1.0)


# ══════════════════════════════════════════════════════════════════════
#  Helper: remove small connected regions
# ══════════════════════════════════════════════════════════════════════

def _remove_small_regions(mask: torch.Tensor, min_area: int) -> torch.Tensor:
    """Remove connected components smaller than min_area in (B, H, W) mask.
    cv2 if available, else erosion/dilation approximation."""
    if min_area <= 0:
        return mask
    B, H, W = mask.shape
    device = mask.device

    if HAS_CV2:
        results = []
        for b in range(B):
            binary = (mask[b].cpu().numpy() > 0.5).astype(np.uint8)
            n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
            filtered = np.zeros_like(binary)
            for i in range(1, n_labels):
                if stats[i, cv2.CC_STAT_AREA] >= min_area:
                    filtered[labels == i] = 1
            results.append(torch.from_numpy(filtered.astype(np.float32)))
        return torch.stack(results, dim=0).to(device)
    else:
        # Torch fallback: erode then dilate to approximate small blob removal
        k_size = max(3, int(math.sqrt(min_area)))
        if k_size % 2 == 0:
            k_size += 1
        pad = k_size // 2
        m4 = mask.unsqueeze(1)
        eroded = -F.max_pool2d(-m4, kernel_size=k_size, stride=1, padding=pad)
        dilated = F.max_pool2d(eroded, kernel_size=k_size, stride=1, padding=pad)
        return dilated.squeeze(1).clamp(0.0, 1.0)


# ══════════════════════════════════════════════════════════════════════
#  Helper: Gaussian temporal smooth along batch dim
# ══════════════════════════════════════════════════════════════════════

def _temporal_smooth(mask: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """Smooth (B, H, W) mask along batch dim with Gaussian kernel."""
    B, H, W = mask.shape
    if B <= 1 or sigma <= 0:
        return mask
    device = mask.device
    radius = max(1, int(math.ceil(2.5 * sigma)))
    size = 2 * radius + 1
    x = torch.arange(size, device=device, dtype=torch.float32) - radius
    k1d = torch.exp(-0.5 * (x / sigma) ** 2)
    k1d = k1d / k1d.sum()

    # (B, H*W) → (H*W, 1, B) for conv1d
    flat = mask.reshape(B, H * W).permute(1, 0).unsqueeze(1)  # (H*W, 1, B)
    kernel = k1d.view(1, 1, -1)
    padded = F.pad(flat, (radius, radius), mode="replicate")
    smoothed = F.conv1d(padded, kernel)  # (H*W, 1, B)
    return smoothed.squeeze(1).permute(1, 0).reshape(B, H, W).clamp(0.0, 1.0)


# ══════════════════════════════════════════════════════════════════════
#  Detection method 1: Pixel difference
# ══════════════════════════════════════════════════════════════════════

def _pixel_diff_masks(images: torch.Tensor,
                      threshold: float) -> torch.Tensor:
    """Absolute pixel difference between consecutive frames.

    diff[n] = |images[n] - images[n-1]|.mean(dim=-1)  # mean across RGB
    mask[n] = (diff[n] > threshold).float()
    mask[0] = zeros (no prev frame)

    images: (B, H, W, C) float32
    Returns: (B, H, W) float32
    """
    B, H, W, C = images.shape
    if B <= 1:
        return torch.zeros(B, H, W, device=images.device, dtype=torch.float32)

    # Vectorized: compute all consecutive diffs at once
    diff = (images[1:] - images[:-1]).abs().mean(dim=-1)  # (B-1, H, W)
    mask = (diff > threshold).float()
    # Prepend zeros for frame 0
    zero_frame = torch.zeros(1, H, W, device=images.device, dtype=torch.float32)
    return torch.cat([zero_frame, mask], dim=0)


# ══════════════════════════════════════════════════════════════════════
#  Detection method 2: Optical flow
# ══════════════════════════════════════════════════════════════════════

def _optical_flow_masks_cv2(images: torch.Tensor,
                            threshold: float) -> torch.Tensor:
    """Dense optical flow via cv2.calcOpticalFlowFarneback.

    For each consecutive pair:
      flow = Farneback(prev_gray, curr_gray)
      magnitude = sqrt(flow_x^2 + flow_y^2)
      mask = (magnitude > threshold)

    images: (B, H, W, C) float32
    Returns: (B, H, W) float32
    """
    B, H, W, C = images.shape
    device = images.device

    if B <= 1:
        return torch.zeros(B, H, W, device=device, dtype=torch.float32)

    results = [np.zeros((H, W), dtype=np.float32)]  # frame 0 = zeros

    for i in range(1, B):
        prev = (images[i - 1].cpu().numpy() * 255).astype(np.uint8)
        curr = (images[i].cpu().numpy() * 255).astype(np.uint8)

        prev_gray = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY)
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_RGB2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2,
            flags=0
        )
        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        mask = (magnitude > threshold).astype(np.float32)
        results.append(mask)

    return torch.from_numpy(np.stack(results, axis=0)).to(device)


def _optical_flow_masks_torch(images: torch.Tensor,
                              threshold: float) -> torch.Tensor:
    """Torch fallback: Laplacian structural difference as motion proxy.

    Where edges shift between frames = motion detected.
    lap_diff[n] = |laplacian(gray[n]) - laplacian(gray[n-1])|
    mask[n] = (lap_diff[n] > threshold * 5).float()

    images: (B, H, W, C) float32
    Returns: (B, H, W) float32
    """
    B, H, W, C = images.shape
    device = images.device

    if B <= 1:
        return torch.zeros(B, H, W, device=device, dtype=torch.float32)

    # Convert to grayscale
    gray = (0.2126 * images[..., 0] + 0.7152 * images[..., 1] +
            0.0722 * images[..., 2])  # (B, H, W)

    # Laplacian kernel
    lap_kernel = torch.tensor(
        [[0.0, 1.0, 0.0],
         [1.0, -4.0, 1.0],
         [0.0, 1.0, 0.0]], dtype=torch.float32, device=device
    ).view(1, 1, 3, 3)

    gray_4d = gray.unsqueeze(1)  # (B, 1, H, W)
    lap = F.conv2d(F.pad(gray_4d, (1, 1, 1, 1), mode="replicate"), lap_kernel)
    lap = lap.squeeze(1)  # (B, H, W)

    # Consecutive Laplacian difference
    lap_diff = (lap[1:] - lap[:-1]).abs()  # (B-1, H, W)

    # Threshold (multiply by 5 since Laplacian values differ from flow magnitudes)
    adjusted_threshold = threshold * 5.0
    mask = (lap_diff > adjusted_threshold).float()

    zero_frame = torch.zeros(1, H, W, device=device, dtype=torch.float32)
    return torch.cat([zero_frame, mask], dim=0)


# ══════════════════════════════════════════════════════════════════════
#  Detection method 3: Background subtraction
# ══════════════════════════════════════════════════════════════════════

def _background_sub_masks(images: torch.Tensor,
                          bg_frames: int,
                          threshold: float) -> torch.Tensor:
    """Build background model from first bg_frames, detect foreground.

    bg_model = median of first bg_frames (robust to outliers).
    For each frame: diff = |frame - bg_model|.mean(dim=-1)
    mask = (diff > threshold).float()

    images: (B, H, W, C) float32
    Returns: (B, H, W) float32
    """
    B, H, W, C = images.shape
    device = images.device

    n_bg = min(bg_frames, B)
    if n_bg <= 0:
        return torch.zeros(B, H, W, device=device, dtype=torch.float32)

    # Build background model: median is robust to moving objects
    if n_bg >= 3:
        bg_model = images[:n_bg].median(dim=0).values  # (H, W, C)
    else:
        bg_model = images[:n_bg].mean(dim=0)  # (H, W, C)

    # Vectorized: diff all frames against bg_model at once
    diff = (images - bg_model.unsqueeze(0)).abs().mean(dim=-1)  # (B, H, W)
    return (diff > threshold).float()


# ══════════════════════════════════════════════════════════════════════
#  Detection method 4: Histogram difference
# ══════════════════════════════════════════════════════════════════════

def _histogram_diff_masks(images: torch.Tensor,
                          grid_size: int,
                          threshold: float) -> torch.Tensor:
    """Per-region histogram change detection.

    1. Divide frame into grid_size x grid_size cells
    2. For each cell: compute per-channel histogram (16 bins) via torch.histc
    3. L2 distance between hist[n] and hist[n-1] per cell
    4. Cells with distance > threshold = changed
    5. Upsample cell-level binary mask to full resolution

    Frame 0 = zeros. Pure torch — no cv2 needed.

    images: (B, H, W, C) float32
    Returns: (B, H, W) float32
    """
    B, H, W, C = images.shape
    device = images.device

    if B <= 1:
        return torch.zeros(B, H, W, device=device, dtype=torch.float32)

    n_bins = 16
    cell_h = max(1, H // grid_size)
    cell_w = max(1, W // grid_size)
    actual_grid_h = max(1, (H + cell_h - 1) // cell_h)
    actual_grid_w = max(1, (W + cell_w - 1) // cell_w)

    # Compute per-cell histograms for all frames
    def _compute_cell_hists(frame: torch.Tensor) -> torch.Tensor:
        """Compute (grid_h, grid_w, C * n_bins) histogram for one frame (H, W, C)."""
        hists = torch.zeros(actual_grid_h, actual_grid_w, C * n_bins,
                            device=device, dtype=torch.float32)
        for gy in range(actual_grid_h):
            for gx in range(actual_grid_w):
                y1 = gy * cell_h
                y2 = min((gy + 1) * cell_h, H)
                x1 = gx * cell_w
                x2 = min((gx + 1) * cell_w, W)
                if y2 <= y1 or x2 <= x1:
                    continue
                cell = frame[y1:y2, x1:x2, :]  # (ch, cw, C)
                for c in range(C):
                    h = torch.histc(cell[:, :, c].float(), bins=n_bins,
                                    min=0.0, max=1.0)
                    # Normalize histogram to sum = 1
                    h_sum = h.sum()
                    if h_sum > 0:
                        h = h / h_sum
                    hists[gy, gx, c * n_bins:(c + 1) * n_bins] = h
        return hists

    # Compute cell-level change masks per frame
    prev_hists = _compute_cell_hists(images[0])
    results = [torch.zeros(H, W, device=device, dtype=torch.float32)]

    for b in range(1, B):
        curr_hists = _compute_cell_hists(images[b])
        # L2 distance between histograms per cell
        diff = (curr_hists - prev_hists).pow(2).sum(dim=-1).sqrt()  # (grid_h, grid_w)
        cell_mask = (diff > threshold).float()  # (grid_h, grid_w)

        # Upsample to full resolution
        cell_mask_4d = cell_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, grid_h, grid_w)
        full_mask = F.interpolate(cell_mask_4d, size=(H, W),
                                  mode="nearest").squeeze(0).squeeze(0)  # (H, W)
        results.append(full_mask)
        prev_hists = curr_hists

    return torch.stack(results, dim=0)


# ══════════════════════════════════════════════════════════════════════
#  NODE: MotionMaskTrackerMEC
# ══════════════════════════════════════════════════════════════════════

class MotionMaskTrackerMEC:
    """Per-frame motion detection mask generator.

    Four detection methods, each independently toggleable:
      1. Pixel difference
      2. Optical flow (Farneback cv2 / Laplacian torch fallback)
      3. Background subtraction
      4. Histogram difference

    Combined result = union (or intersection) of all enabled methods.
    Frame 0 always outputs zeros (no previous frame to compare).
    Motion intensity output: per-frame scalar useful for routing.
    """

    VRAM_TIER = 1

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Video batch (B,H,W,C). Minimum 2 frames for motion detection.",
                }),

                # ── Camera stabilization ──────────────────────────────
                "camera_compensation": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Estimate and subtract global camera motion before detection. "
                        "Critical for hand-held or moving camera footage — isolates "
                        "actual object motion from camera pan/tilt/rotation."
                    ),
                }),
                "stabilization_method": (["homography", "affine", "translation"], {
                    "default": "homography",
                    "tooltip": (
                        "homography: full perspective correction (pan/tilt/rotate/zoom, requires cv2). "
                        "affine: rotation + scale + translation (requires cv2). "
                        "translation: shift only (pure torch, fastest)."
                    ),
                }),

                "detection_mode": (["combined", "pixel_diff", "optical_flow",
                                    "background_sub", "histogram_diff"], {
                    "default": "combined",
                    "tooltip": (
                        "combined: union/intersection of all enabled methods. "
                        "Others: single method only."
                    ),
                }),

                # ── Pixel difference ──────────────────────────────────
                "pixel_diff_enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable absolute pixel difference detection between consecutive frames.",
                }),
                "pixel_diff_threshold": ("FLOAT", {
                    "default": 0.05, "min": 0.001, "max": 1.0, "step": 0.001,
                    "tooltip": "Pixel intensity change to count as moved. 0.05 = 5% brightness change.",
                }),

                # ── Optical flow ──────────────────────────────────────
                "flow_enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable optical flow magnitude detection (Farneback if cv2 available).",
                }),
                "flow_threshold": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 50.0, "step": 0.1,
                    "tooltip": "Flow magnitude in pixels/frame to count as moved.",
                }),
                "flow_algorithm": (["farneback", "phase_correlation"], {
                    "default": "farneback",
                    "tooltip": (
                        "farneback: cv2 dense optical flow (accurate, requires cv2). "
                        "phase_correlation: Laplacian shift detection (pure torch, faster)."
                    ),
                }),

                # ── Background subtraction ────────────────────────────
                "bg_sub_enabled": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable background subtraction (static BG model from first N frames).",
                }),
                "bg_model_frames": ("INT", {
                    "default": 5, "min": 1, "max": 30, "step": 1,
                    "tooltip": "Number of frames to average for background model. More = more stable.",
                }),
                "bg_sub_threshold": ("FLOAT", {
                    "default": 0.1, "min": 0.001, "max": 1.0, "step": 0.001,
                    "tooltip": "Distance from background model to count as foreground.",
                }),

                # ── Histogram difference ──────────────────────────────
                "hist_enabled": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable per-region histogram change detection.",
                }),
                "hist_grid_size": ("INT", {
                    "default": 16, "min": 4, "max": 64, "step": 4,
                    "tooltip": "Divide frame into NxN cells for histogram comparison.",
                }),
                "hist_threshold": ("FLOAT", {
                    "default": 0.15, "min": 0.01, "max": 1.0, "step": 0.01,
                    "tooltip": "Histogram L2 distance to count cell as changed.",
                }),

                # ── Post-processing ───────────────────────────────────
                "combine_method": (["union", "intersection"], {
                    "default": "union",
                    "tooltip": "union: any method triggers. intersection: all enabled methods must agree.",
                }),
                "grow_pixels": ("FLOAT", {
                    "default": 4.0, "min": 0.0, "max": 64.0, "step": 1.0,
                    "tooltip": "Expand detected regions by N pixels (fills gaps in motion mask).",
                }),
                "min_region_size": ("INT", {
                    "default": 100, "min": 0, "max": 10000, "step": 10,
                    "tooltip": "Remove isolated motion regions smaller than N pixels (noise filter).",
                }),
                "temporal_smooth": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Gaussian temporal smoothing across frames to suppress single-frame noise.",
                }),
            },
        }

    RETURN_TYPES = ("MASK", "FLOAT", "STRING")
    RETURN_NAMES = ("motion_mask", "motion_intensity", "info")
    FUNCTION = "execute"
    CATEGORY = "MaskEditControl/Video"
    DESCRIPTION = (
        "Per-frame motion detection using pixel diff, optical flow, "
        "background subtraction, and histogram analysis. "
        "Outputs motion mask + per-frame intensity score."
    )

    def execute(self, images: torch.Tensor,
                camera_compensation: bool, stabilization_method: str,
                detection_mode: str,
                pixel_diff_enabled: bool, pixel_diff_threshold: float,
                flow_enabled: bool, flow_threshold: float, flow_algorithm: str,
                bg_sub_enabled: bool, bg_model_frames: int, bg_sub_threshold: float,
                hist_enabled: bool, hist_grid_size: int, hist_threshold: float,
                combine_method: str,
                grow_pixels: float, min_region_size: int,
                temporal_smooth: bool) -> tuple:

        B, H, W, C = images.shape
        device = images.device

        if B <= 1:
            empty_mask = torch.zeros(B, H, W, device=device, dtype=torch.float32)
            return (empty_mask, 0.0, "[MEC] MotionMaskTracker: need >= 2 frames for motion detection.")

        with torch.no_grad():
            # ── Camera motion compensation ────────────────────────────
            camera_info = ""
            motion_magnitudes = None
            if camera_compensation:
                aligned_images, transforms = compensate_camera_motion(
                    images, method=stabilization_method, reference="previous"
                )
                motion_magnitudes = compute_motion_magnitudes(transforms)
                n_compensated = sum(1 for t in transforms if t is not None)
                avg_motion = (sum(motion_magnitudes) / len(motion_magnitudes)
                              if motion_magnitudes else 0.0)
                camera_info = (
                    f"  camera_compensation: {stabilization_method}, "
                    f"{n_compensated}/{B} frames aligned, "
                    f"avg_motion={avg_motion:.1f}px"
                )
                logger.info(
                    "[MEC] MotionMaskTracker: camera compensation applied "
                    "(%s, %d/%d frames, avg=%.1fpx)",
                    stabilization_method, n_compensated, B, avg_motion,
                )
                # Use aligned images for detection
                detect_images = aligned_images
            else:
                detect_images = images
            masks_list = []
            method_names = []

            # ── Which methods to run ──────────────────────────────────
            run_pixel = (detection_mode == "combined" and pixel_diff_enabled) or detection_mode == "pixel_diff"
            run_flow = (detection_mode == "combined" and flow_enabled) or detection_mode == "optical_flow"
            run_bg = (detection_mode == "combined" and bg_sub_enabled) or detection_mode == "background_sub"
            run_hist = (detection_mode == "combined" and hist_enabled) or detection_mode == "histogram_diff"

            # ── Method 1: Pixel diff ──────────────────────────────────
            if run_pixel:
                m = _pixel_diff_masks(detect_images, pixel_diff_threshold)
                masks_list.append(m)
                method_names.append("pixel_diff")

            # ── Method 2: Optical flow ────────────────────────────────
            if run_flow:
                if flow_algorithm == "farneback" and HAS_CV2:
                    m = _optical_flow_masks_cv2(detect_images, flow_threshold)
                else:
                    m = _optical_flow_masks_torch(detect_images, flow_threshold)
                    if flow_algorithm == "farneback" and not HAS_CV2:
                        logger.info("[MEC] MotionMaskTracker: cv2 not available, using Laplacian fallback for flow.")
                masks_list.append(m)
                method_names.append("optical_flow")

            # ── Method 3: Background subtraction ──────────────────────
            if run_bg:
                m = _background_sub_masks(detect_images, bg_model_frames, bg_sub_threshold)
                masks_list.append(m)
                method_names.append("bg_sub")

            # ── Method 4: Histogram diff ──────────────────────────────
            if run_hist:
                m = _histogram_diff_masks(detect_images, hist_grid_size, hist_threshold)
                masks_list.append(m)
                method_names.append("hist_diff")

            # ── Combine results ───────────────────────────────────────
            if not masks_list:
                # No methods enabled — return empty
                combined = torch.zeros(B, H, W, device=device, dtype=torch.float32)
                method_names.append("none")
            elif len(masks_list) == 1:
                combined = masks_list[0]
            else:
                stacked = torch.stack(masks_list, dim=0)  # (N_methods, B, H, W)
                if combine_method == "intersection":
                    # All enabled methods must agree: min across methods
                    combined = stacked.min(dim=0).values
                else:
                    # Union: any method triggers: max across methods
                    combined = stacked.max(dim=0).values

            # ── Post-processing: grow ─────────────────────────────────
            if grow_pixels > 0:
                combined = _grow_mask(combined, grow_pixels)

            # ── Post-processing: remove small regions ─────────────────
            if min_region_size > 0:
                combined = _remove_small_regions(combined, min_region_size)

            # ── Post-processing: temporal smooth ──────────────────────
            if temporal_smooth and B > 2:
                combined = motion_adaptive_temporal_smooth(
                    combined, sigma_base=1.0,
                    motion_magnitudes=motion_magnitudes,
                    motion_sensitivity=0.5,
                )

            # ── Compute motion intensity (per-frame mean coverage) ────
            per_frame_coverage = combined.mean(dim=(-2, -1))  # (B,)
            # Return overall mean as the single FLOAT output
            motion_intensity = float(per_frame_coverage.mean().item())

            # ── Build info string ─────────────────────────────────────
            per_frame_pcts = [f"f{i}={per_frame_coverage[i].item() * 100:.1f}%"
                              for i in range(min(B, 10))]
            if B > 10:
                per_frame_pcts.append(f"... ({B - 10} more)")

            info_lines = [
                f"[MEC] MotionMaskTracker: {B} frames | methods=[{', '.join(method_names)}]",
                f"  combine={combine_method} | grow={grow_pixels:.0f}px | min_region={min_region_size}",
                f"  temporal_smooth={temporal_smooth} (motion-adaptive)" if temporal_smooth else f"  temporal_smooth={temporal_smooth}",
                f"  overall motion intensity: {motion_intensity * 100:.1f}%",
                f"  per-frame coverage: {', '.join(per_frame_pcts)}",
            ]
            if camera_info:
                info_lines.insert(1, camera_info)
            if not HAS_CV2 and run_flow and flow_algorithm == "farneback":
                info_lines.append("  NOTE: cv2 not available, used Laplacian fallback for flow")
            info = "\n".join(info_lines)

            return (combined, motion_intensity, info)
