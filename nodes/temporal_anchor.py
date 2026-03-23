"""
TemporalAnchorMEC – Temporal Anchor System for mask interpolation over time.

Given anchor frames with their masks, computes signed distance fields (SDF)
for each anchor, then interpolates SDFs between anchors using configurable
easing to produce smooth mask transitions over a video sequence.

Optional: optical flow refinement via cv2 Farneback or torch phase correlation.

Outputs:
  - full_masks MASK (total_frames, H, W): interpolated mask per frame
  - confidence FLOAT batch: 1.0 at anchors, lower at midpoints
  - info STRING: computed SDF / flow metrics

Pure tensor SDF — no scipy dependency. VRAM Tier 2.
"""

from __future__ import annotations

import gc
import math
from typing import List, Tuple

import torch
import torch.nn.functional as F

# ── Optional cv2 with torch fallback ──────────────────────────────────
try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# ══════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════

def _get_device(tensor: torch.Tensor) -> torch.device:
    """Return the device of the tensor — never hardcode 'cuda'."""
    return tensor.device


def _parse_anchor_frames(anchor_frames_str: str, total_frames: int) -> List[int]:
    """Parse comma-separated frame indices, clamp to [0, total_frames-1], sort."""
    raw = [s.strip() for s in anchor_frames_str.split(",") if s.strip()]
    indices: List[int] = []
    for s in raw:
        try:
            idx = int(s)
            idx = max(0, min(idx, total_frames - 1))
            indices.append(idx)
        except ValueError:
            continue
    if not indices:
        indices = [0]
    indices = sorted(set(indices))
    return indices


# ══════════════════════════════════════════════════════════════════════
#  Easing functions — alpha in [0,1] → remapped alpha in [0,1]
# ══════════════════════════════════════════════════════════════════════

def _ease_linear(t: float) -> float:
    return t


def _ease_in(t: float) -> float:
    """Quadratic ease in: slow start, accelerating."""
    return t * t


def _ease_out(t: float) -> float:
    """Quadratic ease out: fast start, decelerating."""
    return 1.0 - (1.0 - t) * (1.0 - t)


def _ease_smooth_step(t: float) -> float:
    """Hermite smooth step: 3t² - 2t³."""
    return t * t * (3.0 - 2.0 * t)


_EASING_MAP = {
    "linear": _ease_linear,
    "ease_in": _ease_in,
    "ease_out": _ease_out,
    "smooth_step": _ease_smooth_step,
}


# ══════════════════════════════════════════════════════════════════════
#  SDF computation — iterative convolution, pure PyTorch
# ══════════════════════════════════════════════════════════════════════

def _compute_sdf_single(mask: torch.Tensor, iterations: int = 64) -> torch.Tensor:
    """Compute approximate signed distance field for a single (H, W) binary mask.

    Method: iterative averaging with boundary enforcement.
      1. Initialize distance field: -1 inside mask, +1 outside.
      2. Detect boundary pixels (where mask transitions).
      3. Iteratively smooth the field with a 3x3 averaging kernel.
      4. After each iteration, re-enforce boundary pixels to 0.
      5. The resulting field approximates the signed distance.

    Returns (H, W) float tensor: negative inside, positive outside, 0 at boundary.
    Scale is in approximate pixel units after normalization.
    """
    device = _get_device(mask)
    H, W = mask.shape

    # Binarize
    binary = (mask > 0.5).float()

    # Initialize: -1 inside, +1 outside
    sdf = torch.where(binary > 0.5, torch.tensor(-1.0, device=device),
                       torch.tensor(1.0, device=device))

    # Detect boundary: pixels where dilated mask != eroded mask
    binary_4d = binary.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    dilated = F.max_pool2d(binary_4d, kernel_size=3, stride=1, padding=1)
    eroded = -F.max_pool2d(-binary_4d, kernel_size=3, stride=1, padding=1)
    boundary = ((dilated - eroded) > 0.5).float().squeeze(0).squeeze(0)  # (H,W)

    # Averaging kernel for diffusion
    avg_kernel = torch.ones(1, 1, 3, 3, device=device, dtype=mask.dtype) / 9.0

    # Iterative diffusion
    for _ in range(iterations):
        sdf_4d = sdf.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        sdf_padded = F.pad(sdf_4d, (1, 1, 1, 1), mode="replicate")
        smoothed = F.conv2d(sdf_padded, avg_kernel, padding=0)
        sdf = smoothed.squeeze(0).squeeze(0)

        # Re-enforce boundary to 0
        sdf = sdf * (1.0 - boundary)

    # Normalize so that the range is approximately [-max_dist, +max_dist]
    # The max meaningful distance is half the diagonal
    max_dist = math.sqrt(H * H + W * W) / 2.0
    abs_max = sdf.abs().max().item()
    if abs_max > 1e-6:
        sdf = sdf * (max_dist / abs_max)

    return sdf


def _compute_sdf_batch(masks: torch.Tensor, iterations: int = 64) -> torch.Tensor:
    """Compute SDF for each mask in a batch. masks: (A, H, W) → (A, H, W) SDFs."""
    A = masks.shape[0]
    sdfs = []
    for i in range(A):
        try:
            sdf_i = _compute_sdf_single(masks[i], iterations=iterations)
        except torch.cuda.OutOfMemoryError:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            sdf_i = _compute_sdf_single(masks[i], iterations=max(8, iterations // 2))
        sdfs.append(sdf_i)
    return torch.stack(sdfs, dim=0)


# ══════════════════════════════════════════════════════════════════════
#  SDF interpolation with easing
# ══════════════════════════════════════════════════════════════════════

def _interpolate_sdf(
    sdf_batch: torch.Tensor,
    anchor_frames: List[int],
    total_frames: int,
    easing_fn,
) -> torch.Tensor:
    """Interpolate SDFs between anchors for every frame.

    sdf_batch: (A, H, W) — one SDF per anchor
    anchor_frames: sorted list of A frame indices
    Returns: (total_frames, H, W) interpolated SDF field
    """
    device = _get_device(sdf_batch)
    A, H, W = sdf_batch.shape
    result = torch.zeros(total_frames, H, W, device=device, dtype=sdf_batch.dtype)

    for t in range(total_frames):
        if t <= anchor_frames[0]:
            # Before or at first anchor → use first anchor's SDF
            result[t] = sdf_batch[0]
        elif t >= anchor_frames[-1]:
            # At or after last anchor → use last anchor's SDF
            result[t] = sdf_batch[-1]
        else:
            # Find surrounding anchors
            left_idx = 0
            for ai in range(len(anchor_frames) - 1):
                if anchor_frames[ai] <= t <= anchor_frames[ai + 1]:
                    left_idx = ai
                    break

            t_a = anchor_frames[left_idx]
            t_b = anchor_frames[left_idx + 1]
            span = t_b - t_a
            if span <= 0:
                result[t] = sdf_batch[left_idx]
            else:
                raw_alpha = (t - t_a) / span
                alpha = easing_fn(raw_alpha)
                result[t] = (1.0 - alpha) * sdf_batch[left_idx] + alpha * sdf_batch[left_idx + 1]

    return result


# ══════════════════════════════════════════════════════════════════════
#  Optical flow estimation — cv2 + torch fallback
# ══════════════════════════════════════════════════════════════════════

def _estimate_flow_cv2(frame_a: torch.Tensor, frame_b: torch.Tensor) -> torch.Tensor:
    """Estimate dense optical flow using cv2 Farneback.

    frame_a, frame_b: (H, W, C) float32 [0,1] tensors
    Returns: (H, W, 2) flow tensor on same device as input.
    """
    device = _get_device(frame_a)
    gray_a = (0.2126 * frame_a[:, :, 0] + 0.7152 * frame_a[:, :, 1] +
              0.0722 * frame_a[:, :, 2])
    gray_b = (0.2126 * frame_b[:, :, 0] + 0.7152 * frame_b[:, :, 1] +
              0.0722 * frame_b[:, :, 2])

    np_a = (gray_a.cpu().numpy() * 255).astype(np.uint8)
    np_b = (gray_b.cpu().numpy() * 255).astype(np.uint8)

    flow_np = cv2.calcOpticalFlowFarneback(
        np_a, np_b, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2,
        flags=0,
    )
    flow = torch.from_numpy(flow_np).to(device=device, dtype=torch.float32)
    return flow  # (H, W, 2)


def _estimate_flow_torch(frame_a: torch.Tensor, frame_b: torch.Tensor,
                          block_size: int = 32) -> torch.Tensor:
    """Estimate optical flow using block-wise phase correlation (pure torch).

    Divides each frame into blocks, computes FFT-based phase correlation per block
    to find the dominant translation, then bilinearly interpolates to a dense flow field.

    frame_a, frame_b: (H, W, C) float32 [0,1]
    Returns: (H, W, 2) flow tensor
    """
    device = _get_device(frame_a)
    H, W, _C = frame_a.shape

    # Convert to grayscale
    gray_a = 0.2126 * frame_a[:, :, 0] + 0.7152 * frame_a[:, :, 1] + 0.0722 * frame_a[:, :, 2]
    gray_b = 0.2126 * frame_b[:, :, 0] + 0.7152 * frame_b[:, :, 1] + 0.0722 * frame_b[:, :, 2]

    # Compute block grid
    n_blocks_h = max(1, H // block_size)
    n_blocks_w = max(1, W // block_size)
    bh = H // n_blocks_h
    bw = W // n_blocks_w

    # Per-block flow
    block_flows = torch.zeros(n_blocks_h, n_blocks_w, 2, device=device, dtype=torch.float32)

    for bi in range(n_blocks_h):
        for bj in range(n_blocks_w):
            y0 = bi * bh
            x0 = bj * bw
            y1 = min(y0 + bh, H)
            x1 = min(x0 + bw, W)

            patch_a = gray_a[y0:y1, x0:x1]
            patch_b = gray_b[y0:y1, x0:x1]

            ph, pw = patch_a.shape
            if ph < 4 or pw < 4:
                continue

            # Hanning window to reduce spectral leakage
            win_h = torch.hann_window(ph, device=device, dtype=torch.float32)
            win_w = torch.hann_window(pw, device=device, dtype=torch.float32)
            window = win_h.unsqueeze(1) * win_w.unsqueeze(0)

            fa = torch.fft.fft2(patch_a * window)
            fb = torch.fft.fft2(patch_b * window)

            # Cross-power spectrum
            cross = fa * fb.conj()
            cross_mag = cross.abs().clamp(min=1e-8)
            cross_norm = cross / cross_mag

            # Inverse FFT → correlation surface
            corr = torch.fft.ifft2(cross_norm).real

            # Find peak
            corr_flat = corr.reshape(-1)
            peak_idx = corr_flat.argmax().item()
            peak_y = peak_idx // pw
            peak_x = peak_idx % pw

            # Convert to signed displacement (wrap around)
            dy = peak_y if peak_y <= ph // 2 else peak_y - ph
            dx = peak_x if peak_x <= pw // 2 else peak_x - pw

            block_flows[bi, bj, 0] = float(dx)  # flow_x
            block_flows[bi, bj, 1] = float(dy)  # flow_y

    # Upscale block flows to dense field via bilinear interpolation
    # block_flows: (n_blocks_h, n_blocks_w, 2) → permute to (1, 2, n_blocks_h, n_blocks_w)
    flow_small = block_flows.permute(2, 0, 1).unsqueeze(0)  # (1, 2, nbh, nbw)
    flow_dense = F.interpolate(flow_small, size=(H, W), mode="bilinear",
                                align_corners=False)  # (1, 2, H, W)
    flow_dense = flow_dense.squeeze(0).permute(1, 2, 0)  # (H, W, 2)

    return flow_dense


def _warp_sdf_with_flow(sdf: torch.Tensor, flow: torch.Tensor,
                         strength: float = 1.0) -> torch.Tensor:
    """Warp a (H, W) SDF field using a (H, W, 2) flow field.

    Uses grid_sample for differentiable warping.
    """
    device = _get_device(sdf)
    H, W = sdf.shape

    # Build identity grid: (1, H, W, 2) with values in [-1, 1]
    grid_y = torch.linspace(-1.0, 1.0, H, device=device)
    grid_x = torch.linspace(-1.0, 1.0, W, device=device)
    grid_yy, grid_xx = torch.meshgrid(grid_y, grid_x, indexing="ij")
    grid = torch.stack([grid_xx, grid_yy], dim=-1).unsqueeze(0)  # (1, H, W, 2)

    # Normalize flow to [-1, 1] range
    flow_norm = flow.clone()
    flow_norm[:, :, 0] = flow_norm[:, :, 0] / (W / 2.0) * strength
    flow_norm[:, :, 1] = flow_norm[:, :, 1] / (H / 2.0) * strength

    # Displaced grid
    displaced = grid + flow_norm.unsqueeze(0)  # (1, H, W, 2)

    # Warp
    sdf_4d = sdf.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    warped = F.grid_sample(sdf_4d, displaced, mode="bilinear",
                            padding_mode="border", align_corners=False)
    return warped.squeeze(0).squeeze(0)  # (H, W)


# ══════════════════════════════════════════════════════════════════════
#  Confidence computation
# ══════════════════════════════════════════════════════════════════════

def _compute_confidence(anchor_frames: List[int], total_frames: int) -> List[float]:
    """Compute per-frame confidence. 1.0 at anchor frames, decreasing toward midpoints.

    For each frame, confidence = 1.0 - (distance_to_nearest_anchor / max_half_span).
    max_half_span is the largest half-gap between consecutive anchors.
    """
    if total_frames <= 0:
        return [0.0]

    # Compute max half-span for normalization
    max_half_span = 0.0
    if len(anchor_frames) == 1:
        max_half_span = max(anchor_frames[0], total_frames - 1 - anchor_frames[0])
        if max_half_span == 0:
            max_half_span = 1.0
    else:
        for i in range(len(anchor_frames) - 1):
            span = anchor_frames[i + 1] - anchor_frames[i]
            max_half_span = max(max_half_span, span / 2.0)
        # Also consider distance from 0 to first anchor and last anchor to end
        max_half_span = max(max_half_span, float(anchor_frames[0]))
        max_half_span = max(max_half_span, float(total_frames - 1 - anchor_frames[-1]))
        if max_half_span == 0:
            max_half_span = 1.0

    confidence = []
    for t in range(total_frames):
        min_dist = min(abs(t - af) for af in anchor_frames)
        conf = 1.0 - (min_dist / max_half_span)
        conf = max(0.0, min(1.0, conf))
        confidence.append(conf)

    return confidence


# ══════════════════════════════════════════════════════════════════════
#  Info string builder
# ══════════════════════════════════════════════════════════════════════

def _build_info(
    anchor_frames: List[int],
    total_frames: int,
    easing_name: str,
    sdf_batch: torch.Tensor,
    full_masks: torch.Tensor,
    confidence: List[float],
    flow_used: bool,
    flow_mag_mean: float = 0.0,
) -> str:
    """Build info string with real computed metrics."""
    lines = [
        "[MEC] Temporal Anchor System",
        f"  Anchor frames: {anchor_frames}",
        f"  Total frames: {total_frames}",
        f"  Easing: {easing_name}",
        f"  SDF shape: {list(sdf_batch.shape)}",
        f"  SDF range: [{sdf_batch.min().item():.3f}, {sdf_batch.max().item():.3f}]",
        f"  SDF mean: {sdf_batch.mean().item():.4f}",
        f"  Output mask coverage (mean): {full_masks.mean().item():.4f}",
        f"  Output mask coverage (std): {full_masks.std().item():.4f}",
        f"  Confidence range: [{min(confidence):.3f}, {max(confidence):.3f}]",
        f"  Confidence mean: {sum(confidence)/len(confidence):.3f}",
        f"  Flow refinement: {'enabled' if flow_used else 'disabled'}",
    ]
    if flow_used:
        lines.append(f"  Flow mean magnitude: {flow_mag_mean:.3f}")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════
#  Main Node Class
# ══════════════════════════════════════════════════════════════════════

class TemporalAnchorMEC:
    """Temporal Anchor System – SDF-based mask interpolation over time.

    Computes signed distance fields for anchor masks, interpolates between
    them with configurable easing, and optionally refines with optical flow.
    """

    VRAM_TIER = 2

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "anchor_masks": ("MASK", {
                    "tooltip": "Mask batch with one mask per anchor frame (A, H, W).",
                }),
                "anchor_frames": ("STRING", {
                    "default": "0",
                    "tooltip": (
                        "Comma-separated frame indices for each anchor mask. "
                        "E.g. '0,10,30'. Count must match number of anchor masks."
                    ),
                }),
                "total_frames": ("INT", {
                    "default": 30, "min": 1, "max": 99999,
                    "tooltip": "Total number of output frames in the sequence.",
                }),
                "easing": (["linear", "ease_in", "ease_out", "smooth_step"], {
                    "default": "smooth_step",
                    "tooltip": (
                        "Easing function for alpha interpolation between anchors.\n"
                        "linear: constant speed.\n"
                        "ease_in: quadratic slow start.\n"
                        "ease_out: quadratic slow end.\n"
                        "smooth_step: Hermite S-curve (3t²-2t³)."
                    ),
                }),
                "sdf_iterations": ("INT", {
                    "default": 64, "min": 4, "max": 512, "step": 4,
                    "tooltip": "Number of SDF diffusion iterations. More = more accurate but slower.",
                }),
                "flow_refinement": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable optical flow refinement of interpolated masks. Requires images input.",
                }),
            },
            "optional": {
                "images": ("IMAGE", {
                    "tooltip": "Video frames (B, H, W, C) for optical flow estimation. Required when flow_refinement is enabled.",
                }),
            },
        }

    RETURN_TYPES = ("MASK", "FLOAT", "STRING")
    RETURN_NAMES = ("full_masks", "confidence", "info")
    FUNCTION = "execute"
    CATEGORY = "MaskEditControl/Temporal"
    DESCRIPTION = (
        "Interpolate masks between anchor frames using Signed Distance Fields (SDF). "
        "Supports easing functions and optional optical flow refinement."
    )
    OUTPUT_IS_LIST = (False, True, False)

    def execute(
        self,
        anchor_masks: torch.Tensor,
        anchor_frames: str,
        total_frames: int,
        easing: str,
        sdf_iterations: int,
        flow_refinement: bool,
        images: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, List[float], str]:
        try:
            return self._execute_inner(
                anchor_masks, anchor_frames, total_frames,
                easing, sdf_iterations, flow_refinement, images,
            )
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _execute_inner(
        self,
        anchor_masks: torch.Tensor,
        anchor_frames_str: str,
        total_frames: int,
        easing: str,
        sdf_iterations: int,
        flow_refinement: bool,
        images: torch.Tensor | None,
    ) -> Tuple[torch.Tensor, List[float], str]:
        device = _get_device(anchor_masks)

        # ── Validate total_frames ────────────────────────────────────
        if total_frames <= 0:
            total_frames = 1

        # ── Ensure anchor_masks is 3D (A, H, W) ─────────────────────
        if anchor_masks.dim() == 2:
            anchor_masks = anchor_masks.unsqueeze(0)

        A, H, W = anchor_masks.shape

        # ── Parse anchor frame indices ───────────────────────────────
        anchor_frames_list = _parse_anchor_frames(anchor_frames_str, total_frames)

        # Trim or pad anchor list to match mask count
        if len(anchor_frames_list) > A:
            anchor_frames_list = anchor_frames_list[:A]
        elif len(anchor_frames_list) < A:
            # Distribute remaining anchors evenly
            for i in range(len(anchor_frames_list), A):
                if total_frames > 1:
                    idx = int(i * (total_frames - 1) / max(1, A - 1))
                else:
                    idx = 0
                anchor_frames_list.append(min(idx, total_frames - 1))
            anchor_frames_list = sorted(set(anchor_frames_list))
            # Re-trim if set() removed duplicates
            if len(anchor_frames_list) > A:
                anchor_frames_list = anchor_frames_list[:A]

        # ── Single anchor → replicate to all frames ──────────────────
        if A == 1 or len(anchor_frames_list) == 1:
            single_mask = (anchor_masks[0] > 0.5).float()
            full_masks = single_mask.unsqueeze(0).expand(total_frames, -1, -1).clone()
            confidence = _compute_confidence(anchor_frames_list, total_frames)
            sdf_single = _compute_sdf_single(anchor_masks[0], iterations=sdf_iterations)
            info = _build_info(
                anchor_frames_list, total_frames, easing,
                sdf_single.unsqueeze(0), full_masks, confidence,
                flow_used=False,
            )
            return (full_masks, confidence, info)

        # ── Resize anchor masks to uniform size ──────────────────────
        # Use first anchor's dimensions as target
        target_H, target_W = H, W
        resized_masks = []
        for i in range(A):
            m = anchor_masks[i]
            if m.shape[0] != target_H or m.shape[1] != target_W:
                m = F.interpolate(
                    m.unsqueeze(0).unsqueeze(0),
                    size=(target_H, target_W), mode="bilinear", align_corners=False,
                ).squeeze(0).squeeze(0)
            resized_masks.append(m)
        anchor_masks_uniform = torch.stack(resized_masks, dim=0)

        # ── Compute SDF for each anchor ──────────────────────────────
        sdf_batch = _compute_sdf_batch(anchor_masks_uniform, iterations=sdf_iterations)

        # ── Interpolate SDF across all frames ────────────────────────
        easing_fn = _EASING_MAP.get(easing, _ease_linear)
        sdf_interp = _interpolate_sdf(sdf_batch, anchor_frames_list, total_frames, easing_fn)

        # ── Optional flow refinement ─────────────────────────────────
        flow_used = False
        flow_mag_mean = 0.0
        if flow_refinement and images is not None:
            B_img = images.shape[0]
            if B_img >= 2:
                flow_used = True
                total_flow_mag = 0.0
                flow_count = 0

                # Resize images if needed for flow computation
                img_H, img_W = images.shape[1], images.shape[2]
                need_resize_flow = (img_H != target_H or img_W != target_W)

                for t in range(total_frames):
                    # Skip anchor frames — they're already exact
                    if t in anchor_frames_list:
                        continue

                    # Find nearest anchor frame for flow reference
                    nearest_anchor = min(anchor_frames_list, key=lambda af: abs(af - t))
                    anchor_img_idx = min(nearest_anchor, B_img - 1)
                    frame_img_idx = min(t, B_img - 1)

                    if anchor_img_idx == frame_img_idx:
                        continue

                    frame_a = images[anchor_img_idx]  # (H, W, C)
                    frame_b = images[frame_img_idx]    # (H, W, C)

                    # Compute flow
                    try:
                        if HAS_CV2:
                            flow_field = _estimate_flow_cv2(frame_a, frame_b)
                        else:
                            flow_field = _estimate_flow_torch(frame_a, frame_b)
                    except Exception:
                        continue

                    # Resize flow if image size differs from mask size
                    if need_resize_flow:
                        flow_resized = F.interpolate(
                            flow_field.permute(2, 0, 1).unsqueeze(0),
                            size=(target_H, target_W), mode="bilinear", align_corners=False,
                        ).squeeze(0).permute(1, 2, 0)
                        flow_resized[:, :, 0] *= target_W / img_W
                        flow_resized[:, :, 1] *= target_H / img_H
                        flow_field = flow_resized

                    # Proportional strength based on distance from anchor
                    nearest_dist = abs(t - nearest_anchor)
                    # Find the span this frame is in
                    max_span = 1
                    for ai in range(len(anchor_frames_list) - 1):
                        if anchor_frames_list[ai] <= t <= anchor_frames_list[ai + 1]:
                            max_span = anchor_frames_list[ai + 1] - anchor_frames_list[ai]
                            break
                    strength = nearest_dist / max(1, max_span)

                    # Warp the interpolated SDF
                    warped = _warp_sdf_with_flow(sdf_interp[t], flow_field, strength=strength)
                    sdf_interp[t] = warped

                    # Track flow magnitude
                    flow_mag = flow_field.norm(dim=-1).mean().item()
                    total_flow_mag += flow_mag
                    flow_count += 1

                if flow_count > 0:
                    flow_mag_mean = total_flow_mag / flow_count

        # ── Threshold SDF → binary masks ─────────────────────────────
        full_masks = (sdf_interp < 0).float()

        # ── Compute confidence ───────────────────────────────────────
        confidence = _compute_confidence(anchor_frames_list, total_frames)

        # ── Build info string ────────────────────────────────────────
        info = _build_info(
            anchor_frames_list, total_frames, easing,
            sdf_batch, full_masks, confidence,
            flow_used=flow_used, flow_mag_mean=flow_mag_mean,
        )

        return (full_masks, confidence, info)
