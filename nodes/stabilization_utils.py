"""
stabilization_utils — Shared camera/motion stabilization primitives.

Provides:
  - estimate_homography(): ORB feature matching + RANSAC homography
  - estimate_affine(): ORB + estimateAffinePartial2D (rotation+scale+translation)
  - warp_frame(): cv2.warpPerspective / warpAffine with torch fallback
  - compensate_camera_motion(): align frame sequence to reference frame
  - motion_adaptive_temporal_smooth(): weight smoothing inversely to motion magnitude
  - smooth_bbox_trajectory(): median + exponential smoothing for bbox coordinates

All functions have pure-torch fallbacks when cv2 is unavailable.
VRAM Tier: 0 (CPU-side feature matching, lightweight tensor ops)
"""

from __future__ import annotations

import math
import logging
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

logger = logging.getLogger("MEC")


# ══════════════════════════════════════════════════════════════════════
#  Homography / affine estimation via feature matching
# ══════════════════════════════════════════════════════════════════════

def _to_gray_uint8(frame: np.ndarray) -> np.ndarray:
    """Convert (H,W,C) float32 [0,1] or uint8 frame to grayscale uint8."""
    if frame.dtype == np.float32 or frame.dtype == np.float64:
        frame = (np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8)
    if frame.ndim == 3 and frame.shape[2] >= 3:
        return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    if frame.ndim == 2:
        return frame
    return frame[..., 0]


def estimate_homography(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    max_features: int = 1000,
    match_ratio: float = 0.75,
    ransac_thresh: float = 5.0,
) -> Optional[np.ndarray]:
    """Estimate 3x3 homography from frame_a to frame_b using ORB + RANSAC.

    Returns None if not enough matches found.
    Requires cv2.
    """
    if not HAS_CV2:
        return None

    gray_a = _to_gray_uint8(frame_a)
    gray_b = _to_gray_uint8(frame_b)

    orb = cv2.ORB_create(nfeatures=max_features)
    kp_a, desc_a = orb.detectAndCompute(gray_a, None)
    kp_b, desc_b = orb.detectAndCompute(gray_b, None)

    if desc_a is None or desc_b is None or len(kp_a) < 4 or len(kp_b) < 4:
        return None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw_matches = bf.knnMatch(desc_a, desc_b, k=2)

    # Lowe's ratio test
    good = []
    for m_pair in raw_matches:
        if len(m_pair) == 2:
            m, n = m_pair
            if m.distance < match_ratio * n.distance:
                good.append(m)

    if len(good) < 4:
        return None

    pts_a = np.float32([kp_a[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts_b = np.float32([kp_b[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, inlier_mask = cv2.findHomography(pts_a, pts_b, cv2.RANSAC, ransac_thresh)

    if H is None:
        return None

    # Validate: a reasonable homography should not have extreme perspective
    if abs(H[2, 0]) > 0.01 or abs(H[2, 1]) > 0.01:
        # Extreme perspective — likely bad match, fall back to affine
        return estimate_affine(frame_a, frame_b, max_features, match_ratio)

    return H


def estimate_affine(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    max_features: int = 1000,
    match_ratio: float = 0.75,
) -> Optional[np.ndarray]:
    """Estimate rigid affine (rotation + scale + translation) from a to b.

    Returns 3x3 matrix (last row = [0,0,1]) or None.
    Requires cv2.
    """
    if not HAS_CV2:
        return None

    gray_a = _to_gray_uint8(frame_a)
    gray_b = _to_gray_uint8(frame_b)

    orb = cv2.ORB_create(nfeatures=max_features)
    kp_a, desc_a = orb.detectAndCompute(gray_a, None)
    kp_b, desc_b = orb.detectAndCompute(gray_b, None)

    if desc_a is None or desc_b is None or len(kp_a) < 3 or len(kp_b) < 3:
        return None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw_matches = bf.knnMatch(desc_a, desc_b, k=2)

    good = []
    for m_pair in raw_matches:
        if len(m_pair) == 2:
            m, n = m_pair
            if m.distance < match_ratio * n.distance:
                good.append(m)

    if len(good) < 3:
        return None

    pts_a = np.float32([kp_a[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts_b = np.float32([kp_b[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, inliers = cv2.estimateAffinePartial2D(pts_a, pts_b, method=cv2.RANSAC)
    if M is None:
        return None

    # Convert 2x3 affine to 3x3
    H = np.eye(3, dtype=np.float64)
    H[:2, :] = M
    return H


def estimate_translation_torch(
    frame_a: torch.Tensor,
    frame_b: torch.Tensor,
) -> Tuple[float, float]:
    """Estimate (dx, dy) translation between two (H,W,C) frames via phase correlation.

    Pure torch — no cv2. Returns (dx, dy) pixel shift.
    """
    # Convert to grayscale
    if frame_a.dim() == 3 and frame_a.shape[-1] >= 3:
        gray_a = 0.2126 * frame_a[..., 0] + 0.7152 * frame_a[..., 1] + 0.0722 * frame_a[..., 2]
        gray_b = 0.2126 * frame_b[..., 0] + 0.7152 * frame_b[..., 1] + 0.0722 * frame_b[..., 2]
    else:
        gray_a = frame_a.squeeze(-1) if frame_a.dim() == 3 else frame_a
        gray_b = frame_b.squeeze(-1) if frame_b.dim() == 3 else frame_b

    # Phase correlation
    fft_a = torch.fft.fft2(gray_a.float())
    fft_b = torch.fft.fft2(gray_b.float())

    cross_power = fft_a * fft_b.conj()
    magnitude = cross_power.abs().clamp(min=1e-8)
    normalized = cross_power / magnitude

    correlation = torch.fft.ifft2(normalized).real

    H, W = correlation.shape
    peak_idx = correlation.argmax()
    peak_y = (peak_idx // W).item()
    peak_x = (peak_idx % W).item()

    # Unwrap to signed shift
    dx = peak_x if peak_x <= W // 2 else peak_x - W
    dy = peak_y if peak_y <= H // 2 else peak_y - H

    return (float(dx), float(dy))


# ══════════════════════════════════════════════════════════════════════
#  Frame warping
# ══════════════════════════════════════════════════════════════════════

def warp_frame_cv2(
    frame: np.ndarray,
    H_matrix: np.ndarray,
    output_size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Warp frame using 3x3 homography/affine matrix. cv2 required.

    frame: (H, W, C) or (H, W) — any dtype
    H_matrix: 3x3 numpy array
    output_size: (W, H) or None to keep same size
    Returns warped frame same shape/dtype.
    """
    h, w = frame.shape[:2]
    if output_size is None:
        output_size = (w, h)

    # Check if it's a pure affine (last row is [0, 0, 1])
    is_affine = (abs(H_matrix[2, 0]) < 1e-10 and
                 abs(H_matrix[2, 1]) < 1e-10 and
                 abs(H_matrix[2, 2] - 1.0) < 1e-10)

    if is_affine:
        warped = cv2.warpAffine(frame, H_matrix[:2, :].astype(np.float64),
                                output_size, flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REPLICATE)
    else:
        warped = cv2.warpPerspective(frame, H_matrix.astype(np.float64),
                                     output_size, flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REPLICATE)
    return warped


def warp_frame_torch(
    frame: torch.Tensor,
    dx: float,
    dy: float,
) -> torch.Tensor:
    """Warp (H,W,C) tensor by translation (dx, dy) using grid_sample.

    Pure torch fallback — translation only.
    """
    H, W = frame.shape[0], frame.shape[1]
    C = frame.shape[2] if frame.dim() == 3 else 1

    if frame.dim() == 2:
        inp = frame.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    else:
        inp = frame.permute(2, 0, 1).unsqueeze(0)  # (1,C,H,W)

    # Build translation grid
    theta = torch.tensor(
        [[1.0, 0.0, -2.0 * dx / W],
         [0.0, 1.0, -2.0 * dy / H]],
        dtype=frame.dtype, device=frame.device
    ).unsqueeze(0)

    grid = F.affine_grid(theta, inp.shape, align_corners=False)
    warped = F.grid_sample(inp, grid, mode="bilinear", padding_mode="border",
                           align_corners=False)

    if frame.dim() == 2:
        return warped.squeeze(0).squeeze(0)
    return warped.squeeze(0).permute(1, 2, 0)


# ══════════════════════════════════════════════════════════════════════
#  Camera motion compensation for frame sequences
# ══════════════════════════════════════════════════════════════════════

def compensate_camera_motion(
    images: torch.Tensor,
    method: str = "homography",
    reference: str = "previous",
) -> Tuple[torch.Tensor, List[Optional[np.ndarray]]]:
    """Align all frames to compensate for global camera motion.

    images: (B, H, W, C) float32 [0,1]
    method: 'homography' | 'affine' | 'translation'
    reference: 'previous' (align each to prior frame) | 'first' (align all to frame 0)

    Returns:
      aligned_images: (B, H, W, C) — camera-compensated
      transforms: list of 3x3 numpy arrays (or None for frame 0 / failures)
    """
    B, H, W, C = images.shape
    device = images.device
    aligned = images.clone()
    transforms: List[Optional[np.ndarray]] = [None]  # frame 0 = identity

    use_cv2 = HAS_CV2 and method in ("homography", "affine")

    for i in range(1, B):
        ref_idx = i - 1 if reference == "previous" else 0

        if use_cv2:
            frame_ref = images[ref_idx].cpu().numpy()
            frame_cur = images[i].cpu().numpy()

            if method == "homography":
                H_mat = estimate_homography(frame_ref, frame_cur)
            else:
                H_mat = estimate_affine(frame_ref, frame_cur)

            if H_mat is not None:
                # Inverse homography: warp current frame to align with reference
                try:
                    H_inv = np.linalg.inv(H_mat)
                except np.linalg.LinAlgError:
                    H_inv = None

                if H_inv is not None:
                    frame_np = images[i].cpu().numpy()
                    if frame_np.dtype != np.float32:
                        frame_np = frame_np.astype(np.float32)
                    warped = warp_frame_cv2(frame_np, H_inv, (W, H))
                    aligned[i] = torch.from_numpy(warped).to(device)
                    transforms.append(H_mat)
                else:
                    transforms.append(None)
            else:
                transforms.append(None)
        else:
            # Pure torch: translation-only via phase correlation
            dx, dy = estimate_translation_torch(images[ref_idx], images[i])
            if abs(dx) > 0.5 or abs(dy) > 0.5:
                aligned[i] = warp_frame_torch(images[i], dx, dy)
                # Store as 3x3 translation matrix
                T = np.eye(3, dtype=np.float64)
                T[0, 2] = dx
                T[1, 2] = dy
                transforms.append(T)
            else:
                transforms.append(None)

    # For 'previous' reference, compose transforms to get cumulative alignment
    if reference == "previous" and B > 2:
        cumulative = np.eye(3, dtype=np.float64)
        for i in range(1, B):
            if transforms[i] is not None:
                cumulative = transforms[i] @ cumulative
                # Re-warp from original frame using cumulative transform
                if use_cv2:
                    try:
                        H_inv = np.linalg.inv(cumulative)
                    except np.linalg.LinAlgError:
                        continue
                    frame_np = images[i].cpu().numpy()
                    if frame_np.dtype != np.float32:
                        frame_np = frame_np.astype(np.float32)
                    warped = warp_frame_cv2(frame_np, H_inv, (W, H))
                    aligned[i] = torch.from_numpy(warped).to(device)
                else:
                    dx = cumulative[0, 2]
                    dy = cumulative[1, 2]
                    aligned[i] = warp_frame_torch(images[i], dx, dy)

    return aligned, transforms


def compute_motion_magnitudes(
    transforms: List[Optional[np.ndarray]],
) -> List[float]:
    """Extract per-frame motion magnitude from transform list.

    Returns list of floats — magnitude of camera displacement per frame.
    """
    magnitudes = []
    for t in transforms:
        if t is None:
            magnitudes.append(0.0)
        else:
            # Translation component
            dx = t[0, 2]
            dy = t[1, 2]
            mag = math.sqrt(dx * dx + dy * dy)
            # Add rotation component if present
            cos_a = t[0, 0]
            sin_a = t[1, 0] if t.shape[0] > 1 else 0.0
            angle = abs(math.atan2(sin_a, cos_a))
            # Combine: pixels of translation + angle-equivalent pixels
            mag += angle * 100.0  # rough: 1 radian ~ 100px equivalent
            magnitudes.append(mag)
    return magnitudes


# ══════════════════════════════════════════════════════════════════════
#  Motion-adaptive temporal smoothing
# ══════════════════════════════════════════════════════════════════════

def _gauss_kernel_1d(sigma: float, device: torch.device,
                     dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """1D Gaussian kernel, normalized."""
    if sigma <= 0:
        return torch.ones(1, device=device, dtype=dtype)
    radius = max(1, int(math.ceil(2.5 * sigma)))
    size = 2 * radius + 1
    x = torch.arange(size, device=device, dtype=dtype) - radius
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def motion_adaptive_temporal_smooth(
    mask: torch.Tensor,
    sigma_base: float = 1.5,
    motion_magnitudes: Optional[List[float]] = None,
    motion_sensitivity: float = 0.5,
) -> torch.Tensor:
    """Smooth (B, H, W) mask along batch dim with motion-adaptive Gaussian.

    At high-motion frames, sigma is reduced (less smoothing = preserve detail).
    At low-motion frames, sigma is increased (more smoothing = eliminate noise).

    mask: (B, H, W) float32
    sigma_base: base smoothing sigma (default 1.5)
    motion_magnitudes: per-frame motion magnitude (len B), or None for uniform
    motion_sensitivity: 0=ignore motion (uniform), 1=full adaptation
    """
    B, H, W = mask.shape
    if B <= 1 or sigma_base <= 0:
        return mask

    device = mask.device

    if motion_magnitudes is None or motion_sensitivity <= 0 or len(motion_magnitudes) != B:
        # Fallback: uniform Gaussian (same as original)
        k1d = _gauss_kernel_1d(sigma_base, device, mask.dtype)
        radius = len(k1d) // 2
        flat = mask.reshape(B, H * W).permute(1, 0).unsqueeze(1)
        kernel = k1d.view(1, 1, -1)
        padded = F.pad(flat, (radius, radius), mode="replicate")
        smoothed = F.conv1d(padded, kernel)
        return smoothed.squeeze(1).permute(1, 0).reshape(B, H, W).clamp(0.0, 1.0)

    # ── Adaptive: per-frame varying sigma ──
    # Normalize motion magnitudes to [0, 1]
    max_mag = max(motion_magnitudes) if motion_magnitudes else 1.0
    if max_mag < 1e-6:
        max_mag = 1.0
    norm_mags = [m / max_mag for m in motion_magnitudes]

    # Per-frame sigma: high motion → low sigma, low motion → high sigma
    # sigma_frame = sigma_base * (1 - sensitivity * norm_mag)
    # Clamp sigma_min to 0.3 to avoid zero-width kernel
    sigmas = [max(0.3, sigma_base * (1.0 - motion_sensitivity * nm))
              for nm in norm_mags]

    # Apply per-frame weighted smoothing
    # For each frame, build a kernel centered at that frame and weight neighbors
    result = torch.zeros_like(mask)
    flat = mask.reshape(B, H * W)  # (B, HW)

    for t in range(B):
        sigma_t = sigmas[t]
        radius = max(1, int(math.ceil(2.5 * sigma_t)))
        weights = torch.zeros(B, device=device, dtype=mask.dtype)
        for j in range(max(0, t - radius), min(B, t + radius + 1)):
            dist = abs(j - t)
            weights[j] = math.exp(-0.5 * (dist / sigma_t) ** 2)
        weights = weights / weights.sum()
        # Weighted average of all frames
        result_flat = (flat * weights.unsqueeze(1)).sum(dim=0)  # (HW,)
        result[t] = result_flat.reshape(H, W)

    return result.clamp(0.0, 1.0)


# ══════════════════════════════════════════════════════════════════════
#  Smooth bounding box trajectory
# ══════════════════════════════════════════════════════════════════════

def smooth_bbox_trajectory(
    bboxes: List[Tuple[int, int, int, int]],
    method: str = "median_then_exponential",
    window_radius: int = 3,
    alpha: float = 0.3,
) -> List[Tuple[int, int, int, int]]:
    """Smooth a sequence of (x, y, w, h) bboxes for stable video cropping.

    Methods:
      'median_then_exponential': median filter for outlier rejection → exponential smooth
      'median': pure median filter (good for removing jumps)
      'exponential': exponential moving average only

    Returns list of smoothed (x, y, w, h) tuples.
    """
    n = len(bboxes)
    if n <= 1:
        return list(bboxes)

    # Convert to arrays for easier manipulation
    arr = np.array(bboxes, dtype=np.float64)  # (N, 4)

    # Step 1: Median filter for outlier rejection
    if method in ("median", "median_then_exponential"):
        filtered = np.copy(arr)
        for i in range(n):
            start = max(0, i - window_radius)
            end = min(n, i + window_radius + 1)
            window = arr[start:end]
            filtered[i] = np.median(window, axis=0)
        arr = filtered

    # Step 2: Exponential smoothing
    if method in ("exponential", "median_then_exponential"):
        smoothed = np.copy(arr)
        for i in range(1, n):
            smoothed[i] = alpha * arr[i] + (1.0 - alpha) * smoothed[i - 1]
        arr = smoothed

    # Convert back to int tuples
    return [(int(round(row[0])), int(round(row[1])),
             int(round(row[2])), int(round(row[3])))
            for row in arr]


def compute_stable_bbox_trajectory(
    mask: torch.Tensor,
    smooth_method: str = "median_then_exponential",
    window_radius: int = 3,
    alpha: float = 0.3,
) -> Tuple[int, int, int, int]:
    """Compute a smoothed stable bounding box from a (B, H, W) mask sequence.

    Instead of simple union, computes per-frame bbox then smooths the trajectory,
    returning the bbox that covers the smoothed trajectory range.

    Returns (x, y, w, h) — single bbox for the whole sequence.
    """
    B, H, W = mask.shape

    # Compute per-frame bboxes
    per_frame = []
    for b in range(B):
        nonzero = torch.nonzero(mask[b] > 0.5, as_tuple=False)
        if nonzero.numel() == 0:
            per_frame.append(None)
        else:
            y_min = nonzero[:, 0].min().item()
            y_max = nonzero[:, 0].max().item()
            x_min = nonzero[:, 1].min().item()
            x_max = nonzero[:, 1].max().item()
            per_frame.append((x_min, y_min, x_max - x_min + 1, y_max - y_min + 1))

    # Fill gaps: interpolate from neighbors
    valid_indices = [i for i, b in enumerate(per_frame) if b is not None]
    if not valid_indices:
        return (0, 0, W, H)

    if len(valid_indices) < B:
        # Fill missing frames with nearest valid bbox
        for i in range(B):
            if per_frame[i] is None:
                nearest = min(valid_indices, key=lambda j: abs(j - i))
                per_frame[i] = per_frame[nearest]

    # Smooth the trajectory
    smoothed = smooth_bbox_trajectory(per_frame, smooth_method, window_radius, alpha)

    # Compute union of the smoothed trajectory
    x_min = min(b[0] for b in smoothed)
    y_min = min(b[1] for b in smoothed)
    x_max = max(b[0] + b[2] for b in smoothed)
    y_max = max(b[1] + b[3] for b in smoothed)

    return (max(0, x_min), max(0, y_min),
            min(W, x_max) - max(0, x_min),
            min(H, y_max) - max(0, y_min))
