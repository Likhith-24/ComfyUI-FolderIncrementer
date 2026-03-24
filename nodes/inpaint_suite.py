"""
InpaintSuiteMEC — Inpaint Crop Pro + Stitch Pro + Mask Prepare.

Three nodes for professional inpainting workflows:
  1. InpaintCropProMEC    — crop around mask with blend mask preparation
  2. InpaintStitchProMEC  — composite inpainted area back seamlessly
  3. InpaintMaskPrepareMEC — standalone mask cleanup and dual-mask output

Key innovations over InpaintCropAndStitch:
  - Separated inpaint_mask_mode (what model sees) from stitch_blend_mode (how stitch composites)
  - Edge-aware blend: Sobel-guided boundary snapping
  - Laplacian pyramid blend: real multi-level frequency decomposition
  - Frequency blend: FFT-domain blending
  - video_stable_crop: lock bbox across all frames
  - Temporal blend mask stabilization

VRAM Tier: 1 (pure tensor ops, no models)
"""

from __future__ import annotations

import gc
import math
import logging
from typing import Dict, List, Tuple, Optional, Any

import torch
import torch.nn.functional as F

import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from scipy.ndimage import binary_closing, binary_fill_holes
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

logger = logging.getLogger("MEC")


# ══════════════════════════════════════════════════════════════════════
#  Device helper
# ══════════════════════════════════════════════════════════════════════

def _get_device(tensor: torch.Tensor) -> torch.device:
    """Return the device of a tensor — never hardcode 'cuda'."""
    return tensor.device


# ══════════════════════════════════════════════════════════════════════
#  Gaussian kernel helpers
# ══════════════════════════════════════════════════════════════════════

def _gauss_kernel_1d(sigma: float, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Create a normalized 1D Gaussian kernel."""
    if sigma <= 0:
        return torch.ones(1, device=device, dtype=dtype)
    radius = max(1, int(math.ceil(3.0 * sigma)))
    size = 2 * radius + 1
    x = torch.arange(size, device=device, dtype=dtype) - radius
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()
    return kernel


def _gaussian_blur_2d(tensor: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable 2D Gaussian blur. tensor: (B, C, H, W) or (B, 1, H, W).

    Pure torch — no cv2 dependency.
    """
    if sigma <= 0:
        return tensor
    device = _get_device(tensor)
    k1d = _gauss_kernel_1d(sigma, device, tensor.dtype)
    pad = len(k1d) // 2
    C = tensor.shape[1]
    # Horizontal pass
    kh = k1d.view(1, 1, 1, -1).expand(C, 1, 1, -1)
    out = F.conv2d(F.pad(tensor, (pad, pad, 0, 0), mode="replicate"), kh, groups=C)
    # Vertical pass
    kv = k1d.view(1, 1, -1, 1).expand(C, 1, -1, 1)
    out = F.conv2d(F.pad(out, (0, 0, pad, pad), mode="replicate"), kv, groups=C)
    return out


def _gaussian_blur_mask(mask: torch.Tensor, sigma: float) -> torch.Tensor:
    """Blur a (B, H, W) mask with 2D Gaussian. Returns (B, H, W)."""
    if sigma <= 0:
        return mask
    m4 = mask.unsqueeze(1)  # (B, 1, H, W)
    blurred = _gaussian_blur_2d(m4, sigma)
    return blurred.squeeze(1)


# ══════════════════════════════════════════════════════════════════════
#  Sobel edge detection — pure torch
# ══════════════════════════════════════════════════════════════════════

_SOBEL_X = torch.tensor(
    [[-1.0, 0.0, 1.0],
     [-2.0, 0.0, 2.0],
     [-1.0, 0.0, 1.0]], dtype=torch.float32
).unsqueeze(0).unsqueeze(0)  # (1,1,3,3)

_SOBEL_Y = torch.tensor(
    [[-1.0, -2.0, -1.0],
     [ 0.0,  0.0,  0.0],
     [ 1.0,  2.0,  1.0]], dtype=torch.float32
).unsqueeze(0).unsqueeze(0)  # (1,1,3,3)


def _sobel_edges(gray: torch.Tensor) -> torch.Tensor:
    """Compute Sobel edge magnitude from (B, H, W) grayscale. Returns (B, H, W)."""
    device = _get_device(gray)
    sx = _SOBEL_X.to(device=device, dtype=gray.dtype)
    sy = _SOBEL_Y.to(device=device, dtype=gray.dtype)
    inp = gray.unsqueeze(1)  # (B, 1, H, W)
    gx = F.conv2d(F.pad(inp, (1, 1, 1, 1), mode="replicate"), sx)
    gy = F.conv2d(F.pad(inp, (1, 1, 1, 1), mode="replicate"), sy)
    magnitude = (gx.pow(2) + gy.pow(2)).sqrt().squeeze(1)  # (B, H, W)
    return magnitude


# ══════════════════════════════════════════════════════════════════════
#  Edge-aware blend mask — Sobel-guided boundary snapping
# ══════════════════════════════════════════════════════════════════════

def _edge_aware_blend_mask(image: torch.Tensor, mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Create an edge-aware blend mask that snaps blend boundaries to image edges.

    image: (B, H, W, C) float32
    mask:  (B, H, W) float32 binary/soft
    radius: blend feather radius
    Returns: (B, H, W) soft blend mask
    """
    B, H, W, C = image.shape

    # Step 1: Compute image luminance
    luma = 0.2126 * image[:, :, :, 0] + 0.7152 * image[:, :, :, 1] + 0.0722 * image[:, :, :, 2]

    # Step 2: Compute Sobel edge magnitude
    edge_mag = _sobel_edges(luma)  # (B, H, W)
    # Normalize edge magnitude to [0, 1] per frame
    for b in range(B):
        emax = edge_mag[b].max()
        if emax > 0:
            edge_mag[b] = edge_mag[b] / emax

    # Step 3: Create base Gaussian blend mask (dilated + blurred)
    binary = (mask > 0.5).float()

    # Step 4: Compute edge-snapped mask
    # Where image edges are strong, the blend boundary should be sharper (snap to edges)
    # Where image edges are weak, use the smooth Gaussian boundary
    sigma_base = radius * 0.4
    sigma_min = max(0.5, sigma_base * 0.15)

    # Create a spatially-varying sharpness via blending sharp + smooth versions
    sharp_blend = _gaussian_blur_mask(binary, sigma_min)   # (B, H, W)
    smooth_blend = _gaussian_blur_mask(binary, sigma_base)  # (B, H, W)

    # Blend: at strong edges use sharp, at weak edges use smooth
    edge_weight = edge_mag.clamp(0.0, 1.0)
    result = edge_weight * sharp_blend + (1.0 - edge_weight) * smooth_blend

    # Ensure the mask interior is 1.0 and exterior is 0.0
    result = torch.where(binary > 0.5, torch.max(result, binary), result)

    return result.clamp(0.0, 1.0)


# ══════════════════════════════════════════════════════════════════════
#  Laplacian pyramid blend — real multi-level decomposition (pure torch)
# ══════════════════════════════════════════════════════════════════════

def _build_laplacian_pyramid_torch(img: torch.Tensor, levels: int) -> List[torch.Tensor]:
    """Build a Laplacian pyramid from (B, C, H, W) image.

    Returns list of tensors: levels Laplacian layers + 1 residual (coarsest).
    Each Laplacian layer = current - upsample(downsample(current)).
    """
    pyramid: List[torch.Tensor] = []
    current = img
    for i in range(levels):
        h, w = current.shape[2], current.shape[3]
        # Downsample by 2x with Gaussian pre-filter
        down = _gaussian_blur_2d(current, sigma=1.0)
        down = F.interpolate(down, size=(max(1, h // 2), max(1, w // 2)), mode="bilinear", align_corners=False)
        # Upsample back
        up = F.interpolate(down, size=(h, w), mode="bilinear", align_corners=False)
        # Laplacian = difference between current and upsampled-downsampled
        laplacian = current - up
        pyramid.append(laplacian)
        current = down
    # Residual (coarsest level)
    pyramid.append(current)
    return pyramid


def _reconstruct_from_pyramid(pyramid: List[torch.Tensor]) -> torch.Tensor:
    """Reconstruct image from Laplacian pyramid. Returns (B, C, H, W)."""
    current = pyramid[-1]
    for i in range(len(pyramid) - 2, -1, -1):
        h, w = pyramid[i].shape[2], pyramid[i].shape[3]
        up = F.interpolate(current, size=(h, w), mode="bilinear", align_corners=False)
        current = up + pyramid[i]
    return current


def _laplacian_pyramid_blend(img_a: torch.Tensor, img_b: torch.Tensor,
                              blend_mask: torch.Tensor, levels: int = 5) -> torch.Tensor:
    """Blend two (B, C, H, W) images using Laplacian pyramid blending.

    blend_mask: (B, 1, H, W) — 0.0 = use img_a, 1.0 = use img_b
    """
    # Clamp levels based on image size
    min_dim = min(img_a.shape[2], img_a.shape[3])
    max_levels = max(1, int(math.log2(max(min_dim, 1))))
    levels = min(levels, max_levels)

    pyr_a = _build_laplacian_pyramid_torch(img_a, levels)
    pyr_b = _build_laplacian_pyramid_torch(img_b, levels)

    # Build Gaussian pyramid for the mask
    mask_pyr: List[torch.Tensor] = []
    current_mask = blend_mask
    for i in range(levels):
        mask_pyr.append(current_mask)
        h, w = current_mask.shape[2], current_mask.shape[3]
        current_mask = F.interpolate(current_mask, size=(max(1, h // 2), max(1, w // 2)),
                                     mode="bilinear", align_corners=False)
    mask_pyr.append(current_mask)

    # Blend each level
    blended_pyr: List[torch.Tensor] = []
    for i in range(len(pyr_a)):
        m = mask_pyr[i]
        blended = (1.0 - m) * pyr_a[i] + m * pyr_b[i]
        blended_pyr.append(blended)

    return _reconstruct_from_pyramid(blended_pyr)


# ══════════════════════════════════════════════════════════════════════
#  Frequency blend — FFT-domain blending
# ══════════════════════════════════════════════════════════════════════

def _frequency_blend(img_a: torch.Tensor, img_b: torch.Tensor,
                     blend_mask: torch.Tensor) -> torch.Tensor:
    """FFT-based frequency domain blending of two (B, C, H, W) images.

    blend_mask: (B, 1, H, W) — 0.0 = use img_a, 1.0 = use img_b
    """
    # Compute FFT of both images
    fft_a = torch.fft.rfft2(img_a)
    fft_b = torch.fft.rfft2(img_b)

    B, C, H, W = img_a.shape
    freq_h, freq_w = fft_a.shape[2], fft_a.shape[3]

    # Build low-pass and high-pass versions of the blend mask in spatial domain
    low_mask = _gaussian_blur_2d(blend_mask, sigma=max(H, W) * 0.1)
    high_mask = blend_mask

    # Map spatial masks to frequency domain shape
    low_mask_freq = F.interpolate(low_mask, size=(freq_h, freq_w), mode="bilinear", align_corners=False)
    high_mask_freq = F.interpolate(high_mask, size=(freq_h, freq_w), mode="bilinear", align_corners=False)

    # Create radial frequency coordinate
    fy = torch.arange(freq_h, device=img_a.device, dtype=torch.float32) / max(freq_h, 1)
    fx = torch.arange(freq_w, device=img_a.device, dtype=torch.float32) / max(freq_w, 1)
    fy_grid, fx_grid = torch.meshgrid(fy, fx, indexing="ij")
    freq_radius = (fy_grid.pow(2) + fx_grid.pow(2)).sqrt()
    max_r = freq_radius.max()
    if max_r > 0:
        freq_radius = freq_radius / max_r
    freq_radius = freq_radius.unsqueeze(0).unsqueeze(0)  # (1,1,fH,fW)

    # Interpolate: at low freq use low_mask_freq, at high freq use high_mask_freq
    freq_weight = freq_radius.clamp(0.0, 1.0)
    final_freq_mask = (1.0 - freq_weight) * low_mask_freq + freq_weight * high_mask_freq

    # Blend in frequency domain
    fft_blended = (1.0 - final_freq_mask) * fft_a + final_freq_mask * fft_b

    # Inverse FFT
    result = torch.fft.irfft2(fft_blended, s=(H, W))
    return result.clamp(0.0, 1.0)


# ══════════════════════════════════════════════════════════════════════
#  Video stable bbox — union of all frame masks
# ══════════════════════════════════════════════════════════════════════

def _compute_bbox_single(mask_2d: torch.Tensor) -> Tuple[int, int, int, int]:
    """Compute (x, y, w, h) bounding box of nonzero region in a 2D mask.

    Returns (-1, -1, -1, -1) if mask is empty.
    """
    nonzero = torch.nonzero(mask_2d > 0.5, as_tuple=False)
    if nonzero.numel() == 0:
        return (-1, -1, -1, -1)
    y_min = nonzero[:, 0].min().item()
    y_max = nonzero[:, 0].max().item()
    x_min = nonzero[:, 1].min().item()
    x_max = nonzero[:, 1].max().item()
    return (x_min, y_min, x_max - x_min + 1, y_max - y_min + 1)


def _compute_stable_bbox(mask: torch.Tensor) -> Tuple[int, int, int, int]:
    """Compute the union bounding box across all frames.

    mask: (B, H, W) float32
    Returns: (x, y, w, h) — single bbox covering all frames' mask regions.
    """
    B, H, W = mask.shape
    union_x_min, union_y_min = W, H
    union_x_max, union_y_max = 0, 0
    any_valid = False

    for b in range(B):
        x, y, w, h = _compute_bbox_single(mask[b])
        if x < 0:
            continue
        any_valid = True
        union_x_min = min(union_x_min, x)
        union_y_min = min(union_y_min, y)
        union_x_max = max(union_x_max, x + w)
        union_y_max = max(union_y_max, y + h)

    if not any_valid:
        return (0, 0, W, H)

    return (union_x_min, union_y_min,
            union_x_max - union_x_min, union_y_max - union_y_min)


# ══════════════════════════════════════════════════════════════════════
#  Temporal Gaussian smoothing along batch dimension
# ══════════════════════════════════════════════════════════════════════

def _temporal_gaussian_smooth(mask: torch.Tensor, sigma: float) -> torch.Tensor:
    """Smooth a (B, H, W) mask along the batch (temporal) dimension.

    Applies 1D Gaussian convolution along dim=0 independently per spatial pixel.
    Returns (B, H, W).
    """
    B, H, W = mask.shape
    if B <= 1 or sigma <= 0:
        return mask
    device = _get_device(mask)
    k1d = _gauss_kernel_1d(sigma, device, mask.dtype)  # (K,)
    K = len(k1d)
    pad = K // 2

    # Reshape: (B, H*W) → (H*W, 1, B) for conv1d
    flat = mask.reshape(B, H * W).permute(1, 0)  # (H*W, B)
    flat = flat.unsqueeze(1)  # (H*W, 1, B)

    kernel = k1d.view(1, 1, -1)  # (1, 1, K)
    padded = F.pad(flat, (pad, pad), mode="replicate")
    smoothed = F.conv1d(padded, kernel)  # (H*W, 1, B)

    # Reshape back: (H*W, 1, B) → (B, H, W)
    smoothed = smoothed.squeeze(1).permute(1, 0).reshape(B, H, W)
    return smoothed.clamp(0.0, 1.0)


# ══════════════════════════════════════════════════════════════════════
#  Morphological helpers — fill holes and remove small regions (torch)
# ══════════════════════════════════════════════════════════════════════

def _fill_holes_torch(mask: torch.Tensor) -> torch.Tensor:
    """Fill interior holes in a (B, H, W) binary mask.

    Uses cv2 contour hierarchy if available, else morphological closing as fallback.
    """
    B, H, W = mask.shape
    device = _get_device(mask)
    binary = (mask > 0.5).float()

    if HAS_CV2:
        results = []
        for b in range(B):
            mask_np = binary[b].cpu().numpy().astype(np.uint8) * 255
            contours, hierarchy = cv2.findContours(
                mask_np, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
            )
            if hierarchy is not None:
                for i, h in enumerate(hierarchy[0]):
                    if h[3] >= 0:
                        cv2.drawContours(mask_np, contours, i, 255, -1)
            results.append(torch.from_numpy((mask_np > 127).astype(np.float32)))
        return torch.stack(results, dim=0).to(device)
    else:
        # Torch fallback: morphological closing with increasing kernel sizes
        result = binary.unsqueeze(1)  # (B, 1, H, W)
        for k_size in [3, 5, 7, 11]:
            pad = k_size // 2
            # Dilation (max_pool)
            dilated = F.max_pool2d(result, kernel_size=k_size, stride=1, padding=pad)
            # Erosion (-max_pool(-x))
            eroded = -F.max_pool2d(-dilated, kernel_size=k_size, stride=1, padding=pad)
            result = eroded
        return result.squeeze(1).clamp(0.0, 1.0)


def _remove_small_regions_torch(mask: torch.Tensor, min_area: int) -> torch.Tensor:
    """Remove connected components smaller than min_area in (B, H, W) mask.

    Uses cv2 connectedComponents if available, else erosion/dilation approximation.
    """
    if min_area <= 0:
        return mask
    B, H, W = mask.shape
    device = _get_device(mask)

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
        # Torch fallback: erode then dilate to remove small blobs
        k_size = max(3, int(math.sqrt(min_area)))
        if k_size % 2 == 0:
            k_size += 1
        pad = k_size // 2
        m4 = mask.unsqueeze(1)
        eroded = -F.max_pool2d(-m4, kernel_size=k_size, stride=1, padding=pad)
        dilated = F.max_pool2d(eroded, kernel_size=k_size, stride=1, padding=pad)
        return dilated.squeeze(1).clamp(0.0, 1.0)


# ══════════════════════════════════════════════════════════════════════
#  Color matching — mean+std transfer
# ══════════════════════════════════════════════════════════════════════

def _color_match_mean_std(source: torch.Tensor, target: torch.Tensor,
                          mask: torch.Tensor) -> torch.Tensor:
    """Match source colors to target using mean+std transfer within masked region.

    source: (B, H, W, C) — the inpainted result to adjust
    target: (B, H, W, C) — the original image reference
    mask: (B, H, W) — region where color matching applies
    Returns adjusted source: (B, H, W, C)
    """
    result = source.clone()
    B = source.shape[0]
    binary = (mask > 0.5).float()

    for b in range(B):
        region_mask = binary[b]  # (H, W)
        if region_mask.sum() < 10:
            continue
        m3 = region_mask.unsqueeze(-1)  # (H, W, 1)

        src_pixels = source[b] * m3
        tgt_pixels = target[b] * m3
        n_pixels = region_mask.sum().clamp(min=1)

        src_mean = src_pixels.sum(dim=(0, 1)) / n_pixels
        tgt_mean = tgt_pixels.sum(dim=(0, 1)) / n_pixels

        src_var = ((source[b] - src_mean) * m3).pow(2).sum(dim=(0, 1)) / n_pixels
        tgt_var = ((target[b] - tgt_mean) * m3).pow(2).sum(dim=(0, 1)) / n_pixels

        src_std = src_var.sqrt().clamp(min=1e-6)
        tgt_std = tgt_var.sqrt().clamp(min=1e-6)

        adjusted = (source[b] - src_mean) * (tgt_std / src_std) + tgt_mean
        result[b] = source[b] * (1.0 - m3) + adjusted * m3

    return result.clamp(0.0, 1.0)


# ══════════════════════════════════════════════════════════════════════
#  Inpaint mask mode application
# ══════════════════════════════════════════════════════════════════════

def _apply_inpaint_mask_mode(mask: torch.Tensor, mode: str) -> torch.Tensor:
    """Convert mask to the format expected by the inpaint model.

    mask: (B, H, W) float32
    mode: 'hard_binary' | 'slight_feather' | 'soft_blend'
    Returns: (B, H, W)
    """
    if mode == "hard_binary":
        return (mask > 0.5).float()
    elif mode == "slight_feather":
        binary = (mask > 0.5).float()
        feathered = _gaussian_blur_mask(binary, sigma=1.5)
        return feathered.clamp(0.0, 1.0)
    elif mode == "soft_blend":
        return _gaussian_blur_mask(mask, sigma=3.0).clamp(0.0, 1.0)
    else:
        return (mask > 0.5).float()


# ══════════════════════════════════════════════════════════════════════
#  Blend mask generation dispatcher
# ══════════════════════════════════════════════════════════════════════

def _generate_stitch_blend_mask(image: torch.Tensor, mask: torch.Tensor,
                                 mode: str, radius: int) -> torch.Tensor:
    """Generate the stitch blend mask based on the selected mode.

    image: (B, H, W, C)
    mask: (B, H, W)
    mode: 'edge_aware' | 'gaussian' | 'laplacian_pyramid' | 'frequency_blend'
    radius: feather radius
    Returns: (B, H, W) soft blend mask
    """
    if mode == "edge_aware":
        return _edge_aware_blend_mask(image, mask, radius)
    elif mode == "gaussian":
        binary = (mask > 0.5).float()
        return _gaussian_blur_mask(binary, sigma=radius * 0.4)
    elif mode == "laplacian_pyramid":
        binary = (mask > 0.5).float()
        return _gaussian_blur_mask(binary, sigma=radius * 0.5)
    elif mode == "frequency_blend":
        binary = (mask > 0.5).float()
        return _gaussian_blur_mask(binary, sigma=radius * 0.6)
    else:
        binary = (mask > 0.5).float()
        return _gaussian_blur_mask(binary, sigma=radius * 0.4)


# ══════════════════════════════════════════════════════════════════════
#  Size mode helpers
# ══════════════════════════════════════════════════════════════════════

def _apply_size_mode(crop_w: int, crop_h: int, size_mode: str,
                     forced_width: int, forced_height: int,
                     min_size: int, max_size: int,
                     padding_multiple: int) -> Tuple[int, int]:
    """Compute the final output dimensions based on size_mode."""
    if size_mode == "forced_size":
        tw, th = forced_width, forced_height
    elif size_mode == "ranged_size":
        aspect = crop_w / max(crop_h, 1)
        tw, th = crop_w, crop_h
        if tw < min_size or th < min_size:
            if tw < th:
                tw = min_size
                th = max(min_size, int(tw / aspect))
            else:
                th = min_size
                tw = max(min_size, int(th * aspect))
        if tw > max_size or th > max_size:
            if tw > th:
                tw = max_size
                th = max(1, int(tw / aspect))
            else:
                th = max_size
                tw = max(1, int(th * aspect))
        tw = max(tw, min_size)
        th = max(th, min_size)
    else:
        tw, th = crop_w, crop_h

    if padding_multiple > 1:
        tw = int(math.ceil(tw / padding_multiple) * padding_multiple)
        th = int(math.ceil(th / padding_multiple) * padding_multiple)

    tw = max(tw, padding_multiple)
    th = max(th, padding_multiple)
    return (tw, th)


# ══════════════════════════════════════════════════════════════════════
#  Resize helpers (IMAGE and MASK)
# ══════════════════════════════════════════════════════════════════════

# Supported interpolation modes for F.interpolate
# "lanczos" is implemented via PIL fallback; others map to torch modes.
_TORCH_INTERP_MODES = {
    "bilinear": "bilinear",
    "bicubic": "bicubic",
    "nearest-exact": "nearest-exact",
    "area": "area",
}


def _resize_image(image: torch.Tensor, target_h: int, target_w: int,
                  mode: str = "bilinear") -> torch.Tensor:
    """Resize (B, H, W, C) image to (B, target_h, target_w, C)."""
    if image.shape[1] == target_h and image.shape[2] == target_w:
        return image
    if mode == "lanczos":
        return _resize_lanczos(image, target_h, target_w)
    torch_mode = _TORCH_INTERP_MODES.get(mode, "bilinear")
    img_bchw = image.permute(0, 3, 1, 2)
    align = False if torch_mode in ("bilinear", "bicubic") else None
    kwargs = {"size": (target_h, target_w), "mode": torch_mode}
    if align is not None:
        kwargs["align_corners"] = align
    resized = F.interpolate(img_bchw, **kwargs)
    return resized.permute(0, 2, 3, 1)


def _resize_mask(mask: torch.Tensor, target_h: int, target_w: int,
                 mode: str = "bilinear") -> torch.Tensor:
    """Resize (B, H, W) mask to (B, target_h, target_w)."""
    if mask.shape[1] == target_h and mask.shape[2] == target_w:
        return mask
    if mode == "lanczos":
        # For masks, convert to 4D, use PIL lanczos, squeeze back
        m4 = mask.unsqueeze(-1).expand(-1, -1, -1, 1)  # (B,H,W,1)
        resized = _resize_lanczos(m4, target_h, target_w)
        return resized[..., 0]
    torch_mode = _TORCH_INTERP_MODES.get(mode, "bilinear")
    m4 = mask.unsqueeze(1)
    kwargs = {"size": (target_h, target_w), "mode": torch_mode}
    if torch_mode in ("bilinear", "bicubic"):
        kwargs["align_corners"] = False
    resized = F.interpolate(m4, **kwargs)
    return resized.squeeze(1)


def _resize_lanczos(image: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """Resize (B,H,W,C) via PIL Lanczos (high quality, slightly slower)."""
    from PIL import Image as PILImage
    B, H, W, C = image.shape
    out = []
    for i in range(B):
        arr = (image[i].cpu().numpy() * 255.0).clip(0, 255).astype("uint8")
        if C == 1:
            arr = arr[:, :, 0]  # PIL needs 2D for grayscale
        pil_img = PILImage.fromarray(arr)
        pil_img = pil_img.resize((target_w, target_h), PILImage.LANCZOS)
        t = torch.from_numpy(np.array(pil_img).astype("float32") / 255.0)
        if C == 1:
            t = t.unsqueeze(-1)  # restore channel dim
        out.append(t)
    return torch.stack(out).to(image.device)


# ══════════════════════════════════════════════════════════════════════
#  NODE 1: InpaintCropProMEC
# ══════════════════════════════════════════════════════════════════════

class InpaintCropProMEC:
    """Crop image around mask region with separate inpaint and stitch blend masks.

    Key improvements over InpaintCropAndStitch:
    - Separated inpaint_mask_mode from stitch_blend_mode
    - Edge-aware, Laplacian pyramid, and frequency blend modes
    - video_stable_crop for consistent bbox across video frames
    - Downscale/upscale factors for quality control
    - Mask pre-processing (blur, grow/shrink)
    - Aspect ratio presets with snap to multiple of 8
    - CROP_MASK output for mask-aware workflows
    """

    VRAM_TIER = 1
    STITCH_BLEND_MODES = ["edge_aware", "gaussian", "laplacian_pyramid", "frequency_blend"]
    INPAINT_MASK_MODES = ["hard_binary", "slight_feather", "soft_blend"]
    SIZE_MODES = ["free_size", "forced_size", "ranged_size"]
    FILL_MODES = ["edge_pad", "neutral_gray", "original"]
    INTERP_METHODS = ["lanczos", "bicubic", "bilinear", "nearest-exact", "area"]
    ASPECT_RATIOS = [
        "none",
        "1:1", "4:3", "3:4", "16:9", "9:16",
        "3:2", "2:3", "21:9", "9:21",
        "5:4", "4:5", "custom",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image batch (B,H,W,C)"}),
                "mask": ("MASK", {"tooltip": "Inpaint mask (B,H,W), white = area to inpaint"}),
                "context_expand": ("FLOAT", {
                    "default": 1.5, "min": 1.0, "max": 4.0, "step": 0.1,
                    "tooltip": "How much to expand the crop bbox beyond the mask bounds (1.0 = tight crop)"}),
                "inpaint_mask_mode": (cls.INPAINT_MASK_MODES, {
                    "default": "hard_binary",
                    "tooltip": "What the inpaint model sees: hard_binary (crisp), slight_feather (gentle edges), soft_blend (very soft)"}),
                "stitch_blend_mode": (cls.STITCH_BLEND_MODES, {
                    "default": "gaussian",
                    "tooltip": "How the result is composited back: edge_aware (Sobel-guided), gaussian, laplacian_pyramid, frequency_blend"}),
                "blend_radius": ("INT", {
                    "default": 32, "min": 1, "max": 256, "step": 1,
                    "tooltip": "Feather radius for the stitch blend mask"}),
                "size_mode": (cls.SIZE_MODES, {
                    "default": "free_size",
                    "tooltip": "free_size (keep crop dims), forced_size (exact WxH), ranged_size (clamp to min/max)"}),
                "forced_width": ("INT", {
                    "default": 1024, "min": 64, "max": 8192, "step": 8,
                    "tooltip": "Target width for forced_size mode"}),
                "forced_height": ("INT", {
                    "default": 1024, "min": 64, "max": 8192, "step": 8,
                    "tooltip": "Target height for forced_size mode"}),
                "min_size": ("INT", {
                    "default": 512, "min": 64, "max": 8192, "step": 8,
                    "tooltip": "Minimum dimension for ranged_size mode"}),
                "max_size": ("INT", {
                    "default": 2048, "min": 64, "max": 8192, "step": 8,
                    "tooltip": "Maximum dimension for ranged_size mode"}),
                "padding_multiple": ("INT", {
                    "default": 8, "min": 2, "max": 128, "step": 2,
                    "tooltip": "Pad output dimensions to be divisible by this value (must be even, default: 8)"}),
                "video_stable_crop": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Lock bbox across all frames using union of all mask regions (for video)"}),
                "fill_masked_area": (cls.FILL_MODES, {
                    "default": "edge_pad",
                    "tooltip": "How to fill the masked area in the crop: edge_pad, neutral_gray, or original"}),
                "downscale_factor": ("FLOAT", {
                    "default": 1.0, "min": 0.25, "max": 1.0, "step": 0.05,
                    "tooltip": "Downscale crop before inpainting (e.g. 0.5 = half resolution). Stitched back at original res."}),
                "downscale_method": (cls.INTERP_METHODS, {
                    "default": "lanczos",
                    "tooltip": "Interpolation method for downscaling: lanczos (best quality), bicubic, bilinear, nearest-exact, area"}),
                "upscale_method": (cls.INTERP_METHODS, {
                    "default": "lanczos",
                    "tooltip": "Interpolation method for upscaling during stitch-back: lanczos, bicubic, bilinear, nearest-exact, area"}),
                "mask_blur": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 64.0, "step": 0.5,
                    "tooltip": "Blur the mask edges before cropping (sigma in pixels)"}),
                "mask_grow": ("INT", {
                    "default": 0, "min": -128, "max": 128, "step": 1,
                    "tooltip": "Grow (positive) or shrink (negative) the mask before cropping (pixels)"}),
                "aspect_ratio": (cls.ASPECT_RATIOS, {
                    "default": "none",
                    "tooltip": "Force crop region to specific aspect ratio. 'none' = follow mask shape."}),
                "custom_aspect_w": ("INT", {
                    "default": 1, "min": 1, "max": 100, "step": 1,
                    "tooltip": "Width component of custom aspect ratio"}),
                "custom_aspect_h": ("INT", {
                    "default": 1, "min": 1, "max": 100, "step": 1,
                    "tooltip": "Height component of custom aspect ratio"}),
                "mask_invert": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Invert the mask before processing (swap inpaint/keep regions)"}),
                "mask_fill_holes": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Fill enclosed holes inside the mask (prevents gaps from breaking inpainting)"}),
                "mask_hipass_filter": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Ignore mask values below this threshold (removes near-transparent noise)"}),
            },
            "optional": {
                "optional_context_mask": ("MASK", {
                    "tooltip": "Extra mask defining additional context area to include in the crop (does not affect inpainting, only crop bounds)"}),
            },
        }

    RETURN_TYPES = ("STITCH_DATA", "IMAGE", "IMAGE", "MASK", "MASK", "MASK", "STRING")
    RETURN_NAMES = ("stitch_data", "cropped_image", "cropped_composite", "inpaint_mask", "stitch_blend_mask", "crop_mask", "info")
    FUNCTION = "crop_for_inpaint"
    CATEGORY = "MaskEditControl/Inpaint"
    DESCRIPTION = "Crop image around mask with separate inpaint and stitch blend masks. Supports edge-aware, Laplacian, frequency blend, downscale, mask processing, and aspect ratio presets."

    @staticmethod
    def _snap(val: int, multiple: int) -> int:
        """Snap value up to nearest multiple."""
        if multiple <= 1:
            return val
        return int(math.ceil(val / multiple) * multiple)

    @staticmethod
    def _parse_aspect_ratio(name: str, custom_w: int, custom_h: int):
        """Return (w_ratio, h_ratio) or None if no aspect ratio enforcement."""
        if name == "none":
            return None
        if name == "custom":
            return (max(1, custom_w), max(1, custom_h))
        parts = name.split(":")
        if len(parts) == 2:
            return (int(parts[0]), int(parts[1]))
        return None

    @staticmethod
    def _fill_mask_holes(mask: torch.Tensor) -> torch.Tensor:
        """Fill enclosed holes inside mask using iterative multi-threshold approach.

        Uses scipy binary_closing + binary_fill_holes at descending thresholds
        to properly handle soft/gradient masks (lquesada's proven method).
        Falls back to simple binary flood-fill when scipy is not available.
        """
        if HAS_SCIPY:
            results = []
            thresholds = [1, 0.99, 0.97, 0.95, 0.93, 0.9, 0.8, 0.7, 0.6, 0.5,
                          0.4, 0.3, 0.2, 0.1]
            for b in range(mask.shape[0]):
                mask_np = mask[b].cpu().numpy()
                for threshold in thresholds:
                    thresholded = mask_np >= threshold
                    closed = binary_closing(thresholded,
                                            structure=np.ones((3, 3)),
                                            border_value=1)
                    filled = binary_fill_holes(closed)
                    mask_np = np.maximum(mask_np,
                                         np.where(filled != 0, threshold, 0))
                results.append(torch.from_numpy(mask_np.astype(np.float32)))
            return torch.stack(results, dim=0).to(mask.device)

        # Torch-only fallback: simple binary flood-fill from borders
        filled = mask.clone()
        for b in range(filled.shape[0]):
            m = filled[b]
            binary = (m > 0.5).float()
            bg = torch.zeros_like(binary)
            H, W = binary.shape
            bg[0, :] = 1.0 - binary[0, :]
            bg[-1, :] = 1.0 - binary[-1, :]
            bg[:, 0] = torch.max(bg[:, 0], 1.0 - binary[:, 0])
            bg[:, -1] = torch.max(bg[:, -1], 1.0 - binary[:, -1])
            inv = 1.0 - binary
            kernel = torch.ones(1, 1, 3, 3, device=mask.device) / 9.0
            for _ in range(max(H, W)):
                prev = bg
                padded = F.pad(bg.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1),
                               mode="constant", value=0)
                bg = (F.conv2d(padded, kernel).squeeze(0).squeeze(0) > 0.01).float() * inv
                if torch.equal(bg, prev):
                    break
            holes = (1.0 - binary) * (1.0 - bg)
            filled[b] = torch.max(m, holes)
        return filled

    @staticmethod
    def _grow_mask(mask: torch.Tensor, pixels: int) -> torch.Tensor:
        """Morphological grow/shrink of mask. Positive = dilate, negative = erode."""
        if pixels == 0:
            return mask
        k = abs(pixels) * 2 + 1
        kernel = torch.ones(1, 1, k, k, device=mask.device, dtype=mask.dtype) / (k * k)
        pad = abs(pixels)
        m4 = mask.unsqueeze(1)
        m4 = F.pad(m4, (pad, pad, pad, pad), mode="replicate")
        conv = F.conv2d(m4, kernel)
        out = conv.squeeze(1)
        if pixels > 0:
            # Dilate: any overlap → 1
            return (out > 0.01).float()
        else:
            # Erode: full overlap → 1
            return (out > 0.99).float()

    def crop_for_inpaint(self, image: torch.Tensor, mask: torch.Tensor,
                         context_expand: float, inpaint_mask_mode: str,
                         stitch_blend_mode: str, blend_radius: int,
                         size_mode: str, forced_width: int, forced_height: int,
                         min_size: int, max_size: int, padding_multiple: int,
                         video_stable_crop: bool, fill_masked_area: str,
                         downscale_factor: float = 1.0,
                         downscale_method: str = "lanczos",
                         upscale_method: str = "lanczos",
                         mask_blur: float = 0.0, mask_grow: int = 0,
                         aspect_ratio: str = "none",
                         custom_aspect_w: int = 1, custom_aspect_h: int = 1,
                         mask_invert: bool = False, mask_fill_holes: bool = False,
                         mask_hipass_filter: float = 0.0,
                         optional_context_mask: Optional[torch.Tensor] = None):
        device = _get_device(image)
        B, H, W, C = image.shape

        # Ensure mask dimensions match image
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        if mask.shape[0] == 1 and B > 1:
            mask = mask.expand(B, -1, -1).clone()
        if mask.shape[1] != H or mask.shape[2] != W:
            mask = _resize_mask(mask, H, W)

        # Prepare optional context mask
        if optional_context_mask is not None:
            if optional_context_mask.dim() == 2:
                optional_context_mask = optional_context_mask.unsqueeze(0)
            if optional_context_mask.shape[0] == 1 and B > 1:
                optional_context_mask = optional_context_mask.expand(B, -1, -1).clone()
            if optional_context_mask.shape[1] != H or optional_context_mask.shape[2] != W:
                optional_context_mask = _resize_mask(optional_context_mask, H, W)

        # ── Mask pre-processing: invert, hipass, fill holes, grow, blur ──
        if mask_invert:
            mask = 1.0 - mask
        if mask_hipass_filter > 0:
            mask = torch.where(mask >= mask_hipass_filter, mask, torch.zeros_like(mask))
        if mask_fill_holes:
            mask = self._fill_mask_holes(mask)
        if mask_grow != 0:
            mask = self._grow_mask(mask, mask_grow)
        if mask_blur > 0:
            mask = _gaussian_blur_mask(mask, sigma=mask_blur).clamp(0.0, 1.0)

        # Step 1: Compute bounding box
        bbox_x, bbox_y, bbox_w, bbox_h = _compute_stable_bbox(mask)

        # Handle empty mask
        if bbox_w <= 0 or bbox_h <= 0:
            bbox_x, bbox_y, bbox_w, bbox_h = 0, 0, W, H

        # Step 2: Expand bbox by context_expand, always centered on mask
        center_x = bbox_x + bbox_w / 2.0
        center_y = bbox_y + bbox_h / 2.0

        crop_w = int(round(bbox_w * context_expand))
        crop_h = int(round(bbox_h * context_expand))

        if crop_w <= 0 or crop_h <= 0:
            crop_w, crop_h = W, H
            center_x, center_y = W / 2.0, H / 2.0

        # Combine with optional_context_mask bbox (lquesada pattern)
        if optional_context_mask is not None:
            ctx_bbox = _compute_stable_bbox(optional_context_mask)
            if ctx_bbox[2] > 0 and ctx_bbox[3] > 0:
                # Union the expanded crop bbox with the context mask bbox
                cx1, cy1 = int(round(center_x - crop_w / 2.0)), int(round(center_y - crop_h / 2.0))
                cx2, cy2 = cx1 + crop_w, cy1 + crop_h
                ox1, oy1 = ctx_bbox[0], ctx_bbox[1]
                ox2, oy2 = ox1 + ctx_bbox[2], oy1 + ctx_bbox[3]
                ux1 = min(cx1, ox1)
                uy1 = min(cy1, oy1)
                ux2 = max(cx2, ox2)
                uy2 = max(cy2, oy2)
                crop_w = ux2 - ux1
                crop_h = uy2 - uy1
                center_x = ux1 + crop_w / 2.0
                center_y = uy1 + crop_h / 2.0

        # ── Enforce aspect ratio on crop region ────────────────────────
        ar = self._parse_aspect_ratio(aspect_ratio, custom_aspect_w, custom_aspect_h)
        if ar is not None:
            ar_w, ar_h = ar
            target_ar = ar_w / ar_h
            current_ar = crop_w / max(crop_h, 1)
            if current_ar > target_ar:
                crop_h = int(round(crop_w / target_ar))
            else:
                crop_w = int(round(crop_h * target_ar))

        # Snap crop dimensions to padding_multiple (expand, not shrink)
        crop_w = self._snap(crop_w, padding_multiple)
        crop_h = self._snap(crop_h, padding_multiple)

        # Step 3: Compute target output size
        target_w, target_h = _apply_size_mode(
            crop_w, crop_h, size_mode,
            forced_width, forced_height,
            min_size, max_size, padding_multiple
        )

        # When forced_size, align crop aspect ratio to target
        if size_mode == "forced_size" and target_w > 0 and target_h > 0:
            target_ar = target_w / target_h
            current_ar = crop_w / max(crop_h, 1)
            if current_ar < target_ar:
                crop_w = self._snap(int(round(crop_h * target_ar)), padding_multiple)
            else:
                crop_h = self._snap(int(round(crop_w / target_ar)), padding_multiple)

        # Apply downscale factor
        if downscale_factor < 1.0:
            target_w = max(padding_multiple, self._snap(int(target_w * downscale_factor), padding_multiple))
            target_h = max(padding_multiple, self._snap(int(target_h * downscale_factor), padding_multiple))

        # Step 4: Position crop centered on mask (allow extending beyond image)
        crop_x = int(round(center_x - crop_w / 2.0))
        crop_y = int(round(center_y - crop_h / 2.0))

        # Calculate canvas padding for out-of-bounds regions
        left_pad = max(0, -crop_x)
        top_pad = max(0, -crop_y)
        right_pad = max(0, (crop_x + crop_w) - W)
        bottom_pad = max(0, (crop_y + crop_h) - H)
        needs_expansion = left_pad > 0 or top_pad > 0 or right_pad > 0 or bottom_pad > 0

        if needs_expansion:
            # Build expanded canvas with edge-replicated borders (lquesada pattern)
            img_bchw = image.permute(0, 3, 1, 2)  # (B, C, H, W)
            canvas_bchw = F.pad(img_bchw,
                                (left_pad, right_pad, top_pad, bottom_pad),
                                mode="replicate")
            canvas_image = canvas_bchw.permute(0, 2, 3, 1)  # (B, exp_H, exp_W, C)

            # Mask canvas: expanded areas are 1.0 (masked)
            exp_H = H + top_pad + bottom_pad
            exp_W = W + left_pad + right_pad
            canvas_mask = torch.ones(B, exp_H, exp_W, device=device, dtype=mask.dtype)
            canvas_mask[:, top_pad:top_pad + H, left_pad:left_pad + W] = mask

            # Coordinates in canvas space
            cto_x, cto_y, cto_w, cto_h = left_pad, top_pad, W, H
            ctc_x = crop_x + left_pad
            ctc_y = crop_y + top_pad

            # Crop from canvas
            cropped = canvas_image[:, ctc_y:ctc_y + crop_h, ctc_x:ctc_x + crop_w, :].clone()
            cropped_mask = canvas_mask[:, ctc_y:ctc_y + crop_h, ctc_x:ctc_x + crop_w].clone()
        else:
            # No expansion needed — clamp and crop directly
            crop_x = max(0, min(crop_x, W - crop_w))
            crop_y = max(0, min(crop_y, H - crop_h))
            # Ensure crop doesn't exceed image
            crop_w = min(crop_w, W - crop_x)
            crop_h = min(crop_h, H - crop_y)

            canvas_image = image
            cto_x, cto_y, cto_w, cto_h = 0, 0, W, H
            ctc_x, ctc_y = crop_x, crop_y

            cropped = image[:, crop_y:crop_y + crop_h, crop_x:crop_x + crop_w, :].clone()
            cropped_mask = mask[:, crop_y:crop_y + crop_h, crop_x:crop_x + crop_w].clone()

        # Step 5: Fill masked area in crop
        if fill_masked_area == "neutral_gray":
            fill_value = 0.5
            binary_3d = (cropped_mask > 0.5).float().unsqueeze(-1)
            cropped = cropped * (1.0 - binary_3d) + fill_value * binary_3d
        elif fill_masked_area == "edge_pad":
            img_bchw_fill = cropped.permute(0, 3, 1, 2)
            blurred_fill = _gaussian_blur_2d(img_bchw_fill, sigma=max(crop_w, crop_h) * 0.15)
            blurred_fill = blurred_fill.permute(0, 2, 3, 1)
            binary_3d = (cropped_mask > 0.5).float().unsqueeze(-1)
            cropped = cropped * (1.0 - binary_3d) + blurred_fill * binary_3d

        # Step 6: Resize crop to target dimensions
        resize_mode = downscale_method if downscale_factor < 1.0 else "bilinear"
        if crop_w != target_w or crop_h != target_h:
            cropped = _resize_image(cropped, target_h, target_w, mode=resize_mode)
            cropped_mask = _resize_mask(cropped_mask, target_h, target_w, mode=resize_mode)

        # Step 7: Generate inpaint mask
        inpaint_mask = _apply_inpaint_mask_mode(cropped_mask, inpaint_mask_mode)

        # Step 8: Generate stitch blend mask
        stitch_blend_mask = _generate_stitch_blend_mask(
            cropped, cropped_mask, stitch_blend_mode, blend_radius
        )

        # Step 8.5: Generate cropped_composite (mask overlaid on cropped image as red tint)
        mask_rgb = inpaint_mask.unsqueeze(-1) * torch.tensor(
            [0.8, 0.15, 0.15], device=device, dtype=cropped.dtype
        ).view(1, 1, 1, 3)
        cropped_composite = (cropped * (1.0 - inpaint_mask.unsqueeze(-1) * 0.5) + mask_rgb * 0.5).clamp(0, 1)

        # Step 8.6: Generate crop_mask (binary mask showing cropped region in original image space)
        crop_mask = torch.zeros(B, H, W, device=device, dtype=torch.float32)
        # Map canvas crop back to original image space
        orig_y1 = max(0, ctc_y - cto_y)
        orig_y2 = min(H, ctc_y + crop_h - cto_y)
        orig_x1 = max(0, ctc_x - cto_x)
        orig_x2 = min(W, ctc_x + crop_w - cto_x)
        if orig_y2 > orig_y1 and orig_x2 > orig_x1:
            crop_mask[:, orig_y1:orig_y2, orig_x1:orig_x2] = 1.0

        # Step 9: Build stitch_data dict (v2 format with canvas coordinates)
        stitch_data = {
            "version": 2,
            "canvas_image": canvas_image.cpu(),
            "cto_x": cto_x, "cto_y": cto_y,
            "cto_w": cto_w, "cto_h": cto_h,
            "ctc_x": ctc_x, "ctc_y": ctc_y,
            "ctc_w": crop_w, "ctc_h": crop_h,
            "target_w": target_w, "target_h": target_h,
            "blend_mode": stitch_blend_mode,
            "blend_radius": blend_radius,
            "stitch_blend_mask_crop": stitch_blend_mask.cpu(),
            "original_mask": mask.cpu(),
            "downscale_factor": downscale_factor,
            "upscale_method": upscale_method,
            "downscale_method": downscale_method,
        }

        # Step 10: Build info string
        expansion_info = ""
        if needs_expansion:
            expansion_info = f"\n  canvas expansion: L={left_pad} T={top_pad} R={right_pad} B={bottom_pad}"
        info_lines = [
            f"InpaintCropProMEC:",
            f"  image: {B}x{H}x{W}x{C}",
            f"  mask bbox: x={bbox_x}, y={bbox_y}, w={bbox_w}, h={bbox_h}",
            f"  crop region: {crop_w}x{crop_h} centered at ({center_x:.0f},{center_y:.0f})",
            f"  context_expand: {context_expand:.2f}",
            f"  aspect_ratio: {aspect_ratio}",
            f"  downscale_factor: {downscale_factor:.2f}",
            f"  mask_blur: {mask_blur:.1f}, mask_grow: {mask_grow}",
            f"  output size: {target_w}x{target_h} (mode={size_mode})",
            f"  inpaint_mask_mode: {inpaint_mask_mode}",
            f"  stitch_blend_mode: {stitch_blend_mode}, radius={blend_radius}",
            f"  video_stable_crop: {video_stable_crop}",
            f"  fill_masked_area: {fill_masked_area}",
            f"  inpaint_mask range: [{inpaint_mask.min().item():.4f}, {inpaint_mask.max().item():.4f}]",
            f"  blend_mask range: [{stitch_blend_mask.min().item():.4f}, {stitch_blend_mask.max().item():.4f}]",
            expansion_info,
        ]
        info = "\n".join(info_lines)

        return (stitch_data, cropped, cropped_composite, inpaint_mask, stitch_blend_mask, crop_mask, info)


# ══════════════════════════════════════════════════════════════════════
#  NODE 2: InpaintStitchProMEC
# ══════════════════════════════════════════════════════════════════════

class InpaintStitchProMEC:
    """Composite inpainted image back into original using stitch_data.

    Supports blend mode override and optional color matching.
    """

    VRAM_TIER = 1
    BLEND_OVERRIDES = ["from_crop", "edge_aware", "gaussian", "laplacian_pyramid", "frequency_blend"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stitch_data": ("STITCH_DATA", {"tooltip": "Stitch data from InpaintCropProMEC"}),
                "inpainted_image": ("IMAGE", {"tooltip": "Inpainted result from model (B,H,W,C)"}),
                "blend_mode_override": (cls.BLEND_OVERRIDES, {
                    "default": "from_crop",
                    "tooltip": "Override the blend mode from crop node, or use 'from_crop' to keep original"}),
                "color_match": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Apply mean+std color transfer before stitching to reduce color shift"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "blend_mask_used", "info")
    FUNCTION = "stitch"
    CATEGORY = "MaskEditControl/Inpaint"
    DESCRIPTION = "Composite inpainted image back using stitch_data. Supports edge-aware, Laplacian pyramid, and frequency domain blending."

    def stitch(self, stitch_data: Dict[str, Any], inpainted_image: torch.Tensor,
               blend_mode_override: str, color_match: bool):
        version = stitch_data.get("version", 1)
        if version >= 2:
            return self._stitch_v2(stitch_data, inpainted_image,
                                   blend_mode_override, color_match)
        return self._stitch_v1(stitch_data, inpainted_image,
                               blend_mode_override, color_match)

    def _stitch_v2(self, stitch_data: Dict[str, Any], inpainted_image: torch.Tensor,
                   blend_mode_override: str, color_match: bool):
        """V2 stitch: uses canvas coordinates for perfect reversal of canvas expansion."""
        canvas_image = stitch_data["canvas_image"]
        ctc_x = stitch_data["ctc_x"]
        ctc_y = stitch_data["ctc_y"]
        ctc_w = stitch_data["ctc_w"]
        ctc_h = stitch_data["ctc_h"]
        cto_x = stitch_data["cto_x"]
        cto_y = stitch_data["cto_y"]
        cto_w = stitch_data["cto_w"]
        cto_h = stitch_data["cto_h"]
        stored_blend_mode = stitch_data["blend_mode"]
        blend_radius = stitch_data["blend_radius"]
        stitch_blend_mask_crop = stitch_data["stitch_blend_mask_crop"]
        upscale_method = stitch_data.get("upscale_method", "lanczos")

        device = _get_device(inpainted_image)
        canvas = canvas_image.to(device).clone()
        B = inpainted_image.shape[0]

        blend_mode = stored_blend_mode if blend_mode_override == "from_crop" else blend_mode_override

        # Resize inpainted to crop region dimensions in canvas
        inp_resized = _resize_image(inpainted_image, ctc_h, ctc_w, mode=upscale_method)

        # Resize blend mask to crop region dimensions
        blend_mask_crop = _resize_mask(stitch_blend_mask_crop.to(device), ctc_h, ctc_w, mode=upscale_method)

        # Regenerate blend mask if overridden
        if blend_mode_override != "from_crop" and blend_mode_override != stored_blend_mode:
            canvas_crop_region = canvas[:, ctc_y:ctc_y + ctc_h, ctc_x:ctc_x + ctc_w, :]
            original_mask = stitch_data.get("original_mask")
            if original_mask is not None:
                crop_mask = _resize_mask(original_mask.to(device), ctc_h, ctc_w)
            else:
                crop_mask = (blend_mask_crop > 0.5).float()
            blend_mask_crop = _generate_stitch_blend_mask(
                canvas_crop_region, crop_mask, blend_mode, blend_radius
            )

        # Optional color matching against canvas context
        if color_match:
            canvas_context = canvas[:, ctc_y:ctc_y + ctc_h, ctc_x:ctc_x + ctc_w, :].clone()
            color_mask = (blend_mask_crop > 0.1).float()
            inp_resized = _color_match_mean_std(inp_resized, canvas_context, color_mask)

        # Composite: blend inpainted onto canvas at crop position
        canvas_crop = canvas[:, ctc_y:ctc_y + ctc_h, ctc_x:ctc_x + ctc_w, :].clone()

        if blend_mode == "laplacian_pyramid":
            a_bchw = canvas_crop.permute(0, 3, 1, 2)
            b_bchw = inp_resized.permute(0, 3, 1, 2)
            m_b1hw = blend_mask_crop.unsqueeze(1)
            blended = _laplacian_pyramid_blend(a_bchw, b_bchw, m_b1hw, levels=5)
            blended = blended.permute(0, 2, 3, 1).clamp(0.0, 1.0)
        elif blend_mode == "frequency_blend":
            a_bchw = canvas_crop.permute(0, 3, 1, 2)
            b_bchw = inp_resized.permute(0, 3, 1, 2)
            m_b1hw = blend_mask_crop.unsqueeze(1)
            blended = _frequency_blend(a_bchw, b_bchw, m_b1hw)
            blended = blended.permute(0, 2, 3, 1).clamp(0.0, 1.0)
        else:
            m3 = blend_mask_crop.unsqueeze(-1)
            blended = canvas_crop * (1.0 - m3) + inp_resized * m3

        canvas[:, ctc_y:ctc_y + ctc_h, ctc_x:ctc_x + ctc_w, :] = blended

        # Extract original image region from canvas (reverses expansion)
        output = canvas[:, cto_y:cto_y + cto_h, cto_x:cto_x + cto_w, :].clamp(0.0, 1.0)

        # Build blend mask in original image space
        H, W = cto_h, cto_w
        blend_canvas = torch.zeros(B, canvas.shape[1], canvas.shape[2],
                                   device=device, dtype=canvas.dtype)
        blend_canvas[:, ctc_y:ctc_y + ctc_h, ctc_x:ctc_x + ctc_w] = blend_mask_crop
        blend_mask_full = blend_canvas[:, cto_y:cto_y + cto_h, cto_x:cto_x + cto_w]

        color_info = "\n  color_match: applied" if color_match else ""
        info = (
            f"InpaintStitchProMEC (v2):\n"
            f"  canvas: {canvas_image.shape[1]}x{canvas_image.shape[2]}\n"
            f"  crop→canvas: x={ctc_x}, y={ctc_y}, w={ctc_w}, h={ctc_h}\n"
            f"  orig→canvas: x={cto_x}, y={cto_y}, w={cto_w}, h={cto_h}\n"
            f"  blend_mode: {blend_mode}"
            + (" (overridden)" if blend_mode_override != "from_crop" else " (from_crop)")
            + f"\n  output: {B}x{H}x{W}"
            + color_info
        )

        return (output, blend_mask_full, info)

    def _stitch_v1(self, stitch_data: Dict[str, Any], inpainted_image: torch.Tensor,
                   blend_mode_override: str, color_match: bool):
        """V1 stitch: legacy format with direct crop coordinates."""
        original_image = stitch_data["original_image"]
        crop_x = stitch_data["crop_x"]
        crop_y = stitch_data["crop_y"]
        crop_w = stitch_data["crop_w"]
        crop_h = stitch_data["crop_h"]
        stored_blend_mode = stitch_data["blend_mode"]
        blend_radius = stitch_data["blend_radius"]
        stitch_blend_mask_crop = stitch_data["stitch_blend_mask_crop"]
        original_mask = stitch_data["original_mask"]
        upscale_method = stitch_data.get("upscale_method", "lanczos")

        device = _get_device(original_image)
        B, H, W, C = original_image.shape

        blend_mode = stored_blend_mode if blend_mode_override == "from_crop" else blend_mode_override

        inp_resized = _resize_image(inpainted_image, crop_h, crop_w, mode=upscale_method)
        blend_mask_crop = _resize_mask(stitch_blend_mask_crop, crop_h, crop_w, mode=upscale_method)

        if blend_mode_override != "from_crop" and blend_mode_override != stored_blend_mode:
            crop_region = original_image[:, crop_y:crop_y + crop_h, crop_x:crop_x + crop_w, :]
            crop_mask = original_mask[:, crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
            blend_mask_crop = _generate_stitch_blend_mask(
                crop_region, crop_mask, blend_mode, blend_radius
            )

        if color_match:
            crop_original = original_image[:, crop_y:crop_y + crop_h, crop_x:crop_x + crop_w, :].clone()
            crop_mask_for_color = original_mask[:, crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
            inp_resized = _color_match_mean_std(inp_resized, crop_original, crop_mask_for_color)

        canvas = original_image.clone()
        canvas_crop = canvas[:, crop_y:crop_y + crop_h, crop_x:crop_x + crop_w, :].clone()

        if blend_mode == "laplacian_pyramid":
            a_bchw = canvas_crop.permute(0, 3, 1, 2)
            b_bchw = inp_resized.permute(0, 3, 1, 2)
            m_b1hw = blend_mask_crop.unsqueeze(1)
            blended_bchw = _laplacian_pyramid_blend(a_bchw, b_bchw, m_b1hw, levels=5)
            blended = blended_bchw.permute(0, 2, 3, 1).clamp(0.0, 1.0)
        elif blend_mode == "frequency_blend":
            a_bchw = canvas_crop.permute(0, 3, 1, 2)
            b_bchw = inp_resized.permute(0, 3, 1, 2)
            m_b1hw = blend_mask_crop.unsqueeze(1)
            blended_bchw = _frequency_blend(a_bchw, b_bchw, m_b1hw)
            blended = blended_bchw.permute(0, 2, 3, 1).clamp(0.0, 1.0)
        else:
            m3 = blend_mask_crop.unsqueeze(-1)
            blended = canvas_crop * (1.0 - m3) + inp_resized * m3

        canvas[:, crop_y:crop_y + crop_h, crop_x:crop_x + crop_w, :] = blended
        blend_mask_full = torch.zeros(B, H, W, device=device, dtype=original_image.dtype)
        blend_mask_full[:, crop_y:crop_y + crop_h, crop_x:crop_x + crop_w] = blend_mask_crop

        color_info = "\n  color_match: applied" if color_match else ""
        info = (
            f"InpaintStitchProMEC (v1):\n"
            f"  stitch region: x={crop_x}, y={crop_y}, w={crop_w}, h={crop_h}\n"
            f"  blend_mode: {blend_mode}\n"
            f"  output: {B}x{H}x{W}x{C}"
            + color_info
        )

        return (canvas, blend_mask_full, info)


# ══════════════════════════════════════════════════════════════════════
#  NODE 3: InpaintMaskPrepareMEC
# ══════════════════════════════════════════════════════════════════════

class InpaintMaskPrepareMEC:
    """Standalone mask preparation: clean up, grow, produce dual inpaint + stitch masks.

    Separates inpaint_mask (what model sees) from stitch_blend_mask (what composite uses).
    Optional temporal smoothing for video batch consistency.
    """

    VRAM_TIER = 1
    INPAINT_EDGE_MODES = ["hard_binary", "slight_feather"]
    STITCH_EDGE_MODES = ["gaussian", "edge_aware"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK", {"tooltip": "Raw input mask (B,H,W)"}),
                "fill_holes": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Fill interior holes in the mask"}),
                "remove_small_regions": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Remove small disconnected blobs"}),
                "min_region_area": ("INT", {
                    "default": 100, "min": 0, "max": 100000, "step": 10,
                    "tooltip": "Minimum region area in pixels to keep"}),
                "grow_pixels": ("INT", {
                    "default": 4, "min": 0, "max": 256, "step": 1,
                    "tooltip": "Dilate mask by this many pixels"}),
                "inpaint_edge_mode": (cls.INPAINT_EDGE_MODES, {
                    "default": "hard_binary",
                    "tooltip": "Edge style for inpaint mask: hard_binary or slight_feather"}),
                "stitch_edge_mode": (cls.STITCH_EDGE_MODES, {
                    "default": "gaussian",
                    "tooltip": "Edge style for stitch blend mask: gaussian or edge_aware"}),
                "stitch_feather_radius": ("INT", {
                    "default": 16, "min": 1, "max": 128, "step": 1,
                    "tooltip": "Feather radius for the stitch blend mask"}),
                "temporal_smooth": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Apply Gaussian temporal smoothing along batch dimension (for video)"}),
                "temporal_sigma": ("FLOAT", {
                    "default": 1.5, "min": 0.1, "max": 10.0, "step": 0.1,
                    "tooltip": "Temporal Gaussian sigma in frames"}),
            },
            "optional": {
                "reference_image": ("IMAGE", {
                    "tooltip": "Reference image for edge-aware stitch blend mask (required for edge_aware mode)"}),
            },
        }

    RETURN_TYPES = ("MASK", "MASK", "IMAGE", "STRING")
    RETURN_NAMES = ("inpaint_mask", "stitch_blend_mask", "debug_preview", "info")
    FUNCTION = "prepare_mask"
    CATEGORY = "MaskEditControl/Inpaint"
    DESCRIPTION = "Clean, grow, and prepare dual masks: inpaint_mask for model + stitch_blend_mask for composite."

    def prepare_mask(self, mask: torch.Tensor, fill_holes: bool,
                     remove_small_regions: bool, min_region_area: int,
                     grow_pixels: int, inpaint_edge_mode: str,
                     stitch_edge_mode: str, stitch_feather_radius: int,
                     temporal_smooth: bool, temporal_sigma: float,
                     reference_image: Optional[torch.Tensor] = None):
        device = _get_device(mask)

        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        B, H, W = mask.shape
        working = mask.clone()

        holes_filled = 0
        regions_removed = 0

        # Step 1: Fill holes
        if fill_holes:
            before_sum = (working > 0.5).float().sum().item()
            working = _fill_holes_torch(working)
            after_sum = (working > 0.5).float().sum().item()
            holes_filled = int(after_sum - before_sum)

        # Step 2: Remove small regions
        if remove_small_regions and min_region_area > 0:
            before_sum = (working > 0.5).float().sum().item()
            working = _remove_small_regions_torch(working, min_region_area)
            after_sum = (working > 0.5).float().sum().item()
            regions_removed = int(before_sum - after_sum)

        # Step 3: Grow mask
        if grow_pixels > 0:
            k = 2 * grow_pixels + 1
            m4 = working.unsqueeze(1)
            working = F.max_pool2d(m4, kernel_size=k, stride=1, padding=grow_pixels).squeeze(1)
            working = working.clamp(0.0, 1.0)

        # Step 4: Generate inpaint mask
        inpaint_mask = _apply_inpaint_mask_mode(working, inpaint_edge_mode)

        # Step 5: Generate stitch blend mask
        if stitch_edge_mode == "edge_aware" and reference_image is not None:
            ref = reference_image
            if ref.shape[0] == 1 and B > 1:
                ref = ref.expand(B, -1, -1, -1)
            if ref.shape[1] != H or ref.shape[2] != W:
                ref = _resize_image(ref, H, W)
            stitch_blend_mask = _edge_aware_blend_mask(ref, working, stitch_feather_radius)
        else:
            binary = (working > 0.5).float()
            stitch_blend_mask = _gaussian_blur_mask(binary, sigma=stitch_feather_radius * 0.4)

        # Step 6: Temporal smoothing
        temporal_variance_before = 0.0
        temporal_variance_after = 0.0
        if temporal_smooth and B > 1:
            temporal_variance_before = stitch_blend_mask.var(dim=0).mean().item()
            stitch_blend_mask = _temporal_gaussian_smooth(stitch_blend_mask, temporal_sigma)
            temporal_variance_after = stitch_blend_mask.var(dim=0).mean().item()

        # Step 7: Build debug preview
        debug_r = inpaint_mask.unsqueeze(-1)
        debug_g = stitch_blend_mask.unsqueeze(-1)
        debug_b = torch.zeros(B, H, W, 1, device=device, dtype=mask.dtype)
        debug_preview = torch.cat([debug_r, debug_g, debug_b], dim=-1)

        # Step 8: Build info string
        info_lines = [
            f"InpaintMaskPrepareMEC:",
            f"  input: {B}x{H}x{W}",
            f"  fill_holes: {fill_holes} (pixels filled: {holes_filled})",
            f"  remove_small_regions: {remove_small_regions} (pixels removed: {regions_removed})",
            f"  grow_pixels: {grow_pixels}",
            f"  inpaint_edge_mode: {inpaint_edge_mode}",
            f"  stitch_edge_mode: {stitch_edge_mode}, radius={stitch_feather_radius}",
            f"  inpaint_mask range: [{inpaint_mask.min().item():.4f}, {inpaint_mask.max().item():.4f}]",
            f"  stitch_blend_mask range: [{stitch_blend_mask.min().item():.4f}, {stitch_blend_mask.max().item():.4f}]",
        ]
        if temporal_smooth and B > 1:
            info_lines.append(
                f"  temporal_smooth: sigma={temporal_sigma:.1f}, "
                f"variance {temporal_variance_before:.6f} → {temporal_variance_after:.6f}"
            )
        info = "\n".join(info_lines)

        return (inpaint_mask, stitch_blend_mask, debug_preview, info)


# ══════════════════════════════════════════════════════════════════════
#  NODE 4: InpaintPasteBackMEC
# ══════════════════════════════════════════════════════════════════════

class InpaintPasteBackMEC:
    """Paste inpainted crop back onto original using stitch_data.

    Simplified companion to InpaintStitchProMEC — no blend mode logic,
    just a clean resize + paste with optional feathered alpha composite.
    Accepts stitch_data from InpaintCropProMEC or a manual CROP_INFO dict.
    """

    VRAM_TIER = 1
    INTERP_METHODS = ["lanczos", "bicubic", "bilinear", "nearest-exact", "area"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stitch_data": ("STITCH_DATA", {"tooltip": "Stitch data from InpaintCropProMEC"}),
                "inpainted_image": ("IMAGE", {"tooltip": "Inpainted crop result (B,H,W,C)"}),
                "upscale_method": (cls.INTERP_METHODS, {
                    "default": "lanczos",
                    "tooltip": "Interpolation method for resizing crop back to original dimensions"}),
                "feather_edges": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Apply Gaussian feather at crop boundary for seamless paste"}),
                "feather_radius": ("INT", {
                    "default": 16, "min": 0, "max": 64, "step": 1,
                    "tooltip": "Feather radius in pixels (only used if feather_edges is True)"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "info")
    FUNCTION = "paste_back"
    CATEGORY = "MaskEditControl/Inpaint"
    DESCRIPTION = "Paste inpainted crop back onto original image using stitch_data with optional feathered edges."

    def paste_back(self, stitch_data: Dict[str, Any], inpainted_image: torch.Tensor,
                   upscale_method: str, feather_edges: bool, feather_radius: int):
        version = stitch_data.get("version", 1)
        device = _get_device(inpainted_image)

        if version >= 2:
            canvas_image = stitch_data["canvas_image"].to(device)
            ctc_x = stitch_data["ctc_x"]
            ctc_y = stitch_data["ctc_y"]
            ctc_w = stitch_data["ctc_w"]
            ctc_h = stitch_data["ctc_h"]
            cto_x = stitch_data["cto_x"]
            cto_y = stitch_data["cto_y"]
            cto_w = stitch_data["cto_w"]
            cto_h = stitch_data["cto_h"]

            B = inpainted_image.shape[0]
            inp_resized = _resize_image(inpainted_image, ctc_h, ctc_w, mode=upscale_method)
            canvas = canvas_image.clone()

            if feather_edges and feather_radius > 0:
                paste_mask = torch.ones(1, 1, ctc_h, ctc_w, device=device, dtype=torch.float32)
                paste_mask = _gaussian_blur_2d(paste_mask, sigma=feather_radius)
                pm3 = paste_mask.squeeze(0).squeeze(0).unsqueeze(-1)
                for b in range(B):
                    crop = canvas[b, ctc_y:ctc_y + ctc_h, ctc_x:ctc_x + ctc_w, :]
                    canvas[b, ctc_y:ctc_y + ctc_h, ctc_x:ctc_x + ctc_w, :] = (
                        crop * (1.0 - pm3) + inp_resized[b] * pm3
                    )
            else:
                canvas[:, ctc_y:ctc_y + ctc_h, ctc_x:ctc_x + ctc_w, :] = inp_resized

            output = canvas[:, cto_y:cto_y + cto_h, cto_x:cto_x + cto_w, :]
            info = (
                f"InpaintPasteBackMEC (v2):\n"
                f"  crop→canvas: x={ctc_x}, y={ctc_y}, w={ctc_w}, h={ctc_h}\n"
                f"  output: {B}x{cto_h}x{cto_w}\n"
                f"  upscale_method: {upscale_method}\n"
                f"  feather_edges: {feather_edges}, radius={feather_radius}"
            )
            return (output.clamp(0, 1), info)

        # V1 fallback
        original_image = stitch_data["original_image"]
        crop_x = stitch_data["crop_x"]
        crop_y = stitch_data["crop_y"]
        crop_w = stitch_data["crop_w"]
        crop_h = stitch_data["crop_h"]

        B, H, W, C = original_image.shape
        inp_resized = _resize_image(inpainted_image, crop_h, crop_w, mode=upscale_method)
        canvas = original_image.clone()

        if feather_edges and feather_radius > 0:
            paste_mask = torch.ones(1, 1, crop_h, crop_w, device=device, dtype=torch.float32)
            paste_mask = _gaussian_blur_2d(paste_mask, sigma=feather_radius)
            paste_mask_3d = paste_mask.squeeze(0).squeeze(0).unsqueeze(-1)
            for b in range(B):
                orig_crop = canvas[b, crop_y:crop_y + crop_h, crop_x:crop_x + crop_w, :]
                canvas[b, crop_y:crop_y + crop_h, crop_x:crop_x + crop_w, :] = (
                    orig_crop * (1.0 - paste_mask_3d) + inp_resized[b] * paste_mask_3d
                )
        else:
            canvas[:, crop_y:crop_y + crop_h, crop_x:crop_x + crop_w, :] = inp_resized

        info = (
            f"InpaintPasteBackMEC (v1):\n"
            f"  paste region: x={crop_x}, y={crop_y}, w={crop_w}, h={crop_h}\n"
            f"  upscale_method: {upscale_method}\n"
            f"  feather_edges: {feather_edges}, radius={feather_radius}\n"
            f"  output: {B}x{H}x{W}x{C}"
        )

        return (canvas.clamp(0, 1), info)
