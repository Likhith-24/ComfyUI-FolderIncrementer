"""
MaskFailureExplainerMEC – Diagnose why a mask failed and suggest fixes.

Input: image (B,H,W,C), mask (B,H,W) of unknown quality.
Runs a pure-tensor analysis pipeline:
  1. Brightness: mean luminance per frame. <0.15 = dark scene.
  2. Blur: Laplacian variance of image. <50 = blurry.
  3. Contrast at boundary: std of image pixels at mask edge ring. <0.05 = similar.
  4. Boundary color confusion: mean color distance inside vs outside mask boundary. <0.1 = too similar.
  5. Background complexity: edge density outside mask region. >0.3 = busy background.

Outputs:
  - explanation STRING: real computed values per-frame, specific actionable advice
  - problem_regions_mask MASK: heatmap of detected issues (not zeros)
  - severity_score FLOAT: 0-100 computed from metrics
  - suggested_method STRING: based on which conditions triggered

No models. Pure tensor math. VRAM Tier 1.
"""

from __future__ import annotations

import gc
import torch
import torch.nn.functional as F

# ── Optional cv2 with torch fallback ──────────────────────────────────
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# ── Device helper ─────────────────────────────────────────────────────

def _get_device(tensor: torch.Tensor) -> torch.device:
    """Return the device of the tensor — never hardcode 'cuda'."""
    return tensor.device


# ── Laplacian kernel (3x3 standard) ──────────────────────────────────

_LAPLACIAN_KERNEL = torch.tensor(
    [[0.0, 1.0, 0.0],
     [1.0, -4.0, 1.0],
     [0.0, 1.0, 0.0]], dtype=torch.float32
).unsqueeze(0).unsqueeze(0)  # (1,1,3,3)

# ── Sobel kernels ────────────────────────────────────────────────────

_SOBEL_X = torch.tensor(
    [[-1.0, 0.0, 1.0],
     [-2.0, 0.0, 2.0],
     [-1.0, 0.0, 1.0]], dtype=torch.float32
).unsqueeze(0).unsqueeze(0)

_SOBEL_Y = torch.tensor(
    [[-1.0, -2.0, -1.0],
     [0.0,  0.0,  0.0],
     [1.0,  2.0,  1.0]], dtype=torch.float32
).unsqueeze(0).unsqueeze(0)


# ══════════════════════════════════════════════════════════════════════
#  Analysis functions — pure torch, batch-aware
# ══════════════════════════════════════════════════════════════════════

def _compute_luminance(image: torch.Tensor) -> torch.Tensor:
    """BT.709 luminance from (B,H,W,C) image → (B,H,W)."""
    return 0.2126 * image[:, :, :, 0] + 0.7152 * image[:, :, :, 1] + 0.0722 * image[:, :, :, 2]


def _compute_brightness(image: torch.Tensor) -> torch.Tensor:
    """Per-frame mean brightness. Returns (B,) tensor."""
    luma = _compute_luminance(image)  # (B,H,W)
    return luma.mean(dim=(-2, -1))  # (B,)


def _compute_blur_score_torch(gray: torch.Tensor) -> torch.Tensor:
    """Laplacian variance per frame via conv2d. gray: (B,H,W) → (B,) scores.

    Convention: higher = sharper. Multiply by 1000 so threshold ~50 is meaningful.
    """
    device = _get_device(gray)
    B, H, W = gray.shape
    kernel = _LAPLACIAN_KERNEL.to(device=device, dtype=gray.dtype)
    # (B,1,H,W) for conv2d
    inp = gray.unsqueeze(1)
    lap = F.conv2d(inp, kernel, padding=1)  # (B,1,H,W)
    lap = lap.squeeze(1)  # (B,H,W)
    # Variance of Laplacian per frame
    var_per_frame = lap.var(dim=(-2, -1))  # (B,)
    return var_per_frame * 1000.0


def _compute_blur_score_cv2(gray_np):
    """Laplacian variance via cv2 for a single HxW numpy array. Returns float."""
    lap = cv2.Laplacian(gray_np, cv2.CV_64F)
    return float(lap.var()) * 1000.0


def _compute_blur_score(image: torch.Tensor) -> torch.Tensor:
    """Blur score per frame. Returns (B,) tensor. Uses cv2 if available, else torch."""
    luma = _compute_luminance(image)  # (B,H,W)
    if HAS_CV2:
        import numpy as np
        scores = []
        for i in range(luma.shape[0]):
            gray_np = luma[i].cpu().numpy().astype(np.float64)
            scores.append(_compute_blur_score_cv2(gray_np))
        return torch.tensor(scores, device=_get_device(image), dtype=image.dtype)
    else:
        return _compute_blur_score_torch(luma)


def _get_mask_edge_ring(mask: torch.Tensor, ring_width: int = 5) -> torch.Tensor:
    """Compute a binary edge ring around the mask boundary.

    mask: (B,H,W) → returns (B,H,W) binary ring.
    Uses morphological dilation minus erosion via max_pool2d.
    """
    B, H, W = mask.shape
    binary = (mask > 0.5).float().unsqueeze(1)  # (B,1,H,W)

    pad = ring_width
    # Dilation via max_pool
    dilated = F.max_pool2d(
        binary, kernel_size=2 * pad + 1, stride=1, padding=pad
    )
    # Erosion via -max_pool(-x)
    eroded = -F.max_pool2d(
        -binary, kernel_size=2 * pad + 1, stride=1, padding=pad
    )
    ring = (dilated - eroded).squeeze(1).clamp(0.0, 1.0)  # (B,H,W)
    return ring


def _compute_boundary_contrast(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Std of image pixels at mask edge ring, per frame. Returns (B,)."""
    ring = _get_mask_edge_ring(mask)  # (B,H,W)
    luma = _compute_luminance(image)  # (B,H,W)
    B = image.shape[0]
    results = []
    for i in range(B):
        ring_pixels = luma[i][ring[i] > 0.5]
        if ring_pixels.numel() < 2:
            results.append(0.0)
        else:
            results.append(ring_pixels.std().item())
    return torch.tensor(results, device=_get_device(image), dtype=image.dtype)


def _compute_boundary_color_confusion(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean color distance between inside and outside mask at boundary. Returns (B,).

    At the mask boundary ring, compare mean color on the mask side vs bg side.
    """
    ring = _get_mask_edge_ring(mask)  # (B,H,W)
    binary = (mask > 0.5).float()
    B = image.shape[0]
    results = []
    for i in range(B):
        ring_mask = ring[i] > 0.5
        inside = ring_mask & (binary[i] > 0.5)
        outside = ring_mask & (binary[i] <= 0.5)
        if inside.sum() < 1 or outside.sum() < 1:
            results.append(0.0)
            continue
        # Mean color inside and outside the ring
        color_inside = image[i][inside].mean(dim=0)   # (C,)
        color_outside = image[i][outside].mean(dim=0)  # (C,)
        dist = (color_inside - color_outside).abs().mean().item()
        results.append(dist)
    return torch.tensor(results, device=_get_device(image), dtype=image.dtype)


def _compute_bg_complexity_torch(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Edge density in background region via Sobel. Returns (B,) in [0,1]."""
    device = _get_device(image)
    luma = _compute_luminance(image)  # (B,H,W)
    B, H, W = luma.shape
    sx = _SOBEL_X.to(device=device, dtype=luma.dtype)
    sy = _SOBEL_Y.to(device=device, dtype=luma.dtype)
    inp = luma.unsqueeze(1)  # (B,1,H,W)
    gx = F.conv2d(inp, sx, padding=1).squeeze(1)  # (B,H,W)
    gy = F.conv2d(inp, sy, padding=1).squeeze(1)  # (B,H,W)
    edges = (gx.pow(2) + gy.pow(2)).sqrt()  # (B,H,W)
    # Threshold edges at 0.1 to get binary edge map
    edge_binary = (edges > 0.1).float()

    bg_mask = (mask <= 0.5).float()  # (B,H,W)
    results = []
    for i in range(B):
        bg_pixels = bg_mask[i].sum().item()
        if bg_pixels < 1:
            results.append(0.0)
        else:
            edge_in_bg = (edge_binary[i] * bg_mask[i]).sum().item()
            results.append(edge_in_bg / bg_pixels)
    return torch.tensor(results, device=device, dtype=image.dtype)


def _compute_bg_complexity_cv2(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Edge density in background region via cv2 Canny. Returns (B,)."""
    import numpy as np
    luma = _compute_luminance(image)  # (B,H,W)
    bg_mask = (mask <= 0.5).float()
    B = image.shape[0]
    results = []
    for i in range(B):
        gray_np = (luma[i].cpu().numpy() * 255).astype(np.uint8)
        edges = cv2.Canny(gray_np, 50, 150)
        edge_binary = (edges > 0).astype(np.float32)
        bg_np = bg_mask[i].cpu().numpy()
        bg_pixels = bg_np.sum()
        if bg_pixels < 1:
            results.append(0.0)
        else:
            results.append(float((edge_binary * bg_np).sum() / bg_pixels))
    return torch.tensor(results, device=_get_device(image), dtype=image.dtype)


def _compute_bg_complexity(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Background complexity (edge density outside mask). Returns (B,)."""
    if HAS_CV2:
        return _compute_bg_complexity_cv2(image, mask)
    else:
        return _compute_bg_complexity_torch(image, mask)


# ══════════════════════════════════════════════════════════════════════
#  Problem regions heatmap
# ══════════════════════════════════════════════════════════════════════

def _build_problem_heatmap(
    image: torch.Tensor,
    mask: torch.Tensor,
    brightness: torch.Tensor,
    blur: torch.Tensor,
    boundary_contrast: torch.Tensor,
    color_confusion: torch.Tensor,
    bg_complexity: torch.Tensor,
) -> torch.Tensor:
    """Build a (B,H,W) heatmap highlighting problematic regions.

    Combines:
      - Low brightness regions → heatmap where image is dark
      - Blurry regions → high-frequency deficit areas
      - Boundary zone → where contrast/color confusion is bad
      - Complex bg → edge-dense background areas
    """
    B, H, W, C = image.shape
    device = _get_device(image)
    heatmap = torch.zeros(B, H, W, device=device, dtype=image.dtype)

    luma = _compute_luminance(image)  # (B,H,W)
    ring = _get_mask_edge_ring(mask)  # (B,H,W)
    bg_mask = (mask <= 0.5).float()

    for i in range(B):
        frame_heat = torch.zeros(H, W, device=device, dtype=image.dtype)

        # Dark regions contribute where brightness is low
        if brightness[i].item() < 0.15:
            dark_map = (1.0 - luma[i]).clamp(0.0, 1.0)
            frame_heat = frame_heat + dark_map * 0.3

        # Boundary problems: highlight ring where contrast is low
        if boundary_contrast[i].item() < 0.05 or color_confusion[i].item() < 0.1:
            frame_heat = frame_heat + ring[i] * 0.4

        # Background complexity: highlight edges in bg
        if bg_complexity[i].item() > 0.3:
            # Compute edge map for this frame
            sx = _SOBEL_X.to(device=device, dtype=image.dtype)
            sy = _SOBEL_Y.to(device=device, dtype=image.dtype)
            inp = luma[i].unsqueeze(0).unsqueeze(0)
            gx = F.conv2d(inp, sx, padding=1).squeeze()
            gy = F.conv2d(inp, sy, padding=1).squeeze()
            edges = (gx.pow(2) + gy.pow(2)).sqrt()
            frame_heat = frame_heat + edges * bg_mask[i] * 0.3

        # If blur is bad, add a uniform low-level heat (blur is global)
        if blur[i].item() < 50.0:
            frame_heat = frame_heat + 0.15

        # Ensure some signal even if no issues detected (baseline from mask edge)
        frame_heat = frame_heat + ring[i] * 0.05

        heatmap[i] = frame_heat

    return heatmap.clamp(0.0, 1.0)


# ══════════════════════════════════════════════════════════════════════
#  Severity scoring
# ══════════════════════════════════════════════════════════════════════

def _compute_severity(
    brightness: torch.Tensor,
    blur: torch.Tensor,
    boundary_contrast: torch.Tensor,
    color_confusion: torch.Tensor,
    bg_complexity: torch.Tensor,
) -> float:
    """Compute a 0-100 severity score (mean across batch).

    Each metric contributes 0-20 points of severity:
      - Brightness: 20 * (1 - clamp(mean_brightness / 0.15, 0, 1)) when dark
      - Blur: 20 * (1 - clamp(mean_blur / 50, 0, 1))
      - Boundary contrast: 20 * (1 - clamp(mean_contrast / 0.05, 0, 1))
      - Color confusion: 20 * (1 - clamp(mean_confusion / 0.1, 0, 1))
      - BG complexity: 20 * clamp(mean_bg_complexity / 0.3, 0, 1)
    """
    # Average across batch
    b = brightness.mean().item()
    bl = blur.mean().item()
    bc = boundary_contrast.mean().item()
    cc = color_confusion.mean().item()
    bg = bg_complexity.mean().item()

    score = 0.0
    # Dark scene penalty (only if dark)
    score += 20.0 * max(0.0, 1.0 - min(b / 0.15, 1.0))
    # Blur penalty
    score += 20.0 * max(0.0, 1.0 - min(bl / 50.0, 1.0))
    # Low boundary contrast penalty
    score += 20.0 * max(0.0, 1.0 - min(bc / 0.05, 1.0))
    # Color confusion penalty
    score += 20.0 * max(0.0, 1.0 - min(cc / 0.1, 1.0))
    # Background complexity penalty
    score += 20.0 * min(bg / 0.3, 1.0)

    return round(min(max(score, 0.0), 100.0), 2)


# ══════════════════════════════════════════════════════════════════════
#  Explanation + suggested method
# ══════════════════════════════════════════════════════════════════════

_THRESHOLD_DARK = 0.15
_THRESHOLD_BLUR = 50.0
_THRESHOLD_CONTRAST = 0.05
_THRESHOLD_COLOR = 0.1
_THRESHOLD_BG = 0.3


def _build_explanation(
    brightness: torch.Tensor,
    blur: torch.Tensor,
    boundary_contrast: torch.Tensor,
    color_confusion: torch.Tensor,
    bg_complexity: torch.Tensor,
    severity: float,
    B: int, H: int, W: int,
) -> str:
    """Build a detailed, per-frame explanation string with actionable advice."""
    lines = []
    lines.append(f"[MEC] Mask Failure Analysis — {B} frame(s), {H}x{W}")
    lines.append(f"Overall severity: {severity:.1f}/100")
    lines.append("")

    issues_found = []

    for i in range(B):
        frame_prefix = f"Frame {i}" if B > 1 else "Image"
        frame_issues = []

        b = brightness[i].item()
        bl = blur[i].item()
        bc = boundary_contrast[i].item()
        cc = color_confusion[i].item()
        bg = bg_complexity[i].item()

        lines.append(f"--- {frame_prefix} ---")
        lines.append(f"  Brightness:         {b:.4f}" + (" ⚠ DARK SCENE" if b < _THRESHOLD_DARK else " ✓"))
        lines.append(f"  Blur score:         {bl:.2f}" + (" ⚠ BLURRY" if bl < _THRESHOLD_BLUR else " ✓"))
        lines.append(f"  Boundary contrast:  {bc:.4f}" + (" ⚠ LOW CONTRAST" if bc < _THRESHOLD_CONTRAST else " ✓"))
        lines.append(f"  Color confusion:    {cc:.4f}" + (" ⚠ COLORS TOO SIMILAR" if cc < _THRESHOLD_COLOR else " ✓"))
        lines.append(f"  BG complexity:      {bg:.4f}" + (" ⚠ BUSY BACKGROUND" if bg > _THRESHOLD_BG else " ✓"))

        if b < _THRESHOLD_DARK:
            frame_issues.append("dark_scene")
        if bl < _THRESHOLD_BLUR:
            frame_issues.append("blurry")
        if bc < _THRESHOLD_CONTRAST:
            frame_issues.append("low_boundary_contrast")
        if cc < _THRESHOLD_COLOR:
            frame_issues.append("color_confusion")
        if bg > _THRESHOLD_BG:
            frame_issues.append("busy_background")

        if frame_issues:
            lines.append(f"  Issues: {', '.join(frame_issues)}")
        else:
            lines.append("  No significant issues detected.")

        issues_found.extend(frame_issues)
        lines.append("")

    # Actionable advice
    unique_issues = list(dict.fromkeys(issues_found))
    if unique_issues:
        lines.append("=== Recommendations ===")
        if "dark_scene" in unique_issues:
            lines.append("• Dark scene: Try boosting image brightness/gamma before masking, or use a model with low-light capability (e.g., SAM2 with auto-point prompts).")
        if "blurry" in unique_issues:
            lines.append("• Blurry image: Apply sharpening before mask generation, or use a matting model (ViTMatte) that handles soft edges.")
        if "low_boundary_contrast" in unique_issues:
            lines.append("• Low boundary contrast: The subject blends with background at the edge. Use trimap-based matting (ViTMatte) or manual boundary refinement.")
        if "color_confusion" in unique_issues:
            lines.append("• Color confusion at boundary: Subject and background have similar colors. Use text-prompt segmentation (GroundingDINO/Florence2) or manual point prompts.")
        if "busy_background" in unique_issues:
            lines.append("• Busy background: High edge density behind subject. Use a model with strong figure-ground separation (RMBG, BiRefNet) or hierarchical SAM2 segmenter.")
    else:
        lines.append("=== No significant issues detected ===")
        lines.append("The image+mask combination appears healthy. If masking still fails, consider increasing model resolution or using manual prompts.")

    return "\n".join(lines)


def _suggest_method(
    brightness: torch.Tensor,
    blur: torch.Tensor,
    boundary_contrast: torch.Tensor,
    color_confusion: torch.Tensor,
    bg_complexity: torch.Tensor,
) -> str:
    """Suggest the best masking method based on which conditions triggered."""
    # Average across batch
    b = brightness.mean().item()
    bl = blur.mean().item()
    bc = boundary_contrast.mean().item()
    cc = color_confusion.mean().item()
    bg = bg_complexity.mean().item()

    suggestions = []

    has_dark = b < _THRESHOLD_DARK
    has_blur = bl < _THRESHOLD_BLUR
    has_low_contrast = bc < _THRESHOLD_CONTRAST
    has_color_confusion = cc < _THRESHOLD_COLOR
    has_busy_bg = bg > _THRESHOLD_BG

    if not any([has_dark, has_blur, has_low_contrast, has_color_confusion, has_busy_bg]):
        return "auto (no significant issues — any segmentation method should work)"

    if has_color_confusion or has_low_contrast:
        suggestions.append("ViTMatte (trimap-based matting handles boundary ambiguity)")
    if has_dark:
        suggestions.append("SAM2 with auto-point prompts (robust in low light)")
    if has_blur:
        suggestions.append("ViTMatte (handles soft/blurry edges via alpha matting)")
    if has_busy_bg:
        suggestions.append("RMBG or BiRefNet (strong figure-ground separation)")

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for s in suggestions:
        key = s.split("(")[0].strip()
        if key not in seen:
            seen.add(key)
            unique.append(s)

    return " → ".join(unique) if unique else "auto"


# ══════════════════════════════════════════════════════════════════════
#  Node class
# ══════════════════════════════════════════════════════════════════════

class MaskFailureExplainerMEC:
    """Diagnose why a mask failed and suggest fixes.

    Runs five pure-tensor analysis metrics on the image+mask pair and
    produces an explanation, a problem-regions heatmap, a severity score,
    and a suggested masking method.
    """

    VRAM_TIER = 1

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Input image(s) — (B,H,W,C) float32 [0,1].",
                }),
                "mask": ("MASK", {
                    "tooltip": "Mask to diagnose — (B,H,W) float32 [0,1]. Can be from any segmentation method.",
                }),
            },
            "optional": {
                "ring_width": ("INT", {
                    "default": 5, "min": 1, "max": 50, "step": 1,
                    "tooltip": "Width in pixels of the boundary ring used for contrast/color analysis.",
                }),
                "blur_threshold": ("FLOAT", {
                    "default": 50.0, "min": 0.0, "max": 1000.0, "step": 1.0,
                    "tooltip": "Laplacian variance threshold below which the image is considered blurry.",
                }),
                "brightness_threshold": ("FLOAT", {
                    "default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Mean brightness threshold below which the scene is considered dark.",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "MASK", "FLOAT", "STRING")
    RETURN_NAMES = ("explanation", "problem_regions_mask", "severity_score", "suggested_method")
    FUNCTION = "analyze"
    CATEGORY = "MaskEditControl/Diagnostics"
    DESCRIPTION = (
        "Diagnose why a mask might be failing. Analyzes brightness, blur, "
        "boundary contrast, color confusion, and background complexity. "
        "Outputs a detailed explanation, problem heatmap, severity score, "
        "and suggested masking method."
    )

    def analyze(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        ring_width: int = 5,
        blur_threshold: float = 50.0,
        brightness_threshold: float = 0.15,
    ) -> tuple[str, torch.Tensor, float, str]:
        try:
            B, H, W, C = image.shape

            # Ensure mask matches image spatial dims
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
            if mask.shape[0] != B:
                # Broadcast single mask to batch
                if mask.shape[0] == 1:
                    mask = mask.expand(B, -1, -1)
                else:
                    raise ValueError(
                        f"[MEC] Mask batch size {mask.shape[0]} does not match "
                        f"image batch size {B}."
                    )
            if mask.shape[1] != H or mask.shape[2] != W:
                mask = F.interpolate(
                    mask.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False
                ).squeeze(1)

            # Move kernels to same device as image
            device = _get_device(image)
            mask = mask.to(device=device, dtype=image.dtype)

            # Downsample very large inputs for the heavy analytical kernels.
            # The metrics are scale-invariant in spirit; the heatmap is upsampled back.
            ANALYZE_MAX_EDGE = 2048
            long_edge = max(H, W)
            if long_edge > ANALYZE_MAX_EDGE:
                scale = ANALYZE_MAX_EDGE / float(long_edge)
                aH = max(1, int(round(H * scale)))
                aW = max(1, int(round(W * scale)))
                image_a = F.interpolate(
                    image.permute(0, 3, 1, 2), size=(aH, aW),
                    mode="bilinear", align_corners=False,
                ).permute(0, 2, 3, 1).contiguous()
                mask_a = F.interpolate(
                    mask.unsqueeze(1), size=(aH, aW),
                    mode="bilinear", align_corners=False,
                ).squeeze(1)
            else:
                image_a, mask_a = image, mask

            # ── Run all 5 analysis metrics ────────────────────────────
            brightness = _compute_brightness(image_a)                          # (B,)
            blur = _compute_blur_score(image_a)                                # (B,)
            boundary_contrast = _compute_boundary_contrast(image_a, mask_a)    # (B,)
            color_confusion = _compute_boundary_color_confusion(image_a, mask_a) # (B,)
            bg_complexity = _compute_bg_complexity(image_a, mask_a)             # (B,)

            # ── Severity score ────────────────────────────────────────
            severity = _compute_severity(
                brightness, blur, boundary_contrast, color_confusion, bg_complexity
            )

            # ── Explanation string ────────────────────────────────────
            explanation = _build_explanation(
                brightness, blur, boundary_contrast, color_confusion,
                bg_complexity, severity, B, H, W,
            )

            # ── Problem regions heatmap ───────────────────────────────
            heatmap = _build_problem_heatmap(
                image_a, mask_a, brightness, blur,
                boundary_contrast, color_confusion, bg_complexity,
            )
            # Upsample heatmap to match the original image size if we downsampled.
            if heatmap.shape[-2:] != (H, W):
                heatmap = F.interpolate(
                    heatmap.unsqueeze(1), size=(H, W),
                    mode="bilinear", align_corners=False,
                ).squeeze(1)

            # ── Suggested method ──────────────────────────────────────
            method = _suggest_method(
                brightness, blur, boundary_contrast, color_confusion, bg_complexity,
            )

            return (explanation, heatmap, severity, method)

        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
