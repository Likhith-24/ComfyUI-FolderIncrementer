"""
LuminanceKeyerMEC – Professional luminance keyer inspired by Nuke's LumaKeyer.

Computes ITU-R BT.709 luminance from an input image, then extracts a matte
using low/high thresholds with smooth S-curve falloff and gamma correction.

Modes:
  - **highlights** – keys bright regions (default low=0.7, high=1.0)
  - **midtones**   – keys mid-range luminance (default low=0.3, high=0.7)
  - **shadows**    – keys dark regions (default low=0.0, high=0.3)
  - **custom**     – uses user-specified low/high thresholds
  - **auto**       – analyzes the image luminance distribution and picks
                     highlights/midtones/shadows automatically

Pure tensor math. No cv2 or model dependencies. VRAM Tier 1.
"""

from __future__ import annotations

import gc
import torch

# ITU-R BT.709 luminance coefficients
_BT709_R = 0.2126
_BT709_G = 0.7152
_BT709_B = 0.0722

# Preset threshold ranges
_PRESETS = {
    "highlights": (0.7, 1.0),
    "midtones":   (0.3, 0.7),
    "shadows":    (0.0, 0.3),
}


def _smooth_step(x: torch.Tensor, falloff: float) -> torch.Tensor:
    """Apply a smooth S-curve to values already in [0, 1].

    falloff controls steepness:
      0   → hard binary threshold (step function)
      1   → standard smoothstep (3t² - 2t³)
      >1  → very gradual transition (raised to power of falloff)
    """
    x = x.clamp(0.0, 1.0)
    if falloff <= 0.0:
        return (x > 0.5).float()
    # Hermite smoothstep: 3t² - 2t³
    smooth = x * x * (3.0 - 2.0 * x)
    if abs(falloff - 1.0) < 1e-6:
        return smooth
    # Raise to power for adjustable falloff: >1 flattens, <1 sharpens
    return smooth.pow(1.0 / max(falloff, 0.01))


def _auto_select_mode(luminance: torch.Tensor) -> str:
    """Pick highlights/midtones/shadows based on luminance statistics.

    Strategy: compute mean luminance across the entire batch.
      mean > 0.6  → image is predominantly bright  → key shadows (the minority)
      mean < 0.4  → image is predominantly dark    → key highlights (the minority)
      otherwise   → balanced image                  → key midtones
    """
    mean_luma = luminance.mean().item()
    if mean_luma > 0.6:
        return "shadows"
    elif mean_luma < 0.4:
        return "highlights"
    return "midtones"


class LuminanceKeyerMEC:
    """Extract a luminance-based matte from an image.

    Computes BT.709 luminance then applies threshold range with smooth
    falloff and gamma correction. Supports preset modes and auto-detection.
    """

    VRAM_TIER = 1

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Input image(s) to extract luminance key from.",
                }),
                "mode": (["auto", "highlights", "midtones", "shadows", "custom"], {
                    "default": "auto",
                    "tooltip": (
                        "Preset luminance range or custom thresholds.\n"
                        "auto: Analyzes image brightness to pick best range.\n"
                        "highlights: Keys bright regions (0.7–1.0).\n"
                        "midtones: Keys mid-range luminance (0.3–0.7).\n"
                        "shadows: Keys dark regions (0.0–0.3).\n"
                        "custom: Uses the low/high sliders directly."
                    ),
                }),
                "low": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": (
                        "Low threshold – pixels with luminance below this become 0 in the mask. "
                        "Only used directly in custom mode; presets override this."
                    ),
                }),
                "high": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": (
                        "High threshold – pixels with luminance above this become 1 in the mask. "
                        "Only used directly in custom mode; presets override this."
                    ),
                }),
                "gamma": ("FLOAT", {
                    "default": 1.0, "min": 0.01, "max": 10.0, "step": 0.01,
                    "tooltip": (
                        "Gamma correction applied after keying. "
                        ">1 compresses mask toward black (reduces coverage). "
                        "<1 expands mask toward white (increases coverage)."
                    ),
                }),
                "falloff": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1,
                    "tooltip": (
                        "Smoothness of the transition between low and high thresholds. "
                        "0 = hard binary edge. 1 = standard smooth. "
                        ">1 = very gradual transition."
                    ),
                }),
                "invert": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Invert the output mask (swap keyed and unkeyed regions).",
                }),
            },
        }

    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("mask", "info")
    FUNCTION = "key_luminance"
    CATEGORY = "MaskEditControl/Keying"
    DESCRIPTION = (
        "Professional luminance keyer inspired by Nuke's LumaKeyer.\n"
        "Computes ITU-R BT.709 luminance from an image and extracts a matte\n"
        "using adjustable thresholds with smooth S-curve falloff and gamma.\n"
        "Modes: auto, highlights, midtones, shadows, custom."
    )

    def key_luminance(
        self,
        image: torch.Tensor,
        mode: str,
        low: float,
        high: float,
        gamma: float,
        falloff: float,
        invert: bool,
    ) -> tuple[torch.Tensor, str]:
        B, H, W, C = image.shape

        try:
            # ── 1. Compute BT.709 luminance ──────────────────────────
            # image is (B, H, W, C) with C >= 3
            r = image[:, :, :, 0]
            g = image[:, :, :, 1]
            b = image[:, :, :, 2]
            luminance = _BT709_R * r + _BT709_G * g + _BT709_B * b  # (B, H, W)
            luminance = luminance.clamp(0.0, 1.0)

            # ── 2. Determine thresholds ──────────────────────────────
            effective_mode = mode
            if mode == "auto":
                effective_mode = _auto_select_mode(luminance)

            if effective_mode in _PRESETS:
                t_low, t_high = _PRESETS[effective_mode]
            else:
                # custom mode
                t_low = low
                t_high = high

            # Ensure low <= high
            if t_low > t_high:
                t_low, t_high = t_high, t_low

            # ── 3. Build mask based on mode ───────────────────────────
            span = t_high - t_low
            if span < 1e-7:
                # Degenerate range: hard binary at the threshold value
                mask = (luminance >= t_low).float()
            elif effective_mode == "shadows":
                # Shadows: dark pixels → mask=1, bright pixels → mask=0
                # Ramp DOWN within [low, high], zero above high
                t = (luminance - t_low) / span
                t = t.clamp(0.0, 1.0)
                mask = _smooth_step(1.0 - t, falloff)
                # Pixels above high get mask=0 (via clamp), pixels below low get mask=1
            elif effective_mode == "midtones":
                # Midtones: tent function peaked at midpoint of [low, high]
                midpoint = (t_low + t_high) * 0.5
                half_span = span * 0.5
                t = 1.0 - ((luminance - midpoint).abs() / half_span).clamp(0.0, 1.0)
                mask = _smooth_step(t, falloff)
            else:
                # Highlights / custom: ramp UP within [low, high]
                t = (luminance - t_low) / span
                t = t.clamp(0.0, 1.0)
                mask = _smooth_step(t, falloff)

            # ── 4. Gamma correction ──────────────────────────────────
            # gamma > 1 → pushes mask darker (pow > 1 compresses toward 0)
            # gamma < 1 → pushes mask brighter (pow < 1 expands toward 1)
            if abs(gamma - 1.0) > 1e-6:
                mask = mask.clamp(0.0, 1.0).pow(max(gamma, 0.01))

            # ── 5. Invert ────────────────────────────────────────────
            if invert:
                mask = 1.0 - mask

            mask = mask.clamp(0.0, 1.0)

            # ── 6. Compute statistics for info string ────────────────
            mean_luma = luminance.mean().item()
            std_luma = luminance.std().item()
            mask_coverage = mask.mean().item() * 100.0

            # Per-frame stats
            frame_coverages = []
            for i in range(B):
                fc = mask[i].mean().item() * 100.0
                frame_coverages.append(f"{fc:.1f}%")

            info_lines = [
                f"Mode: {effective_mode}" + (f" (auto-selected from '{mode}')" if mode == "auto" else ""),
                f"Thresholds: low={t_low:.3f}, high={t_high:.3f}",
                f"Gamma: {gamma:.2f} | Falloff: {falloff:.1f} | Invert: {invert}",
                f"Mean luminance: {mean_luma:.4f} (std: {std_luma:.4f})",
                f"Mask coverage: {mask_coverage:.1f}%",
                f"Batch: {B} frame(s), {H}x{W}",
            ]
            if B > 1:
                info_lines.append(f"Per-frame coverage: {', '.join(frame_coverages)}")

            info = "\n".join(info_lines)

            return (mask, info)

        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
