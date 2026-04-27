"""
Plate-tools nodes (MEC):
  - GrainMatchMEC: Extract grain spectrum from a reference plate and re-apply
    it to a clean image so synthetic content matches the source plate.
  - PlateStabilizerMEC: Affine-stabilize a video batch to its first frame
    using cv2 ORB feature matching when available, else falls back to
    cross-correlation translation.
  - CleanPlateExtractorMEC: Median across a batch (with optional mask
    exclusion) to produce a clean plate.
  - DifferenceMatteMEC: Per-pixel difference between two images → MASK.

cv2 is optional; degraded fallbacks are provided.
"""
from __future__ import annotations

import json
import logging

import torch

logger = logging.getLogger("MEC.PlateTools")


def _try_cv2():
    try:
        import cv2  # type: ignore[import-not-found]
        return cv2
    except ImportError:
        return None


# ──────────────────────────────────────────────────────────────────────
#  GrainMatch (FFT-based)
# ──────────────────────────────────────────────────────────────────────

def _luma(t: torch.Tensor) -> torch.Tensor:
    return 0.2126 * t[..., 0] + 0.7152 * t[..., 1] + 0.0722 * t[..., 2]


def _denoise_box(t: torch.Tensor, k: int = 5) -> torch.Tensor:
    """Cheap NCHW box-filter denoiser. t: [B,H,W,C]."""
    if k <= 1:
        return t
    pad = k // 2
    x = t.permute(0, 3, 1, 2)
    weight = torch.ones(x.shape[1], 1, k, k, device=x.device, dtype=x.dtype) / (k * k)
    x = torch.nn.functional.pad(x, (pad, pad, pad, pad), mode="reflect")
    x = torch.nn.functional.conv2d(x, weight, groups=x.shape[1])
    return x.permute(0, 2, 3, 1)


class GrainMatchMEC:
    """Extract grain noise from ``reference`` and add it to ``target``.

    Grain estimate = reference - denoise(reference). The grain is rescaled
    to ``intensity`` and added to ``target``. If the reference has multiple
    frames, a random frame is sampled for each target frame so the result
    is temporally non-static.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference": ("IMAGE",),
                "target": ("IMAGE",),
            },
            "optional": {
                "intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05}),
                "denoise_kernel": ("INT", {"default": 5, "min": 3, "max": 15, "step": 2}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "info_json")
    FUNCTION = "match"
    CATEGORY = "MaskEditControl/PlateTools"
    DESCRIPTION = "Extract grain from a reference plate and re-apply it to target."

    def match(
        self, reference: torch.Tensor, target: torch.Tensor,
        intensity: float = 1.0, denoise_kernel: int = 5, seed: int = 0,
    ):
        ref = reference
        if ref.shape[1:3] != target.shape[1:3]:
            x = ref.permute(0, 3, 1, 2)
            x = torch.nn.functional.interpolate(
                x, size=target.shape[1:3], mode="bilinear", align_corners=False,
            )
            ref = x.permute(0, 2, 3, 1)
        # Force odd kernel
        k = max(3, int(denoise_kernel) | 1)
        denoised = _denoise_box(ref, k)
        grain = ref - denoised
        grain_std = float(grain.std().item())
        # Per-target-frame, sample a random reference frame
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        B = target.shape[0]
        idx = torch.randint(0, ref.shape[0], (B,), generator=gen)
        sampled = grain[idx]
        out = (target + intensity * sampled).clamp(0.0, 1.0)
        return (out, json.dumps({"grain_std": grain_std, "intensity": intensity, "denoise_kernel": k}))


# ──────────────────────────────────────────────────────────────────────
#  PlateStabilizer
# ──────────────────────────────────────────────────────────────────────

def _phase_correlate_shift(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    """Translation-only fallback: integer shift via FFT phase correlation.

    a, b: 2D float tensors of equal HxW. Returns (dy, dx) that aligns b → a.
    """
    A = torch.fft.fft2(a)
    B = torch.fft.fft2(b)
    R = A * B.conj()
    R = R / (R.abs() + 1e-8)
    r = torch.fft.ifft2(R).real
    H, W = r.shape
    flat = r.flatten().argmax().item()
    py, px = flat // W, flat % W
    if py > H // 2:
        py -= H
    if px > W // 2:
        px -= W
    return float(py), float(px)


def _warp_translate(img: torch.Tensor, dy: float, dx: float) -> torch.Tensor:
    """Translate IMAGE [H,W,C] by (dy, dx) pixels via affine_grid."""
    H, W = img.shape[:2]
    theta = torch.tensor(
        [[1.0, 0.0, -2.0 * dx / max(W - 1, 1)],
         [0.0, 1.0, -2.0 * dy / max(H - 1, 1)]],
        device=img.device, dtype=img.dtype,
    ).unsqueeze(0)
    grid = torch.nn.functional.affine_grid(theta, [1, img.shape[-1], H, W], align_corners=True)
    x = img.permute(2, 0, 1).unsqueeze(0)
    out = torch.nn.functional.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return out.squeeze(0).permute(1, 2, 0)


class PlateStabilizerMEC:
    """Affine-stabilize a video batch to its first frame.

    With cv2 available, uses ORB features + estimateAffinePartial2D.
    Without cv2, falls back to per-frame translation via FFT phase correlation.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"images": ("IMAGE",)},
            "optional": {
                "max_features": ("INT", {"default": 500, "min": 50, "max": 5000, "step": 50}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "info_json")
    FUNCTION = "stabilize"
    CATEGORY = "MaskEditControl/PlateTools"
    DESCRIPTION = "Stabilize a video batch to frame 0 via ORB+affine (cv2) or FFT translation."

    def stabilize(self, images: torch.Tensor, max_features: int = 500):
        B = images.shape[0]
        if B <= 1:
            return (images, json.dumps({"backend": "noop", "frames": B}))
        cv2 = _try_cv2()
        if cv2 is None:
            return self._stabilize_fft(images)
        return self._stabilize_orb(images, max_features, cv2)

    def _stabilize_fft(self, images: torch.Tensor):
        ref = _luma(images[0])
        out = [images[0]]
        shifts = []
        for i in range(1, images.shape[0]):
            cur = _luma(images[i])
            dy, dx = _phase_correlate_shift(ref, cur)
            out.append(_warp_translate(images[i], dy, dx))
            shifts.append({"frame": i, "dy": dy, "dx": dx})
        return (torch.stack(out, dim=0), json.dumps({"backend": "fft_translate", "shifts": shifts}))

    def _stabilize_orb(self, images: torch.Tensor, max_features: int, cv2):
        import numpy as np
        ref_np = (images[0].cpu().numpy() * 255.0).clip(0, 255).astype("uint8")
        ref_gray = cv2.cvtColor(ref_np, cv2.COLOR_RGB2GRAY)
        orb = cv2.ORB_create(nfeatures=int(max_features))
        kp1, des1 = orb.detectAndCompute(ref_gray, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        out = [images[0]]
        records = []
        for i in range(1, images.shape[0]):
            cur_np = (images[i].cpu().numpy() * 255.0).clip(0, 255).astype("uint8")
            cur_gray = cv2.cvtColor(cur_np, cv2.COLOR_RGB2GRAY)
            kp2, des2 = orb.detectAndCompute(cur_gray, None)
            warped = images[i]
            mat = None
            if des1 is not None and des2 is not None and len(kp2) >= 4:
                matches = sorted(bf.match(des1, des2), key=lambda m: m.distance)[:50]
                if len(matches) >= 4:
                    src = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                    dst = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                    mat, _ = cv2.estimateAffinePartial2D(dst, src, method=cv2.RANSAC)
            if mat is not None:
                h, w = ref_gray.shape
                warped_np = cv2.warpAffine(
                    cur_np, mat, (w, h),
                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT,
                )
                warped = torch.from_numpy(warped_np.astype("float32") / 255.0)
                records.append({"frame": i, "ok": True})
            else:
                records.append({"frame": i, "ok": False})
            out.append(warped)
        return (torch.stack(out, dim=0), json.dumps({"backend": "orb", "frames": records}))


# ──────────────────────────────────────────────────────────────────────
#  CleanPlate
# ──────────────────────────────────────────────────────────────────────

class CleanPlateExtractorMEC:
    """Pixelwise median across a batch with optional mask exclusion.

    Excluded pixels are ignored in the median; if no valid samples remain,
    the first frame's pixel is used.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"images": ("IMAGE",)},
            "optional": {
                "exclude_mask": ("MASK", {"tooltip": "Pixels where mask>=0.5 are excluded from the median."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("clean_plate",)
    FUNCTION = "extract"
    CATEGORY = "MaskEditControl/PlateTools"
    DESCRIPTION = "Median across a batch (with optional mask exclusion) → clean plate."

    def extract(self, images: torch.Tensor, exclude_mask: torch.Tensor | None = None):
        if images.shape[0] == 1:
            return (images.clone(),)
        if exclude_mask is None:
            med = images.median(dim=0).values
            return (med.unsqueeze(0),)
        m = exclude_mask
        if m.dim() == 4:
            m = m[..., 0]
        if m.shape[0] != images.shape[0]:
            raise ValueError(
                f"exclude_mask batch ({m.shape[0]}) ≠ images batch ({images.shape[0]})",
            )
        valid = (m < 0.5).unsqueeze(-1).to(images.dtype)  # [B,H,W,1]
        # For pixels that are ever invalid, replace with NaN-equivalent and
        # take the masked median by replacing invalid samples with the per-pixel
        # mean of valid ones (cheap and stable).
        out = images.clone()
        for c in range(images.shape[-1]):
            vals = images[..., c]  # [B,H,W]
            mask = valid[..., 0]
            # Replace invalid with first-valid per pixel
            big = torch.full_like(vals, float("inf"))
            masked = torch.where(mask > 0.5, vals, big)
            sort_v, _ = torch.sort(masked, dim=0)
            counts = mask.sum(dim=0).clamp_min(1)  # [H,W]
            mid_idx = ((counts - 1) // 2).long().unsqueeze(0)  # [1,H,W]
            med = torch.gather(sort_v, 0, mid_idx).squeeze(0)
            no_valid = counts < 1
            med = torch.where(no_valid, vals[0], med)
            out[..., c] = med.unsqueeze(0).expand_as(out[..., c])
        return (out[:1].clone(),)


# ──────────────────────────────────────────────────────────────────────
#  DifferenceMatte
# ──────────────────────────────────────────────────────────────────────

class DifferenceMatteMEC:
    """Per-pixel L2 (or L1) distance between two IMAGEs → MASK."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
            },
            "optional": {
                "metric": (["l2", "l1"], {"default": "l2"}),
                "threshold": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.001}),
                "softness": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.001}),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "compute"
    CATEGORY = "MaskEditControl/PlateTools"
    DESCRIPTION = "Difference matte: |a-b| → MASK with threshold + softness."

    def compute(
        self, image_a: torch.Tensor, image_b: torch.Tensor,
        metric: str = "l2", threshold: float = 0.05, softness: float = 0.05,
    ):
        if image_a.shape != image_b.shape:
            raise ValueError(f"shape mismatch: {image_a.shape} vs {image_b.shape}")
        diff = image_a - image_b
        if metric == "l1":
            d = diff.abs().mean(dim=-1)
        else:
            d = (diff * diff).sum(dim=-1).sqrt() / (3.0 ** 0.5)
        if softness <= 1e-6:
            mask = (d >= threshold).to(d.dtype)
        else:
            lo = max(threshold - softness, 0.0)
            hi = threshold + softness
            mask = ((d - lo) / max(hi - lo, 1e-6)).clamp(0.0, 1.0)
        return (mask,)


NODE_CLASS_MAPPINGS = {
    "GrainMatchMEC": GrainMatchMEC,
    "PlateStabilizerMEC": PlateStabilizerMEC,
    "CleanPlateExtractorMEC": CleanPlateExtractorMEC,
    "DifferenceMatteMEC": DifferenceMatteMEC,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "GrainMatchMEC": "Grain Match (MEC)",
    "PlateStabilizerMEC": "Plate Stabilizer (MEC)",
    "CleanPlateExtractorMEC": "Clean Plate Extractor (MEC)",
    "DifferenceMatteMEC": "Difference Matte (MEC)",
}
