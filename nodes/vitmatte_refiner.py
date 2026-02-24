"""
ViTMatteRefinerMEC – Edge refinement using ViTMatte-style alpha matting.

Takes a coarse mask from SAM/points and refines its edges using the
original image as guidance.  Produces cleaner, anti-aliased alpha mattes
suitable for compositing.

Refinement backends (best to simplest):
  1. ViTMatte neural matting (requires transformers)
  2. Multi-scale guided filter (best non-neural)
  3. LAB color-aware refinement
  4. Single-scale guided filter
  5. Laplacian pyramid blending
  6. Gaussian blur fallback (always available)

All heavy computation delegates to nodes.utils for DRY code.
"""

import torch
import torch.nn.functional as F
import numpy as np

from .utils import (
    HAS_CV2,
    refine_with_vitmatte,
    multi_scale_guided_refine,
    color_aware_refine,
    guided_filter,
    compute_edge_band_np,
    compute_edge_band_torch,
    gaussian_edge_refine,
    boost_edge_contrast,
    generate_trimap,
    build_laplacian_pyramid,
    reconstruct_laplacian_pyramid,
    make_mask_overlay_preview,
)

try:
    import cv2
except ImportError:
    pass


class ViTMatteRefinerMEC:
    """Refine mask edges using image-guided matting techniques.

    Supports multiple backends with automatic fallback chain.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "method": (["auto", "vitmatte", "multi_scale_guided", "color_aware",
                            "guided_filter", "laplacian_blend", "gaussian_blur"], {
                    "default": "auto",
                    "tooltip": (
                        "Refinement method.\n"
                        "auto: tries vitmatte → multi_scale_guided → guided_filter → gaussian.\n"
                        "vitmatte: ViTMatte neural matting (requires transformers).\n"
                        "multi_scale_guided: guided filter at 3 scales – best non-neural.\n"
                        "color_aware: LAB-space refinement robust to lighting changes.\n"
                        "guided_filter: single-scale edge-aware smoothing.\n"
                        "laplacian_blend: frequency-domain blending.\n"
                        "gaussian_blur: simple blur of mask edges."
                    ),
                }),
                "edge_radius": ("INT", {
                    "default": 10, "min": 1, "max": 200, "step": 1,
                    "tooltip": "Radius in pixels around mask edges to refine",
                }),
                "edge_softness": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "How soft/feathered the refined edges should be (0=sharp, 1=very soft)",
                }),
                "erode_amount": ("INT", {
                    "default": 0, "min": -50, "max": 50, "step": 1,
                    "tooltip": "Erode (negative values expand) the mask before refinement",
                }),
                "detail_level": ("FLOAT", {
                    "default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "How much fine detail to preserve (hair, fur, lace). 0=smooth, 1=max detail.",
                }),
                "iterations": ("INT", {
                    "default": 1, "min": 1, "max": 5, "step": 1,
                    "tooltip": "Refinement iterations. 2-3 improves convergence on difficult edges.",
                }),
                "edge_contrast_boost": ("FLOAT", {
                    "default": 1.0, "min": 0.5, "max": 3.0, "step": 0.1,
                    "tooltip": "Boost edge contrast. >1 gives sharper boundaries.",
                }),
            },
            "optional": {
                "trimap_mask": ("MASK", {
                    "tooltip": "Optional trimap for ViTMatte (white=fg, black=bg, gray=unknown)",
                }),
            },
        }

    RETURN_TYPES = ("MASK", "MASK", "IMAGE",)
    RETURN_NAMES = ("refined_mask", "edge_mask", "preview",)
    FUNCTION = "refine"
    CATEGORY = "MaskEditControl/Refinement"
    DESCRIPTION = (
        "Refine mask edges using image-guided matting. "
        "Supports ViTMatte, multi-scale guided filter, LAB color-aware, "
        "Laplacian blending, and Gaussian blur."
    )

    def refine(self, image, mask, method, edge_radius, edge_softness,
               erode_amount, detail_level, iterations=1,
               edge_contrast_boost=1.0, trimap_mask=None):

        img = image[0]  # (H, W, C)
        H, W = img.shape[:2]

        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        m = mask[0].clone()  # (H, W)

        # Resize mask to match image
        if m.shape[0] != H or m.shape[1] != W:
            m = F.interpolate(m.unsqueeze(0).unsqueeze(0), size=(H, W),
                              mode="bilinear", align_corners=False)[0, 0]

        # Erode/expand
        if erode_amount != 0 and HAS_CV2:
            m_np = (m.cpu().numpy() * 255).astype(np.uint8)
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (abs(erode_amount) * 2 + 1,) * 2
            )
            if erode_amount > 0:
                m_np = cv2.erode(m_np, kernel, iterations=1)
            else:
                m_np = cv2.dilate(m_np, kernel, iterations=1)
            m = torch.from_numpy(m_np.astype(np.float32) / 255.0)

        # Iterative refinement
        refined = m
        for _ in range(iterations):
            refined = self._dispatch_refine(
                method, img, refined, trimap_mask,
                edge_radius, edge_softness, detail_level,
            )

        refined = refined.clamp(0, 1)

        # Edge contrast boost
        if edge_contrast_boost != 1.0:
            refined = boost_edge_contrast(refined, m, edge_contrast_boost, edge_radius)
            refined = refined.clamp(0, 1)

        # Edge mask: difference between refined and binary original
        binary = (m > 0.5).float()
        edge_mask = torch.abs(refined - binary)

        # Preview
        preview = make_mask_overlay_preview(img, refined, color=(0.0, 1.0, 0.0))

        # Add red edge highlight
        edge_vis = torch.abs(
            F.avg_pool2d(refined.unsqueeze(0).unsqueeze(0), 3, 1, 1)[0, 0] - refined
        )
        preview[:, :, 0] = torch.clamp(preview[:, :, 0] + edge_vis * 2.0, 0, 1)

        return (refined.unsqueeze(0), edge_mask.unsqueeze(0), preview.unsqueeze(0))

    def _dispatch_refine(self, method, img, m, trimap_mask,
                          edge_radius, edge_softness, detail_level):
        """Route to the chosen refinement method with automatic fallback."""

        img_np = (img.cpu().numpy() * 255).astype(np.uint8)
        m_np = m.cpu().numpy().astype(np.float32)

        if method == "auto":
            result = self._try_vitmatte_wrapper(img, m, trimap_mask, edge_radius)
            if result is None:
                r = multi_scale_guided_refine(img_np, m_np, edge_radius, detail_level)
                if r is not None:
                    result = torch.from_numpy(r).to(m.device)
            if result is None:
                result = self._try_guided_single(img_np, m_np, m.device,
                                                  edge_radius, detail_level)
            if result is None:
                result = gaussian_edge_refine(m, edge_radius)
            return self._apply_softness(result, m, edge_radius, edge_softness)

        if method == "vitmatte":
            result = self._try_vitmatte_wrapper(img, m, trimap_mask, edge_radius)
            if result is None:
                r = multi_scale_guided_refine(img_np, m_np, edge_radius, detail_level)
                result = torch.from_numpy(r).to(m.device) if r is not None else None
            if result is None:
                result = gaussian_edge_refine(m, edge_radius)
            return self._apply_softness(result, m, edge_radius, edge_softness)

        if method == "multi_scale_guided":
            r = multi_scale_guided_refine(img_np, m_np, edge_radius, detail_level)
            result = torch.from_numpy(r).to(m.device) if r is not None else None
            if result is None:
                result = gaussian_edge_refine(m, edge_radius)
            return self._apply_softness(result, m, edge_radius, edge_softness)

        if method == "color_aware":
            r = color_aware_refine(img_np, m_np, edge_radius, detail_level)
            result = torch.from_numpy(r).to(m.device) if r is not None else None
            if result is None:
                result = gaussian_edge_refine(m, edge_radius)
            return self._apply_softness(result, m, edge_radius, edge_softness)

        if method == "guided_filter":
            result = self._try_guided_single(img_np, m_np, m.device,
                                              edge_radius, detail_level)
            if result is None:
                result = gaussian_edge_refine(m, edge_radius)
            return self._apply_softness(result, m, edge_radius, edge_softness)

        if method == "laplacian_blend":
            result = self._laplacian_blend(m_np, m.device, edge_radius, edge_softness)
            return result if result is not None else gaussian_edge_refine(m, edge_radius)

        return gaussian_edge_refine(m, edge_radius)

    # ── ViTMatte wrapper ──────────────────────────────────────────────
    @staticmethod
    def _try_vitmatte_wrapper(img, mask, trimap_mask, edge_radius):
        """Wrap the shared ViTMatte refinement with trimap support."""
        tri = None
        if trimap_mask is not None:
            tri = trimap_mask[0] if trimap_mask.dim() == 3 else trimap_mask
        return refine_with_vitmatte(img, mask, edge_radius, trimap_input=tri)

    # ── Single-scale guided filter ────────────────────────────────────
    @staticmethod
    def _try_guided_single(img_np, m_np, device, edge_radius, detail_level):
        if not HAS_CV2:
            return None
        try:
            guide = img_np[:, :, :3]
            gray = cv2.cvtColor(guide, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            eps = (1 - detail_level) ** 2 * 0.1 + 1e-6
            filtered = guided_filter(gray, m_np, max(1, edge_radius), eps)
            edge_band = compute_edge_band_np(m_np, edge_radius)
            result = m_np * (1 - edge_band) + np.clip(filtered, 0, 1) * edge_band
            return torch.from_numpy(result.astype(np.float32)).to(device)
        except Exception:
            return None

    # ── Laplacian pyramid blending ────────────────────────────────────
    @staticmethod
    def _laplacian_blend(m_np, device, edge_radius, edge_softness):
        if not HAS_CV2:
            return None
        try:
            H, W = m_np.shape
            levels = min(4, int(np.log2(min(H, W))) - 2)
            if levels < 1:
                return None

            blur_k = max(1, edge_radius * 2) | 1
            soft = cv2.GaussianBlur(m_np, (blur_k, blur_k),
                                     edge_radius * edge_softness * 0.5 + 0.1)

            pyr_mask = build_laplacian_pyramid(m_np, levels)
            pyr_soft = build_laplacian_pyramid(soft, levels)

            result_pyr = []
            for i, (lm, ls) in enumerate(zip(pyr_mask, pyr_soft)):
                w = (i + 1) / len(pyr_mask) * edge_softness
                result_pyr.append(lm * (1 - w) + ls * w)

            result = reconstruct_laplacian_pyramid(result_pyr)
            return torch.from_numpy(np.clip(result, 0, 1).astype(np.float32)).to(device)
        except Exception:
            return None

    # ── Softness blending ─────────────────────────────────────────────
    @staticmethod
    def _apply_softness(result, original_mask, edge_radius, edge_softness):
        """Apply edge softness by blending with Gaussian version at edges."""
        if edge_softness <= 0 or not HAS_CV2:
            return result

        try:
            r_np = result.cpu().numpy().astype(np.float32)
            blur_k = max(1, int(edge_radius * edge_softness * 2)) | 1
            blurred = cv2.GaussianBlur(r_np, (blur_k, blur_k), 0)
            edge_band = compute_edge_band_np(
                original_mask.cpu().numpy(), edge_radius
            )
            fused = r_np * (1 - edge_band * edge_softness) + blurred * (edge_band * edge_softness)
            return torch.from_numpy(np.clip(fused, 0, 1).astype(np.float32)).to(result.device)
        except Exception:
            return result
