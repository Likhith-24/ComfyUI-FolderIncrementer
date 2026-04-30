"""
SeCMatAnyonePipelineMEC – SeC segmentation + MatAnyone2 alpha matting pipeline.

Pipeline stages:
  1. **SeC coarse segmentation** – MLLM-based object tracking (video-aware)
  2. **MatAnyone2 alpha matting** – Temporal-consistent alpha with warmup protocol
  3. **Edge refinement** – Optional ViTMatte / guided-filter edge cleanup
  4. **Post-processing** – Hole filling, small region removal

Best for: Video object segmentation with compositing-grade alpha edges,
especially scenes with occlusions, re-appearances, and complex motion.

Key differences from SAM + ViTMatte Pipeline:
  - SeC uses a Large Vision-Language Model for semantic understanding (text prompts)
  - MatAnyone2 provides temporal consistency across video frames
  - Better for long video sequences with appearance changes
"""

from __future__ import annotations

import gc
import json
import logging

import numpy as np
import torch
import torch.nn.functional as F

from .model_manager import (
    MODEL_REGISTRY,
    get_or_load_model,
    clear_cache,
    scan_model_dir,
)
from .utils import (
    HAS_CV2,
    compute_edge_band_np,
    fill_holes,
    remove_small_regions,
    make_split_preview,
)

logger = logging.getLogger("MEC")


class SeCMatAnyonePipelineMEC:
    """End-to-end SeC → MatAnyone2 pipeline.

    Combines SeC MLLM segmentation with MatAnyone2 temporal alpha matting
    for production-grade video masking with compositing-quality edges.
    """

    REFINE_METHODS = ["none", "vitmatte", "guided_filter", "multi_scale_guided"]

    @classmethod
    def INPUT_TYPES(cls):
        seg_models = []
        for name, reg in sorted(MODEL_REGISTRY.items()):
            if reg.get("family") in ("sec", "sam2", "sam3"):
                seg_models.append(name)
        if not seg_models:
            seg_models = ["sec_4b"]

        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Single image or video frames (B>1 for video).",
                }),
                "segmentation_model": (seg_models, {
                    "default": seg_models[0],
                    "tooltip": (
                        "Segmentation model for coarse masks.\n"
                        "SeC: best for video with text prompts.\n"
                        "SAM2/3: best for point/bbox prompts."
                    ),
                }),
                "text_prompt": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": (
                        "Text description of target object (e.g. 'cat', 'person in red').\n"
                        "Used by SeC for semantic tracking. Leave empty for point/bbox prompts."
                    ),
                }),
                "points_json": ("STRING", {
                    "default": "[]",
                    "multiline": True,
                    "tooltip": 'Point prompts: [{"x":100,"y":200,"label":1}, ...]',
                }),
                "bbox_json": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Bounding box: [x1,y1,x2,y2]",
                }),
                "matting_backend": (["matanyone2", "vitmatte_small", "vitmatte_base", "auto"], {
                    "default": "auto",
                    "tooltip": (
                        "Alpha matting backend.\n"
                        "auto: MatAnyone2 for video (B>1), ViTMatte for single images.\n"
                        "matanyone2: Video matting with temporal consistency.\n"
                        "vitmatte_small/base: Neural matting (best edge quality per frame)."
                    ),
                }),
                "edge_radius": ("INT", {
                    "default": 15, "min": 1, "max": 200, "step": 1,
                    "tooltip": "Edge refinement radius in pixels.",
                }),
                "n_warmup": ("INT", {
                    "default": 5, "min": 1, "max": 30, "step": 1,
                    "tooltip": "MatAnyone2 warmup frames (more = better temporal init).",
                }),
                "precision": (["fp16", "bf16", "fp32"], {
                    "default": "fp16",
                    "tooltip": "Segmentation model precision.",
                }),
                "fill_holes_enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Fill interior holes in the final alpha.",
                }),
                "min_region_size": ("INT", {
                    "default": 64, "min": 0, "max": 10000, "step": 1,
                    "tooltip": "Remove isolated regions smaller than N pixels.",
                }),
            },
            "optional": {
                "positive_coords": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Positive points from Points Mask Editor.",
                }),
                "negative_coords": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Negative points from Points Mask Editor.",
                }),
                "bbox": ("BBOX", {
                    "tooltip": "Positive bbox from upstream node.",
                }),
                "edge_refine_method": (cls.REFINE_METHODS, {
                    "default": "none",
                    "tooltip": (
                        "Optional post-matting edge refinement.\n"
                        "none: use raw MatAnyone2 output.\n"
                        "vitmatte/guided_filter: refine edges after matting."
                    ),
                }),
                "keep_model_loaded": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Keep models in VRAM between runs.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "IMAGE", "STRING")
    RETURN_NAMES = ("rgb", "alpha_mask", "coarse_mask", "preview", "info")
    FUNCTION = "process"
    CATEGORY = "MaskEditControl/Pipeline"
    DESCRIPTION = (
        "SeC + MatAnyone2 end-to-end pipeline:\n"
        "1. SeC/SAM segmentation → coarse masks\n"
        "2. MatAnyone2 temporal alpha matting → compositing-grade alpha\n"
        "3. Optional edge refinement\n"
        "4. Post-processing (hole fill, region cleanup)\n\n"
        "Best for video scenes with occlusions, re-appearances, and complex motion."
    )

    def process(
        self,
        image: torch.Tensor,
        segmentation_model: str,
        text_prompt: str,
        points_json: str,
        bbox_json: str,
        matting_backend: str,
        edge_radius: int,
        n_warmup: int,
        precision: str,
        fill_holes_enabled: bool,
        min_region_size: int,
        positive_coords: str | None = None,
        negative_coords: str | None = None,
        bbox=None,
        edge_refine_method: str = "none",
        keep_model_loaded: bool = True,
    ):
        B, H, W, C = image.shape
        is_video = B > 1
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # ── Stage 1: Coarse Segmentation ──────────────────────────────
        # MANUAL bug-fix (Apr 2026): the legacy import targeted
        # `unified_segmentation_node.UnifiedSegmentationNode` which lives in
        # `_deprecated/`.  The active replacement is
        # `unified_segmentation.UnifiedSegmentation` whose `segment()` exposes
        # a smaller kwarg surface — call it with only the supported keys.
        from .unified_segmentation import UnifiedSegmentation
        seg_node = UnifiedSegmentation()
        coarse_masks, score, seg_info = seg_node.segment(
            image=image,
            model_name=segmentation_model,
            points_json=points_json,
            bbox_json=bbox_json,
            multimask=True,
            mask_index=0,
            precision=precision,
            bbox=bbox,
            text_prompt=text_prompt,
            existing_mask=None,
        )

        # ── Stage 2: Alpha Matting ────────────────────────────────────
        actual_backend = matting_backend
        if actual_backend == "auto":
            actual_backend = "matanyone2" if is_video else "vitmatte_small"

        if actual_backend == "matanyone2":
            alpha_mask = self._run_matanyone2(
                image, coarse_masks, n_warmup, B, H, W, device,
            )
        else:
            alpha_mask = self._run_vitmatte(
                image, coarse_masks, actual_backend, edge_radius, B, H, W, device,
            )

        # ── Stage 3: Optional Edge Refinement ─────────────────────────
        if edge_refine_method != "none":
            alpha_mask = self._edge_refine(
                image, alpha_mask, edge_refine_method, edge_radius, B, H, W,
            )

        # ── Stage 4: Post-processing ─────────────────────────────────
        if fill_holes_enabled or min_region_size > 0:
            alpha_mask = self._post_process(
                alpha_mask, fill_holes_enabled, min_region_size, B, H, W,
            )

        alpha_mask = alpha_mask.float().clamp(0.0, 1.0)

        # Build outputs
        rgb = image[:B] * alpha_mask.unsqueeze(-1)

        # Preview: side-by-side original vs masked
        try:
            preview = make_split_preview(
                image[0], coarse_masks[0], alpha_mask[0],
            )
            preview = preview.unsqueeze(0)
        except Exception:
            preview = rgb[:1]

        # Free VRAM
        if not keep_model_loaded:
            clear_cache()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        info = json.dumps({
            "segmentation_model": segmentation_model,
            "matting_backend": actual_backend,
            "is_video": is_video,
            "frames": B,
            "seg_score": round(score, 4),
            "edge_refine": edge_refine_method,
            "n_warmup": n_warmup,
        }, indent=2)

        return (rgb, alpha_mask, coarse_masks, preview, info)

    # ── MatAnyone2 matting ────────────────────────────────────────────

    def _run_matanyone2(self, image, mask, n_warmup, B, H, W, device):
        """MatAnyone2 temporal alpha matting with warmup protocol."""
        loaded = get_or_load_model("matanyone2", precision="fp32", device=device)
        core = loaded["core"] if isinstance(loaded, dict) else loaded

        alphas = []

        first_img = image[0].permute(2, 0, 1).to(device)
        first_mask = mask[0].unsqueeze(0).to(device)
        first_mask_bin = (first_mask > 0.5).float()

        try:
            core.step(first_img, first_mask_bin)
        except Exception as exc:
            logger.debug("[MEC] MatAnyone2 initial step: %s", exc)

        for _ in range(n_warmup):
            try:
                core.step(first_img, first_frame_pred=True)
            except TypeError:
                try:
                    core.step(first_img, first_mask_bin)
                except Exception:
                    break
            except Exception:
                break

        for i in range(B):
            img_t = image[min(i, image.shape[0] - 1)].permute(2, 0, 1).to(device)
            m_t = mask[i].unsqueeze(0).to(device)
            m_bin = (m_t > 0.5).float()

            try:
                alpha = core.step(img_t, m_bin) if i == 0 else core.step(img_t)
                if isinstance(alpha, torch.Tensor):
                    alphas.append(alpha.cpu().squeeze())
                else:
                    alphas.append(torch.from_numpy(np.array(alpha, dtype=np.float32)))
            except Exception as exc:
                logger.warning("[MEC] MatAnyone2 frame %d: %s", i, exc)
                alphas.append(mask[i].cpu())

        return torch.stack(alphas)

    # ── ViTMatte matting ──────────────────────────────────────────────

    def _run_vitmatte(self, image, mask, variant, edge_radius, B, H, W, device):
        """ViTMatte per-frame alpha matting."""
        from PIL import Image as PILImage

        loaded = get_or_load_model(variant, precision="fp32", device=device)
        if isinstance(loaded, dict):
            model, processor = loaded["model"], loaded["processor"]
        else:
            model, processor = loaded, None

        if processor is None:
            from transformers import VitMatteImageProcessor
            processor = VitMatteImageProcessor()

        dev = next(model.parameters()).device
        alphas = []

        for i in range(B):
            img_np = (image[min(i, image.shape[0] - 1)].cpu().numpy() * 255).astype(np.uint8)
            mask_np = mask[i].cpu().numpy().astype(np.float32)

            trimap_u8 = self._generate_trimap(mask_np, edge_radius)
            pil_img = PILImage.fromarray(img_np[:, :, :3])
            pil_tri = PILImage.fromarray(trimap_u8, mode="L")

            inputs = processor(images=pil_img, trimaps=pil_tri, return_tensors="pt")
            inputs = {k: v.to(dev) for k, v in inputs.items()}

            with torch.no_grad():
                out = model(**inputs)

            a = out.alphas[0, 0].cpu()
            if a.shape[0] != H or a.shape[1] != W:
                a = F.interpolate(
                    a.unsqueeze(0).unsqueeze(0), (H, W),
                    mode="bilinear", align_corners=False,
                )[0, 0]

            edge_band = compute_edge_band_np(mask_np, edge_radius)
            eb_t = torch.from_numpy(edge_band)
            blended = mask[i].cpu() * (1 - eb_t) + a * eb_t
            alphas.append(blended)

        return torch.stack(alphas)

    # ── Edge refinement ───────────────────────────────────────────────

    def _edge_refine(self, image, alpha, method, edge_radius, B, H, W):
        """Post-matting edge refinement."""
        if method == "vitmatte":
            return self._run_vitmatte(
                image, alpha, "vitmatte_small", edge_radius, B, H, W,
                "cuda" if torch.cuda.is_available() else "cpu",
            )
        elif method == "guided_filter" and HAS_CV2:
            import cv2
            refined = []
            for i in range(B):
                img_np = (image[min(i, image.shape[0] - 1)].cpu().numpy() * 255).astype(np.uint8)
                a_np = alpha[i].cpu().numpy()
                guide = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
                r = cv2.ximgproc.guidedFilter(
                    guide, a_np.astype(np.float32), edge_radius, 1e-4,
                ) if hasattr(cv2, 'ximgproc') else a_np
                refined.append(torch.from_numpy(r))
            return torch.stack(refined)
        elif method == "multi_scale_guided":
            from .utils import multi_scale_guided_refine
            refined = []
            for i in range(B):
                img_np = (image[min(i, image.shape[0] - 1)].cpu().numpy() * 255).astype(np.uint8)
                a_np = alpha[i].cpu().numpy()
                r = multi_scale_guided_refine(img_np, a_np, edge_radius, 0.5)
                if r is not None:
                    refined.append(torch.from_numpy(r))
                else:
                    refined.append(alpha[i])
            return torch.stack(refined)

        return alpha

    # ── Post-processing ───────────────────────────────────────────────

    def _post_process(self, alpha, do_fill, min_size, B, H, W):
        """Fill holes and remove small regions."""
        processed = []
        for i in range(B):
            a = alpha[i].cpu().numpy()
            if do_fill:
                a = fill_holes(a)
            if min_size > 0:
                a = remove_small_regions(a, min_size)
            processed.append(torch.from_numpy(a))
        return torch.stack(processed)

    # ── Trimap generation ─────────────────────────────────────────────

    @staticmethod
    def _generate_trimap(mask_np: np.ndarray, edge_radius: int) -> np.ndarray:
        """Generate 3-region trimap: 255=fg, 128=unknown, 0=bg."""
        if not HAS_CV2:
            out = np.zeros(mask_np.shape, dtype=np.uint8)
            out[mask_np > 0.5] = 255
            return out

        import cv2
        binary = (mask_np > 0.5).astype(np.uint8) * 255
        ksize = edge_radius * 2 + 1
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        eroded = cv2.erode(binary, kern, iterations=1)
        dilated = cv2.dilate(binary, kern, iterations=1)
        unknown = (dilated > 127) & (eroded < 128)

        trimap = np.zeros(mask_np.shape, dtype=np.uint8)
        trimap[eroded > 127] = 255
        trimap[unknown] = 128
        return trimap
