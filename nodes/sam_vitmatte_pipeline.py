"""
SAMViTMattePipelineMEC – Combined SAM + ViTMatte pipeline for the
highest-quality mask generation in any lighting / environment.

Pipeline stages:
  1. **SAM coarse mask** – initial segmentation from points + bbox
  2. **Iterative refinement** – re-run SAM with mask-derived prompts
  3. **Edge-aware matting** – ViTMatte / guided-filter alpha refinement
  4. **Multi-scale fusion** – blend multiple scales for fine detail
  5. **Post-processing** – morphological cleanup, hole filling

All shared computation delegates to nodes.utils.
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import gc

from .utils import (
    HAS_CV2,
    get_sam_predictor,
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
    fill_holes,
    remove_small_regions,
    mask_to_bbox,
    make_split_preview,
)

try:
    import cv2
except ImportError:
    pass


class SAMViTMattePipelineMEC:
    """End-to-end SAM → ViTMatte pipeline.

    Combines SAM segmentation with ViTMatte-quality edge refinement
    in a single node for maximum accuracy and precision.
    """

    REFINE_METHODS = ["auto", "vitmatte", "guided_filter", "multi_scale_guided",
                      "color_aware", "laplacian_blend"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sam_model": ("SAM_MODEL",),
                "image": ("IMAGE",),
                "points_json": ("STRING", {
                    "default": "[]",
                    "multiline": True,
                    "tooltip": 'JSON array: [{"x":100,"y":200,"label":1}, ...]',
                }),
                "bbox_json": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": 'Bounding box: [x1,y1,x2,y2]. Leave empty for points-only.',
                }),
                "sam_iterations": ("INT", {
                    "default": 2, "min": 1, "max": 5, "step": 1,
                    "tooltip": (
                        "Number of SAM refinement iterations.  Each pass uses the "
                        "previous mask to generate better prompts.  2-3 is ideal."
                    ),
                }),
                "refine_method": (cls.REFINE_METHODS, {
                    "default": "auto",
                    "tooltip": (
                        "Edge refinement backend.\n"
                        "auto: best available (vitmatte → multi_scale_guided → guided_filter)\n"
                        "vitmatte: HuggingFace ViTMatte neural matting\n"
                        "guided_filter: fast image-guided alpha\n"
                        "multi_scale_guided: guided filter at 3 scales (best non-neural)\n"
                        "color_aware: LAB-space color-sensitive edge refinement\n"
                        "laplacian_blend: Laplacian pyramid frequency blending"
                    ),
                }),
                "edge_radius": ("INT", {
                    "default": 12, "min": 1, "max": 200, "step": 1,
                    "tooltip": "Pixels around edges to refine (larger = softer transitions)",
                }),
                "detail_preservation": ("FLOAT", {
                    "default": 0.85, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "How much fine detail (hair, fur, lace) to preserve. 0=smooth, 1=maximum detail.",
                }),
                "edge_contrast": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 3.0, "step": 0.1,
                    "tooltip": "Boost edge contrast for challenging lighting. >1 sharpens boundaries.",
                }),
                "fill_holes_enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Fill interior holes in the mask",
                }),
                "min_region_size": ("INT", {
                    "default": 64, "min": 0, "max": 10000, "step": 1,
                    "tooltip": "Remove isolated mask regions smaller than N pixels (0=disabled)",
                }),
                "multimask_output": ("BOOLEAN", {"default": True}),
                "mask_index": ("INT", {"default": 0, "min": 0, "max": 2}),
                "score_threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "bbox": ("BBOX", {"tooltip": "Bounding box from BBox node (overrides bbox_json)"}),
                "existing_mask": ("MASK", {"tooltip": "Use as initial mask instead of SAM first pass"}),
                "trimap": ("MASK", {"tooltip": "Custom trimap for ViTMatte"}),
            },
        }

    RETURN_TYPES = ("MASK", "MASK", "MASK", "IMAGE", "BBOX", "FLOAT", "STRING",)
    RETURN_NAMES = ("refined_mask", "coarse_mask", "edge_mask",
                    "preview", "detected_bbox", "score", "info",)
    FUNCTION = "execute"
    CATEGORY = "MaskEditControl/Pipeline"
    DESCRIPTION = (
        "SAM + ViTMatte combined pipeline for compositing-grade alpha mattes. "
        "Iterative SAM refinement → edge-aware matting → multi-scale fusion → cleanup."
    )

    def execute(self, sam_model, image, points_json, bbox_json,
                sam_iterations, refine_method, edge_radius,
                detail_preservation, edge_contrast, fill_holes_enabled,
                min_region_size, multimask_output, mask_index,
                score_threshold, bbox=None, existing_mask=None, trimap=None):

        model_info = sam_model
        model = model_info["model"]
        model_type = model_info["model_type"]
        target_device = model_info["device"]
        offload = model_info["offload_to_cpu"]
        model_dtype = model_info["dtype"]

        img_tensor = image[0]  # (H, W, C)
        img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
        H, W = img_np.shape[:2]

        # Parse prompts
        points_list = self._parse_points(points_json)
        box_np = self._parse_bbox(bbox_json, bbox)

        # ── Stage 1: SAM coarse mask (with iterative refinement) ──────
        if offload and hasattr(model, "to"):
            model.to(target_device)

        try:
            coarse_mask, best_score = self._iterative_sam(
                model, model_type, img_np, points_list, box_np,
                sam_iterations, multimask_output, mask_index,
                score_threshold, target_device, model_dtype,
                existing_mask, H, W,
            )
        finally:
            if offload and hasattr(model, "to"):
                model.to("cpu")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # ── Stage 2: Post-process coarse mask ─────────────────────────
        coarse_np = coarse_mask.cpu().numpy()
        if fill_holes_enabled and HAS_CV2:
            coarse_np = fill_holes(coarse_np)
        if min_region_size > 0 and HAS_CV2:
            coarse_np = remove_small_regions(coarse_np, min_region_size)
        coarse_mask = torch.from_numpy(coarse_np)

        # ── Stage 3: Edge-aware matting refinement ────────────────────
        refined_mask = self._refine_edges(
            img_tensor, img_np, coarse_mask, coarse_np,
            refine_method, edge_radius, detail_preservation, trimap,
        )

        # ── Stage 4: Edge contrast boost ──────────────────────────────
        if edge_contrast != 1.0:
            refined_mask = boost_edge_contrast(
                refined_mask, coarse_mask, edge_contrast, 10
            )

        refined_mask = refined_mask.clamp(0, 1)

        # ── Outputs ───────────────────────────────────────────────────
        edge_mask = torch.abs(refined_mask - (coarse_mask > 0.5).float())
        det_bbox = mask_to_bbox(refined_mask, W, H)
        preview = make_split_preview(img_tensor, refined_mask, coarse_mask)

        info = json.dumps({
            "model_type": model_type,
            "sam_iterations": sam_iterations,
            "refine_method": refine_method,
            "edge_radius": edge_radius,
            "detail_preservation": detail_preservation,
            "score": best_score,
            "detected_bbox": det_bbox,
            "mask_area_px": int((refined_mask > 0.5).sum().item()),
            "mask_area_pct": round(float((refined_mask > 0.5).float().mean().item()) * 100, 2),
        }, indent=2)

        return (
            refined_mask.unsqueeze(0),
            coarse_mask.unsqueeze(0),
            edge_mask.unsqueeze(0),
            preview.unsqueeze(0),
            det_bbox,
            best_score,
            info,
        )

    # ══════════════════════════════════════════════════════════════════
    #  STAGE 1 – Iterative SAM refinement
    # ══════════════════════════════════════════════════════════════════

    def _iterative_sam(self, model, model_type, img_np, points_list,
                       box_np, iterations, multimask, mask_index,
                       score_threshold, device, dtype,
                       existing_mask, H, W):
        """Run SAM multiple times, using previous mask to refine prompts."""

        predictor = get_sam_predictor(model, model_type, img_np)
        if predictor is None:
            return torch.zeros((H, W), dtype=torch.float32), 0.0

        point_coords, point_labels = self._points_to_arrays(points_list)

        # Use existing mask as starting point if provided
        current_mask = None
        if existing_mask is not None:
            m = existing_mask[0] if existing_mask.dim() == 3 else existing_mask
            if m.shape[0] != H or m.shape[1] != W:
                m = F.interpolate(m.unsqueeze(0).unsqueeze(0), (H, W),
                                  mode="bilinear", align_corners=False)[0, 0]
            current_mask = (m.cpu().numpy() > 0.5).astype(np.float32)

        best_score = 0.0

        for iteration in range(iterations):
            iter_coords = point_coords
            iter_labels = point_labels
            iter_box = box_np

            if iteration > 0 and current_mask is not None:
                aug_coords, aug_labels, aug_box = self._augment_prompts_from_mask(
                    current_mask, point_coords, point_labels, box_np, H, W
                )
                iter_coords = aug_coords
                iter_labels = aug_labels
                if aug_box is not None:
                    iter_box = aug_box

            # Run SAM prediction with optional mask input from previous iteration
            try:
                mask_input = None
                if iteration > 0 and current_mask is not None:
                    m_logit = np.clip(current_mask, 1e-6, 1 - 1e-6)
                    m_logit = np.log(m_logit / (1 - m_logit))
                    if HAS_CV2:
                        mask_input = cv2.resize(m_logit, (256, 256),
                                                 interpolation=cv2.INTER_LINEAR)
                    else:
                        mask_input = m_logit
                    mask_input = mask_input[np.newaxis, :, :]  # (1, 256, 256)

                masks_np, scores, _ = predictor.predict(
                    point_coords=iter_coords,
                    point_labels=iter_labels,
                    box=iter_box,
                    mask_input=mask_input,
                    multimask_output=multimask,
                )
            except TypeError:
                masks_np, scores, _ = predictor.predict(
                    point_coords=iter_coords,
                    point_labels=iter_labels,
                    box=iter_box,
                    multimask_output=multimask,
                )
            except Exception:
                break

            if masks_np is None or len(masks_np) == 0:
                break

            scores_list = scores.tolist() if hasattr(scores, 'tolist') else list(scores)
            if score_threshold > 0:
                valid = [i for i, s in enumerate(scores_list) if s >= score_threshold]
                idx = valid[0] if valid else 0
            else:
                idx = min(mask_index, len(scores_list) - 1)

            current_mask = masks_np[idx].astype(np.float32)
            best_score = float(scores_list[idx])

            if best_score > 0.98:
                break

        if current_mask is None:
            current_mask = np.zeros((H, W), dtype=np.float32)

        return torch.from_numpy(current_mask), best_score

    def _augment_prompts_from_mask(self, mask, orig_coords, orig_labels,
                                    orig_box, H, W):
        """Generate additional prompts from the previous mask iteration."""
        coords_list = []
        labels_list = []

        if orig_coords is not None:
            coords_list.append(orig_coords)
            labels_list.append(orig_labels)

        if not HAS_CV2:
            c = np.concatenate(coords_list) if coords_list else None
            l = np.concatenate(labels_list) if labels_list else None
            return c, l, orig_box

        binary = (mask > 0.5).astype(np.uint8)

        # Eroded interior → strong positive points
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        interior = cv2.erode(binary, kern, iterations=1)
        interior_pts = np.argwhere(interior > 0)  # (y, x)
        if len(interior_pts) > 0:
            n = min(3, len(interior_pts))
            indices = np.linspace(0, len(interior_pts) - 1, n, dtype=int)
            sampled = interior_pts[indices]
            coords_list.append(sampled[:, ::-1].astype(np.float32))
            labels_list.append(np.ones(n, dtype=np.int32))

        # Dilated boundary exterior → negative points
        dilated = cv2.dilate(binary, kern, iterations=1)
        exterior = dilated - binary
        exterior_pts = np.argwhere(exterior > 0)
        if len(exterior_pts) > 0:
            n = min(2, len(exterior_pts))
            indices = np.linspace(0, len(exterior_pts) - 1, n, dtype=int)
            sampled = exterior_pts[indices]
            coords_list.append(sampled[:, ::-1].astype(np.float32))
            labels_list.append(np.zeros(n, dtype=np.int32))

        all_coords = np.concatenate(coords_list).astype(np.float32) if coords_list else None
        all_labels = np.concatenate(labels_list).astype(np.int32) if labels_list else None

        # Derive tighter bbox from mask
        ys, xs = np.where(binary > 0)
        if len(xs) > 0:
            pad = max(5, int(min(H, W) * 0.02))
            box = np.array([
                max(0, xs.min() - pad),
                max(0, ys.min() - pad),
                min(W, xs.max() + pad),
                min(H, ys.max() + pad),
            ], dtype=np.float32)
        else:
            box = orig_box

        return all_coords, all_labels, box

    # ══════════════════════════════════════════════════════════════════
    #  STAGE 3 – Edge-aware matting (delegates to utils)
    # ══════════════════════════════════════════════════════════════════

    def _refine_edges(self, img, img_np, mask, mask_np,
                      method, edge_radius, detail_pres, trimap_input):
        """Dispatch to the chosen refinement backend via shared utils."""

        if method == "auto":
            tri = trimap_input[0] if (trimap_input is not None and trimap_input.dim() == 3) else trimap_input
            result = refine_with_vitmatte(img, mask, edge_radius, trimap_input=tri)
            if result is None:
                r = multi_scale_guided_refine(img_np, mask_np, edge_radius, detail_pres)
                if r is not None:
                    result = torch.from_numpy(r)
            if result is None:
                result = self._try_guided_single(img_np, mask_np, edge_radius, detail_pres)
            if result is None:
                result = gaussian_edge_refine(mask, edge_radius)
            return result

        if method == "vitmatte":
            tri = trimap_input[0] if (trimap_input is not None and trimap_input.dim() == 3) else trimap_input
            r = refine_with_vitmatte(img, mask, edge_radius, trimap_input=tri)
            if r is not None:
                return r
            r2 = multi_scale_guided_refine(img_np, mask_np, edge_radius, detail_pres)
            return torch.from_numpy(r2) if r2 is not None else gaussian_edge_refine(mask, edge_radius)

        if method == "multi_scale_guided":
            r = multi_scale_guided_refine(img_np, mask_np, edge_radius, detail_pres)
            return torch.from_numpy(r) if r is not None else gaussian_edge_refine(mask, edge_radius)

        if method == "guided_filter":
            r = self._try_guided_single(img_np, mask_np, edge_radius, detail_pres)
            return r if r is not None else gaussian_edge_refine(mask, edge_radius)

        if method == "color_aware":
            r = color_aware_refine(img_np, mask_np, edge_radius, detail_pres)
            return torch.from_numpy(r) if r is not None else gaussian_edge_refine(mask, edge_radius)

        if method == "laplacian_blend":
            r = self._try_laplacian(mask_np, edge_radius, detail_pres)
            return r if r is not None else gaussian_edge_refine(mask, edge_radius)

        return gaussian_edge_refine(mask, edge_radius)

    @staticmethod
    def _try_guided_single(img_np, mask_np, edge_radius, detail_pres):
        """Single-scale guided filter refinement."""
        if not HAS_CV2:
            return None
        try:
            guide = img_np[:, :, :3]
            gray = cv2.cvtColor(guide, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            eps = (1 - detail_pres) ** 2 * 0.1 + 1e-6
            filtered = guided_filter(gray, mask_np, max(1, edge_radius), eps)
            edge_band = compute_edge_band_np(mask_np, edge_radius)
            result = mask_np * (1 - edge_band) + np.clip(filtered, 0, 1) * edge_band
            return torch.from_numpy(result.astype(np.float32))
        except Exception:
            return None

    @staticmethod
    def _try_laplacian(mask_np, edge_radius, detail_pres):
        """Laplacian pyramid refinement."""
        if not HAS_CV2:
            return None
        try:
            H, W = mask_np.shape
            levels = min(4, int(np.log2(min(H, W))) - 2)
            if levels < 1:
                return None

            blur_k = max(1, edge_radius * 2) | 1
            soft = cv2.GaussianBlur(mask_np, (blur_k, blur_k), edge_radius * 0.4 + 0.1)

            pyr_m = build_laplacian_pyramid(mask_np, levels)
            pyr_s = build_laplacian_pyramid(soft, levels)

            result_pyr = []
            for i, (pm, ps) in enumerate(zip(pyr_m, pyr_s)):
                w = (i + 1) / len(pyr_m) * (1 - detail_pres)
                result_pyr.append(pm * (1 - w) + ps * w)

            result = reconstruct_laplacian_pyramid(result_pyr)
            return torch.from_numpy(np.clip(result, 0, 1).astype(np.float32))
        except Exception:
            return None

    # ══════════════════════════════════════════════════════════════════
    #  Prompt parsing
    # ══════════════════════════════════════════════════════════════════

    @staticmethod
    def _parse_points(points_json):
        try:
            return json.loads(points_json) if isinstance(points_json, str) else points_json
        except json.JSONDecodeError:
            return []

    @staticmethod
    def _points_to_arrays(points_list):
        if not points_list:
            return None, None
        coords = [[float(p["x"]), float(p["y"])] for p in points_list]
        labels = [int(p.get("label", 1)) for p in points_list]
        return np.array(coords, dtype=np.float32), np.array(labels, dtype=np.int32)

    @staticmethod
    def _parse_bbox(bbox_json, bbox_input):
        if bbox_input is not None:
            bx, by, bw, bh = bbox_input
            return np.array([bx, by, bx + bw, by + bh], dtype=np.float32)
        if bbox_json and bbox_json.strip():
            try:
                bdata = json.loads(bbox_json)
                if isinstance(bdata, list) and len(bdata) == 4:
                    return np.array(bdata, dtype=np.float32)
                elif isinstance(bdata, dict):
                    bx = float(bdata.get("x", 0))
                    by = float(bdata.get("y", 0))
                    bw = float(bdata.get("w", bdata.get("width", 0)))
                    bh = float(bdata.get("h", bdata.get("height", 0)))
                    return np.array([bx, by, bx + bw, by + bh], dtype=np.float32)
            except json.JSONDecodeError:
                pass
        return None
