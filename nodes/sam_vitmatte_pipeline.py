"""
SAMViTMattePipelineMEC – Combined SAM + ViTMatte pipeline for the
highest-quality mask generation in any lighting / environment.

Pipeline stages:
  1. **SAM coarse mask** – initial segmentation from points + bbox
  2. **Iterative refinement** – re-run SAM with mask-derived prompts
  3. **Edge-aware matting** – ViTMatte / guided-filter alpha refinement
  4. **Multi-scale fusion** – blend multiple scales for fine detail
  5. **Post-processing** – morphological cleanup, hole filling

Designed for compositing-grade alpha mattes from a single click.
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import gc

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


class SAMViTMattePipelineMEC:
    """End-to-end SAM → ViTMatte pipeline.

    Combines SAM segmentation with ViTMatte-quality edge refinement
    in a single node for maximum accuracy and precision.
    """

    # Class-level cache so the ViTMatte model is loaded only once
    _vitmatte_model = None
    _vitmatte_processor = None

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
                "fill_holes": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Fill interior holes in the mask",
                }),
                "remove_small_regions": ("INT", {
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
                detail_preservation, edge_contrast, fill_holes,
                remove_small_regions, multimask_output, mask_index,
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
        if fill_holes and HAS_CV2:
            coarse_np = self._fill_holes(coarse_np)
        if remove_small_regions > 0 and HAS_CV2:
            coarse_np = self._remove_small(coarse_np, remove_small_regions)
        coarse_mask = torch.from_numpy(coarse_np)

        # ── Stage 3: Edge-aware matting refinement ────────────────────
        refined_mask = self._refine_edges(
            img_tensor, coarse_mask, refine_method, edge_radius,
            detail_preservation, edge_contrast, trimap,
        )

        # ── Stage 4: Edge contrast boost ──────────────────────────────
        if edge_contrast != 1.0:
            refined_mask = self._boost_edge_contrast(
                refined_mask, coarse_mask, edge_contrast
            )

        refined_mask = refined_mask.clamp(0, 1)

        # ── Outputs ───────────────────────────────────────────────────
        edge_mask = torch.abs(refined_mask - (coarse_mask > 0.5).float())
        det_bbox = self._mask_to_bbox(refined_mask, W, H)
        preview = self._make_pipeline_preview(img_tensor, refined_mask, coarse_mask)

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

        # Build predictor once, set image once
        predictor = self._get_predictor(model, model_type, img_np)
        if predictor is None:
            empty = np.zeros((H, W), dtype=np.float32)
            return torch.from_numpy(empty), 0.0

        # Initial prompts
        point_coords, point_labels = self._points_to_arrays(points_list)

        # If we have an existing mask, use it as starting point
        current_mask = None
        if existing_mask is not None:
            m = existing_mask[0] if existing_mask.dim() == 3 else existing_mask
            if m.shape[0] != H or m.shape[1] != W:
                m = F.interpolate(m.unsqueeze(0).unsqueeze(0), (H, W),
                                  mode="bilinear", align_corners=False)[0, 0]
            current_mask = (m.cpu().numpy() > 0.5).astype(np.float32)

        best_score = 0.0

        for iteration in range(iterations):
            # On iterations > 0, augment prompts from previous mask
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

            # Run SAM prediction
            try:
                # Provide mask_input from previous iteration for SAM2
                mask_input = None
                if iteration > 0 and current_mask is not None:
                    # SAM expects logits as mask input (inverse sigmoid)
                    m_logit = np.clip(current_mask, 1e-6, 1 - 1e-6)
                    m_logit = np.log(m_logit / (1 - m_logit))
                    # Resize to 256x256 (SAM's internal mask resolution)
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
                # Some predictors don't support mask_input
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

            # Select best mask
            scores_list = scores.tolist() if hasattr(scores, 'tolist') else list(scores)
            if score_threshold > 0:
                valid = [i for i, s in enumerate(scores_list) if s >= score_threshold]
                idx = valid[0] if valid else 0
            else:
                idx = min(mask_index, len(scores_list) - 1)

            current_mask = masks_np[idx].astype(np.float32)
            best_score = float(scores_list[idx])

            # If score is very high, no need for more iterations
            if best_score > 0.98:
                break

        if current_mask is None:
            current_mask = np.zeros((H, W), dtype=np.float32)

        return torch.from_numpy(current_mask), best_score

    def _augment_prompts_from_mask(self, mask, orig_coords, orig_labels,
                                    orig_box, H, W):
        """Generate additional prompts from the previous mask iteration.

        Strategy:
          - Sample positive points from mask interior (eroded)
          - Sample negative points from just outside mask boundary
          - Derive tighter bbox from mask
        """
        coords_list = []
        labels_list = []

        # Keep original prompts
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
            # Convert (y,x) → (x,y) for SAM
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
    #  STAGE 3 – Edge-aware matting
    # ══════════════════════════════════════════════════════════════════

    def _refine_edges(self, img, mask, method, edge_radius,
                      detail_pres, edge_contrast, trimap):
        """Dispatch to the chosen refinement backend."""
        H, W = img.shape[:2]

        if method == "auto":
            result = self._try_vitmatte(img, mask, edge_radius, detail_pres, trimap)
            if result is None:
                result = self._multi_scale_guided(img, mask, edge_radius, detail_pres)
            if result is None:
                result = self._guided_filter_refine(img, mask, edge_radius, detail_pres)
            if result is None:
                result = self._gaussian_edge_refine(mask, edge_radius)
            return result

        elif method == "vitmatte":
            r = self._try_vitmatte(img, mask, edge_radius, detail_pres, trimap)
            return r if r is not None else self._multi_scale_guided(img, mask, edge_radius, detail_pres) or self._gaussian_edge_refine(mask, edge_radius)

        elif method == "multi_scale_guided":
            r = self._multi_scale_guided(img, mask, edge_radius, detail_pres)
            return r if r is not None else self._gaussian_edge_refine(mask, edge_radius)

        elif method == "guided_filter":
            r = self._guided_filter_refine(img, mask, edge_radius, detail_pres)
            return r if r is not None else self._gaussian_edge_refine(mask, edge_radius)

        elif method == "color_aware":
            r = self._color_aware_refine(img, mask, edge_radius, detail_pres)
            return r if r is not None else self._gaussian_edge_refine(mask, edge_radius)

        elif method == "laplacian_blend":
            r = self._laplacian_refine(img, mask, edge_radius, detail_pres)
            return r if r is not None else self._gaussian_edge_refine(mask, edge_radius)

        return self._gaussian_edge_refine(mask, edge_radius)

    # ── ViTMatte ──────────────────────────────────────────────────────
    @classmethod
    def _load_vitmatte_model(cls):
        """Load ViTMatte model once and cache at class level.
        Tries local path first, then auto-downloads from HuggingFace."""
        if cls._vitmatte_model is not None:
            return cls._vitmatte_model, cls._vitmatte_processor

        from transformers import VitMatteForImageMatting, VitMatteImageProcessor

        model = None
        # Try local model path: ComfyUI/models/vitmatte/
        try:
            import folder_paths
            local_dirs = []
            if "vitmatte" in folder_paths.folder_names_and_paths:
                local_dirs = folder_paths.get_folder_paths("vitmatte")
            else:
                import os
                for base in folder_paths.base_path, folder_paths.models_dir:
                    candidate = os.path.join(base, "vitmatte")
                    if os.path.isdir(candidate):
                        local_dirs.append(candidate)

            for local_dir in local_dirs:
                import os
                if os.path.isdir(local_dir) and any(
                    f.endswith(('.safetensors', '.bin', '.pt'))
                    for f in os.listdir(local_dir)
                ):
                    model = VitMatteForImageMatting.from_pretrained(local_dir)
                    break
        except Exception:
            pass

        if model is None:
            # Auto-download from HuggingFace (cached locally after first download)
            model = VitMatteForImageMatting.from_pretrained(
                "hustvl/vitmatte-small-distinctions-646"
            )

        processor = VitMatteImageProcessor()
        model.eval()

        cls._vitmatte_model = model
        cls._vitmatte_processor = processor
        return model, processor

    def _try_vitmatte(self, img, mask, edge_radius, detail_pres, trimap_input):
        try:
            from transformers import VitMatteForImageMatting, VitMatteImageProcessor
            from PIL import Image as PILImage
        except ImportError:
            return None

        try:
            img_np = (img.cpu().numpy() * 255).astype(np.uint8)
            mask_np = mask.cpu().numpy()

            if trimap_input is not None:
                tri = trimap_input[0].cpu().numpy() if trimap_input.dim() == 3 else trimap_input.cpu().numpy()
            else:
                tri = self._generate_trimap(mask_np, edge_radius)

            model, processor = self._load_vitmatte_model()

            pil_img = PILImage.fromarray(img_np)
            pil_tri = PILImage.fromarray((tri * 255).astype(np.uint8), mode="L")

            inputs = processor(images=pil_img, trimaps=pil_tri, return_tensors="pt")

            device = next(model.parameters()).device if hasattr(model, 'parameters') else "cpu"
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                alpha = model(**inputs).alphas[0, 0]

            # Blend ViTMatte alpha with coarse mask in non-edge regions
            edge_band = self._compute_edge_band(mask_np, edge_radius)
            edge_band_t = torch.from_numpy(edge_band)
            result = mask * (1 - edge_band_t) + alpha.cpu() * edge_band_t

            return result
        except Exception:
            return None

    # ── Multi-scale guided filter (best non-neural) ───────────────────
    def _multi_scale_guided(self, img, mask, edge_radius, detail_pres):
        """Run guided filter at 3 scales, fuse for fine + coarse detail."""
        if not HAS_CV2:
            return None

        try:
            guide = (img.cpu().numpy() * 255).astype(np.uint8)
            if guide.shape[-1] == 4:
                guide = guide[:, :, :3]
            guide_f = guide.astype(np.float32) / 255.0
            m_np = mask.cpu().numpy()
            H, W = m_np.shape

            # Three scales: fine, medium, coarse
            scales = [
                {"radius": max(1, edge_radius // 3), "eps": (1 - detail_pres) ** 2 * 0.01 + 1e-6},
                {"radius": max(1, edge_radius),       "eps": (1 - detail_pres) ** 2 * 0.05 + 1e-4},
                {"radius": max(1, edge_radius * 3),   "eps": (1 - detail_pres) ** 2 * 0.2  + 1e-3},
            ]

            results = []
            for s in scales:
                # Per-channel guided filter for better color-edge adherence
                acc = np.zeros_like(m_np)
                for ch in range(min(3, guide_f.shape[2])):
                    g = guide_f[:, :, ch]
                    filtered = self._guided_filter_impl(g, m_np, s["radius"], s["eps"])
                    acc += filtered
                acc /= min(3, guide_f.shape[2])
                results.append(np.clip(acc, 0, 1))

            # Fuse: fine detail near edges, coarse elsewhere
            edge_band = self._compute_edge_band(m_np, edge_radius)
            fine_band  = np.clip(edge_band * 2, 0, 1)  # narrow band
            coarse_inv = 1 - fine_band

            fused = (results[0] * fine_band * detail_pres +
                     results[1] * fine_band * (1 - detail_pres) +
                     results[2] * coarse_inv * 0.3 +
                     m_np * coarse_inv * 0.7)

            return torch.from_numpy(np.clip(fused, 0, 1).astype(np.float32))
        except Exception:
            return None

    # ── Single-scale guided filter ────────────────────────────────────
    def _guided_filter_refine(self, img, mask, edge_radius, detail_pres):
        if not HAS_CV2:
            return None
        try:
            guide = (img.cpu().numpy() * 255).astype(np.uint8)
            if guide.shape[-1] == 4:
                guide = guide[:, :, :3]
            gray = cv2.cvtColor(guide, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            m_np = mask.cpu().numpy()
            eps = (1 - detail_pres) ** 2 * 0.1 + 1e-6

            filtered = self._guided_filter_impl(gray, m_np, max(1, edge_radius), eps)
            edge_band = self._compute_edge_band(m_np, edge_radius)
            result = m_np * (1 - edge_band) + np.clip(filtered, 0, 1) * edge_band

            return torch.from_numpy(result.astype(np.float32))
        except Exception:
            return None

    # ── Color-aware refinement (LAB space) ────────────────────────────
    def _color_aware_refine(self, img, mask, edge_radius, detail_pres):
        """Use LAB color space for lighting-invariant edge detection."""
        if not HAS_CV2:
            return None
        try:
            rgb = (img.cpu().numpy() * 255).astype(np.uint8)
            if rgb.shape[-1] == 4:
                rgb = rgb[:, :, :3]
            lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
            # Normalize LAB channels
            lab[:, :, 0] /= 100.0   # L: 0-100 → 0-1
            lab[:, :, 1] = (lab[:, :, 1] + 128) / 255.0  # a: -128..127 → 0-1
            lab[:, :, 2] = (lab[:, :, 2] + 128) / 255.0  # b: -128..127 → 0-1

            m_np = mask.cpu().numpy()
            H, W = m_np.shape
            eps = (1 - detail_pres) ** 2 * 0.02 + 1e-6

            # Guided filter per LAB channel
            acc = np.zeros_like(m_np)
            weights = [0.5, 0.25, 0.25]  # L gets more weight (luminance invariance)
            for ch, w in zip(range(3), weights):
                g = lab[:, :, ch]
                filtered = self._guided_filter_impl(g, m_np, max(1, edge_radius), eps)
                acc += filtered * w

            edge_band = self._compute_edge_band(m_np, edge_radius)
            result = m_np * (1 - edge_band) + np.clip(acc, 0, 1) * edge_band

            return torch.from_numpy(result.astype(np.float32))
        except Exception:
            return None

    # ── Laplacian pyramid refinement ──────────────────────────────────
    def _laplacian_refine(self, img, mask, edge_radius, detail_pres):
        if not HAS_CV2:
            return None
        try:
            m_np = mask.cpu().numpy()
            H, W = m_np.shape
            levels = min(4, int(np.log2(min(H, W))) - 2)
            if levels < 1:
                return self._gaussian_edge_refine(mask, edge_radius)

            blur_k = max(1, edge_radius * 2) | 1
            soft = cv2.GaussianBlur(m_np, (blur_k, blur_k), edge_radius * 0.4 + 0.1)

            pyr_m = self._build_lap_pyr(m_np, levels)
            pyr_s = self._build_lap_pyr(soft, levels)

            result_pyr = []
            for i, (pm, ps) in enumerate(zip(pyr_m, pyr_s)):
                w = (i + 1) / len(pyr_m) * (1 - detail_pres)
                result_pyr.append(pm * (1 - w) + ps * w)

            result = self._reconstruct_lap_pyr(result_pyr)
            return torch.from_numpy(np.clip(result, 0, 1).astype(np.float32))
        except Exception:
            return None

    # ── Gaussian fallback ─────────────────────────────────────────────
    def _gaussian_edge_refine(self, mask, edge_radius):
        m = mask.clone()
        sigma = max(0.5, edge_radius * 0.4)
        k = int(sigma * 6) | 1
        if k < 3: k = 3

        x = torch.arange(k, dtype=torch.float32, device=m.device) - k // 2
        kernel = torch.exp(-x ** 2 / (2 * sigma ** 2))
        kernel /= kernel.sum()

        m4 = m.unsqueeze(0).unsqueeze(0)
        pad = k // 2
        kh = kernel.view(1, 1, 1, -1)
        kv = kernel.view(1, 1, -1, 1)
        m4 = F.conv2d(F.pad(m4, (pad, pad, 0, 0), "replicate"), kh)
        m4 = F.conv2d(F.pad(m4, (0, 0, pad, pad), "replicate"), kv)
        blurred = m4[0, 0]

        edge_band = self._get_edge_band_torch(mask, edge_radius)
        return mask * (1 - edge_band) + blurred * edge_band

    # ══════════════════════════════════════════════════════════════════
    #  STAGE 4 – Edge contrast boost
    # ══════════════════════════════════════════════════════════════════

    def _boost_edge_contrast(self, refined, coarse, contrast):
        """Apply contrast curve to edge regions for sharper/softer boundaries."""
        edge_band = self._get_edge_band_torch(coarse, 10)

        # Sigmoid-based contrast at edges
        shifted = (refined - 0.5) * contrast
        boosted = torch.sigmoid(shifted * 5)  # steeper sigmoid

        result = refined * (1 - edge_band) + boosted * edge_band
        return result

    # ══════════════════════════════════════════════════════════════════
    #  Utility methods
    # ══════════════════════════════════════════════════════════════════

    def _get_predictor(self, model, model_type, img_np):
        """Get the appropriate SAM predictor and set its image."""
        predictor = None

        if model_type in ("sam2", "sam2.1"):
            try:
                from sam2.sam2_image_predictor import SAM2ImagePredictor
                predictor = SAM2ImagePredictor(model)
            except ImportError:
                pass

        elif model_type == "sam3":
            try:
                from sam3.predictor import SAM3Predictor
                predictor = SAM3Predictor(model)
            except ImportError:
                pass

        else:
            try:
                from segment_anything import SamPredictor
                predictor = SamPredictor(model)
            except ImportError:
                pass

        if predictor is not None:
            predictor.set_image(img_np)

        return predictor

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

    @staticmethod
    def _generate_trimap(mask_np, edge_radius):
        """Generate a high-quality trimap from binary mask."""
        if not HAS_CV2:
            return mask_np
        binary = (mask_np > 0.5).astype(np.uint8) * 255
        # Use two kernel sizes for asymmetric unknown band
        kern_inner = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (edge_radius * 2 + 1, edge_radius * 2 + 1))
        kern_outer = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (int(edge_radius * 1.5) * 2 + 1,) * 2)
        fg = cv2.erode(binary, kern_inner)
        bg_region = 255 - cv2.dilate(binary, kern_outer)

        trimap = np.full_like(mask_np, 0.5, dtype=np.float32)
        trimap[fg > 127] = 1.0
        trimap[bg_region > 127] = 0.0
        return trimap

    @staticmethod
    def _compute_edge_band(mask_np, radius):
        """Compute a smooth edge band around the mask boundary."""
        if not HAS_CV2:
            return np.zeros_like(mask_np)
        binary = (mask_np > 0.5).astype(np.uint8)
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                          (radius * 2 + 1, radius * 2 + 1))
        dilated = cv2.dilate(binary, kern)
        eroded  = cv2.erode(binary, kern)
        band = (dilated - eroded).astype(np.float32)
        # Smooth the band
        blur_k = max(3, radius) | 1
        band = cv2.GaussianBlur(band, (blur_k, blur_k), radius * 0.3)
        return np.clip(band, 0, 1)

    @staticmethod
    def _get_edge_band_torch(mask, radius):
        """PyTorch edge band computation."""
        k = radius * 2 + 1
        pad = radius
        kernel = torch.ones(1, 1, k, k, device=mask.device) / (k * k)
        m4 = mask.unsqueeze(0).unsqueeze(0)
        avg = F.conv2d(F.pad(m4, (pad, pad, pad, pad), "replicate"), kernel)[0, 0]
        band = 1.0 - torch.abs(avg * 2 - 1)
        return band.clamp(0, 1)

    def _guided_filter_impl(self, guide, src, radius, eps):
        """Manual guided filter (works without cv2.ximgproc)."""
        ksize = radius * 2 + 1
        mean_I  = cv2.blur(guide, (ksize, ksize))
        mean_p  = cv2.blur(src,   (ksize, ksize))
        mean_Ip = cv2.blur(guide * src,   (ksize, ksize))
        mean_II = cv2.blur(guide * guide, (ksize, ksize))

        cov_Ip = mean_Ip - mean_I * mean_p
        var_I  = mean_II - mean_I * mean_I

        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I

        mean_a = cv2.blur(a, (ksize, ksize))
        mean_b = cv2.blur(b, (ksize, ksize))
        return mean_a * guide + mean_b

    @staticmethod
    def _build_lap_pyr(img, levels):
        pyr = []
        cur = img.copy()
        for _ in range(levels):
            down = cv2.pyrDown(cur)
            up = cv2.pyrUp(down, dstsize=(cur.shape[1], cur.shape[0]))
            pyr.append(cur - up)
            cur = down
        pyr.append(cur)
        return pyr

    @staticmethod
    def _reconstruct_lap_pyr(pyr):
        cur = pyr[-1]
        for i in range(len(pyr) - 2, -1, -1):
            up = cv2.pyrUp(cur, dstsize=(pyr[i].shape[1], pyr[i].shape[0]))
            cur = up + pyr[i]
        return cur

    @staticmethod
    def _fill_holes(mask_np):
        """Fill interior holes using contour hierarchy."""
        binary = (mask_np > 0.5).astype(np.uint8) * 255
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP,
                                                cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is not None:
            for i, h in enumerate(hierarchy[0]):
                if h[3] >= 0:  # has parent → is a hole
                    cv2.drawContours(binary, contours, i, 255, -1)
        return (binary > 127).astype(np.float32)

    @staticmethod
    def _remove_small(mask_np, min_area):
        """Remove connected components smaller than min_area pixels."""
        binary = (mask_np > 0.5).astype(np.uint8)
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        result = np.zeros_like(binary)
        for i in range(1, n_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                result[labels == i] = 1
        return result.astype(np.float32)

    @staticmethod
    def _mask_to_bbox(mask, W, H):
        coords = torch.nonzero(mask > 0.5, as_tuple=False)
        if coords.shape[0] > 0:
            y_min = int(coords[:, 0].min().item())
            y_max = int(coords[:, 0].max().item())
            x_min = int(coords[:, 1].min().item())
            x_max = int(coords[:, 1].max().item())
            return [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1]
        return [0, 0, W, H]

    @staticmethod
    def _make_pipeline_preview(img, refined, coarse):
        """Side-by-side preview: left=coarse overlay, right=refined overlay."""
        H, W = img.shape[:2]

        preview = img.clone()
        if preview.shape[-1] == 4:
            preview = preview[:, :, :3]

        half = W // 2

        # Left half: coarse mask (blue tint)
        left = preview[:, :half, :].clone()
        cm = coarse[:, :half]
        left[:, :, 2] = torch.clamp(left[:, :, 2] + cm * 0.35, 0, 1)

        # Right half: refined mask (green tint)
        right = preview[:, half:, :].clone()
        rm = refined[:, half:]
        right[:, :, 1] = torch.clamp(right[:, :, 1] + rm * 0.35, 0, 1)

        # Dividing line
        result = preview.clone()
        result[:, :half] = left
        result[:, half:] = right
        if half > 0 and half < W:
            result[:, half-1:half+1, 0] = 1.0
            result[:, half-1:half+1, 1] = 1.0
            result[:, half-1:half+1, 2] = 0.0

        return result
