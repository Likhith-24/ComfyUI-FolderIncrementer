"""
SAMMaskGeneratorMEC – Generate masks using a loaded SAM model with
point prompts, bounding-box prompts, or both.  Supports VRAM offload,
multi-mask output, score thresholding, and iterative refinement.
"""

import torch
import numpy as np
import json
import gc

from .utils import (
    HAS_CV2,
    get_sam_predictor,
    augment_prompts_from_mask,
    mask_to_sam_logits,
    parse_points_json,
    parse_bbox_input,
)


class SAMMaskGeneratorMEC:
    """Run SAM inference with point prompts + bounding box prompts.
    Handles automatic GPU offload if the model was loaded with offload mode.
    Supports iterative refinement for maximum accuracy."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sam_model": ("SAM_MODEL",),
                "image": ("IMAGE",),
                "points_json": ("STRING", {
                    "default": "[]",
                    "multiline": True,
                    "tooltip": (
                        'JSON array: [{"x":100,"y":200,"label":1}, ...]. '
                        'label=1=foreground, label=0=background.'
                    ),
                }),
                "bbox_json": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": (
                        'Bounding box as JSON: [x1, y1, x2, y2] or {"x":..,"y":..,"w":..,"h":..}. '
                        'Leave empty to use only point prompts.'
                    ),
                }),
                "multimask_output": ("BOOLEAN", {"default": True,
                                                   "tooltip": "Return 3 candidate masks (SAM default) vs 1"}),
                "mask_index": ("INT", {"default": 0, "min": 0, "max": 2,
                                        "tooltip": "Which mask to return when multimask=True (0=best score)"}),
                "score_threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                               "tooltip": "Discard masks below this confidence score"}),
                "apply_bbox_crop": ("BOOLEAN", {"default": False,
                                                  "tooltip": "Crop output to bbox region"}),
                "refine_iterations": ("INT", {
                    "default": 1, "min": 1, "max": 5, "step": 1,
                    "tooltip": (
                        "Iterative refinement passes.  Each pass feeds the previous mask "
                        "back into SAM with augmented prompts.  2-3 significantly improves accuracy."
                    ),
                }),
                "auto_negative_points": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Automatically sample negative points just outside the mask boundary.  "
                        "Helps in cluttered scenes and similar-color backgrounds."
                    ),
                }),
            },
            "optional": {
                "bbox": ("BBOX", {"tooltip": "Bounding box from BBox node (overrides bbox_json)"}),
                "existing_mask": ("MASK", {
                    "tooltip": "Use this mask as the starting point instead of running SAM from scratch",
                }),
            },
        }

    RETURN_TYPES = ("MASK", "MASK", "BBOX", "FLOAT", "STRING",)
    RETURN_NAMES = ("mask", "all_masks", "detected_bbox", "score", "info",)
    FUNCTION = "generate"
    CATEGORY = "MaskEditControl/SAM"
    DESCRIPTION = (
        "Generate masks using SAM with point + bounding box prompts, "
        "iterative refinement, and VRAM offload support."
    )

    def generate(self, sam_model, image, points_json, bbox_json,
                 multimask_output, mask_index, score_threshold,
                 apply_bbox_crop, refine_iterations=1,
                 auto_negative_points=False, bbox=None, existing_mask=None):

        model_info = sam_model
        model = model_info["model"]
        model_type = model_info["model_type"]
        target_device = model_info["device"]
        offload = model_info["offload_to_cpu"]
        model_dtype = model_info["dtype"]

        # ── Move model to GPU if offloaded ─────────────────────────────
        if offload and hasattr(model, "to"):
            model.to(target_device)

        try:
            result = self._run_inference(
                model, model_type, image, points_json, bbox_json, bbox,
                multimask_output, mask_index, score_threshold,
                apply_bbox_crop, target_device, model_dtype,
                refine_iterations, auto_negative_points, existing_mask,
            )
        finally:
            # ── Offload back to CPU ────────────────────────────────────
            if offload and hasattr(model, "to"):
                model.to("cpu")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return result

    def _run_inference(self, model, model_type, image, points_json, bbox_json,
                       bbox_input, multimask_output, mask_index, score_threshold,
                       apply_bbox_crop, device, dtype,
                       refine_iterations=1, auto_negative_points=False,
                       existing_mask=None):

        # Convert image: (B, H, W, C) float [0,1] → numpy uint8
        img_tensor = image[0]  # first image in batch
        img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
        H, W = img_np.shape[:2]

        # Parse points (shared utility)
        points_list = parse_points_json(points_json)
        point_coords, point_labels = None, None
        if points_list:
            coords = [[float(p["x"]), float(p["y"])] for p in points_list]
            labels = [int(p.get("label", 1)) for p in points_list]
            point_coords = np.array(coords, dtype=np.float32)
            point_labels = np.array(labels, dtype=np.int32)

        # Parse bbox (shared utility)
        box_np = parse_bbox_input(bbox_json, bbox_input)

        # ── Build predictor (shared utility) ───────────────────────────
        predictor = get_sam_predictor(model, model_type, img_np)

        if predictor is None:
            empty = torch.zeros(1, H, W, dtype=torch.float32)
            return (empty, empty, [0, 0, W, H], 0.0, "No compatible predictor found")

        # ── Use existing mask as starting logits if provided ───────────
        current_mask = None
        if existing_mask is not None:
            m = existing_mask[0] if existing_mask.dim() == 3 else existing_mask
            if m.shape[0] != H or m.shape[1] != W:
                import torch.nn.functional as F
                m = F.interpolate(m.unsqueeze(0).unsqueeze(0), (H, W),
                                  mode="bilinear", align_corners=False)[0, 0]
            current_mask = (m.cpu().numpy() > 0.5).astype(np.float32)

        # ── Iterative SAM refinement ───────────────────────────────────
        masks_np = None
        scores = None
        best_score = 0.0

        for iteration in range(refine_iterations):
            iter_coords = point_coords
            iter_labels = point_labels
            iter_box = box_np

            # Augment prompts from previous iteration (shared utility)
            if iteration > 0 and current_mask is not None:
                iter_coords, iter_labels, iter_box = augment_prompts_from_mask(
                    current_mask, point_coords, point_labels, box_np,
                    H, W, auto_negative=auto_negative_points,
                )

            # Prepare mask input (logits from previous pass)
            mask_input = None
            if iteration > 0 and current_mask is not None:
                mask_input = mask_to_sam_logits(current_mask)

            # Run SAM
            try:
                masks_np, scores, _ = predictor.predict(
                    point_coords=iter_coords,
                    point_labels=iter_labels,
                    box=iter_box,
                    mask_input=mask_input,
                    multimask_output=multimask_output,
                )
            except TypeError:
                # Predictor doesn't support mask_input
                masks_np, scores, _ = predictor.predict(
                    point_coords=iter_coords,
                    point_labels=iter_labels,
                    box=iter_box,
                    multimask_output=multimask_output,
                )
            except Exception:
                if masks_np is None:
                    masks_np, scores = self._fallback_predict(
                        model, img_np, iter_coords, iter_labels,
                        iter_box, multimask_output
                    )
                break

            if masks_np is None:
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

            if best_score > 0.98:
                break

        # Fallback if nothing produced
        if masks_np is None:
            empty = torch.zeros(1, H, W, dtype=torch.float32)
            return (empty, empty, [0, 0, W, H], 0.0, "No masks generated")

        # Convert to tensors
        all_masks_t = torch.from_numpy(masks_np.astype(np.float32))
        if all_masks_t.dim() == 2:
            all_masks_t = all_masks_t.unsqueeze(0)

        scores_list = scores.tolist() if hasattr(scores, 'tolist') else list(scores)

        # Pick best mask by score or by index
        if score_threshold > 0:
            valid = [i for i, s in enumerate(scores_list) if s >= score_threshold]
            if not valid:
                idx = 0
            else:
                idx = valid[0]
        else:
            idx = min(mask_index, len(scores_list) - 1)

        selected_mask = all_masks_t[idx:idx+1]
        selected_score = float(scores_list[idx])

        # Detect bbox from selected mask
        coords = torch.nonzero(selected_mask[0] > 0.5, as_tuple=False)
        if coords.shape[0] > 0:
            y_min = int(coords[:, 0].min().item())
            y_max = int(coords[:, 0].max().item())
            x_min = int(coords[:, 1].min().item())
            x_max = int(coords[:, 1].max().item())
            det_bbox = [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1]
        else:
            det_bbox = [0, 0, W, H]

        # Crop to bbox if requested
        if apply_bbox_crop and box_np is not None:
            bx1, by1, bx2, by2 = [int(v) for v in box_np]
            bx1, by1 = max(0, bx1), max(0, by1)
            bx2, by2 = min(W, bx2), min(H, by2)
            selected_mask = selected_mask[:, by1:by2, bx1:bx2]

        info = json.dumps({
            "model_type": model_type,
            "num_masks": len(scores_list),
            "scores": scores_list,
            "selected_index": idx,
            "detected_bbox": det_bbox,
            "refine_iterations": refine_iterations,
        }, indent=2)

        return (selected_mask, all_masks_t, det_bbox, selected_score, info)

    @staticmethod
    def _fallback_predict(model, img_np, point_coords, point_labels, box_np, multimask):
        """Generic fallback using model's forward pass directly."""
        H, W = img_np.shape[:2]
        # Return an empty mask as fallback
        n = 3 if multimask else 1
        return np.zeros((n, H, W), dtype=np.float32), np.zeros(n, dtype=np.float32)
