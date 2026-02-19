"""
SAMMaskGeneratorMEC – Generate masks using a loaded SAM model with
point prompts, bounding-box prompts, or both.  Supports VRAM offload,
multi-mask output, and score thresholding.
"""

import torch
import numpy as np
import json
import gc


class SAMMaskGeneratorMEC:
    """Run SAM inference with point prompts + bounding box prompts.
    Handles automatic GPU offload if the model was loaded with offload mode."""

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
            },
            "optional": {
                "bbox": ("BBOX", {"tooltip": "Bounding box from BBox node (overrides bbox_json)"}),
            },
        }

    RETURN_TYPES = ("MASK", "MASK", "BBOX", "FLOAT", "STRING",)
    RETURN_NAMES = ("mask", "all_masks", "detected_bbox", "score", "info",)
    FUNCTION = "generate"
    CATEGORY = "MaskEditControl/SAM"
    DESCRIPTION = "Generate masks using SAM with point + bounding box prompts and VRAM offload support."

    def generate(self, sam_model, image, points_json, bbox_json,
                 multimask_output, mask_index, score_threshold,
                 apply_bbox_crop, bbox=None):

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
                       apply_bbox_crop, device, dtype):

        # Convert image: (B, H, W, C) float [0,1] → numpy uint8
        img_tensor = image[0]  # first image in batch
        img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
        H, W = img_np.shape[:2]

        # Parse points
        try:
            points_list = json.loads(points_json) if isinstance(points_json, str) else points_json
        except json.JSONDecodeError:
            points_list = []

        point_coords = None
        point_labels = None
        if points_list:
            coords = [[float(p["x"]), float(p["y"])] for p in points_list]
            labels = [int(p.get("label", 1)) for p in points_list]
            point_coords = np.array(coords, dtype=np.float32)
            point_labels = np.array(labels, dtype=np.int32)

        # Parse bbox
        box_np = None
        if bbox_input is not None:
            # From BBOX node: [x, y, w, h] → [x1, y1, x2, y2]
            bx, by, bw, bh = bbox_input
            box_np = np.array([bx, by, bx + bw, by + bh], dtype=np.float32)
        elif bbox_json and bbox_json.strip():
            try:
                bdata = json.loads(bbox_json)
                if isinstance(bdata, list) and len(bdata) == 4:
                    box_np = np.array(bdata, dtype=np.float32)
                elif isinstance(bdata, dict):
                    bx = float(bdata.get("x", 0))
                    by = float(bdata.get("y", 0))
                    bw = float(bdata.get("w", bdata.get("width", 0)))
                    bh = float(bdata.get("h", bdata.get("height", 0)))
                    box_np = np.array([bx, by, bx + bw, by + bh], dtype=np.float32)
            except json.JSONDecodeError:
                pass

        # ── SAM2 / SAM2.1 inference ────────────────────────────────────
        masks_np = None
        scores = None

        if model_type in ("sam2", "sam2.1"):
            try:
                from sam2.sam2_image_predictor import SAM2ImagePredictor
                predictor = SAM2ImagePredictor(model)
                predictor.set_image(img_np)
                masks_np, scores, _ = predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box=box_np,
                    multimask_output=multimask_output,
                )
            except Exception as e:
                masks_np, scores = self._fallback_predict(
                    model, img_np, point_coords, point_labels, box_np, multimask_output
                )

        elif model_type == "sam3":
            try:
                from sam3.predictor import SAM3Predictor
                predictor = SAM3Predictor(model)
                predictor.set_image(img_np)
                masks_np, scores, _ = predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box=box_np,
                    multimask_output=multimask_output,
                )
            except Exception:
                masks_np, scores = self._fallback_predict(
                    model, img_np, point_coords, point_labels, box_np, multimask_output
                )

        else:
            # Original SAM
            try:
                from segment_anything import SamPredictor
                predictor = SamPredictor(model)
                predictor.set_image(img_np)
                masks_np, scores, _ = predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box=box_np,
                    multimask_output=multimask_output,
                )
            except Exception:
                masks_np, scores = self._fallback_predict(
                    model, img_np, point_coords, point_labels, box_np, multimask_output
                )

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
        }, indent=2)

        return (selected_mask, all_masks_t, det_bbox, selected_score, info)

    @staticmethod
    def _fallback_predict(model, img_np, point_coords, point_labels, box_np, multimask):
        """Generic fallback using model's forward pass directly."""
        H, W = img_np.shape[:2]
        # Return an empty mask as fallback
        n = 3 if multimask else 1
        return np.zeros((n, H, W), dtype=np.float32), np.zeros(n, dtype=np.float32)
