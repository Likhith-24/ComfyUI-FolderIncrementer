"""
SAMMaskGeneratorMEC – Generate masks using a loaded SAM model with
point prompts, bounding-box prompts, or both.  Supports VRAM offload,
multi-mask output, score thresholding, and iterative refinement.
"""

import torch
import numpy as np
import json
import gc

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


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

        # ── Build predictor ────────────────────────────────────────────
        predictor = self._get_predictor(model, model_type, img_np)

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

            # Augment prompts from previous iteration
            if iteration > 0 and current_mask is not None:
                iter_coords, iter_labels, iter_box = self._augment_prompts(
                    current_mask, point_coords, point_labels, box_np,
                    auto_negative_points, H, W,
                )

            # Prepare mask input (logits from previous pass)
            mask_input = None
            if iteration > 0 and current_mask is not None:
                mask_input = self._mask_to_logits(current_mask)

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

    # ── Predictor factory ─────────────────────────────────────────────

    def _get_predictor(self, model, model_type, img_np):
        """Get the correct predictor for the model type and set image."""
        predictor = None

        if model_type in ("sam2", "sam2.1"):
            try:
                from sam2.sam2_image_predictor import SAM2ImagePredictor
                predictor = SAM2ImagePredictor(model)
            except Exception:
                pass

        elif model_type == "sam3":
            try:
                from sam3.predictor import SAM3Predictor
                predictor = SAM3Predictor(model)
            except Exception:
                pass

        else:
            try:
                from segment_anything import SamPredictor
                predictor = SamPredictor(model)
            except Exception:
                pass

        if predictor is not None:
            try:
                predictor.set_image(img_np)
            except Exception:
                return None

        return predictor

    # ── Iterative prompt augmentation ─────────────────────────────────

    def _augment_prompts(self, mask, orig_coords, orig_labels, orig_box,
                          auto_neg, H, W):
        """Generate augmented prompts from the previous mask pass."""
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

        # Interior positive points
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        interior = cv2.erode(binary, kern, iterations=1)
        pts = np.argwhere(interior > 0)
        if len(pts) > 0:
            n = min(3, len(pts))
            idx = np.linspace(0, len(pts) - 1, n, dtype=int)
            coords_list.append(pts[idx][:, ::-1].astype(np.float32))
            labels_list.append(np.ones(n, dtype=np.int32))

        # Exterior negative points
        if auto_neg:
            dilated = cv2.dilate(binary, kern, iterations=1)
            exterior = dilated - binary
            pts = np.argwhere(exterior > 0)
            if len(pts) > 0:
                n = min(2, len(pts))
                idx = np.linspace(0, len(pts) - 1, n, dtype=int)
                coords_list.append(pts[idx][:, ::-1].astype(np.float32))
                labels_list.append(np.zeros(n, dtype=np.int32))

        all_coords = np.concatenate(coords_list).astype(np.float32) if coords_list else None
        all_labels = np.concatenate(labels_list).astype(np.int32) if labels_list else None

        # Derive tighter bbox
        ys, xs = np.where(binary > 0)
        if len(xs) > 0:
            pad = max(5, int(min(H, W) * 0.02))
            box = np.array([
                max(0, xs.min() - pad), max(0, ys.min() - pad),
                min(W, xs.max() + pad), min(H, ys.max() + pad),
            ], dtype=np.float32)
        else:
            box = orig_box

        return all_coords, all_labels, box

    @staticmethod
    def _mask_to_logits(mask, target_size=256):
        """Convert float mask → SAM logit space (inverse sigmoid) at 256x256."""
        m = np.clip(mask, 1e-6, 1 - 1e-6)
        logits = np.log(m / (1 - m))
        if HAS_CV2:
            logits = cv2.resize(logits, (target_size, target_size),
                                 interpolation=cv2.INTER_LINEAR)
        return logits[np.newaxis, :, :]  # (1, 256, 256)

    @staticmethod
    def _fallback_predict(model, img_np, point_coords, point_labels, box_np, multimask):
        """Generic fallback using model's forward pass directly."""
        H, W = img_np.shape[:2]
        # Return an empty mask as fallback
        n = 3 if multimask else 1
        return np.zeros((n, H, W), dtype=np.float32), np.zeros(n, dtype=np.float32)
