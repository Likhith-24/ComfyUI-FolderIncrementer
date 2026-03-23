"""
SamMultiMaskPickerMEC – Interactive multi-mask picker for SAM inference.

Runs SAM inference with point/bbox prompts, returns ALL 3 candidate masks
with IoU scores. A companion JS widget renders thumbnails for interactive
selection via click or keyboard (1/2/3).

Files CREATED: nodes/sam_multi_mask_picker.py, js/sam_multi_mask_picker.js, tests/test_sam_multi_mask_picker.py
Files MODIFIED: __init__.py (import + mapping registration)
Files UNTOUCHED: All other existing node files
"""

from __future__ import annotations

import gc
import json
import logging
from typing import Optional

import numpy as np
import torch

from .model_manager import (
    MODEL_REGISTRY,
    get_or_load_model,
    scan_model_dir,
)
from .utils import (
    get_sam_predictor,
    sam_predict,
    parse_points_json,
    parse_bbox_input,
)

logger = logging.getLogger("MEC")


def _get_device() -> str:
    """Detect the best available device without hardcoding 'cuda'."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _available_sam_models() -> list[str]:
    """Return list of SAM model names from registry (sam1 family only for this node)."""
    names = []
    for name, reg in MODEL_REGISTRY.items():
        if reg.get("family") in ("sam1", "sam2", "sam_hq"):
            names.append(name)
    if not names:
        names = ["sam_vit_h", "sam_vit_l", "sam_vit_b"]
    return sorted(names)


class SamMultiMaskPickerMEC:
    """Run SAM inference and expose all 3 candidate masks for interactive picking.

    The JS companion widget renders thumbnail previews of each mask candidate
    overlaid on the source image. Users pick via click or keyboard 1/2/3.
    """

    VRAM_TIER = 2
    CATEGORY = "MaskEditControl/SAM"
    FUNCTION = "pick_mask"
    RETURN_TYPES = ("MASK", "MASK", "INT", "STRING", "STRING")
    RETURN_NAMES = ("selected_mask", "all_masks", "selected_index", "scores", "info")
    DESCRIPTION = (
        "Run SAM inference to get 3 candidate masks with confidence scores. "
        "Pick interactively via the JS widget (click thumbnail or press 1/2/3). "
        "Press R to re-run. Works with SAM1, SAM2, and HQ-SAM models."
    )

    @classmethod
    def INPUT_TYPES(cls):
        sam_models = _available_sam_models()
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Input image (B, H, W, C) float32 [0, 1]. First frame used for inference.",
                }),
                "model_name": (sam_models, {
                    "default": sam_models[0] if sam_models else "sam_vit_h",
                    "tooltip": "SAM model variant to use for inference. Larger models = better quality, more VRAM.",
                }),
                "points_json": ("STRING", {
                    "default": '[{"x": 256, "y": 256, "label": 1}]',
                    "multiline": True,
                    "tooltip": (
                        'JSON array of point prompts: [{"x":100,"y":200,"label":1}, ...]. '
                        'label=1=foreground, label=0=background.'
                    ),
                }),
                "bbox_json": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": (
                        'Optional bounding box as JSON: [x1, y1, x2, y2]. '
                        'Leave empty to use only point prompts.'
                    ),
                }),
                "precision": (["fp32", "fp16", "bf16"], {
                    "default": "fp32",
                    "tooltip": "Model precision. fp16/bf16 use less VRAM but may reduce quality on some GPUs.",
                }),
                "selected_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2,
                    "step": 1,
                    "tooltip": "Which of the 3 candidate masks to output (0-2). Updated by JS widget on click/keyboard.",
                }),
            },
            "optional": {
                "sam_model": ("SAM_MODEL", {
                    "tooltip": "Pre-loaded SAM model from SAM Model Loader node. Overrides model_name if connected.",
                }),
                "bbox": ("BBOX", {
                    "tooltip": "Bounding box from BBox node (overrides bbox_json).",
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Force re-execution when selected_index changes (JS widget interaction)."""
        return float("nan")

    def pick_mask(
        self,
        image: torch.Tensor,
        model_name: str,
        points_json: str,
        bbox_json: str,
        precision: str,
        selected_index: int,
        sam_model: Optional[dict] = None,
        bbox: Optional[list] = None,
    ):
        B, H, W, C = image.shape

        # Convert first image to numpy uint8 for SAM
        img_np = (image[0].cpu().numpy() * 255).astype(np.uint8)

        # Parse prompts
        points_list = parse_points_json(points_json)
        point_coords, point_labels = None, None
        if points_list:
            coords = [[float(p["x"]), float(p["y"])] for p in points_list]
            labels = [int(p.get("label", 1)) for p in points_list]
            point_coords = np.array(coords, dtype=np.float32)
            point_labels = np.array(labels, dtype=np.int32)

        box_np = parse_bbox_input(bbox_json, bbox)

        # Determine device
        device = _get_device()

        # Load model or use pre-loaded
        model = None
        model_type = None
        model_info = None
        loaded_here = False

        if sam_model is not None:
            model = sam_model["model"]
            model_type = sam_model["model_type"]
            model_info = sam_model
        else:
            loaded_here = True

        masks_np = None
        scores_np = None

        try:
            if loaded_here:
                model_obj = get_or_load_model(model_name, precision=precision, device=device)
                if isinstance(model_obj, dict):
                    model = model_obj["model"]
                    model_type = model_obj.get("family", self._detect_model_type(model_name))
                    model_info = model_obj
                else:
                    model = model_obj
                    model_type = self._detect_model_type(model_name)
                    model_info = {
                        "model": model,
                        "model_type": model_type,
                        "device": device,
                        "dtype": torch.float32 if precision == "fp32" else (
                            torch.float16 if precision == "fp16" else torch.bfloat16
                        ),
                    }

            # Build predictor
            with torch.no_grad():
                predictor = get_sam_predictor(model, model_type, img_np)
                if predictor is None:
                    return self._empty_result(H, W, "No compatible SAM predictor found")

                try:
                    masks_np, scores_np, _ = sam_predict(
                        predictor,
                        model_info,
                        point_coords=point_coords,
                        point_labels=point_labels,
                        box=box_np,
                        multimask_output=True,
                    )
                except Exception as e:
                    logger.warning("[MEC] SAM predict failed: %s", e)
                    return self._empty_result(H, W, f"SAM predict error: {e}")

        except torch.cuda.OutOfMemoryError:
            logger.warning("[MEC] OOM during SAM inference, retrying on CPU")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            try:
                if loaded_here:
                    model_obj = get_or_load_model(model_name, precision="fp32", device="cpu")
                    if isinstance(model_obj, dict):
                        model = model_obj["model"]
                        model_type = model_obj.get("family", self._detect_model_type(model_name))
                        model_info = model_obj
                    else:
                        model = model_obj
                        model_type = self._detect_model_type(model_name)
                        model_info = {
                            "model": model,
                            "model_type": model_type,
                            "device": "cpu",
                            "dtype": torch.float32,
                        }
                elif hasattr(model, "to"):
                    model.to("cpu")
                    model_info = dict(model_info) if model_info else {}
                    model_info["device"] = "cpu"
                    model_info["model"] = model

                with torch.no_grad():
                    predictor = get_sam_predictor(model, model_type, img_np)
                    if predictor is None:
                        return self._empty_result(H, W, "No predictor after OOM fallback")

                    masks_np, scores_np, _ = sam_predict(
                        predictor,
                        model_info,
                        point_coords=point_coords,
                        point_labels=point_labels,
                        box=box_np,
                        multimask_output=True,
                    )
            except Exception as e:
                logger.error("[MEC] CPU fallback also failed: %s", e)
                return self._empty_result(H, W, f"CPU fallback failed: {e}")

        finally:
            if loaded_here and model is not None:
                if hasattr(model, "cpu"):
                    try:
                        model.cpu()
                    except Exception:
                        pass
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if masks_np is None:
            return self._empty_result(H, W, "No masks generated")

        # Ensure we have exactly 3 masks (pad if fewer)
        num_masks = masks_np.shape[0]
        if num_masks < 3:
            pad_masks = np.zeros((3 - num_masks, H, W), dtype=np.float32)
            masks_np = np.concatenate([masks_np, pad_masks], axis=0)
            pad_scores = np.zeros(3 - num_masks, dtype=np.float32)
            scores_np = np.concatenate([scores_np, pad_scores])
        elif num_masks > 3:
            masks_np = masks_np[:3]
            scores_np = scores_np[:3]

        # Convert to tensors — MASK shape is (B, H, W)
        all_masks_t = torch.from_numpy(masks_np.astype(np.float32))  # (3, H, W)

        # Clamp selected_index
        idx = max(0, min(selected_index, 2))
        selected_mask_t = all_masks_t[idx:idx + 1]  # (1, H, W)

        # Build scores string
        scores_list = scores_np.tolist() if hasattr(scores_np, "tolist") else list(scores_np)
        scores_str = json.dumps([round(float(s) * 100, 1) for s in scores_list])

        # Build detailed info
        info = json.dumps({
            "model_name": model_name,
            "num_masks": 3,
            "scores_pct": [round(float(s) * 100, 1) for s in scores_list],
            "scores_raw": [round(float(s), 6) for s in scores_list],
            "selected_index": idx,
            "image_size": [W, H],
            "num_points": len(points_list) if points_list else 0,
            "has_bbox": box_np is not None,
            "precision": precision,
            "device": device,
        }, indent=2)

        return (selected_mask_t, all_masks_t, idx, scores_str, info)

    @staticmethod
    def _detect_model_type(model_name: str) -> str:
        """Detect SAM family from model name."""
        name_lower = model_name.lower()
        if "sam2" in name_lower or "sam2.1" in name_lower:
            return "sam2"
        if "sam3" in name_lower:
            return "sam3"
        if "hq" in name_lower:
            return "sam_hq"
        return "sam_vit_h"

    @staticmethod
    def _empty_result(H: int, W: int, reason: str):
        """Return empty masks + error info when inference fails."""
        empty_single = torch.zeros(1, H, W, dtype=torch.float32)
        empty_batch = torch.zeros(3, H, W, dtype=torch.float32)
        scores_str = json.dumps([0.0, 0.0, 0.0])
        info = json.dumps({"error": reason, "num_masks": 0, "scores_pct": [0.0, 0.0, 0.0]})
        return (empty_single, empty_batch, 0, scores_str, info)
