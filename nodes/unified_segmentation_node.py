"""
UnifiedSegmentationNode – Dispatcher-pattern node for SAM2/2.1, SAM3, SeC, VideoMaMa.

Image mode (B=1):  point/bbox prompts → mask
Video mode (B>1):  prompts on frame 0 → propagate to all frames

Architecture follows the dispatcher pattern:
  segment() → _run_sam2() | _run_sam3() | _run_sec() | _run_videomama()

All model I/O delegated to model_manager.py.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import shutil
import tempfile
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F

from .model_manager import (
    MODEL_REGISTRY,
    clear_cache,
    get_or_load_model,
    precision_to_dtype,
    scan_model_dir,
)
from .utils import parse_bbox_input, parse_points_json, points_to_arrays

logger = logging.getLogger("MEC")


# ══════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════

@contextmanager
def _autocast(dtype: torch.dtype, device: str):
    """Autocast context for fp16/bf16 on CUDA; noop otherwise."""
    if dtype in (torch.float16, torch.bfloat16) and device != "cpu":
        with torch.autocast(device, dtype=dtype):
            yield
    else:
        yield


def _parse_coords(points_json: str):
    """Parse point coordinates from JSON string.

    Args:
        points_json: JSON array of {x, y, label} dicts.

    Returns:
        Tuple (pt_coords, pt_labels) as numpy arrays,
        or (None, None) if empty.
    """
    pts = parse_points_json(points_json)
    if not pts:
        return None, None
    return points_to_arrays(pts)


def _parse_bboxes(
    bbox_json: str,
    bbox_input=None,
    neg_bbox_json: str = "",
):
    """Parse bounding boxes from JSON + optional upstream BBOX.

    Args:
        bbox_json: JSON string "[x1, y1, x2, y2]".
        bbox_input: Optional upstream BBOX tensor/list.
        neg_bbox_json: JSON string for negative bbox (SAM3 only).

    Returns:
        Tuple (pos_bbox_np, neg_bbox_np).  Each is np.ndarray or None.
    """
    pos_box = parse_bbox_input(bbox_json, bbox_input)
    neg_box = None
    if neg_bbox_json and neg_bbox_json.strip():
        neg_box = parse_bbox_input(neg_bbox_json)
    return pos_box, neg_box


def _validate_output(
    masks: torch.Tensor,
    expected_B: int,
    H: int,
    W: int,
) -> torch.Tensor:
    """Validate and normalize segmentation output.

    Ensures:
      - Shape is [B, H, W] float32 in [0, 1]
      - Batch dimension matches expected_B

    Args:
        masks: Raw output masks.
        expected_B: Expected batch size.
        H: Height.
        W: Width.

    Returns:
        Validated [B, H, W] float32 tensor.
    """
    if masks is None or masks.numel() == 0:
        return torch.zeros(expected_B, H, W, dtype=torch.float32)

    if masks.dim() == 2:
        masks = masks.unsqueeze(0)
    elif masks.dim() == 4:
        masks = masks.squeeze(1)

    # Resize if spatial dims don't match
    if masks.shape[-2] != H or masks.shape[-1] != W:
        masks = F.interpolate(
            masks.unsqueeze(1).float(), size=(H, W),
            mode="bilinear", align_corners=False,
        ).squeeze(1)

    # Pad or truncate batch
    if masks.shape[0] < expected_B:
        pad = torch.zeros(
            expected_B - masks.shape[0], H, W,
            dtype=masks.dtype, device=masks.device,
        )
        masks = torch.cat([masks, pad], dim=0)
    elif masks.shape[0] > expected_B:
        masks = masks[:expected_B]

    return masks.float().clamp(0.0, 1.0)


# ══════════════════════════════════════════════════════════════════════
#  Segmentation Families  (for scan filtering)
# ══════════════════════════════════════════════════════════════════════

_SEG_FAMILIES = {"sam2", "sam3", "sec", "videomama"}


# ══════════════════════════════════════════════════════════════════════
#  Node
# ══════════════════════════════════════════════════════════════════════

class UnifiedSegmentationNode:
    """Unified segmentation: SAM2/2.1, SAM3, SeC, VideoMaMa in one node.

    Dispatcher pattern:
      segment() → _run_sam2() | _run_sam3() | _run_sec() | _run_videomama()

    Automatically uses video propagation when input IMAGE batch size > 1.
    """

    # ── Scan available models ─────────────────────────────────────────

    @classmethod
    def _scan_models(cls) -> list[str]:
        """List downloadable/available segmentation models."""
        all_models = scan_model_dir()
        # Filter to segmentation families only
        result: list[str] = []
        for entry in all_models:
            name = entry.replace("[download] ", "")
            reg = MODEL_REGISTRY.get(name)
            if reg and reg.get("family") in _SEG_FAMILIES:
                result.append(entry)
        return result or ["(no models — select a [download] option)"]

    # ── INPUT_TYPES ───────────────────────────────────────────────────

    @classmethod
    def INPUT_TYPES(cls):
        models = cls._scan_models()
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Single image (B=1) or video frames (B>1).",
                }),
                "model_name": (models, {
                    "tooltip": (
                        "Segmentation model checkpoint.\n"
                        "[download] prefix auto-downloads from HuggingFace Hub."
                    ),
                }),
                "points_json": ("STRING", {
                    "default": "[]",
                    "multiline": True,
                    "tooltip": (
                        'JSON array: [{"x":100,"y":200,"label":1}, ...].\n'
                        "label 1 = foreground, 0 = background."
                    ),
                }),
                "bbox_json": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": (
                        "Bounding box: [x1, y1, x2, y2].  Leave empty for "
                        "points-only prompts."
                    ),
                }),
                "multimask": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Return 3 candidate masks (True) or 1 (False).",
                }),
                "mask_index": ("INT", {
                    "default": 0, "min": 0, "max": 2,
                    "tooltip": "Which candidate mask to select (0 = highest score).",
                }),
                "precision": (["fp16", "bf16", "fp32"], {
                    "default": "fp16",
                    "tooltip": "Inference precision. fp16 saves VRAM, bf16 for newer GPUs.",
                }),
            },
            "optional": {
                "bbox": ("BBOX", {
                    "tooltip": "Bounding box from upstream node (overrides bbox_json).",
                }),
                "neg_bbox_json": ("STRING", {
                    "default": "",
                    "tooltip": "Negative bounding box [x1,y1,x2,y2] — SAM3 exclusive.",
                }),
                "text_prompt": ("STRING", {
                    "default": "",
                    "tooltip": "Text description of target — SeC models only.",
                }),
                "existing_mask": ("MASK", {
                    "tooltip": "Initial mask for refinement iterations.",
                }),
            },
        }

    RETURN_TYPES = ("MASK", "FLOAT", "STRING")
    RETURN_NAMES = ("masks", "best_score", "info")
    FUNCTION = "segment"
    CATEGORY = "MaskEditControl/Segmentation"
    DESCRIPTION = (
        "Unified segmentation supporting SAM2/2.1, SAM3, SeC, and VideoMaMa.  "
        "Auto-detects image vs video mode from input batch size.  "
        "Auto-downloads models from HuggingFace on first use."
    )

    # ══════════════════════════════════════════════════════════════════
    #  Main Entry Point (dispatcher)
    # ══════════════════════════════════════════════════════════════════

    def segment(
        self,
        image: torch.Tensor,
        model_name: str,
        points_json: str,
        bbox_json: str,
        multimask: bool,
        mask_index: int,
        precision: str,
        bbox=None,
        neg_bbox_json: str = "",
        text_prompt: str = "",
        existing_mask: torch.Tensor | None = None,
    ):
        clean = model_name.replace("[download] ", "")
        if clean not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model '{clean}'.  Available: {sorted(MODEL_REGISTRY)}"
            )

        reg = MODEL_REGISTRY[clean]
        family = reg["family"]
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = get_or_load_model(clean, precision=precision, device=device)

        # Parse prompts
        pt_coords, pt_labels = _parse_coords(points_json)
        pos_box, neg_box = _parse_bboxes(bbox_json, bbox, neg_bbox_json)

        B, H, W, _C = image.shape
        is_video = B > 1
        torch_dtype = precision_to_dtype(precision)

        # ── Dispatch by family ────────────────────────────────────────
        if family == "sam2":
            masks, score = self._run_sam2(
                model, image, pt_coords, pt_labels, pos_box,
                multimask, mask_index, torch_dtype, device,
                B, H, W, is_video, reg,
            )
        elif family == "sam3":
            masks, score = self._run_sam3(
                model, image, pt_coords, pt_labels, pos_box, neg_box,
                multimask, mask_index, torch_dtype, device,
                B, H, W, is_video,
            )
        elif family == "sec":
            masks, score = self._run_sec(
                model, image, pt_coords, pt_labels, pos_box,
                text_prompt, torch_dtype, device, B, H, W, is_video,
            )
        elif family == "videomama":
            masks, score = self._run_videomama(
                model, image, existing_mask,
                torch_dtype, device, B, H, W,
            )
        else:
            raise ValueError(f"Unsupported family: {family}")

        masks = _validate_output(masks, B, H, W)

        info = json.dumps({
            "model": clean,
            "family": family,
            "mode": "video" if is_video else "image",
            "frames": B,
            "best_score": round(score, 4),
            "precision": precision,
        }, indent=2)

        return (masks, score, info)

    # ══════════════════════════════════════════════════════════════════
    #  SAM2 / SAM2.1 Dispatcher
    # ══════════════════════════════════════════════════════════════════

    def _run_sam2(
        self,
        model,
        image: torch.Tensor,
        pt_coords,
        pt_labels,
        box_np,
        multimask: bool,
        mask_idx: int,
        dtype: torch.dtype,
        device: str,
        B: int,
        H: int,
        W: int,
        is_video: bool,
        reg: dict,
    ):
        """SAM2/2.1 segmentation — image or video mode."""
        if is_video:
            return self._sam2_video(
                model, image, pt_coords, pt_labels, box_np,
                dtype, device, B, H, W, reg,
            )
        else:
            return self._sam2_image(
                model, image[0], pt_coords, pt_labels, box_np,
                multimask, mask_idx, dtype, device, H, W,
            )

    def _sam2_image(
        self, model, frame, pt_coords, pt_labels, box_np,
        multimask, mask_idx, dtype, device, H, W,
    ):
        """Single-image segmentation with SAM2ImagePredictor."""
        try:
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError:
            raise RuntimeError("sam2 package is required for image prediction.")

        predictor = SAM2ImagePredictor(model)
        img_np = (frame.cpu().numpy() * 255).astype(np.uint8)

        with _autocast(dtype, device):
            predictor.set_image(img_np)

        kwargs: dict = {"multimask_output": multimask}
        if pt_coords is not None:
            kwargs["point_coords"] = pt_coords
            kwargs["point_labels"] = pt_labels
        if box_np is not None:
            kwargs["box"] = box_np

        with _autocast(dtype, device):
            masks_np, scores, _logits = predictor.predict(**kwargs)

        if masks_np is None or len(masks_np) == 0:
            return torch.zeros(1, H, W, dtype=torch.float32), 0.0

        scores_list = scores.tolist() if hasattr(scores, "tolist") else list(scores)
        idx = min(mask_idx, len(scores_list) - 1)
        best = float(scores_list[idx])
        mask_t = torch.from_numpy(masks_np[idx].astype(np.float32)).unsqueeze(0)
        return mask_t, best

    def _sam2_video(
        self, model, frames, pt_coords, pt_labels, box_np,
        dtype, device, B, H, W, reg,
    ):
        """Video propagation with SAM2VideoPredictor.

        Note: SAM2.0 does NOT support bbox in video mode.
              SAM2.1 adds bbox support.
        """
        try:
            from sam2.sam2_video_predictor import SAM2VideoPredictor
        except ImportError:
            raise RuntimeError("sam2 package is required for video propagation.")

        # Upgrade SAM2Base → SAM2VideoPredictor in-place (avoids reloading).
        # SAM2VideoPredictor inherits from SAM2Base and only adds 4 attrs.
        if not isinstance(model, SAM2VideoPredictor):
            model.__class__ = SAM2VideoPredictor
            model.fill_hole_area = 8
            model.non_overlap_masks = False
            model.clear_non_cond_mem_around_input = False
            model.add_all_frames_to_correct_as_cond = False
        video_pred = model

        # Save frames to temp JPEG dir (required by init_state)
        tmp = tempfile.mkdtemp(prefix="mec_vid_")
        try:
            from PIL import Image as PILImage

            for i in range(B):
                arr = (frames[i].cpu().numpy() * 255).astype(np.uint8)
                PILImage.fromarray(arr).save(
                    os.path.join(tmp, f"{i:06d}.jpg"), quality=95,
                )

            with _autocast(dtype, device):
                state = video_pred.init_state(video_path=tmp)

            # Add prompts on frame 0
            pkw: dict = {"inference_state": state, "frame_idx": 0, "obj_id": 1}
            if pt_coords is not None:
                pkw["points"] = pt_coords
                pkw["labels"] = pt_labels

            # SAM2.1 supports bbox in video; SAM2.0 does not
            version = reg.get("version", "2.0")
            if box_np is not None and version != "2.0":
                pkw["box"] = box_np

            with _autocast(dtype, device):
                video_pred.add_new_points_or_box(**pkw)

            # Propagate
            collected: dict[int, torch.Tensor] = {}
            with _autocast(dtype, device):
                for fidx, _oids, logits in video_pred.propagate_in_video(state):
                    collected[fidx] = (logits[0, 0] > 0.0).float().cpu()

            out: list[torch.Tensor] = []
            for i in range(B):
                out.append(collected.get(i, torch.zeros(H, W, dtype=torch.float32)))
            return torch.stack(out), 1.0

        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    # ══════════════════════════════════════════════════════════════════
    #  SAM3 Dispatcher
    # ══════════════════════════════════════════════════════════════════

    def _run_sam3(
        self,
        model,
        image: torch.Tensor,
        pt_coords,
        pt_labels,
        box_np,
        neg_box,
        multimask: bool,
        mask_idx: int,
        dtype: torch.dtype,
        device: str,
        B: int,
        H: int,
        W: int,
        is_video: bool,
    ):
        """SAM3 segmentation — image or video mode.

        SAM3 uses the SAM2 architecture but adds negative bbox support
        and uses a BPE tokenizer for text grounding.
        """
        if is_video:
            return self._sam3_video(
                model, image, pt_coords, pt_labels, box_np, neg_box,
                dtype, device, B, H, W,
            )
        else:
            return self._sam3_image(
                model, image[0], pt_coords, pt_labels, box_np, neg_box,
                multimask, mask_idx, dtype, device, H, W,
            )

    def _sam3_image(
        self, model, frame, pt_coords, pt_labels, box_np, neg_box,
        multimask, mask_idx, dtype, device, H, W,
    ):
        """Single-image SAM3 segmentation.

        Falls back to SAM2 image predictor since SAM3 checkpoint is
        loaded into SAM2.1 large architecture.
        """
        try:
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError:
            raise RuntimeError("sam2 package is required for SAM3 image prediction.")

        predictor = SAM2ImagePredictor(model)
        img_np = (frame.cpu().numpy() * 255).astype(np.uint8)

        with _autocast(dtype, device):
            predictor.set_image(img_np)

        kwargs: dict = {"multimask_output": multimask}
        if pt_coords is not None:
            kwargs["point_coords"] = pt_coords
            kwargs["point_labels"] = pt_labels
        if box_np is not None:
            kwargs["box"] = box_np

        with _autocast(dtype, device):
            masks_np, scores, _logits = predictor.predict(**kwargs)

        if masks_np is None or len(masks_np) == 0:
            return torch.zeros(1, H, W, dtype=torch.float32), 0.0

        scores_list = scores.tolist() if hasattr(scores, "tolist") else list(scores)
        idx = min(mask_idx, len(scores_list) - 1)
        best = float(scores_list[idx])
        mask_t = torch.from_numpy(masks_np[idx].astype(np.float32)).unsqueeze(0)
        return mask_t, best

    def _sam3_video(
        self, model, frames, pt_coords, pt_labels, box_np, neg_box,
        dtype, device, B, H, W,
    ):
        """Video propagation for SAM3.

        Uses SAM2VideoPredictor under the hood since SAM3 architecture
        is SAM2.1 compatible.
        """
        try:
            from sam2.sam2_video_predictor import SAM2VideoPredictor
        except ImportError:
            raise RuntimeError("sam2 package is required for SAM3 video propagation.")

        video_pred = SAM2VideoPredictor(model)
        tmp = tempfile.mkdtemp(prefix="mec_sam3v_")
        try:
            from PIL import Image as PILImage

            for i in range(B):
                arr = (frames[i].cpu().numpy() * 255).astype(np.uint8)
                PILImage.fromarray(arr).save(
                    os.path.join(tmp, f"{i:06d}.jpg"), quality=95,
                )

            with _autocast(dtype, device):
                state = video_pred.init_state(video_path=tmp)

            pkw: dict = {"inference_state": state, "frame_idx": 0, "obj_id": 1}
            if pt_coords is not None:
                pkw["points"] = pt_coords
                pkw["labels"] = pt_labels
            if box_np is not None:
                pkw["box"] = box_np

            with _autocast(dtype, device):
                video_pred.add_new_points_or_box(**pkw)

            collected: dict[int, torch.Tensor] = {}
            with _autocast(dtype, device):
                for fidx, _oids, logits in video_pred.propagate_in_video(state):
                    collected[fidx] = (logits[0, 0] > 0.0).float().cpu()

            out: list[torch.Tensor] = []
            for i in range(B):
                out.append(collected.get(i, torch.zeros(H, W, dtype=torch.float32)))
            return torch.stack(out), 1.0

        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    # ══════════════════════════════════════════════════════════════════
    #  SeC Dispatcher
    # ══════════════════════════════════════════════════════════════════

    def _run_sec(
        self,
        model,
        image: torch.Tensor,
        pt_coords,
        pt_labels,
        box_np,
        text_prompt: str,
        dtype: torch.dtype,
        device: str,
        B: int,
        H: int,
        W: int,
        is_video: bool,
    ):
        """SeC (MLLM + SAM2) segmentation.

        Uses the grounding_encoder from the SeC model to perform
        concept-driven video/image segmentation.  Saves frames as
        temp JPEGs for the SAM2 video predictor interface.
        """
        from PIL import Image

        # SeC always works in video mode via grounding_encoder
        # For single images, wrap as 1-frame video
        tmp = tempfile.mkdtemp(prefix="mec_sec_")
        try:
            # Save frames as JPEG files for the SAM2 video predictor
            for i in range(B):
                frame_np = (image[i].cpu().numpy() * 255).astype(np.uint8)
                pil_img = Image.fromarray(frame_np, mode="RGB")
                pil_img.save(os.path.join(tmp, f"{i:05d}.jpg"), "JPEG", quality=95)

            # Initialize inference state
            try:
                offload_state_to_cpu = str(getattr(model, "device", device)) == "cpu"
            except Exception:
                offload_state_to_cpu = False

            inference_state = model.grounding_encoder.init_state(
                video_path=tmp,
                offload_video_to_cpu=False,
                offload_state_to_cpu=offload_state_to_cpu,
            )
            model.grounding_encoder.reset_state(inference_state)

            # Combine positive and negative points
            points = None
            labels = None
            if pt_coords is not None and pt_labels is not None:
                points = pt_coords.astype(np.float32)
                labels = pt_labels.astype(np.int32)

            # Handle bbox + points combination (bbox first, then refine with points)
            if box_np is not None and points is not None:
                _, _, _ = model.grounding_encoder.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=0,
                    obj_id=1,
                    points=None,
                    labels=None,
                    box=box_np.astype(np.float32),
                )
                _, _, out_mask_logits = model.grounding_encoder.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=0,
                    obj_id=1,
                    points=points,
                    labels=labels,
                    box=None,
                )
            elif points is not None or box_np is not None:
                _, _, out_mask_logits = model.grounding_encoder.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=0,
                    obj_id=1,
                    points=points,
                    labels=labels if points is not None else None,
                    box=box_np.astype(np.float32) if box_np is not None else None,
                )
            else:
                raise ValueError(
                    "SeC requires at least one visual prompt (points or bbox)."
                )

            init_mask = (out_mask_logits[0] > 0.0).cpu().numpy()

            # Propagate through video
            masks_tensor = torch.zeros(B, H, W, dtype=torch.float32)

            for out_frame_idx, out_obj_ids, out_mask_logits in model.propagate_in_video(
                inference_state,
                start_frame_idx=0,
                max_frame_num_to_track=B,
                reverse=False,
                init_mask=init_mask,
                mllm_memory_size=12,
            ):
                if out_frame_idx < B:
                    mask = (out_mask_logits[0] > 0.0).cpu().float()
                    # Resize if needed
                    if mask.shape[-2] != H or mask.shape[-1] != W:
                        mask = F.interpolate(
                            mask.unsqueeze(0).unsqueeze(0), size=(H, W),
                            mode="bilinear", align_corners=False,
                        ).squeeze(0).squeeze(0)
                    masks_tensor[out_frame_idx] = mask

            return masks_tensor, 1.0

        finally:
            shutil.rmtree(tmp, ignore_errors=True)
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    # ══════════════════════════════════════════════════════════════════
    #  VideoMaMa Dispatcher
    # ══════════════════════════════════════════════════════════════════

    def _run_videomama(
        self,
        model,
        image: torch.Tensor,
        existing_mask: torch.Tensor | None,
        dtype: torch.dtype,
        device: str,
        B: int,
        H: int,
        W: int,
    ):
        """VideoMaMa – mask-conditioned video matting.

        Takes video frames + initial masks, runs SVD-based UNet inference
        to produce refined alpha matte masks for all frames.
        """
        from PIL import Image

        if existing_mask is None:
            raise ValueError(
                "VideoMaMa requires an existing_mask input. "
                "Connect a mask from SAM2/SAM3/SeC segmentation."
            )

        pipeline = model["pipeline"]

        # Convert frames to PIL and resize
        max_resolution = 1024
        orig_h, orig_w = H, W
        if orig_w >= orig_h:
            target_w = max_resolution
            target_h = int(orig_h * max_resolution / orig_w)
        else:
            target_h = max_resolution
            target_w = int(orig_w * max_resolution / orig_h)
        target_w = max((target_w // 8) * 8, 8)
        target_h = max((target_h // 8) * 8, 8)

        cond_frames = []
        mask_frames = []

        for i in range(B):
            img_np = (image[i].cpu().numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np, mode="RGB")
            cond_frames.append(pil_img.resize((target_w, target_h), Image.LANCZOS))

            if existing_mask.dim() == 2:
                mask_np = (existing_mask.cpu().numpy() * 255).astype(np.uint8)
            elif i < existing_mask.shape[0]:
                mask_np = (existing_mask[i].cpu().numpy() * 255).astype(np.uint8)
            else:
                # Repeat last mask if not enough masks
                mask_np = (existing_mask[-1].cpu().numpy() * 255).astype(np.uint8)

            if mask_np.ndim > 2:
                mask_np = np.squeeze(mask_np)
                while mask_np.ndim > 2:
                    mask_np = mask_np[0]

            pil_mask = Image.fromarray(mask_np, mode="L")
            mask_frames.append(pil_mask.resize((target_w, target_h), Image.LANCZOS))

        logger.info("[MEC] Running VideoMaMa on %d frames (%dx%d → %dx%d)",
                     B, orig_w, orig_h, target_w, target_h)

        output_frames_pil = pipeline.run(
            cond_frames=cond_frames,
            mask_frames=mask_frames,
            seed=42,
            fps=7,
            motion_bucket_id=127,
            noise_aug_strength=0.0,
        )

        # Convert output PIL frames back to mask tensor at original resolution
        output_masks = []
        for frame_pil in output_frames_pil:
            frame_pil = frame_pil.resize((orig_w, orig_h), Image.LANCZOS)
            frame_np = np.array(frame_pil).astype(np.float32) / 255.0
            if frame_np.ndim == 3:
                frame_np = frame_np.mean(axis=-1)
            output_masks.append(frame_np)

        masks_tensor = torch.from_numpy(np.stack(output_masks, axis=0))
        logger.info("[MEC] VideoMaMa complete: %s", masks_tensor.shape)
        return masks_tensor, 1.0
