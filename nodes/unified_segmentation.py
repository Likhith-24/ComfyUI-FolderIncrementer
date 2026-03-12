"""
UnifiedSegmentation – One node for SAM2/2.1, SAM3, and SeC segmentation.

Image mode (B=1):  point/bbox prompts → mask
Video mode (B>1):  prompts on frame 0 → propagate to all frames via
                   SAM2VideoPredictor / SAM3VideoPredictor.

Features:
  - MODEL_REGISTRY with all supported checkpoints + HF repos
  - Filesystem scan + auto-download from HuggingFace Hub
  - Module-level model cache (single model, evict on change)
  - Autocast for fp16 / bf16 inference
  - SAM3 neg_bbox support
  - SeC text-prompt stub (requires sec package)
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

logger = logging.getLogger("MEC")

# ── ComfyUI path helpers ──────────────────────────────────────────────
try:
    import folder_paths

    _MODELS_DIR: str = getattr(
        folder_paths, "models_dir",
        os.path.join(folder_paths.base_path, "models"),
    )
except ImportError:
    folder_paths = None  # type: ignore[assignment]
    _MODELS_DIR = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "models",
    )

from .utils import parse_bbox_input, parse_points_json, points_to_arrays


# ══════════════════════════════════════════════════════════════════════
#  Model Registry
# ══════════════════════════════════════════════════════════════════════

MODEL_REGISTRY: dict[str, dict] = {
    # ── SAM 2.0 ──────────────────────────────────────────────────────
    "sam2_hiera_tiny": {
        "family": "sam2",
        "repo_id": "Kijai/sam2-safetensors",
        "filename": "sam2_hiera_tiny.safetensors",
        "config": "configs/sam2/sam2_hiera_t.yaml",
        "model_dir": "sam2",
    },
    "sam2_hiera_small": {
        "family": "sam2",
        "repo_id": "Kijai/sam2-safetensors",
        "filename": "sam2_hiera_small.safetensors",
        "config": "configs/sam2/sam2_hiera_s.yaml",
        "model_dir": "sam2",
    },
    "sam2_hiera_base_plus": {
        "family": "sam2",
        "repo_id": "Kijai/sam2-safetensors",
        "filename": "sam2_hiera_base_plus.safetensors",
        "config": "configs/sam2/sam2_hiera_b+.yaml",
        "model_dir": "sam2",
    },
    "sam2_hiera_large": {
        "family": "sam2",
        "repo_id": "Kijai/sam2-safetensors",
        "filename": "sam2_hiera_large.safetensors",
        "config": "configs/sam2/sam2_hiera_l.yaml",
        "model_dir": "sam2",
    },
    # ── SAM 2.1 ──────────────────────────────────────────────────────
    "sam2.1_hiera_tiny": {
        "family": "sam2",
        "repo_id": "Kijai/sam2-safetensors",
        "filename": "sam2.1_hiera_tiny.safetensors",
        "config": "configs/sam2.1/sam2.1_hiera_t.yaml",
        "model_dir": "sam2",
    },
    "sam2.1_hiera_small": {
        "family": "sam2",
        "repo_id": "Kijai/sam2-safetensors",
        "filename": "sam2.1_hiera_small.safetensors",
        "config": "configs/sam2.1/sam2.1_hiera_s.yaml",
        "model_dir": "sam2",
    },
    "sam2.1_hiera_base_plus": {
        "family": "sam2",
        "repo_id": "Kijai/sam2-safetensors",
        "filename": "sam2.1_hiera_base_plus.safetensors",
        "config": "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "model_dir": "sam2",
    },
    "sam2.1_hiera_large": {
        "family": "sam2",
        "repo_id": "Kijai/sam2-safetensors",
        "filename": "sam2.1_hiera_large.safetensors",
        "config": "configs/sam2.1/sam2.1_hiera_l.yaml",
        "model_dir": "sam2",
    },
    # ── SAM 3 ────────────────────────────────────────────────────────
    "sam3": {
        "family": "sam3",
        "repo_id": "apozz/sam3-safetensors",
        "filename": "sam3.safetensors",
        "config": None,  # SAM3 uses SAM2 large architecture
        "model_dir": "sam3",
    },
    # ── SeC (MLLM + SAM2) ───────────────────────────────────────────
    "sec_4b_fp16": {
        "family": "sec",
        "repo_id": "OpenIXCLab/SeC-4B",
        "filename": "SeC-4B-fp16.safetensors",
        "config": None,
        "model_dir": "sams",
    },
    "sec_4b_bf16": {
        "family": "sec",
        "repo_id": "OpenIXCLab/SeC-4B",
        "filename": "SeC-4B-bf16.safetensors",
        "config": None,
        "model_dir": "sams",
    },
    "sec_4b_fp32": {
        "family": "sec",
        "repo_id": "OpenIXCLab/SeC-4B",
        "filename": "SeC-4B-fp32.safetensors",
        "config": None,
        "model_dir": "sams",
    },
}


# ══════════════════════════════════════════════════════════════════════
#  Module-Level Model Cache  (one model at a time)
# ══════════════════════════════════════════════════════════════════════

_cache: dict = {"name": None, "model": None, "family": None,
                "dtype": None, "device": None}


def _flush_cache() -> None:
    global _cache
    old = _cache.get("model")
    _cache = {"name": None, "model": None, "family": None,
              "dtype": None, "device": None}
    del old
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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


def _load_state_dict(path: str) -> dict:
    """Load state_dict from .safetensors / .pt / .pth."""
    if path.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file
            return load_file(path)
        except ImportError:
            pass
    try:
        from comfy.utils import load_torch_file
        sd = load_torch_file(path)
        if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
            sd = sd["model"]
        return sd
    except ImportError:
        pass
    sd = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    return sd


# ══════════════════════════════════════════════════════════════════════
#  Node
# ══════════════════════════════════════════════════════════════════════

class UnifiedSegmentation:
    """Unified segmentation: SAM2 / 2.1 / SAM3 / SeC in one node.

    Automatically uses video propagation when the input IMAGE batch
    has more than one frame.
    """

    # ── Scan available models ─────────────────────────────────────────

    @classmethod
    def _scan_models(cls) -> list[str]:
        found: set[str] = set()
        for name, info in MODEL_REGISTRY.items():
            # Direct filesystem check
            for sub in (info["model_dir"], "sams"):
                candidate = os.path.join(_MODELS_DIR, sub, info["filename"])
                if os.path.exists(candidate):
                    found.add(name)
                    break
            # folder_paths check
            if folder_paths is not None and name not in found:
                for key in (info["model_dir"], "sams"):
                    if key in getattr(folder_paths, "folder_names_and_paths", {}):
                        try:
                            p = folder_paths.get_full_path(key, info["filename"])
                            if p and os.path.exists(p):
                                found.add(name)
                                break
                        except Exception:
                            pass

        opts = sorted(found)
        for name in sorted(MODEL_REGISTRY):
            if name not in found:
                opts.append(f"[download] {name}")
        return opts or ["(no models — select a [download] option)"]

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
        "Unified segmentation supporting SAM2/2.1, SAM3, and SeC.  "
        "Auto-detects image vs video mode from input batch size.  "
        "Auto-downloads models from HuggingFace on first use."
    )

    # ── Main entry point ──────────────────────────────────────────────

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
        need_dl = model_name.startswith("[download] ")

        if clean not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model '{clean}'.  "
                f"Available: {sorted(MODEL_REGISTRY)}"
            )

        reg = MODEL_REGISTRY[clean]
        family = reg["family"]

        dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
        torch_dtype = dtype_map[precision]
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = self._ensure_model(clean, reg, torch_dtype, device, need_dl)

        # Parse prompts
        pts = parse_points_json(points_json)
        pt_coords, pt_labels = points_to_arrays(pts)
        box_np = parse_bbox_input(bbox_json, bbox)

        neg_box = None
        if neg_bbox_json and neg_bbox_json.strip() and family == "sam3":
            neg_box = parse_bbox_input(neg_bbox_json)

        B, H, W, _C = image.shape
        is_video = B > 1

        if is_video:
            masks, score = self._video(
                model, family, image, pt_coords, pt_labels,
                box_np, neg_box, torch_dtype, device, existing_mask,
            )
        else:
            masks, score = self._image(
                model, family, image[0], pt_coords, pt_labels,
                box_np, neg_box, text_prompt, multimask, mask_index,
                torch_dtype, device, existing_mask, H, W,
            )

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
    #  Model Loading
    # ══════════════════════════════════════════════════════════════════

    def _ensure_model(self, name, reg, dtype, device, need_dl):
        global _cache
        if _cache["name"] == name and _cache["dtype"] == dtype:
            m = _cache["model"]
            if _cache["device"] != device and hasattr(m, "to"):
                m.to(device)
                _cache["device"] = device
            return m

        _flush_cache()
        path = self._resolve(reg, need_dl)
        family = reg["family"]

        if family in ("sam2", "sam3"):
            model = self._build_sam(path, reg, dtype, device)
        elif family == "sec":
            model = self._build_sec(path, reg, dtype, device)
        else:
            raise ValueError(f"Unknown family: {family}")

        _cache.update(name=name, model=model, family=family,
                      dtype=dtype, device=device)
        return model

    # ── Path Resolution + Download ────────────────────────────────────

    @staticmethod
    def _resolve(reg: dict, need_dl: bool) -> str:
        fname = reg["filename"]
        dirs_to_check: list[str] = []

        for sub in (reg["model_dir"], "sams"):
            dirs_to_check.append(os.path.join(_MODELS_DIR, sub))

        if folder_paths is not None:
            for key in (reg["model_dir"], "sams"):
                if key in getattr(folder_paths, "folder_names_and_paths", {}):
                    try:
                        p = folder_paths.get_full_path(key, fname)
                        if p and os.path.exists(p):
                            return p
                    except Exception:
                        pass

        for d in dirs_to_check:
            c = os.path.join(d, fname)
            if os.path.exists(c):
                return c

        # Auto-download as fallback
        dest_dir = dirs_to_check[0] if dirs_to_check else os.path.join(_MODELS_DIR, reg["model_dir"])
        return UnifiedSegmentation._download(reg, dest_dir)

    @staticmethod
    def _download(reg: dict, dest_dir: str) -> str:
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, reg["filename"])
        if os.path.exists(dest):
            return dest

        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise RuntimeError(
                f"huggingface_hub not installed.  pip install huggingface_hub\n"
                f"Or download '{reg['filename']}' from "
                f"https://huggingface.co/{reg['repo_id']}"
            )

        logger.info("[MEC] Downloading %s from %s …", reg["filename"], reg["repo_id"])
        downloaded = hf_hub_download(
            repo_id=reg["repo_id"],
            filename=reg["filename"],
            local_dir=dest_dir,
        )
        if downloaded != dest and os.path.exists(downloaded):
            shutil.move(downloaded, dest)
        logger.info("[MEC] Saved → %s", dest)
        return dest

    # ── SAM2 / SAM3 Builder ───────────────────────────────────────────

    @staticmethod
    def _build_sam(path: str, reg: dict, dtype: torch.dtype, device: str):
        state_dict = _load_state_dict(path)
        config = reg.get("config")

        # For SAM3 without explicit config, use SAM2.1 large
        if config is None:
            config = "configs/sam2.1/sam2.1_hiera_l.yaml"

        try:
            from sam2.build_sam import build_sam2
        except ImportError:
            raise RuntimeError(
                "sam2 package is required.  Install with:\n"
                "  pip install git+https://github.com/facebookresearch/sam2.git"
            )

        model = build_sam2(config_file=config, ckpt_path=None, device="cpu")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.debug("[MEC] SAM missing keys: %d", len(missing))
        if unexpected:
            logger.debug("[MEC] SAM unexpected keys: %d", len(unexpected))

        model = model.to(dtype).to(device).eval()
        logger.info(
            "[MEC] Loaded %s  (%s, %s, missing=%d, unexpected=%d)",
            reg["filename"], dtype, device, len(missing), len(unexpected),
        )
        return model

    # ── SeC Builder (stub) ────────────────────────────────────────────

    @staticmethod
    def _build_sec(path: str, reg: dict, dtype: torch.dtype, device: str):
        raise NotImplementedError(
            "SeC model support requires the sec package.  Install:\n"
            "  pip install git+https://github.com/OpenIXCLab/SeC.git\n"
            "This backend will be fully implemented in a future release."
        )

    # ══════════════════════════════════════════════════════════════════
    #  Single-Image Segmentation
    # ══════════════════════════════════════════════════════════════════

    def _image(
        self, model, family, frame, pt_coords, pt_labels,
        box_np, neg_box, text_prompt, multimask, mask_idx,
        dtype, device, existing_mask, H, W,
    ):
        img_np = (frame.cpu().numpy() * 255).astype(np.uint8)

        if family in ("sam2", "sam3"):
            return self._sam_image(
                model, img_np, pt_coords, pt_labels, box_np, neg_box,
                multimask, mask_idx, dtype, device, H, W, family,
            )
        elif family == "sec":
            raise NotImplementedError("SeC image inference not yet available.")

        return torch.zeros(1, H, W, dtype=torch.float32), 0.0

    def _sam_image(
        self, model, img_np, pt_coords, pt_labels, box_np, neg_box,
        multimask, mask_idx, dtype, device, H, W, family,
    ):
        try:
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError:
            raise RuntimeError("sam2 package is required for image prediction.")

        predictor = SAM2ImagePredictor(model)

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

    # ══════════════════════════════════════════════════════════════════
    #  Video Segmentation (propagation)
    # ══════════════════════════════════════════════════════════════════

    def _video(
        self, model, family, frames, pt_coords, pt_labels,
        box_np, neg_box, dtype, device, existing_mask,
    ):
        B, H, W, _C = frames.shape

        if family in ("sam2", "sam3"):
            return self._sam_video(
                model, frames, pt_coords, pt_labels, box_np,
                dtype, device, B, H, W,
            )

        # Fallback: per-frame image segmentation
        logger.warning("[MEC] No video propagation for %s — running per-frame.", family)
        masks_list: list[torch.Tensor] = []
        best = 0.0
        for i in range(B):
            m, s = self._image(
                model, family, frames[i], pt_coords, pt_labels,
                box_np, neg_box, "", True, 0, dtype, device, None, H, W,
            )
            masks_list.append(m[0])
            best = max(best, s)
        return torch.stack(masks_list), best

    def _sam_video(
        self, model, frames, pt_coords, pt_labels, box_np,
        dtype, device, B, H, W,
    ):
        try:
            from sam2.sam2_video_predictor import SAM2VideoPredictor
        except ImportError:
            raise RuntimeError(
                "sam2 package is required for video propagation."
            )

        video_pred = SAM2VideoPredictor(model)

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
            if box_np is not None:
                pkw["box"] = box_np

            with _autocast(dtype, device):
                video_pred.add_new_points_or_box(**pkw)

            # Propagate
            collected: dict[int, torch.Tensor] = {}
            with _autocast(dtype, device):
                for fidx, _oids, logits in video_pred.propagate_in_video(state):
                    # logits: (num_obj, 1, H, W)
                    collected[fidx] = (logits[0, 0] > 0.0).float().cpu()

            # Assemble batch
            out: list[torch.Tensor] = []
            for i in range(B):
                out.append(collected.get(i, torch.zeros(H, W, dtype=torch.float32)))
            return torch.stack(out), 1.0

        finally:
            shutil.rmtree(tmp, ignore_errors=True)
