"""
Metadata / routing nodes (MEC):
  - MetadataWriterMEC: Write a JSON sidecar next to image outputs.
  - FrameRangeRouterMEC: Slice a video batch to a sub-range.
  - ShotMetadataNodeMEC: Read a shot.json descriptor from disk.
"""
from __future__ import annotations

import json
import logging
import os

import torch

logger = logging.getLogger("MEC.Metadata")


class MetadataWriterMEC:
    """Write an arbitrary JSON sidecar to disk and pass IMAGE through."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "sidecar_path": ("STRING", {"default": ""}),
                "metadata_json": ("STRING", {"default": "{}", "multiline": True}),
            },
            "optional": {
                "merge_existing": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "written_path")
    FUNCTION = "write"
    OUTPUT_NODE = True
    CATEGORY = "MaskEditControl/Metadata"
    DESCRIPTION = "Write a JSON sidecar; pass the image through."

    def write(self, image: torch.Tensor, sidecar_path: str, metadata_json: str, merge_existing: bool = False):
        if not sidecar_path:
            raise ValueError("sidecar_path is required.")
        try:
            payload = json.loads(metadata_json) if metadata_json.strip() else {}
        except json.JSONDecodeError as exc:
            raise ValueError(f"metadata_json is not valid JSON: {exc}") from exc
        if merge_existing and os.path.isfile(sidecar_path):
            try:
                with open(sidecar_path, "r", encoding="utf-8") as fh:
                    existing = json.load(fh)
                if isinstance(existing, dict) and isinstance(payload, dict):
                    existing.update(payload)
                    payload = existing
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("[MEC] could not merge existing sidecar (%s); overwriting.", exc)
        os.makedirs(os.path.dirname(os.path.abspath(sidecar_path)) or ".", exist_ok=True)
        with open(sidecar_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        return (image, sidecar_path.replace("\\", "/"))


class FrameRangeRouterMEC:
    """Slice a video batch (IMAGE/MASK) by frame index range.

    ``start`` is inclusive, ``end`` is exclusive. Negative values index from
    the end (Python slice semantics). Step controls stride.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "start": ("INT", {"default": 0, "min": -100000, "max": 100000}),
                "end": ("INT", {"default": -1, "min": -100000, "max": 100000}),
                "step": ("INT", {"default": 1, "min": 1, "max": 1000}),
            },
            "optional": {
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT")
    RETURN_NAMES = ("images", "mask", "frame_count")
    FUNCTION = "route"
    CATEGORY = "MaskEditControl/Metadata"
    DESCRIPTION = "Slice a video batch by [start:end:step]."

    def route(
        self, images: torch.Tensor, start: int, end: int, step: int,
        mask: torch.Tensor | None = None,
    ):
        B = images.shape[0]
        # Translate negative end "-1" → "to the end" (B), like a sentinel.
        if end == -1:
            end = B
        s = max(min(start if start >= 0 else max(B + start, 0), B), 0)
        e = max(min(end if end >= 0 else max(B + end, 0), B), 0)
        if e < s:
            s, e = e, s
        out_imgs = images[s:e:step]
        if mask is not None:
            mb = mask.shape[0]
            ee = min(e, mb)
            ss = min(s, mb)
            out_mask = mask[ss:ee:step]
        else:
            out_mask = torch.zeros((out_imgs.shape[0], out_imgs.shape[1], out_imgs.shape[2]), dtype=images.dtype)
        return (out_imgs, out_mask, int(out_imgs.shape[0]))


class ShotMetadataNodeMEC:
    """Read a shot.json descriptor from disk and surface common fields."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"shot_json_path": ("STRING", {"default": ""})},
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "INT", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("show", "shot", "task", "frame_in", "frame_out", "fps", "raw_json")
    FUNCTION = "read"
    CATEGORY = "MaskEditControl/Metadata"
    DESCRIPTION = "Read a shot.json descriptor; missing fields return empty defaults."

    def read(self, shot_json_path: str):
        if not shot_json_path or not os.path.isfile(shot_json_path):
            raise FileNotFoundError(f"shot json not found: {shot_json_path!r}")
        with open(shot_json_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            raise ValueError(f"{shot_json_path!r} did not contain a JSON object.")
        return (
            str(data.get("show", "")),
            str(data.get("shot", "")),
            str(data.get("task", "")),
            int(data.get("frame_in", 0) or 0),
            int(data.get("frame_out", 0) or 0),
            float(data.get("fps", 24.0) or 24.0),
            json.dumps(data, indent=2),
        )


NODE_CLASS_MAPPINGS = {
    "MetadataWriterMEC": MetadataWriterMEC,
    "FrameRangeRouterMEC": FrameRangeRouterMEC,
    "ShotMetadataNodeMEC": ShotMetadataNodeMEC,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MetadataWriterMEC": "Metadata Writer (MEC)",
    "FrameRangeRouterMEC": "Frame Range Router (MEC)",
    "ShotMetadataNodeMEC": "Shot Metadata Reader (MEC)",
}
