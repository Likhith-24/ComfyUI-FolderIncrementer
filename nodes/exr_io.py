"""
EXR I/O nodes (MEC):
  - LoadEXRMEC: Load an EXR file as IMAGE [B,H,W,3] in scene-linear.
  - SaveEXRMEC: Save IMAGE as EXR (16-bit half by default).

Backend priority:
  1. ``OpenEXR`` + ``Imath`` (fastest, full feature set).
  2. ``imageio`` with the ``freeimage`` plugin (decent fallback).
  3. ``imageio`` falling back to a 16-bit TIFF write at the *.exr* path
     with a warning logged. (We never silently change the extension.)

Headless and read-only safe; never imports unavailable libs at module
import time. All paths use forward slashes in the info JSON.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

import numpy as np
import torch

logger = logging.getLogger("MEC.EXRIO")


def _try_openexr_load(path: str) -> tuple[np.ndarray, dict]:
    import OpenEXR  # type: ignore[import-not-found]
    import Imath  # type: ignore[import-not-found]
    f = OpenEXR.InputFile(path)
    try:
        h = f.header()
        dw = h["dataWindow"]
        w = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1
        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        chans = []
        for c in ("R", "G", "B"):
            if c not in h["channels"]:
                raise ValueError(f"EXR {path!r} missing channel {c}")
            buf = f.channel(c, pt)
            arr = np.frombuffer(buf, dtype=np.float32).reshape(height, w)
            chans.append(arr)
        rgb = np.stack(chans, axis=-1)
    finally:
        f.close()
    info = {"backend": "OpenEXR", "width": w, "height": height}
    return rgb, info


def _try_imageio_load(path: str) -> tuple[np.ndarray, dict]:
    import imageio.v3 as iio  # type: ignore[import-not-found]
    arr = iio.imread(path)  # may pick freeimage if installed
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    arr = arr[..., :3].astype(np.float32)
    return arr, {"backend": "imageio"}


class LoadEXRMEC:
    """Load an EXR file as scene-linear IMAGE [1,H,W,3] float32."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"file_path": ("STRING", {"default": ""})},
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "info_json")
    FUNCTION = "load"
    CATEGORY = "MaskEditControl/IO"
    DESCRIPTION = "Load EXR as scene-linear IMAGE. Tries OpenEXR → imageio."

    def load(self, file_path: str):
        if not file_path or not os.path.isfile(file_path):
            raise FileNotFoundError(f"EXR not found: {file_path!r}")
        info: dict[str, Any]
        try:
            rgb, info = _try_openexr_load(file_path)
        except ImportError:
            rgb, info = _try_imageio_load(file_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[MEC] OpenEXR load failed (%s); using imageio.", exc)
            rgb, info = _try_imageio_load(file_path)
        info["file"] = os.path.basename(file_path)
        t = torch.from_numpy(np.ascontiguousarray(rgb)).unsqueeze(0)
        return (t, json.dumps(info, indent=2))


def _try_openexr_save(path: str, rgb: np.ndarray, half: bool) -> dict:
    import OpenEXR  # type: ignore[import-not-found]
    import Imath  # type: ignore[import-not-found]
    h, w, _ = rgb.shape
    pt = Imath.PixelType(Imath.PixelType.HALF if half else Imath.PixelType.FLOAT)
    header = OpenEXR.Header(w, h)
    header["channels"] = {c: Imath.Channel(pt) for c in ("R", "G", "B")}
    out = OpenEXR.OutputFile(path, header)
    try:
        dtype = np.float16 if half else np.float32
        bufs = {c: rgb[..., i].astype(dtype).tobytes() for i, c in enumerate(("R", "G", "B"))}
        out.writePixels(bufs)
    finally:
        out.close()
    return {"backend": "OpenEXR", "half": half}


def _try_imageio_save(path: str, rgb: np.ndarray) -> dict:
    import imageio.v3 as iio  # type: ignore[import-not-found]
    try:
        iio.imwrite(path, rgb.astype(np.float32))
        return {"backend": "imageio"}
    except Exception as exc:  # noqa: BLE001
        # Last-ditch: write a 16-bit TIFF at the same stem and warn.
        alt = os.path.splitext(path)[0] + "_fallback.tif"
        scaled = (np.clip(rgb, 0.0, 65.535) * 1000.0).astype(np.uint16)
        iio.imwrite(alt, scaled)
        logger.warning(
            "[MEC] EXR write failed (%s); wrote 16-bit TIFF fallback to %s", exc, alt,
        )
        return {"backend": "tiff_fallback", "fallback_path": alt}


class SaveEXRMEC:
    """Save an IMAGE batch to EXR (one file per frame; index suffix appended)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "file_path": ("STRING", {
                    "default": "",
                    "tooltip": "Absolute output path. Batches add _0001, _0002 suffixes.",
                }),
            },
            "optional": {
                "half_float": ("BOOLEAN", {"default": True, "tooltip": "16-bit half (smaller, recommended)."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("info_json",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "MaskEditControl/IO"
    DESCRIPTION = "Save IMAGE batch as EXR(s)."

    def save(self, image: torch.Tensor, file_path: str, half_float: bool = True):
        if not file_path:
            raise ValueError("file_path is required.")
        os.makedirs(os.path.dirname(os.path.abspath(file_path)) or ".", exist_ok=True)
        B = int(image.shape[0])
        results = []
        stem, ext = os.path.splitext(file_path)
        if not ext:
            ext = ".exr"
        for i in range(B):
            out_path = file_path if B == 1 else f"{stem}_{i + 1:04d}{ext}"
            rgb = image[i].cpu().numpy().astype(np.float32)
            try:
                info = _try_openexr_save(out_path, rgb, half_float)
            except ImportError:
                info = _try_imageio_save(out_path, rgb)
            except Exception as exc:  # noqa: BLE001
                logger.warning("[MEC] OpenEXR save failed (%s); using imageio.", exc)
                info = _try_imageio_save(out_path, rgb)
            info["path"] = out_path.replace("\\", "/")
            results.append(info)
        return (json.dumps({"frames": results}, indent=2),)


NODE_CLASS_MAPPINGS = {"LoadEXRMEC": LoadEXRMEC, "SaveEXRMEC": SaveEXRMEC}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadEXRMEC": "Load EXR (MEC)",
    "SaveEXRMEC": "Save EXR (MEC)",
}
