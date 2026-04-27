"""
EXRMetadataReaderMEC – Read EXR header metadata without decoding pixels.

Strategy:
  1. If the ``OpenEXR`` Python package is installed, use it (full
     metadata: channels, compression, attributes, all stringifiable
     values). Preferred path.
  2. Otherwise fall back to a pure-Python OpenEXR header parser using
     ``struct`` — reads only the header block (magic, version, then
     the null-terminated attribute records up to the empty-name
     terminator). This is well-defined per the OpenEXR file layout
     spec and works for both scanline and tiled images.

Outputs a ``STRING`` JSON of header attributes plus convenience scalars
(width, height, channel names, pixel_aspect_ratio).
"""
from __future__ import annotations

import json
import logging
import os
import struct
from typing import Any

logger = logging.getLogger("MEC.EXRMetadataReader")

_EXR_MAGIC = 20000630


# ──────────────────────────────────────────────────────────────────
#  Pure-Python header parser (fallback)
# ──────────────────────────────────────────────────────────────────

def _read_null_terminated(fh) -> str:
    out = bytearray()
    while True:
        b = fh.read(1)
        if not b or b == b"\x00":
            break
        out += b
    return out.decode("utf-8", errors="replace")


def _parse_attribute_value(name: str, type_name: str, raw: bytes) -> Any:
    """Best-effort decode of a few common EXR attribute types.

    Unknown types return an ``{"hex": "..."}`` dict so the user still
    sees something. We never raise.
    """
    try:
        if type_name == "string":
            return raw.rstrip(b"\x00").decode("utf-8", errors="replace")
        if type_name == "int":
            return struct.unpack("<i", raw[:4])[0]
        if type_name == "float":
            return struct.unpack("<f", raw[:4])[0]
        if type_name == "double":
            return struct.unpack("<d", raw[:8])[0]
        if type_name == "v2i":
            return list(struct.unpack("<2i", raw[:8]))
        if type_name == "v2f":
            return list(struct.unpack("<2f", raw[:8]))
        if type_name == "v3f":
            return list(struct.unpack("<3f", raw[:12]))
        if type_name == "box2i":
            return list(struct.unpack("<4i", raw[:16]))
        if type_name == "box2f":
            return list(struct.unpack("<4f", raw[:16]))
        if type_name == "compression":
            mapping = {0: "NO", 1: "RLE", 2: "ZIPS", 3: "ZIP", 4: "PIZ",
                       5: "PXR24", 6: "B44", 7: "B44A", 8: "DWAA", 9: "DWAB"}
            return mapping.get(raw[0], int(raw[0]))
        if type_name == "lineOrder":
            mapping = {0: "INCREASING_Y", 1: "DECREASING_Y", 2: "RANDOM_Y"}
            return mapping.get(raw[0], int(raw[0]))
        if type_name == "chlist":
            chans = []
            i = 0
            while i < len(raw):
                # name (null-terminated) ; if next byte is 0 we hit the terminator
                if raw[i] == 0:
                    break
                end = raw.index(b"\x00", i)
                cname = raw[i:end].decode("utf-8", errors="replace")
                i = end + 1
                # 16 bytes channel record: pixelType(i), pLinear(B), reserved[3], xSampling(i), ySampling(i)
                if i + 16 > len(raw):
                    break
                pixel_type = struct.unpack("<i", raw[i:i + 4])[0]
                pt_map = {0: "uint", 1: "half", 2: "float"}
                chans.append({
                    "name": cname,
                    "pixel_type": pt_map.get(pixel_type, pixel_type),
                })
                i += 16
            return chans
    except Exception as exc:  # noqa: BLE001
        return {"_decode_error": str(exc), "_hex": raw.hex()[:80]}
    return {"_raw_hex": raw.hex()[:80], "_size": len(raw)}


def _parse_exr_header_pure(path: str) -> dict:
    """Parse an EXR header without decoding pixels."""
    with open(path, "rb") as fh:
        magic = struct.unpack("<i", fh.read(4))[0]
        if magic != _EXR_MAGIC:
            raise ValueError(f"Not an OpenEXR file: bad magic 0x{magic:08x}")
        version_field = struct.unpack("<i", fh.read(4))[0]
        version = version_field & 0xFF
        flags = version_field >> 8
        attrs: dict[str, Any] = {}
        while True:
            name = _read_null_terminated(fh)
            if name == "":
                break  # end-of-header
            type_name = _read_null_terminated(fh)
            size_b = fh.read(4)
            if len(size_b) < 4:
                break
            size = struct.unpack("<i", size_b)[0]
            if size < 0 or size > 16 * 1024 * 1024:  # 16 MB safety cap
                raise ValueError(f"Implausible attribute size {size} for {name}")
            raw = fh.read(size)
            attrs[name] = {
                "type": type_name,
                "value": _parse_attribute_value(name, type_name, raw),
            }
    return {"version": version, "flags_bits": flags, "attributes": attrs}


def _read_with_openexr(path: str) -> dict:
    import OpenEXR  # type: ignore[import-not-found]
    import Imath  # type: ignore[import-not-found]
    f = OpenEXR.InputFile(path)
    try:
        h = f.header()
    finally:
        f.close()
    out: dict[str, Any] = {}
    for k, v in h.items():
        try:
            json.dumps(v)
            out[k] = v
        except TypeError:
            out[k] = repr(v)
    return {"attributes": out, "library": "OpenEXR"}


# ──────────────────────────────────────────────────────────────────
#  Node
# ──────────────────────────────────────────────────────────────────

class EXRMetadataReaderMEC:
    """Read OpenEXR header metadata without decoding pixels."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {
                    "default": "",
                    "tooltip": "Absolute path to a .exr file.",
                }),
            },
            "optional": {
                "force_pure_python": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Skip OpenEXR even if installed; useful for benchmarking.",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "INT", "INT", "STRING")
    RETURN_NAMES = ("metadata_json", "width", "height", "channels_csv")
    FUNCTION = "read"
    CATEGORY = "MaskEditControl/IO"
    DESCRIPTION = (
        "Read OpenEXR header (compression, channels, custom attributes) without "
        "decoding pixels. Uses OpenEXR if installed, otherwise pure-Python parser."
    )

    def read(self, file_path: str, force_pure_python: bool = False):
        if not file_path or not os.path.isfile(file_path):
            raise FileNotFoundError(f"EXR not found: {file_path!r}")

        meta: dict[str, Any]
        used = "pure_python"
        if not force_pure_python:
            try:
                meta = _read_with_openexr(file_path)
                used = "OpenEXR"
            except ImportError:
                meta = _parse_exr_header_pure(file_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("[MEC] OpenEXR read failed (%s); using pure parser.", exc)
                meta = _parse_exr_header_pure(file_path)
        else:
            meta = _parse_exr_header_pure(file_path)

        meta["library"] = used
        meta["file"] = os.path.basename(file_path)
        meta["bytes"] = os.path.getsize(file_path)

        # Extract convenience scalars (width/height from dataWindow box2i)
        width = 0
        height = 0
        channels: list[str] = []
        attrs = meta.get("attributes", {})
        dw = attrs.get("dataWindow")
        if isinstance(dw, dict) and isinstance(dw.get("value"), list) and len(dw["value"]) == 4:
            xmin, ymin, xmax, ymax = dw["value"]
            width = max(0, xmax - xmin + 1)
            height = max(0, ymax - ymin + 1)
        elif used == "OpenEXR":
            try:
                box = attrs["dataWindow"]
                width = box.max.x - box.min.x + 1  # type: ignore[union-attr]
                height = box.max.y - box.min.y + 1  # type: ignore[union-attr]
            except Exception:
                pass

        chlist = attrs.get("channels")
        if isinstance(chlist, dict) and isinstance(chlist.get("value"), list):
            channels = [c.get("name", "?") for c in chlist["value"]]
        elif used == "OpenEXR" and chlist is not None:
            try:
                channels = list(chlist.keys())  # type: ignore[union-attr]
            except Exception:
                pass

        return (json.dumps(meta, indent=2, default=str), width, height, ",".join(channels))


NODE_CLASS_MAPPINGS = {"EXRMetadataReaderMEC": EXRMetadataReaderMEC}
NODE_DISPLAY_NAME_MAPPINGS = {"EXRMetadataReaderMEC": "EXR Metadata Reader (MEC)"}
