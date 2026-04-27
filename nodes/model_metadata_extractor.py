"""
ModelMetadataExtractorMEC – Read model file metadata without loading weights.

Supports:
  - ``.safetensors``: spec-compliant header parse — first 8 bytes are
    a little-endian uint64 header length, followed by a UTF-8 JSON
    blob describing tensors and a ``__metadata__`` block. Constant-time;
    no torch import needed.
  - ``.ckpt`` / ``.pt`` / ``.pth``: these are pickle archives. We do
    NOT execute them. We use ``zipfile`` (modern PyTorch save format
    is a zip) to list members and extract sizes; if it's an old
    legacy pickle we fall back to ``pickletools.dis`` against the
    raw bytes to enumerate top-level opcodes safely (no globals
    are imported, no objects constructed).

Outputs JSON metadata + the model's training type (if recognized) +
a SHA256 (head + tail) fingerprint suitable for cache keys.

Security: never imports unknown modules; never unpickles untrusted
data. The ``pickletools.dis`` text output is bounded to first 1 MB.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import pickletools
import struct
import zipfile

logger = logging.getLogger("MEC.ModelMetadataExtractor")


_FINGERPRINT_BYTES = 1024 * 1024  # 1 MB head, 1 MB tail


def _quick_fingerprint(path: str) -> str:
    """SHA256 of (size || head || tail) — fast & collision-resistant for caches."""
    size = os.path.getsize(path)
    h = hashlib.sha256()
    h.update(struct.pack("<Q", size))
    with open(path, "rb") as fh:
        h.update(fh.read(_FINGERPRINT_BYTES))
        if size > _FINGERPRINT_BYTES * 2:
            fh.seek(-_FINGERPRINT_BYTES, os.SEEK_END)
            h.update(fh.read(_FINGERPRINT_BYTES))
    return h.hexdigest()


def _read_safetensors_header(path: str) -> dict:
    with open(path, "rb") as fh:
        hdr_len_b = fh.read(8)
        if len(hdr_len_b) < 8:
            raise ValueError("Truncated safetensors header.")
        hdr_len = struct.unpack("<Q", hdr_len_b)[0]
        if hdr_len <= 0 or hdr_len > 100 * 1024 * 1024:  # 100 MB sanity cap
            raise ValueError(f"Implausible safetensors header length: {hdr_len}")
        hdr = fh.read(hdr_len)
    parsed = json.loads(hdr.decode("utf-8"))
    custom = parsed.pop("__metadata__", None)
    tensors = []
    total_params = 0
    for name, spec in parsed.items():
        shape = spec.get("shape", [])
        dtype = spec.get("dtype", "?")
        n = 1
        for d in shape:
            n *= int(d)
        total_params += n
        tensors.append({
            "name": name,
            "dtype": dtype,
            "shape": shape,
            "params": n,
        })
    return {
        "format": "safetensors",
        "tensor_count": len(tensors),
        "total_params": total_params,
        "metadata": custom or {},
        "tensors": tensors[:64],  # cap output size
        "tensor_truncated": len(tensors) > 64,
    }


def _read_torch_zip(path: str) -> dict:
    """Modern .pt/.pth/.ckpt are zip archives. List members, no unpickle."""
    info: dict = {"format": "torch_zip", "members": []}
    with zipfile.ZipFile(path, "r") as zf:
        for zi in zf.infolist():
            info["members"].append({
                "name": zi.filename,
                "size": zi.file_size,
                "compressed": zi.compress_size,
            })
        # The 'data.pkl' member is the structure descriptor — we can
        # safely run pickletools.dis on it (no execution).
        for cand in ("data.pkl", "archive/data.pkl"):
            try:
                with zf.open(cand) as fh:
                    data = fh.read(1_000_000)  # cap 1 MB
                # Disassemble to a bounded text trace
                import io as _io
                buf = _io.StringIO()
                try:
                    pickletools.dis(data, buf)
                    info["pickle_disassembly_excerpt"] = buf.getvalue()[:8000]
                except Exception as exc:  # noqa: BLE001
                    info["pickle_disassembly_excerpt"] = f"<dis failed: {exc}>"
                break
            except KeyError:
                continue
    info["member_count"] = len(info["members"])
    info["total_uncompressed"] = sum(m["size"] for m in info["members"])
    return info


def _read_legacy_pickle(path: str) -> dict:
    """Old-style raw-pickle .pt — disassemble safely, no unpickling."""
    with open(path, "rb") as fh:
        data = fh.read(1_000_000)
    import io as _io
    buf = _io.StringIO()
    try:
        pickletools.dis(data, buf)
    except Exception as exc:  # noqa: BLE001
        return {"format": "legacy_pickle_unreadable", "error": str(exc)}
    return {
        "format": "legacy_pickle",
        "pickle_disassembly_excerpt": buf.getvalue()[:8000],
    }


def _detect_model_kind(meta: dict) -> str:
    """Heuristic identification of common Stable Diffusion model families."""
    tensors = meta.get("tensors", [])
    names = {t["name"] for t in tensors}
    if any(n.startswith("first_stage_model.") for n in names) and any(
        n.startswith("model.diffusion_model.") for n in names
    ):
        return "stable_diffusion_checkpoint"
    if any("encoder.down" in n and "first_stage_model" not in n for n in names):
        return "vae"
    if any(n.startswith("lora_") or ".lora_up" in n or ".lora_down" in n for n in names):
        return "lora"
    if any(n.startswith("text_model.") or n.startswith("transformer.") for n in names):
        return "text_encoder"
    if meta.get("format") == "safetensors" and tensors:
        return "safetensors_other"
    return "unknown"


class ModelMetadataExtractorMEC:
    """Read .safetensors / .pt / .pth / .ckpt metadata without loading weights."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {
                    "default": "",
                    "tooltip": "Absolute path to a model file (.safetensors / .pt / .pth / .ckpt).",
                }),
            },
            "optional": {
                "compute_fingerprint": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Compute SHA256 over (size, first 1 MB, last 1 MB). "
                        "Suitable cache key; far faster than full-file hashing."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "STRING")
    RETURN_NAMES = ("metadata_json", "model_kind", "total_params", "fingerprint")
    FUNCTION = "extract"
    CATEGORY = "MaskEditControl/Diagnostics"
    DESCRIPTION = (
        "Inspect model file metadata WITHOUT unpickling or loading weights. "
        "Safe to run on untrusted .ckpt files. Reports tensor count, params, "
        "training metadata, and a quick fingerprint."
    )

    def extract(self, file_path: str, compute_fingerprint: bool = True):
        if not file_path or not os.path.isfile(file_path):
            raise FileNotFoundError(f"Model file not found: {file_path!r}")

        ext = os.path.splitext(file_path)[1].lower()
        size = os.path.getsize(file_path)
        meta: dict
        try:
            if ext == ".safetensors":
                meta = _read_safetensors_header(file_path)
            else:
                # Try modern torch-zip first; fall back to legacy pickle.
                if zipfile.is_zipfile(file_path):
                    meta = _read_torch_zip(file_path)
                else:
                    meta = _read_legacy_pickle(file_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[MEC] Metadata extraction failed: %s", exc)
            meta = {"format": "error", "error": str(exc)}

        meta["file"] = os.path.basename(file_path)
        meta["bytes"] = size
        kind = _detect_model_kind(meta)
        meta["detected_kind"] = kind
        total_params = int(meta.get("total_params", 0))
        fp = _quick_fingerprint(file_path) if compute_fingerprint else ""
        meta["fingerprint_sha256"] = fp

        return (json.dumps(meta, indent=2, default=str), kind, total_params, fp)


NODE_CLASS_MAPPINGS = {"ModelMetadataExtractorMEC": ModelMetadataExtractorMEC}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelMetadataExtractorMEC": "Model Metadata Extractor (MEC)"
}
