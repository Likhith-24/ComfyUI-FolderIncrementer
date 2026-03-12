"""
model_manager.py – Shared model download, cache, and loading for MaskEditControl.

All segmentation and matting nodes delegate model I/O through this module.
Provides:
  - _MODEL_CACHE: Dict keyed by (model_type, variant, precision)
  - get_model_path(): Resolve local path or trigger download
  - ensure_downloaded(): HuggingFace Hub download with progress
  - get_or_load_model(): Cache-aware model loading
  - clear_cache(): Free VRAM / RAM
  - precision_to_dtype(): String → torch.dtype
  - scan_model_dir(): List available checkpoints on disk
"""

from __future__ import annotations

import gc
import logging
import os
import shutil
from typing import Any, Optional

import torch

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


# ══════════════════════════════════════════════════════════════════════
#  Model Registry  – populated from third_party reference code
# ══════════════════════════════════════════════════════════════════════

MODEL_REGISTRY: dict[str, dict] = {
    # ── SAM 2.0 ──────────────────────────────────────────────────────
    "sam2_hiera_tiny": {
        "family": "sam2",
        "repo_id": "Kijai/sam2-safetensors",
        "filename": "sam2_hiera_tiny.safetensors",
        "config": "sam2_hiera_t.yaml",
        "version": "2.0",
        "model_dir": "sam2",
    },
    "sam2_hiera_small": {
        "family": "sam2",
        "repo_id": "Kijai/sam2-safetensors",
        "filename": "sam2_hiera_small.safetensors",
        "config": "sam2_hiera_s.yaml",
        "version": "2.0",
        "model_dir": "sam2",
    },
    "sam2_hiera_base_plus": {
        "family": "sam2",
        "repo_id": "Kijai/sam2-safetensors",
        "filename": "sam2_hiera_base_plus.safetensors",
        "config": "sam2_hiera_b+.yaml",
        "version": "2.0",
        "model_dir": "sam2",
    },
    "sam2_hiera_large": {
        "family": "sam2",
        "repo_id": "Kijai/sam2-safetensors",
        "filename": "sam2_hiera_large.safetensors",
        "config": "sam2_hiera_l.yaml",
        "version": "2.0",
        "model_dir": "sam2",
    },
    # ── SAM 2.1 ──────────────────────────────────────────────────────
    "sam2.1_hiera_tiny": {
        "family": "sam2",
        "repo_id": "Kijai/sam2-safetensors",
        "filename": "sam2.1_hiera_tiny.safetensors",
        "config": "sam2.1_hiera_t.yaml",
        "version": "2.1",
        "model_dir": "sam2",
    },
    "sam2.1_hiera_small": {
        "family": "sam2",
        "repo_id": "Kijai/sam2-safetensors",
        "filename": "sam2.1_hiera_small.safetensors",
        "config": "sam2.1_hiera_s.yaml",
        "version": "2.1",
        "model_dir": "sam2",
    },
    "sam2.1_hiera_base_plus": {
        "family": "sam2",
        "repo_id": "Kijai/sam2-safetensors",
        "filename": "sam2.1_hiera_base_plus.safetensors",
        "config": "sam2.1_hiera_b+.yaml",
        "version": "2.1",
        "model_dir": "sam2",
    },
    "sam2.1_hiera_large": {
        "family": "sam2",
        "repo_id": "Kijai/sam2-safetensors",
        "filename": "sam2.1_hiera_large.safetensors",
        "config": "sam2.1_hiera_l.yaml",
        "version": "2.1",
        "model_dir": "sam2",
    },
    # ── SAM 3 ────────────────────────────────────────────────────────
    "sam3": {
        "family": "sam3",
        "repo_id": "apozz/sam3-safetensors",
        "filename": "sam3.safetensors",
        "config": None,
        "version": "3.0",
        "model_dir": "sam3",
    },
    # ── SeC (MLLM + SAM2) ───────────────────────────────────────────
    "sec_4b": {
        "family": "sec",
        "repo_id": "OpenIXCLab/SeC-4B",
        "filename": None,  # sharded / directory model
        "config": None,
        "version": "4b",
        "model_dir": "sams",
    },
    # ── VideoMaMa ────────────────────────────────────────────────────
    "videomama": {
        "family": "videomama",
        "repo_id": "SammyLim/VideoMaMa",
        "filename": None,  # directory model (fine-tuned UNet)
        "config": None,
        "version": "1.0",
        "model_dir": "VideoMaMa",
    },
    # ── Matting models ───────────────────────────────────────────────
    "vitmatte_small": {
        "family": "vitmatte",
        "repo_id": "hustvl/vitmatte-small-distinctions-646",
        "filename": None,  # HF Transformers model
        "config": None,
        "version": "small",
        "model_dir": "vitmatte",
    },
    "vitmatte_base": {
        "family": "vitmatte",
        "repo_id": "hustvl/vitmatte-base-distinctions-646",
        "filename": None,
        "config": None,
        "version": "base",
        "model_dir": "vitmatte",
    },
    "matanyone2": {
        "family": "matanyone2",
        "repo_id": "pq-yang/MatAnyone2",
        "filename": "matanyone2.pth",
        "config": None,
        "version": "2.0",
        "model_dir": "matanyone2",
    },
}


# ══════════════════════════════════════════════════════════════════════
#  Module-Level Cache  – keyed by (model_type, variant, precision)
# ══════════════════════════════════════════════════════════════════════

_MODEL_CACHE: dict[tuple, dict[str, Any]] = {}


def precision_to_dtype(precision: str) -> torch.dtype:
    """Convert precision string to torch dtype."""
    mapping = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    if precision not in mapping:
        raise ValueError(f"Unknown precision '{precision}'. Use: {sorted(mapping)}")
    return mapping[precision]


def get_model_path(name: str) -> str:
    """Resolve filesystem path for a registered model, downloading if needed.

    Args:
        name: Key from MODEL_REGISTRY.

    Returns:
        Absolute path to the checkpoint file or model directory.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {sorted(MODEL_REGISTRY)}")

    reg = MODEL_REGISTRY[name]
    fname = reg["filename"]
    model_dir = reg["model_dir"]

    # For directory-based models (HF Transformers), return the directory
    if fname is None:
        local_dir = os.path.join(_MODELS_DIR, model_dir, name)
        if os.path.isdir(local_dir):
            return local_dir
        # Not downloaded yet — return repo_id for from_pretrained()
        return reg["repo_id"]

    # Check local filesystem
    dirs_to_check = [os.path.join(_MODELS_DIR, model_dir)]
    if model_dir != "sams":
        dirs_to_check.append(os.path.join(_MODELS_DIR, "sams"))

    if folder_paths is not None:
        for key in (model_dir, "sams"):
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

    # Not found locally — download
    dest_dir = dirs_to_check[0]
    return ensure_downloaded(name, dest_dir)


def ensure_downloaded(name: str, dest_dir: Optional[str] = None) -> str:
    """Download a model from HuggingFace Hub if not present locally.

    Args:
        name: Key from MODEL_REGISTRY.
        dest_dir: Override download destination.

    Returns:
        Absolute path to the downloaded file.
    """
    reg = MODEL_REGISTRY[name]
    fname = reg["filename"]
    repo_id = reg["repo_id"]

    if dest_dir is None:
        dest_dir = os.path.join(_MODELS_DIR, reg["model_dir"])
    os.makedirs(dest_dir, exist_ok=True)

    if fname is not None:
        dest = os.path.join(dest_dir, fname)
        if os.path.exists(dest):
            return dest
    else:
        # Directory model — check if dir exists with content
        dest = os.path.join(dest_dir, name)
        if os.path.isdir(dest) and os.listdir(dest):
            return dest

    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError:
        raise RuntimeError(
            f"huggingface_hub not installed.  pip install huggingface_hub\n"
            f"Or download '{fname or name}' from https://huggingface.co/{repo_id}"
        )

    if fname is not None:
        # SAM2 2.1 models: use fp16 variant for non-fp32 precision
        logger.info("[MEC] Downloading %s from %s …", fname, repo_id)
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=fname,
            local_dir=dest_dir,
        )
        target = os.path.join(dest_dir, fname)
        if downloaded != target and os.path.exists(downloaded):
            shutil.move(downloaded, target)
        logger.info("[MEC] Saved → %s", target)
        return target
    else:
        # Snapshot download for directory models
        logger.info("[MEC] Downloading %s snapshot from %s …", name, repo_id)
        downloaded = snapshot_download(
            repo_id=repo_id,
            local_dir=os.path.join(dest_dir, name),
        )
        logger.info("[MEC] Saved → %s", downloaded)
        return downloaded


def get_or_load_model(
    name: str,
    precision: str = "fp16",
    device: str = "cuda",
    force_reload: bool = False,
) -> Any:
    """Load a model from cache or disk.  Handles caching and device placement.

    Args:
        name: Key from MODEL_REGISTRY.
        precision: "fp16", "bf16", or "fp32".
        device: Target device.
        force_reload: If True, ignore cache and reload.

    Returns:
        The loaded model object (type depends on family).
    """
    dtype = precision_to_dtype(precision)
    cache_key = (name, precision)

    if not force_reload and cache_key in _MODEL_CACHE:
        cached = _MODEL_CACHE[cache_key]
        model = cached["model"]
        if cached.get("device") != device and hasattr(model, "to"):
            model.to(device)
            cached["device"] = device
        return model

    # Evict old entry if same name but different precision
    for k in list(_MODEL_CACHE.keys()):
        if k[0] == name:
            _evict(k)

    path = get_model_path(name)
    reg = MODEL_REGISTRY[name]
    family = reg["family"]

    print(f"[MEC] Loading model: {name} ({family}, {precision}, {device})")

    if family == "sam2":
        model = _load_sam2(path, reg, dtype, device)
    elif family == "sam3":
        model = _load_sam3(path, reg, dtype, device)
    elif family == "vitmatte":
        model = _load_vitmatte(path, reg, device)
    elif family == "matanyone2":
        model = _load_matanyone2(path, reg, dtype, device)
    elif family == "sec":
        model = _load_sec(path, reg, dtype, device)
    elif family == "videomama":
        model = _load_videomama(path, reg, dtype, device)
    else:
        raise ValueError(f"Unknown model family: {family}")

    _MODEL_CACHE[cache_key] = {
        "model": model,
        "family": family,
        "dtype": dtype,
        "device": device,
    }
    print(f"[MEC] Model loaded: {name}")
    return model


def clear_cache(name: Optional[str] = None) -> None:
    """Free cached models.

    Args:
        name: Specific model to free, or None for all.
    """
    if name is None:
        keys = list(_MODEL_CACHE.keys())
        for k in keys:
            _evict(k)
    else:
        keys = [k for k in _MODEL_CACHE if k[0] == name]
        for k in keys:
            _evict(k)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _evict(key: tuple) -> None:
    """Remove a single entry from the cache."""
    entry = _MODEL_CACHE.pop(key, None)
    if entry is not None:
        model = entry.get("model")
        if hasattr(model, "to"):
            try:
                model.to("cpu")
            except Exception:
                pass
        del model
        del entry


def scan_model_dir(family: Optional[str] = None) -> list[str]:
    """List model names available on disk (optionally filtered by family).

    Returns:
        Sorted list of model names found locally. Models not found get
        a "[download]" prefix appended to a second section.
    """
    found: list[str] = []
    downloadable: list[str] = []

    for name, reg in sorted(MODEL_REGISTRY.items()):
        if family and reg["family"] != family:
            continue

        fname = reg["filename"]
        model_dir = reg["model_dir"]
        located = False

        if fname is not None:
            for sub in (model_dir, "sams"):
                candidate = os.path.join(_MODELS_DIR, sub, fname)
                if os.path.exists(candidate):
                    located = True
                    break
        else:
            local_dir = os.path.join(_MODELS_DIR, model_dir, name)
            if os.path.isdir(local_dir) and os.listdir(local_dir):
                located = True

        if located:
            found.append(name)
        else:
            downloadable.append(f"[download] {name}")

    return found + downloadable if (found or downloadable) else ["(no models found)"]


# ══════════════════════════════════════════════════════════════════════
#  Private Loaders  (one per family)
# ══════════════════════════════════════════════════════════════════════

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


def _load_sam2(path: str, reg: dict, dtype: torch.dtype, device: str):
    """Load SAM2/2.1 model via sam2.build_sam.build_sam2."""
    try:
        from sam2.build_sam import build_sam2
    except ImportError:
        raise RuntimeError(
            "sam2 package is required.  Install with:\n"
            "  pip install git+https://github.com/facebookresearch/sam2.git"
        )

    config = reg.get("config")
    version = reg.get("version", "2.0")

    # build_sam2 resolves configs from sam2 package internal paths
    config_file = f"configs/sam2{'.' + version.replace('2.', '') if version != '2.0' else ''}/{config}" if config else "configs/sam2.1/sam2.1_hiera_l.yaml"
    # Simplify: use the config name directly — build_sam2 expects relative path within sam2 package
    if version == "2.0":
        config_file = f"configs/sam2/{config}"
    else:
        config_file = f"configs/sam2.1/{config}"

    state_dict = _load_state_dict(path)
    model = build_sam2(config_file=config_file, ckpt_path=None, device="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.debug("[MEC] SAM2 missing keys: %d", len(missing))
    if unexpected:
        logger.debug("[MEC] SAM2 unexpected keys: %d", len(unexpected))

    model = model.to(dtype).to(device).eval()
    logger.info("[MEC] SAM2 loaded: %s (%s, %s)", reg["filename"], dtype, device)
    return model


def _load_sam3(path: str, reg: dict, dtype: torch.dtype, device: str):
    """Load SAM3 model.

    SAM3 uses a specialized loader — we build the SAM2 large architecture
    and load SAM3 weights into it.
    """
    try:
        from sam2.build_sam import build_sam2
    except ImportError:
        raise RuntimeError(
            "sam2 package is required for SAM3.  Install with:\n"
            "  pip install git+https://github.com/facebookresearch/sam2.git"
        )

    # SAM3 checkpoint is compatible with SAM2.1 large architecture
    config_file = "configs/sam2.1/sam2.1_hiera_l.yaml"
    state_dict = _load_state_dict(path)
    model = build_sam2(config_file=config_file, ckpt_path=None, device="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    logger.debug("[MEC] SAM3 loaded (missing=%d, unexpected=%d)", len(missing), len(unexpected))
    model = model.to(dtype).to(device).eval()
    logger.info("[MEC] SAM3 loaded: %s (%s, %s)", reg["filename"], dtype, device)
    return model


def _load_vitmatte(path: str, reg: dict, device: str):
    """Load ViTMatte via HuggingFace Transformers."""
    try:
        from transformers import VitMatteForImageMatting, VitMatteImageProcessor
    except ImportError:
        raise RuntimeError(
            "transformers package required for ViTMatte.  Install:\n"
            "  pip install transformers"
        )

    repo_or_path = path
    # If path is a HF repo ID (not a local dir), use it directly
    if os.path.isdir(path) and any(
        f.endswith((".safetensors", ".bin")) for f in os.listdir(path)
    ):
        logger.info("[MEC] Loading ViTMatte from local: %s", path)
    else:
        repo_or_path = reg["repo_id"]
        logger.info("[MEC] Loading ViTMatte from HF: %s", repo_or_path)

    model = VitMatteForImageMatting.from_pretrained(repo_or_path)
    processor = VitMatteImageProcessor()
    model.eval().to(device)
    logger.info("[MEC] ViTMatte ready (%s)", reg.get("version", "unknown"))
    return {"model": model, "processor": processor}


def _load_matanyone2(path: str, reg: dict, dtype: torch.dtype, device: str):
    """Load MatAnyone2 via InferenceCore.

    MatAnyone2 uses:
      - MatAnyone2.from_pretrained(path) via HubMixin, OR
      - Direct state_dict loading into the MatAnyone2 model class
    """
    try:
        from matanyone2.model.matanyone2 import MatAnyone2 as MatAnyone2Model
        from matanyone2.inference.inference_core import InferenceCore
    except ImportError:
        raise RuntimeError(
            "matanyone2 package required.  Install:\n"
            "  pip install git+https://github.com/pq-yang/MatAnyone2.git\n"
            "Or place matanyone2.pth in models/matanyone2/"
        )

    # Try from_pretrained first (handles HF Hub download)
    try:
        network = MatAnyone2Model.from_pretrained(path)
    except Exception:
        # Fallback: manual loading
        network = MatAnyone2Model()
        sd = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
            sd = sd["model"]
        network.load_state_dict(sd, strict=False)

    network = network.to(device).eval()
    core = InferenceCore(network, device=device)
    logger.info("[MEC] MatAnyone2 ready (%s)", device)
    return {"core": core, "network": network}


def _load_sec(path: str, reg: dict, dtype: torch.dtype, device: str):
    """Load SeC model (MLLM + SAM2 grounding encoder)."""
    raise NotImplementedError(
        "SeC model support requires the sec inference package.  "
        "This backend will be fully implemented in a future release."
    )


def _load_videomama(path: str, reg: dict, dtype: torch.dtype, device: str):
    """Load VideoMaMa model (SVD + fine-tuned UNet)."""
    raise NotImplementedError(
        "VideoMaMa model support requires the VideoMaMa pipeline package.  "
        "This backend will be fully implemented in a future release."
    )
