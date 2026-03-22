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
    # ── SAM 1 (Original Segment Anything) ────────────────────────────
    "sam_vit_h": {
        "family": "sam1",
        "repo_id": "ybelkada/segment-anything",
        "filename": "sam_vit_h_4b8939.pth",
        "config": None,
        "version": "vit_h",
        "model_dir": "sams",
    },
    "sam_vit_l": {
        "family": "sam1",
        "repo_id": "ybelkada/segment-anything",
        "filename": "sam_vit_l_0b3195.pth",
        "config": None,
        "version": "vit_l",
        "model_dir": "sams",
    },
    "sam_vit_b": {
        "family": "sam1",
        "repo_id": "ybelkada/segment-anything",
        "filename": "sam_vit_b_01ec64.pth",
        "config": None,
        "version": "vit_b",
        "model_dir": "sams",
    },
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
    # ── HQ-SAM (High-Quality Segment Anything) ──────────────────────
    "sam_hq_vit_h": {
        "family": "sam_hq",
        "repo_id": "lkeab/hq-sam",
        "filename": "sam_hq_vit_h.pth",
        "config": None,
        "version": "vit_h",
        "model_dir": "sams",
    },
    "sam_hq_vit_l": {
        "family": "sam_hq",
        "repo_id": "lkeab/hq-sam",
        "filename": "sam_hq_vit_l.pth",
        "config": None,
        "version": "vit_l",
        "model_dir": "sams",
    },
    "sam_hq_vit_b": {
        "family": "sam_hq",
        "repo_id": "lkeab/hq-sam",
        "filename": "sam_hq_vit_b.pth",
        "config": None,
        "version": "vit_b",
        "model_dir": "sams",
    },
    "sam_hq_vit_tiny": {
        "family": "sam_hq",
        "repo_id": "lkeab/hq-sam",
        "filename": "sam_hq_vit_tiny.pth",
        "config": None,
        "version": "vit_tiny",
        "model_dir": "sams",
    },
    # ── RMBG (Background Removal) ────────────────────────────────────
    "rmbg_2.0": {
        "family": "rmbg",
        "repo_id": "briaai/RMBG-2.0",
        "filename": None,
        "config": None,
        "version": "2.0",
        "model_dir": "rmbg",
    },
    "birefnet_general": {
        "family": "birefnet",
        "repo_id": "ZhengPeng7/BiRefNet",
        "filename": None,
        "config": None,
        "version": "general",
        "model_dir": "birefnet",
    },
    "birefnet_portrait": {
        "family": "birefnet",
        "repo_id": "ZhengPeng7/BiRefNet-portrait",
        "filename": None,
        "config": None,
        "version": "portrait",
        "model_dir": "birefnet",
    },
    # ── Semantic Segmentation (Face / Body / Clothes) ────────────────
    "segformer_face": {
        "family": "segformer_face",
        "repo_id": "jonathandinu/face-parsing",
        "filename": None,
        "config": None,
        "version": "face",
        "model_dir": "segformer",
    },
    "segformer_clothes": {
        "family": "segformer_clothes",
        "repo_id": "mattmdjaga/segformer_b2_clothes",
        "filename": None,
        "config": None,
        "version": "clothes",
        "model_dir": "segformer",
    },
    # ── GroundingDINO (Text-to-BBox) ─────────────────────────────────
    "groundingdino_swint_ogc": {
        "family": "groundingdino",
        "repo_id": "ShilongLiu/GroundingDINO",
        "filename": "groundingdino_swint_ogc.pth",
        "config": None,
        "version": "swint",
        "model_dir": "grounding-dino",
    },
    "groundingdino_swinb_cogcoor": {
        "family": "groundingdino",
        "repo_id": "ShilongLiu/GroundingDINO",
        "filename": "groundingdino_swinb_cogcoor.pth",
        "config": None,
        "version": "swinb",
        "model_dir": "grounding-dino",
    },
    # ── RobustVideoMatting ───────────────────────────────────────────
    "rvm_mobilenetv3": {
        "family": "rvm",
        "repo_id": "PeterL1n/RobustVideoMatting",
        "filename": "rvm_mobilenetv3.pth",
        "config": None,
        "version": "mobilenetv3",
        "model_dir": "rvm",
    },
    "rvm_resnet50": {
        "family": "rvm",
        "repo_id": "PeterL1n/RobustVideoMatting",
        "filename": "rvm_resnet50.pth",
        "config": None,
        "version": "resnet50",
        "model_dir": "rvm",
    },
    # ── CutIE (Video Object Segmentation) ────────────────────────────
    "cutie_base_mega": {
        "family": "cutie",
        "repo_id": "hkchengrex/Cutie",
        "filename": "cutie-base-mega.pth",
        "config": None,
        "version": "base-mega",
        "model_dir": "cutie",
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


def _check_vram(device: str, model_name: str) -> None:
    """Log a warning if available VRAM looks tight for the requested model."""
    if device == "cpu" or not torch.cuda.is_available():
        return
    try:
        free, total = torch.cuda.mem_get_info()
        free_gb = free / (1024 ** 3)
        if free_gb < 2.0:
            logger.warning(
                "[MEC] Low VRAM (%.1f GB free). Loading '%s' may cause OOM. "
                "Consider enabling 'offload_to_cpu' or using a smaller model variant.",
                free_gb, model_name,
            )
    except Exception:
        pass


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

    _check_vram(device, name)

    path = get_model_path(name)
    reg = MODEL_REGISTRY[name]
    family = reg["family"]

    print(f"[MEC] Loading model: {name} ({family}, {precision}, {device})")

    try:
        if family == "sam1":
            model = _load_sam1(path, reg, dtype, device)
        elif family == "sam2":
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
        elif family == "sam_hq":
            model = _load_sam_hq(path, reg, dtype, device)
        elif family == "groundingdino":
            model = _load_groundingdino(path, reg, dtype, device)
        elif family == "rvm":
            model = _load_rvm(path, reg, dtype, device)
        elif family == "cutie":
            model = _load_cutie(path, reg, dtype, device)
        elif family == "rmbg":
            model = _load_rmbg(path, reg, device)
        elif family == "birefnet":
            model = _load_birefnet(path, reg, device)
        elif family in ("segformer_face", "segformer_clothes"):
            model = _load_segformer(path, reg, device)
        else:
            raise ValueError(f"Unknown model family: {family}")
    except torch.cuda.OutOfMemoryError:
        gc.collect()
        torch.cuda.empty_cache()
        raise RuntimeError(
            f"[MEC] Out of GPU memory loading '{name}'. "
            f"Try: enable 'offload_to_cpu', use float16 precision, "
            f"or choose a smaller model variant (e.g. tiny/small instead of large)."
        )

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

            # For SeC: also check for SeC-4B-*.safetensors or SeC-4B/ directory
            if not located and reg.get("family") == "sec":
                sams_dir = os.path.join(_MODELS_DIR, model_dir)
                if os.path.isdir(sams_dir):
                    for f in os.listdir(sams_dir):
                        if f.startswith("SeC-4B") and (
                            f.endswith(".safetensors") or
                            os.path.isdir(os.path.join(sams_dir, f))
                        ):
                            located = True
                            break

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
    """Load SeC model (MLLM + SAM2 grounding encoder).

    Supports single-file safetensors (fp8/fp16/bf16/fp32) and sharded
    directory models.  Uses the SeC inference package from third_party.
    """
    # Import SeC components — look in third_party first, then installed package
    _sec_pkg = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "third_party", "Comfyui-SecNodes",
    )
    import sys
    if _sec_pkg not in sys.path and os.path.isdir(_sec_pkg):
        sys.path.insert(0, _sec_pkg)

    try:
        from inference.configuration_sec import SeCConfig
        from inference.modeling_sec import SeCModel
    except ImportError:
        raise RuntimeError(
            "SeC inference package not found.  Ensure third_party/Comfyui-SecNodes "
            "is present or install the sec inference package."
        )
    try:
        from transformers import AutoTokenizer
    except ImportError:
        raise RuntimeError("transformers package required for SeC.  pip install transformers")

    # Determine if path is a single safetensors file or a directory
    is_single_file = os.path.isfile(path) and path.endswith(".safetensors")
    is_directory = os.path.isdir(path)

    if not is_single_file and not is_directory:
        # Try to scan models/sams/ for SeC files
        sams_dir = os.path.join(_MODELS_DIR, "sams")
        candidates = []
        if os.path.isdir(sams_dir):
            for f in os.listdir(sams_dir):
                if f.startswith("SeC-4B") and f.endswith(".safetensors"):
                    candidates.append(os.path.join(sams_dir, f))
            # Also check for sharded directory
            sec_dir = os.path.join(sams_dir, "SeC-4B")
            if os.path.isdir(sec_dir):
                candidates.append(sec_dir)

        if not candidates:
            raise RuntimeError(
                "No SeC model found. Download a SeC-4B model to ComfyUI/models/sams/\n"
                "Supported formats: SeC-4B-fp16.safetensors, SeC-4B-bf16.safetensors, "
                "SeC-4B-fp32.safetensors, or SeC-4B/ sharded directory."
            )
        path = candidates[0]
        is_single_file = os.path.isfile(path)
        is_directory = os.path.isdir(path)

    # Detect precision from filename
    precision_str = "fp16"
    path_lower = path.lower()
    if "fp8" in path_lower:
        precision_str = "fp8"
    elif "fp32" in path_lower:
        precision_str = "fp32"
    elif "bf16" in path_lower:
        precision_str = "bf16"

    # Config path — use the third_party package config
    config_path = os.path.join(_sec_pkg, "model_config", "SeC-4B")
    if not os.path.isdir(config_path):
        # Fallback: use the model directory itself if sharded
        config_path = path if is_directory else os.path.dirname(path)

    # Map precision to torch dtype
    dtype_map = {
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
        "float16": torch.float16, "fp16": torch.float16,
        "float32": torch.float32, "fp32": torch.float32,
        "fp8": torch.float16,  # FP8 is loaded then converted to FP16
    }
    torch_dtype = dtype_map.get(precision_str, dtype)

    if device == "cpu" and torch_dtype != torch.float32:
        logger.warning("[MEC] CPU mode requires float32 for SeC. Converting.")
        torch_dtype = torch.float32

    use_flash_attn = torch_dtype != torch.float32

    logger.info("[MEC] Loading SeC model from %s [%s]", path, precision_str.upper())

    config = SeCConfig.from_pretrained(config_path)
    config.hydra_overrides_extra = ["++model.non_overlap_masks=false"]

    if is_single_file:
        try:
            from safetensors.torch import load_file
        except ImportError:
            raise RuntimeError("safetensors required for single-file SeC models.")

        try:
            from accelerate import init_empty_weights
            from accelerate.utils import set_module_tensor_to_device

            with init_empty_weights():
                model = SeCModel(config, use_flash_attn=use_flash_attn)

            state_dict = load_file(path)

            if precision_str == "fp8":
                for key in list(state_dict.keys()):
                    if state_dict[key].dtype == torch.float8_e4m3fn:
                        state_dict[key] = state_dict[key].to(torch.float16)

            for name, param in state_dict.items():
                set_module_tensor_to_device(model, name, device="cpu", value=param)

            model = model.eval()
        except ImportError:
            model = SeCModel(config, use_flash_attn=use_flash_attn)
            state_dict = load_file(path)
            if precision_str == "fp8":
                for key in list(state_dict.keys()):
                    if state_dict[key].dtype == torch.float8_e4m3fn:
                        state_dict[key] = state_dict[key].to(torch.float16)
            model.load_state_dict(state_dict, strict=True)
            model = model.eval()

        model = model.to(device=device, dtype=torch_dtype)
    else:
        # Directory-based (sharded) loading
        load_kwargs = {
            "config": config,
            "torch_dtype": torch_dtype,
            "use_flash_attn": use_flash_attn,
            "low_cpu_mem_usage": True,
        }
        if device != "cpu":
            load_kwargs["device_map"] = {"": device}

        model = SeCModel.from_pretrained(path, **load_kwargs).eval()

    # Set up tokenizer and prepare for generation
    tokenizer = AutoTokenizer.from_pretrained(config_path, trust_remote_code=True)
    model.preparing_for_generation(tokenizer=tokenizer, torch_dtype=torch_dtype)

    # Store metadata for potential reload
    model._sec_loading_metadata = {
        "model_path": path,
        "is_single_file": is_single_file,
        "config_path": config_path,
        "torch_dtype": torch_dtype,
        "device": device,
        "use_flash_attn": use_flash_attn,
    }

    logger.info("[MEC] SeC model loaded on %s (%s)", device, precision_str.upper())
    return model


def _load_videomama(path: str, reg: dict, dtype: torch.dtype, device: str):
    """Load VideoMaMa model (SVD + fine-tuned UNet).

    VideoMaMa requires:
      - Base SVD model (stabilityai/stable-video-diffusion-img2vid-xt)
      - Fine-tuned UNet checkpoint (SammyLim/VideoMaMa)

    Returns a dict with the pipeline and metadata.
    """
    _vmama_pkg = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "third_party", "ComfyUI-VideoMaMa",
    )
    import sys
    if _vmama_pkg not in sys.path and os.path.isdir(_vmama_pkg):
        sys.path.insert(0, _vmama_pkg)

    try:
        from pipeline_svd_mask import VideoInferencePipeline
    except ImportError:
        raise RuntimeError(
            "VideoMaMa pipeline not found.  Ensure third_party/ComfyUI-VideoMaMa "
            "is present or install the VideoMaMa package."
        )

    # Resolve paths for base SVD model and UNet checkpoint
    videomama_dir = os.path.join(_MODELS_DIR, "VideoMaMa")
    base_model_path = os.path.join(videomama_dir, "stable-video-diffusion-img2vid-xt")
    unet_path = os.path.join(videomama_dir, "VideoMaMa")

    # Auto-download if needed
    if not os.path.isdir(base_model_path) or not os.listdir(base_model_path):
        try:
            from huggingface_hub import snapshot_download
            logger.info("[MEC] Downloading SVD base model for VideoMaMa...")
            snapshot_download(
                repo_id="stabilityai/stable-video-diffusion-img2vid-xt",
                local_dir=base_model_path,
            )
        except ImportError:
            raise RuntimeError(
                "SVD base model not found and huggingface_hub not installed.\n"
                "Download stabilityai/stable-video-diffusion-img2vid-xt to "
                f"{base_model_path}"
            )

    if not os.path.isdir(unet_path) or not os.listdir(unet_path):
        try:
            from huggingface_hub import snapshot_download
            logger.info("[MEC] Downloading VideoMaMa UNet checkpoint...")
            snapshot_download(
                repo_id="SammyLim/VideoMaMa",
                local_dir=unet_path,
            )
        except ImportError:
            raise RuntimeError(
                "VideoMaMa UNet checkpoint not found and huggingface_hub not installed.\n"
                "Download SammyLim/VideoMaMa to " + unet_path
            )

    weight_dtype = dtype

    logger.info("[MEC] Loading VideoMaMa pipeline...")
    pipeline = VideoInferencePipeline(
        base_model_path=base_model_path,
        unet_checkpoint_path=unet_path,
        weight_dtype=weight_dtype,
        device=device,
        enable_model_cpu_offload=True,
        vae_encode_chunk_size=4,
        attention_mode="auto",
        enable_vae_tiling=False,
        enable_vae_slicing=True,
    )

    logger.info("[MEC] VideoMaMa pipeline loaded on %s", device)
    return {"pipeline": pipeline, "device": device, "dtype": dtype}


def _load_sam_hq(path: str, reg: dict, dtype: torch.dtype, device: str):
    """Load HQ-SAM model."""
    try:
        from segment_anything_hq import sam_model_registry
    except ImportError:
        try:
            from segment_anything import sam_model_registry
        except ImportError:
            raise RuntimeError(
                "segment_anything_hq or segment_anything is required for HQ-SAM.\n"
                "  pip install segment-anything-hq"
            )

    vit_type = reg.get("version", "vit_h")
    model = sam_model_registry[vit_type](checkpoint=path)
    model = model.to(dtype).to(device).eval()
    logger.info("[MEC] HQ-SAM loaded: %s (%s)", vit_type, device)
    return {"model": model, "model_type": vit_type}


def _load_groundingdino(path: str, reg: dict, dtype: torch.dtype, device: str):
    """Load GroundingDINO model for text-to-bounding-box grounding."""
    _gdino_pkg = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "third_party", "GroundingDINO",
    )
    import sys
    if _gdino_pkg not in sys.path and os.path.isdir(_gdino_pkg):
        sys.path.insert(0, _gdino_pkg)

    try:
        from groundingdino.util.inference import load_model as gdino_load_model
    except ImportError:
        try:
            from GroundingDINO.groundingdino.util.inference import load_model as gdino_load_model
        except ImportError:
            raise RuntimeError(
                "groundingdino package required for text prompt masking.\n"
                "  pip install groundingdino-py"
            )

    # Resolve config file for the variant
    version = reg.get("version", "swint")
    config_name = (
        "GroundingDINO_SwinT_OGC.py" if version == "swint"
        else "GroundingDINO_SwinB_cfg.py"
    )
    # Try to find config in common locations
    config_candidates = [
        os.path.join(_gdino_pkg, "groundingdino", "config", config_name),
        os.path.join(os.path.dirname(path), config_name),
    ]
    # Also try installed package config
    try:
        import groundingdino
        pkg_dir = os.path.dirname(groundingdino.__file__)
        config_candidates.append(os.path.join(pkg_dir, "config", config_name))
    except (ImportError, AttributeError):
        pass

    config_path = None
    for c in config_candidates:
        if os.path.isfile(c):
            config_path = c
            break

    if config_path is None:
        raise RuntimeError(
            f"GroundingDINO config '{config_name}' not found.\n"
            f"Searched: {config_candidates}"
        )

    model = gdino_load_model(config_path, path, device=device)
    model = model.eval()
    logger.info("[MEC] GroundingDINO loaded: %s (%s)", version, device)
    return {"model": model, "version": version, "device": device}


def _load_sam1(path: str, reg: dict, dtype: torch.dtype, device: str):
    """Load original SAM (Segment Anything v1) model."""
    try:
        from segment_anything import sam_model_registry
    except ImportError:
        raise RuntimeError(
            "segment_anything package is required for SAM1.  Install with:\n"
            "  pip install segment-anything"
        )

    vit_type = reg.get("version", "vit_h")
    model = sam_model_registry[vit_type](checkpoint=path)
    model = model.to(dtype).to(device).eval()
    logger.info("[MEC] SAM1 loaded: %s (%s)", vit_type, device)
    return {"model": model, "model_type": vit_type}


def _load_rmbg(path: str, reg: dict, device: str):
    """Load RMBG-2.0 background removal model via transformers."""
    try:
        from transformers import AutoModelForImageSegmentation, AutoImageProcessor
    except ImportError:
        raise RuntimeError(
            "transformers is required for RMBG.  Install:\n"
            "  pip install transformers"
        )

    repo_or_path = path if os.path.isdir(path) else reg["repo_id"]
    logger.info("[MEC] Loading RMBG from %s", repo_or_path)
    model = AutoModelForImageSegmentation.from_pretrained(
        repo_or_path, trust_remote_code=True,
    )
    try:
        processor = AutoImageProcessor.from_pretrained(repo_or_path, trust_remote_code=True)
    except Exception:
        processor = None
    model = model.to(device).eval()
    logger.info("[MEC] RMBG loaded (%s)", device)
    return {"model": model, "processor": processor}


def _load_birefnet(path: str, reg: dict, device: str):
    """Load BiRefNet bilateral reference network."""
    try:
        from transformers import AutoModelForImageSegmentation, AutoImageProcessor
    except ImportError:
        raise RuntimeError(
            "transformers is required for BiRefNet.  Install:\n"
            "  pip install transformers"
        )

    repo_or_path = path if os.path.isdir(path) else reg["repo_id"]
    logger.info("[MEC] Loading BiRefNet from %s", repo_or_path)
    model = AutoModelForImageSegmentation.from_pretrained(
        repo_or_path, trust_remote_code=True,
    )
    try:
        processor = AutoImageProcessor.from_pretrained(repo_or_path, trust_remote_code=True)
    except Exception:
        processor = None
    model = model.to(device).eval()
    logger.info("[MEC] BiRefNet loaded: %s (%s)", reg.get("version"), device)
    return {"model": model, "processor": processor}


def _load_segformer(path: str, reg: dict, device: str):
    """Load SegFormer model for face/body/clothes parsing."""
    try:
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
    except ImportError:
        raise RuntimeError(
            "transformers is required for SegFormer.  Install:\n"
            "  pip install transformers"
        )

    repo_or_path = path if os.path.isdir(path) else reg["repo_id"]
    logger.info("[MEC] Loading SegFormer from %s", repo_or_path)
    model = SegformerForSemanticSegmentation.from_pretrained(repo_or_path)
    processor = SegformerImageProcessor.from_pretrained(repo_or_path)
    model = model.to(device).eval()
    logger.info("[MEC] SegFormer loaded: %s (%s)", reg.get("version"), device)
    return {"model": model, "processor": processor}


def _load_rvm(path: str, reg: dict, dtype: torch.dtype, device: str):
    """Load RobustVideoMatting model."""
    _rvm_pkg = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "third_party", "RobustVideoMatting",
    )
    import sys
    if _rvm_pkg not in sys.path and os.path.isdir(_rvm_pkg):
        sys.path.insert(0, _rvm_pkg)

    try:
        from model import MattingNetwork
    except ImportError:
        raise RuntimeError(
            "RobustVideoMatting model module not found.\n"
            "Ensure third_party/RobustVideoMatting is present."
        )

    variant = reg.get("version", "mobilenetv3")
    model = MattingNetwork(variant)
    sd = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(sd)
    model = model.to(dtype).to(device).eval()
    logger.info("[MEC] RVM loaded: %s (%s)", variant, device)
    return {"model": model, "variant": variant}


def _load_cutie(path: str, reg: dict, dtype: torch.dtype, device: str):
    """Load CutIE (video object segmentation) model."""
    _cutie_pkg = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "third_party", "Cutie",
    )
    import sys
    if _cutie_pkg not in sys.path and os.path.isdir(_cutie_pkg):
        sys.path.insert(0, _cutie_pkg)

    try:
        from cutie.model.cutie import CUTIE
        from cutie.inference.inference_core import InferenceCore as CutieInferenceCore
        from omegaconf import open_dict
        from hydra import compose, initialize_config_dir
    except ImportError:
        raise RuntimeError(
            "Cutie package required. Ensure third_party/Cutie is present.\n"
            "  pip install hydra-core omegaconf"
        )

    config_dir = os.path.join(_cutie_pkg, "cutie", "config")
    with initialize_config_dir(config_path=config_dir, version_base="1.1"):
        cfg = compose(config_name="eval_config.yaml")

    with open_dict(cfg):
        cfg.weights = path

    cutie = CUTIE(cfg).to(device).eval()
    sd = torch.load(path, map_location=device, weights_only=True)
    cutie.load_weights(sd)
    processor = CutieInferenceCore(cutie, cfg=cfg)
    logger.info("[MEC] CutIE loaded: %s (%s)", reg.get("version"), device)
    return {"processor": processor, "model": cutie}
