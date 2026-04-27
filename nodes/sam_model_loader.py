"""
SAMModelLoaderMEC – Load SAM2/SAM2.1/SAM3 (or original SAM) checkpoints.

Supports:
  - SAM2 / SAM2.1 via official sam2 package (pip install SAM-2)
  - SAM3 via sam2 package (uses SAM2 infrastructure)
  - Original SAM (vit_h/l/b) via segment_anything package
  - .safetensors / .pt / .pth checkpoint formats
  - Optional CPU-offload to save VRAM
  - Automatic model type detection from filename
  - Auto-download from HuggingFace Hub when model not found locally
"""

import os
import torch
import gc
import logging

logger = logging.getLogger("MEC")

try:
    import folder_paths
    HAS_FOLDER_PATHS = True
except ImportError:
    HAS_FOLDER_PATHS = False


# ──────────────────────────────────────────────────────────────────────
#  Device-juggling helpers for inference nodes
# ──────────────────────────────────────────────────────────────────────
#
# SAM/SAM2 wrappers may have ``offload_to_cpu=True`` set, in which case
# the ``model`` object lives on CPU between calls. Inference nodes
# should call ``move_to_inference_device(wrapper)`` at entry and
# ``restore_device(wrapper)`` in their ``finally`` block to avoid
# device-mismatch errors when chaining multiple SAM inferences.
#
# Both helpers are no-ops when the wrapper is missing a device field
# or when the underlying model has no ``.to()`` method (e.g. a raw
# state dict). They never raise.

def _model_to(obj, device):
    """Best-effort .to(device); silently ignore objects that can't move."""
    if obj is None:
        return obj
    try:
        if hasattr(obj, "to"):
            obj.to(device)
        # Some SAM2 wrappers expose .model rather than being nn.Module
        inner = getattr(obj, "model", None)
        if inner is not None and hasattr(inner, "to") and inner is not obj:
            inner.to(device)
    except Exception as exc:
        logger.warning("[MEC] SAM .to(%s) failed: %s", device, exc)
    return obj


def move_to_inference_device(sam_wrapper):
    """Move a SAM wrapper's underlying model onto its inference device.

    Returns the device string actually used. Safe to call repeatedly.
    """
    if not isinstance(sam_wrapper, dict):
        return None
    device = sam_wrapper.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    if sam_wrapper.get("offload_to_cpu"):
        # When offload is enabled, the resting place is CPU; we still
        # need to move it onto the inference device for the call.
        target = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        target = device
    _model_to(sam_wrapper.get("model"), target)
    return target


def restore_device(sam_wrapper):
    """Move the SAM wrapper's model back to its resting device.

    For ``offload_to_cpu=True`` wrappers, this returns the model to CPU.
    Otherwise it's a no-op (the model never left ``original_device``).
    Always followed by an empty_cache() to free VRAM.
    """
    if not isinstance(sam_wrapper, dict):
        return
    if sam_wrapper.get("offload_to_cpu"):
        _model_to(sam_wrapper.get("model"), "cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── Official SAM2 config mapping for build_sam2 ──────────────────────
# Keys match filename stems; values are config paths for the official
# sam2 package (pip install SAM-2).
SAM2_CONFIGS = {
    "sam2_hiera_tiny":    "configs/sam2/sam2_hiera_t.yaml",
    "sam2_hiera_small":   "configs/sam2/sam2_hiera_s.yaml",
    "sam2_hiera_base":    "configs/sam2/sam2_hiera_b+.yaml",
    "sam2_hiera_large":   "configs/sam2/sam2_hiera_l.yaml",
    "sam2.1_hiera_tiny":  "configs/sam2.1/sam2.1_hiera_t.yaml",
    "sam2.1_hiera_small": "configs/sam2.1/sam2.1_hiera_s.yaml",
    "sam2.1_hiera_base":  "configs/sam2.1/sam2.1_hiera_b+.yaml",
    "sam2.1_hiera_large": "configs/sam2.1/sam2.1_hiera_l.yaml",
}

# ── HuggingFace Hub auto-download registry ────────────────────────────
_DOWNLOAD_REGISTRY = {
    "sam2.1_hiera_large.pt": {
        "repo_id": "facebook/sam2.1-hiera-large",
        "filename": "sam2.1_hiera_large.pt",
    },
    "sam2.1_hiera_base_plus.pt": {
        "repo_id": "facebook/sam2.1-hiera-base-plus",
        "filename": "sam2.1_hiera_base_plus.pt",
    },
    "sam2.1_hiera_small.pt": {
        "repo_id": "facebook/sam2.1-hiera-small",
        "filename": "sam2.1_hiera_small.pt",
    },
    "sam2.1_hiera_tiny.pt": {
        "repo_id": "facebook/sam2.1-hiera-tiny",
        "filename": "sam2.1_hiera_tiny.pt",
    },
    "sam2_hiera_large.pt": {
        "repo_id": "facebook/sam2-hiera-large",
        "filename": "sam2_hiera_large.pt",
    },
    "sam2_hiera_base_plus.pt": {
        "repo_id": "facebook/sam2-hiera-base-plus",
        "filename": "sam2_hiera_base_plus.pt",
    },
    "sam2_hiera_small.pt": {
        "repo_id": "facebook/sam2-hiera-small",
        "filename": "sam2_hiera_small.pt",
    },
    "sam2_hiera_tiny.pt": {
        "repo_id": "facebook/sam2-hiera-tiny",
        "filename": "sam2_hiera_tiny.pt",
    },
    "sam_vit_h_4b8939.pth": {
        "repo_id": "ybelkada/segment-anything",
        "filename": "checkpoints/sam_vit_h_4b8939.pth",
        "subfolder": "checkpoints",
    },
    "sam_vit_l_0b3195.pth": {
        "repo_id": "ybelkada/segment-anything",
        "filename": "checkpoints/sam_vit_l_0b3195.pth",
        "subfolder": "checkpoints",
    },
    "sam_vit_b_01ec64.pth": {
        "repo_id": "ybelkada/segment-anything",
        "filename": "checkpoints/sam_vit_b_01ec64.pth",
        "subfolder": "checkpoints",
    },
}


def _detect_config(model_name):
    """Detect SAM2 config path from model filename.

    Returns a config name suitable for the official sam2 ``build_sam2``
    API (e.g. ``configs/sam2.1/sam2.1_hiera_t.yaml``).
    """
    name = model_name.lower().replace("-", "_").replace(" ", "_")
    for key, config in SAM2_CONFIGS.items():
        normalized_key = key.replace(".", "").replace("_", "")
        normalized_name = name.replace(".", "").replace("_", "")
        if normalized_key in normalized_name:
            return config
    # Heuristic fallback
    is_21 = "2.1" in name or "2_1" in name
    prefix = "configs/sam2.1/sam2.1" if is_21 else "configs/sam2/sam2"
    if "tiny" in name or "_t." in name:
        return f"{prefix}_hiera_t.yaml"
    if "small" in name or "_s." in name:
        return f"{prefix}_hiera_s.yaml"
    if "base" in name or "_b+" in name or "_b." in name:
        return f"{prefix}_hiera_b+.yaml"
    # Default to large
    return f"{prefix}_hiera_l.yaml"


class SAMModelLoaderMEC:
    """Load a Segment Anything Model (SAM / SAM2 / SAM2.1 / SAM3).

    Uses the official ``sam2`` Python package (pip install SAM-2) for
    SAM2/2.1/SAM3 models.  Falls back to ``segment_anything`` for
    original SAM (ViT-H/L/B).
    """

    SUPPORTED_TYPES = [
        "auto", "sam2", "sam2.1", "sam3",
        "sam_vit_h", "sam_vit_l", "sam_vit_b",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        model_files = []
        if HAS_FOLDER_PATHS:
            for key in ("sams", "sam2", "sam3"):
                if key in folder_paths.folder_names_and_paths:
                    try:
                        model_files += folder_paths.get_filename_list(key)
                    except Exception:
                        pass
            # Also scan common extra locations
            cls._scan_extra_paths(model_files)
        if not model_files:
            model_files = ["(place model in models/sams/ or models/sam2/)"]

        # Add well-known downloadable models that aren't already present
        for name in _DOWNLOAD_REGISTRY:
            if name not in model_files:
                model_files.append(f"[download] {name}")

        return {
            "required": {
                "model_name": (sorted(set(model_files)), {
                    "tooltip": (
                        "SAM checkpoint (.pth/.pt/.safetensors).\n"
                        "Models prefixed with [download] will be auto-downloaded "
                        "from HuggingFace Hub on first use."
                    ),
                }),
                "model_type": (cls.SUPPORTED_TYPES, {
                    "default": "auto",
                    "tooltip": (
                        "Model architecture. 'auto' detects from filename.\n"
                        "sam2/sam2.1: Segment Anything 2 (requires sam2 package)\n"
                        "sam3: SAM3 (uses SAM2 infrastructure)\n"
                        "sam_vit_*: Original SAM (requires segment_anything)"
                    ),
                }),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
                "offload_to_cpu": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Keep model on CPU between inferences. "
                        "Saves ~2-4 GB VRAM at cost of slower inference."
                    ),
                }),
                "dtype": (["float16", "bfloat16", "float32"], {
                    "default": "float16",
                    "tooltip": "Model precision. float16 saves VRAM, bfloat16 for newer GPUs.",
                }),
            },
        }

    @classmethod
    def _scan_extra_paths(cls, model_files):
        """Scan additional common model directories."""
        if not HAS_FOLDER_PATHS:
            return
        try:
            base = folder_paths.base_path  # ComfyUI root
            extra_dirs = [
                os.path.join(base, "models", "sam2"),
                os.path.join(base, "models", "sams"),
                os.path.join(base, "models", "sam3"),
            ]
            for d in extra_dirs:
                if os.path.isdir(d):
                    for f in os.listdir(d):
                        if f.endswith((".pth", ".pt", ".safetensors", ".bin")):
                            if f not in model_files:
                                model_files.append(f)
        except Exception:
            pass

    RETURN_TYPES = ("SAM_MODEL",)
    RETURN_NAMES = ("sam_model",)
    FUNCTION = "load"
    CATEGORY = "MaskEditControl/SAM"
    DESCRIPTION = (
        "Load SAM/SAM2/SAM2.1/SAM3 model. "
        "Auto-detects architecture from filename. "
        "Supports VRAM offload."
    )

    def load(self, model_name: str, model_type: str, device: str,
             offload_to_cpu: bool, dtype: str):

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map.get(dtype, torch.float16)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        model_path = self._resolve_path(model_name)
        clean_name = model_name.replace("[download] ", "")
        detected_type = self._detect_type(clean_name) if model_type == "auto" else model_type

        sam_model = None
        load_method = "unknown"

        if detected_type in ("sam2", "sam2.1", "sam3"):
            sam_model, load_method = self._load_sam2_family(
                model_path, clean_name, detected_type, torch_dtype, device, offload_to_cpu
            )
        else:
            sam_model, load_method = self._load_original_sam(
                model_path, detected_type, torch_dtype, device, offload_to_cpu
            )

        if sam_model is None:
            sam_model, load_method = self._load_generic_fallback(
                model_path, torch_dtype, device, offload_to_cpu
            )

        result = {
            "model": sam_model,
            "model_type": detected_type,
            "device": device,
            "original_device": device,  # the device the model was loaded onto;
                                          # callers should restore here in finally:
                                          # see move_to_inference_device / restore_device.
            "offload_to_cpu": offload_to_cpu,
            "dtype": torch_dtype,
            "model_path": model_path,
            "load_method": load_method,
        }

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(
            f"[MEC] Loaded {detected_type} via {load_method}: "
            f"{model_name} (dtype={dtype}, device={device})"
        )
        return (result,)

    # ── Path resolution ───────────────────────────────────────────────
    @staticmethod
    def _resolve_path(model_name):
        # Handle auto-download prefix
        clean_name = model_name
        needs_download = False
        if model_name.startswith("[download] "):
            clean_name = model_name[len("[download] "):]
            needs_download = True

        # Strategy 1: folder_paths registered keys
        if HAS_FOLDER_PATHS:
            for key in ("sams", "sam2", "sam3"):
                if key in folder_paths.folder_names_and_paths:
                    try:
                        path = folder_paths.get_full_path(key, clean_name)
                        if path and os.path.exists(path):
                            return path
                    except Exception:
                        continue

        # Strategy 2: Direct filesystem scan of common model directories
        if HAS_FOLDER_PATHS:
            try:
                models_dir = getattr(folder_paths, 'models_dir', None)
                if not models_dir:
                    models_dir = os.path.join(folder_paths.base_path, "models")
                for subdir in ("sam2", "sams", "sam3"):
                    candidate = os.path.join(models_dir, subdir, clean_name)
                    if os.path.exists(candidate):
                        return candidate
            except Exception:
                pass

        # Strategy 3: Check if it's an absolute path that exists
        if os.path.isabs(clean_name) and os.path.exists(clean_name):
            return clean_name

        # Auto-download from HuggingFace Hub if flagged
        if needs_download:
            return SAMModelLoaderMEC._auto_download(clean_name)

        # Not found — give a helpful error
        search_paths = []
        if HAS_FOLDER_PATHS:
            models_dir = getattr(folder_paths, 'models_dir',
                                 os.path.join(getattr(folder_paths, 'base_path', ''), "models"))
            search_paths = [os.path.join(models_dir, d) for d in ("sam2", "sams", "sam3")]
        raise FileNotFoundError(
            f"SAM model '{clean_name}' not found.\n"
            f"Searched: {', '.join(search_paths)}\n"
            f"Place the model file in one of those directories, "
            f"or select a [download] model to auto-download from HuggingFace."
        )

    @staticmethod
    def _auto_download(model_name):
        """Download model from HuggingFace Hub to the sam2 models directory."""
        entry = _DOWNLOAD_REGISTRY.get(model_name)
        if not entry:
            raise FileNotFoundError(
                f"Model '{model_name}' not in download registry. "
                f"Available: {', '.join(_DOWNLOAD_REGISTRY.keys())}"
            )

        # Determine download directory
        download_dir = None
        if HAS_FOLDER_PATHS:
            for key in ("sam2", "sams"):
                if key in folder_paths.folder_names_and_paths:
                    paths = folder_paths.folder_names_and_paths[key]
                    if isinstance(paths, (list, tuple)) and len(paths) > 0:
                        candidate = paths[0] if isinstance(paths[0], str) else paths[0][0] if isinstance(paths[0], (list, tuple)) else None
                        if candidate:
                            download_dir = candidate
                            break
        if not download_dir:
            download_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                         "models", "sam2")

        os.makedirs(download_dir, exist_ok=True)
        dest_path = os.path.join(download_dir, model_name)

        # If already downloaded, return
        if os.path.exists(dest_path):
            logger.info(f"[MEC] Model already downloaded: {dest_path}")
            return dest_path

        repo_id = entry["repo_id"]
        filename = entry["filename"]
        logger.info(f"[MEC] Auto-downloading {model_name} from {repo_id}...")

        try:
            from huggingface_hub import hf_hub_download
            downloaded = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=download_dir,
                local_dir_use_symlinks=False,
            )
            # hf_hub_download may put file in a subdirectory; move to expected location
            if os.path.exists(downloaded) and downloaded != dest_path:
                import shutil
                shutil.move(downloaded, dest_path)
            logger.info(f"[MEC] Downloaded {model_name} → {dest_path}")
            return dest_path
        except ImportError:
            raise RuntimeError(
                f"huggingface_hub package not installed. Install with:\n"
                f"  pip install huggingface_hub\n"
                f"Or manually download '{model_name}' from https://huggingface.co/{repo_id} "
                f"and place it in {download_dir}"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to download {model_name} from {repo_id}: {e}")

    # ── Type detection ────────────────────────────────────────────────
    @staticmethod
    def _detect_type(model_name):
        name = model_name.lower()
        if "sam3" in name:
            return "sam3"
        if "sam2.1" in name or "sam2_1" in name:
            return "sam2.1"
        if "sam2" in name:
            return "sam2"
        if "vit_h" in name:
            return "sam_vit_h"
        if "vit_l" in name:
            return "sam_vit_l"
        if "vit_b" in name:
            return "sam_vit_b"
        return "sam2"

    # ── SAM2 / SAM2.1 / SAM3 ─────────────────────────────────────────
    def _load_sam2_family(self, model_path, model_name, model_type,
                          torch_dtype, device, offload):
        """Load SAM2/2.1/SAM3 model using the official sam2 package.

        Strategy:
          1. Load state_dict from any format (safetensors/pt/pth)
          2. Build model architecture via ``build_sam2`` (no checkpoint)
          3. Inject state_dict into the architecture
          4. Create a proper SAM2Base model ready for SAM2ImagePredictor
        """
        config_name = _detect_config(model_name)

        # ── Step 1: Load state_dict ────────────────────────────────────
        state_dict = self._load_state_dict(model_path)

        # ── Step 2: Build architecture + load weights ──────────────────
        try:
            from sam2.build_sam import build_sam2

            model = build_sam2(
                config_file=config_name,
                ckpt_path=None,      # architecture only
                device="cpu",
            )

            # Inject weights (strict=False tolerates minor key mismatches
            # between different SAM2 revisions / fp16 conversions)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                logger.debug(f"[MEC] SAM2 missing keys: {len(missing)}")
            if unexpected:
                logger.debug(f"[MEC] SAM2 unexpected keys: {len(unexpected)}")

            model = model.to(torch_dtype)
            if not offload:
                model = model.to(device)
            model.eval()
            return model, "build_sam2"

        except ImportError:
            logger.error(
                "[MEC] sam2 package not found. Install with:\n"
                "  pip install git+https://github.com/facebookresearch/sam2.git"
            )
        except Exception as e:
            logger.warning(f"[MEC] build_sam2 failed: {e}")

        # ── Fallback: return state_dict wrapper ────────────────────────
        logger.warning("[MEC] Returning raw state_dict — SAM2 inference may fail")
        return {"state_dict": state_dict, "dtype": torch_dtype, "device": device}, "state_dict_only"

    # ── State dict loader (safetensors / pt / pth) ────────────────────
    @staticmethod
    def _load_state_dict(model_path):
        """Load state_dict from any supported checkpoint format."""

        # safetensors (preferred — fast, safe)
        if model_path.endswith(".safetensors"):
            try:
                from safetensors.torch import load_file
                return load_file(model_path)
            except ImportError:
                pass

        # comfy.utils (handles both safetensors and pt)
        try:
            from comfy.utils import load_torch_file
            sd = load_torch_file(model_path)
            # Official checkpoints nest under "model" key
            if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
                sd = sd["model"]
            return sd
        except ImportError:
            pass

        # torch.load fallback (pt/pth only)
        sd = torch.load(model_path, map_location="cpu", weights_only=True)
        if isinstance(sd, dict) and "model" in sd:
            sd = sd["model"]
        return sd

    # ── Original SAM ──────────────────────────────────────────────────
    @staticmethod
    def _load_original_sam(model_path, detected_type, torch_dtype, device, offload):
        try:
            from segment_anything import sam_model_registry
            arch_map = {"sam_vit_h": "vit_h", "sam_vit_l": "vit_l", "sam_vit_b": "vit_b"}
            arch = arch_map.get(detected_type, "vit_h")
            model = sam_model_registry[arch](checkpoint=model_path)
            model = model.to(torch_dtype)
            if not offload:
                model = model.to(device)
            return model, "sam_registry"
        except ImportError:
            logger.debug("[MEC] segment_anything not available")
        except Exception as e:
            logger.debug(f"[MEC] Original SAM failed: {e}")
        return None, "failed"

    # ── Generic fallback ──────────────────────────────────────────────
    @staticmethod
    def _load_generic_fallback(model_path, torch_dtype, device, offload):
        """Last-resort loader: return raw state_dict."""
        try:
            sd = SAMModelLoaderMEC._load_state_dict(model_path)
            logger.warning("[MEC] Using generic state_dict loader — predictor may not work")
            return {"state_dict": sd, "dtype": torch_dtype, "device": device}, "generic"
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model '{model_path}': {e}\n"
                f"If this is a .safetensors file, ensure 'safetensors' package is installed.\n"
                f"For SAM2/2.1 models, install: pip install git+https://github.com/facebookresearch/sam2.git"
            )
