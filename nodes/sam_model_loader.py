"""
SAMModelLoaderMEC – Load SAM2/SAM2.1/SAM3 (or original SAM) checkpoints.

Supports:
  - SAM2 / SAM2.1 via sam2 package (kijai/ComfyUI-segment-anything-2 pattern)
  - SAM3 via sam2 package (PozzettiAndrea/ComfyUI-SAM3 uses SAM2 infra)
  - Original SAM (vit_h/l/b) via segment_anything package
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


# ── Known SAM2 config mapping (SAM2 requires a matching config) ───────
SAM2_CONFIGS = {
    "sam2_hiera_tiny":    "sam2_hiera_t.yaml",
    "sam2_hiera_small":   "sam2_hiera_s.yaml",
    "sam2_hiera_base":    "sam2_hiera_b+.yaml",
    "sam2_hiera_large":   "sam2_hiera_l.yaml",
    "sam2.1_hiera_tiny":  "sam2.1_hiera_t.yaml",
    "sam2.1_hiera_small": "sam2.1_hiera_s.yaml",
    "sam2.1_hiera_base":  "sam2.1_hiera_b+.yaml",
    "sam2.1_hiera_large": "sam2.1_hiera_l.yaml",
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
    """Detect SAM2 config file from model filename."""
    name = model_name.lower().replace("-", "_").replace(" ", "_")
    for key, config in SAM2_CONFIGS.items():
        normalized_key = key.replace(".", "").replace("_", "")
        normalized_name = name.replace(".", "").replace("_", "")
        if normalized_key in normalized_name:
            return config
    # Heuristic fallback
    is_21 = "2.1" in name or "2_1" in name
    if "tiny" in name or "_t." in name:
        return "sam2.1_hiera_t.yaml" if is_21 else "sam2_hiera_t.yaml"
    if "small" in name or "_s." in name:
        return "sam2.1_hiera_s.yaml" if is_21 else "sam2_hiera_s.yaml"
    if "base" in name or "_b+" in name or "_b." in name:
        return "sam2.1_hiera_b+.yaml" if is_21 else "sam2_hiera_b+.yaml"
    # Default to large
    return "sam2.1_hiera_l.yaml" if is_21 else "sam2_hiera_l.yaml"


class SAMModelLoaderMEC:
    """Load a Segment Anything Model (SAM / SAM2 / SAM2.1 / SAM3).

    Loading patterns follow reference implementations:
      - SAM2/2.1: kijai/ComfyUI-segment-anything-2 (build_sam2 with config)
      - SAM3: PozzettiAndrea/ComfyUI-SAM3 (uses SAM2 infrastructure)
      - Original SAM: segment_anything package
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

        # Try to find locally first
        if HAS_FOLDER_PATHS:
            for key in ("sams", "sam2", "sam3"):
                if key in folder_paths.folder_names_and_paths:
                    try:
                        path = folder_paths.get_full_path(key, clean_name)
                        if path and os.path.exists(path):
                            return path
                    except Exception:
                        continue

        # Auto-download from HuggingFace Hub if needed
        if needs_download or not HAS_FOLDER_PATHS:
            return SAMModelLoaderMEC._auto_download(clean_name)

        # Not found locally and not marked for download
        raise FileNotFoundError(
            f"SAM model '{clean_name}' not found. "
            f"Place it in ComfyUI/models/sams/ or models/sam2/, "
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
        """Load using sam2 package (covers SAM2, SAM2.1, and SAM3)."""
        # Method 1: build_sam2 with config (correct pattern)
        try:
            from sam2.build_sam import build_sam2
            config = _detect_config(model_name)
            model = build_sam2(
                config_file=config,
                ckpt_path=model_path,
                device="cpu",
            )
            model = model.to(torch_dtype)
            if not offload:
                model = model.to(device)
            return model, "build_sam2"
        except ImportError:
            logger.debug("[MEC] sam2 package not available")
        except Exception as e:
            logger.debug(f"[MEC] build_sam2 failed: {e}")

        # Method 2: direct state_dict load
        try:
            state = torch.load(model_path, map_location="cpu", weights_only=False)
            if isinstance(state, dict):
                if "model" in state:
                    state = state["model"]
                elif "state_dict" in state:
                    state = state["state_dict"]
            if hasattr(state, 'image_encoder'):
                model = state.to(torch_dtype)
                if not offload:
                    model = model.to(device)
                return model, "direct_model"
        except Exception:
            pass

        return None, "failed"

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
        try:
            state = torch.load(model_path, map_location="cpu", weights_only=False)
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            logger.warning("[MEC] Using generic state_dict loader — predictor may not work")
            return {"state_dict": state, "dtype": torch_dtype, "device": device}, "generic"
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
