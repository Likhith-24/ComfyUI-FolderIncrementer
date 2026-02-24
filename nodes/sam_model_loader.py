"""
SAMModelLoaderMEC – Load SAM2/SAM2.1/SAM3 (or original SAM) checkpoints.

Supports:
  - SAM2 / SAM2.1 via sam2 package (kijai/ComfyUI-segment-anything-2 pattern)
  - SAM3 via sam2 package (PozzettiAndrea/ComfyUI-SAM3 uses SAM2 infra)
  - Original SAM (vit_h/l/b) via segment_anything package
  - Optional CPU-offload to save VRAM
  - Automatic model type detection from filename
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
        if not model_files:
            model_files = ["(place model in models/sams/ or models/sam2/)"]

        return {
            "required": {
                "model_name": (sorted(set(model_files)), {
                    "tooltip": "SAM checkpoint (.pth/.pt/.safetensors)",
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
        detected_type = self._detect_type(model_name) if model_type == "auto" else model_type

        sam_model = None
        load_method = "unknown"

        if detected_type in ("sam2", "sam2.1", "sam3"):
            sam_model, load_method = self._load_sam2_family(
                model_path, model_name, detected_type, torch_dtype, device, offload_to_cpu
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
        if not HAS_FOLDER_PATHS:
            raise FileNotFoundError("folder_paths not available")
        for key in ("sams", "sam2", "sam3"):
            if key in folder_paths.folder_names_and_paths:
                try:
                    path = folder_paths.get_full_path(key, model_name)
                    if path and os.path.exists(path):
                        return path
                except Exception:
                    continue
        raise FileNotFoundError(
            f"SAM model '{model_name}' not found. "
            f"Place it in ComfyUI/models/sams/ or models/sam2/"
        )

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
