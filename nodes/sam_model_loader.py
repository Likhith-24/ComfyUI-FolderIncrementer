"""
SAMModelLoaderMEC – Load SAM2/SAM3 (or original SAM) checkpoints with
optional CPU-offload to save VRAM.  Supports automatic model type detection.
"""

import os
import torch
import gc
import folder_paths


class SAMModelLoaderMEC:
    """Load a Segment Anything Model (SAM / SAM2 / SAM3) from a checkpoint
    file.  Provides a VRAM-saving offload toggle that keeps the model on CPU
    until inference and moves it back after."""

    SUPPORTED_TYPES = ["sam2", "sam2.1", "sam3", "sam_vit_h", "sam_vit_l", "sam_vit_b", "auto"]

    @classmethod
    def INPUT_TYPES(cls):
        # Discover .pth / .pt / .safetensors in ComfyUI models dirs
        model_files = (
            folder_paths.get_filename_list("sams")
            if "sams" in folder_paths.folder_names_and_paths
            else []
        )
        # Also check sam2 folder
        if "sam2" in folder_paths.folder_names_and_paths:
            model_files += folder_paths.get_filename_list("sam2")

        if not model_files:
            model_files = ["(place model in models/sams/)"]

        return {
            "required": {
                "model_name": (model_files, {"tooltip": "SAM checkpoint file (.pth/.pt/.safetensors)"}),
                "model_type": (cls.SUPPORTED_TYPES, {
                    "default": "auto",
                    "tooltip": "Model architecture. 'auto' will attempt to detect from checkpoint.",
                }),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
                "offload_to_cpu": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "When enabled, model is kept on CPU and only moved to GPU for "
                        "inference, then moved back.  Saves ~2-4 GB VRAM at the cost of "
                        "slightly slower inference."
                    ),
                }),
                "dtype": (["float32", "float16", "bfloat16"], {
                    "default": "float16",
                    "tooltip": "Model precision.  float16 saves VRAM, bfloat16 for newer GPUs.",
                }),
            },
        }

    RETURN_TYPES = ("SAM_MODEL",)
    RETURN_NAMES = ("sam_model",)
    FUNCTION = "load"
    CATEGORY = "MaskEditControl/SAM"
    DESCRIPTION = "Load SAM/SAM2/SAM3 model with optional VRAM offload."

    def load(self, model_name: str, model_type: str, device: str,
             offload_to_cpu: bool, dtype: str):

        dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
        torch_dtype = dtype_map.get(dtype, torch.float16)

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Resolve path
        model_path = None
        for folder_key in ("sams", "sam2"):
            if folder_key in folder_paths.folder_names_and_paths:
                try:
                    model_path = folder_paths.get_full_path(folder_key, model_name)
                    if model_path and os.path.exists(model_path):
                        break
                except Exception:
                    continue

        if model_path is None or not os.path.exists(str(model_path)):
            raise FileNotFoundError(
                f"SAM model '{model_name}' not found.  Place it in ComfyUI/models/sams/ or models/sam2/"
            )

        # Detect model type from filename if auto
        detected_type = model_type
        if model_type == "auto":
            name_lower = model_name.lower()
            if "sam3" in name_lower:
                detected_type = "sam3"
            elif "sam2.1" in name_lower or "sam2_1" in name_lower:
                detected_type = "sam2.1"
            elif "sam2" in name_lower:
                detected_type = "sam2"
            elif "vit_h" in name_lower:
                detected_type = "sam_vit_h"
            elif "vit_l" in name_lower:
                detected_type = "sam_vit_l"
            elif "vit_b" in name_lower:
                detected_type = "sam_vit_b"
            else:
                detected_type = "sam2"  # default fallback

        sam_model = None

        # ── SAM2 / SAM2.1 loading ──────────────────────────────────────
        if detected_type in ("sam2", "sam2.1"):
            try:
                from sam2.build_sam import build_sam2
                from sam2.sam2_image_predictor import SAM2ImagePredictor

                # Try building from checkpoint
                sam_model = build_sam2(
                    config_file=None,
                    ckpt_path=model_path,
                    device="cpu",
                )
                sam_model = sam_model.to(torch_dtype)
                if not offload_to_cpu:
                    sam_model = sam_model.to(device)
            except ImportError:
                # Fallback: try loading as plain state_dict
                sam_model = self._load_generic(model_path, torch_dtype, device, offload_to_cpu)
            except Exception as e:
                sam_model = self._load_generic(model_path, torch_dtype, device, offload_to_cpu)

        # ── SAM3 loading ───────────────────────────────────────────────
        elif detected_type == "sam3":
            try:
                from sam3.build_sam import build_sam3
                sam_model = build_sam3(ckpt_path=model_path, device="cpu")
                sam_model = sam_model.to(torch_dtype)
                if not offload_to_cpu:
                    sam_model = sam_model.to(device)
            except ImportError:
                sam_model = self._load_generic(model_path, torch_dtype, device, offload_to_cpu)

        # ── Original SAM (vit_h/l/b) ──────────────────────────────────
        else:
            try:
                from segment_anything import sam_model_registry
                arch_map = {
                    "sam_vit_h": "vit_h",
                    "sam_vit_l": "vit_l",
                    "sam_vit_b": "vit_b",
                }
                arch = arch_map.get(detected_type, "vit_h")
                sam_model = sam_model_registry[arch](checkpoint=model_path)
                sam_model = sam_model.to(torch_dtype)
                if not offload_to_cpu:
                    sam_model = sam_model.to(device)
            except ImportError:
                sam_model = self._load_generic(model_path, torch_dtype, device, offload_to_cpu)

        # Package metadata
        result = {
            "model": sam_model,
            "model_type": detected_type,
            "device": device,
            "offload_to_cpu": offload_to_cpu,
            "dtype": torch_dtype,
            "model_path": model_path,
        }

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return (result,)

    @staticmethod
    def _load_generic(path, dtype, device, offload):
        """Fallback loader – just loads the state dict into a wrapper."""
        state = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        return {"state_dict": state, "dtype": dtype, "device": device}
