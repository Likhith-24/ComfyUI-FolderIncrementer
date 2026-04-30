"""
BackgroundRemoverMEC – One-click background removal using RMBG-2.0 or BiRefNet.

Models supported:
  - **RMBG-2.0** (briaai | ISnet variant) – General-purpose, fast
  - **BiRefNet-General** – Bilateral reference, high-detail edges
  - **BiRefNet-Portrait** – Optimized for human portraits

The node outputs a clean foreground (RGB pre-multiplied) and a high-quality
alpha MASK. Ideal as a preprocessing step before compositing or inpainting.
"""

from __future__ import annotations

import gc
import json
import logging

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image as PILImage

from .model_manager import (
    MODEL_REGISTRY,
    get_or_load_model,
    clear_cache,
)

logger = logging.getLogger("MEC")


class BackgroundRemoverMEC:
    """Remove background from images using RMBG or BiRefNet models."""

    @classmethod
    def INPUT_TYPES(cls):
        models = []
        for name, reg in sorted(MODEL_REGISTRY.items()):
            if reg.get("family") in ("rmbg", "birefnet"):
                models.append(name)
        if not models:
            models = ["rmbg_2.0"]

        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Input image(s) to remove background from.",
                }),
                "model_name": (models, {
                    "default": models[0],
                    "tooltip": (
                        "Background removal model:\n"
                        "rmbg_2.0: General-purpose, fast.\n"
                        "birefnet_general: High-detail edges.\n"
                        "birefnet_portrait: Optimized for human portraits."
                    ),
                }),
                "threshold": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Alpha threshold for hard mask (0=soft, 0.5=balanced, 1=hard).",
                }),
                "invert": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Invert the alpha mask (keep background instead).",
                }),
                "mask_blur": ("INT", {
                    "default": 0, "min": 0, "max": 50, "step": 1,
                    "tooltip": "Gaussian blur applied to final mask edges.",
                }),
            },
            "optional": {
                "keep_model_loaded": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Keep model in VRAM between runs.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("foreground", "mask", "info")
    FUNCTION = "remove_bg"
    CATEGORY = "MaskEditControl/Matting"
    DESCRIPTION = (
        "One-click background removal using RMBG-2.0 or BiRefNet.\n"
        "Outputs foreground (premultiplied RGB) and high-quality alpha mask.\n"
        "Ideal for portraits, product photos, and compositing workflows."
    )

    def remove_bg(
        self,
        image: torch.Tensor,
        model_name: str,
        threshold: float,
        invert: bool,
        mask_blur: int,
        keep_model_loaded: bool = True,
    ):
        B, H, W, C = image.shape
        # MANUAL bug-fix (Apr 2026): full device autodetect (cuda > mps > cpu)
        # so Apple-Silicon and AMD-ROCm users get GPU acceleration too.
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        loaded = get_or_load_model(model_name, precision="fp32", device=device)
        model = loaded["model"]
        processor = loaded.get("processor")
        dev = next(model.parameters()).device

        masks = []

        for i in range(B):
            img_np = (image[i].cpu().numpy() * 255).astype(np.uint8)
            pil_img = PILImage.fromarray(img_np[:, :, :3])

            if processor is not None:
                inputs = processor(images=pil_img, return_tensors="pt")
                inputs = {k: v.to(dev) for k, v in inputs.items()}
                input_t = None
            else:
                from torchvision import transforms
                transform = transforms.Compose([
                    transforms.Resize((1024, 1024)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])
                input_t = transform(pil_img).unsqueeze(0).to(dev)
                inputs = None

            # MANUAL bug-fix (Apr 2026): BiRefNet's custom forward(x) does not
            # accept the keyword `pixel_values`; call positionally when the
            # processor was unavailable (BiRefNet path).  Also cast input dtype
            # to match the loaded model parameters (fp16 BiRefNet variant).
            with torch.no_grad():
                if inputs is not None:
                    out = model(**inputs)
                else:
                    p_dtype = next(model.parameters()).dtype
                    if input_t.dtype != p_dtype:
                        input_t = input_t.to(p_dtype)
                    try:
                        out = model(input_t)
                    except TypeError:
                        out = model(pixel_values=input_t)

            # Extract alpha map from model output (MANUAL bug-fix Apr 2026:
            # use single getattr instead of fragile elif-chain that broke on
            # diffusers Output dataclasses without .logits but with __getitem__).
            if hasattr(out, "logits"):
                logits = out.logits
            elif isinstance(out, torch.Tensor):
                logits = out
            elif isinstance(out, (list, tuple)) and len(out) > 0:
                logits = out[-1]
            else:
                # last-resort: index access for HF ModelOutput-style mappings
                try:
                    logits = out[0]
                except (KeyError, IndexError, TypeError) as exc:
                    raise RuntimeError(
                        f"BackgroundRemoverMEC: cannot extract alpha logits from "
                        f"model output of type {type(out).__name__}: {exc}"
                    )

            alpha = torch.sigmoid(logits[0, 0]).cpu()

            # Resize to original dimensions
            if alpha.shape[0] != H or alpha.shape[1] != W:
                alpha = F.interpolate(
                    alpha.unsqueeze(0).unsqueeze(0),
                    (H, W),
                    mode="bilinear",
                    align_corners=False,
                )[0, 0]

            masks.append(alpha)

        alpha_mask = torch.stack(masks).clamp(0.0, 1.0)

        # Apply threshold if not fully soft
        if threshold > 0:
            alpha_mask = (alpha_mask > threshold).float()

        # Invert
        if invert:
            alpha_mask = 1.0 - alpha_mask

        # Blur
        if mask_blur > 0:
            try:
                import cv2
                blurred = []
                for i in range(B):
                    a_np = alpha_mask[i].numpy()
                    ksize = mask_blur * 2 + 1
                    a_np = cv2.GaussianBlur(a_np, (ksize, ksize), 0)
                    blurred.append(torch.from_numpy(a_np))
                alpha_mask = torch.stack(blurred)
            except ImportError:
                pass

        # Pre-multiplied foreground
        fg = image[:B].clone()
        fg = fg * alpha_mask.unsqueeze(-1)

        if not keep_model_loaded:
            clear_cache()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        info = json.dumps({
            "model": model_name,
            "frames": B,
            "threshold": threshold,
            "invert": invert,
            "mask_blur": mask_blur,
        }, indent=2)

        return (fg, alpha_mask, info)
