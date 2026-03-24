"""ImageComparerMEC – interactive before / after image comparison widget."""

import os
import hashlib

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import folder_paths


class ImageComparerMEC:
    """Show two images in an interactive slider / overlay / difference view.

    The frontend widget renders a draggable divider (slider mode),
    adjustable-opacity overlay, or amplified difference heatmap
    directly inside the node.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_a": ("IMAGE", {"tooltip": "Left / base image (B,H,W,C)"}),
                "image_b": ("IMAGE", {"tooltip": "Right / compare image (B,H,W,C)"}),
            },
            "optional": {
                "label_a": ("STRING", {"default": "Before", "tooltip": "Label for image A"}),
                "label_b": ("STRING", {"default": "After", "tooltip": "Label for image B"}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "compare"
    CATEGORY = "MaskEditControl/Preview"
    OUTPUT_NODE = True
    DESCRIPTION = "Interactive image comparer: drag-slider, overlay blend, and difference heatmap."

    def compare(
        self,
        image_a: torch.Tensor,
        image_b: torch.Tensor,
        label_a: str = "Before",
        label_b: str = "After",
    ):
        B, H, W, _C = image_a.shape

        # Match spatial dims if needed
        if image_b.shape[1] != H or image_b.shape[2] != W:
            image_b = F.interpolate(
                image_b.permute(0, 3, 1, 2), size=(H, W),
                mode="bilinear", align_corners=False,
            ).permute(0, 2, 3, 1)

        return {
            "ui": {
                "image_a": [_save_temp(image_a[0], "cmp_a")],
                "image_b": [_save_temp(image_b[0], "cmp_b")],
                "label_a": [label_a],
                "label_b": [label_b],
            }
        }


def _save_temp(tensor: torch.Tensor, prefix: str) -> dict:
    """Save a single (H,W,C) tensor as a temp PNG and return ComfyUI file info."""
    arr = (tensor.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    digest = hashlib.sha256(arr.tobytes()[:8192]).hexdigest()[:10]
    name = f"{prefix}_{digest}.png"
    Image.fromarray(arr).save(
        os.path.join(folder_paths.get_temp_directory(), name),
        compress_level=1,
    )
    return {"filename": name, "subfolder": "", "type": "temp"}
