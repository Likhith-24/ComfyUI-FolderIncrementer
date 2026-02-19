"""
PointsMaskEditor – Generate precise masks from click-points with sub-pixel
accuracy.  Supports positive / negative points, variable radius per point,
and Gaussian soft-brush falloff.
"""

import torch
import torch.nn.functional as F
import numpy as np
import json


class PointsMaskEditor:
    """Create masks from a list of (x, y, label, radius) points with
    sub-pixel Gaussian blending.  Points come from the JS widget or
    as a JSON string."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 1, "max": 16384}),
                "height": ("INT", {"default": 512, "min": 1, "max": 16384}),
                "points_json": ("STRING", {
                    "default": "[]",
                    "multiline": True,
                    "tooltip": (
                        'JSON array of points: [{"x":100,"y":200,"label":1,"radius":5}, ...]. '
                        'label=1 positive (add), label=0 negative (subtract). '
                        'Coordinates are in pixel space (float for sub-pixel).'
                    ),
                }),
                "default_radius": ("FLOAT", {"default": 3.0, "min": 0.5, "max": 256.0, "step": 0.5,
                                              "tooltip": "Fallback radius when a point has no radius field"}),
                "softness": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1,
                                        "tooltip": "Gaussian sigma multiplier (0 = hard circle)"}),
                "normalize": ("BOOLEAN", {"default": True,
                                           "tooltip": "Clamp output to [0,1]"}),
            },
            "optional": {
                "reference_image": ("IMAGE", {"tooltip": "If provided, width/height are taken from this image"}),
                "existing_mask": ("MASK", {"tooltip": "Existing mask to combine points onto"}),
            },
        }

    RETURN_TYPES = ("MASK", "STRING",)
    RETURN_NAMES = ("mask", "points_out",)
    FUNCTION = "generate"
    CATEGORY = "MaskEditControl/Points"
    DESCRIPTION = "Generate a precise mask from click-points with sub-pixel Gaussian accuracy."

    def generate(self, width: int, height: int, points_json: str,
                 default_radius: float, softness: float, normalize: bool,
                 reference_image=None, existing_mask=None):

        # resolve dimensions
        if reference_image is not None:
            _, h, w, _ = reference_image.shape
            height, width = int(h), int(w)

        # parse points
        try:
            points = json.loads(points_json) if isinstance(points_json, str) else points_json
        except json.JSONDecodeError:
            points = []

        device = "cpu"
        if existing_mask is not None:
            device = existing_mask.device
            mask = existing_mask.clone()
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
        else:
            mask = torch.zeros(1, height, width, dtype=torch.float32, device=device)

        # coordinate grids (sub-pixel)
        yy = torch.arange(height, dtype=torch.float32, device=device).unsqueeze(1).expand(height, width)
        xx = torch.arange(width, dtype=torch.float32, device=device).unsqueeze(0).expand(height, width)

        for pt in points:
            px = float(pt.get("x", 0))
            py = float(pt.get("y", 0))
            label = int(pt.get("label", 1))
            radius = float(pt.get("radius", default_radius))

            if softness > 0:
                sigma = radius * softness
                dist_sq = (xx - px) ** 2 + (yy - py) ** 2
                brush = torch.exp(-dist_sq / (2.0 * sigma ** 2))
                # zero out beyond 3-sigma for efficiency
                brush[dist_sq > (3.0 * sigma) ** 2] = 0.0
            else:
                dist_sq = (xx - px) ** 2 + (yy - py) ** 2
                brush = (dist_sq <= radius ** 2).float()

            if label == 1:
                mask[0] = torch.max(mask[0], brush)
            else:
                mask[0] = mask[0] * (1.0 - brush)

        if normalize:
            mask = mask.clamp(0.0, 1.0)

        return (mask, json.dumps(points, indent=2))
