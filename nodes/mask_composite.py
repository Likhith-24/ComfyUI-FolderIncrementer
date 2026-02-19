"""
MaskCompositeAdvanced – Combine multiple masks with boolean and blending
operations (union, intersect, subtract, XOR, blend, min, max).
"""

import torch
import torch.nn.functional as F


class MaskCompositeAdvanced:
    """Combine two masks using various boolean / blending operations with
    optional per-mask invert and output threshold."""

    OPERATIONS = ["union", "intersect", "subtract", "xor", "blend", "min", "max", "difference"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask_a": ("MASK",),
                "mask_b": ("MASK",),
                "operation": (cls.OPERATIONS, {"default": "union"}),
                "blend_factor": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                                            "tooltip": "Blend ratio (only for 'blend' mode).  0=all A, 1=all B"}),
                "invert_a": ("BOOLEAN", {"default": False}),
                "invert_b": ("BOOLEAN", {"default": False}),
                "threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                         "tooltip": "Binarize output (0 = keep soft)"}),
            },
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "composite"
    CATEGORY = "MaskEditControl/Composite"
    DESCRIPTION = "Combine two masks with union, intersect, subtract, xor, blend, min, max, or difference."

    def composite(self, mask_a, mask_b, operation, blend_factor,
                  invert_a, invert_b, threshold):
        a = mask_a.clone()
        b = mask_b.clone()

        # Match dimensions
        if a.dim() == 2:
            a = a.unsqueeze(0)
        if b.dim() == 2:
            b = b.unsqueeze(0)

        # Match batch sizes
        if a.shape[0] != b.shape[0]:
            target_b = max(a.shape[0], b.shape[0])
            if a.shape[0] < target_b:
                a = a.expand(target_b, -1, -1).clone()
            if b.shape[0] < target_b:
                b = b.expand(target_b, -1, -1).clone()

        # Match spatial dimensions
        if a.shape[1:] != b.shape[1:]:
            b = F.interpolate(b.unsqueeze(1), size=a.shape[1:],
                              mode="bilinear", align_corners=False).squeeze(1)

        if invert_a:
            a = 1.0 - a
        if invert_b:
            b = 1.0 - b

        if operation == "union":
            out = torch.max(a, b)
        elif operation == "intersect":
            out = torch.min(a, b)
        elif operation == "subtract":
            out = (a - b).clamp(0, 1)
        elif operation == "xor":
            out = torch.abs(a - b)
        elif operation == "blend":
            out = a * (1 - blend_factor) + b * blend_factor
        elif operation == "min":
            out = torch.min(a, b)
        elif operation == "max":
            out = torch.max(a, b)
        elif operation == "difference":
            out = torch.abs(a - b)
        else:
            out = torch.max(a, b)

        if threshold > 0:
            out = (out >= threshold).float()

        return (out.clamp(0, 1),)
