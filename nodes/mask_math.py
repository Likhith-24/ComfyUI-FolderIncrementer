"""
MaskMath – Pixel-level arithmetic on masks: add, multiply, power, invert,
clamp, remap range, quantize, and threshold with hysteresis.
"""

import torch


class MaskMath:
    """Perform mathematical operations on masks for fine-grained control."""

    OPERATIONS = [
        "add_scalar", "multiply_scalar", "power", "invert",
        "clamp", "remap_range", "quantize", "threshold_hysteresis",
        "gamma", "contrast", "abs_diff_from_value",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "operation": (cls.OPERATIONS, {"default": "invert"}),
                "value_a": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 0.01,
                                       "tooltip": "Primary parameter (meaning depends on operation)"}),
                "value_b": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01,
                                       "tooltip": "Secondary parameter (meaning depends on operation)"}),
            },
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "compute"
    CATEGORY = "MaskEditControl/Math"
    DESCRIPTION = "Pixel-level math operations on masks: arithmetic, gamma, contrast, remap, quantize."

    def compute(self, mask, operation, value_a, value_b):
        m = mask.clone()

        if operation == "add_scalar":
            m = m + value_a
        elif operation == "multiply_scalar":
            m = m * value_a
        elif operation == "power":
            m = m.clamp(0, 1).pow(max(0.01, value_a))
        elif operation == "invert":
            m = 1.0 - m
        elif operation == "clamp":
            m = m.clamp(value_a, value_b)
        elif operation == "remap_range":
            # Remap [value_a, value_b] → [0, 1]
            low, high = min(value_a, value_b), max(value_a, value_b)
            if high - low > 1e-6:
                m = (m - low) / (high - low)
            m = m.clamp(0, 1)
        elif operation == "quantize":
            levels = max(2, int(value_a))
            m = (m * (levels - 1)).round() / (levels - 1)
        elif operation == "threshold_hysteresis":
            # value_a = low threshold, value_b = high threshold
            high_mask = (m >= value_b).float()
            low_mask = (m >= value_a).float()
            # Simple connected-component-free hysteresis: dilate high into low
            import torch.nn.functional as F
            kernel = torch.ones(1, 1, 3, 3, device=m.device)
            expanded = high_mask
            for _ in range(int(max(m.shape[-2], m.shape[-1]) * 0.1)):
                if expanded.dim() == 2:
                    expanded = expanded.unsqueeze(0).unsqueeze(0)
                elif expanded.dim() == 3:
                    expanded = expanded.unsqueeze(1)
                expanded = F.conv2d(F.pad(expanded, (1,1,1,1), mode="constant", value=0),
                                     kernel, padding=0)
                expanded = (expanded > 0.5).float()
                if mask.dim() == 2:
                    expanded = expanded.squeeze(0).squeeze(0)
                else:
                    expanded = expanded.squeeze(1)
                expanded = expanded * low_mask
                if torch.equal(expanded, high_mask):
                    break
                high_mask = expanded
            m = expanded
        elif operation == "gamma":
            gamma = max(0.01, value_a)
            m = m.clamp(0, 1).pow(1.0 / gamma)
        elif operation == "contrast":
            # value_a = contrast factor, value_b = midpoint
            m = (m - value_b) * value_a + value_b
        elif operation == "abs_diff_from_value":
            m = torch.abs(m - value_a)

        return (m.clamp(0, 1),)
