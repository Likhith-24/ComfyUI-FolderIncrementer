"""
BBoxNode – Create, manipulate, and convert bounding boxes for mask/SAM workflows.
Supports manual entry, extraction from masks, padding, clamping, and
conversion to/from masks.
"""

import torch
import json


class BBoxCreate:
    """Manually define a bounding box (x, y, w, h)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "x": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 1}),
                "y": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 1}),
                "width": ("INT", {"default": 128, "min": 1, "max": 16384, "step": 1}),
                "height": ("INT", {"default": 128, "min": 1, "max": 16384, "step": 1}),
            },
        }

    RETURN_TYPES = ("BBOX", "STRING",)
    RETURN_NAMES = ("bbox", "bbox_str",)
    FUNCTION = "create"
    CATEGORY = "MaskEditControl/BBox"
    DESCRIPTION = "Manually create a bounding box [x, y, w, h]."

    def create(self, x, y, width, height):
        bbox = [int(x), int(y), int(width), int(height)]
        return (bbox, json.dumps(bbox))


class BBoxFromMask:
    """Extract the tight bounding box of non-zero pixels in a mask."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "padding": ("INT", {"default": 0, "min": 0, "max": 512, "step": 1,
                                     "tooltip": "Uniform padding around the detected bbox"}),
                "padding_x": ("INT", {"default": 0, "min": 0, "max": 512, "step": 1,
                                       "tooltip": "Extra horizontal padding (added to both sides)"}),
                "padding_y": ("INT", {"default": 0, "min": 0, "max": 512, "step": 1,
                                       "tooltip": "Extra vertical padding (added to both sides)"}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("BBOX", "INT", "INT", "INT", "INT", "STRING",)
    RETURN_NAMES = ("bbox", "x", "y", "width", "height", "bbox_str",)
    FUNCTION = "extract"
    CATEGORY = "MaskEditControl/BBox"
    DESCRIPTION = "Extract tightest bounding box from a mask with optional per-axis padding."

    def extract(self, mask, padding, padding_x, padding_y, threshold):
        m = mask
        if m.dim() == 3:
            m = m[0]
        binary = (m >= threshold).float()
        coords = torch.nonzero(binary, as_tuple=False)

        if coords.shape[0] == 0:
            h, w = m.shape
            bbox = [0, 0, int(w), int(h)]
            return (bbox, 0, 0, int(w), int(h), json.dumps(bbox))

        y_min = int(coords[:, 0].min().item())
        y_max = int(coords[:, 0].max().item())
        x_min = int(coords[:, 1].min().item())
        x_max = int(coords[:, 1].max().item())

        total_px = padding + padding_x
        total_py = padding + padding_y
        h, w = m.shape
        x_min = max(0, x_min - total_px)
        y_min = max(0, y_min - total_py)
        x_max = min(w - 1, x_max + total_px)
        y_max = min(h - 1, y_max + total_py)

        bw = x_max - x_min + 1
        bh = y_max - y_min + 1
        bbox = [x_min, y_min, bw, bh]
        return (bbox, x_min, y_min, bw, bh, json.dumps(bbox))


class BBoxToMask:
    """Convert a bounding box to a filled rectangular mask."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bbox": ("BBOX",),
                "image_width": ("INT", {"default": 512, "min": 1, "max": 16384}),
                "image_height": ("INT", {"default": 512, "min": 1, "max": 16384}),
            },
            "optional": {
                "reference_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "convert"
    CATEGORY = "MaskEditControl/BBox"
    DESCRIPTION = "Convert a bounding box into a rectangular mask."

    def convert(self, bbox, image_width, image_height, reference_image=None):
        if reference_image is not None:
            _, h, w, _ = reference_image.shape
            image_height, image_width = int(h), int(w)
        x, y, bw, bh = bbox
        mask = torch.zeros(1, image_height, image_width, dtype=torch.float32)
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(image_width, x + bw)
        y2 = min(image_height, y + bh)
        mask[0, y1:y2, x1:x2] = 1.0
        return (mask,)


class BBoxPad:
    """Add asymmetric padding to a bounding box with clamping."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bbox": ("BBOX",),
                "pad_left": ("INT", {"default": 0, "min": -4096, "max": 4096}),
                "pad_right": ("INT", {"default": 0, "min": -4096, "max": 4096}),
                "pad_top": ("INT", {"default": 0, "min": -4096, "max": 4096}),
                "pad_bottom": ("INT", {"default": 0, "min": -4096, "max": 4096}),
                "image_width": ("INT", {"default": 512, "min": 1, "max": 16384}),
                "image_height": ("INT", {"default": 512, "min": 1, "max": 16384}),
            },
        }

    RETURN_TYPES = ("BBOX", "STRING",)
    RETURN_NAMES = ("bbox", "bbox_str",)
    FUNCTION = "pad"
    CATEGORY = "MaskEditControl/BBox"
    DESCRIPTION = "Pad a bounding box asymmetrically and clamp to image bounds."

    def pad(self, bbox, pad_left, pad_right, pad_top, pad_bottom, image_width, image_height):
        x, y, bw, bh = bbox
        x1 = max(0, x - pad_left)
        y1 = max(0, y - pad_top)
        x2 = min(image_width, x + bw + pad_right)
        y2 = min(image_height, y + bh + pad_bottom)
        out = [x1, y1, x2 - x1, y2 - y1]
        return (out, json.dumps(out))


class BBoxCrop:
    """Crop an image and its mask to a bounding box region."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "bbox": ("BBOX",),
            },
            "optional": {
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "BBOX",)
    RETURN_NAMES = ("cropped_image", "cropped_mask", "bbox",)
    FUNCTION = "crop"
    CATEGORY = "MaskEditControl/BBox"
    DESCRIPTION = "Crop image (and optionally mask) to the bounding box region."

    def crop(self, image, bbox, mask=None):
        x, y, bw, bh = bbox
        B, H, W, C = image.shape
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(W, x + bw)
        y2 = min(H, y + bh)
        cropped = image[:, y1:y2, x1:x2, :]
        if mask is not None:
            m = mask
            if m.dim() == 2:
                m = m.unsqueeze(0)
            cropped_mask = m[:, y1:y2, x1:x2]
        else:
            cropped_mask = torch.ones(B, y2 - y1, x2 - x1, dtype=torch.float32, device=image.device)
        return (cropped, cropped_mask, [x1, y1, x2 - x1, y2 - y1])
