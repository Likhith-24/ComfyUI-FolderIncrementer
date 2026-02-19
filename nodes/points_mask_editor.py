"""
PointsMaskEditor – Unified interactive editor for points AND bounding boxes.

Click = add point (left=positive, right=negative).
Drag  = draw bounding box (auto-detected when mouse moves >5px).
No mode switching needed – both coexist in the same canvas.

Outputs points_json, bbox_json, and a direct mask from the points.
Designed to feed directly into SAM2, SAM3, and ViTMatte nodes.
Also compatible with SeC (Comfyui-SecNodes) via standard MASK / point formats.
"""

import torch
import torch.nn.functional as F
import numpy as np
import json


class PointsMaskEditor:
    """Unified points + bounding-box editor with sub-pixel accuracy.

    - Left-click  → positive point (label=1)
    - Right-click → negative point (label=0)
    - Left-drag   → bounding box (auto-detected by movement)
    - Scroll      → adjust point radius
    - Ctrl+Scroll → zoom
    - Middle-drag → pan
    - Shift+click → delete point
    - Delete      → remove hovered point
    - Ctrl+Z/Y    → undo / redo
    - C           → clear all

    Outputs are designed to be directly compatible with SAM2, SAM3,
    and ViTMatte pipelines.  Also compatible with SeC (Comfyui-SecNodes)
    as they share standard point coordinate and MASK formats.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 1, "max": 16384}),
                "height": ("INT", {"default": 512, "min": 1, "max": 16384}),
                "editor_data": ("STRING", {
                    "default": '{"points":[],"bboxes":[]}',
                    "multiline": True,
                    "tooltip": (
                        "JSON from the interactive editor. Contains both points and bboxes.\n"
                        "points: [{x, y, label, radius}, ...]\n"
                        "bboxes: [[x1, y1, x2, y2], ...]\n"
                        "Automatically populated by the canvas widget."
                    ),
                }),
                "default_radius": ("FLOAT", {"default": 3.0, "min": 0.5, "max": 256.0, "step": 0.5,
                                              "tooltip": "Default brush radius for points"}),
                "softness": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1,
                                        "tooltip": "Gaussian sigma multiplier (0 = hard circle)"}),
                "normalize": ("BOOLEAN", {"default": True,
                                           "tooltip": "Clamp output to [0,1]"}),
            },
            "optional": {
                "reference_image": ("IMAGE", {"tooltip": "If provided, width/height are inferred from this image"}),
                "existing_mask": ("MASK", {"tooltip": "Existing mask to layer points onto"}),
            },
        }

    RETURN_TYPES = ("MASK", "STRING", "STRING", "BBOX",)
    RETURN_NAMES = ("mask", "points_json", "bbox_json", "primary_bbox",)
    FUNCTION = "generate"
    CATEGORY = "MaskEditControl/Editor"
    DESCRIPTION = (
        "Unified points + bbox editor. Click for points, drag for boxes – "
        "no mode switching. Outputs feed directly into SAM2/SAM3/ViTMatte."
    )

    def generate(self, width: int, height: int, editor_data: str,
                 default_radius: float, softness: float, normalize: bool,
                 reference_image=None, existing_mask=None):

        # Resolve dimensions from image
        if reference_image is not None:
            _, h, w, _ = reference_image.shape
            height, width = int(h), int(w)

        # Parse unified editor data
        try:
            data = json.loads(editor_data) if isinstance(editor_data, str) else editor_data
        except json.JSONDecodeError:
            data = {}

        points = data.get("points", [])
        bboxes = data.get("bboxes", [])

        # Backward compat: if editor_data is a plain list, treat as points
        if isinstance(data, list):
            points = data
            bboxes = []

        device = "cpu"
        if existing_mask is not None:
            device = existing_mask.device
            mask = existing_mask.clone()
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
        else:
            mask = torch.zeros(1, height, width, dtype=torch.float32, device=device)

        # ── Render points onto mask ────────────────────────────────────
        if points:
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
                    brush[dist_sq > (3.0 * sigma) ** 2] = 0.0
                else:
                    dist_sq = (xx - px) ** 2 + (yy - py) ** 2
                    brush = (dist_sq <= radius ** 2).float()

                if label == 1:
                    mask[0] = torch.max(mask[0], brush)
                else:
                    mask[0] = mask[0] * (1.0 - brush)

        # ── Render bboxes onto mask (filled rectangles) ────────────────
        for box in bboxes:
            if len(box) >= 4:
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                x1, y1 = max(0, min(x1, width)), max(0, min(y1, height))
                x2, y2 = max(0, min(x2, width)), max(0, min(y2, height))
                if x2 > x1 and y2 > y1:
                    mask[0, y1:y2, x1:x2] = 1.0

        if normalize:
            mask = mask.clamp(0.0, 1.0)

        # ── Build outputs ──────────────────────────────────────────────
        points_json_out = json.dumps(points)

        # First bbox as primary_bbox in [x, y, w, h] format for BBOX type
        if bboxes and len(bboxes[0]) >= 4:
            b = bboxes[0]
            primary_bbox = [int(b[0]), int(b[1]),
                            int(b[2]) - int(b[0]), int(b[3]) - int(b[1])]
        else:
            primary_bbox = [0, 0, width, height]

        # bbox_json: first bbox as [x1, y1, x2, y2] for SAM/SeC
        bbox_json_out = json.dumps(bboxes[0] if bboxes else [])

        return (mask, points_json_out, bbox_json_out, primary_bbox)
