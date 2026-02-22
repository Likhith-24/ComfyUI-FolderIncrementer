"""
PointsMaskEditor – Unified interactive editor for points AND bounding boxes.

Click = add point (left=positive, right=negative).
Ctrl+Drag = draw bounding box (left=positive, right=negative).
No mode switching needed – both coexist in the same canvas.

Outputs are designed for full compatibility with:
  • Sam2Segmentation (comfyui-segment-anything-2) – positive_coords, negative_coords, bboxes
  • SAM3 – positive_coords, negative_coords, pos_bboxes, neg_bboxes
  • SAMMaskGeneratorMEC (this pack) – points_json, bbox_json
  • BBox pipeline nodes – primary_bbox [x, y, w, h]
  • ViTMatte / SeC – mask, points_json
"""

import torch
import torch.nn.functional as F
import numpy as np
import json


class PointsMaskEditor:
    """Unified points + bounding-box editor with sub-pixel accuracy.

    - Left-click  → positive point (label=1)
    - Right-click → negative point (label=0)
    - Ctrl+Left-drag  → positive bounding box (green)
    - Ctrl+Right-drag → negative bounding box (red)
    - Scroll      → adjust point radius
    - Ctrl+Scroll → zoom
    - Middle-drag → pan
    - Shift+click → delete element
    - Delete      → remove hovered element
    - Ctrl+Z/Y    → undo / redo

    Outputs are designed to be directly compatible with SAM2, SAM2.1,
    SAM3, SeC, and ViTMatte pipelines.
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
                        "bboxes: [[x1, y1, x2, y2, label], ...]  label: 1=positive, 0=negative\n"
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

    RETURN_TYPES = ("MASK", "STRING", "STRING", "BBOX", "BBOX", "STRING", "STRING", "BBOX",)
    RETURN_NAMES = ("mask", "positive_coords", "negative_coords", "bboxes", "neg_bboxes",
                    "points_json", "bbox_json", "primary_bbox",)
    FUNCTION = "generate"
    CATEGORY = "MaskEditControl/Editor"
    DESCRIPTION = (
        "Unified points + bbox editor. Click for points, Ctrl+drag for boxes – "
        "no mode switching. Outputs feed directly into SAM2/SAM2.1/SAM3/SeC/ViTMatte.\n\n"
        "positive_coords / negative_coords → Sam2Segmentation coordinates_positive / coordinates_negative\n"
        "bboxes → Sam2Segmentation / SAM2.1 / SeC bboxes input (positive only)\n"
        "neg_bboxes → SAM3 negative bounding boxes\n"
        "points_json / bbox_json → SAMMaskGeneratorMEC\n"
        "primary_bbox → BBox pipeline nodes [x, y, w, h]"
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
        bboxes_raw = data.get("bboxes", [])

        # Backward compat: if editor_data is a plain list, treat as points
        if isinstance(data, list):
            points = data
            bboxes_raw = []

        # ── Separate positive / negative points ───────────────────────
        pos_points = []
        neg_points = []
        for pt in points:
            coord = {"x": int(round(float(pt.get("x", 0)))),
                     "y": int(round(float(pt.get("y", 0))))}
            if int(pt.get("label", 1)) == 1:
                pos_points.append(coord)
            else:
                neg_points.append(coord)

        # ── Separate positive / negative bboxes ───────────────────────
        pos_bboxes = []  # [[x1,y1,x2,y2], ...]
        neg_bboxes = []  # [[x1,y1,x2,y2], ...]
        for box in bboxes_raw:
            if len(box) >= 4:
                coords = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
                label = int(box[4]) if len(box) >= 5 else 1
                if label == 1:
                    pos_bboxes.append(coords)
                else:
                    neg_bboxes.append(coords)

        # ── Build mask ─────────────────────────────────────────────────
        device = "cpu"
        if existing_mask is not None:
            device = existing_mask.device
            mask = existing_mask.clone()
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
        else:
            mask = torch.zeros(1, height, width, dtype=torch.float32, device=device)

        # Render points onto mask
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

        # Render bboxes onto mask (positive add, negative subtract)
        for coords in pos_bboxes:
            x1, y1, x2, y2 = coords
            x1, y1 = max(0, min(x1, width)), max(0, min(y1, height))
            x2, y2 = max(0, min(x2, width)), max(0, min(y2, height))
            if x2 > x1 and y2 > y1:
                mask[0, y1:y2, x1:x2] = 1.0

        for coords in neg_bboxes:
            x1, y1, x2, y2 = coords
            x1, y1 = max(0, min(x1, width)), max(0, min(y1, height))
            x2, y2 = max(0, min(x2, width)), max(0, min(y2, height))
            if x2 > x1 and y2 > y1:
                mask[0, y1:y2, x1:x2] = 0.0

        if normalize:
            mask = mask.clamp(0.0, 1.0)

        # ── Build outputs ──────────────────────────────────────────────

        # 1. positive_coords / negative_coords — STRING for Sam2Segmentation
        #    Format: [{"x": int, "y": int}, ...]
        #    IMPORTANT: output None (not '[]') when empty, so downstream
        #    `if coordinates is not None:` checks work correctly.
        positive_coords = json.dumps(pos_points) if pos_points else None
        negative_coords = json.dumps(neg_points) if neg_points else None

        # 2. bboxes — BBOX for Sam2Segmentation / SAM2.1 / SeC
        #    Sam2Segmentation expects: for bbox_list in bboxes → for bbox in bbox_list
        #    So we pass the list of bbox coordinate-lists directly.
        #    IMPORTANT: pass None (not []) when empty, so the
        #    `if bboxes is not None:` check in Sam2Segmentation is skipped.
        bboxes_out = pos_bboxes if pos_bboxes else None

        # 3. neg_bboxes — BBOX for SAM3 negative bounding boxes
        neg_bboxes_out = neg_bboxes if neg_bboxes else None

        # 4. points_json — STRING unified format for SAMMaskGeneratorMEC
        #    Format: [{x, y, label, radius}, ...]
        points_json_out = json.dumps(points)

        # 5. bbox_json — STRING first positive bbox as [x1,y1,x2,y2] for SAMMaskGeneratorMEC
        bbox_json_out = json.dumps(pos_bboxes[0] if pos_bboxes else [])

        # 6. primary_bbox — BBOX [x, y, w, h] for BBox pipeline / legacy nodes
        if pos_bboxes:
            b = pos_bboxes[0]
            primary_bbox = [b[0], b[1], b[2] - b[0], b[3] - b[1]]
        else:
            primary_bbox = [0, 0, width, height]

        return (mask, positive_coords, negative_coords,
                bboxes_out, neg_bboxes_out,
                points_json_out, bbox_json_out, primary_bbox)
