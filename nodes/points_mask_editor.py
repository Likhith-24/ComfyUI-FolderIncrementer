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
import base64
from io import BytesIO
from PIL import Image as PILImage


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

    # ══════════════════════════════════════════════════════════════════
    #  Private Helpers
    # ══════════════════════════════════════════════════════════════════

    @staticmethod
    def _parse_editor_data(editor_data: str) -> tuple[list, list, list, list]:
        """Parse JSON editor data into separated point and bbox lists.

        Args:
            editor_data: JSON string from the canvas widget.

        Returns:
            Tuple of (pos_points, neg_points, pos_bboxes, neg_bboxes).
            Each point is {"x": float, "y": float}.
            Each bbox is [x1, y1, x2, y2] ints.
        """
        try:
            data = json.loads(editor_data) if isinstance(editor_data, str) else editor_data
        except json.JSONDecodeError:
            data = {}

        # Backward compat: if editor_data is a plain list, treat as points
        if isinstance(data, list):
            points_raw = data
            bboxes_raw = []
        else:
            points_raw = data.get("points", [])
            bboxes_raw = data.get("bboxes", [])

        pos_points: list[dict] = []
        neg_points: list[dict] = []
        for pt in points_raw:
            coord = {"x": round(float(pt.get("x", 0)), 2),
                     "y": round(float(pt.get("y", 0)), 2)}
            if int(pt.get("label", 1)) == 1:
                pos_points.append(coord)
            else:
                neg_points.append(coord)

        pos_bboxes: list[list[int]] = []
        neg_bboxes: list[list[int]] = []
        for box in bboxes_raw:
            if len(box) >= 4:
                coords = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
                label = int(box[4]) if len(box) >= 5 else 1
                if label == 1:
                    pos_bboxes.append(coords)
                else:
                    neg_bboxes.append(coords)

        return pos_points, neg_points, pos_bboxes, neg_bboxes

    @staticmethod
    def _render_point_brush(
        mask: torch.Tensor,
        points: list,
        default_radius: float,
        softness: float,
        height: int,
        width: int,
        device: str,
    ) -> torch.Tensor:
        """Render point brushes onto the mask tensor.

        Positive points (label=1) add to mask via max.
        Negative points (label=0) subtract via multiply.

        Args:
            mask: [1, H, W] float32 tensor.
            points: List of point dicts {x, y, label, radius}.
            default_radius: Fallback radius.
            softness: Gaussian sigma multiplier (0 = hard circle).
            height: Canvas height.
            width: Canvas width.
            device: Torch device.

        Returns:
            Modified [1, H, W] mask tensor.
        """
        if not points:
            return mask

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

        return mask

    @staticmethod
    def _render_bbox_region(
        mask: torch.Tensor,
        pos_bboxes: list,
        neg_bboxes: list,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Fill/clear rectangular bbox regions on the mask.

        Positive bboxes set pixels to 1.0.
        Negative bboxes set pixels to 0.0.

        Args:
            mask: [1, H, W] float32 tensor.
            pos_bboxes: List of [x1, y1, x2, y2] positive boxes.
            neg_bboxes: List of [x1, y1, x2, y2] negative boxes.
            height: Canvas height.
            width: Canvas width.

        Returns:
            Modified [1, H, W] mask tensor.
        """
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

        return mask

    @staticmethod
    def _encode_reference_image(
        reference_image: torch.Tensor,
        width: int,
        height: int,
        max_preview: int = 1536,
    ) -> dict | None:
        """Encode reference image as base64 JPEG for frontend display.

        Args:
            reference_image: [B, H, W, C] float32 tensor.
            width: Original image width.
            height: Original image height.
            max_preview: Max pixel dimension for preview.

        Returns:
            Dict with "bg_image", "bg_image_width", "bg_image_height"
            keys for the UI payload, or None on failure.
        """
        try:
            img_np = (reference_image[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            pil_img = PILImage.fromarray(img_np)
            if max(pil_img.size) > max_preview:
                ratio = max_preview / max(pil_img.size)
                new_size = (int(pil_img.width * ratio), int(pil_img.height * ratio))
                pil_img = pil_img.resize(new_size, PILImage.LANCZOS)
            buf = BytesIO()
            pil_img.save(buf, format='JPEG', quality=85)
            bg_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            return {
                "bg_image": [bg_b64],
                "bg_image_width": [width],
                "bg_image_height": [height],
            }
        except Exception as e:
            print(f"[MEC] Warning: failed to encode reference image for frontend: {e}")
            return None

    # ══════════════════════════════════════════════════════════════════
    #  Main Entry Point
    # ══════════════════════════════════════════════════════════════════

    def generate(self, width: int, height: int, editor_data: str,
                 default_radius: float, softness: float, normalize: bool,
                 reference_image=None, existing_mask=None):

        # Resolve dimensions from image
        if reference_image is not None:
            _, h, w, _ = reference_image.shape
            height, width = int(h), int(w)

        # Parse editor data into separated lists
        pos_points, neg_points, pos_bboxes, neg_bboxes = self._parse_editor_data(editor_data)

        # Reconstruct combined points list for rendering
        all_points_raw = []
        try:
            data = json.loads(editor_data) if isinstance(editor_data, str) else editor_data
            if isinstance(data, list):
                all_points_raw = data
            else:
                all_points_raw = data.get("points", [])
        except json.JSONDecodeError:
            pass

        # ── Build mask ─────────────────────────────────────────────────
        device = "cpu"
        if existing_mask is not None:
            device = existing_mask.device
            mask = existing_mask.clone()
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
        else:
            mask = torch.zeros(1, height, width, dtype=torch.float32, device=device)

        # Render points and bboxes via private helpers
        mask = self._render_point_brush(mask, all_points_raw, default_radius, softness, height, width, device)
        mask = self._render_bbox_region(mask, pos_bboxes, neg_bboxes, height, width)

        if normalize:
            mask = mask.clamp(0.0, 1.0)

        # ── Build named outputs ────────────────────────────────────────

        # positive_coords / negative_coords — STRING for Sam2Segmentation
        positive_coords = json.dumps(pos_points) if pos_points else "[]"
        negative_coords = json.dumps(neg_points) if neg_points else "[]"

        # bboxes — BBOX for Sam2Segmentation / SAM2.1 / SeC
        bboxes_out = [pos_bboxes] if pos_bboxes else None

        # neg_bboxes — BBOX for SAM3 negative bounding boxes
        neg_bboxes_out = [neg_bboxes] if neg_bboxes else None

        # points_json — STRING unified format for SAMMaskGeneratorMEC
        points_json_out = json.dumps(all_points_raw)

        # bbox_json — STRING first positive bbox as [x1,y1,x2,y2]
        bbox_json_out = json.dumps(pos_bboxes[0] if pos_bboxes else [])

        # primary_bbox — BBOX [x, y, w, h] for BBox pipeline / legacy nodes
        if pos_bboxes:
            b = pos_bboxes[0]
            primary_bbox = [b[0], b[1], b[2] - b[0], b[3] - b[1]]
        else:
            primary_bbox = [0, 0, width, height]

        result = (mask, positive_coords, negative_coords,
                  bboxes_out, neg_bboxes_out,
                  points_json_out, bbox_json_out, primary_bbox)

        # ── Send reference image to frontend for display ──────────────
        if reference_image is not None:
            ui_data = self._encode_reference_image(reference_image, width, height)
            if ui_data is not None:
                return {"ui": ui_data, "result": result}

        return result
