"""
MaskDrawFrame – Draw shapes (circle, rect, ellipse, polygon) onto a mask
for a specific frame in a sequence.  Great for manually creating masks.
"""

import torch
import numpy as np
import json


class MaskDrawFrame:
    """Draw geometric shapes onto a mask at specific coordinates.
    Useful for creating masks from scratch or adding to existing ones."""

    SHAPES = ["circle", "rectangle", "ellipse", "polygon", "line"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 1, "max": 16384}),
                "height": ("INT", {"default": 512, "min": 1, "max": 16384}),
                "shape": (cls.SHAPES, {"default": "circle"}),
                "shape_params_json": ("STRING", {
                    "default": '{"cx": 256, "cy": 256, "radius": 50}',
                    "multiline": True,
                    "tooltip": (
                        "Shape parameters as JSON.\n"
                        "circle: {cx, cy, radius}\n"
                        "rectangle: {x, y, w, h}\n"
                        "ellipse: {cx, cy, rx, ry, angle}\n"
                        "polygon: {points: [[x1,y1],[x2,y2],...]}\n"
                        "line: {x1, y1, x2, y2, thickness}"
                    ),
                }),
                "value": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                     "tooltip": "Fill value for the drawn shape"}),
                "feather": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 128.0, "step": 0.5,
                                       "tooltip": "Soft edge feathering"}),
                "operation": (["set", "add", "subtract", "max", "min"], {"default": "set",
                               "tooltip": "How to combine with existing mask"}),
            },
            "optional": {
                "existing_mask": ("MASK",),
                "reference_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "draw"
    CATEGORY = "MaskEditControl/Draw"
    DESCRIPTION = "Draw precise geometric shapes onto a mask with feathering and blend operations."

    def draw(self, width, height, shape, shape_params_json, value, feather,
             operation, existing_mask=None, reference_image=None):

        if reference_image is not None:
            _, h, w, _ = reference_image.shape
            height, width = int(h), int(w)

        try:
            params = json.loads(shape_params_json) if isinstance(shape_params_json, str) else shape_params_json
        except json.JSONDecodeError:
            params = {}

        # Create drawing canvas
        canvas = torch.zeros(height, width, dtype=torch.float32)

        yy = torch.arange(height, dtype=torch.float32).unsqueeze(1).expand(height, width)
        xx = torch.arange(width, dtype=torch.float32).unsqueeze(0).expand(height, width)

        if shape == "circle":
            cx = float(params.get("cx", width // 2))
            cy = float(params.get("cy", height // 2))
            r = float(params.get("radius", 50))
            dist = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            if feather > 0:
                canvas = (1.0 - ((dist - r) / feather).clamp(0, 1)) * value
                canvas[dist <= r] = value
            else:
                canvas[dist <= r] = value

        elif shape == "rectangle":
            x = float(params.get("x", 0))
            y = float(params.get("y", 0))
            w = float(params.get("w", 100))
            h = float(params.get("h", 100))
            if feather > 0:
                dx = torch.max(x - xx, xx - (x + w)).clamp(min=0)
                dy = torch.max(y - yy, yy - (y + h)).clamp(min=0)
                dist = torch.sqrt(dx ** 2 + dy ** 2)
                canvas = (1.0 - (dist / feather).clamp(0, 1)) * value
            else:
                mask_region = (xx >= x) & (xx <= x + w) & (yy >= y) & (yy <= y + h)
                canvas[mask_region] = value

        elif shape == "ellipse":
            cx = float(params.get("cx", width // 2))
            cy = float(params.get("cy", height // 2))
            rx = float(params.get("rx", 100))
            ry = float(params.get("ry", 50))
            angle = float(params.get("angle", 0)) * np.pi / 180.0
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            dx = xx - cx
            dy = yy - cy
            rx_ = (dx * cos_a + dy * sin_a) / max(rx, 1e-6)
            ry_ = (-dx * sin_a + dy * cos_a) / max(ry, 1e-6)
            dist = torch.sqrt(rx_ ** 2 + ry_ ** 2)
            if feather > 0:
                canvas = (1.0 - ((dist - 1.0) * min(rx, ry) / feather).clamp(0, 1)) * value
                canvas[dist <= 1.0] = value
            else:
                canvas[dist <= 1.0] = value

        elif shape == "polygon":
            pts = params.get("points", [])
            if pts and len(pts) >= 3:
                try:
                    import cv2
                    pts_np = np.array(pts, dtype=np.int32)
                    mask_np = np.zeros((height, width), dtype=np.float32)
                    cv2.fillPoly(mask_np, [pts_np], float(value))
                    if feather > 0:
                        kernel_size = int(feather * 4) | 1
                        mask_np = cv2.GaussianBlur(mask_np, (kernel_size, kernel_size), feather)
                    canvas = torch.from_numpy(mask_np)
                except ImportError:
                    # Simple scanline fallback
                    canvas = self._fill_polygon_torch(pts, height, width, value)

        elif shape == "line":
            x1 = float(params.get("x1", 0))
            y1 = float(params.get("y1", 0))
            x2 = float(params.get("x2", width))
            y2 = float(params.get("y2", height))
            thickness = float(params.get("thickness", 3))
            # Distance from point to line segment
            line_len = max(1e-6, np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
            dx = (x2 - x1) / line_len
            dy = (y2 - y1) / line_len
            # Project
            t = ((xx - x1) * dx + (yy - y1) * dy).clamp(0, line_len)
            proj_x = x1 + t * dx
            proj_y = y1 + t * dy
            dist = torch.sqrt((xx - proj_x) ** 2 + (yy - proj_y) ** 2)
            half_t = thickness / 2
            if feather > 0:
                canvas = (1.0 - ((dist - half_t) / feather).clamp(0, 1)) * value
                canvas[dist <= half_t] = value
            else:
                canvas[dist <= half_t] = value

        # Combine with existing mask
        if existing_mask is not None:
            base = existing_mask.clone()
            if base.dim() == 3:
                base = base[0]
            if base.shape != canvas.shape:
                import torch.nn.functional as Fn
                canvas = Fn.interpolate(
                    canvas.unsqueeze(0).unsqueeze(0),
                    size=base.shape, mode="bilinear", align_corners=False
                ).squeeze(0).squeeze(0)

            if operation == "set":
                mask_region = canvas > 0
                base[mask_region] = canvas[mask_region]
            elif operation == "add":
                base = (base + canvas).clamp(0, 1)
            elif operation == "subtract":
                base = (base - canvas).clamp(0, 1)
            elif operation == "max":
                base = torch.max(base, canvas)
            elif operation == "min":
                base = torch.min(base, canvas)
            canvas = base

        return (canvas.unsqueeze(0).clamp(0, 1),)

    @staticmethod
    def _fill_polygon_torch(pts, h, w, value):
        """Simple scanline polygon fill (no cv2 dependency)."""
        canvas = torch.zeros(h, w, dtype=torch.float32)
        n = len(pts)
        for y in range(h):
            nodes = []
            j = n - 1
            for i in range(n):
                yi, xi = float(pts[i][1]), float(pts[i][0])
                yj, xj = float(pts[j][1]), float(pts[j][0])
                if (yi < y <= yj) or (yj < y <= yi):
                    x_intersect = xi + (y - yi) / (yj - yi) * (xj - xi)
                    nodes.append(x_intersect)
                j = i
            nodes.sort()
            for k in range(0, len(nodes) - 1, 2):
                x_start = max(0, int(nodes[k]))
                x_end = min(w, int(nodes[k + 1]) + 1)
                canvas[y, x_start:x_end] = value
        return canvas
