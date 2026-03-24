"""
MaskDrawFrame – Draw shapes onto a mask for a specific frame in a sequence.
Supports: circle, rectangle, ellipse, polygon, line, triangle, star, diamond,
cross, rounded_rectangle, heart, arrow.  All shapes support rotation.
"""

import torch
import numpy as np
import json
import math


class MaskDrawFrame:
    """Draw geometric shapes onto a mask at specific coordinates.
    Useful for creating masks from scratch or adding to existing ones."""

    SHAPES = [
        "circle", "rectangle", "ellipse", "polygon", "line",
        "triangle", "star", "diamond", "cross", "rounded_rectangle",
        "heart", "arrow",
    ]

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
                        "line: {x1, y1, x2, y2, thickness}\n"
                        "triangle: {cx, cy, size} or {points: [[x1,y1],[x2,y2],[x3,y3]]}\n"
                        "star: {cx, cy, outer_r, inner_r, num_points}\n"
                        "diamond: {cx, cy, w, h}\n"
                        "cross: {cx, cy, size, thickness}\n"
                        "rounded_rectangle: {x, y, w, h, corner_radius}\n"
                        "heart: {cx, cy, size}\n"
                        "arrow: {cx, cy, length, width, head_length, head_width}"
                    ),
                }),
                "value": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                     "tooltip": "Fill value for the drawn shape"}),
                "feather": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 128.0, "step": 0.5,
                                       "tooltip": "Soft edge feathering"}),
                "rotation": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 0.5,
                                        "tooltip": "Rotation angle in degrees (around shape center)"}),
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
    DESCRIPTION = "Draw precise geometric shapes onto a mask with feathering, rotation, and blend operations."

    # ── Rotation helper ───────────────────────────────────────────────
    @staticmethod
    def _rotate_grid(xx: torch.Tensor, yy: torch.Tensor,
                     cx: float, cy: float, angle_deg: float):
        """Return rotated (xx, yy) grids around (cx, cy) by angle_deg."""
        if abs(angle_deg) < 1e-6:
            return xx, yy
        rad = math.radians(angle_deg)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        dx = xx - cx
        dy = yy - cy
        rx = dx * cos_a + dy * sin_a
        ry = -dx * sin_a + dy * cos_a
        return rx + cx, ry + cy

    # ── Polygon-to-SDF helper (for feathered polygon shapes) ─────────
    @staticmethod
    def _polygon_sdf(pts_list: list, xx: torch.Tensor, yy: torch.Tensor) -> torch.Tensor:
        """Signed distance field for a convex/concave polygon.
        Negative inside, positive outside."""
        n = len(pts_list)
        if n < 3:
            return torch.ones_like(xx) * 1e6

        # Winding number test for inside/outside + min edge distance
        pts_np = np.array(pts_list, dtype=np.float64)
        h, w = xx.shape
        xx_np = xx.numpy().astype(np.float64)
        yy_np = yy.numpy().astype(np.float64)

        min_dist = np.full((h, w), 1e12, dtype=np.float64)
        winding = np.zeros((h, w), dtype=np.float64)

        for i in range(n):
            j = (i + 1) % n
            ex = pts_np[j, 0] - pts_np[i, 0]
            ey = pts_np[j, 1] - pts_np[i, 1]
            wx = xx_np - pts_np[i, 0]
            wy = yy_np - pts_np[i, 1]

            # Closest point on edge
            edge_len_sq = ex * ex + ey * ey
            if edge_len_sq < 1e-12:
                continue
            t = np.clip((wx * ex + wy * ey) / edge_len_sq, 0.0, 1.0)
            dx = wx - t * ex
            dy = wy - t * ey
            d = dx * dx + dy * dy
            min_dist = np.minimum(min_dist, d)

            # Winding number contribution
            cond1 = (pts_np[i, 1] <= yy_np) & (pts_np[j, 1] > yy_np)
            cond2 = (pts_np[j, 1] <= yy_np) & (pts_np[i, 1] > yy_np)
            cross = ex * wy - ey * wx
            winding = np.where(cond1 & (cross > 0), winding + 1, winding)
            winding = np.where(cond2 & (cross < 0), winding - 1, winding)

        min_dist = np.sqrt(min_dist)
        sign = np.where(winding != 0, -1.0, 1.0)
        return torch.from_numpy((sign * min_dist).astype(np.float32))

    # ── Generate regular polygon vertices ─────────────────────────────
    @staticmethod
    def _regular_polygon_pts(cx: float, cy: float, r: float, n: int,
                             start_angle: float = -math.pi / 2) -> list:
        pts = []
        for i in range(n):
            a = start_angle + 2 * math.pi * i / n
            pts.append([cx + r * math.cos(a), cy + r * math.sin(a)])
        return pts

    def draw(self, width, height, shape, shape_params_json, value, feather,
             rotation, operation, existing_mask=None, reference_image=None):

        if reference_image is not None:
            _, h, w, _ = reference_image.shape
            height, width = int(h), int(w)

        try:
            params = json.loads(shape_params_json) if isinstance(shape_params_json, str) else shape_params_json
        except json.JSONDecodeError:
            params = {}

        # ── Normalise params: list → dict based on shape ──────────────
        if isinstance(params, list):
            if shape == "circle" and len(params) >= 3:
                params = {"cx": params[0], "cy": params[1], "radius": params[2]}
            elif shape == "rectangle" and len(params) >= 4:
                params = {"x": params[0], "y": params[1], "w": params[2], "h": params[3]}
            elif shape == "ellipse" and len(params) >= 4:
                params = {"cx": params[0], "cy": params[1], "rx": params[2], "ry": params[3],
                          "angle": params[4] if len(params) >= 5 else 0}
            elif shape in ("polygon", "triangle"):
                params = {"points": params}
            elif shape == "line" and len(params) >= 4:
                params = {"x1": params[0], "y1": params[1], "x2": params[2], "y2": params[3],
                          "thickness": params[4] if len(params) >= 5 else 3}
            else:
                params = {}
        elif not isinstance(params, dict):
            params = {}

        # Create drawing canvas
        canvas = torch.zeros(height, width, dtype=torch.float32)

        yy = torch.arange(height, dtype=torch.float32).unsqueeze(1).expand(height, width)
        xx = torch.arange(width, dtype=torch.float32).unsqueeze(0).expand(height, width)

        if shape == "circle":
            cx = float(params.get("cx", width // 2))
            cy = float(params.get("cy", height // 2))
            r = float(params.get("radius", 50))
            rxx, ryy = self._rotate_grid(xx, yy, cx, cy, rotation)
            dist = torch.sqrt((rxx - cx) ** 2 + (ryy - cy) ** 2)
            if feather > 0:
                canvas = ((1.0 - ((dist - r) / feather).clamp(0, 1)) * value).clamp(0, value)
            else:
                canvas[dist <= r] = value

        elif shape == "rectangle":
            x = float(params.get("x", 0))
            y = float(params.get("y", 0))
            w = float(params.get("w", 100))
            h = float(params.get("h", 100))
            rcx, rcy = x + w / 2, y + h / 2
            rxx, ryy = self._rotate_grid(xx, yy, rcx, rcy, rotation)
            dx = (torch.abs(rxx - rcx) - w / 2).clamp(min=0)
            dy = (torch.abs(ryy - rcy) - h / 2).clamp(min=0)
            dist = torch.sqrt(dx ** 2 + dy ** 2)
            if feather > 0:
                canvas = (1.0 - (dist / feather).clamp(0, 1)) * value
            else:
                inside = (torch.abs(rxx - rcx) <= w / 2) & (torch.abs(ryy - rcy) <= h / 2)
                canvas[inside] = value

        elif shape == "ellipse":
            cx = float(params.get("cx", width // 2))
            cy = float(params.get("cy", height // 2))
            rx = float(params.get("rx", 100))
            ry = float(params.get("ry", 50))
            # Ellipse has its own angle param + global rotation
            angle = float(params.get("angle", 0)) + rotation
            rad = math.radians(angle)
            cos_a, sin_a = math.cos(rad), math.sin(rad)
            dx = xx - cx
            dy = yy - cy
            rx_ = (dx * cos_a + dy * sin_a) / max(rx, 1e-6)
            ry_ = (-dx * sin_a + dy * cos_a) / max(ry, 1e-6)
            dist = torch.sqrt(rx_ ** 2 + ry_ ** 2)
            if feather > 0:
                canvas = torch.where(dist <= 1.0,
                                     torch.tensor(value),
                                     ((1.0 - ((dist - 1.0) * min(rx, ry) / feather).clamp(0, 1)) * value))
            else:
                canvas[dist <= 1.0] = value

        elif shape == "polygon":
            pts = params.get("points", [])
            if pts and len(pts) >= 3:
                cx = sum(p[0] for p in pts) / len(pts)
                cy = sum(p[1] for p in pts) / len(pts)
                if abs(rotation) > 1e-6:
                    rad = math.radians(rotation)
                    cos_a, sin_a = math.cos(rad), math.sin(rad)
                    pts = [[cx + (p[0] - cx) * cos_a - (p[1] - cy) * sin_a,
                            cy + (p[0] - cx) * sin_a + (p[1] - cy) * cos_a] for p in pts]
                if feather > 0:
                    sdf = self._polygon_sdf(pts, xx, yy)
                    canvas = (1.0 - (sdf / feather).clamp(0, 1)) * value
                    canvas = canvas.clamp(0, value)
                else:
                    try:
                        import cv2
                        pts_np = np.array(pts, dtype=np.int32)
                        mask_np = np.zeros((height, width), dtype=np.float32)
                        cv2.fillPoly(mask_np, [pts_np], float(value))
                        canvas = torch.from_numpy(mask_np)
                    except ImportError:
                        canvas = self._fill_polygon_torch(pts, height, width, value)

        elif shape == "line":
            x1 = float(params.get("x1", 0))
            y1 = float(params.get("y1", 0))
            x2 = float(params.get("x2", width))
            y2 = float(params.get("y2", height))
            thickness = float(params.get("thickness", 3))
            lcx, lcy = (x1 + x2) / 2, (y1 + y2) / 2
            rxx, ryy = self._rotate_grid(xx, yy, lcx, lcy, rotation)
            line_len = max(1e-6, math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
            dx = (x2 - x1) / line_len
            dy = (y2 - y1) / line_len
            t = ((rxx - x1) * dx + (ryy - y1) * dy).clamp(0, line_len)
            proj_x = x1 + t * dx
            proj_y = y1 + t * dy
            dist = torch.sqrt((rxx - proj_x) ** 2 + (ryy - proj_y) ** 2)
            half_t = thickness / 2
            if feather > 0:
                canvas = (1.0 - ((dist - half_t) / feather).clamp(0, 1)) * value
                canvas = canvas.clamp(0, value)
            else:
                canvas[dist <= half_t] = value

        elif shape == "triangle":
            pts = params.get("points", None)
            if pts is None or len(pts) < 3:
                cx = float(params.get("cx", width // 2))
                cy = float(params.get("cy", height // 2))
                size = float(params.get("size", 100))
                pts = self._regular_polygon_pts(cx, cy, size, 3)
            else:
                cx = sum(p[0] for p in pts) / len(pts)
                cy = sum(p[1] for p in pts) / len(pts)
            if abs(rotation) > 1e-6:
                rad = math.radians(rotation)
                cos_a, sin_a = math.cos(rad), math.sin(rad)
                pts = [[cx + (p[0] - cx) * cos_a - (p[1] - cy) * sin_a,
                        cy + (p[0] - cx) * sin_a + (p[1] - cy) * cos_a] for p in pts]
            if feather > 0:
                sdf = self._polygon_sdf(pts, xx, yy)
                canvas = (1.0 - (sdf / feather).clamp(0, 1)) * value
                canvas = canvas.clamp(0, value)
            else:
                try:
                    import cv2
                    pts_np = np.array(pts, dtype=np.int32)
                    mask_np = np.zeros((height, width), dtype=np.float32)
                    cv2.fillPoly(mask_np, [pts_np], float(value))
                    canvas = torch.from_numpy(mask_np)
                except ImportError:
                    canvas = self._fill_polygon_torch(pts, height, width, value)

        elif shape == "star":
            cx = float(params.get("cx", width // 2))
            cy = float(params.get("cy", height // 2))
            outer_r = float(params.get("outer_r", 100))
            inner_r = float(params.get("inner_r", 40))
            num_points = int(params.get("num_points", 5))
            pts = []
            for i in range(num_points * 2):
                a = -math.pi / 2 + math.pi * i / num_points
                r = outer_r if i % 2 == 0 else inner_r
                pts.append([cx + r * math.cos(a), cy + r * math.sin(a)])
            if abs(rotation) > 1e-6:
                rad = math.radians(rotation)
                cos_a, sin_a = math.cos(rad), math.sin(rad)
                pts = [[cx + (p[0] - cx) * cos_a - (p[1] - cy) * sin_a,
                        cy + (p[0] - cx) * sin_a + (p[1] - cy) * cos_a] for p in pts]
            if feather > 0:
                sdf = self._polygon_sdf(pts, xx, yy)
                canvas = (1.0 - (sdf / feather).clamp(0, 1)) * value
                canvas = canvas.clamp(0, value)
            else:
                try:
                    import cv2
                    pts_np = np.array(pts, dtype=np.int32)
                    mask_np = np.zeros((height, width), dtype=np.float32)
                    cv2.fillPoly(mask_np, [pts_np], float(value))
                    canvas = torch.from_numpy(mask_np)
                except ImportError:
                    canvas = self._fill_polygon_torch(pts, height, width, value)

        elif shape == "diamond":
            cx = float(params.get("cx", width // 2))
            cy = float(params.get("cy", height // 2))
            dw = float(params.get("w", 100))
            dh = float(params.get("h", 100))
            pts = [[cx, cy - dh / 2], [cx + dw / 2, cy],
                   [cx, cy + dh / 2], [cx - dw / 2, cy]]
            if abs(rotation) > 1e-6:
                rad = math.radians(rotation)
                cos_a, sin_a = math.cos(rad), math.sin(rad)
                pts = [[cx + (p[0] - cx) * cos_a - (p[1] - cy) * sin_a,
                        cy + (p[0] - cx) * sin_a + (p[1] - cy) * cos_a] for p in pts]
            if feather > 0:
                sdf = self._polygon_sdf(pts, xx, yy)
                canvas = (1.0 - (sdf / feather).clamp(0, 1)) * value
                canvas = canvas.clamp(0, value)
            else:
                try:
                    import cv2
                    pts_np = np.array(pts, dtype=np.int32)
                    mask_np = np.zeros((height, width), dtype=np.float32)
                    cv2.fillPoly(mask_np, [pts_np], float(value))
                    canvas = torch.from_numpy(mask_np)
                except ImportError:
                    canvas = self._fill_polygon_torch(pts, height, width, value)

        elif shape == "cross":
            cx = float(params.get("cx", width // 2))
            cy = float(params.get("cy", height // 2))
            size = float(params.get("size", 100))
            thickness = float(params.get("thickness", 30))
            hs, ht = size / 2, thickness / 2
            rxx, ryy = self._rotate_grid(xx, yy, cx, cy, rotation)
            dx = torch.abs(rxx - cx)
            dy = torch.abs(ryy - cy)
            h_bar = (dx <= hs) & (dy <= ht)
            v_bar = (dx <= ht) & (dy <= hs)
            inside = h_bar | v_bar
            if feather > 0:
                # Approximate distance: min of dist to each bar
                d_hbar_x = (dx - hs).clamp(min=0)
                d_hbar_y = (dy - ht).clamp(min=0)
                d_hbar = torch.sqrt(d_hbar_x ** 2 + d_hbar_y ** 2)
                d_vbar_x = (dx - ht).clamp(min=0)
                d_vbar_y = (dy - hs).clamp(min=0)
                d_vbar = torch.sqrt(d_vbar_x ** 2 + d_vbar_y ** 2)
                dist = torch.min(d_hbar, d_vbar)
                canvas = (1.0 - (dist / feather).clamp(0, 1)) * value
            else:
                canvas[inside] = value

        elif shape == "rounded_rectangle":
            x = float(params.get("x", 0))
            y = float(params.get("y", 0))
            w = float(params.get("w", 200))
            h = float(params.get("h", 100))
            cr = float(params.get("corner_radius", 20))
            cr = min(cr, w / 2, h / 2)
            rcx, rcy = x + w / 2, y + h / 2
            rxx, ryy = self._rotate_grid(xx, yy, rcx, rcy, rotation)
            # SDF for rounded rectangle
            dx = (torch.abs(rxx - rcx) - (w / 2 - cr)).clamp(min=0)
            dy = (torch.abs(ryy - rcy) - (h / 2 - cr)).clamp(min=0)
            dist = torch.sqrt(dx ** 2 + dy ** 2) - cr
            # Clamp: inside the full rect but outside the rounded rect
            inside_rect = (torch.abs(rxx - rcx) <= w / 2) & (torch.abs(ryy - rcy) <= h / 2)
            dist = torch.where(inside_rect & (dist < 0), dist, dist.clamp(min=0))
            if feather > 0:
                canvas = (1.0 - (dist / feather).clamp(0, 1)) * value
                # Ensure fully inside is solid
                canvas = torch.where(dist <= 0, torch.tensor(value), canvas)
            else:
                canvas[dist <= 0] = value

        elif shape == "heart":
            cx = float(params.get("cx", width // 2))
            cy = float(params.get("cy", height // 2))
            size = float(params.get("size", 100))
            rxx, ryy = self._rotate_grid(xx, yy, cx, cy, rotation)
            # Parametric heart SDF: normalize coordinates
            px = (rxx - cx) / size
            py = -(ryy - cy) / size + 0.4  # Shift up so center feels natural
            # Heart implicit: (x^2 + y^2 - 1)^3 - x^2 * y^3 < 0
            x2 = px * px
            y2 = py * py
            dist_val = (x2 + y2 - 1.0) ** 3 - x2 * (py ** 3)
            if feather > 0:
                # Approximate feather via gradient magnitude
                grad_scale = size * 0.3
                canvas = (1.0 - (dist_val / (feather / grad_scale)).clamp(0, 1)) * value
                canvas = canvas.clamp(0, value)
            else:
                canvas[dist_val <= 0] = value

        elif shape == "arrow":
            cx = float(params.get("cx", width // 2))
            cy = float(params.get("cy", height // 2))
            length = float(params.get("length", 200))
            arrow_w = float(params.get("width", 30))
            head_length = float(params.get("head_length", 60))
            head_width = float(params.get("head_width", 80))
            rxx, ryy = self._rotate_grid(xx, yy, cx, cy, rotation)
            # Arrow pointing right: shaft + triangular head
            shaft_len = length - head_length
            half_shaft = arrow_w / 2
            lx = rxx - cx
            ly = ryy - cy
            # Shaft: centered, from -length/2 to -length/2 + shaft_len
            shaft_x_start = -length / 2
            shaft_x_end = shaft_x_start + shaft_len
            shaft = (lx >= shaft_x_start) & (lx <= shaft_x_end) & \
                    (torch.abs(ly) <= half_shaft)
            # Head: triangle from shaft_x_end to length/2
            head_start = shaft_x_end
            head_end = length / 2
            head_progress = ((lx - head_start) / max(head_length, 1e-6)).clamp(0, 1)
            head_half_w = head_width / 2 * (1.0 - head_progress)
            head = (lx >= head_start) & (lx <= head_end) & \
                   (torch.abs(ly) <= head_half_w)
            inside = shaft | head
            if feather > 0:
                # Compute approximate distance
                shaft_dx = torch.max(shaft_x_start - lx, lx - shaft_x_end).clamp(min=0)
                shaft_dy = (torch.abs(ly) - half_shaft).clamp(min=0)
                shaft_dist = torch.sqrt(shaft_dx ** 2 + shaft_dy ** 2)
                head_dy = (torch.abs(ly) - head_half_w).clamp(min=0)
                head_dx = torch.max(head_start - lx, lx - head_end).clamp(min=0)
                head_dist = torch.sqrt(head_dx ** 2 + head_dy ** 2)
                dist = torch.min(shaft_dist, head_dist)
                canvas = (1.0 - (dist / feather).clamp(0, 1)) * value
            else:
                canvas[inside] = value

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
        """Scanline polygon fill (no cv2 dependency)."""
        canvas = torch.zeros(h, w, dtype=torch.float32)
        if not pts or len(pts) < 3:
            return canvas
        n = len(pts)
        for y in range(h):
            y_center = float(y) + 0.5
            nodes = []
            j = n - 1
            for i in range(n):
                yi, xi = float(pts[i][1]), float(pts[i][0])
                yj, xj = float(pts[j][1]), float(pts[j][0])
                if (yi <= y_center < yj) or (yj <= y_center < yi):
                    if abs(yj - yi) > 1e-8:
                        x_intersect = xi + (y_center - yi) / (yj - yi) * (xj - xi)
                        nodes.append(x_intersect)
                j = i
            nodes.sort()
            for k in range(0, len(nodes) - 1, 2):
                x_start = max(0, int(nodes[k]))
                x_end = min(w, int(nodes[k + 1]) + 1)
                canvas[y, x_start:x_end] = value
        return canvas
