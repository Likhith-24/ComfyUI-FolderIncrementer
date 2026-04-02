"""
SplineMaskEditorMEC — Interactive spline drawing tool for mask creation.

Draw closed or open spline shapes on an image canvas inside ComfyUI.
Supports Catmull-Rom, Bezier (with tangent handles), and polyline modes.

Outputs THREE things simultaneously:
  1. mask         — MASK (B,H,W): rasterized filled spline region
  2. coords_json  — STRING: SAM-compatible point coords from control points
  3. spline_data_out — SPLINE_DATA: structured dict for downstream nodes

Single frame: connect mask to any mask input directly.
Video seed:   connect mask to SAM2 video predictor as frame-0 seed mask.
SAM prompts:  connect coords_json to SAM Mask Generator positive_coords.

JS companion: js/spline_mask_editor.js — interactive canvas widget.

VRAM Tier: 1 (pure tensor ops, no models)

Files CREATED: nodes/spline_mask_editor.py, js/spline_mask_editor.js
Files MODIFIED: __init__.py (import + mapping)
Files UNTOUCHED: All existing node files
"""

from __future__ import annotations

import json
import math
import logging
from typing import List, Tuple, Optional

import torch
import torch.nn.functional as F
import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

logger = logging.getLogger("MEC")


# ══════════════════════════════════════════════════════════════════════
#  Gaussian blur helper (local, no cross-file dependency)
# ══════════════════════════════════════════════════════════════════════

def _gauss_kernel_1d(sigma: float, device: torch.device,
                     dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Create a normalized 1D Gaussian kernel."""
    if sigma <= 0:
        return torch.ones(1, device=device, dtype=dtype)
    radius = max(1, int(math.ceil(3.0 * sigma)))
    size = 2 * radius + 1
    x = torch.arange(size, device=device, dtype=dtype) - radius
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum()


def _gaussian_blur_mask(mask: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable 2D Gaussian blur on (B, H, W) mask. Pure torch."""
    if sigma <= 0:
        return mask
    device = mask.device
    k1d = _gauss_kernel_1d(sigma, device, mask.dtype)
    pad = len(k1d) // 2
    m4 = mask.unsqueeze(1)  # (B, 1, H, W)
    kh = k1d.view(1, 1, 1, -1)
    out = F.conv2d(F.pad(m4, (pad, pad, 0, 0), mode="replicate"), kh)
    kv = k1d.view(1, 1, -1, 1)
    out = F.conv2d(F.pad(out, (0, 0, pad, pad), mode="replicate"), kv)
    return out.squeeze(1)


# ══════════════════════════════════════════════════════════════════════
#  Spline sampling algorithms
# ══════════════════════════════════════════════════════════════════════

def _catmull_rom_sample(points: List[Tuple[float, float]],
                        samples_per_segment: int,
                        closed: bool) -> List[Tuple[float, float]]:
    """Centripetal Catmull-Rom spline interpolation through control points.

    For each segment of 4 consecutive points (P0, P1, P2, P3), computes the
    curve passing through P1→P2 using centripetal parameterization (alpha=0.5)
    which avoids cusps and self-intersections.

    If closed=True, wraps points so the curve forms a closed loop.
    Returns list of (x, y) sampled curve points.
    """
    n = len(points)
    if n < 2:
        return list(points)
    if n == 2:
        # Linear interpolation for 2 points
        result = []
        p0, p1 = points[0], points[1]
        for i in range(samples_per_segment + 1):
            t = i / max(samples_per_segment, 1)
            result.append((p0[0] + t * (p1[0] - p0[0]),
                           p0[1] + t * (p1[1] - p0[1])))
        return result

    # Build extended point list for closed/open curves
    if closed:
        ext = [points[-1]] + list(points) + [points[0], points[1]]
    else:
        # Reflect endpoints for open curves
        p_start = (2 * points[0][0] - points[1][0],
                   2 * points[0][1] - points[1][1])
        p_end = (2 * points[-1][0] - points[-2][0],
                 2 * points[-1][1] - points[-2][1])
        ext = [p_start] + list(points) + [p_end]

    result = []
    num_segments = n if closed else (n - 1)

    for seg in range(num_segments):
        p0 = ext[seg]
        p1 = ext[seg + 1]
        p2 = ext[seg + 2]
        p3 = ext[seg + 3]

        # Centripetal parameterization (alpha = 0.5)
        def _dist(a, b):
            return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) + 1e-8

        d01 = _dist(p0, p1) ** 0.5  # alpha = 0.5 → sqrt of distance
        d12 = _dist(p1, p2) ** 0.5
        d23 = _dist(p2, p3) ** 0.5

        # Knot values
        t0 = 0.0
        t1 = t0 + d01
        t2 = t1 + d12
        t3 = t2 + d23

        for i in range(samples_per_segment):
            t = t1 + (t2 - t1) * (i / max(samples_per_segment, 1))

            # Barry and Goldman's pyramidal formulation
            def _lerp_pt(pa, pb, ta, tb, t_val):
                w = (t_val - ta) / max(tb - ta, 1e-10)
                return (pa[0] + w * (pb[0] - pa[0]),
                        pa[1] + w * (pb[1] - pa[1]))

            a1 = _lerp_pt(p0, p1, t0, t1, t)
            a2 = _lerp_pt(p1, p2, t1, t2, t)
            a3 = _lerp_pt(p2, p3, t2, t3, t)

            b1 = _lerp_pt(a1, a2, t0, t2, t)
            b2 = _lerp_pt(a2, a3, t1, t3, t)

            c = _lerp_pt(b1, b2, t1, t2, t)
            result.append(c)

    # Add final point
    if not closed and result:
        result.append(points[-1])

    return result


def _bezier_sample(points: List[Tuple[float, float]],
                   handles: List[dict],
                   samples_per_segment: int) -> List[Tuple[float, float]]:
    """Cubic Bezier spline with explicit control point handles.

    For each segment between points[i] and points[i+1], uses:
      P0 = points[i]
      CP1 = handles[i]["cp2x"], handles[i]["cp2y"]   (out-handle of point i)
      CP2 = handles[i+1]["cp1x"], handles[i+1]["cp1y"] (in-handle of point i+1)
      P1 = points[i+1]

    B(t) = (1-t)^3 * P0 + 3*(1-t)^2*t * CP1 + 3*(1-t)*t^2 * CP2 + t^3 * P1

    Returns list of (x, y) sampled curve points.
    """
    n = len(points)
    if n < 2:
        return list(points)

    result = []
    for seg in range(n - 1):
        p0 = points[seg]
        p1 = points[seg + 1]

        # Get control points from handles
        if handles and seg < len(handles):
            h0 = handles[seg]
            cp1 = (float(h0.get("cp2x", p0[0])), float(h0.get("cp2y", p0[1])))
        else:
            cp1 = p0

        if handles and (seg + 1) < len(handles):
            h1 = handles[seg + 1]
            cp2 = (float(h1.get("cp1x", p1[0])), float(h1.get("cp1y", p1[1])))
        else:
            cp2 = p1

        for i in range(samples_per_segment):
            t = i / max(samples_per_segment, 1)
            omt = 1.0 - t
            # Cubic Bezier formula
            x = (omt ** 3 * p0[0] +
                 3 * omt ** 2 * t * cp1[0] +
                 3 * omt * t ** 2 * cp2[0] +
                 t ** 3 * p1[0])
            y = (omt ** 3 * p0[1] +
                 3 * omt ** 2 * t * cp1[1] +
                 3 * omt * t ** 2 * cp2[1] +
                 t ** 3 * p1[1])
            result.append((x, y))

    # Add final point
    if points:
        result.append(points[-1])
    return result


def _polyline_sample(points: List[Tuple[float, float]],
                     closed: bool) -> List[Tuple[float, float]]:
    """Straight-line segments through all control points.

    If closed, connects last point to first.
    Returns list of (x, y) — just the points themselves (vertices).
    """
    result = list(points)
    if closed and len(points) >= 3:
        result.append(points[0])  # close the loop
    return result


# ══════════════════════════════════════════════════════════════════════
#  Polygon rasterization (mask generation)
# ══════════════════════════════════════════════════════════════════════

def _fill_polygon_cv2(pts: List[Tuple[float, float]],
                      H: int, W: int) -> np.ndarray:
    """Rasterize filled polygon using cv2.fillPoly. Returns (H,W) float32 [0,1]."""
    if len(pts) < 3:
        return np.zeros((H, W), dtype=np.float32)
    pts_np = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
    canvas = np.zeros((H, W), dtype=np.float32)
    cv2.fillPoly(canvas, [pts_np], 1.0)
    return canvas


def _fill_polygon_scanline(pts: List[Tuple[float, float]],
                           H: int, W: int) -> np.ndarray:
    """Scanline polygon fill (pure Python/numpy, no cv2 dependency).

    For each row y, find all x-intersections with polygon edges,
    sort them, fill between consecutive pairs.
    """
    canvas = np.zeros((H, W), dtype=np.float32)
    if len(pts) < 3:
        return canvas

    n = len(pts)
    for y in range(H):
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
            x_start = max(0, int(math.floor(nodes[k])))
            x_end = min(W, int(math.ceil(nodes[k + 1])))
            canvas[y, x_start:x_end] = 1.0
    return canvas


def _rasterize_splines(spline_data_json: str, H: int, W: int,
                       spline_type: str, closed: bool,
                       samples_per_segment: int,
                       feather_radius: float,
                       invert: bool,
                       device: torch.device) -> torch.Tensor:
    """Parse spline JSON, sample curves, rasterize to filled mask.

    1. Parse spline_data_json (list of shape dicts)
    2. For each shape: sample curve points using appropriate algorithm
    3. Rasterize filled polygon (cv2 or scanline fallback)
    4. Union all shapes: max across shapes
    5. Apply feather if > 0: Gaussian blur on mask
    6. Optionally invert

    Returns: (1, H, W) float32 tensor [0, 1]
    """
    try:
        shapes = json.loads(spline_data_json) if isinstance(spline_data_json, str) else spline_data_json
    except (json.JSONDecodeError, TypeError):
        shapes = []

    if not isinstance(shapes, list):
        shapes = []

    combined = np.zeros((H, W), dtype=np.float32)

    for shape in shapes:
        if not isinstance(shape, dict):
            continue

        raw_pts = shape.get("points", [])
        if len(raw_pts) < 2:
            continue

        pts = [(float(p["x"] if isinstance(p, dict) else p[0]),
                float(p["y"] if isinstance(p, dict) else p[1]))
               for p in raw_pts]

        shape_type = shape.get("type", spline_type)
        shape_closed = shape.get("closed", closed)
        handles = shape.get("handles", None)

        # Sample curve points
        if shape_type == "bezier" and handles:
            curve_pts = _bezier_sample(pts, handles, samples_per_segment)
        elif shape_type == "polyline":
            curve_pts = _polyline_sample(pts, shape_closed)
        else:
            # Default: Catmull-Rom
            curve_pts = _catmull_rom_sample(pts, samples_per_segment, shape_closed)

        if len(curve_pts) < 3 and shape_closed:
            continue

        # Rasterize filled polygon
        if shape_closed:
            if HAS_CV2:
                poly_mask = _fill_polygon_cv2(curve_pts, H, W)
            else:
                poly_mask = _fill_polygon_scanline(curve_pts, H, W)
        else:
            # Open polylines: rasterize as thick stroke (2px)
            poly_mask = np.zeros((H, W), dtype=np.float32)
            if HAS_CV2:
                line_pts = np.array(curve_pts, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(poly_mask, [line_pts], isClosed=False,
                              color=1.0, thickness=2)
            else:
                # Approximate: draw points
                for px, py in curve_pts:
                    ix, iy = int(round(px)), int(round(py))
                    if 0 <= iy < H and 0 <= ix < W:
                        poly_mask[iy, ix] = 1.0

        combined = np.maximum(combined, poly_mask)

    mask = torch.from_numpy(combined).to(device=device, dtype=torch.float32)

    # Apply feather (Gaussian blur)
    if feather_radius > 0 and mask.max() > 0:
        mask = _gaussian_blur_mask(mask.unsqueeze(0), feather_radius).squeeze(0)

    # Invert if requested
    if invert:
        mask = 1.0 - mask

    return mask.unsqueeze(0).clamp(0.0, 1.0)  # (1, H, W)


# ══════════════════════════════════════════════════════════════════════
#  Coords extraction for SAM compatibility
# ══════════════════════════════════════════════════════════════════════

def _coords_from_splines(spline_data_json: str) -> str:
    """Extract control points from all closed splines.

    Returns SAM-compatible JSON: [{"x": int, "y": int, "label": 1}, ...]
    One entry per control point from all closed shapes.
    """
    try:
        shapes = json.loads(spline_data_json) if isinstance(spline_data_json, str) else spline_data_json
    except (json.JSONDecodeError, TypeError):
        return "[]"

    if not isinstance(shapes, list):
        return "[]"

    coords = []
    for shape in shapes:
        if not isinstance(shape, dict):
            continue
        raw_pts = shape.get("points", [])
        for p in raw_pts:
            if isinstance(p, dict):
                x = int(round(float(p.get("x", 0))))
                y = int(round(float(p.get("y", 0))))
            elif isinstance(p, (list, tuple)) and len(p) >= 2:
                x = int(round(float(p[0])))
                y = int(round(float(p[1])))
            else:
                continue
            coords.append({"x": x, "y": y, "label": 1})

    return json.dumps(coords)


# ══════════════════════════════════════════════════════════════════════
#  SPLINE_DATA type builder
# ══════════════════════════════════════════════════════════════════════

def _build_spline_data(spline_data_json: str, canvas_w: int, canvas_h: int) -> dict:
    """Build the SPLINE_DATA custom type dict from serialized JSON.

    Structure:
    {
        "shapes": [
            {
                "type": "catmull_rom" | "bezier" | "polyline",
                "closed": bool,
                "points": [{"x": int, "y": int}, ...],
                "handles": None | [{"cp1x":f,"cp1y":f,"cp2x":f,"cp2y":f}, ...]
            }
        ],
        "canvas_width": int,
        "canvas_height": int,
        "frame_index": 0
    }
    """
    try:
        shapes = json.loads(spline_data_json) if isinstance(spline_data_json, str) else spline_data_json
    except (json.JSONDecodeError, TypeError):
        shapes = []

    if not isinstance(shapes, list):
        shapes = []

    return {
        "shapes": shapes,
        "canvas_width": canvas_w,
        "canvas_height": canvas_h,
        "frame_index": 0,
    }


# ══════════════════════════════════════════════════════════════════════
#  NODE: SplineMaskEditorMEC
# ══════════════════════════════════════════════════════════════════════

class SplineMaskEditorMEC:
    """Interactive spline drawing tool. Draw closed shapes on the image canvas.
    Supports Catmull-Rom, Bezier (with handles), and polyline modes.

    Single frame: connect mask output to any mask input.
    Video seed:   connect mask to SAM2 video predictor as frame-0 seed mask.
    SAM prompts:  connect coords_json to SAM Mask Generator positive_coords.
    Downstream:   connect spline_data to Motion Mask Tracker or shape nodes.
    """

    VRAM_TIER = 1

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Reference image shown in editor canvas (B,H,W,C).",
                }),
                "spline_data": ("STRING", {
                    "default": "[]",
                    "multiline": True,
                    "tooltip": "Internal serialized spline state from the JS editor. Do not edit manually.",
                }),
                "spline_type": (["catmull_rom", "bezier", "polyline"], {
                    "default": "catmull_rom",
                    "tooltip": (
                        "catmull_rom: smooth curve through all control points. "
                        "bezier: smooth curve with editable tangent handles. "
                        "polyline: straight segments between points."
                    ),
                }),
                "closed": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Close the spline loop (filled region). False = open path for trajectories.",
                }),
                "smoothing": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable spline smoothing. Disable for polygonal/hard shapes.",
                }),
                "samples_per_segment": ("INT", {
                    "default": 20, "min": 2, "max": 100, "step": 1,
                    "tooltip": "Curve resolution per segment. Higher = smoother mask edge.",
                }),
                "feather_radius": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 64.0, "step": 0.5,
                    "tooltip": "Gaussian blur on mask edge after rasterization. 0 = hard edge.",
                }),
                "invert": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Fill outside the spline region instead of inside.",
                }),
            },
            "optional": {
                "width": ("INT", {
                    "default": 0, "min": 0, "max": 16384, "step": 1,
                    "tooltip": "Output width. 0 = match image width.",
                }),
                "height": ("INT", {
                    "default": 0, "min": 0, "max": 16384, "step": 1,
                    "tooltip": "Output height. 0 = match image height.",
                }),
            },
        }

    RETURN_TYPES = ("MASK", "STRING", "SPLINE_DATA")
    RETURN_NAMES = ("mask", "coords_json", "spline_data_out")
    FUNCTION = "execute"
    CATEGORY = "MaskEditControl/Spline"
    DESCRIPTION = (
        "Draw closed or open spline shapes on the image canvas. "
        "Supports Catmull-Rom, Bezier, and polyline. Outputs mask, "
        "SAM-compatible coords, and spline data for downstream nodes."
    )

    def execute(self, image: torch.Tensor, spline_data: str,
                spline_type: str, closed: bool, smoothing: bool,
                samples_per_segment: int, feather_radius: float,
                invert: bool,
                width: int = 0, height: int = 0) -> tuple:

        B, img_H, img_W, C = image.shape
        device = image.device

        # Determine output dimensions
        out_W = width if width > 0 else img_W
        out_H = height if height > 0 else img_H

        # Override samples_per_segment if smoothing disabled
        actual_samples = samples_per_segment if smoothing else 1

        # Rasterize splines to mask
        mask = _rasterize_splines(
            spline_data_json=spline_data,
            H=out_H, W=out_W,
            spline_type=spline_type,
            closed=closed,
            samples_per_segment=actual_samples,
            feather_radius=feather_radius,
            invert=invert,
            device=device,
        )  # (1, H, W)

        # Expand to batch size if needed (same mask for all frames)
        if B > 1:
            mask = mask.expand(B, -1, -1).contiguous()

        # Extract SAM-compatible coords
        coords_json = _coords_from_splines(spline_data)

        # Build SPLINE_DATA custom type
        spline_data_out = _build_spline_data(spline_data, out_W, out_H)

        # Info logging
        try:
            shapes = json.loads(spline_data) if isinstance(spline_data, str) else spline_data
            n_shapes = len(shapes) if isinstance(shapes, list) else 0
            n_points = sum(len(s.get("points", [])) for s in shapes
                          if isinstance(s, dict)) if isinstance(shapes, list) else 0
        except (json.JSONDecodeError, TypeError):
            n_shapes = 0
            n_points = 0

        info_msg = (
            f"[MEC] SplineMaskEditor: {n_shapes} shape(s), {n_points} control points | "
            f"type={spline_type} | closed={closed} | "
            f"mask {out_W}x{out_H} | feather={feather_radius:.1f}"
        )
        logger.info(info_msg)

        return (mask, coords_json, spline_data_out)
