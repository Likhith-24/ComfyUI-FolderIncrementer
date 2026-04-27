"""
Color science nodes (MEC):
  - ColorSpaceConvertMEC: sRGB ↔ Linear ↔ Rec.709 ↔ ACEScg conversions.
  - LUTApplyMEC: Apply a .cube LUT (1D or 3D) to an IMAGE.
  - ExposureGradeMEC: Exposure (stops), white-balance temp/tint, contrast pivot.

Pure torch on the operating side. ``.cube`` parser is plain-Python.
No PyOpenColorIO dependency required (lazy import only if user explicitly
opts into ``colorspace='ocio_<name>'`` — not enabled here to keep the
node self-contained).
"""
from __future__ import annotations

import json
import logging
import os
import re

import torch

logger = logging.getLogger("MEC.ColorScience")


# ──────────────────────────────────────────────────────────────────────
#  Transfer curves
# ──────────────────────────────────────────────────────────────────────

def _srgb_to_linear(t: torch.Tensor) -> torch.Tensor:
    a = 0.055
    return torch.where(
        t <= 0.04045,
        t / 12.92,
        ((t + a) / (1 + a)).clamp_min(1e-12) ** 2.4,
    )


def _linear_to_srgb(t: torch.Tensor) -> torch.Tensor:
    a = 0.055
    return torch.where(
        t <= 0.0031308,
        12.92 * t,
        (1 + a) * t.clamp_min(1e-12) ** (1 / 2.4) - a,
    )


def _rec709_to_linear(t: torch.Tensor) -> torch.Tensor:
    # Rec.709 OETF inverse
    return torch.where(
        t < 0.081,
        t / 4.5,
        ((t + 0.099) / 1.099).clamp_min(1e-12) ** (1 / 0.45),
    )


def _linear_to_rec709(t: torch.Tensor) -> torch.Tensor:
    return torch.where(
        t < 0.018,
        4.5 * t,
        1.099 * t.clamp_min(1e-12) ** 0.45 - 0.099,
    )


# ACEScg ↔ sRGB matrices (linear). Source: ACES TB-2014-004.
_M_SRGB_TO_ACESCG = torch.tensor([
    [0.4397010, 0.3829780, 0.1773350],
    [0.0897923, 0.8134230, 0.0967926],
    [0.0175440, 0.1115440, 0.8707040],
])
# Numerical inverse of the forward matrix (so round-trip is bit-exact in
# float32). Recomputed at module load to avoid drift between published
# tables.
_M_ACESCG_TO_SRGB = torch.linalg.inv(_M_SRGB_TO_ACESCG)


def _apply_matrix(rgb: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """rgb: [...,3], m: [3,3] → [...,3]"""
    out = torch.einsum("...c,kc->...k", rgb, m.to(rgb.dtype).to(rgb.device))
    return out


_SPACES = ["srgb", "linear", "rec709", "acescg"]


def _convert(img: torch.Tensor, src: str, dst: str) -> torch.Tensor:
    """Convert IMAGE [B,H,W,3] from src space to dst space."""
    if src == dst:
        return img
    # Move everything through "linear-sRGB" as the reference.
    if src == "srgb":
        ref = _srgb_to_linear(img)
    elif src == "rec709":
        ref = _rec709_to_linear(img)
    elif src == "acescg":
        ref = _apply_matrix(img, _M_ACESCG_TO_SRGB)
    elif src == "linear":
        ref = img
    else:
        raise ValueError(f"Unknown source space: {src}")

    if dst == "srgb":
        return _linear_to_srgb(ref).clamp(0.0, 1.0)
    if dst == "rec709":
        return _linear_to_rec709(ref).clamp(0.0, 1.0)
    if dst == "acescg":
        return _apply_matrix(ref, _M_SRGB_TO_ACESCG)
    if dst == "linear":
        return ref
    raise ValueError(f"Unknown destination space: {dst}")


class ColorSpaceConvertMEC:
    """Convert IMAGE between sRGB / linear / Rec.709 / ACEScg."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "src_space": (_SPACES, {"default": "srgb"}),
                "dst_space": (_SPACES, {"default": "linear"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "convert"
    CATEGORY = "MaskEditControl/Color"
    DESCRIPTION = "Convert IMAGE between sRGB, linear, Rec.709, and ACEScg."

    def convert(self, image: torch.Tensor, src_space: str, dst_space: str):
        return (_convert(image, src_space, dst_space),)


# ──────────────────────────────────────────────────────────────────────
#  .cube LUT parser
# ──────────────────────────────────────────────────────────────────────

_CUBE_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def parse_cube_lut(path: str) -> dict:
    """Parse an Adobe ``.cube`` LUT file (1D or 3D).

    Returns ``{"dim": int, "size": int, "domain_min":[..], "domain_max":[..], "data": tensor}``.
    For 1D LUT: data shape (size, 3).
    For 3D LUT: data shape (size, size, size, 3) ordered R-fastest.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"LUT not found: {path!r}")

    dim = 3
    size = 0
    domain_min = [0.0, 0.0, 0.0]
    domain_max = [1.0, 1.0, 1.0]
    rows: list[list[float]] = []

    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            up = line.upper()
            if up.startswith("LUT_3D_SIZE"):
                size = int(line.split()[-1])
                dim = 3
            elif up.startswith("LUT_1D_SIZE"):
                size = int(line.split()[-1])
                dim = 1
            elif up.startswith("DOMAIN_MIN"):
                vals = [float(x) for x in _CUBE_NUM_RE.findall(line)]
                if len(vals) >= 3:
                    domain_min = vals[:3]
            elif up.startswith("DOMAIN_MAX"):
                vals = [float(x) for x in _CUBE_NUM_RE.findall(line)]
                if len(vals) >= 3:
                    domain_max = vals[:3]
            elif up.startswith("TITLE"):
                continue
            else:
                vals = [float(x) for x in _CUBE_NUM_RE.findall(line)]
                if len(vals) >= 3:
                    rows.append(vals[:3])

    if size <= 0:
        raise ValueError(f"LUT {path!r} declared no size.")
    expected = size if dim == 1 else size ** 3
    if len(rows) != expected:
        raise ValueError(
            f"LUT {path!r}: expected {expected} rows for {dim}D size {size}, got {len(rows)}."
        )
    data = torch.tensor(rows, dtype=torch.float32)
    if dim == 3:
        # In a .cube file the R channel varies fastest, then G, then B.
        # After view(size,size,size,3) the axes are (B, G, R, channel);
        # permute so that indexing as data[r, g, b] works directly.
        data = data.view(size, size, size, 3).permute(2, 1, 0, 3).contiguous()
    return {
        "dim": dim, "size": size,
        "domain_min": domain_min, "domain_max": domain_max,
        "data": data,
    }


def _apply_lut_3d(img: torch.Tensor, lut: dict) -> torch.Tensor:
    """Trilinear-interp 3D LUT lookup. img: [B,H,W,3] in [0,1]."""
    size = lut["size"]
    data = lut["data"].to(img.device, dtype=img.dtype)  # [S,S,S,3]
    dmin = torch.tensor(lut["domain_min"], device=img.device, dtype=img.dtype)
    dmax = torch.tensor(lut["domain_max"], device=img.device, dtype=img.dtype)
    rng = (dmax - dmin).clamp_min(1e-8)
    norm = ((img - dmin) / rng).clamp(0.0, 1.0)
    coord = norm * (size - 1)
    lo = coord.floor().long().clamp(0, size - 1)
    hi = (lo + 1).clamp(0, size - 1)
    f = coord - lo.to(coord.dtype)
    fr, fg, fb = f[..., 0:1], f[..., 1:2], f[..., 2:3]
    lr, lg, lb = lo[..., 0], lo[..., 1], lo[..., 2]
    hr, hg, hb = hi[..., 0], hi[..., 1], hi[..., 2]

    def _g(r, g, b):
        return data[r, g, b]
    c000 = _g(lr, lg, lb)
    c100 = _g(hr, lg, lb)
    c010 = _g(lr, hg, lb)
    c110 = _g(hr, hg, lb)
    c001 = _g(lr, lg, hb)
    c101 = _g(hr, lg, hb)
    c011 = _g(lr, hg, hb)
    c111 = _g(hr, hg, hb)
    c00 = c000 * (1 - fr) + c100 * fr
    c01 = c001 * (1 - fr) + c101 * fr
    c10 = c010 * (1 - fr) + c110 * fr
    c11 = c011 * (1 - fr) + c111 * fr
    c0 = c00 * (1 - fg) + c10 * fg
    c1 = c01 * (1 - fg) + c11 * fg
    out = c0 * (1 - fb) + c1 * fb
    return out


def _apply_lut_1d(img: torch.Tensor, lut: dict) -> torch.Tensor:
    """Per-channel 1D LUT lookup with linear interp."""
    size = lut["size"]
    data = lut["data"].to(img.device, dtype=img.dtype)  # [S,3]
    coord = img.clamp(0.0, 1.0) * (size - 1)
    lo = coord.floor().long().clamp(0, size - 1)
    hi = (lo + 1).clamp(0, size - 1)
    f = coord - lo.to(coord.dtype)
    out = torch.empty_like(img)
    for c in range(3):
        v_lo = data[lo[..., c], c]
        v_hi = data[hi[..., c], c]
        out[..., c] = v_lo * (1 - f[..., c]) + v_hi * f[..., c]
    return out


class LUTApplyMEC:
    """Apply a .cube LUT (1D or 3D) to an IMAGE."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "lut_path": ("STRING", {"default": ""}),
            },
            "optional": {
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "info_json")
    FUNCTION = "apply"
    CATEGORY = "MaskEditControl/Color"
    DESCRIPTION = "Apply a .cube LUT (Adobe format, 1D or 3D) with optional strength blend."

    def apply(self, image: torch.Tensor, lut_path: str, strength: float = 1.0):
        lut = parse_cube_lut(lut_path)
        if lut["dim"] == 3:
            graded = _apply_lut_3d(image, lut)
        else:
            graded = _apply_lut_1d(image, lut)
        out = image * (1.0 - strength) + graded * strength
        info = {"dim": lut["dim"], "size": lut["size"], "strength": strength}
        return (out.clamp(0.0, 1.0), json.dumps(info))


# ──────────────────────────────────────────────────────────────────────
#  Exposure / WB / Contrast
# ──────────────────────────────────────────────────────────────────────

def _temp_tint_to_rgb_gain(temp: float, tint: float) -> tuple[float, float, float]:
    """Cheap photographic temp/tint → per-channel gain.

    temp in [-100, 100]: positive = warmer (boost R, cut B).
    tint in [-100, 100]: positive = magenta (boost R+B, cut G).
    """
    t = temp / 200.0
    g = tint / 200.0
    r = 1.0 + t + g
    grn = 1.0 - g
    b = 1.0 - t + g
    return (max(r, 0.0), max(grn, 0.0), max(b, 0.0))


class ExposureGradeMEC:
    """Exposure (in stops), WB temp/tint, and contrast around mid-grey pivot."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "exposure_stops": ("FLOAT", {
                    "default": 0.0, "min": -10.0, "max": 10.0, "step": 0.05,
                    "tooltip": "Exposure adjustment in stops (linear multiply by 2**stops).",
                }),
                "temperature": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "tint": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "contrast": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05,
                    "tooltip": "Contrast multiplier around the mid-grey pivot.",
                }),
                "pivot": ("FLOAT", {
                    "default": 0.18, "min": 0.001, "max": 0.999, "step": 0.001,
                    "tooltip": "Mid-grey pivot for contrast (0.18 = scene-linear grey).",
                }),
            },
            "optional": {
                "operate_in_linear": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "If True (recommended), input is treated as sRGB-encoded, "
                        "linearized for the math, then re-encoded. If False, the math "
                        "is done directly on the encoded values (legacy / display-referred)."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "grade"
    CATEGORY = "MaskEditControl/Color"
    DESCRIPTION = "Exposure (stops), WB (temp/tint), and contrast around a pivot."

    def grade(
        self,
        image: torch.Tensor,
        exposure_stops: float,
        temperature: float,
        tint: float,
        contrast: float,
        pivot: float,
        operate_in_linear: bool = True,
    ):
        x = image
        if operate_in_linear:
            x = _srgb_to_linear(x)
        # Exposure
        if exposure_stops != 0.0:
            x = x * (2.0 ** exposure_stops)
        # WB
        if temperature != 0.0 or tint != 0.0:
            r, g, b = _temp_tint_to_rgb_gain(temperature, tint)
            gain = torch.tensor([r, g, b], device=x.device, dtype=x.dtype)
            x = x * gain
        # Contrast around pivot
        if contrast != 1.0:
            x = (x - pivot) * contrast + pivot
        if operate_in_linear:
            x = _linear_to_srgb(x.clamp_min(0.0))
        return (x.clamp(0.0, 1.0),)


NODE_CLASS_MAPPINGS = {
    "ColorSpaceConvertMEC": ColorSpaceConvertMEC,
    "LUTApplyMEC": LUTApplyMEC,
    "ExposureGradeMEC": ExposureGradeMEC,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ColorSpaceConvertMEC": "Color Space Convert (MEC)",
    "LUTApplyMEC": "LUT Apply (.cube) (MEC)",
    "ExposureGradeMEC": "Exposure Grade (MEC)",
}
