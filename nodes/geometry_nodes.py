"""
Geometry / pass-data nodes (MEC):
  - DepthWarpMEC: Depth-driven parallax warp (cheap stereo synth).
  - NormalToCurvatureMEC: Compute curvature/divergence from a normal pass.
  - PositionPassSplitterMEC: Split a position pass into X/Y/Z masks (or images).

Pure torch.
"""
from __future__ import annotations

import logging

import torch

logger = logging.getLogger("MEC.Geometry")


class DepthWarpMEC:
    """Warp an IMAGE horizontally by depth (parallax shift).

    shift_pixels(x,y) = max_shift * (depth(x,y) - pivot)

    Positive shift moves pixels right (use negative ``max_shift`` for the
    other eye). Pixels are interpolated bilinearly via ``grid_sample``;
    holes are filled with border replication.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "depth": ("IMAGE",),
            },
            "optional": {
                "max_shift_pixels": ("FLOAT", {"default": 16.0, "min": -512.0, "max": 512.0, "step": 0.5}),
                "pivot": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "warp"
    CATEGORY = "MaskEditControl/Geometry"
    DESCRIPTION = "Horizontal parallax warp driven by a depth pass."

    def warp(
        self, image: torch.Tensor, depth: torch.Tensor,
        max_shift_pixels: float = 16.0, pivot: float = 0.5,
    ):
        if image.shape[1:3] != depth.shape[1:3]:
            x = depth.permute(0, 3, 1, 2)
            x = torch.nn.functional.interpolate(
                x, size=image.shape[1:3], mode="bilinear", align_corners=False,
            )
            depth = x.permute(0, 2, 3, 1)
        d = depth[..., 0]  # [B,H,W]
        B, H, W = d.shape
        device, dtype = image.device, image.dtype
        ys = torch.linspace(-1, 1, H, device=device, dtype=dtype).view(1, H, 1).expand(B, H, W)
        xs = torch.linspace(-1, 1, W, device=device, dtype=dtype).view(1, 1, W).expand(B, H, W)
        shift_norm = (d - pivot) * max_shift_pixels * (2.0 / max(W - 1, 1))
        grid = torch.stack([xs - shift_norm, ys], dim=-1)  # [B,H,W,2]
        x = image.permute(0, 3, 1, 2)
        out = torch.nn.functional.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=True)
        return (out.permute(0, 2, 3, 1),)


class NormalToCurvatureMEC:
    """Approximate curvature from a tangent-space normal pass.

    Computes divergence of the (Nx, Ny) field via central differences:
        curvature = ∂Nx/∂x + ∂Ny/∂y

    Output is normalized to [0,1] mask per frame. Useful for edge wear,
    cavity AO synthesis, etc.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"normal": ("IMAGE",)},
            "optional": {
                "scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 32.0, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("curvature",)
    FUNCTION = "compute"
    CATEGORY = "MaskEditControl/Geometry"
    DESCRIPTION = "Compute curvature mask from a tangent-space normal pass."

    def compute(self, normal: torch.Tensor, scale: float = 1.0):
        # Map [0,1] normals to [-1,1]
        n = normal * 2.0 - 1.0
        nx = n[..., 0]
        ny = n[..., 1]
        # Central differences with reflect-pad
        def cd(a: torch.Tensor, axis: int) -> torch.Tensor:
            x = a.unsqueeze(1)  # [B,1,H,W]
            x = torch.nn.functional.pad(x, (1, 1, 1, 1), mode="reflect")
            if axis == 0:  # along H
                d = x[:, :, 2:, 1:-1] - x[:, :, :-2, 1:-1]
            else:  # along W
                d = x[:, :, 1:-1, 2:] - x[:, :, 1:-1, :-2]
            return d.squeeze(1) * 0.5
        dnx_dx = cd(nx, axis=1)
        dny_dy = cd(ny, axis=0)
        curv = (dnx_dx + dny_dy) * scale
        # Normalize per-frame to [0,1] with 0.5 as "flat"
        curv = (curv * 0.5 + 0.5).clamp(0.0, 1.0)
        return (curv,)


class PositionPassSplitterMEC:
    """Split a world-position pass (XYZ encoded as RGB) into per-axis masks.

    Each axis is normalized via either provided (min, max) or per-frame
    auto-range. Useful for slicing scenes by world-space position.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"position": ("IMAGE",)},
            "optional": {
                "auto_normalize": ("BOOLEAN", {"default": True}),
                "x_min": ("FLOAT", {"default": 0.0, "min": -1e6, "max": 1e6}),
                "x_max": ("FLOAT", {"default": 1.0, "min": -1e6, "max": 1e6}),
                "y_min": ("FLOAT", {"default": 0.0, "min": -1e6, "max": 1e6}),
                "y_max": ("FLOAT", {"default": 1.0, "min": -1e6, "max": 1e6}),
                "z_min": ("FLOAT", {"default": 0.0, "min": -1e6, "max": 1e6}),
                "z_max": ("FLOAT", {"default": 1.0, "min": -1e6, "max": 1e6}),
            },
        }

    RETURN_TYPES = ("MASK", "MASK", "MASK")
    RETURN_NAMES = ("x_mask", "y_mask", "z_mask")
    FUNCTION = "split"
    CATEGORY = "MaskEditControl/Geometry"
    DESCRIPTION = "Split position pass into X/Y/Z masks (auto- or manually-ranged)."

    def split(
        self, position: torch.Tensor,
        auto_normalize: bool = True,
        x_min: float = 0.0, x_max: float = 1.0,
        y_min: float = 0.0, y_max: float = 1.0,
        z_min: float = 0.0, z_max: float = 1.0,
    ):
        out = []
        bounds = [(x_min, x_max), (y_min, y_max), (z_min, z_max)]
        for axis in range(3):
            chan = position[..., axis]
            if auto_normalize:
                lo = float(chan.amin())
                hi = float(chan.amax())
            else:
                lo, hi = bounds[axis]
            rng = max(hi - lo, 1e-8)
            out.append(((chan - lo) / rng).clamp(0.0, 1.0))
        return tuple(out)


NODE_CLASS_MAPPINGS = {
    "DepthWarpMEC": DepthWarpMEC,
    "NormalToCurvatureMEC": NormalToCurvatureMEC,
    "PositionPassSplitterMEC": PositionPassSplitterMEC,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DepthWarpMEC": "Depth Warp (MEC)",
    "NormalToCurvatureMEC": "Normal → Curvature (MEC)",
    "PositionPassSplitterMEC": "Position Pass Splitter (MEC)",
}
