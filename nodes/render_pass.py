"""
Render-pass nodes (MEC):
  - MergeRenderPassesMEC: Composite beauty + diffuse/specular/emission/AO passes.
  - DepthOfFieldMaskMEC: Convert a Z-depth pass into a per-pixel CoC alpha
    mask suitable for driving a blur node.

Pure tensor math; no extra dependencies.
"""
from __future__ import annotations

import logging

import torch

logger = logging.getLogger("MEC.RenderPass")


def _match_shape(a: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Resize a to spatial dims of target ([B,H,W,C] or [B,H,W])."""
    if a.shape[1:3] == target.shape[1:3]:
        return a
    # NCHW interpolate
    if a.dim() == 4:
        x = a.permute(0, 3, 1, 2)
        x = torch.nn.functional.interpolate(
            x, size=target.shape[1:3], mode="bilinear", align_corners=False,
        )
        return x.permute(0, 2, 3, 1)
    if a.dim() == 3:
        x = a.unsqueeze(1)
        x = torch.nn.functional.interpolate(
            x, size=target.shape[1:3], mode="bilinear", align_corners=False,
        )
        return x.squeeze(1)
    return a


class MergeRenderPassesMEC:
    """Composite beauty/diffuse/specular/emission/AO into a single IMAGE.

    out = beauty * (ao_strength * AO + (1 - ao_strength)) +
          diffuse_gain * diffuse +
          specular_gain * specular +
          emission_gain * emission

    All inputs are optional except ``beauty``. Missing passes contribute 0.
    AO defaults to multiplicative on beauty.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "beauty": ("IMAGE",),
            },
            "optional": {
                "diffuse": ("IMAGE",),
                "specular": ("IMAGE",),
                "emission": ("IMAGE",),
                "ao": ("IMAGE",),
                "diffuse_gain": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 4.0, "step": 0.05}),
                "specular_gain": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 4.0, "step": 0.05}),
                "emission_gain": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05}),
                "ao_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "merge"
    CATEGORY = "MaskEditControl/Render"
    DESCRIPTION = "Composite beauty + auxiliary render passes."

    def merge(
        self,
        beauty: torch.Tensor,
        diffuse: torch.Tensor | None = None,
        specular: torch.Tensor | None = None,
        emission: torch.Tensor | None = None,
        ao: torch.Tensor | None = None,
        diffuse_gain: float = 0.0,
        specular_gain: float = 0.0,
        emission_gain: float = 1.0,
        ao_strength: float = 1.0,
    ):
        out = beauty.clone()
        if ao is not None:
            ao_m = _match_shape(ao, beauty)
            ao_m = ao_m[..., :1] if ao_m.shape[-1] >= 1 else ao_m
            mix = ao_strength * ao_m + (1.0 - ao_strength)
            out = out * mix
        if diffuse is not None and diffuse_gain != 0.0:
            out = out + diffuse_gain * _match_shape(diffuse, beauty)
        if specular is not None and specular_gain != 0.0:
            out = out + specular_gain * _match_shape(specular, beauty)
        if emission is not None and emission_gain != 0.0:
            out = out + emission_gain * _match_shape(emission, beauty)
        return (out.clamp(0.0, 1e6),)


class DepthOfFieldMaskMEC:
    """Convert a depth pass into a per-pixel CoC (defocus) alpha mask.

    coc = clamp(|depth - focus_distance| / aperture, 0, 1)

    Higher coc → more out-of-focus. Output shape: MASK [B,H,W] in [0,1].
    The depth pass may be a 1-channel or 3-channel IMAGE; if 3-channel,
    the R channel is used.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "depth": ("IMAGE",),
                "focus_distance": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 100.0, "step": 0.001}),
                "aperture": ("FLOAT", {"default": 0.1, "min": 0.001, "max": 100.0, "step": 0.001}),
            },
            "optional": {
                "invert": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Invert the focus mask (1 = in focus, 0 = defocus).",
                }),
                "depth_channel": (["R", "G", "B", "luma"], {"default": "R"}),
            },
        }

    RETURN_TYPES = ("MASK", "MASK")
    RETURN_NAMES = ("coc_mask", "in_focus_mask")
    FUNCTION = "compute"
    CATEGORY = "MaskEditControl/Render"
    DESCRIPTION = "Convert depth pass → CoC mask (defocus alpha) and in-focus mask."

    def compute(
        self,
        depth: torch.Tensor,
        focus_distance: float,
        aperture: float,
        invert: bool = False,
        depth_channel: str = "R",
    ):
        if depth.dim() != 4:
            raise ValueError(f"depth must be IMAGE [B,H,W,C], got shape {tuple(depth.shape)}")
        if depth.shape[-1] >= 3 and depth_channel == "luma":
            d = (0.2126 * depth[..., 0] + 0.7152 * depth[..., 1] + 0.0722 * depth[..., 2])
        else:
            idx = {"R": 0, "G": 1, "B": 2}.get(depth_channel, 0)
            d = depth[..., min(idx, depth.shape[-1] - 1)]
        coc = ((d - focus_distance).abs() / max(aperture, 1e-6)).clamp(0.0, 1.0)
        if invert:
            coc = 1.0 - coc
        in_focus = 1.0 - coc
        return (coc, in_focus)


NODE_CLASS_MAPPINGS = {
    "MergeRenderPassesMEC": MergeRenderPassesMEC,
    "DepthOfFieldMaskMEC": DepthOfFieldMaskMEC,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MergeRenderPassesMEC": "Merge Render Passes (MEC)",
    "DepthOfFieldMaskMEC": "Depth-of-Field Mask (MEC)",
}
