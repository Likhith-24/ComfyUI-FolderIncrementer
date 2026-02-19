"""
MaskTransformXY – Independent X/Y axis mask erode, expand (dilate), blur, 
offset, and feather with sub-pixel precision.
"""

import torch
import torch.nn.functional as F
import numpy as np


class MaskTransformXY:
    """Erode / expand a mask independently on X and Y axes with optional
    Gaussian blur, feathering, offset, and threshold controls."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "expand_x": ("INT", {"default": 0, "min": -512, "max": 512, "step": 1,
                                      "tooltip": "Positive = dilate horizontally, Negative = erode horizontally"}),
                "expand_y": ("INT", {"default": 0, "min": -512, "max": 512, "step": 1,
                                      "tooltip": "Positive = dilate vertically, Negative = erode vertically"}),
                "blur_x": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 128.0, "step": 0.5,
                                      "tooltip": "Gaussian blur sigma along X axis"}),
                "blur_y": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 128.0, "step": 0.5,
                                      "tooltip": "Gaussian blur sigma along Y axis"}),
                "offset_x": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": 1,
                                      "tooltip": "Shift mask horizontally (pixels)"}),
                "offset_y": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": 1,
                                      "tooltip": "Shift mask vertically (pixels)"}),
                "feather": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 128.0, "step": 0.5,
                                       "tooltip": "Feather (smooth edge) radius after transform"}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                                         "tooltip": "Binarize mask after morph ops (0 = keep soft)"}),
                "invert": ("BOOLEAN", {"default": False, "tooltip": "Invert output mask"}),
            },
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "transform"
    CATEGORY = "MaskEditControl/Transform"
    DESCRIPTION = "Erode/expand mask independently on X & Y with blur, offset, feather, and threshold."

    # ── helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _gauss_kernel_1d(sigma: float, device: torch.device) -> torch.Tensor:
        """Create a 1-D Gaussian kernel."""
        if sigma <= 0:
            return torch.ones(1, device=device)
        radius = int(3 * sigma + 0.5)
        if radius < 1:
            radius = 1
        size = 2 * radius + 1
        x = torch.arange(size, dtype=torch.float32, device=device) - radius
        kernel = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel /= kernel.sum()
        return kernel

    @staticmethod
    def _morph_1d(mask: torch.Tensor, amount: int, axis: str) -> torch.Tensor:
        """Dilate (amount > 0) or erode (amount < 0) along one axis."""
        if amount == 0:
            return mask
        abs_amount = abs(amount)
        # Build 1-D structuring element
        if axis == "x":
            kernel = torch.ones(1, 1, 1, 2 * abs_amount + 1, device=mask.device)
        else:
            kernel = torch.ones(1, 1, 2 * abs_amount + 1, 1, device=mask.device)
        pad_h = abs_amount if axis == "y" else 0
        pad_w = abs_amount if axis == "x" else 0
        m = mask.unsqueeze(0).unsqueeze(0) if mask.dim() == 2 else mask.unsqueeze(1)
        m_pad = F.pad(m, (pad_w, pad_w, pad_h, pad_h), mode="constant",
                       value=0.0 if amount > 0 else 1.0)
        if amount > 0:
            # max-pool style dilation via convolution then threshold
            out = F.conv2d(m_pad, kernel, padding=0)
            out = (out > 0.5).float()
        else:
            # erosion = invert → dilate → invert
            out = F.conv2d(1.0 - m_pad, kernel, padding=0)
            out = 1.0 - (out > 0.5).float()
        if mask.dim() == 2:
            out = out.squeeze(0).squeeze(0)
        else:
            out = out.squeeze(1)
        return out

    def _blur_separable(self, mask: torch.Tensor, sigma_x: float, sigma_y: float) -> torch.Tensor:
        if sigma_x <= 0 and sigma_y <= 0:
            return mask
        need_batch = mask.dim() == 2
        if need_batch:
            mask = mask.unsqueeze(0).unsqueeze(0)
        else:
            mask = mask.unsqueeze(1)
        # X blur
        if sigma_x > 0:
            kx = self._gauss_kernel_1d(sigma_x, mask.device)
            kx = kx.view(1, 1, 1, -1)
            px = kx.shape[-1] // 2
            mask = F.pad(mask, (px, px, 0, 0), mode="reflect")
            mask = F.conv2d(mask, kx, padding=0)
        # Y blur
        if sigma_y > 0:
            ky = self._gauss_kernel_1d(sigma_y, mask.device)
            ky = ky.view(1, 1, -1, 1)
            py = ky.shape[-2] // 2
            mask = F.pad(mask, (0, 0, py, py), mode="reflect")
            mask = F.conv2d(mask, ky, padding=0)
        if need_batch:
            mask = mask.squeeze(0).squeeze(0)
        else:
            mask = mask.squeeze(1)
        return mask

    @staticmethod
    def _offset(mask: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
        if dx == 0 and dy == 0:
            return mask
        need_batch = mask.dim() == 2
        if need_batch:
            mask = mask.unsqueeze(0)
        B, H, W = mask.shape
        out = torch.zeros_like(mask)
        # compute src/dst slices
        src_y0 = max(0, -dy)
        src_y1 = min(H, H - dy)
        dst_y0 = max(0, dy)
        dst_y1 = min(H, H + dy)
        src_x0 = max(0, -dx)
        src_x1 = min(W, W - dx)
        dst_x0 = max(0, dx)
        dst_x1 = min(W, W + dx)
        if src_y1 > src_y0 and src_x1 > src_x0:
            out[:, dst_y0:dst_y1, dst_x0:dst_x1] = mask[:, src_y0:src_y1, src_x0:src_x1]
        if need_batch:
            out = out.squeeze(0)
        return out

    # ── main ─────────────────────────────────────────────────────────────

    def transform(self, mask: torch.Tensor, expand_x: int, expand_y: int,
                  blur_x: float, blur_y: float, offset_x: int, offset_y: int,
                  feather: float, threshold: float, invert: bool):
        m = mask.clone()
        # morphological ops per axis
        m = self._morph_1d(m, expand_x, "x")
        m = self._morph_1d(m, expand_y, "y")
        # directional blur
        m = self._blur_separable(m, blur_x, blur_y)
        # offset
        m = self._offset(m, offset_x, offset_y)
        # feather (isotropic blur)
        if feather > 0:
            m = self._blur_separable(m, feather, feather)
        # threshold
        if threshold > 0:
            m = (m >= threshold).float()
        # invert
        if invert:
            m = 1.0 - m
        return (m.clamp(0.0, 1.0),)
