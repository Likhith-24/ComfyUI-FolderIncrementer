"""
TrimapGeneratorMEC – Standalone trimap generation from segmentation masks.

Converts a coarse segmentation mask into a 3-region trimap:
  - Foreground (white, 1.0)
  - Background (black, 0.0)
  - Unknown boundary (gray, 0.5)

Designed to feed directly into ViTMatte for alpha matting.
"""

import torch
import torch.nn.functional as F
import numpy as np

from .utils import generate_trimap, compute_edge_band_np, HAS_CV2


class TrimapGeneratorMEC:
    """Generate a trimap from a segmentation mask.

    The unknown region width is controlled by edge_radius.
    Supports asymmetric inner/outer erosion for fine-tuned control.
    Post-processing can smooth the trimap edges for stable ViTMatte input.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK", {
                    "tooltip": "Coarse segmentation mask to convert to trimap",
                }),
                "edge_radius": ("INT", {
                    "default": 15, "min": 1, "max": 200, "step": 1,
                    "tooltip": "Width of the unknown boundary region in pixels",
                }),
                "inner_erosion": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 3.0, "step": 0.1,
                    "tooltip": (
                        "Scale factor for inner (foreground) erosion. "
                        "<1 = tighter fg region, >1 = wider fg region"
                    ),
                }),
                "outer_dilation": ("FLOAT", {
                    "default": 1.5, "min": 0.5, "max": 5.0, "step": 0.1,
                    "tooltip": (
                        "Scale factor for outer (background) dilation. "
                        ">1 = wider unknown band for better edge capture"
                    ),
                }),
                "smooth": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 20.0, "step": 0.5,
                    "tooltip": (
                        "Gaussian smoothing of the trimap boundaries. "
                        "Helps reduce staircasing artifacts. 0 = no smoothing."
                    ),
                }),
                "threshold": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Binarization threshold for the input mask",
                }),
            },
            "optional": {
                "image": ("IMAGE", {
                    "tooltip": (
                        "Reference image for edge-aware trimap generation. "
                        "When provided, unknown regions follow image edges."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("MASK", "MASK", "MASK",)
    RETURN_NAMES = ("trimap", "foreground", "unknown",)
    FUNCTION = "generate"
    CATEGORY = "MaskEditControl/Trimap"
    DESCRIPTION = (
        "Generate a trimap from a segmentation mask. "
        "White=foreground, black=background, gray=unknown boundary. "
        "Feed into ViTMatte for alpha matting."
    )

    def generate(self, mask, edge_radius, inner_erosion, outer_dilation,
                 smooth, threshold, image=None):

        # Handle batch dimension
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        B, H, W = mask.shape
        trimaps = []
        fg_masks = []
        unknown_masks = []

        for i in range(B):
            m = mask[i].cpu().numpy()

            # Apply threshold
            m_binary = (m > threshold).astype(np.float32)

            # Generate trimap
            trimap = generate_trimap(
                m_binary, edge_radius,
                inner_scale=inner_erosion,
                outer_scale=outer_dilation,
            )

            # Edge-aware refinement with image
            if image is not None and HAS_CV2:
                import cv2
                frame_idx = min(i, image.shape[0] - 1)
                img_np = (image[frame_idx].cpu().numpy() * 255).astype(np.uint8)
                trimap = self._edge_aware_trimap(trimap, img_np, edge_radius)

            # Smooth boundaries
            if smooth > 0 and HAS_CV2:
                import cv2
                # Only smooth the unknown region transitions
                k = max(3, int(smooth * 2)) | 1
                smoothed = cv2.GaussianBlur(trimap, (k, k), smooth)
                # Re-quantize: fg stays fg, bg stays bg, transitions become unknown
                fg_hard = (trimap > 0.9).astype(np.float32)
                bg_hard = (trimap < 0.1).astype(np.float32)
                trimap = fg_hard * 1.0 + (1 - fg_hard - bg_hard) * smoothed

            # Extract regions
            fg = (trimap > 0.9).astype(np.float32)
            unknown = ((trimap > 0.1) & (trimap < 0.9)).astype(np.float32)

            trimaps.append(torch.from_numpy(trimap))
            fg_masks.append(torch.from_numpy(fg))
            unknown_masks.append(torch.from_numpy(unknown))

        trimap_out = torch.stack(trimaps)
        fg_out = torch.stack(fg_masks)
        unknown_out = torch.stack(unknown_masks)

        return (trimap_out, fg_out, unknown_out)

    @staticmethod
    def _edge_aware_trimap(trimap, img_np, edge_radius):
        """Refine trimap unknown region to follow image edges."""
        if not HAS_CV2:
            return trimap

        import cv2

        # Detect edges in the image
        gray = cv2.cvtColor(img_np[:, :, :3], cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges_dilated = cv2.dilate(
            edges,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )

        # Where there are image edges in the unknown region, keep unknown
        # Where there are no edges, lean towards fg/bg based on proximity
        unknown = (trimap > 0.1) & (trimap < 0.9)
        edge_mask = (edges_dilated > 0).astype(np.float32)

        # Use distance transform for gradient
        fg_region = (trimap > 0.9).astype(np.uint8)
        dist_to_fg = cv2.distanceTransform(1 - fg_region, cv2.DIST_L2, 5)
        max_dist = max(1, dist_to_fg.max())
        dist_normalized = dist_to_fg / max_dist

        # In unknown region: if near fg and no strong edge, bias towards fg
        confidence = np.clip(1.0 - dist_normalized * 2, 0, 1)
        refined = trimap.copy()
        # Only modify unknown region
        unknown_float = unknown.astype(np.float32)
        refined = refined * (1 - unknown_float) + (
            confidence * 0.8 + 0.1
        ) * unknown_float

        return np.clip(refined, 0, 1).astype(np.float32)
