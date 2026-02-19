"""
MaskPreviewOverlay – Visualise masks overlaid on images with colour,
opacity, edge-highlight, and bbox drawing options.
"""

import torch
import torch.nn.functional as F


class MaskPreviewOverlay:
    """Generate a preview image with the mask overlaid on the source image.
    Supports custom overlay colour, edge highlight, bbox drawing, and
    side-by-side comparison."""

    DISPLAY_MODES = ["overlay", "mask_only", "side_by_side", "checkerboard", "edge_highlight"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "display_mode": (cls.DISPLAY_MODES, {"default": "overlay"}),
                "overlay_color_r": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "overlay_color_g": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "overlay_color_b": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "opacity": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.01}),
                "edge_width": ("INT", {"default": 2, "min": 0, "max": 20, "step": 1,
                                        "tooltip": "Edge contour width in pixels"}),
                "show_bbox": ("BOOLEAN", {"default": False, "tooltip": "Draw bounding box of mask region"}),
                "bbox_color_r": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "bbox_color_g": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "bbox_color_b": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "bbox": ("BBOX", {"tooltip": "External bbox to draw (overrides auto-detect)"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("preview",)
    FUNCTION = "preview"
    CATEGORY = "MaskEditControl/Preview"
    OUTPUT_NODE = True
    DESCRIPTION = "Overlay mask on image with customisable colour, opacity, edge highlight, and bbox display."

    def preview(self, image, mask, display_mode, overlay_color_r, overlay_color_g,
                overlay_color_b, opacity, edge_width, show_bbox,
                bbox_color_r, bbox_color_g, bbox_color_b, bbox=None):

        B, H, W, C = image.shape
        m = mask.clone()
        if m.dim() == 2:
            m = m.unsqueeze(0)
        # Match batch
        if m.shape[0] < B:
            m = m.expand(B, -1, -1).clone()
        # Match spatial
        if m.shape[1] != H or m.shape[2] != W:
            m = F.interpolate(m.unsqueeze(1), size=(H, W),
                              mode="bilinear", align_corners=False).squeeze(1)

        color = torch.tensor([overlay_color_r, overlay_color_g, overlay_color_b],
                             device=image.device, dtype=image.dtype)
        bbox_clr = torch.tensor([bbox_color_r, bbox_color_g, bbox_color_b],
                                device=image.device, dtype=image.dtype)

        out = image.clone()

        for i in range(B):
            frame = out[i]            # (H, W, C)
            mi = m[i]                 # (H, W)

            if display_mode == "overlay":
                mask_3d = mi.unsqueeze(-1)
                overlay = color.view(1, 1, 3).expand(H, W, 3)
                frame = frame * (1 - mask_3d * opacity) + overlay * mask_3d * opacity

            elif display_mode == "mask_only":
                frame = mi.unsqueeze(-1).expand(H, W, 3)

            elif display_mode == "side_by_side":
                mask_vis = mi.unsqueeze(-1).expand(H, W, 3)
                frame = torch.cat([frame, mask_vis], dim=1)  # wider

            elif display_mode == "checkerboard":
                checker = self._checkerboard(H, W, 16, image.device)
                bg = checker.unsqueeze(-1).expand(H, W, 3) * 0.5 + 0.25
                mask_3d = mi.unsqueeze(-1)
                frame = frame * mask_3d + bg * (1 - mask_3d)

            elif display_mode == "edge_highlight":
                edge = self._detect_edge(mi, edge_width)
                edge_3d = edge.unsqueeze(-1)
                overlay = color.view(1, 1, 3).expand(H, W, 3)
                frame = frame * (1 - edge_3d) + overlay * edge_3d

            # Edge overlay for non-edge modes
            if display_mode != "edge_highlight" and edge_width > 0:
                edge = self._detect_edge(mi, edge_width)
                edge_3d = edge.unsqueeze(-1)
                overlay_edge = color.view(1, 1, 3).expand(H, W, 3)
                frame = frame * (1 - edge_3d) + overlay_edge * edge_3d

            # BBox overlay
            if show_bbox:
                box = self._get_bbox(mi, bbox)
                if box is not None:
                    frame = self._draw_bbox(frame, box, bbox_clr, 2)

            if display_mode == "side_by_side":
                # Need to handle differently – output is wider
                # Rebuild output for this batch item
                if i == 0:
                    out = torch.zeros(B, H, W * 2, C, device=image.device, dtype=image.dtype)
                out[i] = frame
            else:
                out[i] = frame

        return (out.clamp(0, 1),)

    @staticmethod
    def _detect_edge(mask, width):
        """Simple morphological edge detection."""
        if width <= 0:
            return torch.zeros_like(mask)
        m = mask.unsqueeze(0).unsqueeze(0)
        kernel = torch.ones(1, 1, 2*width+1, 2*width+1, device=mask.device)
        dilated = F.conv2d(F.pad(m, (width, width, width, width), mode="constant", value=0),
                           kernel, padding=0)
        dilated = (dilated > 0.5).float()
        eroded = F.conv2d(F.pad(m, (width, width, width, width), mode="constant", value=1),
                          kernel, padding=0)
        total = kernel.numel()
        eroded = (eroded >= total).float()
        edge = (dilated - eroded).squeeze(0).squeeze(0).clamp(0, 1)
        return edge

    @staticmethod
    def _checkerboard(h, w, size, device):
        y = torch.arange(h, device=device) // size
        x = torch.arange(w, device=device) // size
        return ((y.unsqueeze(1) + x.unsqueeze(0)) % 2).float()

    @staticmethod
    def _get_bbox(mask, external_bbox):
        if external_bbox is not None:
            return external_bbox
        coords = torch.nonzero(mask > 0.5, as_tuple=False)
        if coords.shape[0] == 0:
            return None
        y_min = int(coords[:, 0].min())
        y_max = int(coords[:, 0].max())
        x_min = int(coords[:, 1].min())
        x_max = int(coords[:, 1].max())
        return [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1]

    @staticmethod
    def _draw_bbox(frame, bbox, color, thickness):
        x, y, bw, bh = bbox
        H, W, C = frame.shape
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(W - 1, x + bw), min(H - 1, y + bh)
        t = thickness
        # Top
        frame[y1:y1+t, x1:x2, :] = color
        # Bottom
        frame[max(0,y2-t):y2, x1:x2, :] = color
        # Left
        frame[y1:y2, x1:x1+t, :] = color
        # Right
        frame[y1:y2, max(0,x2-t):x2, :] = color
        return frame
