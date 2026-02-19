"""
MaskPropagateVideo – Draw/define a mask on one frame and propagate it
across a video sequence.  Supports static copy, motion-compensated
propagation (optical flow), and SAM2 video-propagation mode.
"""

import torch
import torch.nn.functional as F
import numpy as np


class MaskPropagateVideo:
    """Take a mask defined on a single frame and apply / propagate it
    to every frame in an image batch (video sequence)."""

    PROPAGATION_MODES = [
        "static",           # Same mask on every frame
        "optical_flow",     # Warp mask using dense optical flow
        "sam2_video",       # Use SAM2's video propagator
        "fade",             # Fade mask in/out over time
        "scale_linear",     # Linearly scale mask over frames
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Video frames as image batch (B, H, W, C)"}),
                "mask": ("MASK", {"tooltip": "Source mask (single frame or batch)"}),
                "source_frame": ("INT", {"default": 0, "min": 0, "max": 99999,
                                          "tooltip": "Frame index where the mask was drawn"}),
                "mode": (cls.PROPAGATION_MODES, {"default": "static"}),
                "flow_threshold": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 50.0, "step": 0.5,
                                              "tooltip": "Optical flow magnitude threshold for mask warping"}),
                "fade_start": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                          "tooltip": "Mask opacity at source frame (for fade mode)"}),
                "fade_end": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                        "tooltip": "Mask opacity at last frame (for fade mode)"}),
                "bidirectional": ("BOOLEAN", {"default": True,
                                               "tooltip": "Propagate both forward and backward from source frame"}),
            },
            "optional": {
                "sam_model": ("SAM_MODEL", {"tooltip": "SAM2 model for video propagation mode"}),
                "points_json": ("STRING", {"default": "", "multiline": True,
                                            "tooltip": "Point prompts for SAM2 video mode"}),
            },
        }

    RETURN_TYPES = ("MASK", "IMAGE",)
    RETURN_NAMES = ("masks", "preview",)
    FUNCTION = "propagate"
    CATEGORY = "MaskEditControl/Video"
    DESCRIPTION = "Propagate a single-frame mask across all video frames using static copy, optical flow, SAM2 video, or fade modes."

    def propagate(self, images, mask, source_frame, mode, flow_threshold,
                  fade_start, fade_end, bidirectional, sam_model=None, points_json=""):

        B, H, W, C = images.shape
        source_frame = min(source_frame, B - 1)

        # Ensure mask is 2D (single frame)
        src_mask = mask
        if src_mask.dim() == 3:
            idx = min(source_frame, src_mask.shape[0] - 1)
            src_mask = src_mask[idx]
        if src_mask.dim() != 2:
            src_mask = src_mask.squeeze()

        # Resize mask if dimensions don't match
        if src_mask.shape[0] != H or src_mask.shape[1] != W:
            src_mask = F.interpolate(
                src_mask.unsqueeze(0).unsqueeze(0),
                size=(H, W), mode="bilinear", align_corners=False
            ).squeeze(0).squeeze(0)

        # ── Dispatch by mode ───────────────────────────────────────────
        if mode == "static":
            out_masks = self._static(src_mask, B)
        elif mode == "optical_flow":
            out_masks = self._optical_flow(images, src_mask, source_frame,
                                           flow_threshold, bidirectional)
        elif mode == "sam2_video":
            out_masks = self._sam2_video(images, src_mask, source_frame,
                                         sam_model, points_json)
        elif mode == "fade":
            out_masks = self._fade(src_mask, B, source_frame, fade_start, fade_end)
        elif mode == "scale_linear":
            out_masks = self._scale_linear(src_mask, B, source_frame, fade_start, fade_end)
        else:
            out_masks = self._static(src_mask, B)

        # Build preview: overlay mask on images
        preview = self._overlay_preview(images, out_masks)

        return (out_masks, preview)

    # ── Mode implementations ─────────────────────────────────────────

    @staticmethod
    def _static(mask, num_frames):
        """Copy the same mask to every frame."""
        return mask.unsqueeze(0).expand(num_frames, -1, -1).clone()

    @staticmethod
    def _fade(mask, num_frames, source_frame, start_opacity, end_opacity):
        """Linearly fade opacity across frames."""
        masks = mask.unsqueeze(0).expand(num_frames, -1, -1).clone()
        for i in range(num_frames):
            if num_frames == 1:
                alpha = start_opacity
            else:
                t = abs(i - source_frame) / max(1, num_frames - 1)
                alpha = start_opacity + (end_opacity - start_opacity) * t
            masks[i] = masks[i] * alpha
        return masks

    @staticmethod
    def _scale_linear(mask, num_frames, source_frame, start_scale, end_scale):
        """Linearly scale mask spatially across frames (zoom in/out effect)."""
        masks = torch.zeros(num_frames, mask.shape[0], mask.shape[1],
                            dtype=mask.dtype, device=mask.device)
        H, W = mask.shape
        for i in range(num_frames):
            if num_frames == 1:
                scale = start_scale
            else:
                t = abs(i - source_frame) / max(1, num_frames - 1)
                scale = start_scale + (end_scale - start_scale) * t
            if scale <= 0:
                continue
            new_h = max(1, int(H * scale))
            new_w = max(1, int(W * scale))
            scaled = F.interpolate(
                mask.unsqueeze(0).unsqueeze(0),
                size=(new_h, new_w), mode="bilinear", align_corners=False
            ).squeeze(0).squeeze(0)
            # Center-paste
            y_off = (H - new_h) // 2
            x_off = (W - new_w) // 2
            src_y0 = max(0, -y_off)
            src_x0 = max(0, -x_off)
            dst_y0 = max(0, y_off)
            dst_x0 = max(0, x_off)
            copy_h = min(new_h - src_y0, H - dst_y0)
            copy_w = min(new_w - src_x0, W - dst_x0)
            if copy_h > 0 and copy_w > 0:
                masks[i, dst_y0:dst_y0+copy_h, dst_x0:dst_x0+copy_w] = \
                    scaled[src_y0:src_y0+copy_h, src_x0:src_x0+copy_w]
        return masks

    def _optical_flow(self, images, src_mask, source_frame, threshold, bidirectional):
        """Warp mask using Farneback optical flow between consecutive frames."""
        B, H, W, C = images.shape
        masks = torch.zeros(B, H, W, dtype=src_mask.dtype, device=src_mask.device)
        masks[source_frame] = src_mask

        try:
            import cv2
        except ImportError:
            # Fallback to static if cv2 not available
            return self._static(src_mask, B)

        imgs_gray = []
        for i in range(B):
            frame = (images[i].cpu().numpy() * 255).astype(np.uint8)
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            imgs_gray.append(gray)

        # Forward propagation
        current_mask = src_mask.cpu().numpy()
        for i in range(source_frame + 1, B):
            flow = cv2.calcOpticalFlowFarneback(
                imgs_gray[i-1], imgs_gray[i],
                None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            h, w = flow.shape[:2]
            flow_map = np.zeros_like(flow)
            flow_map[:, :, 0] = np.arange(w) + flow[:, :, 0]
            flow_map[:, :, 1] = np.arange(h).reshape(-1, 1) + flow[:, :, 1]
            warped = cv2.remap(current_mask, flow_map[:, :, 0].astype(np.float32),
                               flow_map[:, :, 1].astype(np.float32),
                               cv2.INTER_LINEAR, borderValue=0)
            # Apply threshold
            if threshold > 0:
                mag = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
                warped[mag < threshold * 0.1] = current_mask[mag < threshold * 0.1]
            current_mask = warped
            masks[i] = torch.from_numpy(current_mask).to(src_mask.device)

        # Backward propagation
        if bidirectional and source_frame > 0:
            current_mask = src_mask.cpu().numpy()
            for i in range(source_frame - 1, -1, -1):
                flow = cv2.calcOpticalFlowFarneback(
                    imgs_gray[i+1], imgs_gray[i],
                    None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                h, w = flow.shape[:2]
                flow_map = np.zeros_like(flow)
                flow_map[:, :, 0] = np.arange(w) + flow[:, :, 0]
                flow_map[:, :, 1] = np.arange(h).reshape(-1, 1) + flow[:, :, 1]
                warped = cv2.remap(current_mask, flow_map[:, :, 0].astype(np.float32),
                                   flow_map[:, :, 1].astype(np.float32),
                                   cv2.INTER_LINEAR, borderValue=0)
                current_mask = warped
                masks[i] = torch.from_numpy(current_mask).to(src_mask.device)

        return masks

    def _sam2_video(self, images, src_mask, source_frame, sam_model, points_json):
        """Use SAM2's video propagation if available, otherwise fall back to static."""
        B, H, W, C = images.shape

        if sam_model is None:
            return self._static(src_mask, B)

        model_info = sam_model
        model = model_info["model"]
        model_type = model_info["model_type"]

        if model_type not in ("sam2", "sam2.1"):
            return self._static(src_mask, B)

        try:
            from sam2.sam2_video_predictor import SAM2VideoPredictor
            import json as json_mod
            import tempfile, os

            predictor = SAM2VideoPredictor(model)

            # SAM2 video predictor expects a video directory or frames
            # We'll create a temporary approach
            frames_np = []
            for i in range(B):
                frame = (images[i].cpu().numpy() * 255).astype(np.uint8)
                frames_np.append(frame)

            # Initialize with mask on source frame
            state = predictor.init_state(video_path=None, frames=frames_np)

            # Add mask prompt
            mask_np = src_mask.cpu().numpy()
            predictor.add_new_mask(state, frame_idx=source_frame,
                                   obj_id=1, mask=mask_np)

            # Parse optional points
            if points_json and points_json.strip():
                try:
                    pts = json_mod.loads(points_json)
                    if pts:
                        coords = np.array([[p["x"], p["y"]] for p in pts], dtype=np.float32)
                        labels = np.array([p.get("label", 1) for p in pts], dtype=np.int32)
                        predictor.add_new_points(state, frame_idx=source_frame,
                                                  obj_id=1, points=coords, labels=labels)
                except Exception:
                    pass

            # Propagate
            masks = torch.zeros(B, H, W, dtype=torch.float32, device=src_mask.device)
            for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
                if len(mask_logits) > 0:
                    m = (mask_logits[0] > 0).float().cpu()
                    if m.dim() == 3:
                        m = m[0]
                    if m.shape[0] != H or m.shape[1] != W:
                        m = F.interpolate(m.unsqueeze(0).unsqueeze(0),
                                          size=(H, W), mode="bilinear",
                                          align_corners=False).squeeze()
                    masks[frame_idx] = m

            return masks

        except (ImportError, Exception):
            # Fallback to optical flow or static
            try:
                return self._optical_flow(images, src_mask, source_frame, 2.0, True)
            except Exception:
                return self._static(src_mask, B)

    @staticmethod
    def _overlay_preview(images, masks):
        """Create a preview with green mask overlay on images."""
        B, H, W, C = images.shape
        preview = images.clone()
        color = torch.tensor([0.0, 1.0, 0.0], device=images.device)  # green
        alpha = 0.35
        for i in range(B):
            m = masks[i]
            if m.shape[0] != H or m.shape[1] != W:
                m = F.interpolate(
                    m.unsqueeze(0).unsqueeze(0),
                    size=(H, W), mode="bilinear", align_corners=False
                ).squeeze()
            mask_3d = m.unsqueeze(-1).expand(-1, -1, 3)
            overlay = color.unsqueeze(0).unsqueeze(0).expand(H, W, 3)
            preview[i] = preview[i] * (1 - mask_3d * alpha) + overlay * mask_3d * alpha
        return preview.clamp(0, 1)
