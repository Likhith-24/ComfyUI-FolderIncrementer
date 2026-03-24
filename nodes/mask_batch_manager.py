"""
MaskBatchManager – Manipulate mask batches: slice, concat, repeat, pick
specific frames, reverse, interleave, temporal smoothing.
"""

import torch
import torch.nn.functional as F


class MaskBatchManager:
    """Utility node for managing mask batches (video mask sequences).
    Supports slicing, concatenation, repeating, picking frames,
    reordering, and temporal smoothing for video workflows."""

    OPERATIONS = [
        "slice",           # Extract a range of frames
        "pick_frames",     # Pick specific frame indices
        "repeat",          # Repeat the batch N times
        "reverse",         # Reverse frame order
        "concat",          # Concatenate mask_b after mask
        "interleave",      # Interleave two mask batches
        "set_frame",       # Replace a specific frame
        "insert_frame",    # Insert a frame at position
        "remove_frame",    # Remove a frame at position
        "smooth_temporal", # Gaussian blur across time axis to reduce flicker
        "reduce_flicker",  # Median filter across time axis
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "operation": (cls.OPERATIONS, {"default": "slice"}),
                "param_a": ("INT", {"default": 0, "min": 0, "max": 99999,
                                     "tooltip": "Start frame (slice), frame index (pick/set/insert/remove), or repeat count"}),
                "param_b": ("INT", {"default": -1, "min": -1, "max": 99999,
                                     "tooltip": "End frame (slice, -1=end). Ignored for other ops."}),
                "frame_indices": ("STRING", {"default": "", "multiline": False,
                                              "tooltip": "Comma-separated frame indices for pick_frames mode"}),
            },
            "optional": {
                "mask_b": ("MASK", {"tooltip": "Second mask batch for concat/interleave/set_frame"}),
            },
        }

    RETURN_TYPES = ("MASK", "INT",)
    RETURN_NAMES = ("mask", "count",)
    FUNCTION = "manage"
    CATEGORY = "MaskEditControl/Batch"
    DESCRIPTION = "Manage mask batches: slice, pick, repeat, reverse, concat, interleave, insert, remove, temporal smoothing."

    def manage(self, mask, operation, param_a, param_b, frame_indices, mask_b=None):
        m = mask.clone()
        if m.dim() == 2:
            m = m.unsqueeze(0)

        if operation == "slice":
            end = param_b if param_b >= 0 else m.shape[0]
            end = min(end, m.shape[0])
            start = min(param_a, end)
            out = m[start:end]

        elif operation == "pick_frames":
            if frame_indices.strip():
                indices = [int(x.strip()) for x in frame_indices.split(",") if x.strip().isdigit()]
                indices = [i for i in indices if 0 <= i < m.shape[0]]
                if indices:
                    out = m[indices]
                else:
                    out = m[:1]
            else:
                idx = min(param_a, m.shape[0] - 1)
                out = m[idx:idx+1]

        elif operation == "repeat":
            count = max(1, param_a)
            out = m.repeat(count, 1, 1)

        elif operation == "reverse":
            out = m.flip(0)

        elif operation == "concat":
            if mask_b is not None:
                mb = mask_b.clone()
                if mb.dim() == 2:
                    mb = mb.unsqueeze(0)
                # Match spatial dims
                if mb.shape[1:] != m.shape[1:]:
                    mb = F.interpolate(mb.unsqueeze(1), size=m.shape[1:],
                                       mode="bilinear", align_corners=False).squeeze(1)
                out = torch.cat([m, mb], dim=0)
            else:
                out = m

        elif operation == "interleave":
            if mask_b is not None:
                mb = mask_b.clone()
                if mb.dim() == 2:
                    mb = mb.unsqueeze(0)
                if mb.shape[1:] != m.shape[1:]:
                    mb = F.interpolate(mb.unsqueeze(1), size=m.shape[1:],
                                       mode="bilinear", align_corners=False).squeeze(1)
                max_len = max(m.shape[0], mb.shape[0])
                parts = []
                for i in range(max_len):
                    if i < m.shape[0]:
                        parts.append(m[i:i+1])
                    if i < mb.shape[0]:
                        parts.append(mb[i:i+1])
                out = torch.cat(parts, dim=0)
            else:
                out = m

        elif operation == "set_frame":
            if mask_b is not None:
                mb = mask_b.clone()
                if mb.dim() == 2:
                    mb = mb.unsqueeze(0)
                if mb.shape[1:] != m.shape[1:]:
                    mb = F.interpolate(mb.unsqueeze(1), size=m.shape[1:],
                                       mode="bilinear", align_corners=False).squeeze(1)
                idx = min(param_a, m.shape[0] - 1)
                m[idx] = mb[0]
            out = m

        elif operation == "insert_frame":
            if mask_b is not None:
                mb = mask_b.clone()
                if mb.dim() == 2:
                    mb = mb.unsqueeze(0)
                if mb.shape[1:] != m.shape[1:]:
                    mb = F.interpolate(mb.unsqueeze(1), size=m.shape[1:],
                                       mode="bilinear", align_corners=False).squeeze(1)
                idx = min(param_a, m.shape[0])
                out = torch.cat([m[:idx], mb[:1], m[idx:]], dim=0)
            else:
                out = m

        elif operation == "remove_frame":
            idx = min(param_a, m.shape[0] - 1)
            out = torch.cat([m[:idx], m[idx+1:]], dim=0) if m.shape[0] > 1 else m

        elif operation == "smooth_temporal":
            # Gaussian blur along the time (batch) dimension to reduce frame-to-frame flicker
            # param_a = temporal radius (default to 2 if 0)
            radius = max(1, param_a) if param_a > 0 else 2
            if m.shape[0] <= 1:
                out = m
            else:
                sigma = radius / 2.0
                k_size = 2 * radius + 1
                t = torch.arange(k_size, dtype=torch.float32, device=m.device) - radius
                kernel = torch.exp(-0.5 * (t / max(sigma, 0.5)) ** 2)
                kernel = kernel / kernel.sum()
                # Reshape: (B, H, W) → (H*W, 1, B) for 1D conv along time
                B, H, W = m.shape
                flat = m.permute(1, 2, 0).reshape(H * W, 1, B)
                padded = F.pad(flat, (radius, radius), mode="reflect")
                smoothed = F.conv1d(padded, kernel.view(1, 1, -1))
                out = smoothed.reshape(H, W, B).permute(2, 0, 1).clamp(0, 1)

        elif operation == "reduce_flicker":
            # Median filter along time axis to remove single-frame outliers
            # param_a = window size (default to 3 if 0)
            window = max(3, param_a) if param_a > 0 else 3
            if window % 2 == 0:
                window += 1
            if m.shape[0] <= 2:
                out = m
            else:
                B, H, W = m.shape
                half = window // 2
                padded = F.pad(m.unsqueeze(0).unsqueeze(0),
                               (0, 0, 0, 0, half, half), mode="reflect").squeeze(0).squeeze(0)
                frames = []
                for i in range(B):
                    window_slice = padded[i:i + window]
                    frames.append(window_slice.median(dim=0).values)
                out = torch.stack(frames, dim=0)

        else:
            out = m

        if out.shape[0] == 0:
            out = m[:1]

        return (out, out.shape[0])
