"""
VideoFrameExtractorMEC – Detect image vs. video input and extract frames.

Handles both single images and video batches transparently:
  - Single image (B=1): passes through unchanged
  - Video batch (B>1): extracts a specific frame (default: first)

This lets downstream nodes always receive a single IMAGE tensor,
regardless of whether the user connected an image loader or video loader.
"""

import torch


class VideoFrameExtractorMEC:
    """Extract a single frame from a video batch, or pass through a single image."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Image batch (B,H,W,C). Single images pass through; video batches select one frame.",
                }),
                "frame_index": ("INT", {
                    "default": 0, "min": 0, "max": 999999, "step": 1,
                    "tooltip": "Which frame to extract (0-based). Clamped to batch length.",
                }),
                "mode": (["specific_frame", "first", "last", "middle"], {
                    "default": "first",
                    "tooltip": (
                        "Frame selection mode:\n"
                        "first: always frame 0\n"
                        "last: final frame\n"
                        "middle: middle frame (B//2)\n"
                        "specific_frame: use frame_index value"
                    ),
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "BOOLEAN",)
    RETURN_NAMES = ("frame", "total_frames", "is_video",)
    FUNCTION = "extract"
    CATEGORY = "MaskEditControl/Video"
    DESCRIPTION = (
        "Extract a single frame from a video batch. "
        "Single images pass through unchanged. "
        "Reports total frame count and whether input is a video batch."
    )

    def extract(self, images, frame_index=0, mode="first"):
        B = images.shape[0]
        is_video = B > 1

        if mode == "first":
            idx = 0
        elif mode == "last":
            idx = B - 1
        elif mode == "middle":
            idx = B // 2
        else:  # specific_frame
            idx = min(frame_index, B - 1)

        frame = images[idx].unsqueeze(0)  # (1, H, W, C)
        return (frame, B, is_video)


NODE_CLASS_MAPPINGS = {
    "VideoFrameExtractorMEC": VideoFrameExtractorMEC,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoFrameExtractorMEC": "Video Frame Extractor (MEC)",
}
