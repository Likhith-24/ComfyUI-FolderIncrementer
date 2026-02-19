"""
ComfyUI-CustomNodePacks
=======================
A growing collection of custom nodes:
  - FolderIncrementer – auto-incrementing version strings
  - MaskEditControl  – pinpoint mask editing, SAM2/SAM3, per-axis erode/expand,
                       point editing, bbox tools, video mask propagation
"""

# ── FolderIncrementer nodes ────────────────────────────────────────────
from .folder_incrementer import (
    NODE_CLASS_MAPPINGS as _FOLDER_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as _FOLDER_DISPLAY,
)

# ── MaskEditControl nodes ─────────────────────────────────────────────
from .nodes.mask_transform_xy import MaskTransformXY
from .nodes.mask_draw_frame import MaskDrawFrame
from .nodes.mask_propagate_video import MaskPropagateVideo
from .nodes.mask_composite import MaskCompositeAdvanced
from .nodes.mask_preview import MaskPreviewOverlay
from .nodes.points_mask_editor import PointsMaskEditor
from .nodes.sam_model_loader import SAMModelLoaderMEC
from .nodes.sam_mask_generator import SAMMaskGeneratorMEC
from .nodes.mask_batch_manager import MaskBatchManager
from .nodes.mask_math import MaskMath
from .nodes.bbox_nodes import BBoxCreate, BBoxFromMask, BBoxToMask, BBoxPad, BBoxCrop
from .nodes.vitmatte_refiner import ViTMatteRefinerMEC
from .nodes.sam_vitmatte_pipeline import SAMViTMattePipelineMEC

_MEC_MAPPINGS = {
    "MaskTransformXY": MaskTransformXY,
    "MaskDrawFrame": MaskDrawFrame,
    "MaskPropagateVideo": MaskPropagateVideo,
    "MaskCompositeAdvanced": MaskCompositeAdvanced,
    "MaskPreviewOverlay": MaskPreviewOverlay,
    "PointsMaskEditor": PointsMaskEditor,
    "SAMModelLoaderMEC": SAMModelLoaderMEC,
    "SAMMaskGeneratorMEC": SAMMaskGeneratorMEC,
    "MaskBatchManager": MaskBatchManager,
    "MaskMath": MaskMath,
    "BBoxCreate": BBoxCreate,
    "BBoxFromMask": BBoxFromMask,
    "BBoxToMask": BBoxToMask,
    "BBoxPad": BBoxPad,
    "BBoxCrop": BBoxCrop,
    "ViTMatteRefinerMEC": ViTMatteRefinerMEC,
    "SAMViTMattePipelineMEC": SAMViTMattePipelineMEC,
}

_MEC_DISPLAY = {
    "MaskTransformXY": "Mask Transform XY (MEC)",
    "MaskDrawFrame": "Mask Draw Frame (MEC)",
    "MaskPropagateVideo": "Mask Propagate Video (MEC)",
    "MaskCompositeAdvanced": "Mask Composite Advanced (MEC)",
    "MaskPreviewOverlay": "Mask Preview Overlay (MEC)",
    "PointsMaskEditor": "Points Mask Editor (MEC)",
    "SAMModelLoaderMEC": "SAM Model Loader (MEC)",
    "SAMMaskGeneratorMEC": "SAM Mask Generator (MEC)",
    "MaskBatchManager": "Mask Batch Manager (MEC)",
    "MaskMath": "Mask Math (MEC)",
    "BBoxCreate": "BBox Create (MEC)",
    "BBoxFromMask": "BBox From Mask (MEC)",
    "BBoxToMask": "BBox To Mask (MEC)",
    "BBoxPad": "BBox Pad (MEC)",
    "BBoxCrop": "BBox Crop (MEC)",
    "ViTMatteRefinerMEC": "ViTMatte Edge Refiner (MEC)",
    "SAMViTMattePipelineMEC": "SAM + ViTMatte Pipeline (MEC)",
}

# ── Merge all mappings ────────────────────────────────────────────────
NODE_CLASS_MAPPINGS = {**_FOLDER_MAPPINGS, **_MEC_MAPPINGS}
NODE_DISPLAY_NAME_MAPPINGS = {**_FOLDER_DISPLAY, **_MEC_DISPLAY}

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
