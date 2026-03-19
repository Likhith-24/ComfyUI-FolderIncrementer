# Example Workflows for ComfyUI-CustomNodePacks

These are sample ComfyUI workflows demonstrating the nodes in this pack. Drag and drop the `.json` files directly into ComfyUI to load them.

## Workflows

### 1. `basic_sam_segmentation.json`
**SAM Point-Based Segmentation** — The core workflow for most users.
- Load an image → Place points in the interactive editor → SAM generates a precise mask → Overlay preview
- Demonstrates: `SAM Model Loader`, `Points Mask Editor`, `SAM Mask Generator`, `Mask Preview Overlay`

### 2. `sam_vitmatte_pipeline.json`
**Compositing-Grade Alpha Matting** — SAM → ViTMatte end-to-end for hair, fur, lace detail.
- Load image → Points + BBox → SAM + ViTMatte Pipeline → Refined matte with edge detail
- Demonstrates: `SAM Model Loader`, `Points Mask Editor`, `SAM + ViTMatte Pipeline`

### 3. `mask_editing_toolkit.json`
**Mask Transform & Composite** — Manipulate masks with math, transforms, shapes, and compositing.
- Create masks via shapes → Transform XY → Composite operations → Preview
- Demonstrates: `Mask Draw Frame`, `Mask Transform XY`, `Mask Composite Advanced`, `Mask Math`, `Mask Preview Overlay`

### 4. `bbox_pipeline.json`
**BBox Workflow** — Create, extract, pad, and crop with bounding boxes.
- Load image → BBox from mask → Pad → Crop region → Preview
- Demonstrates: `BBox Create`, `BBox From Mask`, `BBox Pad`, `BBox Crop`, `BBox To Mask`

### 5. `video_mask_propagation.json`
**Video Masking** — Draw a mask on one frame, propagate it across the entire video.
- Load video → Extract frame → Editor → Propagate mask → Preview all frames
- Demonstrates: `Video Frame Extractor`, `Points Mask Editor`, `Mask Propagate Video`

## How to Use

1. Open ComfyUI in your browser (`http://127.0.0.1:8188`)
2. Drag a `.json` file from this folder into the ComfyUI canvas
3. Connect a `Load Image` node or use the default setup
4. Click **Queue Prompt** to run

## Node Categories

| Category | Nodes |
|----------|-------|
| **Editor** | Points Mask Editor |
| **SAM** | SAM Model Loader, SAM Mask Generator |
| **Pipeline** | SAM + ViTMatte Pipeline |
| **Refinement** | ViTMatte Edge Refiner, Trimap Generator |
| **Transform** | Mask Transform XY |
| **Draw** | Mask Draw Frame |
| **Composite** | Mask Composite Advanced |
| **Math** | Mask Math |
| **Batch** | Mask Batch Manager |
| **Preview** | Mask Preview Overlay |
| **BBox** | BBox Create, BBox From Mask, BBox To Mask, BBox Pad, BBox Crop |
| **Video** | Video Frame Extractor, Mask Propagate Video |
| **Segmentation** | Unified Segmentation |
| **Matting** | Matting Node |
| **Utils** | Folder Version Incrementer, Folder Version Check, Folder Version Set |
