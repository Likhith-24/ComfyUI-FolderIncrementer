<p align="center">
  <img src="https://i.pinimg.com/736x/5d/52/19/5d5219ef27c1530bea027ff95d352e05.jpg" alt="MEC Logo" width="120" />
</p>

<h1 align="center">ComfyUI-CustomNodePacks</h1>

<p align="center">
  <strong>MaskEditControl (MEC) + FolderIncrementer</strong><br/>
  Production-grade mask editing, SAM2/SAM3 segmentation, alpha matting, and auto-versioned file output for ComfyUI.
</p>

<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#node-reference">Node Reference</a> •
  <a href="#workflows">Workflows</a> •
  <a href="#troubleshooting">Troubleshooting</a> •
  <a href="#license">License</a>
</p>

---

## Overview

**ComfyUI-CustomNodePacks** ships **26 nodes** organized into two packs:

| Pack | Nodes | Purpose |
|------|------:|---------|
| **MaskEditControl (MEC)** | 23 | Pinpoint mask editing, SAM2/3 segmentation, ViTMatte alpha matting, video propagation, compositing tools |
| **FolderIncrementer** | 3 | Filesystem-safe auto-versioned output (`v001`, `v002`, …) |

All nodes are prefixed with **(MEC)** in the ComfyUI node menu for easy discovery.

---

## Installation

### 1. Clone the repo

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Likhith-24/ComfyUI-CustomNodePacks.git
```

Or install via **ComfyUI Manager** → search "CustomNodePacks".

### 2. Install dependencies

> **⚠️ IMPORTANT: Check your existing package versions before installing.**
>
> ComfyUI bundles its own `torch`, `torchvision`, `numpy`, and `Pillow`. Blindly running `pip install -r requirements.txt` can **overwrite** them with incompatible versions and break your ComfyUI installation.
>
> **Recommended approach:** Open `requirements.txt`, comment out anything already installed, and only install what you're missing:
>
> ```bash
> # Check what you already have:
> pip list | grep -i "opencv\|scipy\|safetensors\|transformers"
>
> # Then install only what's missing:
> pip install opencv-python>=4.7.0 scipy>=1.10.0 safetensors>=0.4.0
> ```

The `requirements.txt` lists:

| Package | Required? | Purpose |
|---------|-----------|---------|
| `opencv-python` | **Yes** | Edge detection, morphological ops, guided filter |
| `scipy` | **Yes** | Gaussian filters, signal processing |
| `safetensors` | **Yes** | Safe model weight loading |
| `transformers` | Optional | ViTMatte neural matting (best quality edges) |
| `pillow` | Optional | Image I/O for ViTMatte |

Core packages (`torch`, `torchvision`, `numpy`) are already provided by ComfyUI — do **not** reinstall them.

### 3. Place SAM model checkpoints

Download SAM / SAM2 / SAM2.1 / SAM3 weights and place them in:

```
ComfyUI/models/sams/        ← SAM ViT-H/L/B, SAM3
ComfyUI/models/sam2/         ← SAM2, SAM2.1
```

| Model | File | Source |
|-------|------|--------|
| SAM ViT-H | `sam_vit_h_4b8939.pth` | [Meta AI](https://github.com/facebookresearch/segment-anything) |
| SAM2.1 Large | `sam2.1_hiera_large.pt` | [Meta AI](https://github.com/facebookresearch/sam2) |
| SAM3 | `sam3_hiera_large.pt` | [Meta AI](https://github.com/facebookresearch/sam3) |

The **SAM Model Loader** node auto-detects model type from the filename.

### 4. (Optional) ViTMatte matting

For the highest-quality edge refinement (hair, fur, glass, lace):

```bash
pip install transformers pillow
```

The ViTMatte model (~400 MB) auto-downloads from HuggingFace on first use. For offline setups, place model files in `ComfyUI/models/vitmatte/`.

### 5. Restart ComfyUI

Look for `[MEC] Loaded 23 MaskEditControl nodes.` in the console to confirm.

---

## Node Reference

### Legend

All nodes appear in the ComfyUI menu under **MaskEditControl/** categories.

| Symbol | Meaning |
|--------|---------|
| ★★★★★ | Best quality (may need optional deps) |
| ★★★★☆ | Great quality |
| ★★★☆☆ | Good / fast fallback |

---

### Segmentation & SAM

#### SAM Model Loader (MEC)

Loads SAM / SAM2 / SAM2.1 / SAM3 checkpoints with optional VRAM offload.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | — | Checkpoint filename (auto-listed from `models/sams/` and `models/sam2/`) |
| `model_type` | `auto` | Force type: `sam_vit_h`, `sam_vit_l`, `sam_vit_b`, `sam2`, `sam2.1`, `sam3`, or `auto` |
| `device` | `cuda` | `cuda` / `cpu` |
| `offload_to_cpu` | `false` | Keep model on CPU, move to GPU only during inference (saves ~2–4 GB VRAM) |
| `dtype` | `float16` | `float16` / `bfloat16` / `float32` |

**Output:** `SAM_MODEL` — connect to SAM Mask Generator or SAM + ViTMatte Pipeline.

---

#### SAM Mask Generator (MEC)

Runs SAM/SAM2/SAM3 inference with point + bbox prompts, iterative refinement, and auto-negative point sampling.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `image` | — | Input image |
| `sam_model` | — | From SAM Model Loader |
| `points_json` | `""` | `[{"x":int, "y":int, "label":1}, ...]` |
| `bbox_json` | `""` | `[x1, y1, x2, y2]` |
| `multimask` | `true` | Return 3 candidate masks |
| `mask_index` | `0` | Which candidate (0–2) to use |
| `refine_iterations` | `1` | Iterative SAM passes (each tightens boundaries) |
| `auto_negative_points` | `0` | Sample N negative points from outside the mask |

**Outputs:** `masks`, `best_score`, `info`

---

#### Unified Segmentation (MEC)

One-node dispatcher for **SAM2/2.1, SAM3, SeC, VideoMaMa** with automatic image vs. video detection.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | — | Model selector |
| `points_json` / `bbox_json` | `""` | Point and box prompts |
| `precision` | `fp16` | `fp16` / `bf16` / `fp32` |
| `attention_mode` | `auto` | `sdpa` / `flash_attn` / `sage_attn` / `xformers` |
| `text_prompt` | — | (optional) Text prompt for grounding models |

**Outputs:** `masks`, `best_score`, `info`

---

#### SAM + ViTMatte Pipeline (MEC)

End-to-end pipeline: **SAM coarse segmentation → iterative refinement → neural alpha matting** in a single node. Best possible masking quality.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sam_iterations` | `2` | Iterative SAM passes |
| `refine_method` | `auto` | `auto` / `vitmatte` / `multi_scale_guided` / `color_aware` / `laplacian_blend` |
| `edge_radius` | `12` | Pixels around edges to refine |
| `detail_preservation` | `0.85` | Fine detail retention (hair, fur, lace) |
| `edge_contrast` | `1.0` | Edge sharpness boost (>1 = sharper) |
| `fill_holes` | `true` | Fill holes inside mask |
| `remove_small_regions` | `64` | Remove noise < N pixels |

**Outputs:** `refined_mask`, `coarse_mask`, `edge_mask`, `preview`, `detected_bbox`, `score`, `info`

**Pipeline stages:**
1. SAM coarse mask from point/bbox prompts
2. Iterative refinement (re-run SAM with mask-derived prompts)
3. Edge-aware matting (ViTMatte / guided filter / LAB color)
4. Edge contrast boost
5. Post-processing (hole fill, small region removal)

---

### Alpha Matting

#### Matting Node (MEC)

Unified alpha matting with **7 backends**. Takes a coarse mask and returns compositing-grade alpha.

| Backend | Quality | Requires | Best for |
|---------|---------|----------|----------|
| `vitmatte_small` / `vitmatte_base` | ★★★★★ | `transformers` | Hair, fur, glass, transparency |
| `matanyone2` | ★★★★★ | Model download | Video matting, temporal consistency |
| `rvm_mobilenetv3` / `rvm_resnet50` | ★★★★☆ | Model download | Real-time video matting |
| `cutie` | ★★★★☆ | Model download | Video object cutout |
| `sam_hq` | ★★★☆☆ | Model download | Quick high-quality mask |

| Parameter | Default | Description |
|-----------|---------|-------------|
| `backend` | `auto` | Backend selection (auto picks best available) |
| `edge_radius` | `15` | Trimap unknown-band width |
| `erode_dilate` | `0` | Pre-process mask morphology (−50 to +50) |
| `n_warmup` | `5` | Warmup frames for video backends |

**Outputs:** `rgb` (premultiplied), `alpha_mask`

---

#### ViTMatte Edge Refiner (MEC)

Standalone edge refinement — feed any coarse mask and get clean edges.

| Method | Quality | Requires | Best for |
|--------|---------|----------|----------|
| `vitmatte` | ★★★★★ | `transformers` | Hair, fur, glass, complex edges |
| `multi_scale_guided` | ★★★★☆ | `opencv-python` | General high-quality |
| `color_aware` | ★★★★☆ | `opencv-python` | Challenging lighting |
| `guided_filter` | ★★★☆☆ | `opencv-python` | Fast good-quality |
| `laplacian_blend` | ★★★☆☆ | `opencv-python` | Smooth blending |
| `gaussian_blur` | ★★☆☆☆ | (none) | Simple fallback |

---

#### Trimap Generator (MEC)

Generates a 3-region trimap (white = foreground, black = background, gray = unknown) for ViTMatte input.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `edge_radius` | `15` | Width of the unknown boundary in pixels |
| `inner_erosion` | `1.0` | Foreground erosion scale (<1 tighter, >1 wider) |
| `outer_dilation` | `1.5` | Background dilation scale |
| `smooth` | `0.0` | Gaussian smoothing of boundaries |
| `threshold` | `0.5` | Binarization threshold |

**Outputs:** `trimap`, `foreground`, `unknown`

---

### Mask Editing & Transform

#### Mask Transform XY (MEC)

Independent per-axis mask manipulation: erode/expand, directional blur, offset, feather, threshold, invert.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `expand_x` / `expand_y` | `0` | Per-axis morphological expand (negative = erode) |
| `blur_x` / `blur_y` | `0.0` | Per-axis directional Gaussian blur |
| `offset_x` / `offset_y` | `0` | Translate mask in pixels |
| `feather` | `0.0` | Isotropic edge feathering |
| `threshold` | `0.0` | Binarization threshold (0 = disabled) |
| `invert` | `false` | Invert the mask |

---

#### Mask Draw Frame (MEC)

Draw geometric shapes onto masks with feathering and blend modes.

| Shape | Parameters |
|-------|-----------|
| `circle` | `cx`, `cy`, `radius` |
| `rectangle` | `x`, `y`, `w`, `h` |
| `ellipse` | `cx`, `cy`, `rx`, `ry` |
| `polygon` | `points` array of `[x, y]` |
| `line` | `x1`, `y1`, `x2`, `y2`, `thickness` |

| Parameter | Default | Description |
|-----------|---------|-------------|
| `shape` | — | Shape type |
| `shape_data` | — | JSON with shape parameters |
| `intensity` | `1.0` | Shape opacity |
| `feather` | `0.0` | Edge softness |
| `blend_mode` | `set` | `set` / `add` / `multiply` / `max_` / `min_` |

---

#### Mask Composite Advanced (MEC)

Combine two masks with compositing operations.

| Operation | Description |
|-----------|-------------|
| `union` | Combine both (max) |
| `intersect` | Overlap only (min) |
| `subtract` | A minus B |
| `xor` | Exclusive — one or the other, not both |
| `blend` | Weighted average |
| `min` / `max` | Per-pixel min / max |
| `difference` | Absolute difference |

Automatically resizes masks to match if spatial dimensions differ.

---

#### Mask Math (MEC)

Mathematical operations on a single mask.

| Operation | Description |
|-----------|-------------|
| `add_scalar` | Add constant value |
| `multiply_scalar` | Multiply by value |
| `power` | Raise to power |
| `invert` | `1 − mask` |
| `clamp` | Clamp to `[value_a, value_b]` range |
| `remap_range` | Remap from `[value_a, value_b]` → `[0, 1]` |
| `quantize` | Snap to N discrete levels |
| `threshold_hysteresis` | Dual-threshold with connected regions |
| `gamma` | Gamma correction |
| `contrast` | Contrast adjustment |
| `abs_diff_from_value` | Absolute difference from value |

All outputs are clamped to `[0, 1]`.

---

### Batch & Video

#### Mask Batch Manager (MEC)

Manipulate mask batches for video workflows.

| Operation | Description |
|-----------|-------------|
| `slice` | Extract frame range `[start:end]` |
| `pick_frames` | Select specific frames by index |
| `repeat` | Repeat mask N times |
| `reverse` | Reverse frame order |
| `concat` | Join two mask batches |
| `interleave` | Alternate frames from two batches |
| `insert` | Insert mask_b at position |
| `remove` | Remove frame at index |

---

#### Mask Propagate Video (MEC)

Draw mask on frame 1 → propagate across all frames.

| Mode | Description |
|------|-------------|
| `static` | Same mask on every frame |
| `fade` | Linear fade to zero |
| `scale_linear` | Linear scale over time |
| `optical_flow` | Track mask using optical flow |
| `sam2_video` | Use SAM2 video predictor for tracking |

---

#### Video Frame Extractor (MEC)

Extract a single frame from a video batch.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mode` | `first` | `first` / `last` / `middle` / `specific_frame` |
| `frame_index` | `0` | Frame index for `specific_frame` mode (clamped to batch size) |

**Outputs:** `frame` (IMAGE), `total_frames` (INT), `is_video` (BOOLEAN)

---

### BBox Tools

Five nodes for bounding box manipulation:

| Node | Description |
|------|-------------|
| **BBox Create** | Manual `[x, y, width, height]` entry |
| **BBox From Mask** | Extract tight bbox from non-zero mask pixels with per-axis padding |
| **BBox To Mask** | Convert bbox to a rectangular mask |
| **BBox Pad** | Asymmetric padding (top/bottom/left/right) with canvas clamping |
| **BBox Crop** | Crop image + mask to bbox region |

All BBox nodes clamp outputs to valid canvas bounds — no negative dimensions, no out-of-bounds errors.

---

### Interactive Editor

#### Points Mask Editor (MEC)

Full-featured interactive canvas editor for placing points and bounding boxes directly on your image.

| Action | Effect |
|--------|--------|
| **Left click** | Add positive point (foreground) |
| **Right click** | Add negative point (background) |
| **CTRL + Left drag** | Draw positive bounding box (green) |
| **CTRL + Right drag** | Draw negative bounding box (red) |
| **Shift + Click** | Delete element under cursor |
| **Scroll wheel** | Adjust point radius |
| **CTRL + Scroll** | Zoom in/out |
| **Middle mouse drag** | Pan canvas |
| **Delete / Backspace** | Delete hovered element |
| **CTRL + Z / CTRL + Shift + Z** | Undo / Redo |
| **R** | Reset view |

**Toolbar:** Pill counters for +pts / −pts / bbox count / radius. Buttons for **✕ Pts**, **✕ BBox**, **✕ All**, **↶ Undo / Redo ↷**, **▣ Fit**.

**Outputs (8):**

| Output | Type | Description |
|--------|------|-------------|
| `mask` | MASK | Rendered points/bboxes mask |
| `positive_coords` | STRING | `[{"x":int,"y":int}, ...]` for SAM |
| `negative_coords` | STRING | `[{"x":int,"y":int}, ...]` for SAM |
| `bboxes` | BBOX | Positive bounding boxes |
| `neg_bboxes` | BBOX | Negative bounding boxes (SAM3) |
| `points_json` | STRING | Full point data for SAM Mask Generator |
| `bbox_json` | STRING | Primary bbox for SAM Mask Generator |
| `primary_bbox` | BBOX | `[x,y,w,h]` for BBox pipeline |

Connect a `reference_image` to see it as the editor background for precise placement.

---

### Preview

#### Mask Preview Overlay (MEC)

Visualize masks with 5 display modes:

| Mode | Description |
|------|-------------|
| `overlay` | Red-tinted mask overlaid on the image |
| `binary_mask` | Black & white mask only |
| `edge_only` | Show mask edges / contours |
| `side_by_side` | Original image next to masked version |
| `alpha_channel` | RGBA with alpha from mask |

Handles batch size mismatches automatically (expands or repeats to match).

---

### Utilities

#### Universal Reroute / Dot (MEC)

Nuke-style reroute dot that accepts **any** connection type (IMAGE, MASK, LATENT, STRING, INT, FLOAT, etc.). Use it to keep your workflow wires clean and organized.

#### Parameter History (MEC)

Tracks every parameter change across ComfyUI runs in a local SQLite database. Query with:

| Mode | Description |
|------|-------------|
| `all_history` | Full parameter history for last N runs |
| `last_run_diff` | What changed between the last two runs |
| `node_class_filter` | Filter history by node class name |

---

### FolderIncrementer

Filesystem-aware auto-versioning for output files.

| Node | Description |
|------|-------------|
| **Folder Version Incrementer** | Scans output directory for `v001`, `v002`, … and returns the next available version |
| **Folder Version Check** | Reports how many versions exist |
| **Folder Version Set** | Reserves version slots by creating placeholder directories |

**Key features:**
- **Filesystem-based** — no global counter file; scans the actual directory
- **Cancel-safe** — if you cancel mid-execution, no version is wasted
- **Extension-preserving** — `photo.png` → `photo/v001/photo.png`

**Outputs:**

| Output | Example | Purpose |
|--------|---------|---------|
| `version_string` | `v001` | Version tag |
| `version_number` | `1` | Integer version |
| `folder_name` | `photo` | Derived from source filename |
| `subfolder_path` | `photo/v001` | For Save Image subfolder |
| `filename_prefix` | `photo/v001/photo` | Without extension |
| `output_filename` | `photo/v001/photo.png` | Full output path |

---

## Workflows

### Best Quality Masking (Single Image)

```
[SAM Model Loader]          [Load Image]
  offload_to_cpu: true         ↓
  dtype: float16          [Points Mask Editor]
        ↓                    ↓ points_json, bbox_json
[SAM + ViTMatte Pipeline] ←─┘
  sam_iterations: 2–3
  refine_method: auto
  detail_preservation: 0.85
  edge_contrast: 1.2
        ↓
  refined_mask → compositing
```

### Fast Iteration

```
[SAM Mask Generator]  →  [ViTMatte Edge Refiner]
  refine_iterations: 2       method: multi_scale_guided
```

### Video Masking

```
[Load Video]  →  [Video Frame Extractor]  →  [Points Mask Editor]
                   mode: first                      ↓
                                            [SAM Mask Generator]
                                                    ↓
                                            [Mask Propagate Video]
                                              mode: sam2_video
                                                    ↓
                                              per-frame masks
```

### Alpha Matting (From Any Coarse Mask)

```
[Any Segmentation Node]  →  [Matting Node]
                               backend: auto
                               edge_radius: 15
                                    ↓
                              alpha_mask → compositing
```

### SeC + MEC (Video Object Segmentation)

[SeC](https://github.com/9nate-drake/Comfyui-SecNodes) handles tracking, MEC handles edge quality:

```
[SeC Video Segmentation]  →  [ViTMatte Edge Refiner (MEC)]  →  refined masks
```

Install SeC: `cd ComfyUI/custom_nodes && git clone https://github.com/9nate-drake/Comfyui-SecNodes`

---

## VRAM Management

All nodes are designed to work on **low-end GPUs** (4–6 GB VRAM):

- **VRAM monitoring** — model loading logs a warning when free VRAM drops below 2 GB
- **OOM recovery** — if GPU runs out of memory during model loading, you get an actionable error message instead of a crash
- **Offload to CPU** — SAM Model Loader has `offload_to_cpu` to keep models in system RAM
- **Auto memory cleanup** — matting nodes run `gc.collect()` + `torch.cuda.empty_cache()` after inference
- **dtype control** — use `float16` or `bfloat16` to halve VRAM usage
- **Torch-only fallbacks** — if `opencv-python` is not installed, all nodes fall back to pure PyTorch implementations (slightly slower but functional)

---

## Running Tests

The test suite covers all pure-tensor nodes (no GPU models required):

```bash
# Using the test runner (recommended):
python run_tests.py

# Or directly with pytest:
python -m pytest tests/test_nodes.py -v --tb=short
```

78 tests covering: MaskMath, MaskComposite, MaskDrawFrame, MaskBatchManager, BBox nodes, MaskTransformXY, MaskPropagateVideo, MaskPreview, PointsMaskEditor, TrimapGenerator, VideoFrameExtractor, SAMViTMattePipeline.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: cv2` | `pip install opencv-python>=4.7.0` — nodes will use torch fallbacks but cv2 gives better quality |
| `ModuleNotFoundError: transformers` | `pip install transformers pillow` — only needed for ViTMatte matting |
| Torch version mismatch after `pip install -r requirements.txt` | Reinstall your ComfyUI torch: check [PyTorch install page](https://pytorch.org/get-started/locally/) for your CUDA version |
| SAM model not found | Place `.pth` / `.pt` / `.safetensors` in `ComfyUI/models/sams/` or `ComfyUI/models/sam2/` |
| CUDA out of memory | Enable `offload_to_cpu` in SAM Model Loader, use `float16` dtype, reduce image resolution |
| Nodes not showing in menu | Check console for `[MEC] Loaded` message. If missing, check for import errors in the console output |
| ViTMatte download fails | Download manually from [HuggingFace](https://huggingface.co/hustvl/vitmatte-small-composition-1k) → place in `ComfyUI/models/vitmatte/` |

---

## Project Structure

```
ComfyUI-CustomNodePacks/
├── __init__.py                 # Node registration & mappings
├── folder_incrementer.py       # FolderIncrementer nodes
├── conftest.py                 # Pytest root configuration
├── run_tests.py                # Test runner script
├── pyproject.toml              # Package metadata
├── requirements.txt            # Dependencies
├── js/
│   ├── folder_incrementer.js   # Frontend for FolderIncrementer
│   └── points_bbox_editor.js   # Interactive canvas editor
├── nodes/
│   ├── mask_transform_xy.py    # Per-axis mask transform
│   ├── mask_draw_frame.py      # Shape drawing
│   ├── mask_propagate_video.py # Video mask propagation
│   ├── mask_composite.py       # Mask compositing ops
│   ├── mask_preview.py         # Mask visualization
│   ├── mask_math.py            # Mathematical mask ops
│   ├── mask_batch_manager.py   # Batch manipulation
│   ├── bbox_nodes.py           # BBox tools (5 nodes)
│   ├── points_mask_editor.py   # Interactive editor
│   ├── sam_model_loader.py     # SAM/SAM2/SAM3 loader
│   ├── sam_mask_generator.py   # SAM inference
│   ├── sam_vitmatte_pipeline.py# End-to-end pipeline
│   ├── vitmatte_refiner.py     # Standalone edge refiner
│   ├── trimap_generator.py     # Trimap generation
│   ├── matting_node.py         # Multi-backend matting
│   ├── unified_segmentation_node.py  # Unified dispatcher
│   ├── unified_segmentation.py # Segmentation backends
│   ├── video_frame_extractor.py# Frame extraction
│   ├── universal_reroute.py    # Reroute dot node
│   ├── parameter_memory.py     # Parameter history
│   ├── model_manager.py        # Shared model cache
│   └── utils.py                # Shared utilities
└── tests/
    └── test_nodes.py           # 78 eval tests
```

---

## License

MIT

---

<p align="center">
  Made with ❤️ for the ComfyUI community
</p>
