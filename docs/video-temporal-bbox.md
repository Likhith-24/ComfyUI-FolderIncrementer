# Video, Temporal & BBox

> **Category:** `MaskEditControl/Video` · `MaskEditControl/Temporal` · `MaskEditControl/BBox`  
> **VRAM Tier:** 1–2 (pure tensor math for BBox/video; optional SAM2 for propagation/temporal)

Eight nodes for video frame handling, mask propagation across frames, temporal interpolation between anchor masks, and bounding box creation/manipulation.

---

## Video Nodes

### 1. Video Frame Extractor (MEC)

Extract a single frame from a video image batch. Single images pass through unchanged.

**File:** [`nodes/video_frame_extractor.py`](../nodes/video_frame_extractor.py)  
**Category:** `MaskEditControl/Video`

#### Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `images` | IMAGE | — | — | Image batch (B,H,W,C). Single images pass through. |
| `frame_index` | INT | `0` | 0 – 999999 | Frame to extract (0-based). Clamped to batch length. |
| `mode` | COMBO | `first` | `specific_frame`, `first`, `last`, `middle` | Frame selection mode |

#### Mode Details

| Mode | Behavior |
|------|----------|
| `first` | Always frame 0 |
| `last` | Final frame |
| `middle` | Middle frame (B//2) |
| `specific_frame` | Uses `frame_index` value |

#### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `frame` | IMAGE | Extracted single frame |
| `total_frames` | INT | Total frames in input batch |
| `is_video` | BOOLEAN | True if input had B>1 |

---

### 2. Mask Propagate Video (MEC)

Propagate a single-frame mask across all video frames using 5 modes: static copy, optical flow warping, SAM2 video tracking, fade, or linear scaling.

**File:** [`nodes/mask_propagate_video.py`](../nodes/mask_propagate_video.py)  
**Category:** `MaskEditControl/Video`

#### Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `images` | IMAGE | — | — | Video frames (B,H,W,C) |
| `mask` | MASK | — | — | Source mask (single frame or batch) |
| `source_frame` | INT | `0` | 0 – 99999 | Frame index where the mask was drawn |
| `mode` | COMBO | `static` | See modes table | Propagation mode |
| `flow_threshold` | FLOAT | `2.0` | 0.0 – 50.0 | Optical flow magnitude threshold for masking |
| `fade_start` | FLOAT | `1.0` | 0.0 – 1.0 | Mask opacity at source frame (fade mode) |
| `fade_end` | FLOAT | `0.0` | 0.0 – 1.0 | Mask opacity at last frame (fade mode) |
| `bidirectional` | BOOLEAN | `true` | — | Propagate both forward and backward from source |

**Optional inputs:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `sam_model` | SAM_MODEL | SAM2 model for `sam2_video` mode |
| `points_json` | STRING | Point prompts for SAM2 video mode |

#### Propagation Modes

| Mode | Requires | Description |
|------|----------|-------------|
| `static` | — | Copy the mask identically to every frame |
| `optical_flow` | — | Warp mask using frame-to-frame optical flow estimation |
| `sam2_video` | SAM_MODEL | Run SAM2 video propagation from the annotated frame |
| `fade` | — | Linearly interpolate mask opacity from `fade_start` to `fade_end` |
| `scale_linear` | — | Linearly scale mask size across frames |

#### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `masks` | MASK | Propagated mask batch (one per frame) |
| `preview` | IMAGE | Visual preview of propagation |

---

### 3. Temporal Anchor (MEC)

Interpolate masks between anchor frames using Signed Distance Fields (SDF). Define keyframe masks at specific frame indices, and the node smoothly morphs between them across the full sequence.

**File:** [`nodes/temporal_anchor.py`](../nodes/temporal_anchor.py)  
**Category:** `MaskEditControl/Temporal`

#### Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `anchor_masks` | MASK | — | — | Mask batch with one mask per anchor frame (A,H,W) |
| `anchor_frames` | STRING | `"0"` | — | Comma-separated frame indices (e.g. `"0,10,30"`). Count must match masks. |
| `total_frames` | INT | `30` | 1 – 99999 | Total output frames |
| `easing` | COMBO | `smooth_step` | See easing table | Interpolation easing function |
| `sdf_iterations` | INT | `64` | 4 – 512, step 4 | SDF diffusion iterations. More = more accurate, slower. |
| `flow_refinement` | BOOLEAN | `false` | — | Enable optical flow refinement. Requires `images` input. |

**Optional:** `images` (IMAGE) — video frames for optical flow estimation

#### Easing Functions

| Easing | Formula | Behavior |
|--------|---------|----------|
| `linear` | $t$ | Constant speed |
| `ease_in` | $t^2$ | Slow start, fast end |
| `ease_out` | $1-(1-t)^2$ | Fast start, slow end |
| `smooth_step` | $3t^2-2t^3$ | Hermite S-curve, smooth start and end |

#### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `full_masks` | MASK | Complete mask sequence (total_frames, H, W) |
| `confidence` | FLOAT (list) | Per-frame confidence values |
| `info` | STRING | Anchor summary and interpolation stats |

---

## BBox Nodes

### 4. BBox Create

Manually create a bounding box by specifying coordinates.

**File:** [`nodes/bbox_nodes.py`](../nodes/bbox_nodes.py)  
**Category:** `MaskEditControl/BBox`

#### Parameters

| Parameter | Type | Default | Range |
|-----------|------|---------|-------|
| `x` | INT | `0` | 0 – 16384 |
| `y` | INT | `0` | 0 – 16384 |
| `width` | INT | `128` | 1 – 16384 |
| `height` | INT | `128` | 1 – 16384 |

#### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `bbox` | BBOX | Bounding box [x, y, w, h] |
| `bbox_str` | STRING | Human-readable string |

---

### 5. BBox From Mask

Extract the tightest bounding box from a mask with optional per-axis padding.

**File:** [`nodes/bbox_nodes.py`](../nodes/bbox_nodes.py)  
**Category:** `MaskEditControl/BBox`

#### Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `mask` | MASK | — | — | Input mask |
| `padding` | INT | `0` | 0 – 512 | Uniform padding around detected bbox |
| `padding_x` | INT | `0` | 0 – 512 | Extra horizontal padding (both sides) |
| `padding_y` | INT | `0` | 0 – 512 | Extra vertical padding (both sides) |
| `threshold` | FLOAT | `0.5` | 0.0 – 1.0 | Mask binarization threshold |

#### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `bbox` | BBOX | Detected bounding box |
| `x` | INT | Left coordinate |
| `y` | INT | Top coordinate |
| `width` | INT | Box width |
| `height` | INT | Box height |
| `bbox_str` | STRING | Human-readable string |

---

### 6. BBox To Mask

Convert a bounding box into a rectangular binary mask.

**File:** [`nodes/bbox_nodes.py`](../nodes/bbox_nodes.py)  
**Category:** `MaskEditControl/BBox`

#### Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `bbox` | BBOX | — | — | Input bounding box |
| `image_width` | INT | `512` | 1 – 16384 | Output mask width |
| `image_height` | INT | `512` | 1 – 16384 | Output mask height |

**Optional:** `reference_image` (IMAGE) — automatically sets width/height from image dimensions

#### Output

| Output | Type | Description |
|--------|------|-------------|
| `mask` | MASK | Rectangular mask at bbox location |

---

### 7. BBox Pad

Pad a bounding box asymmetrically and clamp to image bounds. Negative padding shrinks the box.

**File:** [`nodes/bbox_nodes.py`](../nodes/bbox_nodes.py)  
**Category:** `MaskEditControl/BBox`

#### Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `bbox` | BBOX | — | — | Input bounding box |
| `pad_left` | INT | `0` | −4096 – 4096 | Left padding (negative = shrink) |
| `pad_right` | INT | `0` | −4096 – 4096 | Right padding |
| `pad_top` | INT | `0` | −4096 – 4096 | Top padding |
| `pad_bottom` | INT | `0` | −4096 – 4096 | Bottom padding |
| `image_width` | INT | `512` | 1 – 16384 | Clamp boundary width |
| `image_height` | INT | `512` | 1 – 16384 | Clamp boundary height |

#### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `bbox` | BBOX | Padded bounding box |
| `bbox_str` | STRING | Human-readable string |

---

### 8. BBox Crop

Crop an image (and optionally a mask) to a bounding box region.

**File:** [`nodes/bbox_nodes.py`](../nodes/bbox_nodes.py)  
**Category:** `MaskEditControl/BBox`

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `image` | IMAGE | Image to crop |
| `bbox` | BBOX | Crop region |

**Optional:** `mask` (MASK) — also crop this mask to the same region

#### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `cropped_image` | IMAGE | Cropped image |
| `cropped_mask` | MASK | Cropped mask (empty if no mask input) |
| `bbox` | BBOX | Pass-through of the input bbox |
