<p align="center">
  <img src="https://i.pinimg.com/736x/5d/52/19/5d5219ef27c1530bea027ff95d352e05.jpg" alt="MEC Logo" width="120" />
</p>

<h1 align="center">ComfyUI-CustomNodePacks</h1>

<p align="center">
  <strong>MaskEditControl (MEC) + FolderIncrementer + VFX Suite</strong><br/>
  Production-grade mask editing, SAM1/2/3 segmentation, alpha matting, inpainting, diagnostics,<br/>
  temporal mask interpolation, luminance keying, color science, EXR I/O, render-pass<br/>
  compositing, plate tools, VAE merging &amp; analysis, and auto-versioned file output for ComfyUI.
</p>

<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#node-reference">Node Reference</a> •
  <a href="#workflows">Workflows</a> •
  <a href="#troubleshooting">Troubleshooting</a> •
  <a href="#acknowledgements">Acknowledgements</a> •
  <a href="#license">License</a>
</p>

---

## Overview

**ComfyUI-CustomNodePacks** ships **72 nodes** organized into the following packs:

| Pack | Nodes | Purpose |
|------|------:|---------|
| **MaskEditControl (MEC)** | 33 | Pinpoint mask editing, SAM1/2/3 segmentation, SAM multi-mask picker, SeC + MatAnyone2 pipeline, background removal, face/clothes parsing, ViTMatte alpha matting, luminance keying, inpaint crop/stitch/paste-back suite, image comparison, mask failure diagnostics, temporal anchor interpolation, video propagation, compositing tools |
| **VAE Tools** | 4 | Merge two/three VAEs (8 algorithms, per-block alpha), latent inspector, similarity analyser, per-block weight inspector |
| **VFX Suite** | 19 | Color-space convert (sRGB/linear/Rec.709/ACEScg), `.cube` LUT apply, exposure/grade, EXR load+save, merge render passes, depth-of-field mask, depth warp, normal→curvature, position-pass splitter, grain match, plate stabilizer, clean-plate extractor, difference matte, metadata writer, frame-range router, shot-metadata reader |
| **Diagnostics** | 5 | EXR metadata reader, batch version manager, temporal consistency checker, model metadata extractor, parameter history |
| **FolderIncrementer** | 3 | Filesystem-safe auto-versioned output (`v001`, `v002`, …) with name sanitization and atomic version reservation |
| **Universal Reroute** | 1 | Nuke-style Dot node — reroute any wire type for cleaner workflow graphs |
| **Parameter Memory** | 1 | Tracks every parameter change with SQLite history, defaults recall, and per-run diffing |

All nodes are prefixed with **(MEC)** in the ComfyUI node menu for easy discovery.

### Who is this for?

- **VFX / compositing artists** who need Nuke-quality mask control inside ComfyUI
- **Video creators** who need temporally consistent masks across hundreds of frames
- **Inpainters** who want crop → inpaint → stitch pipelines with edge-aware blending
- **Anyone** tired of SAM giving 3 masks and not knowing which one to pick
- **Beginners** who want one-click background removal or a node that tells them *why* their mask failed

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

Download SAM 2.1 / SAM 3 weights and place them in:

```
ComfyUI/models/sams/        ← SAM 3
ComfyUI/models/sam2/        ← SAM 2.1
```

| Model | File | Source |
|-------|------|--------|
| SAM2.1 Large | `sam2.1_hiera_large.pt` | [Meta AI](https://github.com/facebookresearch/sam2) |
| SAM3 | `sam3_hiera_large.pt` | [Meta AI](https://github.com/facebookresearch/sam3) |

The **SAM Model Loader** node auto-detects model type from the filename.

> **Note:** Legacy SAM 1 (ViT-H/L/B) and the original SAM 2.0 line are no longer supported. Only SAM 2.1 and SAM 3 are exposed.

### 4. (Optional) ViTMatte matting

For the highest-quality edge refinement (hair, fur, glass, lace):

```bash
pip install transformers pillow
```

The ViTMatte model (~400 MB) auto-downloads from HuggingFace on first use. For offline setups, place model files in `ComfyUI/models/vitmatte/`.

### 5. Restart ComfyUI

Look for `[MEC] Loaded 44 MaskEditControl nodes.` in the console to confirm.

---

## Detailed Documentation

Every node's parameters, modes, and outputs are documented in depth in the `docs/` folder:

| Guide | Nodes | What's inside |
|-------|------:|---------------|
| [Inpaint Suite](docs/inpaint-suite.md) | 4 | Crop Pro, Stitch Pro, Paste Back, Mask Prepare — full pipeline with fill modes, blend modes, interpolation methods |
| [Mask Editing](docs/mask-editing.md) | 8 | Transform XY, Draw Frame (12 shapes), Draw Shape (unified dropdown), Composite Advanced (8 ops), Math (11 ops), Batch Manager, Preview Overlay, Spline Mask Editor |
| [SAM & Segmentation](docs/sam-segmentation.md) | 8 | Model Loader, Mask Generator, Multi-Mask Picker, Unified Segmentation, Semantic Segment, Background Remover, both pipelines |
| [Matting & Refinement](docs/matting-refinement.md) | 4 | Matting Node (7 backends), ViTMatte Refiner (7 methods), Trimap Generator, Luminance Keyer |
| [Video, Temporal & BBox](docs/video-temporal-bbox.md) | 10 | Frame Extractor, Mask Propagate (5 modes), Temporal Anchor (SDF), Motion Mask Tracker (4 methods), 6 BBox nodes |
| [Utility & Interactive](docs/utility-nodes.md) | 8 | Points Mask Editor, Image Comparer, Mask Failure Explainer, Parameter History, Universal Reroute, 3 Folder Incrementer nodes |
| [VAE Tools](docs/vae-merge.md) | 4 | VAE Merge (8 algorithms), Latent Inspector, Similarity Analyser, Block Inspector |
| [Color Science](docs/color-science.md) | 3 | sRGB/linear/Rec.709/ACEScg convert, `.cube` LUT apply, exposure/WB/contrast grade |
| [EXR I/O](docs/exr-io.md) | 2 | Load and save OpenEXR with imageio + TIFF fallback chain |
| [Render Passes](docs/render-pass.md) | 2 | Merge beauty + AO/diffuse/spec/emission, depth→CoC mask |
| [Plate Tools](docs/plate-tools.md) | 4 | Grain match, plate stabilizer (ORB/FFT), clean-plate extractor, difference matte |

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

Loads SAM 2.1 / SAM 3 checkpoints with optional VRAM offload.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | — | Checkpoint filename (auto-listed from `models/sams/` and `models/sam2/`) |
| `model_type` | `auto` | Force type: `sam2.1`, `sam3`, or `auto` |
| `device` | `cuda` | `cuda` / `cpu` |
| `offload_to_cpu` | `false` | Keep model on CPU, move to GPU only during inference (saves ~2–4 GB VRAM) |
| `dtype` | `float16` | `float16` / `bfloat16` / `float32` |

**Output:** `SAM_MODEL` — connect to SAM Mask Generator, SAM Multi-Mask Picker, or SAM + ViTMatte Pipeline.

---

#### SAM Mask Generator (MEC)

Runs SAM 2.1 / SAM 3 inference with point + bbox prompts, iterative refinement, and auto-negative point sampling.

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

#### SAM Multi-Mask Picker (MEC)

Run SAM inference and view **all 3 candidate masks** side-by-side with IoU scores. An interactive JS widget renders thumbnails — click a mask or press **1** / **2** / **3** to select it instantly.

**How it works:** SAM always outputs 3 masks ranked by confidence. Instead of guessing which `mask_index` is best, this node shows all three with quality scores and lets you pick visually. Works with SAM1, SAM2, and HQ-SAM.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `image` | — | Input image |
| `model_name` | — | SAM model (SAM1 / SAM2 / HQ-SAM, auto-listed) |
| `points_json` | `[{"x":256,"y":256,"label":1}]` | Point prompts JSON |
| `bbox_json` | `""` | Optional bounding box `[x1,y1,x2,y2]` |
| `precision` | `fp32` | `fp32` / `fp16` / `bf16` |
| `selected_index` | `0` | Which candidate (0–2) — updated by widget click |
| `sam_model` | (optional) | Pre-loaded SAM model from SAM Model Loader |
| `bbox` | (optional) | BBox from BBox pipeline (overrides `bbox_json`) |

**Outputs:** `selected_mask`, `all_masks` (3×H×W), `selected_index`, `scores` (JSON), `info`

> **Who is it for?** When SAM gives you 3 options and you don't know which is best — see them all at once, pick with one click, and pipe the winner downstream.

---

#### Unified Segmentation (MEC)

One-node dispatcher for **SAM 2.1, SAM 3, SeC, VideoMaMa** with automatic image vs. video detection.

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

End-to-end pipeline: **SAM coarse segmentation → iterative refinement → neural alpha matting** in a single node. Best possible masking quality for single images.

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

#### SeC + MatAnyone2 Pipeline (MEC)

End-to-end pipeline: **SeC MLLM segmentation → MatAnyone2 temporal alpha matting** in a single node. Best for video masking with temporal consistency.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `segmentation_model` | — | SeC / SAM2 / SAM3 model selector |
| `text_prompt` | `""` | Text description of target object (e.g. "cat", "person in red") |
| `matting_backend` | `auto` | `auto` / `matanyone2` / `vitmatte_small` / `vitmatte_base` |
| `edge_radius` | `15` | Edge refinement radius in pixels |
| `n_warmup` | `5` | MatAnyone2 warmup frames (more = better temporal init) |
| `edge_refine_method` | `none` | Optional post-matting refinement: `vitmatte` / `guided_filter` / `multi_scale_guided` |
| `fill_holes_enabled` | `true` | Fill interior holes in alpha |
| `min_region_size` | `64` | Remove isolated regions < N pixels |

**Outputs:** `rgb` (premultiplied), `alpha_mask`, `coarse_mask`, `preview`, `info`

**Pipeline stages:**
1. SeC/SAM coarse segmentation (text or point/bbox prompts)
2. MatAnyone2 temporal alpha matting with warmup protocol
3. Optional edge refinement (ViTMatte / guided filter)
4. Post-processing (hole fill, small region removal)

**Key advantages over SAM + ViTMatte:**
- SeC uses a Vision-Language Model for semantic understanding (text prompts)
- MatAnyone2 provides temporal consistency across video frames
- Better for long sequences with occlusions and re-appearances

---

#### Semantic Segment (MEC)

Face / body / clothes semantic parsing using SegFormer. Select classes by name to build a combined mask.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | — | `segformer_face` (19-class facial) or `segformer_clothes` (18-class apparel) |
| `classes_csv` | `"skin,hair"` | Comma-separated class names to include |
| `threshold` | `0.5` | Confidence threshold |
| `invert` | `false` | Invert output mask |

**Face classes:** skin, l_brow, r_brow, l_eye, r_eye, eye_g, l_ear, r_ear, ear_r, nose, mouth, u_lip, l_lip, neck, necklace, cloth, hair, hat

**Clothes classes:** hat, hair, sunglasses, upper_clothes, skirt, pants, dress, belt, left_shoe, right_shoe, face, left_leg, right_leg, left_arm, right_arm, bag, scarf

**Output:** `mask`, `info`

> **Who is it for?** Portrait retouchers who need to isolate specific facial features or clothing items without manual masking.

---

### Alpha Matting

#### Background Remover (MEC)

One-click background removal using RMBG-2.0 or BiRefNet. Outputs a clean foreground and alpha mask.

| Model | Quality | Best for |
|-------|---------|----------|
| `rmbg_2.0` | ★★★★☆ | General-purpose, fast |
| `birefnet_general` | ★★★★★ | High-detail edges |
| `birefnet_portrait` | ★★★★★ | Human portraits |

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | — | Background removal model selector |
| `threshold` | `0.5` | Alpha threshold (0 = soft, 1 = hard) |
| `invert` | `false` | Keep background instead |
| `mask_blur` | `0` | Gaussian blur on mask edges |

**Outputs:** `foreground` (premultiplied RGB), `mask`, `info`

> **Who is it for?** E-commerce product photography, quick compositing, batch background removal.

---

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

### Keying

#### Luminance Keyer (MEC)

Professional luminance keyer inspired by Nuke's LumaKeyer. Extracts a matte based on image brightness using ITU-R BT.709 luminance with smooth S-curve falloff and gamma correction.

**How it works:** Converts the image to BT.709 luminance (`0.2126R + 0.7152G + 0.0722B`), then applies a threshold range with Hermite smoothstep falloff. Five modes target different brightness ranges — **auto** mode analyzes the image and picks the best range automatically.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `image` | — | Input image(s) to key |
| `mode` | `auto` | `auto` / `highlights` (0.7–1.0) / `midtones` (0.3–0.7) / `shadows` (0.0–0.3) / `custom` |
| `low` | `0.0` | Low threshold (custom mode only) |
| `high` | `1.0` | High threshold (custom mode only) |
| `gamma` | `1.0` | Gamma correction — >1 shrinks mask, <1 expands |
| `falloff` | `1.0` | Edge smoothness — 0 = hard binary, 1 = smooth, >1 = very gradual |
| `invert` | `false` | Flip the mask |

**Outputs:** `mask`, `info` (mode, thresholds, luminance stats, per-frame coverage)

> **Who is it for?** VFX artists pulling luminance keys (sky replacement, highlight isolation, shadow grading), colorists building luminance-driven masks for selective color grading, anyone who needs brightness-based masking without a model.

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

Low-level shape drawing. Accepts a `shape` + JSON `shape_params_json`. Supports all 12 shapes (circle, rectangle, ellipse, polygon, line, triangle, star, diamond, cross, rounded_rectangle, heart, arrow) with SDF rendering, feathering, rotation, and blend operations.

> **Prefer Draw Shape (MEC)** for new workflows — it exposes all parameters as named inputs instead of raw JSON.

---

#### Draw Shape (MEC) ★ New

Unified 12-shape drawing node with a single dropdown. All parameters are visible as named inputs with tooltips — irrelevant ones are ignored per shape. Replaces the 5 legacy per-shape wrapper nodes.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `shape` | `circle` | Dropdown: circle, rectangle, ellipse, polygon, line, triangle, star, diamond, cross, rounded_rectangle, heart, arrow |
| `cx` / `cy` | `256` | Center position (center-based shapes) |
| `radius` | `50` | Circle radius, triangle/heart size |
| `size_w` / `size_h` | `200` / `100` | Rectangle, rounded rect, diamond, arrow width/height |
| `rx` / `ry` | `100` / `50` | Ellipse radii |
| `top_left_x` / `top_left_y` | `100` | Rectangle/line start position |
| `x2` / `y2` | `400` | Line end position |
| `thickness` | `5` | Line, cross thickness |
| `outer_r` / `inner_r` | `100` / `40` | Star radii |
| `num_points` | `5` | Star/polygon point count |
| `corner_radius` | `20` | Rounded rectangle corners |
| `arrow_length` / `head_length` / `head_width` | — | Arrow dimensions |
| `points_json` | — | Polygon vertices: `[[x1,y1], ...]` |
| `value` | `1.0` | Fill intensity |
| `feather` | `0.0` | Edge softness |
| `rotation` | `0.0` | Rotation in degrees |
| `operation` | `set` | `set` / `add` / `subtract` / `max` / `min` |
| `batch_size` | `1` | Number of mask frames |

**Optional:** `coords_json` (per-frame position override), `existing_mask`, `reference_image`

**Output:** `mask` (MASK)

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

### Inpainting

#### Inpaint Crop Pro (MEC)

Crop image tightly around the mask region for focused inpainting, with separate inpaint and stitch blend masks. Feed the crop to any inpaint model, then stitch back seamlessly.

**How it works:** Computes tight bounding box around the mask, expands by `context_expand`, produces two separate masks — one for the inpaint model (crisp or feathered) and one for compositing back (Gaussian, edge-aware, Laplacian pyramid, or FFT frequency blend). Supports video-stable cropping (union bbox across all frames).

| Parameter | Default | Description |
|-----------|---------|-------------|
| `image` | — | Input image batch |
| `mask` | — | Inpaint mask (white = area to inpaint) |
| `context_expand` | `1.5` | Crop expansion factor beyond mask bounds (1.0 = tight) |
| `inpaint_mask_mode` | `hard_binary` | What the model sees: `hard_binary` / `slight_feather` / `soft_blend` |
| `stitch_blend_mode` | `gaussian` | Compositing mode: `edge_aware` / `gaussian` / `laplacian_pyramid` / `frequency_blend` |
| `blend_radius` | `32` | Feather radius for stitch blend mask |
| `size_mode` | `free_size` | `free_size` / `forced_size` (exact W×H) / `ranged_size` (min/max clamp) |
| `forced_width` / `forced_height` | `1024` | Target dimensions for `forced_size` mode |
| `min_size` / `max_size` | `512` / `2048` | Dimension clamp for `ranged_size` mode |
| `padding_multiple` | `8` | Pad output to be divisible by N |
| `video_stable_crop` | `false` | Lock bbox across all frames for video consistency |
| `fill_masked_area` | `edge_pad` | Fill masked area in crop: `edge_pad` / `neutral_gray` / `original` |
| `downscale_method` | `lanczos` | Interpolation for downscaling: `lanczos` / `bicubic` / `bilinear` / `nearest-exact` / `area` |
| `upscale_method` | `lanczos` | Interpolation for upscaling: same options |

**Outputs:** `stitch_data`, `cropped_image`, `inpaint_mask`, `stitch_blend_mask`, `crop_mask`, `cropped_composite`, `info`

> `cropped_composite` is a red-tinted debug preview showing the mask overlay on the crop — useful for verifying region coverage.

> **Who is it for?** Inpainting power users who want crop → inpaint → stitch with professional blend modes (Laplacian pyramid, frequency blend, edge-aware).

---

#### Inpaint Stitch Pro (MEC)

Composite inpainted image back into the original using stitch data from Inpaint Crop Pro.

**How it works:** Takes the inpainted result and the stitch data, resizes and places it back at the original crop location, blending using the stored blend mask. Optional mean+std color matching reduces color shift at boundaries.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `stitch_data` | — | From Inpaint Crop Pro |
| `inpainted_image` | — | Inpainted result from any model |
| `blend_mode_override` | `from_crop` | Override blend mode or keep original |
| `color_match` | `false` | Apply mean+std color transfer before stitching |

**Outputs:** `image`, `blend_mask_used`, `info`

---

#### Inpaint Mask Prepare (MEC)

Standalone mask cleanup and dual-mask preparation for inpainting workflows.

**How it works:** Fills holes, removes small disconnected blobs, dilates the mask, then produces two outputs — a clean inpaint mask (for the model) and a stitch blend mask (for compositing). Optional temporal smoothing for video batch consistency.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mask` | — | Raw input mask |
| `fill_holes` | `true` | Fill interior holes |
| `remove_small_regions` | `true` | Remove disconnected blobs |
| `min_region_area` | `100` | Minimum region area in pixels |
| `grow_pixels` | `4` | Dilate mask by N pixels |
| `inpaint_edge_mode` | `hard_binary` | `hard_binary` / `slight_feather` |
| `stitch_edge_mode` | `gaussian` | `gaussian` / `edge_aware` |
| `stitch_feather_radius` | `16` | Feather radius for stitch blend |
| `temporal_smooth` | `false` | Gaussian smoothing along batch dimension |
| `temporal_sigma` | `1.5` | Temporal smoothing sigma (frames) |
| `reference_image` | (optional) | For edge-aware stitch blend |

**Outputs:** `inpaint_mask`, `stitch_blend_mask`, `debug_preview`, `info`

---

#### Inpaint Paste Back (MEC)

Simple paste-back node: resize the inpainted crop and composite it onto the original with optional Gaussian-feathered alpha blending.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `stitch_data` | — | From Inpaint Crop Pro |
| `inpainted_image` | — | Inpainted result from any model |
| `feather_edges` | `true` | Gaussian-feathered alpha blending at crop boundary |
| `feather_radius` | `8` | Blur radius for edge feathering (pixels) |

**Outputs:** `image`, `info`

> **Paste Back vs Stitch Pro:** Paste Back is simpler — just resize + paste with optional feathering. Stitch Pro uses the advanced blend mask (Laplacian pyramid, edge-aware, FFT frequency blend) stored in stitch_data for professional compositing.

---

#### Image Comparer (MEC)

Interactive before/after comparison widget with 3 modes: drag-slider split, overlay blend, and difference heatmap.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `image_a` | — | Left / "before" image |
| `image_b` | — | Right / "after" image |
| `label_a` | `"Before"` | (optional) Label for image A |
| `label_b` | `"After"` | (optional) Label for image B |

**Modes:** ◧ Compare (drag slider) · ⊕ Overlay (alpha blend) · ≠ Diff (heatmap)

> **Who is it for?** Quick visual comparison of before/after results — inpainting QA, model comparison, parameter tuning.

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

#### Temporal Anchor System (MEC)

Mask interpolation over time using Signed Distance Fields (SDF). Define masks on a few key frames and let the node smoothly morph between them across the full video.

**How it works:** Computes an SDF for each anchor mask (distance from boundary, negative inside, positive outside), then interpolates the SDF fields between anchor frames using configurable easing. The zero-crossing of the blended SDF produces the interpolated mask boundary. Optional optical flow refinement (Farneback or FFT phase correlation) warps the SDF to follow motion.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `anchor_masks` | — | One mask per anchor frame (A, H, W) |
| `anchor_frames` | `"0"` | Comma-separated frame indices (e.g. `"0,10,30"`) |
| `total_frames` | `30` | Total output frames |
| `easing` | `smooth_step` | `linear` / `ease_in` / `ease_out` / `smooth_step` |
| `sdf_iterations` | `64` | SDF diffusion iterations (more = more accurate) |
| `flow_refinement` | `false` | Enable optical flow refinement |
| `images` | (optional) | Video frames for optical flow estimation |

**Outputs:** `full_masks` (total_frames × H × W), `confidence` (per-frame float list), `info`

> **Who is it for?** Video editors who need smooth mask morphing between keyframes — rotoscoping helpers, animated mask transitions, temporal mask interpolation where SAM tracking is overkill.

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

Six nodes for bounding box manipulation:

| Node | Description |
|------|-------------|
| **BBox Create** | Manual `[x, y, width, height]` entry |
| **BBox From Mask** | Extract tight bbox from non-zero mask pixels with per-axis padding |
| **BBox To Mask** | Convert bbox to a rectangular mask |
| **BBox Pad** | Asymmetric padding (top/bottom/left/right) with canvas clamping |
| **BBox Crop** | Crop image + mask to bbox region |
| **BBox Smooth Temporal** | Smooth bbox sequences across video frames (moving-average / exponential) |

All BBox nodes clamp outputs to valid canvas bounds — no negative dimensions, no out-of-bounds errors.

---

### Diagnostics

#### Mask Failure Explainer (MEC)

Diagnose why a mask failed and get actionable fix suggestions. Pure tensor analysis — no models, no VRAM.

**How it works:** Runs a 5-metric analysis pipeline on your image + mask pair:

| Metric | What it measures | Failure threshold |
|--------|-----------------|-------------------|
| **Brightness** | BT.709 mean luminance | Dark scene < 0.15 |
| **Blur** | Laplacian variance × 1000 | Blurry image < 50 |
| **Boundary contrast** | Std deviation at mask edge ring | Low contrast < 0.05 |
| **Color confusion** | Mean color distance inside vs. outside mask | High confusion < 0.1 |
| **Background complexity** | Sobel edge density outside mask | Busy background > 0.3 |

Each metric contributes up to 20 points to a severity score (0–100). The node outputs a human-readable explanation, a problem-regions heatmap, the severity score, and a suggested method (e.g. "try ViTMatte for complex edges" or "use BiRefNet for busy backgrounds").

| Parameter | Default | Description |
|-----------|---------|-------------|
| `image` | — | Input image(s) |
| `mask` | — | Mask to diagnose |
| `ring_width` | `5` | Boundary ring width for contrast analysis |
| `blur_threshold` | `50.0` | Laplacian variance threshold |
| `brightness_threshold` | `0.15` | Dark scene threshold |

**Outputs:** `explanation` (text), `problem_regions_mask` (heatmap), `severity_score` (0–100), `suggested_method` (string)

> **Who is it for?** Anyone whose mask looks wrong and doesn't know why. Plug in your image + bad mask, read the diagnosis, follow the suggestion. Especially useful for beginners learning which segmentation method works for which scenario.

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

#### Spline Mask Editor (MEC) ★ New

Interactive canvas for drawing spline-based masks directly on your image. Supports Catmull-Rom (smooth curves through points), Bezier (with tangent handles), and polyline (straight segments) modes.

| Action | Effect |
|--------|--------|
| **Left click** | Add control point (or close path by clicking first point) |
| **Shift + click** | Delete point under cursor |
| **Ctrl + click** | Insert point on nearest curve segment |
| **Right-click** | Context menu (Delete Point, Open/Close, Smooth/Sharp) |
| **S key** | Toggle smooth / sharp interpolation for current path |
| **Scroll wheel** | Zoom in/out |
| **Middle mouse drag** | Pan canvas |

**Key improvements:** Normalized [0,1] coordinates (resolution-independent), zoom-relative point sizes, segment insertion, close-by-clicking-first-point affordance (highlighted orange), property-based persistence.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `spline_type` | `catmull_rom` | `catmull_rom` / `bezier` / `polyline` |
| `closed` | `true` | Close the spline loop for filled region |
| `smoothing` | `true` | Enable spline smoothing |
| `samples_per_segment` | `20` | Curve resolution (higher = smoother) |
| `feather_radius` | `0.0` | Gaussian edge softness |
| `invert` | `false` | Fill outside spline instead of inside |

**Outputs:** `mask` (MASK), `coords_json` (SAM-compatible points), `spline_data_out` (for Motion Mask Tracker)

---

#### Motion Mask Tracker (MEC) ★ New

Per-frame motion detection with 4 independent methods and camera stabilization. Feed a video batch — get a motion mask highlighting what moved between frames.

| Method | Description |
|--------|-------------|
| `pixel_diff` | Absolute per-pixel brightness change |
| `optical_flow` | Farneback dense flow or phase correlation |
| `background_sub` | Static background model from first N frames |
| `histogram_diff` | Per-region color histogram distance |

| Parameter | Default | Description |
|-----------|---------|-------------|
| `camera_compensation` | `true` | Subtract global camera motion (homography/affine/translation) |
| `detection_mode` | `combined` | `combined` / individual method |
| `combine_method` | `union` | `union` (any triggers) / `intersection` (all agree) |
| `grow_pixels` | `4.0` | Expand detected regions |
| `min_region_size` | `100` | Remove noise regions < N pixels |
| `temporal_smooth` | `true` | Gaussian smoothing across frames |

**Outputs:** `motion_mask` (MASK), `motion_intensity` (FLOAT), `info` (STRING)

> **Who is it for?** Video editors who need automatic motion masks for selective effects, stabilization-aware change detection, or motion-triggered workflows.

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

## Node Quick-Reference Table

All 47 nodes at a glance:

| # | Node | Category | VRAM Tier | What it does |
|---|------|----------|-----------|-------------|
| 1 | SAM Model Loader | SAM | 2 | Load SAM 2.1 / SAM 3 checkpoints |
| 2 | SAM Mask Generator | SAM | 2 | SAM inference with point + bbox prompts |
| 3 | SAM Multi-Mask Picker | SAM | 2 | View all 3 SAM candidates, pick interactively |
| 4 | Unified Segmentation | SAM | 2 | One-node dispatcher for all segmentation backends |
| 5 | SAM + ViTMatte Pipeline | SAM / Matting | 2 | SAM → refinement → ViTMatte end-to-end |
| 6 | SeC + MatAnyone2 Pipeline | SAM / Matting | 3 | SeC → MatAnyone2 temporal video pipeline |
| 7 | Semantic Segment | Segmentation | 2 | SegFormer face/body/clothes parsing |
| 8 | Background Remover | Matting | 2 | One-click RMBG / BiRefNet removal |
| 9 | Matting Node | Matting | 2 | Unified 7-backend alpha matting |
| 10 | ViTMatte Edge Refiner | Matting | 2 | Standalone edge refinement (7 methods) |
| 11 | Trimap Generator | Matting | 1 | Generate trimap for ViTMatte input |
| 12 | Luminance Keyer | Keying | 1 | BT.709 luminance keying with smoothstep |
| 13 | Mask Transform XY | Editing | 1 | Per-axis erode/expand/blur/offset |
| 14 | Mask Draw Frame | Editing | 1 | Draw 12 shapes with rotation and SDF |
| 15 | Draw Shape | Editing | 1 | Unified 12-shape dropdown with all params — replaces per-shape wrappers |
| 16 | Mask Composite Advanced | Editing | 1 | Boolean/blend two masks (8 operations) |
| 17 | Mask Math | Editing | 1 | Mathematical mask operations (11 ops) |
| 18 | Spline Mask Editor | Interactive | 1 | Interactive Catmull-Rom / Bezier / polyline spline drawing on canvas |
| 19 | Inpaint Crop Pro | Inpaint | 1 | Crop around mask for inpainting |
| 20 | Inpaint Stitch Pro | Inpaint | 1 | Composite inpainted result back |
| 21 | Inpaint Paste Back | Inpaint | 1 | Simple paste-back with feathered alpha |
| 22 | Inpaint Mask Prepare | Inpaint | 1 | Clean + dual-mask preparation |
| 23 | Image Comparer | Preview | 1 | Interactive before/after comparison (3 modes) |
| 24 | Mask Batch Manager | Video | 1 | Slice/concat/interleave mask batches |
| 25 | Mask Propagate Video | Video | 1–2 | Propagate mask across video frames |
| 26 | Temporal Anchor System | Video | 2 | SDF-based mask interpolation between keyframes |
| 27 | Motion Mask Tracker | Video | 1 | Per-frame motion detection (4 methods, camera stabilization) |
| 28 | Video Frame Extractor | Video | 1 | Extract single frame from batch |
| 29 | BBox Create | BBox | 1 | Manual bbox entry |
| 30 | BBox From Mask | BBox | 1 | Extract bbox from mask |
| 31 | BBox To Mask | BBox | 1 | Convert bbox to mask |
| 32 | BBox Pad | BBox | 1 | Asymmetric bbox padding |
| 33 | BBox Crop | BBox | 1 | Crop image + mask to bbox |
| 34 | BBox Smooth Temporal | BBox | 1 | Smooth bbox sequences across video frames |
| 35 | Mask Failure Explainer | Diagnostics | 1 | Diagnose bad masks, suggest fixes |
| 36 | Points Mask Editor | Interactive | 1 | Canvas editor for points/bboxes |
| 37 | Mask Preview Overlay | Preview | 1 | 5-mode mask visualization |
| 38 | Universal Reroute / Dot | Utility | 1 | Any-type wire reroute |
| 39 | Parameter History | Utility | 1 | Track parameter changes over runs |
| 40 | Folder Version Incrementer | Output | 1 | Auto-versioned file output |
| 41 | Folder Version Check | Output | 1 | Check existing versions |
| 42 | Folder Version Set | Output | 1 | Reserve version slots |
| 43–47 | Draw Circle / Rectangle / Ellipse / Polygon / Line | Editing | 1 | *(Deprecated)* Legacy per-shape wrappers — use Draw Shape instead |
| 48 | VAE Merge | VAE | 1 | Merge 2/3 VAEs (8 algorithms, per-block alpha) |
| 49 | VAE Latent Inspector | VAE | 1 | Per-channel stats, NaN/Inf checks, health verdict |
| 50 | VAE Similarity Analyser | VAE | 1 | Cosine similarity (global + per-block) between two VAEs |
| 51 | VAE Block Inspector | VAE | 1 | Per-block weight stats for a single VAE |
| 52 | Color Space Convert | Color | 1 | sRGB ↔ linear ↔ Rec.709 ↔ ACEScg |
| 53 | LUT Apply (.cube) | Color | 1 | Adobe `.cube` 1D/3D LUT with strength blend |
| 54 | Exposure Grade | Color | 1 | Stops + WB temp/tint + contrast pivot |
| 55 | Load EXR | I/O | 1 | OpenEXR → imageio → TIFF fallback |
| 56 | Save EXR | I/O | 1 | Half-float by default, TIFF fallback |
| 57 | EXR Metadata Reader | I/O | 1 | Pure-python header parser (no OpenEXR needed) |
| 58 | Merge Render Passes | Render | 1 | Beauty + diffuse/spec/emission/AO compositing |
| 59 | Depth-of-Field Mask | Render | 1 | Depth pass → CoC alpha mask |
| 60 | Depth Warp | Geometry | 1 | Horizontal parallax warp from depth |
| 61 | Normal → Curvature | Geometry | 1 | Curvature mask from a tangent-space normal pass |
| 62 | Position Pass Splitter | Geometry | 1 | Split position pass into X/Y/Z masks |
| 63 | Grain Match | Plate | 1 | Extract grain from reference plate, re-apply to target |
| 64 | Plate Stabilizer | Plate | 1 | ORB+RANSAC affine (cv2) → FFT translation fallback |
| 65 | Clean Plate Extractor | Plate | 1 | Median across batch with optional mask exclusion |
| 66 | Difference Matte | Plate | 1 | L1/L2 image diff → MASK with threshold + softness |
| 67 | Metadata Writer | Metadata | 1 | Write/merge JSON sidecars next to outputs |
| 68 | Frame Range Router | Metadata | 1 | Slice IMAGE/MASK batches `[start:end:step]` |
| 69 | Shot Metadata Reader | Metadata | 1 | Read shot.json (show / shot / task / frame in/out / fps) |
| 70 | Batch Version Manager | Output | 1 | `<root>/<show>/<shot>/<task>/v###/` with atomic reservation |
| 71 | Temporal Consistency Checker | Diagnostics | 1 | Flicker score (mask_iou / pixel_diff / Farneback flow) |
| 72 | Model Metadata Extractor | Diagnostics | 1 | Inspect safetensors/checkpoints **without unpickling** |

**VRAM Tiers:** 1 = pure tensor math (CPU/GPU, no models), 2 = loads a model (~1–4 GB), 3 = loads multiple models

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

### Pick the Best SAM Mask

```
[Load Image]  →  [Points Mask Editor]  →  [SAM Multi-Mask Picker]
                                              ↓ click thumbnail
                                         selected_mask → downstream
```

### Inpaint Pipeline

```
[Load Image] + [Mask]  →  [Inpaint Crop Pro]     →  [Any Inpaint Model]
                              context_expand: 1.5        ↓
                              stitch_blend: edge_aware   [Inpaint Stitch Pro]
                                    ↓ stitch_data ──────→     ↓
                                                         seamless result
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

### Temporal Mask Morphing

```
[Define masks on frames 0, 15, 30]  →  [Temporal Anchor System]
                                          anchor_frames: "0,15,30"
                                          total_frames: 60
                                          easing: smooth_step
                                                ↓
                                          60 interpolated masks
```

### Alpha Matting (From Any Coarse Mask)

```
[Any Segmentation Node]  →  [Matting Node]
                               backend: auto
                               edge_radius: 15
                                    ↓
                              alpha_mask → compositing
```

### Diagnose a Bad Mask

```
[Load Image] + [Bad Mask]  →  [Mask Failure Explainer]
                                    ↓
                              explanation: "Dark scene (mean=0.12), low boundary contrast"
                              suggested_method: "Try ViTMatte with edge_radius=20"
                              severity: 65/100
```

### Luminance Keying

```
[Load Image]  →  [Luminance Keyer]
                    mode: auto
                    falloff: 1.0
                    gamma: 1.0
                       ↓
                  luminance mask (highlights / shadows / midtones)
```

### SeC + MEC (Video Object Segmentation)

[SeC](https://github.com/9nate-drake/Comfyui-SecNodes) handles tracking, MEC handles edge quality:

```
[SeC Video Segmentation]  →  [ViTMatte Edge Refiner (MEC)]  →  refined masks
```

Install SeC: `cd ComfyUI/custom_nodes && git clone https://github.com/9nate-drake/Comfyui-SecNodes`

### Spline Mask Drawing

```
[Load Image]  →  [Spline Mask Editor]
                    spline_type: catmull_rom
                    closed: true
                    feather_radius: 2.0
                       ↓
                  mask → any mask input
                  coords_json → SAM Mask Generator
                  spline_data → Motion Mask Tracker
```

### Motion Detection (Video)

```
[Load Video Frames]  →  [Motion Mask Tracker]
                           camera_compensation: true
                           detection_mode: combined
                           grow_pixels: 4
                                ↓
                          motion_mask → compositing / routing
                          motion_intensity → Switch node
```

### Draw Shape (Quick Masking)

```
[Draw Shape (MEC)]
   shape: star
   cx: 256, cy: 256
   outer_r: 100, inner_r: 40
   num_points: 5
   feather: 4.0
        ↓
   mask → downstream
```

---

## VRAM Management

All nodes are designed to work on **low-end GPUs** (4–6 GB VRAM):

- **VRAM monitoring** — model loading logs a warning when free VRAM drops below 2 GB
- **OOM recovery** — if GPU runs out of memory, nodes fall back to CPU instead of crashing
- **Offload to CPU** — SAM Model Loader has `offload_to_cpu` to keep models in system RAM
- **Auto memory cleanup** — all nodes run `gc.collect()` + `torch.cuda.empty_cache()` in `finally` blocks
- **dtype control** — use `float16` or `bfloat16` to halve VRAM usage
- **Torch-only fallbacks** — if `opencv-python` is not installed, all nodes fall back to pure PyTorch implementations (slightly slower but functional)
- **VRAM tiers** — each node declares its tier (1 = no model, 2 = one model, 3 = multiple models) so you can plan your workflow

---

## Running Tests

The test suite covers all pure-tensor nodes (no GPU models required):

```bash
# Run all tests:
python -m pytest tests/ -v --tb=short

# Run a specific node's tests:
python -m pytest tests/test_luminance_keyer.py -v
python -m pytest tests/test_inpaint_suite.py -v
python -m pytest tests/test_mask_failure_explainer.py -v
python -m pytest tests/test_temporal_anchor.py -v
python -m pytest tests/test_sam_multi_mask_picker.py -v
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: cv2` | `pip install opencv-python>=4.7.0` — nodes will use torch fallbacks but cv2 gives better quality |
| `ModuleNotFoundError: transformers` | `pip install transformers pillow` — only needed for ViTMatte matting |
| Torch version mismatch after `pip install -r requirements.txt` | Reinstall your ComfyUI torch: check [PyTorch install page](https://pytorch.org/get-started/locally/) for your CUDA version |
| SAM model not found | Place `.pth` / `.pt` / `.safetensors` in `ComfyUI/models/sams/` or `ComfyUI/models/sam2/` |
| CUDA out of memory | Enable `offload_to_cpu` in SAM Model Loader, use `float16` dtype, reduce image resolution |
| Nodes not showing in menu | Check console for `[MEC] Loaded 44` message. If missing, check for import errors in the console output |
| ViTMatte download fails | Download manually from [HuggingFace](https://huggingface.co/hustvl/vitmatte-small-composition-1k) → place in `ComfyUI/models/vitmatte/` |
| My mask looks bad | Connect your image + mask to **Mask Failure Explainer** — it will diagnose the issue and suggest the right method |

---

## Project Structure

```
ComfyUI-CustomNodePacks/
├── __init__.py                     # Node registration (47 nodes)
├── folder_incrementer.py           # FolderIncrementer nodes (3)
├── conftest.py                     # Pytest root configuration
├── pyproject.toml                  # Package metadata
├── requirements.txt                # Dependencies
├── js/
│   ├── folder_incrementer.js       # Frontend: FolderIncrementer
│   ├── image_comparer.js           # Frontend: before/after comparison widget
│   ├── points_bbox_editor.js       # Frontend: interactive canvas editor
│   ├── parameter_memory.js         # Frontend: parameter history UI
│   ├── sam_multi_mask_picker.js    # Frontend: 3-mask thumbnail picker
│   ├── spline_mask_editor.js       # Frontend: spline drawing canvas (normalized coords)
│   └── universal_reroute.js        # Frontend: Nuke-style dot
├── nodes/
│   ├── sam_model_loader.py         # SAM 2.1 / SAM 3 model loader
│   ├── sam_mask_generator.py       # SAM inference engine
│   ├── sam_multi_mask_picker.py    # Multi-mask picker + JS widget
│   ├── sam_vitmatte_pipeline.py    # SAM → ViTMatte end-to-end pipeline
│   ├── sec_matanyone_pipeline.py   # SeC → MatAnyone2 video pipeline
│   ├── unified_segmentation_node.py# Unified segmentation dispatcher
│   ├── unified_segmentation.py     # Segmentation backends (deprecated)
│   ├── semantic_segment.py         # SegFormer face/clothes parsing
│   ├── background_remover.py       # RMBG / BiRefNet background removal
│   ├── matting_node.py             # 7-backend alpha matting
│   ├── vitmatte_refiner.py         # Standalone edge refiner
│   ├── trimap_generator.py         # Trimap generation
│   ├── luminance_keyer.py          # BT.709 luminance keyer
│   ├── mask_transform_xy.py        # Per-axis mask transform
│   ├── mask_draw_frame.py          # Shape drawing (12 shapes + DrawShapeMEC unified node)
│   ├── mask_composite.py           # Mask compositing ops
│   ├── mask_math.py                # Mathematical mask ops
│   ├── mask_batch_manager.py       # Batch manipulation
│   ├── mask_propagate_video.py     # Video mask propagation
│   ├── mask_preview.py             # Mask visualization
│   ├── mask_failure_explainer.py   # Mask diagnostics engine
│   ├── temporal_anchor.py          # SDF interpolation system
│   ├── inpaint_suite.py            # Crop/stitch/paste-back/prepare (4 nodes)
│   ├── bbox_nodes.py               # BBox tools (6 nodes incl. BBoxSmooth)
│   ├── points_mask_editor.py       # Interactive point/bbox editor
│   ├── spline_mask_editor.py       # Interactive spline mask drawing
│   ├── motion_mask_tracker.py      # Per-frame motion detection (4 methods)
│   ├── stabilization_utils.py      # Camera stabilization helpers
│   ├── video_frame_extractor.py    # Frame extraction
│   ├── image_comparer.py           # Before/after image comparison
│   ├── universal_reroute.py        # Nuke-style reroute dot
│   ├── parameter_memory.py         # Parameter history + SQLite
│   ├── model_manager.py            # Shared model cache & download
│   └── utils.py                    # Shared utilities
├── docs/
│   ├── inpaint-suite.md            # Inpaint Crop/Stitch/Paste Back/Prepare
│   ├── mask-editing.md             # Transform, Draw, Composite, Math, Batch, Preview
│   ├── sam-segmentation.md         # SAM, Unified Segmentation, Pipelines
│   ├── matting-refinement.md       # Matting, ViTMatte, Trimap, Keyer
│   ├── video-temporal-bbox.md      # Video, Temporal Anchor, BBox tools
│   └── utility-nodes.md            # Editor, Comparer, Diagnostics, Utils
├── tests/
│   ├── test_luminance_keyer.py     # 38 tests
│   ├── test_inpaint_suite.py       # Inpaint suite tests
│   ├── test_mask_failure_explainer.py # Diagnostics tests
│   ├── test_temporal_anchor.py     # Temporal anchor tests
│   └── test_sam_multi_mask_picker.py  # Multi-mask picker tests
├── example_workflows/
│   ├── basic_sam_segmentation.json
│   ├── sam_vitmatte_pipeline.json
│   ├── mask_editing_toolkit.json
│   ├── bbox_pipeline.json
│   └── video_mask_propagation.json
└── third_party/                    # Reference implementations
```

---

## Acknowledgements

This project builds on or references the following open-source work:

| Project | Author | License | Description |
|---------|--------|---------|-------------|
| [Segment Anything](https://github.com/facebookresearch/segment-anything) | Meta AI | Apache-2.0 | SAM foundation model |
| [Segment Anything 2](https://github.com/facebookresearch/sam2) | Meta AI | Apache-2.0 | SAM2 / SAM2.1 video segmentation |
| [SAM-HQ](https://github.com/SysCV/sam-hq) | SysCV (ETH Zurich) | Apache-2.0 | High-quality SAM variant (NeurIPS 2023) |
| [ViTMatte](https://github.com/hustvl/ViTMatte) | HUST VL | MIT | Vision Transformer alpha matting |
| [Matte-Anything](https://github.com/hustvl/Matte-Anything) | HUST VL | MIT | Interactive SAM + matting pipeline |
| [MatAnyone2](https://github.com/pq-yang/MatAnyone2) | NTU S-Lab | S-Lab-1.0 | Video matting with learned quality evaluator |
| [RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting) | PeterL1n / ByteDance | GPL-3.0 | Real-time human video matting |
| [Cutie](https://github.com/hkchengrex/Cutie) | hkchengrex | MIT | Video object segmentation (CVPR 2024) |
| [ComfyUI-segment-anything-2](https://github.com/kijai/ComfyUI-segment-anything-2) | kijai | Apache-2.0 | SAM2 ComfyUI nodes with points editor |
| [ComfyUI-SecNodes](https://github.com/9nate-drake/Comfyui-SecNodes) | 9nate-drake | Apache-2.0 | SeC 4B video segmentation nodes |
| [ComfyUI-MatAnyone](https://github.com/FuouM/ComfyUI-MatAnyone) | FuouM | MIT | MatAnyone ComfyUI wrapper |
| [ComfyUI-RMBG](https://github.com/1038lab/ComfyUI-RMBG) | 1038lab | GPL-3.0 | RMBG-2.0 / BiRefNet background removal |
| [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes) | kijai | GPL-3.0 | Utility node collection |
| [ComfyUI-Inpaint-CropAndStitch](https://github.com/lquesada/ComfyUI-Inpaint-CropAndStitch) | lquesada | GPL-3.0 | Crop-and-stitch inpainting reference |
| [ComfyUI-VideoMaMa](https://github.com/okdalto/ComfyUI-VideoMaMa) | okdalto | — | VideoMaMa mask-guided video matting |
| [ComfyUI LayerStyle](https://github.com/chflame163/ComfyUI_LayerStyle) | chflame163 | MIT | Layer compositing node suite |
| [ComfyUI LayerStyle Advance](https://github.com/chflame163/ComfyUI_LayerStyle_Advance) | chflame163 | MIT | Advanced nodes (BiRefNet, Florence2, etc.) |

---

## License

MIT

---

<p align="center">
  Made with ❤️ for the ComfyUI community
</p>
