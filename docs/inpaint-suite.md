# Inpaint Suite

> **Category:** `MaskEditControl/Inpaint`  
> **VRAM Tier:** 1 (pure tensor math — no models loaded)  
> **File:** [`nodes/inpaint_suite.py`](../nodes/inpaint_suite.py)

The Inpaint Suite is a 4-node pipeline for crop → inpaint → stitch workflows. Crop tightly around a mask, send the crop to any inpainting model, then composite the result back seamlessly with professional blend modes.

---

## Pipeline Overview

```
                                ┌─────────────────┐
                                │  Any Inpaint     │
[Image] + [Mask]                │  Model (KSampler,│
       ↓                       │  SDXL Inpaint,   │
┌──────────────────┐            │  Flux, etc.)     │
│ Inpaint Crop Pro │──cropped──→│                  │──inpainted──┐
│                  │──stitch_data─────────────────────────────┐  │
└──────────────────┘                                          │  │
                                                              ↓  ↓
                                                   ┌──────────────────┐
                                                   │ Inpaint Stitch   │
                                                   │ Pro              │
                                                   └────────┬─────────┘
                                                            ↓
                                                     Seamless Result
```

**Alternative simple paste:** Use **Inpaint Paste Back** instead of Inpaint Stitch Pro for a quick resize + paste without blend mode logic.

**Standalone mask cleanup:** Use **Inpaint Mask Prepare** to clean a raw mask and generate dual masks (inpaint + stitch blend) without the crop pipeline.

---

## Nodes

### 1. Inpaint Crop Pro (MEC)

Crops the image tightly around the mask region, producing a focused crop for inpainting plus all the data needed to stitch it back.

**How it works:**
1. Computes a tight bounding box around the mask
2. Expands the box by `context_expand` to give the inpaint model surrounding context
3. Optionally enforces an aspect ratio on the crop region
4. Applies mask pre-processing (blur, grow/shrink)
5. Fills the masked area in the crop (`edge_pad`, `neutral_gray`, or `original`)
6. Resizes to target dimensions if needed
7. Generates two separate masks — one for the inpaint model and one for compositing
8. Builds a `stitch_data` dict that stores everything needed for the stitch step

#### Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `image` | IMAGE | — | — | Input image batch (B, H, W, C) |
| `mask` | MASK | — | — | Inpaint mask (B, H, W). White = area to inpaint |
| `context_expand` | FLOAT | `1.5` | 1.0 – 4.0 | How much to expand the crop beyond mask bounds. 1.0 = tight crop, 2.0 = double the padding |
| `inpaint_mask_mode` | CHOICE | `hard_binary` | `hard_binary` · `slight_feather` · `soft_blend` | What the inpaint model sees. `hard_binary` = crisp edges, `slight_feather` = gentle edge blur, `soft_blend` = very soft falloff |
| `stitch_blend_mode` | CHOICE | `gaussian` | `edge_aware` · `gaussian` · `laplacian_pyramid` · `frequency_blend` | Compositing method for stitching back. Stored in stitch_data and used by Inpaint Stitch Pro |
| `blend_radius` | INT | `32` | 1 – 256 | Feather radius in pixels for the stitch blend mask |
| `size_mode` | CHOICE | `free_size` | `free_size` · `forced_size` · `ranged_size` | `free_size` = keep natural crop dimensions; `forced_size` = resize to exact WxH; `ranged_size` = clamp dimensions to min/max |
| `forced_width` | INT | `1024` | 64 – 8192 | Target width when `size_mode` = `forced_size` |
| `forced_height` | INT | `1024` | 64 – 8192 | Target height when `size_mode` = `forced_size` |
| `min_size` | INT | `512` | 64 – 8192 | Minimum dimension when `size_mode` = `ranged_size` |
| `max_size` | INT | `2048` | 64 – 8192 | Maximum dimension when `size_mode` = `ranged_size` |
| `padding_multiple` | INT | `8` | 2 – 128 (step 2) | Pad output dimensions to be divisible by this value. Must be even. Most diffusion models need multiples of 8 or 64 |
| `video_stable_crop` | BOOLEAN | `false` | — | Lock the crop region across all frames using the union of all per-frame mask bounding boxes. Essential for video inpainting |
| `fill_masked_area` | CHOICE | `edge_pad` | `edge_pad` · `neutral_gray` · `original` | How to fill the masked region inside the crop before sending to the model |
| `downscale_factor` | FLOAT | `1.0` | 0.25 – 1.0 | Downscale the crop before inpainting (e.g. 0.5 = half resolution). The result is upscaled back during stitch |
| `downscale_method` | CHOICE | `lanczos` | `lanczos` · `bicubic` · `bilinear` · `nearest-exact` · `area` | Interpolation for downscaling. Lanczos gives best quality |
| `upscale_method` | CHOICE | `lanczos` | `lanczos` · `bicubic` · `bilinear` · `nearest-exact` · `area` | Interpolation for upscaling during stitch-back. Stored in stitch_data |
| `mask_blur` | FLOAT | `0.0` | 0.0 – 64.0 | Gaussian blur sigma on the mask before computing bbox |
| `mask_grow` | INT | `0` | −128 – 128 | Morphologically grow (+) or shrink (−) the mask before cropping |
| `aspect_ratio` | CHOICE | `none` | `none` · `1:1` · `4:3` · `3:4` · `16:9` · `9:16` · `3:2` · `2:3` · `21:9` · `9:21` · `5:4` · `4:5` · `custom` | Force the crop region to a specific aspect ratio |
| `custom_aspect_w` | INT | `1` | 1 – 100 | Width component for `custom` aspect ratio |
| `custom_aspect_h` | INT | `1` | 1 – 100 | Height component for `custom` aspect ratio |

#### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `stitch_data` | STITCH_DATA | Dictionary with crop coordinates, blend settings, original image — connect to Inpaint Stitch Pro or Inpaint Paste Back |
| `cropped_image` | IMAGE | The cropped image with masked area filled. Connect this to your inpainting model |
| `cropped_composite` | IMAGE | Debug preview: the cropped image with a semi-transparent red overlay showing where the mask is |
| `inpaint_mask` | MASK | The inpaint mask within the crop region. Connect this to your inpainting model's mask input |
| `stitch_blend_mask` | MASK | The blend mask for compositing (Gaussian, edge-aware, Laplacian, or frequency) |
| `crop_mask` | MASK | Binary mask showing the crop region in original image space |
| `info` | STRING | Detailed text with all computed values (crop region, sizes, modes, mask ranges) |

#### Fill Modes Explained

| Mode | What it does | When to use |
|------|-------------|-------------|
| `edge_pad` | Blurs the image and fills the masked area with the blurred version | Best default — gives the inpaint model smooth color gradients to work from |
| `neutral_gray` | Fills the masked area with 50% gray | Good for models that expect a clean slate |
| `original` | Keeps the original pixels in the masked area | Use when partial inpainting or touch-up is desired |

#### Blend Modes Explained

| Mode | How it works | Quality | Best for |
|------|-------------|---------|----------|
| `gaussian` | Simple Gaussian feathered border | Good | General-purpose, fast |
| `edge_aware` | Sobel edge-guided feathering that follows image structure | Great | Images with strong edges at the boundary |
| `laplacian_pyramid` | Multi-scale pyramid decomposition blending | Excellent | Complex textures, gradual color transitions |
| `frequency_blend` | FFT-based frequency domain blending | Excellent | Very smooth results, large inpaint regions |

---

### 2. Inpaint Stitch Pro (MEC)

Composites the inpainted crop back into the original image using the stitch_data from Inpaint Crop Pro. This is the full-featured stitch node with blend mode override and color matching.

#### Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `stitch_data` | STITCH_DATA | — | — | From Inpaint Crop Pro |
| `inpainted_image` | IMAGE | — | — | The inpainted result from your model |
| `blend_mode_override` | CHOICE | `from_crop` | `from_crop` · `edge_aware` · `gaussian` · `laplacian_pyramid` · `frequency_blend` | Override the blend mode set during crop. `from_crop` = use whatever was set in Inpaint Crop Pro |
| `color_match` | BOOLEAN | `false` | — | Apply mean + standard deviation color transfer before stitching. Reduces color shift at the boundary between inpainted and original regions |

#### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `image` | IMAGE | The final composited image with the inpainted region seamlessly blended |
| `blend_mask_used` | MASK | The actual blend mask that was applied (useful for debugging) |
| `info` | STRING | Stitch details (regions, modes, color match stats) |

---

### 3. Inpaint Paste Back (MEC)

Lightweight alternative to Inpaint Stitch Pro. Simply resizes the inpainted crop and pastes it back at the original location with optional feathered edges. No blend mode logic — just a clean paste.

#### Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `stitch_data` | STITCH_DATA | — | — | From Inpaint Crop Pro |
| `inpainted_image` | IMAGE | — | — | The inpainted crop result |
| `upscale_method` | CHOICE | `lanczos` | `lanczos` · `bicubic` · `bilinear` · `nearest-exact` · `area` | Interpolation for resizing the crop back to original dimensions |
| `feather_edges` | BOOLEAN | `false` | — | Apply Gaussian feather at the crop boundary for a softer transition |
| `feather_radius` | INT | `16` | 0 – 64 | Feather radius in pixels (only used when `feather_edges` is enabled) |

#### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `image` | IMAGE | The composited result |
| `info` | STRING | Paste region details |

#### When to use Paste Back vs Stitch Pro

| Use case | Recommended node |
|----------|-----------------|
| Quick preview of inpaint result | **Paste Back** |
| Production compositing | **Stitch Pro** (edge-aware / Laplacian blending) |
| Simple hard paste, no blending | **Paste Back** with `feather_edges` = false |
| Color shift at boundaries | **Stitch Pro** with `color_match` = true |

---

### 4. Inpaint Mask Prepare (MEC)

Standalone mask cleanup and dual-mask preparation. Use this when you already have a mask but need to clean it up and produce separate inpaint + stitch blend masks — without the crop pipeline.

**How it works:**
1. Fills holes inside the mask (optional)
2. Removes small disconnected blobs below a threshold area
3. Dilates the mask by a configurable number of pixels
4. Generates the inpaint mask (hard binary or slight feather)
5. Generates the stitch blend mask (Gaussian or edge-aware)
6. Optional temporal smoothing for video batches

#### Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `mask` | MASK | — | — | Raw input mask to clean |
| `fill_holes` | BOOLEAN | `true` | — | Fill interior holes in the mask |
| `remove_small_regions` | BOOLEAN | `true` | — | Remove disconnected blobs below threshold |
| `min_region_area` | INT | `100` | 0 – 100,000 | Minimum region area in pixels to keep |
| `grow_pixels` | INT | `4` | 0 – 256 | Dilate the mask by N pixels |
| `inpaint_edge_mode` | CHOICE | `hard_binary` | `hard_binary` · `slight_feather` | Edge style for the inpaint mask |
| `stitch_edge_mode` | CHOICE | `gaussian` | `gaussian` · `edge_aware` | Edge style for the stitch blend mask |
| `stitch_feather_radius` | INT | `16` | 1 – 128 | Feather radius for the stitch blend mask |
| `temporal_smooth` | BOOLEAN | `false` | — | Gaussian smoothing along the batch (time) dimension for video consistency |
| `temporal_sigma` | FLOAT | `1.5` | 0.1 – 10.0 | Temporal smoothing sigma in frames |
| `reference_image` | IMAGE | _(optional)_ | — | Reference image for edge-aware stitch mode |

#### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `inpaint_mask` | MASK | Cleaned mask for the inpaint model |
| `stitch_blend_mask` | MASK | Feathered blend mask for compositing |
| `debug_preview` | IMAGE | Visual preview with mask regions color-coded |
| `info` | STRING | Cleanup statistics (regions removed, pixels grown, etc.) |

---

## Interpolation Methods

All resize operations in the Inpaint Suite support 5 interpolation methods:

| Method | Engine | Quality | Speed | Best for |
|--------|--------|---------|-------|----------|
| `lanczos` | PIL (per-image) | Highest | Slower | Production quality, final upscale |
| `bicubic` | torch F.interpolate | Very high | Fast | Good balance of quality and speed |
| `bilinear` | torch F.interpolate | Good | Fastest | Quick previews, small resizes |
| `nearest-exact` | torch F.interpolate | Pixel-perfect | Fast | Pixel art, hard masks, no blur |
| `area` | torch F.interpolate | High for downscale | Fast | Downscaling without aliasing |

> **Note:** `lanczos` uses PIL internally (converting tensors to PIL images and back) because PyTorch's `F.interpolate` does not support Lanczos. This gives the best quality but is slightly slower for large batches.
