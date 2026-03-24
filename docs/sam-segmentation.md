# SAM & Segmentation

> **Category:** `MaskEditControl/SAM` · `MaskEditControl/Segmentation` · `MaskEditControl/Pipeline`  
> **VRAM Tier:** 2–3 (loads SAM/GroundingDINO/SeC models)

Eight nodes covering model loading, mask generation with point/bbox/text prompts, multi-mask interactive selection, unified multi-model segmentation, semantic parsing, one-click background removal, and end-to-end pipelines.

---

## Nodes

### 1. SAM Model Loader (MEC)

Load SAM, SAM2, SAM2.1, or SAM3 checkpoints with auto-detection from filename. Supports VRAM offload and automatic HuggingFace download.

**File:** [`nodes/sam_model_loader.py`](../nodes/sam_model_loader.py)  
**Category:** `MaskEditControl/SAM`

#### Parameters

| Parameter | Type | Default | Options | Description |
|-----------|------|---------|---------|-------------|
| `model_name` | COMBO | — | Scanned from `sams/`, `sam2/`, `sam3/` dirs + `[download]` entries | SAM checkpoint file. Models prefixed with `[download]` will be auto-downloaded from HuggingFace Hub. |
| `model_type` | COMBO | `auto` | `auto`, `sam2`, `sam2.1`, `sam3`, `sam_vit_h`, `sam_vit_l`, `sam_vit_b` | Architecture. `auto` detects from filename. |
| `device` | COMBO | `auto` | `auto`, `cuda`, `cpu` | Compute device |
| `offload_to_cpu` | BOOLEAN | `false` | — | Keep model on CPU between inferences. Saves 2–4 GB VRAM at cost of slower inference. |
| `dtype` | COMBO | `float16` | `float16`, `bfloat16`, `float32` | Model precision |

#### Supported Auto-Download Models

| Model | Repository | Family |
|-------|-----------|--------|
| SAM2 Hiera (tiny/small/base/large) | `facebook/sam2-hiera-*` | sam2 |
| SAM2.1 Hiera (tiny/small/base/large) | `facebook/sam2.1-hiera-*` | sam2.1 |
| SAM ViT (H/L/B) | `ybelkada/segment-anything` | sam1 |

#### Output

| Output | Type | Description |
|--------|------|-------------|
| `sam_model` | SAM_MODEL | Loaded model ready for inference |

---

### 2. SAM Mask Generator (MEC)

Generate masks using SAM with point prompts, bounding boxes, text grounding, iterative refinement, and automatic negative points.

**File:** [`nodes/sam_mask_generator.py`](../nodes/sam_mask_generator.py)  
**Category:** `MaskEditControl/SAM`

#### Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `sam_model` | SAM_MODEL | — | — | SAM model from SAM Model Loader |
| `image` | IMAGE | — | — | Input image |
| `points_json` | STRING | `"[]"` | multiline | JSON array of point prompts. `label=1` is foreground, `label=0` is background. Example: `[{"x":100,"y":200,"label":1}]` |
| `bbox_json` | STRING | `""` | — | Bounding box as `[x1, y1, x2, y2]` or `{"x":..,"y":..,"w":..,"h":..}` |
| `text_prompt` | STRING | `""` | — | Text description of target (e.g. "person", "dog"). Requires GroundingDINO model. |
| `negative_text_prompt` | STRING | `""` | — | Text description of objects to exclude. Generates negative points in those regions. |
| `grounding_model` | COMBO | `none` | Dynamic list | GroundingDINO model for text-to-bbox. `none` disables text prompting. |
| `text_threshold` | FLOAT | `0.25` | 0.0 – 1.0, step 0.01 | GroundingDINO box confidence threshold |
| `text_box_threshold` | FLOAT | `0.3` | 0.0 – 1.0, step 0.01 | GroundingDINO text-box association threshold |
| `multimask_output` | BOOLEAN | `true` | — | Return 3 candidate masks vs 1 |
| `mask_index` | INT | `0` | 0 – 2 | Which mask to return when multimask=True (0 = best score) |
| `score_threshold` | FLOAT | `0.0` | 0.0 – 1.0, step 0.01 | Discard masks below this confidence |
| `apply_bbox_crop` | BOOLEAN | `false` | — | Crop output to bbox region |
| `refine_iterations` | INT | `1` | 1 – 5 | Iterative refinement passes. 2–3 significantly improves accuracy. |
| `auto_negative_points` | BOOLEAN | `false` | — | Sample negative points outside mask boundary. Helps in cluttered scenes. |

**Optional inputs:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `bbox` | BBOX | BBox from upstream node (overrides bbox_json) |
| `existing_mask` | MASK | Starting mask instead of running SAM from scratch |

#### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `mask` | MASK | Best/selected mask |
| `all_masks` | MASK | All candidate masks stacked |
| `detected_bbox` | BBOX | Detected bounding box |
| `score` | FLOAT | Confidence score |
| `info` | STRING | Diagnostic info |

---

### 3. SAM Multi-Mask Picker (MEC)

Run SAM inference to get 3 candidate masks, displayed in an interactive JS widget. Click a thumbnail or press 1/2/3 to select. Press R to re-run.

**File:** [`nodes/sam_multi_mask_picker.py`](../nodes/sam_multi_mask_picker.py)  
**Category:** `MaskEditControl/SAM`

#### Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `image` | IMAGE | — | — | Input image (first frame used) |
| `model_name` | COMBO | — | Dynamic SAM model list | SAM model variant. Larger = better quality, more VRAM. |
| `points_json` | STRING | `[{"x":256,"y":256,"label":1}]` | multiline | Point prompts JSON array |
| `bbox_json` | STRING | `""` | — | Optional bounding box `[x1, y1, x2, y2]` |
| `precision` | COMBO | `fp32` | `fp32`, `fp16`, `bf16` | Model precision |
| `selected_index` | INT | `0` | 0 – 2 | Which candidate to output. Updated by JS widget. |

**Optional inputs:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `sam_model` | SAM_MODEL | Pre-loaded model (overrides model_name) |
| `bbox` | BBOX | BBox from upstream node |

#### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `selected_mask` | MASK | The chosen mask |
| `all_masks` | MASK | All 3 candidates stacked |
| `selected_index` | INT | Currently selected index |
| `scores` | STRING | JSON array of 3 confidence scores |
| `info` | STRING | Diagnostic info |

#### Interactive Widget

The node displays 3 mask thumbnails overlaid on the image in the ComfyUI canvas. Interaction:
- **Click** a thumbnail to select it
- **Press 1/2/3** to select by index
- **Press R** to re-run inference

---

### 4. Unified Segmentation Node (MEC)

All-in-one segmentation dispatcher supporting SAM2/2.1, SAM3, SeC, VideoMaMa, and HQ-SAM. Auto-detects image vs video mode from batch size. Supports GroundingDINO text prompting, bidirectional video tracking, and multiple attention backends.

**File:** [`nodes/unified_segmentation_node.py`](../nodes/unified_segmentation_node.py)  
**Category:** `MaskEditControl/Segmentation`

#### Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `image` | IMAGE | — | — | Single image (B=1) or video frames (B>1) |
| `model_name` | COMBO | — | Dynamic list | Segmentation model. `[download]` prefix auto-downloads from HuggingFace. |
| `points_json` | STRING | `"[]"` | multiline | Point prompts JSON. Used when positive/negative coords not connected. |
| `bbox_json` | STRING | `""` | — | Bounding box `[x1, y1, x2, y2]` |
| `multimask` | BOOLEAN | `true` | — | Return 3 candidates vs 1 |
| `mask_index` | INT | `0` | 0 – 2 | Candidate selection |
| `precision` | COMBO | `fp16` | `fp16`, `bf16`, `fp32` | Inference precision |
| `attention_mode` | COMBO | `auto` | `auto`, `sdpa`, `flash_attn`, `sage_attn`, `xformers` | Attention backend. `auto` selects best available. |

**Optional inputs:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `positive_coords` | STRING | — | Positive points from Points Mask Editor |
| `negative_coords` | STRING | — | Negative points from Points Mask Editor |
| `bbox` | BBOX | — | Positive bbox from upstream node |
| `neg_bbox_json` | STRING | `""` | Negative bbox (SAM3 exclusive) |
| `neg_bboxes` | BBOX | — | Negative bboxes from Points Mask Editor |
| `text_prompt` | STRING | `""` | Target object text. GroundingDINO converts to bbox for SAM; SeC uses native grounding. |
| `grounding_model` | COMBO | `none` | GroundingDINO model for text-to-bbox |
| `text_threshold` | FLOAT | `0.25` | GroundingDINO confidence threshold |
| `existing_mask` | MASK | — | Initial mask for refinement |
| `keep_model_loaded` | BOOLEAN | `true` | Keep model in VRAM between executions |
| `tracking_direction` | COMBO | `forward` | `forward`, `backward`, `bidirectional` — video propagation direction |
| `annotation_frame_idx` | INT | `0` | Frame index where prompts are placed (0-based) |
| `individual_objects` | BOOLEAN | `false` | Each positive point as separate tracked object |

#### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `masks` | MASK | Segmentation masks |
| `best_score` | FLOAT | Confidence score |
| `info` | STRING | Model info, timing, prompt summary |

---

### 5. Semantic Segment (MEC)

Semantic face/clothes parsing using SegFormer. Select classes by name to build a combined mask.

**File:** [`nodes/semantic_segment.py`](../nodes/semantic_segment.py)  
**Category:** `MaskEditControl/Segmentation`

#### Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `image` | IMAGE | — | — | Input image(s) |
| `model_name` | COMBO | — | `segformer_face`, `segformer_clothes` | Parsing model |
| `classes_csv` | STRING | `"skin,hair"` | — | Comma-separated class names to include |
| `threshold` | FLOAT | `0.5` | 0.0 – 1.0 | Confidence threshold |
| `invert` | BOOLEAN | `false` | — | Invert output mask |

**Optional:** `keep_model_loaded` (BOOLEAN, default `true`)

#### Available Classes

**Face model (19 classes):** `background`, `skin`, `l_brow`, `r_brow`, `l_eye`, `r_eye`, `eye_g`, `l_ear`, `r_ear`, `ear_r`, `nose`, `mouth`, `u_lip`, `l_lip`, `neck`, `necklace`, `cloth`, `hair`, `hat`

**Clothes model (18 classes):** `background`, `hat`, `hair`, `sunglasses`, `upper_clothes`, `skirt`, `pants`, `dress`, `belt`, `left_shoe`, `right_shoe`, `face`, `left_leg`, `right_leg`, `left_arm`, `right_arm`, `bag`, `scarf`

#### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `mask` | MASK | Combined class mask |
| `info` | STRING | Detected classes and pixel counts |

---

### 6. Background Remover (MEC)

One-click background removal using RMBG-2.0 or BiRefNet. No prompts needed.

**File:** [`nodes/background_remover.py`](../nodes/background_remover.py)  
**Category:** `MaskEditControl/Matting`

#### Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `image` | IMAGE | — | — | Input image(s) |
| `model_name` | COMBO | — | `birefnet_general`, `birefnet_portrait`, `rmbg_2.0` | Removal model. `rmbg_2.0`: fast general-purpose. `birefnet_general`: high-detail edges. `birefnet_portrait`: optimized for humans. |
| `threshold` | FLOAT | `0.5` | 0.0 – 1.0 | Alpha threshold (0=soft, 1=hard) |
| `invert` | BOOLEAN | `false` | — | Keep background instead |
| `mask_blur` | INT | `0` | 0 – 50 | Gaussian blur on final mask edges |

**Optional:** `keep_model_loaded` (BOOLEAN, default `true`)

#### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `foreground` | IMAGE | Premultiplied RGB (image x alpha) |
| `mask` | MASK | Alpha mask |
| `info` | STRING | Model used, resolution, timing |

---

### 7. SAM + ViTMatte Pipeline (MEC)

End-to-end pipeline: iterative SAM refinement → edge-aware matting → multi-scale fusion → cleanup. Produces compositing-grade alpha mattes.

**File:** [`nodes/sam_vitmatte_pipeline.py`](../nodes/sam_vitmatte_pipeline.py)  
**Category:** `MaskEditControl/Pipeline`

#### Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `sam_model` | SAM_MODEL | — | — | From SAM Model Loader |
| `image` | IMAGE | — | — | Input image |
| `points_json` | STRING | `"[]"` | multiline | Point prompts JSON |
| `bbox_json` | STRING | `""` | — | Bounding box JSON |
| `sam_iterations` | INT | `2` | 1 – 5 | SAM refinement passes. 2–3 is ideal. |
| `refine_method` | COMBO | `auto` | See methods table | Edge refinement backend |
| `edge_radius` | INT | `12` | 1 – 200 | Pixels around edges to refine |
| `detail_preservation` | FLOAT | `0.85` | 0.0 – 1.0 | Fine detail preservation (hair, fur). 0=smooth, 1=max detail. |
| `edge_contrast` | FLOAT | `1.0` | 0.0 – 3.0 | Edge contrast boost. >1 sharpens boundaries. |
| `fill_holes_enabled` | BOOLEAN | `true` | — | Fill interior holes |
| `min_region_size` | INT | `64` | 0 – 10000 | Remove isolated regions smaller than N px |
| `multimask_output` | BOOLEAN | `true` | — | Return 3 candidates |
| `mask_index` | INT | `0` | 0 – 2 | Candidate selection |
| `score_threshold` | FLOAT | `0.0` | 0.0 – 1.0 | Minimum confidence |

**Optional:** `bbox` (BBOX), `existing_mask` (MASK), `trimap` (MASK)

#### Refinement Methods

| Method | Engine | Description |
|--------|--------|-------------|
| `auto` | — | Best available (vitmatte > multi_scale_guided > guided_filter) |
| `vitmatte` | Neural (HuggingFace) | ViTMatte neural matting — highest quality |
| `guided_filter` | Classical | Fast single-scale edge-aware smoothing |
| `multi_scale_guided` | Classical | Guided filter at 3 scales — best non-neural |
| `color_aware` | Classical | LAB-space color-sensitive refinement |
| `laplacian_blend` | Classical | Frequency-domain Laplacian pyramid blending |

#### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `refined_mask` | MASK | Compositing-grade alpha matte |
| `coarse_mask` | MASK | Raw SAM mask before refinement |
| `edge_mask` | MASK | Edge region that was refined |
| `preview` | IMAGE | Visual preview |
| `detected_bbox` | BBOX | Detected bounding box |
| `score` | FLOAT | Confidence score |
| `info` | STRING | Pipeline summary |

---

### 8. SeC + MatAnyone Pipeline (MEC)

End-to-end video pipeline: SeC/SAM segmentation → MatAnyone2 temporal alpha matting → optional edge refinement → cleanup. Best for video with occlusions, re-appearances, and complex motion.

**File:** [`nodes/sec_matanyone_pipeline.py`](../nodes/sec_matanyone_pipeline.py)  
**Category:** `MaskEditControl/Pipeline`

#### Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `image` | IMAGE | — | — | Single image or video frames (B>1) |
| `segmentation_model` | COMBO | — | Dynamic (SeC/SAM2/SAM3 models) | Coarse segmentation model |
| `text_prompt` | STRING | `""` | — | Target object text (SeC native grounding). Leave empty for point/bbox. |
| `points_json` | STRING | `"[]"` | multiline | Point prompts |
| `bbox_json` | STRING | `""` | — | Bounding box |
| `matting_backend` | COMBO | `auto` | `auto`, `matanyone2`, `vitmatte_small`, `vitmatte_base` | Alpha matting backend. `auto`: MatAnyone2 for video, ViTMatte for images. |
| `edge_radius` | INT | `15` | 1 – 200 | Edge refinement radius |
| `n_warmup` | INT | `5` | 1 – 30 | MatAnyone2 warmup frames |
| `precision` | COMBO | `fp16` | `fp16`, `bf16`, `fp32` | Segmentation precision |
| `fill_holes_enabled` | BOOLEAN | `true` | — | Fill interior holes |
| `min_region_size` | INT | `64` | 0 – 10000 | Remove small isolated regions |

**Optional:** `positive_coords`, `negative_coords` (STRING), `bbox` (BBOX), `edge_refine_method` (COMBO: `none`/`vitmatte`/`guided_filter`/`multi_scale_guided`), `keep_model_loaded` (BOOLEAN)

#### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `rgb` | IMAGE | Premultiplied foreground |
| `alpha_mask` | MASK | Compositing-grade alpha |
| `coarse_mask` | MASK | Raw segmentation before matting |
| `preview` | IMAGE | Side-by-side preview |
| `info` | STRING | Pipeline summary and timing |
