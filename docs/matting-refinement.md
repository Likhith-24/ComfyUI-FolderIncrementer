# Matting & Refinement

> **Category:** `MaskEditControl/Matting` · `MaskEditControl/Refinement` · `MaskEditControl/Trimap` · `MaskEditControl/Keying`  
> **VRAM Tier:** 2 (loads ViTMatte / matting models)

Four nodes for producing compositing-grade alpha mattes from coarse segmentation masks: a unified matting hub, a standalone edge refiner, a trimap generator, and a luminance keyer.

---

## Nodes

### 1. Matting Node (MEC)

Unified alpha matting from coarse segmentation masks with 7 backends. Auto-selects the best backend based on input type (single image vs video batch).

**File:** [`nodes/matting_node.py`](../nodes/matting_node.py)  
**Category:** `MaskEditControl/Matting`

#### Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `image` | IMAGE | — | — | RGB image(s). Single frame or video batch. |
| `mask` | MASK | — | — | Coarse segmentation mask(s) to refine |
| `backend` | COMBO | `auto` | See backends table | Matting backend |
| `edge_radius` | INT | `15` | 1 – 200 | Trimap unknown-band width in pixels |
| `erode_dilate` | INT | `0` | −50 – 50 | Pre-process mask: positive = erode (shrink), negative = dilate (expand) |

**Optional inputs:**

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `trimap` | MASK | — | — | Custom trimap override (0=bg, 0.5=unknown, 1=fg) |
| `n_warmup` | INT | `5` | 1 – 30 | MatAnyone2 / RVM warmup frames |
| `sam_model` | SAM_MODEL | — | — | Required for `sam_hq` backend |

#### Backends

| Backend | Input | Description |
|---------|-------|-------------|
| `auto` | any | ViTMatte for B=1, MatAnyone2 for B>1 |
| `vitmatte_small` | image | ViTMatte Small — fast neural matting |
| `vitmatte_base` | image | ViTMatte Base — higher quality neural matting |
| `matanyone2` | video | Video matting with temporal warmup protocol |
| `rvm_mobilenetv3` | video | RobustVideoMatting (MobileNetV3) — trimap-free, human-focused |
| `rvm_resnet50` | video | RobustVideoMatting (ResNet50) — higher quality, more VRAM |
| `cutie` | video | CutIE video object segmentation (mask propagation) |
| `sam_hq` | image | HQ-SAM refinement — needs SAM model + automatic point prompts |

#### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `rgb` | IMAGE | Premultiplied image (image x alpha) |
| `alpha_mask` | MASK | Compositing-grade alpha matte |

---

### 2. ViTMatte Refiner (MEC)

Standalone mask edge refinement using 7 methods. Feed it any coarse mask + the original image to get clean, feathered edges without re-running segmentation.

**File:** [`nodes/vitmatte_refiner.py`](../nodes/vitmatte_refiner.py)  
**Category:** `MaskEditControl/Refinement`

#### Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `image` | IMAGE | — | — | Original image for edge guidance |
| `mask` | MASK | — | — | Coarse mask to refine |
| `method` | COMBO | `auto` | See methods table | Refinement method |
| `edge_radius` | INT | `10` | 1 – 200 | Radius in pixels around edges to refine |
| `edge_softness` | FLOAT | `0.5` | 0.0 – 1.0 | Edge feathering (0=sharp, 1=very soft) |
| `erode_amount` | INT | `0` | −50 – 50 | Erode/expand mask before refinement |
| `detail_level` | FLOAT | `0.8` | 0.0 – 1.0 | Fine detail preservation (hair, fur). 0=smooth, 1=max. |
| `iterations` | INT | `1` | 1 – 5 | Refinement passes. 2–3 improves difficult edges. |
| `edge_contrast_boost` | FLOAT | `1.0` | 0.5 – 3.0 | Edge contrast sharpening. >1 gives crisper boundaries. |

**Optional:** `trimap_mask` (MASK) — custom trimap for ViTMatte

#### Refinement Methods

| Method | Engine | Quality | Speed | Best For |
|--------|--------|---------|-------|----------|
| `auto` | — | — | — | Falls through: vitmatte > multi_scale_guided > guided_filter > gaussian |
| `vitmatte` | Neural | Highest | Slow | Hair, fur, translucent objects |
| `multi_scale_guided` | Classical | High | Medium | Best non-neural option; 3-scale guided filter |
| `color_aware` | Classical | High | Medium | Challenging lighting, LAB-space color sensitive |
| `guided_filter` | Classical | Good | Fast | Quick edge cleanup |
| `laplacian_blend` | Classical | Good | Medium | Frequency-domain pyramid blending |
| `gaussian_blur` | Classical | Basic | Very fast | Simple edge softening |

#### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `refined_mask` | MASK | Refined alpha matte |
| `edge_mask` | MASK | Edge region that was processed |
| `preview` | IMAGE | Before/after preview |

---

### 3. Trimap Generator (MEC)

Convert a coarse segmentation mask into a trimap (white=foreground, black=background, gray=unknown boundary). Feed into ViTMatte or other trimap-based matting.

**File:** [`nodes/trimap_generator.py`](../nodes/trimap_generator.py)  
**Category:** `MaskEditControl/Trimap`

#### Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `mask` | MASK | — | — | Coarse mask to convert |
| `edge_radius` | INT | `15` | 1 – 200 | Width of the unknown boundary region in pixels |
| `inner_erosion` | FLOAT | `1.0` | 0.1 – 3.0 | Scale factor for foreground erosion. <1 = tighter fg, >1 = wider fg. |
| `outer_dilation` | FLOAT | `1.5` | 0.5 – 5.0 | Scale factor for background dilation. >1 = wider unknown band for better edge capture. |
| `smooth` | FLOAT | `0.0` | 0.0 – 20.0 | Gaussian smoothing of trimap boundaries. Reduces staircasing. |
| `threshold` | FLOAT | `0.5` | 0.0 – 1.0 | Binarization threshold for input mask |

**Optional:** `image` (IMAGE) — reference image for edge-aware trimap where unknown regions follow image edges

#### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `trimap` | MASK | Full trimap (0=bg, 0.5=unknown, 1=fg) |
| `foreground` | MASK | Foreground region only |
| `unknown` | MASK | Unknown boundary region only |

---

### 4. Luminance Keyer (MEC)

Professional luminance keyer using ITU-R BT.709 coefficients. Extract mattes based on image brightness with presets and smooth S-curve falloff.

**File:** [`nodes/luminance_keyer.py`](../nodes/luminance_keyer.py)  
**Category:** `MaskEditControl/Keying`

#### Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `image` | IMAGE | — | — | Input image(s) |
| `mode` | COMBO | `auto` | `auto`, `highlights`, `midtones`, `shadows`, `custom` | Preset or custom thresholds |
| `low` | FLOAT | `0.0` | 0.0 – 1.0 | Low threshold (only used in `custom` mode) |
| `high` | FLOAT | `1.0` | 0.0 – 1.0 | High threshold (only used in `custom` mode) |
| `gamma` | FLOAT | `1.0` | 0.01 – 10.0 | Post-keying gamma. >1 reduces coverage, <1 expands coverage. |
| `falloff` | FLOAT | `1.0` | 0.0 – 10.0 | Transition smoothness. 0=hard binary, 1=standard, >1=very gradual. |
| `invert` | BOOLEAN | `false` | — | Swap keyed/unkeyed regions |

#### Mode Presets

| Mode | Low | High | Description |
|------|-----|------|-------------|
| `auto` | — | — | Analyzes image brightness to pick best range |
| `highlights` | 0.7 | 1.0 | Key bright regions |
| `midtones` | 0.3 | 0.7 | Key mid-range luminance |
| `shadows` | 0.0 | 0.3 | Key dark regions |
| `custom` | slider | slider | Uses `low` and `high` values directly |

#### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `mask` | MASK | Luminance matte |
| `info` | STRING | Mode used, threshold values, coverage stats |
