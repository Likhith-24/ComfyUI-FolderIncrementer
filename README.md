# ComfyUI-CustomNodePacks

A growing collection of custom nodes for ComfyUI.

---

## 1. FolderIncrementer

Auto-incrementing version string node (`v001`, `v002`, …) for automating folder/file versioning.

- **Folder Version Incrementer** – outputs next version string + number
- **Folder Version Reset** – reset a counter to 0
- **Folder Version Set** – manually set a counter value

---

## 2. MaskEditControl (MEC)

Pinpoint-accurate mask editing suite with **SAM2 / SAM3 + ViTMatte** integration, independent X/Y axis control, unified interactive point & bounding-box editor, video-frame mask propagation, and compositing-grade alpha matting.

### Nodes (18 total)

| Node | Category | Purpose |
|------|----------|---------|
| **Mask Transform XY** | Transform | Independent X/Y erode/expand, directional blur, offset, feather, threshold |
| **Points Mask Editor** | Editor | Unified sub-pixel point & bbox editor – no mode switching needed |
| **SAM Model Loader** | SAM | Load SAM / SAM2 / SAM2.1 / SAM3 with VRAM offload |
| **SAM Mask Generator** | SAM | Iterative SAM inference with point + bbox prompts, auto-negative points |
| **SAM + ViTMatte Pipeline** | Pipeline | **Combined SAM → ViTMatte end-to-end** for maximum accuracy |
| **ViTMatte Edge Refiner** | Refinement | Multi-scale guided filter / LAB color-aware / ViTMatte alpha matting |
| **Mask Propagate Video** | Video | Draw mask on 1 frame → propagate via static / optical-flow / SAM2-video / fade |
| **Mask Draw Frame** | Draw | Draw circle / rect / ellipse / polygon / line with feathering + blend ops |
| **Mask Composite Advanced** | Composite | Union / intersect / subtract / XOR / blend / min / max / diff |
| **Mask Preview Overlay** | Preview | Overlay / mask-only / side-by-side / checkerboard / edge-highlight + bbox |
| **Mask Math** | Math | Gamma, contrast, quantize, power, remap, hysteresis threshold |
| **Mask Batch Manager** | Batch | Slice / pick / repeat / reverse / concat / interleave / insert / remove frames |
| **BBox Create** | BBox | Manual bbox entry |
| **BBox From Mask** | BBox | Extract tight bbox with per-axis padding |
| **BBox To Mask** | BBox | Convert bbox to rectangular mask |
| **BBox Pad** | BBox | Asymmetric padding with clamping |
| **BBox Crop** | BBox | Crop image + mask to bbox region |

---

### SAM + ViTMatte Pipeline (Best Masking)

The **SAM + ViTMatte Pipeline** node combines SAM segmentation with ViTMatte-quality edge refinement in a single node for the **best possible masking in any lighting condition / environment**:

```
[Load SAM Model] → [SAM + ViTMatte Pipeline] → refined_mask
                         ↑
                    [Load Image]
```

**Pipeline stages:**

1. **SAM coarse mask** – initial segmentation from your point / bbox prompts
2. **Iterative refinement** – re-run SAM 2-3× with mask-derived prompts for tighter boundaries
3. **Edge-aware matting** – ViTMatte neural matting, multi-scale guided filter, or LAB color-aware refinement
4. **Edge contrast boost** – sharpen boundaries for challenging lighting
5. **Post-processing** – fill interior holes, remove small isolated regions

**Key parameters:**

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `sam_iterations` | 2 | Number of iterative SAM passes (each tightens the mask) |
| `refine_method` | auto | Edge refinement: auto / vitmatte / multi_scale_guided / color_aware / laplacian_blend |
| `edge_radius` | 12 | Pixels around edges to refine |
| `detail_preservation` | 0.85 | How much fine detail (hair, fur, lace) to keep |
| `edge_contrast` | 1.0 | Boost edge sharpness for tricky lighting (>1 = sharper) |
| `fill_holes` | true | Fill holes inside the mask |
| `remove_small_regions` | 64 | Remove isolated noise < N pixels |

**Outputs:** `refined_mask`, `coarse_mask`, `edge_mask`, `preview` (side-by-side comparison), `detected_bbox`, `score`, `info`

---

### ViTMatte Edge Refiner (Standalone)

Use this standalone if you already have a coarse mask from any source and want to refine its edges:

| Method | Quality | Requires | Best for |
|--------|---------|----------|----------|
| `vitmatte` | ★★★★★ | `transformers` | Hair, fur, glass, complex edges |
| `multi_scale_guided` | ★★★★☆ | `opencv-python` | General high-quality refinement |
| `color_aware` | ★★★★☆ | `opencv-python` | **Challenging lighting conditions** |
| `guided_filter` | ★★★☆☆ | `opencv-python` | Fast good-quality refinement |
| `laplacian_blend` | ★★★☆☆ | `opencv-python` | Smooth frequency-domain blending |
| `gaussian_blur` | ★★☆☆☆ | (none) | Simple fallback |

New parameters: `iterations` (1-5, run refinement multiple times), `edge_contrast_boost` (0.5-3.0, sharper edges).

---

### SAM Mask Generator (Enhanced)

The standalone SAM generator now supports:

- **Iterative refinement** (`refine_iterations`, 1-5) – each pass feeds the previous mask back as logits
- **Auto-negative points** (`auto_negative_points`) – samples negative prompts from just outside the boundary
- **Existing mask input** – use a pre-existing mask as the starting point
- **All model types**: SAM, SAM2, SAM2.1, SAM3

---

### Model Support

| Model | Loader Type | Notes |
|-------|-------------|-------|
| SAM ViT-H/L/B | `sam_vit_h` / `sam_vit_l` / `sam_vit_b` | Original Meta SAM |
| SAM2 | `sam2` | Meta SAM2 |
| SAM2.1 | `sam2.1` | Meta SAM2.1 |
| SAM3 | `sam3` | SAM3 |
| Auto-detect | `auto` | Detects from filename |

Place model checkpoints in `ComfyUI/models/sams/` or `ComfyUI/models/sam2/`.

---

### VRAM Offload

Enable **"offload_to_cpu"** in the SAM Model Loader node. The model stays on CPU RAM and is only moved to GPU during inference, then moved back. Saves ~2-4 GB VRAM.

---

### Unified Points & BBox Editor

The interactive canvas editor supports **both points and bounding boxes without any mode switching**:

| Action | Effect |
|--------|--------|
| **Left click** | Add positive point (foreground) |
| **Right click** | Add negative point (background) |
| **CTRL + Left drag** | Draw positive bounding box (green) |
| **CTRL + Right drag** | Draw negative bounding box (red) |
| **Shift + Click** | Delete point or bbox under cursor |
| **Scroll wheel** | Adjust point radius |
| **CTRL + Scroll** | Zoom in/out |
| **Middle mouse drag** | Pan canvas |
| **Delete / Backspace** | Delete hovered element |
| **CTRL + Z** | Undo |
| **CTRL + Shift + Z** | Redo |
| **CTRL + C** | Clear all |
| **R** | Reset view |

The editor outputs a unified `editor_data` JSON containing both `points` and `bboxes`, directly compatible with SAM / SAM2 / SAM3.

**Image input**: Connect a `reference_image` to see it as the editor background for precise point placement.

---

### Recommended Workflow

For the **best masking quality** in any lighting / environment:

```
Load SAM2.1 Model (offload_to_cpu=true, dtype=float16)
    ↓
SAM + ViTMatte Pipeline
    • sam_iterations: 2-3
    • refine_method: auto
    • detail_preservation: 0.85
    • edge_contrast: 1.2 (for tricky lighting)
    • fill_holes: true
    ↓
refined_mask → use for compositing
```

For **fast iteration**:

```
SAM Mask Generator (refine_iterations: 2) → ViTMatte Edge Refiner (multi_scale_guided)
```

---

### SeC (Segment Concept) Interoperability

**SeC** ([Comfyui-SecNodes](https://github.com/9nate-drake/Comfyui-SecNodes)) is a state-of-the-art **video object segmentation** method that uses a Large Vision-Language Model for semantic understanding. It outperforms SAM2.1 on complex scenes (+11.8 points on SeCVOS benchmark).

SeC is a separate ComfyUI node pack — install it alongside MEC for the best combined workflow:

```
[SeC Model Loader] → [SeC Video Segmentation] → per-frame masks
                                                       ↓
                                              [ViTMatte Edge Refiner (MEC)]
                                                       ↓
                                                 refined masks
```

**Why this works:** SeC excels at **tracking objects across video frames** through occlusions and appearance changes. MEC's ViTMatte Refiner then **cleans up edges** (hair, fur, transparency) on each frame. Together they give you compositing-grade video masks.

**Format compatibility:**
- SeC outputs standard `MASK` type → feeds directly into MEC's ViTMatte Edge Refiner
- MEC's Points Editor output coordinates `{"x":..., "y":...}` match SeC's point format
- MEC mask outputs → feed into SeC's `input_mask` for guided tracking

Install SeC: `cd ComfyUI/custom_nodes && git clone https://github.com/9nate-drake/Comfyui-SecNodes`
Model: Download from [HuggingFace](https://huggingface.co/VeryAladeen/Sec-4B) → place in `ComfyUI/models/sams/`

---

## Installation

1. Clone into `ComfyUI/custom_nodes/`:
   ```
   cd ComfyUI/custom_nodes
   git clone https://github.com/Likhith-24/ComfyUI-CustomNodePacks.git
   ```

2. Install dependencies:
   ```
   pip install -r ComfyUI-CustomNodePacks/requirements.txt
   ```

3. For SAM models, place checkpoints in `ComfyUI/models/sams/` or `ComfyUI/models/sam2/`.

4. **(Optional)** For ViTMatte neural matting:
   ```
   pip install transformers pillow
   ```
   The model (~400MB) auto-downloads from HuggingFace on first use and is cached locally.
   Alternatively, place model files in `ComfyUI/models/vitmatte/` for fully offline use.

5. Restart ComfyUI.

## License

MIT
