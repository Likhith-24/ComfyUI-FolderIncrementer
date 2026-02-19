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

Pinpoint-accurate mask editing suite with SAM2/SAM3 integration, independent X/Y axis control, interactive point & bounding-box editors, and video-frame mask propagation.

### Nodes

| Node | Category | Purpose |
|------|----------|---------|
| **Mask Transform XY** | Transform | Independent X/Y erode/expand, directional blur, offset, feather, threshold |
| **Points Mask Editor** | Points | Sub-pixel Gaussian point-based mask creation with interactive canvas |
| **SAM Model Loader** | SAM | Load SAM/SAM2/SAM2.1/SAM3 with VRAM offload |
| **SAM Mask Generator** | SAM | Inference with point prompts + bbox prompts, multi-mask, score threshold |
| **Mask Propagate Video** | Video | Draw mask on 1 frame → propagate via static/optical-flow/SAM2-video/fade |
| **Mask Draw Frame** | Draw | Draw circle/rect/ellipse/polygon/line with feathering + blend ops |
| **Mask Composite Advanced** | Composite | Union/intersect/subtract/XOR/blend/min/max/diff |
| **Mask Preview Overlay** | Preview | Overlay/mask-only/side-by-side/checkerboard/edge-highlight + bbox |
| **Mask Math** | Math | Gamma, contrast, quantize, power, remap, hysteresis threshold |
| **Mask Batch Manager** | Batch | Slice/pick/repeat/reverse/concat/interleave/insert/remove frames |
| **BBox Create** | BBox | Manual bbox entry |
| **BBox From Mask** | BBox | Extract tight bbox with per-axis padding |
| **BBox To Mask** | BBox | Convert bbox to rectangular mask |
| **BBox Pad** | BBox | Asymmetric padding with clamping |
| **BBox Crop** | BBox | Crop image+mask to bbox region |

### VRAM Offload

Enable **"offload_to_cpu"** in the SAM Model Loader node. The model stays on CPU RAM and is only moved to GPU during inference, then moved back. Saves ~2-4 GB VRAM.

### Points Editor Keyboard Shortcuts

| Key | Action |
|-----|--------|
| Left click | Add positive point |
| Right click | Add negative point |
| Scroll wheel | Adjust point radius |
| Ctrl+Scroll | Zoom in/out |
| Middle mouse | Pan |
| Shift+Click | Delete point |
| B | Switch to BBox mode |
| P | Switch to Points mode |
| Ctrl+Z | Undo |
| Ctrl+Shift+Z | Redo |
| Ctrl+C | Clear all |
| Delete | Delete hovered point |
| R | Reset view |

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

4. Restart ComfyUI.

## License

MIT
