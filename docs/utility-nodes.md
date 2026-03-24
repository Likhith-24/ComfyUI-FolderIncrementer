# Utility, Interactive & Diagnostics

> **Category:** `MaskEditControl/Editor` · `MaskEditControl/Preview` · `MaskEditControl/Diagnostics` · `MaskEditControl/Utils` · `utils`  
> **VRAM Tier:** 1 (pure CPU/tensor operations)

Nine nodes covering the interactive canvas editor, image comparison, mask diagnostics, parameter history, universal reroute, and folder version management.

---

## Interactive Nodes

### 1. Points Mask Editor (MEC)

Interactive canvas widget for placing point prompts and drawing bounding boxes. Click for points, Ctrl+drag for boxes — no mode switching needed. Outputs feed directly into SAM2/SAM3/SeC/ViTMatte.

**File:** [`nodes/points_mask_editor.py`](../nodes/points_mask_editor.py)  
**Category:** `MaskEditControl/Editor`

#### Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `width` | INT | `512` | 1 – 16384 | Canvas width |
| `height` | INT | `512` | 1 – 16384 | Canvas height |
| `editor_data` | STRING | `{"points":[],"bboxes":[]}` | multiline | JSON from the interactive canvas. Auto-populated by the widget. |
| `default_radius` | FLOAT | `3.0` | 0.5 – 256.0 | Default brush radius for point visualization |
| `softness` | FLOAT | `1.0` | 0.0 – 10.0 | Gaussian sigma multiplier (0 = hard circles) |
| `normalize` | BOOLEAN | `true` | — | Clamp output mask to [0, 1] |

**Optional inputs:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `reference_image` | IMAGE | Auto-sets width/height from image dimensions |
| `existing_mask` | MASK | Layer new points onto this mask |

#### Editor Data Format

```json
{
  "points": [
    {"x": 100, "y": 200, "label": 1, "radius": 5},
    {"x": 300, "y": 150, "label": 0, "radius": 3}
  ],
  "bboxes": [
    [50, 50, 400, 300, 1],
    [10, 10, 100, 100, 0]
  ]
}
```

- **Points:** `label=1` is foreground (green), `label=0` is background (red)
- **BBoxes:** `[x1, y1, x2, y2, label]` — `label=1` positive, `label=0` negative

#### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `mask` | MASK | Rendered point mask |
| `positive_coords` | STRING | Positive points JSON for SAM2 `coordinates_positive` |
| `negative_coords` | STRING | Negative points JSON for SAM2 `coordinates_negative` |
| `bboxes` | BBOX | Positive bounding boxes |
| `neg_bboxes` | BBOX | Negative bounding boxes (SAM3) |
| `points_json` | STRING | Full points JSON for SAMMaskGeneratorMEC |
| `bbox_json` | STRING | Primary bbox as `[x1,y1,x2,y2]` string |
| `primary_bbox` | BBOX | First positive bbox as [x, y, w, h] |

---

### 2. Image Comparer (MEC)

Interactive before/after comparison widget with 3 modes. Drag the slider to compare, toggle to overlay blend, or view a difference heatmap.

**File:** [`nodes/image_comparer.py`](../nodes/image_comparer.py)  
**Category:** `MaskEditControl/Preview`  
**Output Node:** Yes (terminal node — no downstream connections needed)

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image_a` | IMAGE | — | Left / "before" image |
| `image_b` | IMAGE | — | Right / "after" image |
| `label_a` | STRING | `"Before"` | _(optional)_ Label for image A |
| `label_b` | STRING | `"After"` | _(optional)_ Label for image B |

#### Modes (Toggle via widget buttons)

| Mode | Icon | Description |
|------|------|-------------|
| Compare | `◧` | Drag-slider split view |
| Overlay | `⊕` | Alpha blend of both images |
| Diff | `≠` | Difference heatmap highlighting changes |

---

## Diagnostics

### 3. Mask Failure Explainer (MEC)

Diagnose why a mask might be failing. Analyzes 5 metrics: brightness, blur, boundary contrast, color confusion, and background complexity. Outputs a human-readable explanation and suggested masking method.

**File:** [`nodes/mask_failure_explainer.py`](../nodes/mask_failure_explainer.py)  
**Category:** `MaskEditControl/Diagnostics`

#### Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `image` | IMAGE | — | — | Input image to analyze |
| `mask` | MASK | — | — | Mask to diagnose |

**Optional tuning:**

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `ring_width` | INT | `5` | 1 – 50 | Boundary ring width for contrast/color analysis |
| `blur_threshold` | FLOAT | `50.0` | 0 – 1000 | Laplacian variance threshold for blur detection |
| `brightness_threshold` | FLOAT | `0.15` | 0.0 – 1.0 | Mean brightness threshold for dark-scene detection |

#### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `explanation` | STRING | Human-readable diagnostic report |
| `problem_regions_mask` | MASK | Heatmap highlighting problematic areas |
| `severity_score` | FLOAT | Overall severity (0=fine, 1=very problematic) |
| `suggested_method` | STRING | Recommended masking approach based on analysis |

---

### 4. Parameter History (MEC)

Query the parameter history database to see what changed between runs. Useful for debugging workflow parameter drift.

**File:** [`nodes/parameter_memory.py`](../nodes/parameter_memory.py)  
**Category:** `MaskEditControl/Utils`  
**Output Node:** Yes

#### Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `mode` | COMBO | `all_history` | `all_history`, `last_run_diff`, `node_class_filter` | Query mode |
| `last_n_runs` | INT | `5` | 1 – 100 | How many recent runs to include |

**Optional:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `node_class_filter` | STRING | `""` | Filter to a specific node class (e.g. "KSampler") |
| `run_a` | INT | `0` | First run number for diff mode (0 = auto) |
| `run_b` | INT | `0` | Second run number for diff mode (0 = auto) |

#### Modes

| Mode | Description |
|------|-------------|
| `all_history` | Recent parameter changes across all nodes |
| `last_run_diff` | What changed between the last two runs |
| `node_class_filter` | Filter history to a specific node class |

#### Output

| Output | Type | Description |
|--------|------|-------------|
| `history_report` | STRING | Formatted parameter change report |

---

## Utility Nodes

### 5. Universal Reroute (MEC)

Drop onto any connection to reroute it. Auto-adapts to any type (IMAGE, MASK, LATENT, STRING, etc.). Renders as a compact dot. Right-click → "Remove Reroute (reconnect)" to dissolve.

**File:** [`nodes/universal_reroute.py`](../nodes/universal_reroute.py)  
**Category:** `MaskEditControl/Utils`

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `anything` | ANY (`*`) | Wildcard — accepts any ComfyUI type |

#### Output

| Output | Type | Description |
|--------|------|-------------|
| `output` | ANY (`*`) | Pass-through of input |

---

## Folder Incrementer Nodes

Three nodes for automatic version-managed output folders. Each queue run creates a new versioned subfolder (e.g. `output/my_project/02-22-2026/v001/`).

### 6. Folder Version Incrementer

Auto-increment version folders for each queue run. Creates dated subdirectories with zero-padded version numbers. Reads the connected node's filename to determine folder naming.

**File:** [`folder_incrementer.py`](../folder_incrementer.py)  
**Category:** `utils`  
**Output Node:** Yes

#### Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `prefix` | STRING | `"v"` | — | Prefix before version number (e.g. `"v"` → `v001`) |
| `padding` | INT | `3` | 1 – 10 | Zero-pad width (3 → 001, 4 → 0001) |
| `label` | STRING | `"default"` | — | Fallback folder name when no source file connected |
| `date_format` | COMBO | `MM-DD-YYYY` | `MM-DD-YYYY`, `DD-MM-YYYY`, `YYYY-MM-DD` | Date format for the date subfolder |

**Optional:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `trigger` | ANY | — | Connect any output — node reads the filename automatically |
| `source_filename` | STRING | `""` | Auto-filled by JS from connected node |
| `base_path` | STRING | `""` | Override base output directory. Empty = ComfyUI output dir. |

#### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `version_string` | STRING | e.g. `"v001"` |
| `version_number` | INT | e.g. `1` |
| `folder_name` | STRING | Label-derived folder name |
| `subfolder_path` | STRING | Full relative subfolder (e.g. `my_project/02-22-2026/v001`) |
| `filename_prefix` | STRING | Filename prefix for save nodes |
| `output_filename` | STRING | Complete output filename |

---

### 7. Folder Version Check

Inspect the current version counter for a label without incrementing it.

**File:** [`folder_incrementer.py`](../folder_incrementer.py)  
**Category:** `utils`  
**Output Node:** Yes

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `label` | STRING | `"default"` | Folder name to inspect |
| `date_format` | COMBO | `MM-DD-YYYY` | Must match what Folder Version Incrementer uses |

**Optional:** `trigger` (ANY), `base_path` (STRING)

#### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `status` | STRING | Human-readable status message |
| `current_version` | INT | Current version number |

---

### 8. Folder Version Set

Manually set the version counter to a specific number (creates placeholder directories up to that version).

**File:** [`folder_incrementer.py`](../folder_incrementer.py)  
**Category:** `utils`  
**Output Node:** Yes

#### Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `label` | STRING | `"default"` | — | Folder name |
| `value` | INT | `1` | 1 – 999999 | Set version counter to this value |

**Optional:** `trigger` (ANY), `prefix` (STRING, `"v"`), `padding` (INT, `3`), `base_path` (STRING), `date_format` (COMBO)

#### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `status` | STRING | Confirmation message |
| `next_version` | INT | The next version that will be used |
