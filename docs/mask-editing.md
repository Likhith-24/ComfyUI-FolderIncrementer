# Mask Editing & Transform

> **Category:** `MaskEditControl/Transform` · `MaskEditControl/Draw` · `MaskEditControl/Spline` · `MaskEditControl/Composite` · `MaskEditControl/Math` · `MaskEditControl/Batch`  
> **VRAM Tier:** 1 (pure tensor math — no models loaded)

Eight nodes for precise mask manipulation: per-axis transform, shape drawing (raw + unified), spline mask editor, compositing, math operations, batch management, and visualization.

---

## Nodes

### 1. Mask Transform XY (MEC)

Independent per-axis mask manipulation. Erode or expand along X and Y separately — unlike most ComfyUI nodes that only offer isotropic (uniform) morph.

**File:** [`nodes/mask_transform_xy.py`](../nodes/mask_transform_xy.py)  
**Category:** `MaskEditControl/Transform`

#### Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `mask` | MASK | — | — | Input mask |
| `expand_x` | INT | `0` | −512 – 512 | Horizontal expand (positive) or erode (negative) in pixels |
| `expand_y` | INT | `0` | −512 – 512 | Vertical expand (positive) or erode (negative) in pixels |
| `blur_x` | FLOAT | `0.0` | 0.0 – 128.0 | Gaussian blur sigma along X axis only |
| `blur_y` | FLOAT | `0.0` | 0.0 – 128.0 | Gaussian blur sigma along Y axis only |
| `offset_x` | INT | `0` | −4096 – 4096 | Shift mask horizontally in pixels |
| `offset_y` | INT | `0` | −4096 – 4096 | Shift mask vertically in pixels |
| `feather` | FLOAT | `0.0` | 0.0 – 128.0 | Isotropic edge feathering (applied after all other transforms) |
| `threshold` | FLOAT | `0.5` | 0.0 – 1.0 | Binarize the mask. Set to 0 to keep soft values |
| `invert` | BOOLEAN | `false` | — | Invert the final mask |

#### Output

| Output | Type | Description |
|--------|------|-------------|
| `mask` | MASK | Transformed mask |

#### Use Cases

- **Erode X only:** Thin out a mask horizontally while keeping vertical coverage (e.g., narrowing a selection along a wall edge)
- **Directional blur:** Blur only on one axis for motion-like feathering
- **Offset + threshold:** Create crisp shadow/drop masks offset from the original

---

### 2. Mask Draw Frame (MEC)

Draw 12 geometric shapes onto a mask with sub-pixel SDF rendering, feathering, rotation, and blend operations.

**File:** [`nodes/mask_draw_frame.py`](../nodes/mask_draw_frame.py)  
**Category:** `MaskEditControl/Draw`

#### Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `width` | INT | `512` | 1 – 16384 | Canvas width |
| `height` | INT | `512` | 1 – 16384 | Canvas height |
| `shape` | CHOICE | `circle` | See shapes table | Which shape to draw |
| `shape_params_json` | STRING | `{"cx":256,"cy":256,"radius":50}` | — | JSON with shape-specific parameters (see below) |
| `value` | FLOAT | `1.0` | 0.0 – 1.0 | Fill intensity for the shape |
| `feather` | FLOAT | `0.0` | 0.0 – 128.0 | Soft edge feathering |
| `rotation` | FLOAT | `0.0` | −360 – 360 | Rotation angle in degrees around the shape center |
| `operation` | CHOICE | `set` | `set` · `add` · `subtract` · `max` · `min` | How to blend the shape with the existing mask |
| `existing_mask` | MASK | _(optional)_ | — | Existing mask to draw onto (instead of blank canvas) |
| `reference_image` | IMAGE | _(optional)_ | — | Reference image for sizing |

#### Shapes and Their JSON Parameters

| Shape | JSON Parameters | Example |
|-------|----------------|---------|
| `circle` | `cx`, `cy`, `radius` | `{"cx": 256, "cy": 256, "radius": 100}` |
| `rectangle` | `x`, `y`, `w`, `h` | `{"x": 50, "y": 50, "w": 200, "h": 150}` |
| `ellipse` | `cx`, `cy`, `rx`, `ry` | `{"cx": 256, "cy": 256, "rx": 120, "ry": 80}` |
| `polygon` | `points`: array of [x,y] | `{"points": [[100,100], [200,50], [300,100]]}` |
| `line` | `x1`, `y1`, `x2`, `y2`, `thickness` | `{"x1": 10, "y1": 10, "x2": 500, "y2": 500, "thickness": 5}` |
| `triangle` | `cx`, `cy`, `size` — or `points` | `{"cx": 256, "cy": 256, "size": 100}` |
| `star` | `cx`, `cy`, `outer_r`, `inner_r`, `num_points` | `{"cx": 256, "cy": 256, "outer_r": 100, "inner_r": 40, "num_points": 5}` |
| `diamond` | `cx`, `cy`, `w`, `h` | `{"cx": 256, "cy": 256, "w": 120, "h": 180}` |
| `cross` | `cx`, `cy`, `size`, `thickness` | `{"cx": 256, "cy": 256, "size": 100, "thickness": 30}` |
| `rounded_rectangle` | `x`, `y`, `w`, `h`, `corner_radius` | `{"x": 50, "y": 50, "w": 200, "h": 150, "corner_radius": 20}` |
| `heart` | `cx`, `cy`, `size` | `{"cx": 256, "cy": 256, "size": 100}` |
| `arrow` | `cx`, `cy`, `length`, `width`, `head_length`, `head_width` | `{"cx": 256, "cy": 256, "length": 200, "width": 30, "head_length": 60, "head_width": 80}` |

#### Output

| Output | Type | Description |
|--------|------|-------------|
| `mask` | MASK | The drawn mask |

#### Tips

- **Multiple shapes:** Chain multiple Mask Draw Frame nodes, each feeding its output as the next node's `existing_mask`
- **Rotation:** All 12 shapes support rotation around their center point
- **SDF rendering:** Shapes use signed distance field math for smooth, anti-aliased edges
- **Feathering:** Works beautifully with SDF — gives a natural gradient at shape edges

---

### 3. Mask Composite Advanced (MEC)

Combine two masks using Boolean, blend, or mathematical operations. Automatically resizes masks if dimensions differ.

**File:** [`nodes/mask_composite.py`](../nodes/mask_composite.py)  
**Category:** `MaskEditControl/Composite`

#### Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `mask_a` | MASK | — | — | First mask |
| `mask_b` | MASK | — | — | Second mask |
| `operation` | CHOICE | `union` | See operations table | Compositing operation |
| `blend_factor` | FLOAT | `0.5` | 0.0 – 1.0 | Blend ratio (only for `blend` mode). 0 = all A, 1 = all B |
| `invert_a` | BOOLEAN | `false` | — | Invert mask A before operation |
| `invert_b` | BOOLEAN | `false` | — | Invert mask B before operation |
| `threshold` | FLOAT | `0.0` | 0.0 – 1.0 | Binarize the result. 0 = keep soft values |

#### Operations

| Operation | Formula | Description |
|-----------|---------|-------------|
| `union` | `max(A, B)` | Combine both masks |
| `intersect` | `min(A, B)` | Only where both masks overlap |
| `subtract` | `clamp(A − B)` | Remove B from A |
| `xor` | `A + B − 2·min(A,B)` | Either one but not both |
| `blend` | `A·(1−f) + B·f` | Weighted average using `blend_factor` |
| `min` | `min(A, B)` | Per-pixel minimum (same as intersect) |
| `max` | `max(A, B)` | Per-pixel maximum (same as union) |
| `difference` | `|A − B|` | Absolute difference |

#### Output

| Output | Type | Description |
|--------|------|-------------|
| `mask` | MASK | Composited mask |

---

### 4. Mask Math (MEC)

Apply mathematical transformations to a single mask. All outputs are clamped to [0, 1].

**File:** [`nodes/mask_math.py`](../nodes/mask_math.py)  
**Category:** `MaskEditControl/Math`

#### Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `mask` | MASK | — | — | Input mask |
| `operation` | CHOICE | `invert` | See operations table | Mathematical operation |
| `value_a` | FLOAT | `0.0` | −100 – 100 | Primary parameter (interpretation depends on operation) |
| `value_b` | FLOAT | `1.0` | −100 – 100 | Secondary parameter |

#### Operations

| Operation | `value_a` meaning | `value_b` meaning | Formula |
|-----------|-------------------|-------------------|---------|
| `add_scalar` | Value to add | _(unused)_ | `mask + a` |
| `multiply_scalar` | Multiplier | _(unused)_ | `mask × a` |
| `power` | Exponent | _(unused)_ | `mask ^ a` |
| `invert` | _(unused)_ | _(unused)_ | `1 − mask` |
| `clamp` | Lower bound | Upper bound | `clamp(mask, a, b)` |
| `remap_range` | Source min | Source max | Map `[a, b]` → `[0, 1]` |
| `quantize` | Number of levels | _(unused)_ | Snap to N discrete steps |
| `threshold_hysteresis` | Low threshold | High threshold | Dual-threshold with connected components |
| `gamma` | Gamma value | _(unused)_ | `mask ^ (1/a)` |
| `contrast` | Contrast strength | Midpoint (default 0.5) | S-curve contrast adjustment |
| `abs_diff_from_value` | Reference value | _(unused)_ | `|mask − a|` |

#### Output

| Output | Type | Description |
|--------|------|-------------|
| `mask` | MASK | Transformed mask, clamped to [0, 1] |

---

### 5. Mask Batch Manager (MEC)

Manipulate mask batches for video workflows: slice, pick specific frames, repeat, reverse, concatenate, and more.

**File:** [`nodes/mask_batch_manager.py`](../nodes/mask_batch_manager.py)  
**Category:** `MaskEditControl/Batch`

#### Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `mask` | MASK | — | — | Input mask batch |
| `operation` | CHOICE | `slice` | See operations table | Batch operation |
| `param_a` | INT | `0` | 0 – 99999 | Start frame (slice), frame index (pick/set/insert/remove), or repeat count |
| `param_b` | INT | `-1` | −1 – 99999 | End frame for slice (−1 = end). Ignored for other ops |
| `frame_indices` | STRING | `""` | — | Comma-separated frame indices for `pick_frames` mode |
| `mask_b` | MASK | _(optional)_ | — | Second mask batch for `concat`, `interleave`, `set_frame` |

#### Operations

| Operation | `param_a` | `param_b` | Description |
|-----------|-----------|-----------|-------------|
| `slice` | Start frame | End frame (−1 = end) | Extract frame range `[a:b]` |
| `pick_frames` | _(unused)_ | _(unused)_ | Select specific frames from `frame_indices` string |
| `repeat` | Repeat count | _(unused)_ | Repeat the mask batch N times |
| `reverse` | _(unused)_ | _(unused)_ | Reverse frame order |
| `concat` | _(unused)_ | _(unused)_ | Join mask + mask_b along batch dimension |
| `interleave` | _(unused)_ | _(unused)_ | Alternate frames: mask[0], mask_b[0], mask[1], mask_b[1], … |
| `set_frame` | Frame index | _(unused)_ | Replace frame at index with mask_b |
| `insert_frame` | Insert position | _(unused)_ | Insert mask_b at position |
| `remove_frame` | Frame index | _(unused)_ | Remove frame at index |

#### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `mask` | MASK | Modified mask batch |
| `count` | INT | Number of frames in the output batch |

---

### 6. Mask Preview Overlay (MEC)

Visualize masks overlaid on images with 5 display modes, customizable colors, edge highlighting, and bounding box display.

**File:** [`nodes/mask_preview.py`](../nodes/mask_preview.py)  
**Category:** `MaskEditControl/Preview`  
**Output Node:** Yes (can be used as a terminal node)

#### Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `image` | IMAGE | — | — | Background image |
| `mask` | MASK | — | — | Mask to visualize |
| `display_mode` | CHOICE | `overlay` | See modes table | Visualization mode |
| `overlay_color_r` | FLOAT | `1.0` | 0.0 – 1.0 | Red component of overlay color |
| `overlay_color_g` | FLOAT | `0.0` | 0.0 – 1.0 | Green component |
| `overlay_color_b` | FLOAT | `0.0` | 0.0 – 1.0 | Blue component |
| `opacity` | FLOAT | `0.4` | 0.0 – 1.0 | Overlay opacity |
| `edge_width` | INT | `2` | 0 – 20 | Edge contour width in pixels |
| `show_bbox` | BOOLEAN | `false` | — | Draw bounding box of the mask region |
| `bbox_color_r/g/b` | FLOAT | `0/1/0` | 0.0 – 1.0 | Bounding box color (default: green) |
| `bbox` | BBOX | _(optional)_ | — | External bbox to draw (overrides auto-detect) |

#### Display Modes

| Mode | Description |
|------|-------------|
| `overlay` | Semi-transparent colored tint over masked region |
| `mask_only` | Black and white mask only (no image) |
| `side_by_side` | Original image next to the overlay version |
| `checkerboard` | Transparent area shown as checkerboard pattern |
| `edge_highlight` | Only the mask boundary contour drawn on the image |

#### Output

| Output | Type | Description |
|--------|------|-------------|
| `preview` | IMAGE | The visualization image |

Handles batch size mismatches automatically (repeats the shorter batch to match the longer one).

---

### 7. Draw Shape (MEC)

Unified 12-shape drawing node with a single dropdown. All parameters are exposed as named inputs (no JSON editing). Irrelevant parameters are ignored per shape. Replaces the 5 legacy per-shape wrapper nodes.

**File:** [`nodes/mask_draw_frame.py`](../nodes/mask_draw_frame.py)  
**Category:** `MaskEditControl/Draw`

#### Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `width` | INT | `512` | 1 – 16384 | Canvas width |
| `height` | INT | `512` | 1 – 16384 | Canvas height |
| `shape` | CHOICE | `circle` | 12 shapes | Shape to draw |
| `cx` / `cy` | FLOAT | `256.0` | −16384 – 16384 | Center position (center-based shapes) |
| `radius` | FLOAT | `50.0` | 0 – 8192 | Circle radius; triangle/heart size |
| `size_w` / `size_h` | FLOAT | `200` / `100` | 0 – 16384 | Rectangle, rounded_rect, diamond, arrow |
| `rx` / `ry` | FLOAT | `100` / `50` | 0 – 8192 | Ellipse X/Y radii |
| `top_left_x` / `top_left_y` | FLOAT | `100` | −16384 – 16384 | Top-left or line start |
| `x2` / `y2` | FLOAT | `400` | −16384 – 16384 | Line end position |
| `thickness` | FLOAT | `5.0` | 0 – 500 | Line/cross thickness |
| `outer_r` / `inner_r` | FLOAT | `100` / `40` | 0 – 8192 | Star outer/inner radii |
| `num_points` | INT | `5` | 3 – 50 | Star/polygon point count |
| `corner_radius` | FLOAT | `20.0` | 0 – 4096 | Rounded rectangle corners |
| `cross_size` | FLOAT | `100.0` | 0 – 8192 | Cross arm length |
| `arrow_length` / `head_length` / `head_width` | FLOAT | — | 0 – 16384 | Arrow dimensions |
| `points_json` | STRING | — | multiline | Polygon vertices `[[x1,y1],...]` |
| `value` | FLOAT | `1.0` | 0.0 – 1.0 | Fill intensity |
| `feather` | FLOAT | `0.0` | 0.0 – 128.0 | Edge feathering |
| `rotation` | FLOAT | `0.0` | −360 – 360 | Rotation degrees |
| `operation` | CHOICE | `set` | `set` · `add` · `subtract` · `max` · `min` | Blend with existing mask |
| `batch_size` | INT | `1` | 1 – 256 | Number of mask frames |

**Optional:** `coords_json`, `existing_mask`, `reference_image`

#### Shape → Parameter Reference

| Shape | Uses |
|-------|------|
| `circle` | cx, cy, radius |
| `rectangle` | top_left_x, top_left_y, size_w, size_h |
| `ellipse` | cx, cy, rx, ry |
| `polygon` | points_json, num_points |
| `line` | top_left_x, top_left_y, x2, y2, thickness |
| `triangle` | cx, cy, radius |
| `star` | cx, cy, outer_r, inner_r, num_points |
| `diamond` | cx, cy, size_w, size_h |
| `cross` | cx, cy, cross_size, thickness |
| `rounded_rectangle` | top_left_x, top_left_y, size_w, size_h, corner_radius |
| `heart` | cx, cy, radius |
| `arrow` | cx, cy, arrow_length, size_w, head_length, head_width |

#### Output

| Output | Type | Description |
|--------|------|-------------|
| `mask` | MASK | Drawn shape mask |

---

### 8. Spline Mask Editor (MEC)

Interactive canvas for drawing spline-based masks directly on your image. Three interpolation modes with normalized-coordinate persistence and segment insertion.

**File:** [`nodes/spline_mask_editor.py`](../nodes/spline_mask_editor.py) + [`js/spline_mask_editor.js`](../js/spline_mask_editor.js)  
**Category:** `MaskEditControl/Spline`

#### Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `image` | IMAGE | — | — | Reference image shown in editor canvas |
| `spline_data` | STRING | `"[]"` | multiline | Internal serialized state (do not edit manually) |
| `spline_type` | CHOICE | `catmull_rom` | `catmull_rom` · `bezier` · `polyline` | Interpolation method |
| `closed` | BOOLEAN | `true` | — | Close the spline loop (filled region) |
| `smoothing` | BOOLEAN | `true` | — | Enable spline smoothing |
| `samples_per_segment` | INT | `20` | 2 – 100 | Curve resolution per segment |
| `feather_radius` | FLOAT | `0.0` | 0.0 – 64.0 | Gaussian blur on mask edge |
| `invert` | BOOLEAN | `false` | — | Fill outside spline instead of inside |

**Optional:** `width` / `height` (INT, override dimensions; 0 = match image)

#### Editor Controls

| Action | Effect |
|--------|--------|
| **Left click** | Add point (or close path by clicking first point) |
| **Shift + click** | Delete point under cursor |
| **Ctrl + click** | Insert point on nearest curve segment |
| **Right-click** | Context menu (Delete, Open/Close, Smooth/Sharp) |
| **S key** | Toggle smooth / sharp |
| **Scroll** | Zoom |
| **Middle drag** | Pan |

#### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `mask` | MASK | Rasterized filled spline region (B, H, W) |
| `coords_json` | STRING | SAM-compatible point coords from control points |
| `spline_data_out` | SPLINE_DATA | Structured data for downstream nodes |
