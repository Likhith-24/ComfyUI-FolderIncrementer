# Changelog

All notable changes to ComfyUI-CustomNodePacks are documented here.

## [1.7.0] – 2026-04-02

### Added

- **DrawShapeMEC** – unified 12-shape drawing node with a single dropdown.
  All parameters exposed as named inputs with descriptive tooltips — no
  more raw JSON editing. Replaces the 5 legacy per-shape wrapper nodes.
- **SplineMaskEditorMEC (JS rewrite)** – complete rewrite of the spline
  editor following Olm SplineMask patterns:
  - Normalized [0,1] coordinates (resolution-independent)
  - Segment insertion via Ctrl+click near curve
  - Close path by clicking first point (highlighted orange)
  - Right-click context menu (Delete, Open/Close, Smooth/Sharp)
  - Zoom-relative point sizes and fonts
  - Property-based persistence with backward-compatible deserialization
  - Status bar with keyboard hints

### Changed

- **DrawCircleMEC, DrawRectangleMEC, DrawEllipseMEC, DrawPolygonMEC,
  DrawLineMEC** – deprecated, now thin wrappers around DrawShapeMEC.
  Kept for backward compatibility with existing workflows.
- Node count updated: 38 → 47 (44 MEC + 3 FolderIncrementer).
- README, docs, and project structure updated for all new nodes.

## [1.6.0] – 2026-03-19

### Added

- **SplineMaskEditorMEC** – interactive spline mask drawing with
  Catmull-Rom, Bezier, and polyline modes. Outputs mask, SAM-compatible
  coords, and SPLINE_DATA for downstream chaining.
- **MotionMaskTrackerMEC** – per-frame motion detection with 4 methods
  (pixel diff, optical flow, background subtraction, histogram diff),
  camera stabilization (homography/affine/translation), and post-processing.
- **BBoxSmooth** – smooth bounding-box sequences across video frames
  using moving-average or exponential smoothing.
- **stabilization_utils.py** – shared camera stabilization helpers for
  motion tracker and inpaint suite.

### Changed

- **InpaintSuite** – uses stabilization_utils for camera-stable cropping.
- **BBoxNodes** – added BBoxSmooth to the 5 existing BBox nodes.

## [1.5.0] – 2025-07-17

### Added

- **InpaintCropProMEC – Canvas Expansion** – crops near image edges now
  extend beyond bounds via `F.pad(mode="replicate")`, producing perfectly
  centered crops everywhere. Ported from lquesada/ComfyUI-Inpaint-CropAndStitch.
- **InpaintCropProMEC – optional_context_mask** – new optional input: union
  of context mask bbox with crop bbox to include surrounding context.
- **InpaintCropProMEC – Iterative Hole-Filling** – `_fill_mask_holes` rewritten
  with 14-threshold iterative approach using scipy `binary_closing` +
  `binary_fill_holes` (with pure-torch fallback). Preserves gradient values
  in soft masks.
- **Stitch Data v2 Format** – new coordinate system: `ctc_x/y/w/h`
  (crop-to-canvas) + `cto_x/y/w/h` (canvas-to-original) for perfect
  reversal of canvas expansion in stitching.

### Changed

- **InpaintStitchProMEC** – v1/v2 dispatch: v2 uses canvas coordinates
  for seamless compositing of expanded crops; v1 fallback preserved for
  backward compatibility with existing workflows.
- **InpaintPasteBackMEC** – updated for v2 stitch format with canvas
  coordinate handling; v1 fallback preserved.
- **Points Editor (JS)** – `computeSize` override prevents LiteGraph
  relayout jitter; widget height computed from actual other-widget heights
  instead of magic number; locked dimensions after image load.
- **Image Comparer (JS)** – `computeSize` override prevents resize jitter
  after image comparison loads.

### Fixed

- **unified_segmentation.py** – marked as DEPRECATED (dead module superseded
  by unified_segmentation_node.py + model_manager.py). Prevents confusion
  from divergent MODEL_REGISTRY.

## [1.4.0] – 2025-07-16

### Added

- **InpaintCropProMEC** – 3 new mask preprocessing inputs from original
  InpaintCropAndStitch: `mask_invert` (flip inpaint/keep regions),
  `mask_fill_holes` (flood-fill enclosed gaps), `mask_hipass_filter`
  (threshold out near-transparent noise).
- **MaskBatchManager** – 2 new temporal operations for video workflows:
  `smooth_temporal` (Gaussian blur along time axis to reduce flicker) and
  `reduce_flicker` (median filter across frames to remove outliers).
- **BBoxSmooth (MEC)** – new node to smooth bounding-box sequences across
  video frames using moving-average or exponential smoothing, eliminating
  jitter in tracked crops.

### Changed

- **InpaintCropProMEC** – crop centering rewrite: bbox now expands
  symmetrically around the mask center, then clamps to image bounds. Fixes
  asymmetric off-center crops when the mask is near image edges.
- **Points Mask Editor (JS)** – complete toolbar UI overhaul: brighter
  button colors, 11 px font, 36 px toolbar height, `textBaseline="middle"`
  vertical centering, 120 ms active-state flash on click, wider separators,
  higher-opacity pills.
- **Points Mask Editor (Python)** – bbox rendering rewritten to use soft
  Gaussian-edged brushes: positive bboxes use `torch.max(mask, brush)`
  (additive, preserves soft point edges), negative bboxes use multiplicative
  erase. Replaces hard `mask = 1.0` overwrite that destroyed soft blending.

### Fixed

- **utils.py** – `multi_scale_guided_refine` and `color_aware_refine` now
  handle 2D grayscale input arrays (was IndexError when `img_np.ndim == 2`).
- **model_manager.py** – float8 dtype comparison guarded with
  `hasattr(torch, "float8_e4m3fn")` to prevent AttributeError on
  PyTorch < 2.1.
- **Points Editor (JS)** – fixed 4 memory leaks: `requestAnimationFrame` now
  cancelled in `onRemoved`, resize debounce timer cleared, keyboard event
  listener properly removed, complete cleanup chain on node deletion.
- **InpaintCropProMEC** – `_resize_lanczos` fix for single-channel masks:
  squeeze channel dim to 2D for PIL, restore after (was crashing with
  `Cannot handle this data type: (1, 1, 1), |u1`).

## [1.3.1] – 2025-07-15

### Added

- **InpaintPasteBackMEC** – new lightweight node to paste inpainted crop back
  onto the original image using stitch_data, with optional Gaussian-feathered
  alpha blending at the crop boundary.
- **InpaintCropProMEC** – `downscale_method` and `upscale_method` inputs
  with 5 interpolation modes: lanczos (PIL-based, highest quality), bicubic,
  bilinear, nearest-exact, area. Upscale method is stored in stitch_data and
  automatically used by InpaintStitchProMEC.

### Changed

- **InpaintCropProMEC** – `padding_multiple` now enforces step=2, min=2 to
  guarantee even-valued padding (required by many tiled/diffusion models).
- **Image Comparer (JS)** – mode labels now use Unicode icons
  (◧ Compare, ⊕ Overlay, ≠ Diff); labels fade during drag; divider grip
  uses a dot-grid pattern; overlay scrubber upgraded to rounded bar + circular
  handle.

### Fixed

- **InpaintStitchProMEC** – now reads `upscale_method` from stitch_data
  (defaulting to lanczos) instead of always using bilinear.

## [1.3.0] – 2025-07-14

### Added

- **ImageComparerMEC** – new interactive before/after comparison node with three
  view modes: drag-slider, adjustable-opacity overlay, and amplified difference
  heatmap. Rendered entirely in-node via a DOM canvas widget.
- **MaskDrawFrame** – 7 new shapes: triangle, star, diamond, cross,
  rounded_rectangle, heart, arrow (total: 12 shapes).
- **MaskDrawFrame** – `rotation` parameter (−360 ° to +360 °) applies to every
  shape including the original five.
- **SAMMaskGeneratorMEC** – `negative_text_prompt` input: describe what to
  exclude; GroundingDINO detects the region and injects negative points into SAM
  inference.
- **InpaintCropProMEC** – `downscale_factor` (0.25–1.0) to shrink the crop
  before sending to the inpainting model and automatically upscale on stitch.
- **InpaintCropProMEC** – `mask_blur` / `mask_grow` pre-processing: blur and
  morphologically dilate/erode the mask before computing the crop region.
- **InpaintCropProMEC** – `aspect_ratio` selector with 12 presets (1:1, 4:3,
  16:9, …) plus custom ratio; crop region is adjusted to match the chosen AR.
- **InpaintCropProMEC** – 6th output `crop_mask`: binary mask showing the
  cropped region in original image space.

### Fixed

- **Points/BBox Editor (JS)** – eliminated canvas jitter caused by
  `updateEditorSize()` re-entrantly calling `node.setSize()` in a
  ResizeObserver loop. Added 50 ms debounce, integer-snap for pixel positions,
  and a re-entrance guard.
- **Points/BBox Editor (JS)** – toolbar separators and pill buttons now render
  at integer pixel positions, removing sub-pixel aliasing artefacts.

### Changed

- **Model Manager** – `.safetensors` format is now preferred over `.pt`/`.pth`/
  `.bin` when a safetensors variant exists locally. SAM3 (`sam3.pt`) is
  explicitly excluded from this mapping.
- **InpaintCropProMEC** – crop dimensions are now snapped up to the nearest
  `padding_multiple` to avoid size mismatches with tiled models.

## [1.2.6] – 2025-07-13

### Fixed

- GitHub Actions publish workflow: added `FORCE_JAVASCRIPT_ACTIONS_TO_NODE24`
  env var and fork guard.
- Added missing `MIT-License` file required by PyPI metadata.

### Changed

- README overview table expanded from 2 to 4 packs.
