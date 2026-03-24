# Changelog

All notable changes to ComfyUI-CustomNodePacks are documented here.

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
