# Plate Tools (MEC)

Four plate-handling nodes under `MaskEditControl/PlateTools` that cover
the most common compositing prep tasks: grain matching, stabilization,
clean-plate extraction, and difference matting.

## GrainMatchMEC

Extracts the high-frequency grain layer from a `reference` plate via
`reference - denoise(reference)` and re-applies it to a `target` image
so synthetic content matches the source plate.

The denoise step is a cheap reflect-padded box filter; the kernel size
controls how much detail counts as "grain" vs "image". Per-frame, a
random reference frame is sampled (seeded) so the resulting grain is
not temporally static. `intensity` rescales the grain before adding.

## PlateStabilizerMEC

Affine-stabilizes a video batch to its first frame.

When `cv2` is installed (already a hard dependency), an ORB feature
detector + RANSAC `estimateAffinePartial2D` is used. Without `cv2` (or
when ORB returns < 4 matches for a frame), the node falls back to FFT
phase-correlation translation, which still removes most camera shake.

The `info_json` output records the chosen backend and per-frame status
for debugging.

## CleanPlateExtractorMEC

Median across a batch with optional mask exclusion.

- No mask: per-pixel `torch.median` across the batch dimension.
- With mask: pixels where the mask is `>= 0.5` are excluded; the median
  is computed only over the remaining samples per pixel. Pixels with
  no valid samples fall back to frame 0.

Useful to derive a clean-plate from a locked-off shot in which a moving
subject has been already segmented.

## DifferenceMatteMEC

Per-pixel L1 or L2 distance between two images, threshold-converted to
a `MASK`. The `softness` parameter creates a smooth ramp around the
threshold rather than a hard step.
