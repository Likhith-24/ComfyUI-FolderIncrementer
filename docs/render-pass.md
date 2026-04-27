# Render-Pass Nodes (MEC)

Two nodes for working with multi-pass renderer output (V-Ray, Arnold,
Redshift, Cycles, Octane, etc.).

## MergeRenderPassesMEC

Composites a beauty pass with optional auxiliary passes:

```text
out = beauty * (ao_strength · AO + (1 - ao_strength))
    + diffuse_gain  · diffuse
    + specular_gain · specular
    + emission_gain · emission
```

All auxiliary passes are optional; missing passes contribute zero.
Inputs are auto-resized (bilinear) to the beauty resolution if needed.
This is intentionally simple — for full re-light workflows use a
dedicated tool, but for quick AO multiplies and emission boosts this
covers 90% of use cases.

## DepthOfFieldMaskMEC

Converts a depth pass into a per-pixel circle-of-confusion (CoC) mask:

```text
coc = clamp(|depth - focus_distance| / aperture, 0, 1)
```

Outputs both the CoC mask (1 = fully defocused, 0 = sharp) and an
`in_focus` mask (its complement). Use these to drive a separable blur
node for cheap depth-of-field, or as alpha layers for compositing.

The `depth_channel` selector lets you pick R / G / B / luma when the
depth pass arrives encoded in an unusual channel layout.
