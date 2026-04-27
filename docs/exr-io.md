# EXR I/O Nodes (MEC)

Two nodes for reading and writing OpenEXR images, under
`MaskEditControl/IO`. Both ship with a fallback chain so they work
without extra dependencies — the EXR libraries are only loaded when
the node actually runs.

## Backend priority

1. **OpenEXR + Imath** — fastest, full feature set.
2. **imageio** (with the `freeimage` plugin if available).
3. **TIFF fallback** on save — when neither backend can write EXR, a
   16-bit TIFF is written next to the requested path with a
   `_fallback.tif` suffix and a warning logged. The original `.exr`
   file is *not* silently created.

Install the OpenEXR backend with:

```bash
pip install OpenEXR Imath
```

## LoadEXRMEC

Loads a single EXR file and returns it as `IMAGE` of shape `[1,H,W,3]`
in scene-linear float32. The R, G, B channels are pulled by name; an
EXR missing any of those channels raises a clear `ValueError`.

The `info_json` output contains `backend`, `width`, `height`, and the
file basename.

## SaveEXRMEC

Saves an `IMAGE` batch as one EXR per frame. With a single frame, the
output path is used verbatim; for batches an `_NNNN` suffix is
appended (4-digit, 1-indexed).

`half_float=True` writes EXR-half (recommended; smaller files with no
visible quality loss for SDR content). Disable for 32-bit float when
storing HDR output for compositing.

## See also

- [`EXRMetadataReaderMEC`](utility-nodes.md) — pure-python EXR header
  reader for shot-metadata workflows that only need the header
  (channels, dataWindow, custom attributes).
