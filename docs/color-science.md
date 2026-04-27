# Color Science Nodes (MEC)

Three nodes for VFX/grading workflows, all under `MaskEditControl/Color`.
None of these nodes pull in extra dependencies — math is done in pure
torch using built-in matrices and OETFs.

## ColorSpaceConvertMEC

Converts an `IMAGE` between four spaces:

| Space | Notes |
| --- | --- |
| `srgb` | sRGB-encoded (default ComfyUI image convention). |
| `linear` | Linear-light (sRGB primaries). |
| `rec709` | Rec.709 OETF-encoded. |
| `acescg` | ACEScg primaries, linear. |

Conversion always routes through linear-sRGB internally. Output is
clamped to [0,1] for display-encoded targets and unclamped for `linear`
and `acescg`.

## LUTApplyMEC

Loads an Adobe `.cube` LUT file (1D or 3D) and applies it. Trilinear
interpolation for 3D LUTs; per-channel linear interp for 1D. The LUT
parser respects `LUT_3D_SIZE`, `LUT_1D_SIZE`, `DOMAIN_MIN`, `DOMAIN_MAX`,
and `TITLE`.

A `strength` slider blends between the input and the graded result.

## ExposureGradeMEC

Photographic exposure in stops, white-balance temp/tint, and contrast
around a configurable mid-grey pivot.

By default `operate_in_linear=True`, which means the input is treated as
sRGB-encoded, linearized for the math, then re-encoded. Disable this
flag for legacy display-referred grading.
