# VAE Tools (MEC)

Four VAE-focused nodes covering merging, latent diagnostics, and model
introspection.

## VAEMergeMEC

Merge two (or three) VAEs using one of eight algorithms:

- `weighted_sum` — `out = (1-α)·A + α·B`
- `add_difference` — `out = A + α·(B - C)` (requires `vae_c`)
- `tensor_sum` — element-wise mean of A and B
- `triple_sum` — equal mean of A, B, C (requires `vae_c`)
- `slerp` — spherical linear interpolation of flattened parameters
- `dare_ties` — DARE/TIES with sparsity drop and sign-resolution
- `block_swap` — replace whole blocks of A with B according to per-block
  weights
- `clamp_interp` — like `weighted_sum` but bounds the result to the per-tensor
  range of A and B

### Per-block alpha

Pass either a JSON object or a comma-separated list to override the
global `alpha` for individual blocks. Recognised names follow the
SD/SDXL VAE block layout (`block_conv_in`, `block_0..3`, `block_mid`,
`block_norm_out`, `block_conv_out`).

### Brightness / contrast

After the merge, two scalar tweaks can be applied to the
`decoder.conv_out` weights only — useful for nudging output luminance
without retraining.

The merged VAE is returned as a fresh `deepcopy`; the inputs are never
modified. The merge runs on CPU in float32 and is cast back to the
source dtype before being installed into the wrapper.

## VAELatentInspectorMEC

Per-channel min / max / mean / std / abs_mean stats across the latent,
plus NaN/Inf counts and an overall `verdict`:

- `corrupt` — non-zero NaN or Inf count
- `saturated` — channel abs_mean exceeds 30 (suggests clipping)
- `low_contrast` — channel std below 0.05
- `healthy` — none of the above

Use `fail_on_corrupt=True` to raise hard on corrupt latents (useful as a
sentinel in long batched runs).

## VAESimilarityAnalyserMEC

Computes cosine similarity between two VAEs:

- Globally (over all common parameters)
- Per block (using the same SD/SDXL layout as VAEMerge)
- Optionally per tensor

Reports tensors that exist only in one model — handy for debugging
mismatched checkpoints.

## VAEBlockInspectorMEC

Per-block weight statistics (mean / std / abs_mean / count) for a single
VAE. Useful to spot blocks dominated by NaN/zeros, or to compare
fine-tunes against a reference.
