"""
MEC Paint Suite
===============

Four nodes that, together, form an interactive paint → fix → refine → build
pipeline inspired by Forbidden Vision but rebuilt around MEC's mask math:

  1. ``MECAdvancedPaintCanvas``  – interactive canvas + Nuke-style mask math
  2. ``MECContextInpainter``    – crop / inpaint / blend with smart logic
  3. ``MECToneRefiner``         – exposure + colour rescue + fake DOF
  4. ``MECBuilderSampler``      – KSampler with adaptive CFG + polish pass

The JS canvas (``js/mec_advanced_paint.js``) writes its drawing into a hidden
``canvas_data`` STRING widget as a base64 PNG.  The Python node decodes that
string into RGBA, optionally composites it on top of ``reference_image``, and
then runs the full Nuke-style mask pipeline (hardness → expansion → blur).
"""
from __future__ import annotations

import base64
import io
import math
import re
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F

try:
    import cv2  # type: ignore
    _HAS_CV2 = True
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore
    _HAS_CV2 = False

try:
    from scipy import ndimage as _ndi  # type: ignore
    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    _ndi = None  # type: ignore
    _HAS_SCIPY = False

try:  # PIL is part of ComfyUI's runtime — used to decode the JS PNG payload.
    from PIL import Image
    _HAS_PIL = True
except Exception:  # pragma: no cover
    Image = None  # type: ignore
    _HAS_PIL = False

# Comfy core (only imported lazily inside methods that need it so this module
# can still be imported in test contexts).


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  small numeric helpers                                                ║
# ╚══════════════════════════════════════════════════════════════════════╝
def _to_bhwc(img: torch.Tensor) -> torch.Tensor:
    """Accept (H,W,C), (B,H,W,C) or (B,C,H,W) and return (B,H,W,C) float."""
    if img.dim() == 3:
        img = img.unsqueeze(0)
    if img.dim() != 4:
        raise ValueError(f"image tensor must be 3-D or 4-D, got {tuple(img.shape)}")
    if img.shape[1] in (1, 3, 4) and img.shape[-1] not in (1, 3, 4):
        img = img.permute(0, 2, 3, 1).contiguous()
    return img.float().clamp(0.0, 1.0)


def _to_mask(mask: torch.Tensor) -> torch.Tensor:
    """Coerce any mask-shaped input to (B,H,W) float in [0,1]."""
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    elif mask.dim() == 4:
        if mask.shape[-1] == 1:
            mask = mask[..., 0]
        elif mask.shape[1] == 1:
            mask = mask[:, 0]
        else:
            mask = mask.mean(dim=-1)
    return mask.float().clamp(0.0, 1.0)


def _gaussian_blur_2d(t: torch.Tensor, radius: float) -> torch.Tensor:
    """Separable Gaussian blur on a (B,1,H,W) tensor with float radius (px)."""
    if radius <= 0.0:
        return t
    sigma = max(radius / 2.0, 1e-3)
    ksize = max(3, int(2 * round(3 * sigma) + 1))
    half = ksize // 2
    x = torch.arange(-half, half + 1, dtype=t.dtype, device=t.device)
    g = torch.exp(-(x ** 2) / (2.0 * sigma * sigma))
    g = g / g.sum()
    g_x = g.view(1, 1, 1, ksize)
    g_y = g.view(1, 1, ksize, 1)
    out = F.conv2d(t, g_x, padding=(0, half))
    out = F.conv2d(out, g_y, padding=(half, 0))
    return out


def _morph_2d(mask_2d: np.ndarray, pixels: int) -> np.ndarray:
    """Morphological dilate (>0) / erode (<0) by ``pixels`` pixels.

    Falls back to scipy or a torch maxpool implementation if cv2 is missing.
    Input/output is a float32 ``(H,W)`` array in [0,1].
    """
    if pixels == 0:
        return mask_2d
    radius = abs(int(pixels))
    if _HAS_CV2:
        ksize = 2 * radius + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        binary = (mask_2d > 0.5).astype(np.uint8) * 255
        if pixels > 0:
            out = cv2.dilate(binary, kernel, iterations=1)
        else:
            out = cv2.erode(binary, kernel, iterations=1)
        return (out.astype(np.float32) / 255.0)
    if _HAS_SCIPY:
        binary = mask_2d > 0.5
        if pixels > 0:
            out = _ndi.binary_dilation(binary, iterations=radius)
        else:
            out = _ndi.binary_erosion(binary, iterations=radius)
        return out.astype(np.float32)
    # last-resort torch maxpool dilation (erosion = invert/dilate/invert)
    t = torch.from_numpy(mask_2d).unsqueeze(0).unsqueeze(0).float()
    if pixels > 0:
        out = F.max_pool2d(t, 2 * radius + 1, stride=1, padding=radius)
    else:
        out = 1.0 - F.max_pool2d(1.0 - t, 2 * radius + 1, stride=1, padding=radius)
    return out.squeeze().numpy()


def _np_to_bhwc(arr: np.ndarray) -> torch.Tensor:
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.ndim == 3:
        arr = arr[None, ...]
    return torch.from_numpy(np.ascontiguousarray(arr.astype(np.float32))).clamp(0.0, 1.0)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Node 1 — Advanced Paint Canvas                                       ║
# ╚══════════════════════════════════════════════════════════════════════╝
class MECAdvancedPaintCanvas:
    """Interactive paint canvas with Nuke-style procedural mask math.

    The JS widget posts a base64 PNG (RGBA) into the hidden ``canvas_data``
    string widget on every serialise.  Python decodes it, optionally composites
    over ``reference_image``, then derives ``processed_mask`` from the alpha
    channel of the painted strokes through four ordered stages:

      1. **Raw mask**:        ``alpha`` of painted pixels, normalised to [0,1].
      2. **mask_hardness**:   anything brighter than ``(1 - hardness)`` is
         clamped to 1.0 — produces a solid inner core whose width is exactly
         the user-selected hardness fraction of the brush profile.
      3. **mask_expansion**:  morphological dilate / erode in pixels.  Negative
         shrinks, positive grows.  Uses cv2 ellipse kernel when available.
      4. **mask_blur**:       Gaussian blur of ``radius`` px, then linearly
         interpolated against the hard mask by ``mask_blur_strength``
         (0 = hard, 1 = full blur).
    """

    CATEGORY = "MEC/Paint"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("painted_image", "processed_mask")
    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "canvas_width":  ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8}),
                "canvas_height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8}),
                "brush_type":     (["paint", "mask_only"], {"default": "paint"}),
                "brush_color":    ("STRING", {"default": "#000000"}),
                "brush_opacity":  ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "brush_hardness": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                "brush_size":     ("INT",   {"default": 20,  "min": 1,   "max": 500, "step": 1}),
                "mask_hardness":     ("FLOAT", {"default": 0.5, "min": 0.0,    "max": 1.0,   "step": 0.01}),
                "mask_expansion":    ("INT",   {"default": 0,   "min": -100,   "max": 100,   "step": 1}),
                "mask_blur_radius":  ("FLOAT", {"default": 0.0, "min": 0.0,    "max": 100.0, "step": 0.1}),
                "mask_blur_strength":("FLOAT", {"default": 1.0, "min": 0.0,    "max": 1.0,   "step": 0.01}),
                "canvas_data":    ("STRING", {"multiline": False, "default": ""}),
            },
            "optional": {
                "reference_image": ("IMAGE",),
            },
        }

    # ----- decode the JS payload -------------------------------------
    @staticmethod
    def _decode_canvas(data: str, w: int, h: int) -> np.ndarray:
        """Decode the hidden ``canvas_data`` STRING into an HxWx4 float array.

        Empty / invalid payloads gracefully fall back to a fully-transparent
        canvas so the node never errors during the very first execution before
        the user has painted anything.
        """
        empty = np.zeros((h, w, 4), dtype=np.float32)
        if not data:
            return empty
        try:
            payload = data.split(",", 1)[1] if data.startswith("data:") else data
            raw = base64.b64decode(payload)
            if not _HAS_PIL:
                return empty
            with Image.open(io.BytesIO(raw)) as im:
                im = im.convert("RGBA")
                if im.size != (w, h):
                    im = im.resize((w, h), Image.BILINEAR)
                arr = np.asarray(im, dtype=np.float32) / 255.0
            return arr
        except Exception:
            return empty

    # ----- mask pipeline (Nuke-style) --------------------------------
    @staticmethod
    def _process_mask(alpha: np.ndarray,
                      hardness: float,
                      expansion: int,
                      blur_radius: float,
                      blur_strength: float) -> np.ndarray:
        m = np.clip(alpha.astype(np.float32), 0.0, 1.0)

        # 2 — hardness: any pixel above (1 - hardness) becomes a solid 1.0
        if hardness > 0.0:
            thresh = 1.0 - float(hardness)
            core = (m >= thresh).astype(np.float32)
            m = np.maximum(m, core)

        # 3 — expansion (morph)
        if expansion != 0:
            m = _morph_2d(m, int(expansion))

        # 4 — blur + lerp
        if blur_radius > 0.0 and blur_strength > 0.0:
            t = torch.from_numpy(m)[None, None, ...].float()
            blurred = _gaussian_blur_2d(t, float(blur_radius)).squeeze().numpy()
            m = (1.0 - float(blur_strength)) * m + float(blur_strength) * blurred
        return np.clip(m, 0.0, 1.0)

    # ----- main --------------------------------------------------------
    def execute(self, canvas_width, canvas_height, brush_type, brush_color,
                brush_opacity, brush_hardness, brush_size,
                mask_hardness, mask_expansion, mask_blur_radius,
                mask_blur_strength, canvas_data, reference_image=None):

        w = int(canvas_width)
        h = int(canvas_height)

        rgba = self._decode_canvas(canvas_data, w, h)            # (H,W,4) [0..1]
        rgb = rgba[..., :3]
        alpha = rgba[..., 3]

        # background (reference image or white)
        if reference_image is not None:
            ref = _to_bhwc(reference_image)[0].cpu().numpy()
            if ref.shape[:2] != (h, w):
                ref_t = torch.from_numpy(ref).permute(2, 0, 1).unsqueeze(0)
                ref_t = F.interpolate(ref_t, size=(h, w), mode="bilinear", align_corners=False)
                ref = ref_t.squeeze(0).permute(1, 2, 0).numpy()
            ref = ref[..., :3]
        else:
            ref = np.ones((h, w, 3), dtype=np.float32)

        # painted image: alpha-composite the strokes over the background.  In
        # ``mask_only`` mode the user only wants the mask, so ``painted_image``
        # is identical to the reference (or blank white) image.
        if brush_type == "mask_only":
            painted = ref
        else:
            a = alpha[..., None]
            painted = rgb * a + ref * (1.0 - a)

        proc = self._process_mask(
            alpha,
            float(mask_hardness),
            int(mask_expansion),
            float(mask_blur_radius),
            float(mask_blur_strength),
        )

        painted_t = _np_to_bhwc(painted)
        mask_t = torch.from_numpy(proc.astype(np.float32))[None, ...]
        return (painted_t, mask_t)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Node 2 — Context Inpainter (Fixer)                                  ║
# ╚══════════════════════════════════════════════════════════════════════╝
def _parse_wildcard_prompt(prompt: str, n_regions: int) -> List[str]:
    """Parse ``[SEP]`` / ``[SKIP]`` / ``[ASC]`` / ``[DSC]`` markers.

    Returns a list of ``n_regions`` prompts, one per region.  ``[SKIP]`` keeps
    the original prompt; ``[ASC]`` / ``[DSC]`` only flip the *order* in which
    the segments are zipped against region indices.  Missing segments are
    padded with the last available segment (or the entire prompt if no [SEP]).
    """
    if not prompt:
        return [""] * n_regions
    order = "ASC"
    p = prompt
    if "[DSC]" in p:
        order = "DSC"
        p = p.replace("[DSC]", "")
    if "[ASC]" in p:
        order = "ASC"
        p = p.replace("[ASC]", "")
    parts = [s.strip() for s in re.split(r"\[SEP\]", p)]
    parts = [s for s in parts if s != ""]
    if not parts:
        return [""] * n_regions
    if order == "DSC":
        parts = list(reversed(parts))
    out: List[str] = []
    for i in range(n_regions):
        seg = parts[i] if i < len(parts) else parts[-1]
        if "[SKIP]" in seg:
            out.append("")
        else:
            out.append(seg)
    return out


def _label_regions(mask_2d: np.ndarray) -> Tuple[np.ndarray, int]:
    """Connected-component labelling with cv2 / scipy / pure-python fallback."""
    binary = (mask_2d > 0.5).astype(np.uint8)
    if _HAS_CV2:
        n, labels = cv2.connectedComponents(binary, connectivity=8)
        return labels.astype(np.int32), max(int(n) - 1, 0)
    if _HAS_SCIPY:
        labels, n = _ndi.label(binary)  # type: ignore
        return labels.astype(np.int32), int(n)
    # last-resort: treat the entire mask as one region
    return binary.astype(np.int32), 1 if binary.any() else 0


class MECContextInpainter:
    """Smart-blend inpainted output back over the original image.

    The math is laid out in the order it runs:

      * **crop_padding** – extends the masked bbox by a multiplier so the
        inpaint sees context.
      * **mask_expansion_blend** – per-blend dilate/erode on the *blend* mask
        only (not the inpaint mask).
      * **blend_softness** – Gaussian feather on that expanded mask.
      * **enable_color_correction** – matches the inpainted region's
        per-channel mean / std to the original region's mean / std (Reinhard).
      * **enable_lightness_rescue** -- CIE LAB L-channel comparison.  If the
        inpainted region is >5 % darker, lerp L upwards by the deficit.
      * **enable_differential_diffusion** – ``|orig − inpaint|`` per pixel is
        used as a soft preservation weight, so unchanged pixels are kept.
      * **sampling_mask_blur*** – additional blur applied to the *output*
        ``debug_mask`` (used when feeding back into another sampler).
    """

    CATEGORY = "MEC/Paint"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("blended_image", "debug_mask")
    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image":   ("IMAGE",),
                "mask":             ("MASK",),
                "inpainted_image":  ("IMAGE",),
                "crop_padding":     ("FLOAT",  {"default": 1.2, "min": 1.0, "max": 2.0, "step": 0.01}),
                "blend_softness":   ("FLOAT",  {"default": 8.0, "min": 0.0, "max": 200.0, "step": 0.5}),
                "mask_expansion_blend": ("INT",  {"default": 0, "min": -100, "max": 100, "step": 1}),
                "enable_color_correction":     ("BOOLEAN", {"default": True}),
                "enable_lightness_rescue":     ("BOOLEAN", {"default": True}),
                "enable_differential_diffusion":("BOOLEAN", {"default": False}),
                "sampling_mask_blur_size":     ("INT",   {"default": 21, "min": 0, "max": 201, "step": 1}),
                "sampling_mask_blur_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "face_positive_prompt": ("STRING", {"multiline": True, "default": ""}),
                "face_negative_prompt": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    # ---- core math ------------------------------------------------
    @staticmethod
    def _color_match(orig: np.ndarray, fake: np.ndarray, m: np.ndarray) -> np.ndarray:
        """Per-channel mean/std match (Reinhard) constrained to mask region."""
        out = fake.copy()
        sel = m > 0.05
        if not sel.any():
            return out
        for c in range(3):
            o = orig[..., c][sel]
            f = fake[..., c][sel]
            mo, so = float(o.mean()), float(o.std() + 1e-6)
            mf, sf = float(f.mean()), float(f.std() + 1e-6)
            out[..., c] = ((fake[..., c] - mf) * (so / sf) + mo).clip(0.0, 1.0)
        return out

    @staticmethod
    def _lightness_rescue(orig: np.ndarray, fake: np.ndarray, m: np.ndarray) -> np.ndarray:
        """Lift L channel of ``fake`` toward ``orig`` if it is >5% darker."""
        if not _HAS_CV2:
            # luminance-only fallback (BT.709)
            lo = (orig * np.array([0.2126, 0.7152, 0.0722])).sum(-1)
            lf = (fake * np.array([0.2126, 0.7152, 0.0722])).sum(-1)
            sel = m > 0.05
            if not sel.any():
                return fake
            d = float(lo[sel].mean() - lf[sel].mean())
            if d <= 0.05:
                return fake
            return np.clip(fake + d * m[..., None], 0.0, 1.0)
        lab_o = cv2.cvtColor((orig * 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
        lab_f = cv2.cvtColor((fake * 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
        sel = m > 0.05
        if not sel.any():
            return fake
        # L is in 0..255 in OpenCV's 8-bit LAB; 5 % is ~12.75
        lo, lf = lab_o[..., 0], lab_f[..., 0]
        d = float(lo[sel].mean() - lf[sel].mean())
        if d <= 12.75:
            return fake
        lab_f[..., 0] = np.clip(lab_f[..., 0] + d * m, 0.0, 255.0)
        out = cv2.cvtColor(lab_f.astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
        return out

    @staticmethod
    def _differential_weight(orig: np.ndarray, fake: np.ndarray) -> np.ndarray:
        """High weight where pixels actually changed, low where they didn't."""
        d = np.abs(fake - orig).max(axis=-1)
        # rescale per-image so the brightest delta is 1
        m = d.max()
        if m > 1e-6:
            d = d / m
        return d.astype(np.float32)

    # ---- main ------------------------------------------------------
    def execute(self, original_image, mask, inpainted_image,
                crop_padding, blend_softness, mask_expansion_blend,
                enable_color_correction, enable_lightness_rescue,
                enable_differential_diffusion,
                sampling_mask_blur_size, sampling_mask_blur_strength,
                face_positive_prompt, face_negative_prompt):
        orig = _to_bhwc(original_image)[0].cpu().numpy()
        fake = _to_bhwc(inpainted_image)[0].cpu().numpy()
        H, W = orig.shape[:2]
        m = _to_mask(mask)[0].cpu().numpy()
        if m.shape != (H, W):
            m_t = torch.from_numpy(m)[None, None].float()
            m_t = F.interpolate(m_t, size=(H, W), mode="bilinear", align_corners=False)
            m = m_t.squeeze().numpy()

        if fake.shape[:2] != (H, W):
            f_t = torch.from_numpy(fake).permute(2, 0, 1).unsqueeze(0)
            f_t = F.interpolate(f_t, size=(H, W), mode="bilinear", align_corners=False)
            fake = f_t.squeeze(0).permute(1, 2, 0).numpy()

        # ---- crop window with padding (purely informative — used for the
        # debug mask and for region detection) -----------------------------
        ys, xs = np.where(m > 0.05)
        if ys.size > 0 and xs.size > 0:
            y0, y1 = int(ys.min()), int(ys.max())
            x0, x1 = int(xs.min()), int(xs.max())
            cy = (y0 + y1) * 0.5
            cx = (x0 + x1) * 0.5
            hh = (y1 - y0) * float(crop_padding) * 0.5
            ww = (x1 - x0) * float(crop_padding) * 0.5
            y0 = max(0, int(cy - hh)); y1 = min(H - 1, int(cy + hh))
            x0 = max(0, int(cx - ww)); x1 = min(W - 1, int(cx + ww))
        else:
            y0 = x0 = 0; y1 = H - 1; x1 = W - 1

        # ---- region split for wildcard prompts ---------------------------
        labels, n_reg = _label_regions(m)
        pos_prompts = _parse_wildcard_prompt(face_positive_prompt, max(n_reg, 1))
        neg_prompts = _parse_wildcard_prompt(face_negative_prompt, max(n_reg, 1))
        # We don't actually run a sampler here; we expose the parsed prompts
        # via the node's print output (deterministic, observable in console).
        if n_reg > 0:
            print(f"[MEC ContextInpainter] {n_reg} mask regions detected.")
            for i in range(n_reg):
                print(f"  region {i}: + {pos_prompts[i]!r}  − {neg_prompts[i]!r}")

        # ---- colour correction & lightness rescue ------------------------
        out = fake
        if enable_color_correction:
            out = self._color_match(orig, out, m)
        if enable_lightness_rescue:
            out = self._lightness_rescue(orig, out, m)

        # ---- differential diffusion --------------------------------------
        if enable_differential_diffusion:
            w = self._differential_weight(orig, fake)
            blend = np.clip(m * np.maximum(w, 0.05), 0.0, 1.0)
        else:
            blend = m.copy()

        # blend mask: expansion + softness
        if int(mask_expansion_blend) != 0:
            blend = _morph_2d(blend, int(mask_expansion_blend))
        if float(blend_softness) > 0.0:
            t = torch.from_numpy(blend)[None, None].float()
            blend = _gaussian_blur_2d(t, float(blend_softness)).squeeze().numpy()
        blend = np.clip(blend, 0.0, 1.0)

        # final composite
        b = blend[..., None]
        composite = out * b + orig * (1.0 - b)
        composite = np.clip(composite, 0.0, 1.0)

        # debug mask: optionally blurred sampling mask
        dbg = m
        if int(sampling_mask_blur_size) > 0 and float(sampling_mask_blur_strength) > 0.0:
            t = torch.from_numpy(dbg)[None, None].float()
            blurred = _gaussian_blur_2d(t, float(sampling_mask_blur_size) / 2.0).squeeze().numpy()
            dbg = (1.0 - float(sampling_mask_blur_strength)) * dbg + \
                  float(sampling_mask_blur_strength) * blurred

        return (_np_to_bhwc(composite), torch.from_numpy(dbg.astype(np.float32))[None, ...])


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Node 3 — Tone Refiner                                                ║
# ╚══════════════════════════════════════════════════════════════════════╝
class MECToneRefiner:
    """Auto-correct tone, optional upscale and centre-focus DOF.

    The neural-corrector is *not* a learned model — it's a deterministic
    histogram-driven exposure / black-level / gray-world correction whose
    effect is gated by the user blend factors.  It runs in three stages:

      1. **Black/white-point lift**: percentile-based remapping of darkest 1 %
         to 0 and brightest 99 % to 1, with a smoothstep curve to avoid
         clipped highlights.
      2. **Gray-world colour balance**: divide each channel by its mean and
         renormalise so the overall mean equals the channel-wide mean.
      3. **Tone / colour blend**: lerp between original and corrected.

    DOF is a fake center-focus blur — depth = 1 at centre, 0 at corners,
    raised to a power that matches ``ai_dof_focus_depth``.
    """

    CATEGORY = "MEC/Paint"
    RETURN_TYPES = ("IMAGE", "LATENT")
    RETURN_NAMES = ("refined_image", "refined_latent")
    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":             ("IMAGE",),
                "neural_corrector":  ("BOOLEAN", {"default": True}),
                "corrector_tone":    ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01}),
                "corrector_color":   ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.01}),
                "enable_upscale":    ("BOOLEAN", {"default": False}),
                "upscale_factor":    ("FLOAT", {"default": 1.5, "min": 1.0, "max": 4.0, "step": 0.05}),
                "ai_enable_dof":     ("BOOLEAN", {"default": False}),
                "ai_dof_strength":   ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05}),
                "ai_dof_focus_depth":("FLOAT", {"default": 0.7, "min": 0.5, "max": 0.99, "step": 0.01}),
            },
            "optional": {
                "latent": ("LATENT",),
                "vae":    ("VAE",),
            },
        }

    @staticmethod
    def _auto_tone(img: np.ndarray) -> np.ndarray:
        """Percentile-based black/white-point with smoothstep curve."""
        out = img.copy()
        for c in range(3):
            ch = out[..., c]
            lo = float(np.percentile(ch, 1.0))
            hi = float(np.percentile(ch, 99.0))
            if hi - lo < 1e-3:
                continue
            n = np.clip((ch - lo) / (hi - lo), 0.0, 1.0)
            n = n * n * (3.0 - 2.0 * n)        # smoothstep
            out[..., c] = n
        return out.clip(0.0, 1.0)

    @staticmethod
    def _gray_world(img: np.ndarray) -> np.ndarray:
        means = img.reshape(-1, 3).mean(axis=0) + 1e-6
        target = float(means.mean())
        scale = target / means
        return np.clip(img * scale[None, None, :], 0.0, 1.0)

    @staticmethod
    def _dof(img: np.ndarray, strength: float, focus: float) -> np.ndarray:
        H, W = img.shape[:2]
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        cx, cy = W * 0.5, H * 0.5
        r = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2)
        r = np.clip(r, 0.0, 1.0)
        # focus = 1 means strong centre-only focus, focus = 0.5 = wide.
        power = 1.0 + (float(focus) - 0.5) * 8.0
        coc = np.clip(r ** power, 0.0, 1.0)
        coc = coc * float(strength)
        radius = max(0.5, 12.0 * float(strength))
        t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
        blurred = _gaussian_blur_2d(t, radius).squeeze(0).permute(1, 2, 0).numpy()
        return np.clip(img * (1.0 - coc[..., None]) + blurred * coc[..., None], 0.0, 1.0)

    def execute(self, image, neural_corrector, corrector_tone, corrector_color,
                enable_upscale, upscale_factor, ai_enable_dof,
                ai_dof_strength, ai_dof_focus_depth, latent=None, vae=None):
        img = _to_bhwc(image)[0].cpu().numpy()
        out = img

        if neural_corrector:
            toned = self._auto_tone(out)
            colored = self._gray_world(toned)
            out = (1.0 - float(corrector_tone)) * out + float(corrector_tone) * toned
            out = (1.0 - float(corrector_color)) * out + float(corrector_color) * colored
            out = np.clip(out, 0.0, 1.0)

        if ai_enable_dof:
            out = self._dof(out, float(ai_dof_strength), float(ai_dof_focus_depth))

        if enable_upscale and float(upscale_factor) > 1.0:
            t = torch.from_numpy(out).permute(2, 0, 1).unsqueeze(0)
            new_h = int(round(out.shape[0] * float(upscale_factor)))
            new_w = int(round(out.shape[1] * float(upscale_factor)))
            t = F.interpolate(t, size=(new_h, new_w), mode="bicubic", align_corners=False)
            out = t.squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()

        out_img = _np_to_bhwc(out)

        # Latent passthrough or re-encode
        if latent is not None:
            out_lat = latent
        elif vae is not None:
            try:
                samples = vae.encode(out_img[:, :, :, :3])
                out_lat = {"samples": samples}
            except Exception:
                out_lat = {"samples": torch.zeros(1, 4, max(out_img.shape[1] // 8, 1),
                                                       max(out_img.shape[2] // 8, 1))}
        else:
            out_lat = {"samples": torch.zeros(1, 4, max(out_img.shape[1] // 8, 1),
                                                   max(out_img.shape[2] // 8, 1))}
        return (out_img, out_lat)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Node 4 — Builder Sampler                                             ║
# ╚══════════════════════════════════════════════════════════════════════╝
def _build_cfg_schedule(mode: str, start: float, finish: float, pivot: float, steps: int) -> List[float]:
    """Per-step CFG schedule.

    * **Constant** – every step uses ``start``.
    * **Linear** – linear ramp ``start → finish`` over ``steps``.
    * **Ease Down** – early steps fall fast toward ``pivot`` (cubic ease-out
      to 70 % of ``steps``) then slow ease from ``pivot → finish`` for the
      remainder.  Used to keep the early steps under firm guidance and let the
      late steps relax for fine detail.
    """
    steps = max(int(steps), 1)
    if mode == "Constant":
        return [float(start)] * steps
    if mode == "Linear":
        return [float(start) + (float(finish) - float(start)) * (i / max(steps - 1, 1))
                for i in range(steps)]
    # Ease Down
    out: List[float] = []
    knee = max(int(round(steps * 0.7)), 1)
    for i in range(steps):
        if i < knee:
            t = i / max(knee - 1, 1)
            t = 1.0 - (1.0 - t) ** 3      # cubic ease-out
            v = float(start) + (float(pivot) - float(start)) * t
        else:
            t = (i - knee) / max(steps - knee - 1, 1)
            t = t ** 2                     # ease-in toward finish
            v = float(pivot) + (float(finish) - float(pivot)) * t
        out.append(float(v))
    return out


class _CFGScheduler:
    """Closure object that swaps the model's CFG per sampling step.

    ComfyUI's ``ksampler`` exposes a ``set_model_sampler_cfg_function`` hook on
    the ModelPatcher which receives ``(args)`` where ``args["cond_scale"]``
    already equals ``cfg``.  We replace that scale with our scheduled value
    selected by the current ``timestep`` (largest sigma first).
    """

    def __init__(self, schedule: List[float], sigmas: torch.Tensor):
        self.schedule = list(schedule)
        # Map sigma -> step index.  ComfyUI sigmas come in *decreasing* order
        # of magnitude, so step 0 == sigmas[0].
        self.sigma_to_step = {float(s.item()): i for i, s in enumerate(sigmas[:-1])}

    def __call__(self, args):
        sigma = args.get("sigma", None)
        cond = args["cond"]
        uncond = args["uncond"]
        scale = args.get("cond_scale", self.schedule[0])
        if sigma is not None:
            try:
                key = float(sigma.flatten()[0].item())
                idx = self.sigma_to_step.get(key, 0)
                if 0 <= idx < len(self.schedule):
                    scale = self.schedule[idx]
            except Exception:
                pass
        return uncond + (cond - uncond) * float(scale)


class MECBuilderSampler:
    """KSampler with adaptive CFG curves and an optional polish pass.

    Uses ComfyUI's standard ``comfy.sample.sample`` so the schedule honours
    the user's ``sampler_name`` / ``scheduler`` exactly the same way the
    built-in KSampler does — we only override ``cond_scale`` per step.
    """

    CATEGORY = "MEC/Paint"
    RETURN_TYPES = ("LATENT", "IMAGE")
    RETURN_NAMES = ("latent", "preview_image")
    FUNCTION = "execute"

    RESOLUTION_PRESETS = {
        "SDXL (1024x1024)": (1024, 1024),
        "SD1.5 (512x512)":  (512, 512),
        "Custom":           None,
    }

    @classmethod
    def INPUT_TYPES(cls):
        try:
            import comfy.samplers
            samplers = comfy.samplers.KSampler.SAMPLERS
            schedulers = comfy.samplers.KSampler.SCHEDULERS
        except Exception:
            samplers = ["euler", "dpmpp_2m"]
            schedulers = ["normal", "karras"]
        return {
            "required": {
                "model":    ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "steps":    ("INT",   {"default": 20, "min": 1, "max": 200}),
                "cfg":      ("FLOAT", {"default": 8.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "sampler_name": (samplers,),
                "scheduler":    (schedulers,),
                "denoise":  ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "cfg_mode":   (["Constant", "Linear", "Ease Down"], {"default": "Constant"}),
                "cfg_finish": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "cfg_pivot":  ("FLOAT", {"default": 5.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "self_correction": ("BOOLEAN", {"default": False}),
                "resolution_preset": (list(cls.RESOLUTION_PRESETS.keys()), {"default": "SD1.5 (512x512)"}),
                "custom_width":  ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8}),
                "custom_height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
            "optional": {
                "vae":          ("VAE",),
                "latent_image": ("LATENT",),
            },
        }

    def _resolve_resolution(self, preset, w, h):
        wh = self.RESOLUTION_PRESETS.get(preset)
        if wh is None:
            return int(w), int(h)
        return wh

    def execute(self, model, positive, negative, steps, cfg,
                sampler_name, scheduler, denoise,
                cfg_mode, cfg_finish, cfg_pivot,
                self_correction, resolution_preset,
                custom_width, custom_height, seed,
                vae=None, latent_image=None):
        import comfy.samplers
        import comfy.sample

        w, h = self._resolve_resolution(resolution_preset, custom_width, custom_height)

        if latent_image is None:
            samples = torch.zeros(1, 4, h // 8, w // 8)
            latent_image = {"samples": samples}
        latent = latent_image["samples"].to(torch.float32)

        device = model.load_device if hasattr(model, "load_device") else \
                 (latent.device if isinstance(latent, torch.Tensor) else torch.device("cpu"))

        # ---- adaptive CFG ----
        sampler = comfy.samplers.sampler_object(sampler_name)
        sigmas = comfy.samplers.calculate_sigmas(
            model.get_model_object("model_sampling"), scheduler, int(steps)
        )
        if 0.0 < float(denoise) < 1.0:
            cut = max(int(round(steps * (1.0 - float(denoise)))), 0)
            sigmas = sigmas[cut:]
        sched = _build_cfg_schedule(cfg_mode, float(cfg), float(cfg_finish),
                                    float(cfg_pivot), max(len(sigmas) - 1, 1))

        m = model.clone()
        try:
            m.set_model_sampler_cfg_function(_CFGScheduler(sched, sigmas))
        except Exception:
            # Older Comfy: fall back to a constant CFG below.
            pass

        noise = comfy.sample.prepare_noise(latent, int(seed), None)
        out_latent = comfy.sample.sample_custom(
            m, noise, float(cfg), sampler, sigmas,
            positive, negative, latent,
            disable_noise=False, seed=int(seed),
        )

        # ---- self-correction polish pass ----
        if self_correction:
            polish_steps = 2
            polish_sigmas = comfy.samplers.calculate_sigmas(
                m.get_model_object("model_sampling"), scheduler, polish_steps + 2
            )[-(polish_steps + 1):]
            noise2 = comfy.sample.prepare_noise(out_latent, int(seed) + 1, None)
            out_latent = comfy.sample.sample_custom(
                m, noise2, float(cfg_finish if cfg_mode != "Constant" else cfg),
                sampler, polish_sigmas,
                positive, negative, out_latent,
                disable_noise=False, seed=int(seed) + 1,
            )

        out_dict = {"samples": out_latent}

        # ---- preview decode ----
        if vae is not None:
            try:
                preview = vae.decode(out_latent)
                preview = _to_bhwc(preview)
            except Exception:
                preview = torch.zeros(1, h, w, 3)
        else:
            preview = torch.zeros(1, h, w, 3)
        return (out_dict, preview)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  ComfyUI registration                                                 ║
# ╚══════════════════════════════════════════════════════════════════════╝
NODE_CLASS_MAPPINGS = {
    "MECAdvancedPaintCanvas": MECAdvancedPaintCanvas,
    "MECContextInpainter":    MECContextInpainter,
    "MECToneRefiner":         MECToneRefiner,
    "MECBuilderSampler":      MECBuilderSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MECAdvancedPaintCanvas": "Advanced Paint Canvas (MEC)",
    "MECContextInpainter":    "Context Inpainter / Fixer (MEC)",
    "MECToneRefiner":         "Tone Refiner (MEC)",
    "MECBuilderSampler":      "Builder Sampler (MEC)",
}
