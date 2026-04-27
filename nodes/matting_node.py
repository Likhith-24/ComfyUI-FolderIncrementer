"""
MattingNode – Alpha matting from coarse segmentation masks.

Backends:
  - **vitmatte_small / vitmatte_base**: HuggingFace ViTMatte neural matting.
    Takes image + trimap → compositing-grade alpha.  Best for single images.
  - **matanyone2**: Video matting via InferenceCore step API.
    Warmup protocol on the first frame, then per-frame alpha generation.
    Best for video sequences.
  - **auto**: VitMatte for single images, MatAnyone2 for video (if available),
    falls back to VitMatte per-frame.

Features:
  - Trimap auto-generation via dilate XOR erode (unknown = dilated & ~eroded)
  - Alpha validation with range clamping and spatial dim checks
  - Model I/O delegated to model_manager.py
  - RGB output (image × alpha) + alpha_mask output
  - Graceful degradation when optional packages are missing

Return convention:
  RETURN_TYPES = ("IMAGE", "MASK")
  RETURN_NAMES = ("rgb", "alpha_mask")
  - rgb: [B, H, W, 3] float32, premultiplied alpha (image × alpha)
  - alpha_mask: [B, H, W] float32, compositing-grade alpha
"""

from __future__ import annotations

import gc
import logging

import numpy as np
import torch
import torch.nn.functional as F

from .model_manager import get_or_load_model, clear_cache
from .utils import (
    HAS_CV2,
    compute_edge_band_np,
    generate_trimap,
    make_mask_overlay_preview,
)

logger = logging.getLogger("MEC")


# ══════════════════════════════════════════════════════════════════════
#  Trimap Generation  (dilate XOR erode)
# ══════════════════════════════════════════════════════════════════════

def _generate_trimap(
    mask_np: np.ndarray,
    edge_radius: int,
) -> np.ndarray:
    """Generate trimap via dilate XOR erode.

    unknown = dilated_mask & ~eroded_mask
    fg = eroded_mask, bg = ~dilated_mask

    Args:
        mask_np: float32 (H, W) in [0, 1].
        edge_radius: kernel radius for morphology ops.

    Returns:
        uint8 (H, W) with 0 = bg, 128 = unknown, 255 = fg.
    """
    if not HAS_CV2:
        # Fallback: hard threshold
        out = np.zeros(mask_np.shape, dtype=np.uint8)
        out[mask_np > 0.5] = 255
        return out

    import cv2

    binary = (mask_np > 0.5).astype(np.uint8) * 255
    ksize = edge_radius * 2 + 1
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))

    eroded = cv2.erode(binary, kern, iterations=1)
    dilated = cv2.dilate(binary, kern, iterations=1)

    # Unknown band = dilated XOR eroded (the ring between them)
    unknown = (dilated > 127) & (eroded < 128)

    trimap = np.zeros(mask_np.shape, dtype=np.uint8)
    trimap[eroded > 127] = 255     # definite foreground
    trimap[unknown] = 128          # unknown band
    # Everything else stays 0 (definite background)

    return trimap


def _trimap_to_pil(trimap_u8: np.ndarray):
    """Convert uint8 trimap array to PIL Image (L mode)."""
    from PIL import Image as PILImage
    return PILImage.fromarray(trimap_u8, mode="L")


# ══════════════════════════════════════════════════════════════════════
#  Alpha Validation
# ══════════════════════════════════════════════════════════════════════

def _validate_alpha(
    alpha: torch.Tensor,
    expected_B: int,
    H: int,
    W: int,
) -> torch.Tensor:
    """Validate and normalize alpha output.

    Ensures:
      - Shape is [B, H, W] float32 in [0, 1]
      - Batch dimension matches expected_B
      - Spatial dimensions match H × W

    Args:
        alpha: Raw alpha tensor.
        expected_B: Expected batch size.
        H: Expected height.
        W: Expected width.

    Returns:
        Validated [B, H, W] float32 tensor.
    """
    if alpha is None or alpha.numel() == 0:
        return torch.zeros(expected_B, H, W, dtype=torch.float32)

    if alpha.dim() == 2:
        alpha = alpha.unsqueeze(0)
    elif alpha.dim() == 4:
        alpha = alpha.squeeze(1)

    # Resize if spatial dims don't match
    if alpha.shape[-2] != H or alpha.shape[-1] != W:
        alpha = F.interpolate(
            alpha.unsqueeze(1).float(), size=(H, W),
            mode="bilinear", align_corners=False,
        ).squeeze(1)

    # Pad or truncate batch
    if alpha.shape[0] < expected_B:
        pad = torch.zeros(
            expected_B - alpha.shape[0], H, W,
            dtype=alpha.dtype, device=alpha.device,
        )
        alpha = torch.cat([alpha, pad], dim=0)
    elif alpha.shape[0] > expected_B:
        alpha = alpha[:expected_B]

    return alpha.float().clamp(0.0, 1.0)


# ══════════════════════════════════════════════════════════════════════
#  Node
# ══════════════════════════════════════════════════════════════════════

class MattingNode:
    """Alpha matting from coarse masks using ViTMatte or MatAnyone2.

    Returns (rgb, alpha_mask):
      - rgb: [B, H, W, 3] premultiplied image (image × alpha)
      - alpha_mask: [B, H, W] compositing-grade alpha
    """

    BACKENDS = [
        "auto",
        "vitmatte_small",
        "vitmatte_base",
        "matanyone2",
        "rvm_mobilenetv3",
        "rvm_resnet50",
        "cutie",
        "sam_hq",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "RGB image(s).  Single frame or video batch.",
                }),
                "mask": ("MASK", {
                    "tooltip": "Coarse segmentation mask(s) to refine.",
                }),
                "backend": (cls.BACKENDS, {
                    "default": "auto",
                    "tooltip": (
                        "Matting backend.\n"
                        "auto: VitMatte for B=1, MatAnyone2 for B>1.\n"
                        "vitmatte_small/base: HuggingFace ViTMatte neural matting.\n"
                        "matanyone2: Video matting with warmup protocol.\n"
                        "rvm_mobilenetv3/resnet50: RobustVideoMatting (trimap-free, human-focused).\n"
                        "cutie: CutIE video object segmentation (mask propagation).\n"
                        "sam_hq: HQ-SAM refinement (requires SAM model + mask→point prompts)."
                    ),
                }),
                "edge_radius": ("INT", {
                    "default": 15, "min": 1, "max": 200, "step": 1,
                    "tooltip": "Trimap unknown-band width in pixels.",
                }),
                "erode_dilate": ("INT", {
                    "default": 0, "min": -50, "max": 50, "step": 1,
                    "tooltip": (
                        "Pre-process mask: positive = erode (shrink), "
                        "negative = dilate (expand).  0 = no change."
                    ),
                }),
            },
            "optional": {
                "trimap": ("MASK", {
                    "tooltip": "Custom trimap override (0=bg, 0.5=unknown, 1=fg).",
                }),
                "n_warmup": ("INT", {
                    "default": 5, "min": 1, "max": 30,
                    "tooltip": "MatAnyone2 / RVM warmup frames.",
                }),
                "sam_model": ("SAM_MODEL", {
                    "tooltip": "Required for sam_hq backend. SAM model from SAM Model Loader.",
                }),
                "auto_download": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "If True, allow ViTMatte / MatAnyone weights to be downloaded "
                        "from HuggingFace on first use. If False and the model is not "
                        "already cached, fall back immediately to guided_filter / "
                        "gaussian_blur refinement instead of attempting a download. "
                        "Useful for offline render farms."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("rgb", "alpha_mask")
    FUNCTION = "matte"
    CATEGORY = "MaskEditControl/Matting"
    DESCRIPTION = (
        "Unified alpha matting from coarse segmentation masks.\n"
        "7 backends: ViTMatte (small/base), MatAnyone2, RVM (mobilenet/resnet), CutIE, HQ-SAM.\n"
        "Auto mode selects the best backend based on input (image vs video).\n\n"
        "Outputs:\n"
        "  rgb: premultiplied image (image × alpha) [B,H,W,3]\n"
        "  alpha_mask: compositing-grade alpha [B,H,W]"
    )

    # ── Main ──────────────────────────────────────────────────────────

    def matte(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        backend: str,
        edge_radius: int,
        erode_dilate: int,
        trimap: torch.Tensor | None = None,
        n_warmup: int = 5,
        sam_model=None,
        auto_download: bool = True,
    ):
        B_img = image.shape[0]
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        B_mask = mask.shape[0]
        H, W = image.shape[1], image.shape[2]

        # Expand / match batch dimensions
        if B_mask == 1 and B_img > 1:
            mask = mask.expand(B_img, -1, -1)
        B = max(B_img, B_mask)

        # Resize mask to image dims if needed
        if mask.shape[1] != H or mask.shape[2] != W:
            mask = F.interpolate(
                mask.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False,
            ).squeeze(1)

        # Erode / dilate preprocessing
        if erode_dilate != 0 and HAS_CV2:
            import cv2
            ksize = abs(erode_dilate) * 2 + 1
            kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
            processed = []
            for i in range(B):
                m = (mask[i].cpu().numpy() * 255).astype(np.uint8)
                if erode_dilate > 0:
                    m = cv2.erode(m, kern, iterations=1)
                else:
                    m = cv2.dilate(m, kern, iterations=1)
                processed.append(torch.from_numpy(m.astype(np.float32) / 255.0))
            mask = torch.stack(processed)

        # Decide backend
        actual_backend = backend
        if backend == "auto":
            if B > 1:
                actual_backend = "matanyone2"
            else:
                actual_backend = "vitmatte_small"

        # Dispatch
        if actual_backend.startswith("vitmatte"):
            alpha_mask = self._run_vitmatte_with_fallback(
                image, mask, actual_backend, edge_radius, trimap, B, H, W,
                auto_download=auto_download,
            )
        elif actual_backend == "matanyone2":
            try:
                alpha_mask = self._run_matanyone(image, mask, n_warmup, B, H, W)
            except (ImportError, RuntimeError) as exc:
                logger.warning(
                    "[MEC] MatAnyone2 unavailable (%s), falling back to VitMatte chain.", exc,
                )
                alpha_mask = self._run_vitmatte_with_fallback(
                    image, mask, "vitmatte_small", edge_radius, trimap, B, H, W,
                    auto_download=auto_download,
                )
        elif actual_backend.startswith("rvm"):
            alpha_mask = self._run_rvm(image, actual_backend, n_warmup, B, H, W)
        elif actual_backend == "cutie":
            alpha_mask = self._run_cutie(image, mask, B, H, W)
        elif actual_backend == "sam_hq":
            if sam_model is None:
                raise ValueError("sam_hq backend requires a SAM model. Connect SAM Model Loader.")
            alpha_mask = self._run_sam_hq_refine(
                image, mask, sam_model, edge_radius, B, H, W,
            )
        else:
            raise ValueError(f"Unknown matting backend: {actual_backend}")

        # Free intermediate GPU memory for low-VRAM users
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Validate alpha
        alpha_mask = _validate_alpha(alpha_mask, B, H, W)

        # Build rgb output: premultiplied (image × alpha)
        # image: [B, H, W, C],  alpha_mask: [B, H, W]
        rgb = image[:B] * alpha_mask.unsqueeze(-1)

        return (rgb, alpha_mask)

    # ══════════════════════════════════════════════════════════════════
    #  ViTMatte Backend
    # ══════════════════════════════════════════════════════════════════

    # ══════════════════════════════════════════════════════════════════
    #  ViTMatte Backend (with explicit fallback chain)
    # ══════════════════════════════════════════════════════════════════
    #
    # Fallback order (any link can fail and we move on):
    #   1. ViTMatte-base   (best quality, ~360 MB)
    #   2. ViTMatte-small  (good quality, ~100 MB)
    #   3. guided_filter   (CV2-based edge-aware refinement, no model)
    #   4. gaussian_blur   (pure-torch soft-edge fallback, always available)

    def _run_vitmatte_with_fallback(
        self, image, mask, variant, edge_radius, trimap_in,
        B, H, W, auto_download: bool = True,
    ):
        chain: list[str] = []
        # Try the requested variant first, then escalate down the chain.
        if variant == "vitmatte_base":
            chain = ["vitmatte_base", "vitmatte_small"]
        else:
            chain = ["vitmatte_small", "vitmatte_base"]

        # If auto_download is disabled, only try variants whose weights
        # appear to be already cached. We cannot reliably introspect
        # model_manager, so we simply attempt and catch network errors.
        last_exc: Exception | None = None
        for v in chain:
            try:
                if not auto_download:
                    # Best-effort: set env so HF will refuse to download.
                    import os as _os
                    prev = _os.environ.get("HF_HUB_OFFLINE")
                    _os.environ["HF_HUB_OFFLINE"] = "1"
                    try:
                        return self._run_vitmatte(
                            image, mask, v, edge_radius, trimap_in, B, H, W,
                        )
                    finally:
                        if prev is None:
                            _os.environ.pop("HF_HUB_OFFLINE", None)
                        else:
                            _os.environ["HF_HUB_OFFLINE"] = prev
                else:
                    return self._run_vitmatte(
                        image, mask, v, edge_radius, trimap_in, B, H, W,
                    )
            except Exception as exc:  # noqa: BLE001 - intentionally broad: network/import/runtime
                logger.warning("[MEC] %s failed (%s); trying next backend.", v, exc)
                last_exc = exc

        # Both ViTMatte variants unavailable — try guided filter.
        try:
            return self._guided_filter_matte(image, mask, edge_radius, B, H, W)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[MEC] guided_filter unavailable (%s); using gaussian_blur fallback.", exc,
            )

        # Final fallback: pure-torch gaussian blur of the input mask.
        logger.info(
            "[MEC] Using gaussian_blur matte fallback. Last ViTMatte error: %s", last_exc,
        )
        return self._gaussian_blur_matte(mask, edge_radius, B, H, W)

    def _guided_filter_matte(self, image, mask, edge_radius, B, H, W):
        """Edge-aware mask refinement using OpenCV's guided filter.

        Requires opencv-contrib (``cv2.ximgproc``). Falls back to
        ``cv2.bilateralFilter`` if ximgproc is unavailable.
        """
        if not HAS_CV2:
            raise RuntimeError("guided_filter requires OpenCV.")
        import cv2
        radius = max(2, edge_radius // 2)
        out: list[torch.Tensor] = []
        try:
            ximgproc = cv2.ximgproc  # type: ignore[attr-defined]
            has_gf = True
        except AttributeError:
            has_gf = False
        for i in range(B):
            img_np = image[min(i, image.shape[0] - 1)].cpu().numpy()
            img_u8 = (img_np * 255).clip(0, 255).astype(np.uint8)
            mask_np = mask[i].cpu().numpy().astype(np.float32)
            if has_gf:
                refined = ximgproc.guidedFilter(  # type: ignore[attr-defined]
                    guide=img_u8, src=mask_np, radius=radius, eps=1e-3,
                )
            else:
                # bilateralFilter on the mask using image as joint signal
                refined = cv2.bilateralFilter(
                    mask_np, d=radius * 2 + 1, sigmaColor=0.1, sigmaSpace=radius,
                )
            out.append(torch.from_numpy(np.clip(refined, 0.0, 1.0)).float())
        return torch.stack(out)

    def _gaussian_blur_matte(self, mask, edge_radius, B, H, W):
        """Pure-torch soft-edge fallback. Always available.

        Applies a single separable Gaussian blur sized to ``edge_radius``
        and clamps to [0, 1]. Quality is far below ViTMatte but the
        output is well-defined on every machine.
        """
        sigma = max(1.0, edge_radius / 3.0)
        ksize = max(3, edge_radius * 2 + 1) | 1  # ensure odd
        # Build 1D gaussian kernel
        ax = torch.arange(ksize, dtype=torch.float32) - (ksize - 1) / 2.0
        kern_1d = torch.exp(-(ax ** 2) / (2 * sigma * sigma))
        kern_1d = kern_1d / kern_1d.sum()
        # Separable conv via two passes
        m = mask[:B].unsqueeze(1)  # [B, 1, H, W]
        kern_h = kern_1d.view(1, 1, 1, ksize)
        kern_v = kern_1d.view(1, 1, ksize, 1)
        pad = ksize // 2
        m = F.conv2d(F.pad(m, (pad, pad, 0, 0), mode="replicate"), kern_h)
        m = F.conv2d(F.pad(m, (0, 0, pad, pad), mode="replicate"), kern_v)
        return m.squeeze(1).clamp(0.0, 1.0)

    def _run_vitmatte(self, image, mask, variant, edge_radius, trimap_in, B, H, W):
        from PIL import Image as PILImage

        device = "cuda" if torch.cuda.is_available() else "cpu"
        loaded = get_or_load_model(variant, precision="fp32", device=device)
        if isinstance(loaded, dict):
            model, processor = loaded["model"], loaded["processor"]
        else:
            model, processor = loaded, None

        if processor is None:
            try:
                from transformers import VitMatteImageProcessor
                processor = VitMatteImageProcessor()
            except ImportError:
                raise RuntimeError("transformers is required for ViTMatte.")

        device = next(model.parameters()).device

        alphas: list[torch.Tensor] = []
        for i in range(B):
            img_np = (image[min(i, image.shape[0] - 1)].cpu().numpy() * 255).astype(np.uint8)
            mask_np = mask[i].cpu().numpy().astype(np.float32)

            # Build trimap using dilate XOR erode
            if trimap_in is not None:
                t = trimap_in[min(i, trimap_in.shape[0] - 1)]
                if t.dim() > 2:
                    t = t[0]
                tri_np = t.cpu().numpy()
                tri_u8 = np.zeros(tri_np.shape, dtype=np.uint8)
                tri_u8[tri_np > 0.9] = 255
                tri_u8[(tri_np > 0.1) & (tri_np < 0.9)] = 128
                pil_tri = _trimap_to_pil(tri_u8)
            else:
                trimap_u8 = _generate_trimap(mask_np, edge_radius)
                pil_tri = _trimap_to_pil(trimap_u8)

            pil_img = PILImage.fromarray(img_np[:, :, :3])
            inputs = processor(images=pil_img, trimaps=pil_tri, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                out = model(**inputs)

            # out.alphas: (1, 1, H', W')
            a = out.alphas[0, 0].cpu()
            if a.shape[0] != H or a.shape[1] != W:
                a = F.interpolate(
                    a.unsqueeze(0).unsqueeze(0), (H, W),
                    mode="bilinear", align_corners=False,
                )[0, 0]

            # Blend: keep coarse mask in non-edge areas, ViTMatte at edges
            edge_band = compute_edge_band_np(mask_np, edge_radius)
            eb_t = torch.from_numpy(edge_band)
            blended = mask[i].cpu() * (1 - eb_t) + a * eb_t
            alphas.append(blended)

        return torch.stack(alphas)

    # ══════════════════════════════════════════════════════════════════
    #  MatAnyone2 Backend
    # ══════════════════════════════════════════════════════════════════

    def _run_matanyone(self, image, mask, n_warmup, B, H, W):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        loaded = get_or_load_model("matanyone2", precision="fp32", device=device)
        if isinstance(loaded, dict):
            core = loaded["core"]
        else:
            core = loaded

        # MatAnyone2 expects:  image (C, H, W) float [0, 1],  mask (1, H, W) float {0, 1}
        alphas: list[torch.Tensor] = []

        # Warmup phase: repeat first frame n_warmup times with first_frame_pred=True
        first_img = image[0].permute(2, 0, 1).to(device)  # (C, H, W)
        first_mask = mask[0].unsqueeze(0).to(device)       # (1, H, W)
        first_mask_bin = (first_mask > 0.5).float()

        # Frame 0: initial step with mask
        try:
            core.step(first_img, first_mask_bin)
        except Exception as exc:
            logger.debug("[MEC] MatAnyone2 initial step error: %s", exc)

        # Warmup: repeat first frame with first_frame_pred=True
        for _ in range(n_warmup):
            try:
                core.step(first_img, first_frame_pred=True)
            except TypeError:
                # Fallback for older API without first_frame_pred
                try:
                    core.step(first_img, first_mask_bin)
                except Exception as exc:
                    logger.debug("[MEC] MatAnyone2 warmup step error: %s", exc)
                    break
            except Exception as exc:
                logger.debug("[MEC] MatAnyone2 warmup step error: %s", exc)
                break

        # Process all frames
        for i in range(B):
            img_t = image[min(i, image.shape[0] - 1)].permute(2, 0, 1).to(device)
            m_t = mask[i].unsqueeze(0).to(device)
            m_bin = (m_t > 0.5).float()

            try:
                if i == 0:
                    alpha = core.step(img_t, m_bin)
                else:
                    alpha = core.step(img_t)

                if isinstance(alpha, torch.Tensor):
                    alphas.append(alpha.cpu().squeeze())
                else:
                    alphas.append(torch.from_numpy(np.array(alpha, dtype=np.float32)))
            except Exception as exc:
                logger.warning("[MEC] MatAnyone2 frame %d error: %s", i, exc)
                alphas.append(mask[i].cpu())

        return torch.stack(alphas)

    # ══════════════════════════════════════════════════════════════════
    #  RobustVideoMatting Backend
    # ══════════════════════════════════════════════════════════════════

    def _run_rvm(self, image, variant, n_warmup, B, H, W):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        loaded = get_or_load_model(variant, precision="fp32", device=device)
        model = loaded["model"]

        # RVM processes video as sequential frames with recurrent hidden state
        # Input: (B, C, H, W) per frame.  Output: alpha (B, 1, H, W), fgr (B, 3, H, W)
        rec = [None] * 4  # recurrent states: r1, r2, r3, r4
        downsample_ratio = 0.25  # default for speed

        alphas = []
        with torch.no_grad():
            for i in range(B):
                frame = image[min(i, image.shape[0] - 1)].permute(2, 0, 1).unsqueeze(0).to(device)
                fgr, pha, *rec = model(frame, *rec, downsample_ratio)
                alphas.append(pha[0, 0].cpu())

        return torch.stack(alphas)

    # ══════════════════════════════════════════════════════════════════
    #  CutIE Backend (Video Object Segmentation)
    # ══════════════════════════════════════════════════════════════════

    def _run_cutie(self, image, mask, B, H, W):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        loaded = get_or_load_model("cutie_base_mega", precision="fp32", device=device)
        processor = loaded["processor"]

        # CutIE: first frame gets mask, subsequent frames are propagated
        first_frame = image[0].permute(2, 0, 1).to(device)  # (C, H, W)
        first_mask = (mask[0] > 0.5).long().to(device)  # (H, W) integer labels

        # Encode first frame with mask (object label = 1)
        with torch.no_grad():
            processor.step(first_frame, first_mask, idx_mask=False)

        alphas = [mask[0].cpu().float()]  # first frame uses the input mask

        with torch.no_grad():
            for i in range(1, B):
                frame = image[min(i, image.shape[0] - 1)].permute(2, 0, 1).to(device)
                output_prob = processor.step(frame)
                # output_prob: (num_objects, H, W) probabilities
                if output_prob.dim() == 3:
                    alpha = output_prob[0].cpu().float()  # first object
                else:
                    alpha = output_prob.cpu().float()
                alphas.append(alpha)

        return torch.stack(alphas)

    # ══════════════════════════════════════════════════════════════════
    #  HQ-SAM Refinement Backend
    # ══════════════════════════════════════════════════════════════════

    def _run_sam_hq_refine(self, image, mask, sam_model, edge_radius, B, H, W):
        """Use HQ-SAM to refine a coarse mask by converting mask to point prompts."""
        from .utils import mask_to_bbox

        model_info = sam_model
        model = model_info["model"]

        try:
            from segment_anything_hq import SamPredictor as HQSamPredictor
        except ImportError:
            from segment_anything import SamPredictor as HQSamPredictor

        predictor = HQSamPredictor(model)

        alphas = []
        for i in range(B):
            img_np = (image[min(i, image.shape[0] - 1)].cpu().numpy() * 255).astype(np.uint8)
            mask_np = mask[i].cpu().numpy()

            predictor.set_image(img_np)

            # Derive bbox from mask
            bbox = mask_to_bbox(mask[i])
            box_np = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]],
                              dtype=np.float32)

            # Sample a positive point from mask center of mass
            ys, xs = np.where(mask_np > 0.5)
            if len(ys) > 0:
                cx, cy = float(xs.mean()), float(ys.mean())
                point_coords = np.array([[cx, cy]], dtype=np.float32)
                point_labels = np.array([1], dtype=np.int32)
            else:
                point_coords = None
                point_labels = None

            kwargs = {"multimask_output": False, "box": box_np}
            if point_coords is not None:
                kwargs["point_coords"] = point_coords
                kwargs["point_labels"] = point_labels

            try:
                masks_np, scores, _ = predictor.predict(**kwargs)
                alpha = torch.from_numpy(masks_np[0].astype(np.float32))
            except Exception:
                alpha = mask[i].cpu()

            alphas.append(alpha)

        return torch.stack(alphas)
