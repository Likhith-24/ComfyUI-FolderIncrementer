"""
Shared utilities for the MaskEditControl node pack.

Centralizes:
  - ViTMatte model loading & caching
  - Trimap generation
  - Edge band computation (numpy + torch)
  - Guided filter implementations
  - Mask post-processing (hole fill, small region removal)
  - Laplacian pyramid helpers
  - Image/video first-frame extraction
"""

import torch
import torch.nn.functional as F
import numpy as np
import gc
import os
import logging

logger = logging.getLogger("MEC")

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# ══════════════════════════════════════════════════════════════════════
#  ViTMatte Model Management (singleton cache)
# ══════════════════════════════════════════════════════════════════════

_vitmatte_model = None
_vitmatte_processor = None


def get_vitmatte_model():
    """Load ViTMatte model once and return cached instance.

    Resolution order:
      1. ComfyUI/models/vitmatte/ (local)
      2. HuggingFace auto-download (hustvl/vitmatte-small-distinctions-646)
    """
    global _vitmatte_model, _vitmatte_processor

    if _vitmatte_model is not None:
        return _vitmatte_model, _vitmatte_processor

    from transformers import VitMatteForImageMatting, VitMatteImageProcessor

    model = None

    # Try local model path
    try:
        import folder_paths
        local_dirs = []
        if "vitmatte" in folder_paths.folder_names_and_paths:
            local_dirs = folder_paths.get_folder_paths("vitmatte")
        else:
            for base in (folder_paths.base_path, folder_paths.models_dir):
                candidate = os.path.join(base, "vitmatte")
                if os.path.isdir(candidate):
                    local_dirs.append(candidate)

        for local_dir in local_dirs:
            if os.path.isdir(local_dir) and any(
                f.endswith(('.safetensors', '.bin', '.pt'))
                for f in os.listdir(local_dir)
            ):
                model = VitMatteForImageMatting.from_pretrained(local_dir)
                logger.info(f"[MEC] ViTMatte loaded from local: {local_dir}")
                break
    except Exception as e:
        logger.debug(f"[MEC] Local ViTMatte not found: {e}")

    if model is None:
        logger.info("[MEC] Downloading ViTMatte from HuggingFace...")
        model = VitMatteForImageMatting.from_pretrained(
            "hustvl/vitmatte-small-distinctions-646"
        )
        logger.info("[MEC] ViTMatte downloaded and cached.")

    processor = VitMatteImageProcessor()
    model.eval()

    _vitmatte_model = model
    _vitmatte_processor = processor
    return model, processor


def release_vitmatte_model():
    """Release ViTMatte model to free memory."""
    global _vitmatte_model, _vitmatte_processor
    _vitmatte_model = None
    _vitmatte_processor = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════════════
#  Trimap Generation
# ══════════════════════════════════════════════════════════════════════

def generate_trimap(mask_np, edge_radius, inner_scale=1.0, outer_scale=1.5):
    """Generate a high-quality trimap from a binary mask.

    Args:
        mask_np: float32 numpy array (H, W) in [0, 1]
        edge_radius: pixel radius for the unknown band
        inner_scale: erosion kernel scale (≤1 = tighter fg)
        outer_scale: dilation kernel scale (≥1 = wider unknown band)

    Returns:
        trimap: float32 (H, W) with 1.0=fg, 0.0=bg, 0.5=unknown
    """
    if not HAS_CV2:
        # Fallback: threshold the mask directly
        return (mask_np > 0.5).astype(np.float32)

    binary = (mask_np > 0.5).astype(np.uint8) * 255

    inner_r = max(1, int(edge_radius * inner_scale))
    outer_r = max(1, int(edge_radius * outer_scale))

    kern_inner = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (inner_r * 2 + 1, inner_r * 2 + 1)
    )
    kern_outer = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (outer_r * 2 + 1, outer_r * 2 + 1)
    )

    fg = cv2.erode(binary, kern_inner)
    bg_region = 255 - cv2.dilate(binary, kern_outer)

    trimap = np.full(mask_np.shape, 0.5, dtype=np.float32)
    trimap[fg > 127] = 1.0
    trimap[bg_region > 127] = 0.0

    return trimap


# ══════════════════════════════════════════════════════════════════════
#  Edge Band Computation
# ══════════════════════════════════════════════════════════════════════

def compute_edge_band_np(mask_np, radius):
    """Compute a smooth float edge band around the mask boundary (numpy/cv2).

    Returns: float32 (H, W) in [0, 1], 1 at edges, 0 away.
    """
    if not HAS_CV2:
        return np.zeros_like(mask_np, dtype=np.float32)

    binary = (mask_np > 0.5).astype(np.uint8)
    kern = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)
    )
    dilated = cv2.dilate(binary, kern)
    eroded = cv2.erode(binary, kern)
    band = (dilated - eroded).astype(np.float32)

    blur_k = max(3, radius) | 1
    band = cv2.GaussianBlur(band, (blur_k, blur_k), radius * 0.3)
    return np.clip(band, 0, 1)


def compute_edge_band_torch(mask, radius):
    """Compute a smooth edge band around mask boundary (PyTorch).

    Args:
        mask: float tensor (H, W)
        radius: int

    Returns: float tensor (H, W) in [0, 1]
    """
    k = radius * 2 + 1
    pad = radius
    kernel = torch.ones(1, 1, k, k, device=mask.device) / (k * k)
    m4 = mask.unsqueeze(0).unsqueeze(0)
    avg = F.conv2d(F.pad(m4, (pad, pad, pad, pad), "replicate"), kernel)[0, 0]
    band = 1.0 - torch.abs(avg * 2 - 1)
    return band.clamp(0, 1)


# ══════════════════════════════════════════════════════════════════════
#  Guided Filter
# ══════════════════════════════════════════════════════════════════════

def guided_filter(guide, src, radius, eps):
    """Box-filter-based guided filter (works without cv2.ximgproc).

    Args:
        guide: float32 (H, W) guide image channel
        src: float32 (H, W) source signal
        radius: filter radius
        eps: regularization
    Returns:
        float32 (H, W) filtered result
    """
    if not HAS_CV2:
        return src

    ksize = radius * 2 + 1
    mean_I = cv2.blur(guide, (ksize, ksize))
    mean_p = cv2.blur(src, (ksize, ksize))
    mean_Ip = cv2.blur(guide * src, (ksize, ksize))
    mean_II = cv2.blur(guide * guide, (ksize, ksize))

    cov_Ip = mean_Ip - mean_I * mean_p
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.blur(a, (ksize, ksize))
    mean_b = cv2.blur(b, (ksize, ksize))

    return mean_a * guide + mean_b


# ══════════════════════════════════════════════════════════════════════
#  Multi-Scale Guided Filter Refinement
# ══════════════════════════════════════════════════════════════════════

def multi_scale_guided_refine(img_np, mask_np, edge_radius, detail_level):
    """Run guided filter at 3 scales and fuse for fine + coarse detail.

    Args:
        img_np: uint8 (H, W, 3) RGB image
        mask_np: float32 (H, W) mask
        edge_radius: base filter radius
        detail_level: 0-1, how much fine detail to preserve

    Returns:
        float32 (H, W) refined mask, or None on failure
    """
    if not HAS_CV2:
        return None

    try:
        if img_np.ndim == 2:
            img_np = img_np[:, :, np.newaxis]
        guide = img_np[:, :, :3] if img_np.shape[2] >= 3 else img_np
        guide_f = guide.astype(np.float32) / 255.0

        scales = [
            {"radius": max(1, edge_radius // 3),
             "eps": (1 - detail_level) ** 2 * 0.01 + 1e-6},
            {"radius": max(1, edge_radius),
             "eps": (1 - detail_level) ** 2 * 0.05 + 1e-4},
            {"radius": max(1, edge_radius * 3),
             "eps": (1 - detail_level) ** 2 * 0.2 + 1e-3},
        ]

        results = []
        for s in scales:
            acc = np.zeros_like(mask_np)
            n_ch = min(3, guide_f.shape[2])
            for ch in range(n_ch):
                filtered = guided_filter(
                    guide_f[:, :, ch], mask_np, s["radius"], s["eps"]
                )
                acc += filtered
            acc /= n_ch
            results.append(np.clip(acc, 0, 1))

        edge_band = compute_edge_band_np(mask_np, edge_radius)
        fine_band = np.clip(edge_band * 2, 0, 1)
        coarse_inv = 1 - fine_band

        fused = (
            results[0] * fine_band * detail_level
            + results[1] * fine_band * (1 - detail_level)
            + results[2] * coarse_inv * 0.3
            + mask_np * coarse_inv * 0.7
        )

        return np.clip(fused, 0, 1).astype(np.float32)

    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════
#  Color-Aware (LAB) Refinement
# ══════════════════════════════════════════════════════════════════════

def color_aware_refine(img_np, mask_np, edge_radius, detail_level):
    """LAB color-space guided filter for lighting-robust edge refinement.

    Returns: float32 (H, W) or None on failure.
    """
    if not HAS_CV2:
        return None

    try:
        if img_np.ndim == 2:
            img_np = np.stack([img_np] * 3, axis=-1)
        rgb = img_np[:, :, :3] if img_np.shape[2] >= 3 else np.stack([img_np[:,:,0]] * 3, axis=-1)
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0

        eps = (1 - detail_level) ** 2 * 0.02 + 1e-6
        acc = np.zeros_like(mask_np)
        weights = [0.5, 0.25, 0.25]

        for ch, w in zip(range(3), weights):
            filtered = guided_filter(
                lab[:, :, ch], mask_np, max(1, edge_radius), eps
            )
            acc += filtered * w

        edge_band = compute_edge_band_np(mask_np, edge_radius)
        result = mask_np * (1 - edge_band) + np.clip(acc, 0, 1) * edge_band
        return result.astype(np.float32)

    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════
#  Mask Post-Processing
# ══════════════════════════════════════════════════════════════════════

def fill_holes(mask_np):
    """Fill interior holes in a mask using contour hierarchy."""
    if not HAS_CV2:
        return mask_np

    binary = (mask_np > 0.5).astype(np.uint8) * 255
    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    if hierarchy is not None:
        for i, h in enumerate(hierarchy[0]):
            if h[3] >= 0:  # has parent → is a hole
                cv2.drawContours(binary, contours, i, 255, -1)
    return (binary > 127).astype(np.float32)


def remove_small_regions(mask_np, min_area):
    """Remove connected components smaller than min_area pixels."""
    if not HAS_CV2 or min_area <= 0:
        return mask_np

    binary = (mask_np > 0.5).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    result = np.zeros_like(binary)
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            result[labels == i] = 1
    return result.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════
#  Laplacian Pyramid
# ══════════════════════════════════════════════════════════════════════

def build_laplacian_pyramid(img, levels):
    """Build a Laplacian pyramid from a 2D array."""
    if not HAS_CV2:
        return [img]

    pyr = []
    cur = img.copy()
    for _ in range(levels):
        down = cv2.pyrDown(cur)
        up = cv2.pyrUp(down, dstsize=(cur.shape[1], cur.shape[0]))
        pyr.append(cur - up)
        cur = down
    pyr.append(cur)
    return pyr


def reconstruct_laplacian_pyramid(pyr):
    """Reconstruct image from Laplacian pyramid."""
    if not HAS_CV2:
        return pyr[0] if pyr else np.zeros((1, 1), dtype=np.float32)

    cur = pyr[-1]
    for i in range(len(pyr) - 2, -1, -1):
        up = cv2.pyrUp(cur, dstsize=(pyr[i].shape[1], pyr[i].shape[0]))
        cur = up + pyr[i]
    return cur


# ══════════════════════════════════════════════════════════════════════
#  Gaussian Edge Refinement (always-available fallback)
# ══════════════════════════════════════════════════════════════════════

def gaussian_edge_refine(mask_t, edge_radius):
    """Simple Gaussian blur applied only at mask edges (PyTorch).

    Args:
        mask_t: float tensor (H, W)
        edge_radius: blur radius

    Returns: float tensor (H, W)
    """
    sigma = max(0.5, edge_radius * 0.4)
    k = int(sigma * 6) | 1
    if k < 3:
        k = 3

    x = torch.arange(k, dtype=torch.float32, device=mask_t.device) - k // 2
    kernel = torch.exp(-x ** 2 / (2 * sigma ** 2))
    kernel /= kernel.sum()

    m4 = mask_t.unsqueeze(0).unsqueeze(0)
    pad = k // 2
    kh = kernel.view(1, 1, 1, -1)
    kv = kernel.view(1, 1, -1, 1)
    m4 = F.conv2d(F.pad(m4, (pad, pad, 0, 0), "replicate"), kh)
    m4 = F.conv2d(F.pad(m4, (0, 0, pad, pad), "replicate"), kv)
    blurred = m4[0, 0]

    edge_band = compute_edge_band_torch(mask_t, edge_radius)
    return mask_t * (1 - edge_band) + blurred * edge_band


# ══════════════════════════════════════════════════════════════════════
#  ViTMatte Refinement
# ══════════════════════════════════════════════════════════════════════

def refine_with_vitmatte(img_t, mask_t, edge_radius, trimap_input=None):
    """Run ViTMatte on an image+mask and return refined alpha.

    Args:
        img_t: float tensor (H, W, C) in [0, 1]
        mask_t: float tensor (H, W) in [0, 1]
        edge_radius: trimap unknown-band radius
        trimap_input: optional float tensor (H, W) trimap override

    Returns: float tensor (H, W) refined alpha, or None on failure
    """
    try:
        from transformers import VitMatteForImageMatting, VitMatteImageProcessor
        from PIL import Image as PILImage
    except ImportError:
        return None

    try:
        img_np = (img_t.cpu().numpy() * 255).astype(np.uint8)
        mask_np = mask_t.cpu().numpy()

        if trimap_input is not None:
            tri = trimap_input.cpu().numpy() if isinstance(trimap_input, torch.Tensor) else trimap_input
            if tri.ndim == 3:
                tri = tri[0]
        else:
            tri = generate_trimap(mask_np, edge_radius)

        model, processor = get_vitmatte_model()

        pil_img = PILImage.fromarray(img_np[:, :, :3])
        pil_tri = PILImage.fromarray((tri * 255).astype(np.uint8), mode="L")

        inputs = processor(images=pil_img, trimaps=pil_tri, return_tensors="pt")

        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            alpha = model(**inputs).alphas[0, 0]

        # Resize alpha to match mask if ViTMatte output resolution differs
        alpha_out = alpha.cpu().to(mask_t.device)
        mH, mW = mask_t.shape
        if alpha_out.shape[0] != mH or alpha_out.shape[1] != mW:
            alpha_out = F.interpolate(
                alpha_out.unsqueeze(0).unsqueeze(0), size=(mH, mW),
                mode="bilinear", align_corners=False,
            ).squeeze(0).squeeze(0)

        # Blend: keep coarse mask in non-edge regions, ViTMatte at edges
        edge_band = compute_edge_band_np(mask_np, edge_radius)
        edge_band_t = torch.from_numpy(edge_band).to(mask_t.device)
        result = mask_t * (1 - edge_band_t) + alpha_out * edge_band_t

        return result

    except Exception as e:
        logger.debug(f"[MEC] ViTMatte refinement failed: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════
#  Edge Contrast Boost
# ══════════════════════════════════════════════════════════════════════

def boost_edge_contrast(refined_t, coarse_t, contrast, edge_radius=10):
    """Apply sigmoid-based contrast curve at mask edges.

    Args:
        refined_t: float tensor (H, W)
        coarse_t: float tensor (H, W)
        contrast: float, >1 sharpens, <1 softens
        edge_radius: band detection radius

    Returns: float tensor (H, W)
    """
    edge_band = compute_edge_band_torch(coarse_t, edge_radius)
    shifted = (refined_t - 0.5) * contrast
    boosted = torch.sigmoid(shifted * 5.0)
    return refined_t * (1 - edge_band) + boosted * edge_band


# ══════════════════════════════════════════════════════════════════════
#  SAM Predictor Factory
# ══════════════════════════════════════════════════════════════════════

def get_sam_predictor(model, model_type, img_np):
    """Create the correct SAM predictor for the given model type and set image.

    Args:
        model: loaded SAM model object (SAM2Base or SamModel)
        model_type: str ("sam2", "sam2.1", "sam3", "sam_vit_h", etc.)
        img_np: uint8 (H, W, 3) RGB image

    Returns: predictor object or None
    """
    predictor = None

    if model_type in ("sam2", "sam2.1", "sam3"):
        try:
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            predictor = SAM2ImagePredictor(model)
        except ImportError:
            logger.error(
                "[MEC] sam2 package not found. Install with:\n"
                "  pip install git+https://github.com/facebookresearch/sam2.git"
            )
            return None
    else:
        try:
            from segment_anything import SamPredictor
            predictor = SamPredictor(model)
        except ImportError:
            logger.error("[MEC] segment_anything package not found.")
            return None

    if predictor is not None:
        try:
            # Detect dtype from model parameters (model.dtype may not exist)
            dtype = None
            device = "cpu"
            if hasattr(model, 'parameters'):
                try:
                    p = next(model.parameters())
                    dtype = p.dtype
                    device = p.device
                except StopIteration:
                    pass
            # Use autocast for fp16/bf16 models to avoid dtype mismatches
            if dtype in (torch.float16, torch.bfloat16) and str(device) != "cpu":
                with torch.autocast(str(device), dtype=dtype):
                    predictor.set_image(img_np)
            else:
                predictor.set_image(img_np)
        except Exception as e:
            logger.error(f"[MEC] predictor.set_image failed: {e}")
            return None

    return predictor


def sam_predict(predictor, model_info, **kwargs):
    """Run ``predictor.predict()`` with proper autocast for fp16/bf16 models.

    Wraps ``predict`` so callers don't need to manage dtype contexts.
    All keyword arguments are forwarded to ``predictor.predict()``.
    """
    dtype = model_info.get("dtype", torch.float32) if isinstance(model_info, dict) else torch.float32
    device = model_info.get("device", "cpu") if isinstance(model_info, dict) else "cpu"

    if dtype in (torch.float16, torch.bfloat16) and str(device) != "cpu":
        with torch.autocast(str(device), dtype=dtype):
            return predictor.predict(**kwargs)
    return predictor.predict(**kwargs)


# ══════════════════════════════════════════════════════════════════════
#  Prompt Augmentation (for iterative SAM refinement)
# ══════════════════════════════════════════════════════════════════════

def augment_prompts_from_mask(mask, orig_coords, orig_labels, orig_box,
                              H, W, auto_negative=True):
    """Generate augmented prompts from a previous mask iteration.

    Args:
        mask: float32 numpy array (H, W) current mask
        orig_coords: (N, 2) float32 array or None — original point coords
        orig_labels: (N,) int32 array or None — original point labels
        orig_box: (4,) float32 array or None — original bounding box [x1,y1,x2,y2]
        H, W: image dimensions
        auto_negative: whether to sample negative points from boundary exterior

    Returns:
        (all_coords, all_labels, box) tuple
    """
    coords_list = []
    labels_list = []

    if orig_coords is not None:
        coords_list.append(orig_coords)
        labels_list.append(orig_labels)

    if not HAS_CV2:
        c = np.concatenate(coords_list) if coords_list else None
        l = np.concatenate(labels_list) if labels_list else None
        return c, l, orig_box

    import cv2
    binary = (mask > 0.5).astype(np.uint8)

    # Eroded interior → strong positive points
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    interior = cv2.erode(binary, kern, iterations=1)
    interior_pts = np.argwhere(interior > 0)  # (y, x)
    if len(interior_pts) > 0:
        n = min(3, len(interior_pts))
        indices = np.linspace(0, len(interior_pts) - 1, n, dtype=int)
        sampled = interior_pts[indices]
        coords_list.append(sampled[:, ::-1].astype(np.float32))
        labels_list.append(np.ones(n, dtype=np.int32))

    # Dilated boundary exterior → negative points
    if auto_negative:
        dilated = cv2.dilate(binary, kern, iterations=1)
        exterior = dilated - binary
        exterior_pts = np.argwhere(exterior > 0)
        if len(exterior_pts) > 0:
            n = min(2, len(exterior_pts))
            indices = np.linspace(0, len(exterior_pts) - 1, n, dtype=int)
            sampled = exterior_pts[indices]
            coords_list.append(sampled[:, ::-1].astype(np.float32))
            labels_list.append(np.zeros(n, dtype=np.int32))

    all_coords = np.concatenate(coords_list).astype(np.float32) if coords_list else None
    all_labels = np.concatenate(labels_list).astype(np.int32) if labels_list else None

    # Derive tighter bbox from mask
    ys, xs = np.where(binary > 0)
    if len(xs) > 0:
        pad = max(5, int(min(H, W) * 0.02))
        box = np.array([
            max(0, xs.min() - pad),
            max(0, ys.min() - pad),
            min(W, xs.max() + pad),
            min(H, ys.max() + pad),
        ], dtype=np.float32)
    else:
        box = orig_box

    return all_coords, all_labels, box


def mask_to_sam_logits(mask_np, target_size=256):
    """Convert float mask → SAM logit space (inverse sigmoid) at target resolution.

    Args:
        mask_np: float32 numpy array (H, W) in [0, 1]
        target_size: output spatial size (SAM expects 256)

    Returns:
        float32 numpy array (1, target_size, target_size)
    """
    m = np.clip(mask_np, 1e-6, 1 - 1e-6)
    logits = np.log(m / (1 - m))
    if HAS_CV2:
        import cv2
        logits = cv2.resize(logits, (target_size, target_size),
                            interpolation=cv2.INTER_LINEAR)
    return logits[np.newaxis, :, :]


def parse_points_json(points_json):
    """Parse points JSON string to a list of dicts.

    Args:
        points_json: JSON string or list

    Returns:
        list of dicts with 'x', 'y', 'label' keys
    """
    import json
    try:
        return json.loads(points_json) if isinstance(points_json, str) else points_json
    except json.JSONDecodeError:
        return []


def points_to_arrays(points_list):
    """Convert points list to numpy arrays for SAM.

    Args:
        points_list: list of dicts with 'x', 'y', 'label'

    Returns:
        (coords, labels) — numpy arrays or (None, None) if empty
    """
    if not points_list:
        return None, None
    coords = [[float(p["x"]), float(p["y"])] for p in points_list]
    labels = [int(p.get("label", 1)) for p in points_list]
    return np.array(coords, dtype=np.float32), np.array(labels, dtype=np.int32)


def parse_bbox_input(bbox_json, bbox_input=None):
    """Parse bounding box from JSON string or BBOX input.

    Args:
        bbox_json: JSON string like '[x1,y1,x2,y2]' or '{"x":..,"y":..,"w":..,"h":..}'
        bbox_input: optional BBOX tuple [x, y, w, h]

    Returns:
        numpy array [x1, y1, x2, y2] or None
    """
    import json
    if bbox_input is not None:
        bx, by, bw, bh = bbox_input
        return np.array([bx, by, bx + bw, by + bh], dtype=np.float32)
    if bbox_json and bbox_json.strip():
        try:
            bdata = json.loads(bbox_json)
            if isinstance(bdata, list) and len(bdata) == 4:
                return np.array(bdata, dtype=np.float32)
            elif isinstance(bdata, dict):
                bx = float(bdata.get("x", 0))
                by = float(bdata.get("y", 0))
                bw = float(bdata.get("w", bdata.get("width", 0)))
                bh = float(bdata.get("h", bdata.get("height", 0)))
                return np.array([bx, by, bx + bw, by + bh], dtype=np.float32)
        except json.JSONDecodeError:
            pass
    return None


# ══════════════════════════════════════════════════════════════════════
#  Mask to BBox
# ══════════════════════════════════════════════════════════════════════

def mask_to_bbox(mask_t, W=None, H=None):
    """Extract bounding box [x, y, w, h] from a mask tensor.

    Args:
        mask_t: float tensor (H, W) or (B, H, W)
        W: fallback width (auto-detected from mask if None)
        H: fallback height (auto-detected from mask if None)
    """
    m = mask_t
    if m.dim() == 3:
        m = m[0]
    if H is None:
        H = m.shape[0]
    if W is None:
        W = m.shape[1]
    coords = torch.nonzero(m > 0.5, as_tuple=False)
    if coords.shape[0] > 0:
        y_min = int(coords[:, 0].min().item())
        y_max = int(coords[:, 0].max().item())
        x_min = int(coords[:, 1].min().item())
        x_max = int(coords[:, 1].max().item())
        return [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1]
    return [0, 0, W, H]


# ══════════════════════════════════════════════════════════════════════
#  Preview Helpers
# ══════════════════════════════════════════════════════════════════════

def make_mask_overlay_preview(img_t, mask_t, color=(0.0, 1.0, 0.0), alpha=0.35):
    """Create an overlay preview with colored mask on image.

    Args:
        img_t: float tensor (H, W, C) in [0, 1]
        mask_t: float tensor (H, W) in [0, 1]
        color: (R, G, B) tuple
        alpha: overlay opacity

    Returns: float tensor (H, W, 3)
    """
    preview = img_t[:, :, :3].clone()
    m = mask_t.unsqueeze(-1)
    overlay = torch.tensor(color, device=img_t.device).view(1, 1, 3).expand_as(preview)
    preview = preview * (1 - m * alpha) + overlay * m * alpha
    return preview.clamp(0, 1)


def make_split_preview(img_t, left_mask, right_mask):
    """Side-by-side preview: left=coarse (blue tint), right=refined (green tint).

    Args:
        img_t: float tensor (H, W, C)
        left_mask: float tensor (H, W)
        right_mask: float tensor (H, W)

    Returns: float tensor (H, W, 3)
    """
    H, W = img_t.shape[:2]
    preview = img_t[:, :, :3].clone()
    half = W // 2

    # Left: blue tint
    preview[:, :half, 2] = torch.clamp(preview[:, :half, 2] + left_mask[:, :half] * 0.35, 0, 1)
    # Right: green tint
    preview[:, half:, 1] = torch.clamp(preview[:, half:, 1] + right_mask[:, half:] * 0.35, 0, 1)
    # Dividing line (yellow)
    if 0 < half < W:
        preview[:, max(0, half - 1):half + 1, 0] = 1.0
        preview[:, max(0, half - 1):half + 1, 1] = 1.0
        preview[:, max(0, half - 1):half + 1, 2] = 0.0

    return preview


# ══════════════════════════════════════════════════════════════════════
#  Video Frame Extraction
# ══════════════════════════════════════════════════════════════════════

def extract_first_frame(images):
    """Extract first frame from an image batch (video sequence).

    Args:
        images: tensor (B, H, W, C)

    Returns: tensor (1, H, W, C) — single frame
    """
    if images.dim() == 4 and images.shape[0] > 0:
        return images[0:1]
    return images


def is_video_batch(images):
    """Check if the input is a video (multi-frame batch)."""
    return images.dim() == 4 and images.shape[0] > 1
