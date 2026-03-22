r"""
Eval test suite for MaskEditControl node pack.

Tests cover:
  - All pure-tensor nodes (no model/GPU required)
  - Edge cases: empty masks, single-pixel, large images, mismatched dims
  - Batch scenarios: B=1, B>1, batch mismatch
  - Type safety: output shapes, dtypes, value ranges
  - Dependency fallback: cv2-optional code paths

Run with:
    cd d:\PROJECT\Custom_Nodes\ComfyUI-CustomNodePacks
    python run_tests.py
"""

import sys
import os
import json
import torch
import numpy as np
import pytest

# Add the node pack root to sys.path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ══════════════════════════════════════════════════════════════════════
#  Fixtures
# ══════════════════════════════════════════════════════════════════════

@pytest.fixture
def small_image():
    """4-frame 64x64 RGB image batch."""
    return torch.rand(4, 64, 64, 3)


@pytest.fixture
def small_mask():
    """4-frame 64x64 mask batch."""
    m = torch.zeros(4, 64, 64)
    m[:, 16:48, 16:48] = 1.0
    return m


@pytest.fixture
def single_mask():
    """Single-frame 64x64 mask."""
    m = torch.zeros(1, 64, 64)
    m[:, 20:44, 20:44] = 1.0
    return m


@pytest.fixture
def empty_mask():
    """All-zeros mask."""
    return torch.zeros(1, 64, 64)


@pytest.fixture
def full_mask():
    """All-ones mask."""
    return torch.ones(1, 64, 64)


# ══════════════════════════════════════════════════════════════════════
#  MaskMath
# ══════════════════════════════════════════════════════════════════════

class TestMaskMath:
    @pytest.fixture(autouse=True)
    def setup(self):
        from nodes.mask_math import MaskMath
        self.node = MaskMath()

    @pytest.mark.parametrize("op", [
        "add_scalar", "multiply_scalar", "power", "invert", "clamp",
        "remap_range", "quantize", "threshold_hysteresis",
        "gamma", "contrast", "abs_diff_from_value",
    ])
    def test_all_operations_run(self, single_mask, op):
        (result,) = self.node.compute(single_mask, op, 0.5, 0.8)
        assert result.shape == single_mask.shape
        assert result.dtype == torch.float32
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_invert(self, single_mask):
        (result,) = self.node.compute(single_mask, "invert", 0, 0)
        # Where mask was 1, result should be 0 and vice versa
        assert torch.allclose(result + single_mask, torch.ones_like(result))

    def test_empty_mask(self, empty_mask):
        (result,) = self.node.compute(empty_mask, "invert", 0, 0)
        assert torch.allclose(result, torch.ones_like(result))

    def test_hysteresis_bounded(self):
        """Hysteresis on large mask shouldn't hang."""
        large = torch.rand(1, 512, 512) * 0.5 + 0.25  # all in [0.25, 0.75]
        (result,) = self.node.compute(large, "threshold_hysteresis", 0.3, 0.6)
        assert result.shape == large.shape

    def test_2d_mask_input(self):
        """2D mask (H, W) without batch dim."""
        m = torch.rand(64, 64)
        (result,) = self.node.compute(m, "invert", 0, 0)
        assert result.min() >= 0.0


# ══════════════════════════════════════════════════════════════════════
#  MaskCompositeAdvanced
# ══════════════════════════════════════════════════════════════════════

class TestMaskComposite:
    @pytest.fixture(autouse=True)
    def setup(self):
        from nodes.mask_composite import MaskCompositeAdvanced
        self.node = MaskCompositeAdvanced()

    @pytest.mark.parametrize("op", [
        "union", "intersect", "subtract", "xor", "blend",
        "min", "max", "difference",
    ])
    def test_all_operations_run(self, single_mask, empty_mask, op):
        (result,) = self.node.composite(single_mask, empty_mask, op, 0.5, False, False, 0.0)
        assert result.shape[0] == 1
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_xor_is_binary(self):
        """XOR should produce only 0s and 1s."""
        a = torch.tensor([[[0.8, 0.2], [0.6, 0.9]]])
        b = torch.tensor([[[0.3, 0.7], [0.1, 0.6]]])
        (result,) = self.node.composite(a, b, "xor", 0.5, False, False, 0.0)
        unique = torch.unique(result)
        assert all(v in [0.0, 1.0] for v in unique.tolist())

    def test_batch_mismatch(self, single_mask):
        """Different batch sizes should be handled."""
        mask_b3 = torch.rand(3, 64, 64)
        (result,) = self.node.composite(single_mask, mask_b3, "union", 0.5, False, False, 0.0)
        assert result.shape[0] == 3

    def test_spatial_mismatch(self):
        """Different spatial sizes should be handled."""
        a = torch.rand(1, 64, 64)
        b = torch.rand(1, 32, 48)
        (result,) = self.node.composite(a, b, "union", 0.5, False, False, 0.0)
        assert result.shape[1:] == a.shape[1:]


# ══════════════════════════════════════════════════════════════════════
#  MaskDrawFrame
# ══════════════════════════════════════════════════════════════════════

class TestMaskDrawFrame:
    @pytest.fixture(autouse=True)
    def setup(self):
        from nodes.mask_draw_frame import MaskDrawFrame
        self.node = MaskDrawFrame()

    @pytest.mark.parametrize("shape,params", [
        ("circle", '{"cx": 32, "cy": 32, "radius": 10}'),
        ("rectangle", '{"x": 10, "y": 10, "w": 20, "h": 20}'),
        ("ellipse", '{"cx": 32, "cy": 32, "rx": 15, "ry": 10}'),
        ("polygon", '{"points": [[10,10],[50,10],[50,50],[10,50]]}'),
        ("line", '{"x1": 0, "y1": 0, "x2": 63, "y2": 63, "thickness": 3}'),
    ])
    def test_all_shapes(self, shape, params):
        (result,) = self.node.draw(64, 64, shape, params, 1.0, 0.0, "set")
        assert result.shape == (1, 64, 64)
        assert result.max() <= 1.0
        assert result.min() >= 0.0

    def test_polygon_fallback_torch(self):
        """Polygon should work even when cv2 import fails."""
        pts = [[10, 10], [50, 10], [50, 50], [10, 50]]
        result = self.node._fill_polygon_torch(pts, 64, 64, 1.0)
        assert result.shape == (64, 64)
        assert result.sum() > 0  # should have filled pixels

    def test_feathered_circle(self):
        (result,) = self.node.draw(64, 64, "circle",
                                   '{"cx": 32, "cy": 32, "radius": 10}',
                                   1.0, 5.0, "set")
        assert result.shape == (1, 64, 64)

    def test_with_existing_mask(self, single_mask):
        (result,) = self.node.draw(64, 64, "circle",
                                   '{"cx": 32, "cy": 32, "radius": 5}',
                                   1.0, 0.0, "add", existing_mask=single_mask)
        assert result.shape == single_mask.shape

    def test_empty_polygon(self):
        """Empty polygon points should not crash."""
        (result,) = self.node.draw(64, 64, "polygon", '{"points": []}', 1.0, 0.0, "set")
        assert result.shape == (1, 64, 64)


# ══════════════════════════════════════════════════════════════════════
#  MaskBatchManager
# ══════════════════════════════════════════════════════════════════════

class TestMaskBatchManager:
    @pytest.fixture(autouse=True)
    def setup(self):
        from nodes.mask_batch_manager import MaskBatchManager
        self.node = MaskBatchManager()

    def test_slice(self, small_mask):
        result = self.node.manage(small_mask, "slice", 1, 3, "")[0]
        assert result.shape[0] == 2

    def test_pick(self, small_mask):
        result = self.node.manage(small_mask, "pick_frames", 0, 0, "0,2")[0]
        assert result.shape[0] == 2

    def test_reverse(self, small_mask):
        result = self.node.manage(small_mask, "reverse", 0, 0, "")[0]
        assert result.shape == small_mask.shape
        assert torch.allclose(result[0], small_mask[3])

    def test_repeat(self, single_mask):
        result = self.node.manage(single_mask, "repeat", 3, 0, "")[0]
        assert result.shape[0] == 3

    def test_empty_result_guard(self):
        """Slicing to empty should return at least 1 frame."""
        m = torch.rand(2, 64, 64)
        result = self.node.manage(m, "slice", 5, 10, "")[0]
        assert result.shape[0] >= 1


# ══════════════════════════════════════════════════════════════════════
#  BBox Nodes
# ══════════════════════════════════════════════════════════════════════

class TestBBoxNodes:
    @pytest.fixture(autouse=True)
    def setup(self):
        from nodes.bbox_nodes import BBoxCreate, BBoxFromMask, BBoxToMask, BBoxPad, BBoxCrop
        self.create = BBoxCreate()
        self.from_mask = BBoxFromMask()
        self.to_mask = BBoxToMask()
        self.pad = BBoxPad()
        self.crop = BBoxCrop()

    def test_create(self):
        (bbox, s) = self.create.create(10, 20, 100, 50)
        assert bbox == [10, 20, 100, 50]

    def test_from_mask(self, single_mask):
        (bbox, x, y, w, h, s) = self.from_mask.extract(single_mask, 5, 0, 0, 0.5)
        assert len(bbox) == 4
        assert bbox[2] > 0 and bbox[3] > 0

    def test_from_empty_mask(self, empty_mask):
        """Empty mask should still produce a valid bbox."""
        (bbox, x, y, w, h, s) = self.from_mask.extract(empty_mask, 0, 0, 0, 0.5)
        assert len(bbox) == 4

    def test_to_mask(self):
        bbox = [10, 10, 30, 30]
        (mask,) = self.to_mask.convert(bbox, 64, 64)
        assert mask.shape == (1, 64, 64)
        assert mask.sum() > 0

    def test_pad_negative_clamp(self):
        """Large negative pad should not produce negative dims."""
        bbox = [10, 10, 20, 20]
        (result, s) = self.pad.pad(bbox, 4096, 4096, 4096, 4096, 64, 64)
        assert result[2] >= 0 and result[3] >= 0

    def test_crop(self, small_image, single_mask):
        bbox = [10, 10, 30, 30]
        (img, mask, out_bbox) = self.crop.crop(small_image, bbox, mask=single_mask)
        assert img.shape[1] == 30  # height
        assert img.shape[2] == 30  # width
        assert mask.shape[1] == img.shape[1]
        assert mask.shape[2] == img.shape[2]

    def test_crop_mask_spatial_mismatch(self, small_image):
        """Mask with different spatial dims than image should still produce correct crop."""
        mismatched_mask = torch.ones(1, 32, 32)
        bbox = [5, 5, 20, 20]
        (img, mask, out_bbox) = self.crop.crop(small_image, bbox, mask=mismatched_mask)
        assert mask.shape[1] == img.shape[1]
        assert mask.shape[2] == img.shape[2]

    def test_crop_empty_region(self, small_image):
        """Crop region that doesn't overlap with image."""
        bbox = [1000, 1000, 10, 10]  # way outside 64x64 image
        (img, mask, out_bbox) = self.crop.crop(small_image, bbox)
        assert img.shape[1] >= 1 and img.shape[2] >= 1


# ══════════════════════════════════════════════════════════════════════
#  MaskTransformXY
# ══════════════════════════════════════════════════════════════════════

class TestMaskTransformXY:
    @pytest.fixture(autouse=True)
    def setup(self):
        from nodes.mask_transform_xy import MaskTransformXY
        self.node = MaskTransformXY()

    def test_identity(self, single_mask):
        (result,) = self.node.transform(
            single_mask, 0, 0, 0.0, 0.0, 0, 0, 0.0, 0.0, False
        )
        assert result.shape == single_mask.shape
        assert torch.allclose(result, single_mask)

    def test_offset(self, single_mask):
        (result,) = self.node.transform(
            single_mask, 10, 5, 0.0, 0.0, 0, 0, 0.0, 0.0, False
        )
        assert result.shape == single_mask.shape

    def test_blur(self, single_mask):
        (result,) = self.node.transform(
            single_mask, 0, 0, 3.0, 3.0, 0, 0, 0.0, 0.0, False
        )
        assert result.shape == single_mask.shape

    def test_empty_mask(self, empty_mask):
        (result,) = self.node.transform(
            empty_mask, 5, 5, 2.0, 2.0, 1, 1, 0.0, 0.0, False
        )
        assert result.shape == empty_mask.shape


# ══════════════════════════════════════════════════════════════════════
#  MaskPropagateVideo
# ══════════════════════════════════════════════════════════════════════

class TestMaskPropagateVideo:
    @pytest.fixture(autouse=True)
    def setup(self):
        from nodes.mask_propagate_video import MaskPropagateVideo
        self.node = MaskPropagateVideo()

    def test_static(self, small_image, single_mask):
        (masks, preview) = self.node.propagate(
            small_image, single_mask, 0, "static", 2.0, 1.0, 0.0, True
        )
        assert masks.shape[0] == small_image.shape[0]
        assert torch.allclose(masks[0], masks[1])

    def test_fade(self, small_image, single_mask):
        (masks, preview) = self.node.propagate(
            small_image, single_mask, 0, "fade", 2.0, 1.0, 0.0, True
        )
        assert masks.shape[0] == small_image.shape[0]
        # Last frame should be dimmer than first
        assert masks[-1].sum() <= masks[0].sum() + 1e-6

    def test_scale_linear(self, small_image, single_mask):
        (masks, preview) = self.node.propagate(
            small_image, single_mask, 0, "scale_linear", 2.0, 1.0, 0.5, True
        )
        assert masks.shape[0] == small_image.shape[0]

    def test_sam2_no_model_fallback(self, small_image, single_mask):
        """SAM2 mode without model should fall back to static."""
        (masks, preview) = self.node.propagate(
            small_image, single_mask, 0, "sam2_video", 2.0, 1.0, 0.0, True
        )
        assert masks.shape[0] == small_image.shape[0]


# ══════════════════════════════════════════════════════════════════════
#  MaskPreviewOverlay
# ══════════════════════════════════════════════════════════════════════

class TestMaskPreview:
    @pytest.fixture(autouse=True)
    def setup(self):
        from nodes.mask_preview import MaskPreviewOverlay
        self.node = MaskPreviewOverlay()

    @pytest.mark.parametrize("mode", [
        "overlay", "binary_mask", "edge_only", "side_by_side", "alpha_channel",
    ])
    def test_all_modes(self, small_image, small_mask, mode):
        (result,) = self.node.preview(
            small_image, small_mask, mode,
            1.0, 0.0, 0.0, 0.5, 2, False, 0.0, 1.0, 0.0,
        )
        assert result.shape[0] == small_image.shape[0]

    def test_batch_mismatch_expand(self, small_image):
        """Mask batch=1 with image batch=4 should expand."""
        mask = torch.zeros(1, 64, 64)
        (result,) = self.node.preview(
            small_image, mask, "overlay",
            1.0, 0.0, 0.0, 0.5, 2, False, 0.0, 1.0, 0.0,
        )
        assert result.shape[0] == 4

    def test_non_singleton_batch_mismatch(self, small_image):
        """Mask batch=2 with image batch=4 should handle via repeat."""
        mask = torch.rand(2, 64, 64)
        (result,) = self.node.preview(
            small_image, mask, "overlay",
            1.0, 0.0, 0.0, 0.5, 2, False, 0.0, 1.0, 0.0,
        )
        assert result.shape[0] == 4

    def test_side_by_side_width(self, small_image, small_mask):
        """Side-by-side should double the width."""
        (result,) = self.node.preview(
            small_image, small_mask, "side_by_side",
            1.0, 0.0, 0.0, 0.5, 2, False, 0.0, 1.0, 0.0,
        )
        assert result.shape[2] == small_image.shape[2] * 2


# ══════════════════════════════════════════════════════════════════════
#  PointsMaskEditor
# ══════════════════════════════════════════════════════════════════════

class TestPointsMaskEditor:
    @pytest.fixture(autouse=True)
    def setup(self):
        from nodes.points_mask_editor import PointsMaskEditor
        self.node = PointsMaskEditor()

    def test_empty_editor(self):
        result = self.node.generate(64, 64, '{"points":[],"bboxes":[]}', 3.0, 1.0, True)
        if isinstance(result, dict):
            mask = result["result"][0]
        else:
            mask = result[0]
        assert mask.shape == (1, 64, 64)
        assert mask.sum() == 0

    def test_positive_point(self):
        data = json.dumps({"points": [{"x": 32, "y": 32, "label": 1, "radius": 5}], "bboxes": []})
        result = self.node.generate(64, 64, data, 3.0, 1.0, True)
        if isinstance(result, dict):
            mask = result["result"][0]
        else:
            mask = result[0]
        assert mask.sum() > 0

    def test_negative_point(self):
        data = json.dumps({"points": [
            {"x": 32, "y": 32, "label": 1, "radius": 20},
            {"x": 32, "y": 32, "label": 0, "radius": 5},
        ], "bboxes": []})
        result = self.node.generate(64, 64, data, 3.0, 1.0, True)
        if isinstance(result, dict):
            mask = result["result"][0]
        else:
            mask = result[0]
        # Center should have a hole
        assert mask[0, 32, 32] < 0.5

    def test_zero_radius_no_nan(self):
        """Zero radius should not produce NaN."""
        data = json.dumps({"points": [{"x": 32, "y": 32, "label": 1, "radius": 0}], "bboxes": []})
        result = self.node.generate(64, 64, data, 3.0, 1.0, True)
        if isinstance(result, dict):
            mask = result["result"][0]
        else:
            mask = result[0]
        assert not torch.isnan(mask).any()

    def test_bbox_output(self):
        data = json.dumps({
            "points": [],
            "bboxes": [[10, 10, 50, 50, 1]],
        })
        result = self.node.generate(64, 64, data, 3.0, 1.0, True)
        if isinstance(result, dict):
            mask = result["result"][0]
        else:
            mask = result[0]
        assert mask[0, 30, 30] == 1.0  # inside bbox

    def test_existing_mask_resize(self):
        """existing_mask with different dims should be resized."""
        existing = torch.ones(1, 32, 32)
        result = self.node.generate(64, 64, '{"points":[],"bboxes":[]}',
                                    3.0, 1.0, True, existing_mask=existing)
        if isinstance(result, dict):
            mask = result["result"][0]
        else:
            mask = result[0]
        assert mask.shape == (1, 64, 64)


# ══════════════════════════════════════════════════════════════════════
#  TrimapGenerator
# ══════════════════════════════════════════════════════════════════════

class TestTrimapGenerator:
    @pytest.fixture(autouse=True)
    def setup(self):
        from nodes.trimap_generator import TrimapGeneratorMEC
        self.node = TrimapGeneratorMEC()

    def test_basic(self, single_mask):
        (trimap, fg, unknown) = self.node.generate(
            single_mask, 5, 1.0, 1.5, 0.0, 0.5
        )
        assert trimap.shape == single_mask.shape
        # Trimap should have 3 regions: bg ≈ 0, unknown ≈ 0.5, fg ≈ 1
        assert trimap.min() >= 0.0
        assert trimap.max() <= 1.0

    def test_empty_mask(self, empty_mask):
        (trimap, fg, unknown) = self.node.generate(
            empty_mask, 5, 1.0, 1.5, 0.0, 0.5
        )
        assert trimap.shape == empty_mask.shape


# ══════════════════════════════════════════════════════════════════════
#  VideoFrameExtractor
# ══════════════════════════════════════════════════════════════════════

class TestVideoFrameExtractor:
    @pytest.fixture(autouse=True)
    def setup(self):
        from nodes.video_frame_extractor import VideoFrameExtractorMEC
        self.node = VideoFrameExtractorMEC()

    def test_extract_first(self, small_image):
        (frame, total, is_video) = self.node.extract(small_image, 0, "first")
        assert frame.shape == (1, 64, 64, 3)
        assert total == 4

    def test_extract_last(self, small_image):
        (frame, total, is_video) = self.node.extract(small_image, 3, "last")
        assert frame.shape == (1, 64, 64, 3)

    def test_extract_out_of_range(self, small_image):
        """Out of range index should clamp, not crash."""
        (frame, total, is_video) = self.node.extract(small_image, 999, "specific_frame")
        assert frame.shape == (1, 64, 64, 3)

    def test_single_frame(self):
        img = torch.rand(1, 64, 64, 3)
        (frame, total, is_video) = self.node.extract(img, 0, "first")
        assert total == 1


# ══════════════════════════════════════════════════════════════════════
#  SAMViTMattePipeline (input parsing only — no model)
# ══════════════════════════════════════════════════════════════════════

class TestSAMViTMattePipeline:
    @pytest.fixture(autouse=True)
    def setup(self):
        from nodes.sam_vitmatte_pipeline import SAMViTMattePipelineMEC
        self.node = SAMViTMattePipelineMEC()

    def test_static_helpers(self):
        """Test static refinement helpers don't crash."""
        mask_np = np.zeros((64, 64), dtype=np.float32)
        mask_np[20:44, 20:44] = 1.0
        result = self.node._try_laplacian(mask_np, 5, 0.85)
        # May return None if cv2 not available, which is fine

    def test_guided_single_no_cv2(self):
        """If cv2 unavailable, should return None gracefully."""
        img_np = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        mask_np = np.zeros((64, 64), dtype=np.float32)
        result = self.node._try_guided_single(img_np, mask_np, 5, 0.85)
        # Should either return a tensor or None — never crash


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
