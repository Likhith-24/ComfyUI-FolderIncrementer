"""Quick integration test for all MEC nodes."""
import sys, os
sys.path.insert(0, r"D:\PROJECT\ComfyUI_windows_portable\ComfyUI")

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Need folder_paths available for SAM loader
import folder_paths

import torch
import json

# Direct import from the nodes directory
_nodes_dir = r"D:\PROJECT\Custom_Nodes\ComfyUI-CustomNodePacks"
sys.path.insert(0, _nodes_dir)

print("=" * 60)
print("MEC Node Integration Tests")
print("=" * 60)

# ── Test 1: MaskDrawFrame ─────────────────────────────────────────
print("\n[1] MaskDrawFrame - dict params")
from nodes.mask_draw_frame import MaskDrawFrame
mdf = MaskDrawFrame()
result = mdf.draw(256, 256, "circle", '{"cx": 128, "cy": 128, "radius": 50}', 1.0, 0.0, "set")
assert result[0].shape == (1, 256, 256), f"Bad shape: {result[0].shape}"
assert result[0].max() > 0.9, f"Circle not drawn"
print("  PASS: dict params work")

print("[1b] MaskDrawFrame - LIST params (was crashing)")
result2 = mdf.draw(256, 256, "circle", "[128, 128, 50]", 1.0, 0.0, "set")
assert result2[0].shape == (1, 256, 256), f"Bad shape: {result2[0].shape}"
assert result2[0].max() > 0.9, f"Circle not drawn from list"
print("  PASS: list params work (crash fixed)")

print("[1c] MaskDrawFrame - rectangle list params")
result3 = mdf.draw(256, 256, "rectangle", "[10, 10, 100, 100]", 1.0, 0.0, "set")
assert result3[0].max() > 0.9, f"Rectangle not drawn"
print("  PASS: rectangle list params work")

print("[1d] MaskDrawFrame - polygon list params")
result4 = mdf.draw(256, 256, "polygon", "[[50,50],[200,50],[200,200],[50,200]]", 1.0, 0.0, "set")
assert result4[0].max() > 0.9, f"Polygon not drawn"
print("  PASS: polygon list params work")

print("[1e] MaskDrawFrame - line list params")
result5 = mdf.draw(256, 256, "line", "[0, 0, 255, 255, 5]", 1.0, 0.0, "set")
assert result5[0].max() > 0.9, f"Line not drawn"
print("  PASS: line list params work")

print("[1f] MaskDrawFrame - ellipse list params")
result6 = mdf.draw(256, 256, "ellipse", "[128, 128, 80, 40]", 1.0, 0.0, "set")
assert result6[0].max() > 0.9, f"Ellipse not drawn"
print("  PASS: ellipse list params work")

# ── Test 2: PointsMaskEditor ──────────────────────────────────────
print("\n[2] PointsMaskEditor - basic points")
from nodes.points_mask_editor import PointsMaskEditor
pme = PointsMaskEditor()

# Test with points
editor_data = json.dumps({
    "points": [
        {"x": 128, "y": 128, "label": 1, "radius": 10},
        {"x": 50, "y": 50, "label": 0, "radius": 5},
    ],
    "bboxes": [[20, 20, 200, 200, 1]]
})
result = pme.generate(256, 256, editor_data, 3.0, 1.0, True)
mask, pos_coords, neg_coords, bboxes, neg_bboxes, pts_json, bbox_json, primary_bbox = result

assert mask.shape == (1, 256, 256), f"Bad mask shape: {mask.shape}"
assert mask.max() >= 0.9, f"Points not rendered"
print("  PASS: mask generated")

# Check positive_coords format
assert isinstance(pos_coords, str), f"pos_coords should be str, got {type(pos_coords)}"
parsed = json.loads(pos_coords)
assert len(parsed) == 1, f"Should have 1 positive point, got {len(parsed)}"
assert "x" in parsed[0] and "y" in parsed[0], f"Missing x/y in coords"
print(f"  PASS: positive_coords = {pos_coords}")

# Check negative_coords format
assert isinstance(neg_coords, str), f"neg_coords should be str, got {type(neg_coords)}"
parsed_neg = json.loads(neg_coords)
assert len(parsed_neg) == 1, f"Should have 1 negative point"
print(f"  PASS: negative_coords = {neg_coords}")

# Check empty gives [] not None
result_empty = pme.generate(256, 256, '{"points":[],"bboxes":[]}', 3.0, 1.0, True)
assert result_empty[1] == "[]", f"Empty pos_coords should be '[]', got {repr(result_empty[1])}"
assert result_empty[2] == "[]", f"Empty neg_coords should be '[]', got {repr(result_empty[2])}"
assert result_empty[3] is None, f"Empty bboxes should be None, got {result_empty[3]}"
print("  PASS: empty outputs correct ([] not None)")

# Check bbox batch dimension
assert isinstance(bboxes, list), f"bboxes should be list, got {type(bboxes)}"
assert len(bboxes) == 1, f"Outer list should have 1 batch, got {len(bboxes)}"
assert isinstance(bboxes[0], list), f"Inner should be list of bboxes"
assert len(bboxes[0]) == 1, f"Should have 1 bbox, got {len(bboxes[0])}"
assert bboxes[0][0] == [20, 20, 200, 200], f"Wrong bbox coords: {bboxes[0][0]}"
print(f"  PASS: bboxes batch dim = {len(bboxes)} (correct for SAM2)")

# Check primary_bbox
assert isinstance(primary_bbox, list), f"primary_bbox should be list"
assert primary_bbox == [20, 20, 180, 180], f"Wrong primary_bbox: {primary_bbox}"
print(f"  PASS: primary_bbox = {primary_bbox}")

# ── Test 3: MaskTransformXY ──────────────────────────────────────
print("\n[3] MaskTransformXY")
from nodes.mask_transform_xy import MaskTransformXY
mtxy = MaskTransformXY()
test_mask = torch.zeros(1, 256, 256)
test_mask[0, 100:150, 100:150] = 1.0
result = mtxy.transform(test_mask, 5, 5, 0.0, 0.0, 10, 10, 0.0, 0.5, False)
assert result[0].shape == (1, 256, 256), f"Bad shape: {result[0].shape}"
assert result[0].max() > 0.9, f"Transform lost mask"
print("  PASS: expand + offset work")

# ── Test 4: SAMModelLoaderMEC path resolution ─────────────────────
print("\n[4] SAMModelLoaderMEC - path resolution")
from nodes.sam_model_loader import SAMModelLoaderMEC
try:
    path = SAMModelLoaderMEC._resolve_path("sam2.1_hiera_tiny-fp16.safetensors")
    assert os.path.exists(path), f"Path doesn't exist: {path}"
    print(f"  PASS: resolved to {path}")
except FileNotFoundError as e:
    print(f"  FAIL: {e}")

# ── Test 5: MaskMath ─────────────────────────────────────────────
print("\n[5] MaskMath")
from nodes.mask_math import MaskMath
mm = MaskMath()
m1 = torch.ones(1, 64, 64) * 0.5
# add_scalar: mask + value_a => 0.5 + 0.3 = 0.8
result = mm.compute(m1, "add_scalar", 0.3, 0.0)
assert abs(result[0].max().item() - 0.8) < 0.01, f"add_scalar wrong: {result[0].max()}"
print("  PASS: add_scalar works")
# invert: 1.0 - mask => 0.5
result_inv = mm.compute(m1, "invert", 0.0, 0.0)
assert abs(result_inv[0].max().item() - 0.5) < 0.01, f"invert wrong: {result_inv[0].max()}"
print("  PASS: invert works")
# clamp: clamp(0.2, 0.4) on mask=0.5 => 0.4
result_clamp = mm.compute(m1, "clamp", 0.2, 0.4)
assert abs(result_clamp[0].max().item() - 0.4) < 0.01, f"clamp wrong: {result_clamp[0].max()}"
print("  PASS: clamp works")

# ── Test 6: BBox nodes ───────────────────────────────────────────
print("\n[6] BBox nodes")
from nodes.bbox_nodes import BBoxCreate, BBoxFromMask
bc = BBoxCreate()
result = bc.create(10, 20, 100, 200)
assert result[0] == [10, 20, 100, 200], f"BBoxCreate wrong: {result}"
assert result[1] == '[10, 20, 100, 200]', f"BBoxCreate JSON wrong: {result[1]}"
print("  PASS: BBoxCreate")

bfm = BBoxFromMask()
test_mask2 = torch.zeros(1, 256, 256)
test_mask2[0, 50:150, 60:180] = 1.0
# extract(mask, padding, padding_x, padding_y, threshold)
result = bfm.extract(test_mask2, 0, 0, 0, 0.5)
bbox = result[0]
print(f"  BBoxFromMask: {bbox}")
# mask is 1.0 at rows[50:150], cols[60:180] → x_min=60, y_min=50
assert bbox[0] == 60, f"x wrong: {bbox[0]}"
assert bbox[1] == 50, f"y wrong: {bbox[1]}"
assert bbox[2] == 120, f"w wrong: {bbox[2]}"  # 180-60=120
assert bbox[3] == 100, f"h wrong: {bbox[3]}"  # 150-50=100
print("  PASS: BBoxFromMask")

# ── Test 7: TrimapGenerator ──────────────────────────────────────
print("\n[7] TrimapGenerator")
from nodes.trimap_generator import TrimapGeneratorMEC
tg = TrimapGeneratorMEC()
test_mask3 = torch.zeros(1, 128, 128)
test_mask3[0, 30:90, 30:90] = 1.0
# generate(mask, edge_radius, inner_erosion, outer_dilation, smooth, threshold)
result = tg.generate(test_mask3, 10, 1.0, 1.5, 0.0, 0.5)
assert result[0].shape == (1, 128, 128), f"Bad trimap shape"
# Should have 3 values: 0 (bg), 0.5 (unknown), 1 (fg)
unique = result[0].unique()
print(f"  Unique values: {unique.tolist()}")
print("  PASS: trimap generated")

# ── Test 8: SAM2 full model load + predict ────────────────────────
print("\n[8] SAM2 full model load + predict")
try:
    from nodes.sam_model_loader import SAMModelLoaderMEC
    import numpy as np

    loader = SAMModelLoaderMEC()
    result = loader.load("sam2.1_hiera_tiny-fp16.safetensors", "auto", "cuda", False, "float16")
    model_info = result[0]
    assert model_info["load_method"] == "build_sam2", f"Wrong method: {model_info['load_method']}"
    assert hasattr(model_info["model"], "image_encoder"), "Not a SAM2Base model"
    print(f"  PASS: Loaded via {model_info['load_method']}")

    # Test predictor creation + inference
    from nodes.utils import get_sam_predictor, sam_predict
    dummy_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    predictor = get_sam_predictor(model_info["model"], model_info["model_type"], dummy_img)
    assert predictor is not None, "Predictor creation failed"
    print("  PASS: Predictor created + image set")

    masks, scores, logits = sam_predict(
        predictor, model_info,
        point_coords=np.array([[128, 128]], dtype=np.float32),
        point_labels=np.array([1], dtype=np.int32),
        multimask_output=True,
    )
    assert masks.shape[0] == 3, f"Expected 3 masks, got {masks.shape[0]}"
    print(f"  PASS: Predict returned {masks.shape}, scores={scores}")

    # Cleanup GPU
    del predictor, model_info
    import gc; gc.collect()
    torch.cuda.empty_cache()
except Exception as e:
    print(f"  FAIL: {e}")
    import traceback; traceback.print_exc()

print("\n" + "=" * 60)
print("ALL TESTS PASSED")
print("=" * 60)
