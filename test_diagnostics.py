"""
ComfyUI-MaskEditControl – Full Diagnostic Tests
Run: conda activate animal_pose && python test_diagnostics.py
"""
import sys
import os
import json
import traceback

print("=" * 60)
print("  ComfyUI-MaskEditControl Diagnostics")
print("=" * 60)
print(f"Python: {sys.version}")
print(f"CWD: {os.getcwd()}")
print()

errors = []
warnings = []

# ── 1. Import all Python modules ──────────────────────────────
print("[1] Importing Python modules...")
try:
    sys.path.insert(0, os.getcwd())
    from folder_incrementer import FolderIncrementer, FolderIncrementerReset, FolderIncrementerSet
    from folder_incrementer import NODE_CLASS_MAPPINGS as FOLDER_MAP
    from folder_incrementer import NODE_DISPLAY_NAME_MAPPINGS as FOLDER_DISPLAY
    print("    folder_incrementer.py: OK")
except Exception as e:
    errors.append(f"folder_incrementer import: {e}")
    traceback.print_exc()

try:
    from nodes.points_mask_editor import PointsMaskEditor
    print("    points_mask_editor.py: OK")
except Exception as e:
    errors.append(f"points_mask_editor import: {e}")
    traceback.print_exc()

try:
    from nodes.mask_transform_xy import MaskTransformXY
    from nodes.mask_draw_frame import MaskDrawFrame
    from nodes.mask_propagate_video import MaskPropagateVideo
    from nodes.mask_composite import MaskCompositeAdvanced
    from nodes.mask_preview import MaskPreviewOverlay
    from nodes.mask_batch_manager import MaskBatchManager
    from nodes.mask_math import MaskMath
    from nodes.bbox_nodes import BBoxCreate, BBoxFromMask, BBoxToMask, BBoxPad, BBoxCrop
    print("    All other node modules: OK")
except Exception as e:
    errors.append(f"Other node import: {e}")
    traceback.print_exc()

try:
    from nodes.sam_model_loader import SAMModelLoaderMEC
    from nodes.sam_mask_generator import SAMMaskGeneratorMEC
    from nodes.vitmatte_refiner import ViTMatteRefinerMEC
    from nodes.sam_vitmatte_pipeline import SAMViTMattePipelineMEC
    print("    SAM/ViTMatte modules: OK")
except Exception as e:
    warnings.append(f"SAM/ViTMatte import (optional deps may be missing): {e}")
    print(f"    SAM/ViTMatte modules: WARNING - {e}")

print()

# ── 2. Validate NODE_CLASS_MAPPINGS ───────────────────────────
print("[2] Validating NODE_CLASS_MAPPINGS...")
try:
    # Can't import __init__ directly (relative imports), so build mappings manually
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

    # Folder incrementer mappings
    NODE_CLASS_MAPPINGS.update(FOLDER_MAP)
    NODE_DISPLAY_NAME_MAPPINGS.update(FOLDER_DISPLAY)

    # MEC node mappings
    _mec_classes = {
        "PointsMaskEditor": PointsMaskEditor,
        "MaskTransformXY": MaskTransformXY,
        "MaskDrawFrame": MaskDrawFrame,
        "MaskPropagateVideo": MaskPropagateVideo,
        "MaskCompositeAdvanced": MaskCompositeAdvanced,
        "MaskPreviewOverlay": MaskPreviewOverlay,
        "MaskBatchManager": MaskBatchManager,
        "MaskMath": MaskMath,
        "BBoxCreate": BBoxCreate,
        "BBoxFromMask": BBoxFromMask,
        "BBoxToMask": BBoxToMask,
        "BBoxPad": BBoxPad,
        "BBoxCrop": BBoxCrop,
    }
    NODE_CLASS_MAPPINGS.update(_mec_classes)
    for name in _mec_classes:
        NODE_DISPLAY_NAME_MAPPINGS[name] = f"{name} (MEC)"

    print(f"    Total nodes registered: {len(NODE_CLASS_MAPPINGS)}")
    for name, cls in NODE_CLASS_MAPPINGS.items():
        has_input = hasattr(cls, "INPUT_TYPES") and callable(cls.INPUT_TYPES)
        has_func = hasattr(cls, "FUNCTION")
        has_ret = hasattr(cls, "RETURN_TYPES")
        has_cat = hasattr(cls, "CATEGORY")
        issues = []
        if not has_input:
            issues.append("missing INPUT_TYPES")
        if not has_func:
            issues.append("missing FUNCTION")
        if not has_ret:
            issues.append("missing RETURN_TYPES")
        if not has_cat:
            issues.append("missing CATEGORY")
        if has_func:
            func_name = cls.FUNCTION
            if not hasattr(cls, func_name):
                issues.append(f'FUNCTION "{func_name}" method not found')
        status = "OK" if not issues else f"ISSUES: {issues}"
        print(f"    {name}: {status}")
        if issues:
            errors.append(f"{name}: {issues}")

    missing_display = set(NODE_CLASS_MAPPINGS.keys()) - set(NODE_DISPLAY_NAME_MAPPINGS.keys())
    if missing_display:
        warnings.append(f"Missing display names: {missing_display}")
        print(f"    WARNING: Missing display names for: {missing_display}")
    else:
        print("    All display names present: OK")
except Exception as e:
    errors.append(f"NODE_CLASS_MAPPINGS validation: {e}")
    traceback.print_exc()

print()

# ── 3. Validate INPUT_TYPES schemas ──────────────────────────
print("[3] Validating INPUT_TYPES schemas...")
for name, cls in NODE_CLASS_MAPPINGS.items():
    try:
        inputs = cls.INPUT_TYPES()
        req = inputs.get("required", {})
        opt = inputs.get("optional", {})
        print(f"    {name}: {len(req)} required, {len(opt)} optional inputs")
    except Exception as e:
        errors.append(f"{name}.INPUT_TYPES(): {e}")
        print(f"    {name}: ERROR - {e}")
print()

# ── 4. FolderIncrementer logic test ───────────────────────────
print("[4] Testing FolderIncrementer logic...")
try:
    inc = FolderIncrementer()

    # Test basic increment
    test_label = "__diag_test__"
    r1 = inc.increment(prefix="v", padding=3, label=test_label)
    print(f"    Basic increment: version={r1[0]}, num={r1[1]}, folder={r1[2]}, subfolder={r1[3]}, prefix={r1[4]}")
    assert len(r1) == 5, f"Expected 5 outputs, got {len(r1)}"
    assert r1[0].startswith("v"), f"Expected v prefix, got {r1[0]}"

    # Test with source_filename
    r2 = inc.increment(prefix="v", padding=3, label="ignored", source_filename="my_video.mp4")
    print(f"    With source_filename: version={r2[0]}, folder={r2[2]}, subfolder={r2[3]}, filename_prefix={r2[4]}")
    assert r2[2] == "my_video", f"Expected folder_name=my_video, got {r2[2]}"
    assert "my_video" in r2[3], f"Expected subfolder to contain my_video"
    assert "my_video" in r2[4], f"Expected filename_prefix to contain my_video"

    # Test with path-style source filename
    r3 = inc.increment(prefix="v", padding=3, label="ignored", source_filename="subfolder/test_image.png")
    print(f"    With path source: version={r3[0]}, folder={r3[2]}, subfolder={r3[3]}, filename_prefix={r3[4]}")
    assert r3[2] == "test_image", f"Expected folder_name=test_image, got {r3[2]}"

    # Test independent counters per filename
    r4 = inc.increment(prefix="v", padding=3, label="ignored", source_filename="my_video.mp4")
    assert r4[1] == r2[1] + 1, f"Expected counter to increment for same filename, got {r4[1]}"
    print(f"    Independent counter: my_video run2 = {r4[0]} (num={r4[1]}) - OK")

    # Cleanup test counters
    from folder_incrementer import _load_counters, _save_counters
    counters = _load_counters()
    for k in [test_label, "my_video", "test_image"]:
        counters.pop(k, None)
    _save_counters(counters)
    print("    Cleanup: OK")
    print("    FolderIncrementer: ALL TESTS PASSED")
except Exception as e:
    errors.append(f"FolderIncrementer test: {e}")
    traceback.print_exc()

print()

# ── 5. PointsMaskEditor mask generation test ──────────────────
print("[5] Testing PointsMaskEditor mask generation...")
try:
    import torch

    editor = PointsMaskEditor()

    # Test empty editor data
    mask, pts_json, bbox_json, primary_bbox = editor.generate(
        width=256, height=256,
        editor_data='{"points":[],"bboxes":[]}',
        default_radius=3.0, softness=1.0, normalize=True,
    )
    assert mask.shape == (1, 256, 256), f"Expected (1,256,256), got {mask.shape}"
    assert mask.sum() == 0, "Empty editor should produce zero mask"
    print(f"    Empty mask: shape={list(mask.shape)}, sum={mask.sum():.4f} - OK")

    # Test with a positive point
    data_pts = json.dumps({"points": [{"x": 128, "y": 128, "label": 1, "radius": 10}], "bboxes": []})
    mask2, _, _, _ = editor.generate(
        width=256, height=256, editor_data=data_pts,
        default_radius=3.0, softness=1.0, normalize=True,
    )
    assert mask2.sum() > 0, "Positive point should produce non-zero mask"
    assert mask2[0, 128, 128] > 0.5, "Center of point should be bright"
    print(f"    Positive point mask: sum={mask2.sum():.2f}, center={mask2[0, 128, 128]:.4f} - OK")

    # Test with a negative point (subtracts from existing)
    data_neg = json.dumps(
        {
            "points": [
                {"x": 128, "y": 128, "label": 1, "radius": 20},
                {"x": 128, "y": 128, "label": 0, "radius": 10},
            ],
            "bboxes": [],
        }
    )
    mask3, _, _, _ = editor.generate(
        width=256, height=256, editor_data=data_neg,
        default_radius=3.0, softness=1.0, normalize=True,
    )
    assert mask3[0, 128, 128] < mask2[0, 128, 128], "Negative point should reduce center"
    print(f"    Negative point subtract: center={mask3[0, 128, 128]:.4f} (< {mask2[0, 128, 128]:.4f}) - OK")

    # Test with bbox
    data_bbox = json.dumps({"points": [], "bboxes": [[50, 50, 200, 200]]})
    mask4, _, bbox_j, prim = editor.generate(
        width=256, height=256, editor_data=data_bbox,
        default_radius=3.0, softness=1.0, normalize=True,
    )
    assert mask4[0, 100, 100] == 1.0, "Inside bbox should be 1.0"
    assert mask4[0, 10, 10] == 0.0, "Outside bbox should be 0.0"
    print(f"    BBox mask: inside={mask4[0, 100, 100]:.1f}, outside={mask4[0, 10, 10]:.1f} - OK")

    # Test primary_bbox output format [x, y, w, h]
    assert prim == [50, 50, 150, 150], f"Expected primary_bbox [50,50,150,150], got {prim}"
    print(f"    Primary bbox format (x,y,w,h): {prim} - OK")

    # Test with reference_image (auto width/height)
    ref_img = torch.zeros(1, 480, 640, 3)
    mask5, _, _, _ = editor.generate(
        width=100, height=100,  # should be overridden
        editor_data='{"points":[],"bboxes":[]}',
        default_radius=3.0, softness=1.0, normalize=True,
        reference_image=ref_img,
    )
    assert mask5.shape == (1, 480, 640), f"Expected (1,480,640) from ref image, got {list(mask5.shape)}"
    print(f"    Auto size from ref image: shape={list(mask5.shape)} - OK")

    # Test mask accuracy: point at exact pixel should affect that pixel
    data_acc = json.dumps({"points": [{"x": 50.0, "y": 75.0, "label": 1, "radius": 5}], "bboxes": []})
    mask6, _, _, _ = editor.generate(
        width=512, height=512, editor_data=data_acc,
        default_radius=3.0, softness=0.0, normalize=True,
    )
    assert mask6[0, 75, 50] == 1.0, f"Hard circle center pixel should be 1.0, got {mask6[0, 75, 50]}"
    assert mask6[0, 0, 0] == 0.0, "Far corner should be 0.0"
    print(f"    Pixel accuracy (hard): [75,50]={mask6[0, 75, 50]:.1f}, [0,0]={mask6[0, 0, 0]:.1f} - OK")

    # Test mask with existing_mask input
    base = torch.ones(1, 256, 256) * 0.5
    data_existing = json.dumps({"points": [{"x": 128, "y": 128, "label": 1, "radius": 5}], "bboxes": []})
    mask7, _, _, _ = editor.generate(
        width=256, height=256, editor_data=data_existing,
        default_radius=3.0, softness=1.0, normalize=True,
        existing_mask=base,
    )
    assert mask7[0, 0, 0] == 0.5, f"Untouched area should keep base value 0.5, got {mask7[0, 0, 0]}"
    assert mask7[0, 128, 128] >= 0.5, f"Center should be at least 0.5"
    print(f"    Existing mask overlay: corner={mask7[0, 0, 0]:.2f}, center={mask7[0, 128, 128]:.4f} - OK")

    # Test points_json and bbox_json output
    data_out = json.dumps({
        "points": [{"x": 10, "y": 20, "label": 1, "radius": 3}],
        "bboxes": [[0, 0, 100, 100]],
    })
    _, pts_out, bbox_out, _ = editor.generate(
        width=256, height=256, editor_data=data_out,
        default_radius=3.0, softness=1.0, normalize=True,
    )
    pts_parsed = json.loads(pts_out)
    bbox_parsed = json.loads(bbox_out)
    assert len(pts_parsed) == 1, f"Expected 1 point in output, got {len(pts_parsed)}"
    assert pts_parsed[0]["x"] == 10, f"Point x should be 10"
    assert bbox_parsed == [0, 0, 100, 100], f"bbox output mismatch: {bbox_parsed}"
    print(f"    JSON outputs: points={pts_out}, bbox={bbox_out} - OK")

    print("    PointsMaskEditor: ALL TESTS PASSED")
except Exception as e:
    errors.append(f"PointsMaskEditor test: {e}")
    traceback.print_exc()

print()

# ── 6. JS file syntax validation ─────────────────────────────
print("[6] Validating JS files...")
js_dir = os.path.join(os.getcwd(), "js")
for fname in os.listdir(js_dir):
    if fname.endswith(".js"):
        fpath = os.path.join(js_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            content = f.read()

        # Basic bracket/brace/paren balance check
        stack = []
        bracket_map = {"{": "}", "[": "]", "(": ")"}
        line_num = 1
        in_string = None
        prev_char = ""
        ok = True
        for i, ch in enumerate(content):
            if ch == "\n":
                line_num += 1
            # Skip string contents
            if in_string:
                if ch == in_string and prev_char != "\\":
                    in_string = None
                prev_char = ch
                continue
            if ch in ('"', "'", "`"):
                in_string = ch
                prev_char = ch
                continue
            # Skip line comments
            if ch == "/" and i + 1 < len(content) and content[i + 1] == "/":
                # skip to end of line
                nl = content.find("\n", i)
                if nl == -1:
                    break
                continue

            if ch in bracket_map:
                stack.append((bracket_map[ch], line_num))
            elif ch in bracket_map.values():
                if stack and stack[-1][0] == ch:
                    stack.pop()
                elif stack:
                    errors.append(f"{fname}:{line_num}: unexpected '{ch}', expected '{stack[-1][0]}'")
                    ok = False
                    break

            prev_char = ch

        if ok and stack:
            errors.append(f"{fname}: unclosed '{stack[-1][0]}' from line {stack[-1][1]}")
            ok = False

        # Check for key patterns
        has_register = "registerExtension" in content
        has_app_import = 'from "../../scripts/app.js"' in content or "from '../../scripts/app.js'" in content

        status = "OK" if ok else "BRACKET MISMATCH"
        print(f"    {fname}: {status}, registerExtension={has_register}, appImport={has_app_import}")

print()

# ── 7. WEB_DIRECTORY check ────────────────────────────────────
print("[7] Checking WEB_DIRECTORY configuration...")
try:
    WEB_DIRECTORY = "./js"
    js_path = os.path.join(os.getcwd(), WEB_DIRECTORY.lstrip("./"))
    exists = os.path.isdir(js_path)
    js_files = os.listdir(js_path) if exists else []
    print(f"    WEB_DIRECTORY = '{WEB_DIRECTORY}' -> {js_path}")
    print(f"    Directory exists: {exists}")
    print(f"    JS files: {js_files}")
    if not exists:
        errors.append(f"WEB_DIRECTORY '{WEB_DIRECTORY}' does not exist")
except Exception as e:
    errors.append(f"WEB_DIRECTORY check: {e}")
    traceback.print_exc()

print()

# ── 8. Torch availability check ──────────────────────────────
print("[8] Torch environment check...")
try:
    import torch
    print(f"    PyTorch version: {torch.__version__}")
    print(f"    CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"    CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"    Torch dtype float32: OK")
except Exception as e:
    errors.append(f"Torch check: {e}")
    traceback.print_exc()

print()

# ── Summary ───────────────────────────────────────────────────
print("=" * 60)
if errors:
    print(f"  ERRORS: {len(errors)}")
    for e in errors:
        print(f"    !! {e}")
else:
    print("  NO ERRORS")

if warnings:
    print(f"  WARNINGS: {len(warnings)}")
    for w in warnings:
        print(f"    ?? {w}")

if not errors:
    print("  ALL DIAGNOSTICS PASSED")
else:
    print(f"  {len(errors)} ISSUE(S) FOUND")
print("=" * 60)
