# MaskEditControl – Implementation Plan (v2)

## Architecture

### model_manager.py (NEW)
- [x] `_MODEL_CACHE` dict keyed by `(model_type, variant, precision)`
- [x] `MODEL_REGISTRY` with all SAM2/2.1/SAM3/SeC/VideoMaMa/ViTMatte/MatAnyone2
- [x] `get_model_path()` – resolve local or trigger download
- [x] `ensure_downloaded()` – HuggingFace Hub download
- [x] `get_or_load_model()` – cache-aware model loading
- [x] `clear_cache()` – free VRAM/RAM
- [x] `precision_to_dtype()` – string → torch.dtype
- [x] `scan_model_dir()` – list available checkpoints

### UnifiedSegmentationNode (`nodes/unified_segmentation_node.py`) (NEW)
- [x] Dispatcher pattern: `segment()` → `_run_sam2()` | `_run_sam3()` | `_run_sec()` | `_run_videomama()`
- [x] `_parse_coords()` – JSON points → numpy arrays
- [x] `_parse_bboxes()` – JSON/BBOX → numpy (pos + neg)
- [x] `_validate_output()` – shape/clamp/batch normalization
- [x] SAM2/2.1 image segmentation via SAM2ImagePredictor
- [x] SAM2/2.1 video propagation via SAM2VideoPredictor (auto when B>1)
- [x] SAM2.0 vs 2.1 bbox-in-video version check
- [x] SAM3 image + video support (SAM2 architecture compatible)
- [x] SeC stub with install instructions
- [x] VideoMaMa stub with install instructions
- [x] Autocast context for fp16/bf16 inference

### MattingNode (`nodes/matting_node.py`) (REWRITTEN)
- [x] RETURN_TYPES = `("IMAGE", "MASK")`, RETURN_NAMES = `("rgb", "alpha_mask")`
- [x] `_generate_trimap()` via dilate XOR erode (unknown = dilated & ~eroded)
- [x] `_validate_alpha()` – shape/clamp/batch normalization
- [x] VitMatte backend via `model_manager.get_or_load_model()`
- [x] MatAnyone2 backend with proper warmup protocol (`first_frame_pred=True`)
- [x] Auto mode (VitMatte for image, MatAnyone2 for video)
- [x] RGB output: premultiplied (image × alpha)

### PointsMaskEditor (`nodes/points_mask_editor.py`) (REFACTORED)
- [x] `_parse_editor_data()` – JSON → separated pos/neg point/bbox lists
- [x] `_render_point_brush()` – Gaussian/hard circle rendering
- [x] `_render_bbox_region()` – positive fill / negative clear
- [x] `_encode_reference_image()` – base64 JPEG for frontend
- [x] Explicit named output variables

### Registration (`__init__.py`) (UPDATED)
- [x] Import `UnifiedSegmentationNode` from `unified_segmentation_node`
- [x] Import `model_manager` module
- [x] Print startup/loaded messages
- [x] Updated mappings and display names

### Tests (`tasks/test_nodes.py`) (NEW)
- [x] Test 1: PointsMaskEditor output types
- [x] Test 2: `_parse_editor_data`
- [x] Test 3: `_render_point_brush`
- [x] Test 4: `_render_bbox_region`
- [x] Test 5: UnifiedSegmentationNode.INPUT_TYPES structure
- [x] Test 6: `_parse_coords` / `_parse_bboxes`
- [x] Test 7: `_validate_output` shape and clamping
- [x] Test 8: MattingNode output types
- [x] Test 9: `_generate_trimap` values (0/128/255)
- [x] Test 10: `_validate_alpha` shape and clamping

## Existing Node Issues
- [ ] FolderIncrementer premature dir creation (remove os.makedirs from node)
- [x] PointsMaskEditor canvas shaking (fixed: unified height callbacks)
- [x] PointsMaskEditor video frames (fixed: bg_image base64 return)
- [x] PointsMaskEditor canvas resize on image change (fixed)

## Model Registry Reference

| Key | Family | HF Repo | Filename | Dir |
|-----|--------|---------|----------|-----|
| sam2_hiera_tiny | sam2 | Kijai/sam2-safetensors | sam2_hiera_tiny.safetensors | sam2 |
| sam2_hiera_small | sam2 | Kijai/sam2-safetensors | sam2_hiera_small.safetensors | sam2 |
| sam2_hiera_base_plus | sam2 | Kijai/sam2-safetensors | sam2_hiera_base_plus.safetensors | sam2 |
| sam2_hiera_large | sam2 | Kijai/sam2-safetensors | sam2_hiera_large.safetensors | sam2 |
| sam2.1_hiera_tiny | sam2 | Kijai/sam2-safetensors | sam2.1_hiera_tiny.safetensors | sam2 |
| sam2.1_hiera_small | sam2 | Kijai/sam2-safetensors | sam2.1_hiera_small.safetensors | sam2 |
| sam2.1_hiera_base_plus | sam2 | Kijai/sam2-safetensors | sam2.1_hiera_base_plus.safetensors | sam2 |
| sam2.1_hiera_large | sam2 | Kijai/sam2-safetensors | sam2.1_hiera_large.safetensors | sam2 |
| sam3 | sam3 | apozz/sam3-safetensors | sam3.safetensors | sam3 |
| sec_4b | sec | OpenIXCLab/SeC-4B | — (sharded dir) | sams |
| videomama | videomama | SammyLim/VideoMaMa | — (dir model) | VideoMaMa |
| vitmatte_small | vitmatte | hustvl/vitmatte-small-distinctions-646 | — (HF Transformers) | vitmatte |
| vitmatte_base | vitmatte | hustvl/vitmatte-base-distinctions-646 | — (HF Transformers) | vitmatte |
| matanyone2 | matanyone2 | pq-yang/MatAnyone2 | matanyone2.pth | matanyone2 |

## Status
- [x] Phase 1: Read codebase + all 7 third-party references
- [x] Phase 2: Create model_manager.py (shared cache/download)
- [x] Phase 3: Create unified_segmentation_node.py (dispatcher pattern)
- [x] Phase 4: Rewrite matting_node.py (correct return types + trimap)
- [x] Phase 5: Refactor points_mask_editor.py (private helpers)
- [x] Phase 6: Create test_nodes.py (10 unit tests)
- [x] Phase 7: Update __init__.py (new imports + print)
- [x] Phase 8: Update todo.md
- [ ] Phase 9: Run verification
