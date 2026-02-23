# Architecture Review & Correction Plan

## System Understanding

### Architecture
- **Python backend**: `folder_incrementer.py` (3 nodes) + `nodes/*.py` (14 nodes)  
- **JS frontend**: `js/folder_incrementer.js` (filename auto-extraction) + `js/points_bbox_editor.js` (canvas editor)  
- **Registration**: `__init__.py` merges all NODE_CLASS_MAPPINGS  
- **Deployment**: Dev copy at `D:\PROJECT\Custom_Nodes\` synced to `D:\PROJECT\ComfyUI_windows_portable\ComfyUI\custom_nodes\`

### Data Flow
1. User uploads file (image/video) in ComfyUI → Load node has widget with filename
2. JS `folder_incrementer.js` traverses graph upstream from trigger input → finds filename widget → writes to `source_filename` widget
3. Python `FolderIncrementer.increment()` reads `source_filename` → derives folder_name, builds date folder, scans versions, creates version dir, returns paths
4. Downstream Save nodes use `subfolder_path`/`filename_prefix` for output

### PointsMaskEditor Flow
1. User interacts with JS canvas (points_bbox_editor.js) → stores JSON in `editor_data` widget
2. Python `PointsMaskEditor.generate()` parses JSON → outputs mask, coords, bboxes for SAM2

---

## Issues Found

### CRITICAL: FolderIncrementer creates folder on SCAN (not on save)
- **Location**: `folder_incrementer.py` line 131-132
- **Problem**: `os.makedirs(version_dir, exist_ok=True)` runs every time the node executes, even if no downstream save happens. If the workflow has this node connected but the save node fails or is disconnected, empty version folders accumulate.
- **Also**: Because `IS_CHANGED` returns NaN, the node re-executes every queue, creating a new version folder each time — even if nothing downstream uses it.
- **Impact**: Version numbers keep incrementing with empty folders.
- **Correct behavior**: The node should ONLY scan and report paths. Folder creation is the responsibility of the Save node (which ComfyUI's Save Image node does automatically when given a subfolder path).

### CRITICAL: PointsMaskEditor outputs crash Sam2Segmentation
- **Location**: `points_mask_editor.py` lines 192-200
- **Problem**: When no bboxes are drawn, output is `None`. When no points are drawn, output is `None`. But Sam2Segmentation's `bboxes` input type is `BBOX` with `forceInput: True` implicit — if the output is connected, ComfyUI will still pass the value. The real crash happens because:
  1. Empty list `[]` passes `if bboxes is not None:` check → loop body never sets `boxes_np` → UnboundLocalError
  2. We already fixed this by outputting `None` instead of `[]` — VERIFIED this is current state.
  3. BUT: When bboxes ARE provided, Sam2Segmentation iterates `for bbox_list in bboxes` where it expects bboxes to be a list of lists (batch of per-image bbox lists). Our output is `[[x1,y1,x2,y2]]` — a list containing one list of 4 ints. Sam2Seg iterates the outer list getting `[x1,y1,x2,y2]`, then iterates `for bbox in bbox_list` getting individual ints. `np.array([x1, y1, x2, y2])` produces shape `(4,)` which works for the non-individual_objects path.
  4. Actually this works correctly for the non-batch case. The issue the user saw was the empty-list case, which is now fixed.

### MODERATE: FolderIncrementer premature directory creation
- **Problem**: Creating the version directory during node execution means every queue creates a folder even if the job fails downstream.
- **Fix**: Remove `os.makedirs()` from the node. Let the Save node handle directory creation. ComfyUI's Save Image already creates `subfolder_path` automatically.

### MINOR: Coordinates output None breaks wire connections  
- **Location**: `points_mask_editor.py` line 191
- **Problem**: Outputting `None` for `positive_coords` when no points exist works for the `if coordinates_positive is not None:` check in Sam2Seg, BUT if the wire is connected, ComfyUI may raise an error because the output type is declared as STRING but actual value is None.
- **Fix**: Output empty string `""` instead of `None` — Sam2Seg parses it with json.loads which will fail, caught by `except: pass`, and `coordinates_positive` remains the original string `""` which is truthy... Actually no. We need to trace this more carefully.

### MINOR: Missing `render` reference in JS
- The `render` function in `points_bbox_editor.js` is defined locally and referenced by the width/height callbacks but may not be accessible due to scoping (it's defined after those callbacks are set up). This could cause a silent failure.

---

## Correction Strategy

### Fix 1: Remove premature directory creation from FolderIncrementer
- Remove `os.makedirs(version_dir, exist_ok=True)` 
- The node outputs paths only; folder creation happens at the Save node level
- This makes versioning deterministic and non-destructive

### Fix 2: Fix PointsMaskEditor coordinate output format  
- When no points: output `None` for coords (current) — this is correct because Sam2Seg checks `if coordinates_positive is not None:`
- When no bboxes: output `None` for bboxes (current) — this is correct because Sam2Seg checks `if bboxes is not None:`
- These outputs are optional connections; if not connected, Sam2Seg defaults to None anyway

### Fix 3: Verify JS scoping for render function
- Check that the `render` function is accessible in widget callbacks

---

## Status
- [x] Phase 1: Read entire codebase  
- [x] Phase 2: Requirement matching  
- [x] Phase 3: Issue detection  
- [ ] Phase 4: Root cause analysis (detailed)  
- [ ] Phase 5: Implement fixes  
- [ ] Phase 6: Verification  
