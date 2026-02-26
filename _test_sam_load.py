"""Test actual SAM model loading (not just path resolution)."""
import sys
import os
import logging

logging.basicConfig(level=logging.DEBUG)

sys.path.insert(0, r"D:\PROJECT\ComfyUI_windows_portable\ComfyUI")
import folder_paths
sys.path.insert(0, r"D:\PROJECT\Custom_Nodes\ComfyUI-CustomNodePacks")

from nodes.sam_model_loader import SAMModelLoaderMEC

loader = SAMModelLoaderMEC()
print("Loading SAM2.1 tiny fp16 safetensors...")
result = loader.load("sam2.1_hiera_tiny-fp16.safetensors", "auto", "cuda", False, "float16")
info = result[0]
print(f"Model type: {info['model_type']}")
print(f"Load method: {info['load_method']}")
print(f"Model class: {type(info['model']).__name__}")
print(f"Device: {info['device']}")

model = info["model"]
if hasattr(model, "image_encoder"):
    print(f"Has image_encoder: True ({type(model.image_encoder).__name__})")
else:
    print("Has image_encoder: False")

# Test creating a predictor
import torch
import numpy as np
from nodes.utils import get_sam_predictor

# Create a dummy image
img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
try:
    predictor = get_sam_predictor(model, info["model_type"], img)
    if predictor is not None:
        print(f"Predictor created: {type(predictor).__name__}")
        print("SUCCESS: Full SAM pipeline works!")
    else:
        print("FAIL: get_sam_predictor returned None")
        # Debug: try importing SAM2ImagePredictor directly
        try:
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            print(f"  SAM2ImagePredictor imported OK")
            pred = SAM2ImagePredictor(model)
            print(f"  Predictor created: {type(pred).__name__}")
            pred.set_image(img)
            print(f"  set_image worked!")
        except Exception as e2:
            print(f"  Direct SAM2ImagePredictor failed: {e2}")
except Exception as e:
    print(f"FAIL: {e}")
    import traceback
    traceback.print_exc()
