"""Pytest configuration for MaskEditControl tests.

Handles sys.path setup and stubs ComfyUI-specific modules so node
modules can be imported outside of ComfyUI.
"""
import sys
import types

# Stub out ComfyUI-specific modules that nodes may import at module level
for mod_name in ("folder_paths", "comfy", "comfy.utils"):
    if mod_name not in sys.modules:
        stub = types.ModuleType(mod_name)
        stub.__path__ = []
        sys.modules[mod_name] = stub
