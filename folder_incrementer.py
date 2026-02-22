import os
import json

# Persistent counter file stored alongside this node
COUNTER_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "counters.json")


def _load_counters():
    if os.path.exists(COUNTER_FILE):
        with open(COUNTER_FILE, "r") as f:
            return json.load(f)
    return {}


def _save_counters(counters):
    with open(COUNTER_FILE, "w") as f:
        json.dump(counters, f, indent=2)


class FolderIncrementer:
    """
    Outputs an auto-incrementing version string (v001, v002, …).
    The counter is tracked per-label and persists across ComfyUI restarts.
    Connect any output to the 'trigger' input – its value is ignored;
    the connection simply ensures this node runs as part of the graph.

    If source_filename is provided (e.g. from a Load Image / Load Video node),
    the folder name is auto-derived from the input filename, version
    subdirectories are created (v001, v002, …), and the output filename
    matches the input filename.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prefix": ("STRING", {"default": "v", "tooltip": "Prefix before the number (e.g. 'v' → v001)"}),
                "padding": ("INT", {"default": 3, "min": 1, "max": 10, "tooltip": "Zero-pad width (3 → 001)"}),
                "label": ("STRING", {"default": "default", "tooltip": "Counter label (ignored when source_filename is provided)"}),
            },
            "optional": {
                "trigger": ("*", {"tooltip": "Connect any output here to ensure this node is part of the execution graph"}),
                "source_filename": ("STRING", {"default": "", "tooltip": "File path/name from Load Image or Load Video. Auto-derives folder name and output filename."}),
            },
        }

    RETURN_TYPES = ("STRING", "INT", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("version_string", "version_number", "folder_name", "subfolder_path", "filename_prefix")
    FUNCTION = "increment"
    CATEGORY = "utils"
    OUTPUT_NODE = False

    # This makes ComfyUI re-execute the node every run (no caching).
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def increment(self, prefix="v", padding=3, label="default", trigger=None, source_filename=""):
        # Auto-derive folder name from source filename if provided
        if source_filename and source_filename.strip():
            basename = os.path.basename(source_filename.strip())
            name_no_ext = os.path.splitext(basename)[0]
            folder_name = name_no_ext
            counter_label = name_no_ext   # Use filename as unique counter key
            output_filename = name_no_ext  # Output filename matches input name
        else:
            folder_name = label
            counter_label = label
            output_filename = ""

        counters = _load_counters()

        # Increment (or initialise) the counter for this label
        current = counters.get(counter_label, 0) + 1
        counters[counter_label] = current
        _save_counters(counters)

        version_string = f"{prefix}{str(current).zfill(padding)}"

        # Build paths
        # subfolder_path: "folder_name/v001" – can be used as output subdirectory
        subfolder_path = f"{folder_name}/{version_string}" if folder_name else version_string

        # filename_prefix: "folder_name/v001/filename" – ready for ComfyUI save nodes
        if output_filename:
            filename_prefix = f"{subfolder_path}/{output_filename}"
        else:
            filename_prefix = f"{subfolder_path}/{version_string}"

        return (version_string, current, folder_name, subfolder_path, filename_prefix)


class FolderIncrementerReset:
    """Reset a counter back to 0 (next run will produce 1 again)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "label": ("STRING", {"default": "default", "tooltip": "Label of the counter to reset"}),
            },
            "optional": {
                "trigger": ("*", {}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "reset"
    CATEGORY = "utils"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def reset(self, label="default", trigger=None):
        counters = _load_counters()
        counters[label] = 0
        _save_counters(counters)
        return (f"Counter '{label}' reset to 0",)


class FolderIncrementerSet:
    """Manually set a counter to a specific value."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "label": ("STRING", {"default": "default", "tooltip": "Label of the counter to set"}),
                "value": ("INT", {"default": 1, "min": 0, "max": 999999, "tooltip": "Set the counter to this value (next output will be this number)"}),
            },
            "optional": {
                "trigger": ("*", {}),
            },
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("version_string", "version_number")
    FUNCTION = "set_counter"
    CATEGORY = "utils"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def set_counter(self, label="default", value=1, trigger=None):
        counters = _load_counters()
        counters[label] = value
        _save_counters(counters)
        version_string = f"v{str(value).zfill(3)}"
        return (version_string, value)


# ----- Registration maps consumed by __init__.py -----
NODE_CLASS_MAPPINGS = {
    "FolderIncrementer": FolderIncrementer,
    "FolderIncrementerReset": FolderIncrementerReset,
    "FolderIncrementerSet": FolderIncrementerSet,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FolderIncrementer": "Folder Version Incrementer",
    "FolderIncrementerReset": "Folder Version Reset",
    "FolderIncrementerSet": "Folder Version Set",
}
