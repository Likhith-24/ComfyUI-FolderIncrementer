import os
import re
from datetime import datetime


def _get_output_dir():
    """Return ComfyUI output directory, with fallback for standalone use."""
    try:
        import folder_paths
        return folder_paths.get_output_directory()
    except Exception:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def _scan_next_version(scan_dir, prefix, padding):
    """
    Scan *scan_dir* for existing sub-directories that match the version
    pattern (e.g. v001, v002 …) and return the **next** version number.
    If no version folders exist yet → returns 1.
    Purely filesystem-based: cancelling a run cannot "waste" a number.
    """
    if not os.path.isdir(scan_dir):
        return 1
    pattern = re.compile(rf"^{re.escape(prefix)}(\d{{{padding},}})$")
    max_ver = 0
    for entry in os.listdir(scan_dir):
        if os.path.isdir(os.path.join(scan_dir, entry)):
            m = pattern.match(entry)
            if m:
                max_ver = max(max_ver, int(m.group(1)))
    return max_ver + 1


class FolderIncrementer:
    """
    Automatic dynamic file output management node.

    How it works
    ────────────
    1. Reads the input filename from whatever is connected (image, video,
       or any file type) via the JS companion that auto-fills
       ``source_filename``.
    2. Creates folder structure:
       ``output/{base_name}/{MM-DD-YYYY}/v###/{original_filename}``
    3. Version scanning happens inside the **date folder**, so each day
       starts fresh at v001.
    4. The version folder is created on execution to "claim" the number.
       Cancelled / stopped runs that never reach this node do NOT waste
       a version number.

    Example (source_filename = ``SC_30_SHT50.mp4``, today = 02-22-2026)
    ────────────────────────────────────────────────────────────────────
    output/
      SC_30_SHT50/
        02-22-2026/
          v001/
            SC_30_SHT50.mp4
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prefix": ("STRING", {"default": "v",
                    "tooltip": "Prefix before the version number (e.g. 'v' → v001)"}),
                "padding": ("INT", {"default": 3, "min": 1, "max": 10,
                    "tooltip": "Zero-pad width (3 → 001)"}),
                "label": ("STRING", {"default": "default",
                    "tooltip": "Fallback folder name (used only when no source file is connected)"}),
            },
            "optional": {
                "trigger": ("*", {
                    "tooltip": "Connect any output here – the node reads the connected filename automatically"}),
                "source_filename": ("STRING", {"default": "",
                    "tooltip": "Auto-filled by JS from the connected node. "
                               "Drives folder name + output filename."}),
                "base_path": ("STRING", {"default": "",
                    "tooltip": "Override base output directory.  Leave empty → ComfyUI output dir."}),
            },
        }

    RETURN_TYPES  = ("STRING", "INT", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES  = ("version_string", "version_number", "folder_name",
                     "subfolder_path", "filename_prefix", "output_filename")
    FUNCTION = "increment"
    CATEGORY = "utils"
    OUTPUT_NODE = True   # ensure the node always executes

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Re-execute every queue so the filesystem scan is up-to-date.
        return float("NaN")

    def increment(self, prefix="v", padding=3, label="default",
                  trigger=None, source_filename="", base_path=""):

        # ── 1. Derive names from source file ──────────────────────────
        if source_filename and source_filename.strip():
            basename = os.path.basename(source_filename.strip())
            name_no_ext, ext = os.path.splitext(basename)  # "SC_30_SHT50", ".mp4"
            folder_name = name_no_ext
        else:
            name_no_ext = ""
            ext = ""
            folder_name = label

        # ── 2. Resolve base directory ─────────────────────────────────
        base_dir = (base_path.strip()
                    if base_path and base_path.strip()
                    else _get_output_dir())

        # ── 3. Build date folder (MM-DD-YYYY) ─────────────────────────
        today_date = datetime.now().strftime("%m-%d-%Y")

        # ── 4. Scan for next version INSIDE the date folder ───────────
        #    Structure: base_dir / folder_name / today_date / v###
        date_dir = os.path.join(base_dir, folder_name, today_date)
        version_num = _scan_next_version(date_dir, prefix, padding)
        version_string = f"{prefix}{str(version_num).zfill(padding)}"

        # ── 5. Create the version folder to claim the number ──────────
        version_dir = os.path.join(date_dir, version_string)
        os.makedirs(version_dir, exist_ok=True)

        # ── 6. Build output paths ─────────────────────────────────────
        # subfolder_path : "SC_30_SHT50/02-22-2026/v001"
        subfolder_path = f"{folder_name}/{today_date}/{version_string}"

        # filename_prefix: "SC_30_SHT50/02-22-2026/v001/SC_30_SHT50"
        if name_no_ext:
            filename_prefix = f"{subfolder_path}/{name_no_ext}"
        else:
            filename_prefix = f"{subfolder_path}/{version_string}"

        # output_filename: "SC_30_SHT50/02-22-2026/v001/SC_30_SHT50.mp4"
        if name_no_ext and ext:
            output_filename = f"{subfolder_path}/{name_no_ext}{ext}"
        else:
            output_filename = filename_prefix

        return (version_string, version_num, folder_name,
                subfolder_path, filename_prefix, output_filename)


class FolderIncrementerReset:
    """
    Report the current version state for a folder (today's date).

    Scans ``<output>/<label>/<MM-DD-YYYY>/`` for version folders and
    reports how many exist.  To truly "reset", delete the version
    directories from disk.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "label": ("STRING", {"default": "default",
                    "tooltip": "Folder name to inspect"}),
            },
            "optional": {
                "trigger": ("*", {}),
                "base_path": ("STRING", {"default": "",
                    "tooltip": "Override base directory.  Leave empty → ComfyUI output dir."}),
            },
        }

    RETURN_TYPES  = ("STRING", "INT")
    RETURN_NAMES  = ("status", "current_version")
    FUNCTION = "check"
    CATEGORY = "utils"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def check(self, label="default", trigger=None, base_path=""):
        base_dir = base_path.strip() if base_path and base_path.strip() else _get_output_dir()
        today_date = datetime.now().strftime("%m-%d-%Y")
        scan_dir = os.path.join(base_dir, label, today_date)
        next_ver = _scan_next_version(scan_dir, "v", 3)
        current  = next_ver - 1
        if current < 1:
            return (f"'{label}/{today_date}': no versions yet – next will be v001", 0)
        return (f"'{label}/{today_date}': {current} version(s) exist – next will be v{str(next_ver).zfill(3)}", current)


class FolderIncrementerSet:
    """
    Reserve version slots by creating empty directories (inside today's
    date folder).

    Creates ``<output>/<label>/<MM-DD-YYYY>/v001`` … ``v{value}`` so that
    the next FolderIncrementer run will output v{value+1}.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "label": ("STRING", {"default": "default",
                    "tooltip": "Folder name under the output directory"}),
                "value": ("INT", {"default": 1, "min": 1, "max": 999999,
                    "tooltip": "Create placeholder dirs up to this version number"}),
            },
            "optional": {
                "trigger": ("*", {}),
                "prefix": ("STRING", {"default": "v"}),
                "padding": ("INT", {"default": 3, "min": 1, "max": 10}),
                "base_path": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES  = ("STRING", "INT")
    RETURN_NAMES  = ("status", "next_version")
    FUNCTION = "set_version"
    CATEGORY = "utils"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def set_version(self, label="default", value=1, trigger=None,
                    prefix="v", padding=3, base_path=""):
        base_dir = base_path.strip() if base_path and base_path.strip() else _get_output_dir()
        today_date = datetime.now().strftime("%m-%d-%Y")
        folder = os.path.join(base_dir, label, today_date)
        for i in range(1, value + 1):
            ver_dir = os.path.join(folder, f"{prefix}{str(i).zfill(padding)}")
            os.makedirs(ver_dir, exist_ok=True)
        next_ver = value + 1
        return (f"Reserved v001–v{str(value).zfill(padding)} for '{label}/{today_date}'. "
                f"Next = v{str(next_ver).zfill(padding)}",
                next_ver)


# ----- Registration maps consumed by __init__.py -----
NODE_CLASS_MAPPINGS = {
    "FolderIncrementer": FolderIncrementer,
    "FolderIncrementerReset": FolderIncrementerReset,
    "FolderIncrementerSet": FolderIncrementerSet,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FolderIncrementer": "Folder Version Incrementer",
    "FolderIncrementerReset": "Folder Version Check",
    "FolderIncrementerSet": "Folder Version Set",
}
