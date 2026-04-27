import os
import re
import sys
from datetime import datetime
from pathlib import Path


# Date format selector → strftime mapping
DATE_FORMAT_MAP = {
    "MM-DD-YYYY": "%m-%d-%Y",
    "DD-MM-YYYY": "%d-%m-%Y",
    "YYYY-MM-DD": "%Y-%m-%d",
}
DATE_FORMAT_CHOICES = list(DATE_FORMAT_MAP.keys())

# Path separator styles — different OSes expect different separators
# when paths are passed to external tools or displayed to users.
# ComfyUI internally handles "/" on all OSes, but users may need
# native separators for downstream scripts or other tools.
PATH_STYLE_CHOICES = ["auto", "windows", "linux", "macos"]


def _get_path_sep(style: str) -> str:
    """Return the path separator for the selected style."""
    if style == "windows":
        return "\\"
    elif style in ("linux", "macos"):
        return "/"
    else:  # auto — detect from current OS
        return os.sep


def _get_output_dir():
    """Return ComfyUI output directory, with fallback for standalone use."""
    try:
        import folder_paths
        return folder_paths.get_output_directory()
    except Exception:
        return str(Path(__file__).resolve().parent / "output")


def _get_current_os() -> str:
    """Return the detected OS name for display."""
    if sys.platform == "win32":
        return "Windows"
    elif sys.platform == "darwin":
        return "macOS"
    else:
        return "Linux"


# Common input file patterns that should not be used as output folder names.
# If source_filename looks like one of these, fall back to the label instead.
_INPUT_FILE_PATTERNS = re.compile(
    r"^(ComfyUI_temp_|input_|ref_|reference_)"
    r"|^\d{5,}_\.png$"  # ComfyUI temp uploads like 00001_.png
    r"|^clipspace/",
    re.IGNORECASE,
)


def _looks_like_input_file(filename: str) -> bool:
    """Return True if filename appears to be a ComfyUI input/temp file
    rather than a meaningful output name."""
    if not filename:
        return False
    return bool(_INPUT_FILE_PATTERNS.search(filename))


# Characters illegal on Windows file systems (also covers Linux/macOS reserved
# slash). NUL and other control chars are stripped too.
_ILLEGAL_FOLDER_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED_WINDOWS_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def _sanitize_folder_name(name: str, max_length: int = 100, fallback: str = "output") -> str:
    """Make *name* safe to use as a folder component on Windows/Linux/macOS.

    Removes illegal characters, strips leading/trailing dots and whitespace,
    rejects reserved Windows names (CON, PRN, ...) and clamps length.
    Returns *fallback* if the result is empty.
    """
    if not name:
        return fallback
    cleaned = _ILLEGAL_FOLDER_CHARS.sub("_", str(name))
    cleaned = cleaned.strip(" .")
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length].rstrip(" .")
    if not cleaned:
        return fallback
    if cleaned.upper() in _RESERVED_WINDOWS_NAMES:
        cleaned = f"_{cleaned}"
    return cleaned


def _scan_next_version(scan_dir, prefix, padding):
    """
    Scan *scan_dir* for existing sub-directories that match the version
    pattern (e.g. v001, v002 …) and return the **next** version number.
    If no version folders exist yet → returns 1.
    Purely filesystem-based: cancelling a run cannot "waste" a number.
    """
    scan_path = Path(scan_dir)
    if not scan_path.is_dir():
        return 1
    pattern = re.compile(rf"^{re.escape(prefix)}(\d{{{padding},}})$")
    max_ver = 0
    for entry in scan_path.iterdir():
        if entry.is_dir():
            m = pattern.match(entry.name)
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
                "date_format": (DATE_FORMAT_CHOICES, {
                    "default": "MM-DD-YYYY",
                    "tooltip": "Date format for the date subfolder (e.g. 02-22-2026 or 2026-02-22)",
                }),
                "path_style": (PATH_STYLE_CHOICES, {
                    "default": "auto",
                    "tooltip": "Path separator style for output strings. "
                               "auto=detect from current OS, windows=backslash, "
                               "linux/macos=forward slash. Use 'auto' unless you "
                               "design workflows on one OS and run on another.",
                }),
            },
            "optional": {
                "trigger": ("*", {
                    "tooltip": "Connect any output here – the node reads the connected filename automatically"}),
                "source_filename": ("STRING", {"default": "",
                    "tooltip": "Auto-filled by JS from the connected node. "
                               "Drives folder name + output filename."}),
                "base_path": ("STRING", {"default": "",
                    "tooltip": "Override base output directory.  Leave empty → ComfyUI output dir."}),
                "folder_name_override": ("STRING", {"default": "",
                    "tooltip": "Force a specific folder name instead of deriving from the input filename. "
                               "Sanitized for cross-platform safety."}),
                "reserve_version": ("BOOLEAN", {"default": False,
                    "tooltip": "If True, create the version directory and write a `.reserved` marker file "
                               "to claim the version number atomically. Prevents collisions in batch/render-farm "
                               "workflows. Leave False for normal use (the directory will be created by "
                               "ComfyUI's Save node when output is actually written)."}),
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
                  date_format="MM-DD-YYYY", path_style="auto",
                  trigger=None, source_filename="", base_path="",
                  folder_name_override="", reserve_version=False):

        sep = _get_path_sep(path_style)
        detected_os = _get_current_os()

        # ── 1. Derive names from source file ──────────────────────────
        if source_filename and source_filename.strip() and not _looks_like_input_file(source_filename.strip()):
            # Normalize Windows backslashes when running on POSIX (and vice-versa).
            raw = source_filename.strip().replace("\\", "/")
            basename = Path(raw).name
            name_no_ext = Path(basename).stem       # "SC_30_SHT50"
            ext = Path(basename).suffix             # ".mp4"
            derived_folder = name_no_ext
        else:
            name_no_ext = ""
            ext = ""
            derived_folder = label

        # Explicit override wins; otherwise sanitize the derived name.
        if folder_name_override and folder_name_override.strip():
            folder_name = _sanitize_folder_name(folder_name_override.strip(), fallback=label or "output")
        else:
            folder_name = _sanitize_folder_name(derived_folder, fallback=label or "output")

        # ── 2. Resolve base directory ─────────────────────────────────
        if base_path and base_path.strip():
            base_dir = Path(base_path.strip())
        else:
            base_dir = Path(_get_output_dir())

        # ── 3. Build date folder ─────────────────────────────────────
        fmt = DATE_FORMAT_MAP.get(date_format, "%m-%d-%Y")
        today_date = datetime.now().strftime(fmt)

        # ── 4. Scan for next version INSIDE the date folder ───────────
        #    Structure: base_dir / folder_name / today_date / v###
        date_dir = base_dir / folder_name / today_date
        version_num = _scan_next_version(date_dir, prefix, padding)
        version_string = f"{prefix}{str(version_num).zfill(padding)}"

        # ── 5. Optional atomic reservation ────────────────────────────
        #    Default behaviour: do NOT create the directory here.
        #    ComfyUI's get_save_image_path() will create it when the
        #    downstream Save node actually writes output. With
        #    reserve_version=True, we claim the version slot up-front
        #    by creating the directory and dropping a marker file —
        #    safe for parallel batch / render-farm workflows.
        if reserve_version:
            try:
                ver_dir = date_dir / version_string
                ver_dir.mkdir(parents=True, exist_ok=True)
                marker = ver_dir / ".reserved"
                if not marker.exists():
                    marker.write_text(
                        f"reserved={datetime.now().isoformat()}\n"
                        f"folder={folder_name}\n"
                        f"version={version_string}\n",
                        encoding="utf-8",
                    )
            except OSError as exc:
                # Reservation is best-effort; never crash the workflow.
                print(f"[MEC] FolderIncrementer: reserve_version failed: {exc}")

        # ── 6. Build output paths using chosen separator ──────────────
        subfolder_path = sep.join([folder_name, today_date, version_string])

        if name_no_ext:
            filename_prefix = sep.join([subfolder_path, name_no_ext])
        else:
            filename_prefix = sep.join([subfolder_path, version_string])

        if name_no_ext and ext:
            output_filename = sep.join([subfolder_path, f"{name_no_ext}{ext}"])
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
                "date_format": (DATE_FORMAT_CHOICES, {
                    "default": "MM-DD-YYYY",
                    "tooltip": "Date format (must match what FolderIncrementer uses)",
                }),
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

    def check(self, label="default", date_format="MM-DD-YYYY",
               trigger=None, base_path=""):
        base_dir = Path(base_path.strip()) if base_path and base_path.strip() else Path(_get_output_dir())
        fmt = DATE_FORMAT_MAP.get(date_format, "%m-%d-%Y")
        today_date = datetime.now().strftime(fmt)
        safe_label = _sanitize_folder_name(label, fallback="default")
        scan_dir = base_dir / safe_label / today_date
        next_ver = _scan_next_version(scan_dir, "v", 3)
        current  = next_ver - 1
        if current < 1:
            return (f"'{safe_label}/{today_date}': no versions yet – next will be v001", 0)
        return (f"'{safe_label}/{today_date}': {current} version(s) exist – next will be v{str(next_ver).zfill(3)}", current)


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
                "date_format": (DATE_FORMAT_CHOICES, {
                    "default": "MM-DD-YYYY",
                    "tooltip": "Date format (must match what FolderIncrementer uses)",
                }),
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
                    prefix="v", padding=3, base_path="",
                    date_format="MM-DD-YYYY"):
        base_dir = Path(base_path.strip()) if base_path and base_path.strip() else Path(_get_output_dir())
        fmt = DATE_FORMAT_MAP.get(date_format, "%m-%d-%Y")
        today_date = datetime.now().strftime(fmt)
        safe_label = _sanitize_folder_name(label, fallback="default")
        folder = base_dir / safe_label / today_date
        for i in range(1, value + 1):
            ver_dir = folder / f"{prefix}{str(i).zfill(padding)}"
            ver_dir.mkdir(parents=True, exist_ok=True)
        next_ver = value + 1
        return (f"Reserved v001–v{str(value).zfill(padding)} for '{safe_label}/{today_date}'. "
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
