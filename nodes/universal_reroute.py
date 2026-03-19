"""
UniversalRerouteMEC — Dynamic reroute / Dot node.

A virtual-style pass-through that accepts ANY data type.
The JS companion (js/universal_reroute.js) renders a compact circle,
auto-adapts slot types on connection, handles bundle-drop, and strips
itself from the backend prompt when running as a virtual relay.

Key design:
  - VALIDATE_INPUTS always returns True → bypasses ComfyUI type checks
  - passthrough() accepts anything=None → never errors on missing input
  - JS sets isVirtualNode=True and strips from prompt before execution
"""

from __future__ import annotations

_AnyType = type("AnyType", (str,), {"__ne__": lambda self, other: False})
ANY = _AnyType("*")


class UniversalRerouteMEC:
    """Dynamic reroute dot — accepts and forwards any connection type."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"anything": (ANY, {})}}

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    RETURN_TYPES = (ANY,)
    RETURN_NAMES = ("output",)
    FUNCTION = "passthrough"
    CATEGORY = "MaskEditControl/Utils"
    DESCRIPTION = (
        "Drop onto any connection to reroute it. Compact dot shape. "
        "Auto-adapts to IMAGE, LATENT, MASK, STRING, etc. "
        "Right-click → 'Remove Reroute (reconnect)' to dissolve."
    )

    def passthrough(self, anything=None):
        return (anything,)
