"""
UniversalRerouteMEC — Nuke-style "Dot" pass-through node.

Accepts any data type on input and passes it through unchanged.
The JS companion (js/universal_reroute.js) provides:
  - Compact dot rendering with colored ring matching the data type
  - Bundle-drop: insert into multiple wires at once
  - Right-click → "Remove Reroute (reconnect)" to dissolve
  - Double-click to toggle type label
"""

from __future__ import annotations


class _AnyType(str):
    """Wildcard type that matches any ComfyUI data type."""

    def __ne__(self, other):
        return False


_ANY = _AnyType("*")


class UniversalRerouteMEC:
    """Pass-through node that accepts and forwards any connection type."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "any": (_ANY, {}),
            },
        }

    RETURN_TYPES = (_ANY,)
    RETURN_NAMES = ("any",)
    FUNCTION = "passthrough"
    CATEGORY = "MaskEditControl/Utils"
    DESCRIPTION = (
        "Nuke-style Dot node. Drop onto any connection to reroute it. "
        "Right-click → 'Remove Reroute (reconnect)' to dissolve. "
        "Handles any data type (IMAGE, LATENT, MASK, STRING, etc.)."
    )

    def passthrough(self, any):
        return (any,)
