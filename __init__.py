"""
__init__.py - Package entrypoint for the Audio Amplifier nodes
"""

WEB_DIRECTORY = "./web"

# Each node module keeps its own NODE_CLASS_MAPPINGS (same pattern as
# nodes_v1.py); merged here into one pack-wide registration.
from .nodes_v1 import NODE_CLASS_MAPPINGS as _CLASSIC_MAPPINGS
from .nodes_v1 import NODE_DISPLAY_NAME_MAPPINGS as _CLASSIC_NAMES
from .nodes_retime import NODE_CLASS_MAPPINGS as _RETIME_MAPPINGS
from .nodes_retime import NODE_DISPLAY_NAME_MAPPINGS as _RETIME_NAMES
from .nodes_waveform_overlay import NODE_CLASS_MAPPINGS as _WAVEFORM_MAPPINGS
from .nodes_waveform_overlay import NODE_DISPLAY_NAME_MAPPINGS as _WAVEFORM_NAMES

NODE_CLASS_MAPPINGS = {**_CLASSIC_MAPPINGS, **_RETIME_MAPPINGS, **_WAVEFORM_MAPPINGS}
NODE_DISPLAY_NAME_MAPPINGS = {**_CLASSIC_NAMES, **_RETIME_NAMES, **_WAVEFORM_NAMES}

print(f"[AudioAmplifier] Registered via classic V1 node mappings "
      f"({len(NODE_CLASS_MAPPINGS)} nodes): {list(NODE_CLASS_MAPPINGS.keys())}")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
