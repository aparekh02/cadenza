"""World-model adapters.

Each adapter wraps a specific world model (pi_0.5, OpenVLA, Octo, GR00T, ...)
and conforms to the WorldModelAdapter ABC. The detector picks the right
adapter by inspecting what model is present at the project root or in the
HuggingFace cache.
"""

from cadenza.stack.adapters.base import (
    WorldModelAdapter,
    AdapterReply,
    ProposedAction,
    register_adapter,
    get_adapter,
    list_adapters,
)
from cadenza.stack.adapters.mock import MockAdapter
from cadenza.stack.adapters.pi_zero import PiZeroAdapter
from cadenza.stack.adapters.openvla import OpenVLAAdapter
from cadenza.stack.adapters.smolvla import SmolVLAAdapter

__all__ = [
    "WorldModelAdapter",
    "AdapterReply",
    "ProposedAction",
    "register_adapter",
    "get_adapter",
    "list_adapters",
    "MockAdapter",
    "PiZeroAdapter",
    "OpenVLAAdapter",
    "SmolVLAAdapter",
]
