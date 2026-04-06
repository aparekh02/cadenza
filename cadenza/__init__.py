import os as _os
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

"""Cadenza — Developer-first action library for Unitree robots.

Quick start::

    import cadenza

    # List available actions
    cadenza.list_actions("go1")

    # Simulate
    cadenza.run("walk forward 2 meters then turn left then jump")

    # Or use the robot API
    go1 = cadenza.go1()
    go1.run([go1.stand(), go1.walk_forward(), go1.jump()])

CLI::

    cadenza list go1
    cadenza sim go1 "walk forward then jump"
    cadenza deploy go1 --ip 192.168.123.15
"""

from cadenza.actions import (
    ActionSpec, ActionPhase, ActionLibrary, ActionCall,
    get_action, list_actions, get_library,
)
from cadenza.sim import Sim, run
from cadenza.go1 import Go1, Step
from cadenza.g1 import G1


def go1(**kwargs) -> Go1:
    """Create a Go1 robot controller.

    Usage::

        import cadenza
        go1 = cadenza.go1()
        go1.run([go1.jump(), go1.walk_forward(speed=1.5)])
    """
    return Go1(**kwargs)


def g1(**kwargs) -> G1:
    """Create a G1 humanoid controller.

    Usage::

        import cadenza
        g1 = cadenza.g1()
        g1.run([g1.stand(), g1.walk_forward(), g1.lift_left_hand()])
    """
    return G1(**kwargs)


# Lazy import for VLA (heavy dependencies)
def __getattr__(name: str):
    if name == "VLAGuardian":
        from cadenza.vla import VLAGuardian
        return VLAGuardian
    raise AttributeError(f"module 'cadenza' has no attribute {name!r}")
