"""cadenza.robots — Canonical primitive libraries for each robot platform.

This package is the single source of truth that the Cadenza OS integration
layer wraps.  Every robot has its own subpackage exporting:

    actions   — ActionLibrary instance (from cadenza.actions)
    model     — Path to the MuJoCo scene.xml
    spec      — RobotSpec (kinematics, limits, gaits)
    controller — High-level controller class (Go1 / G1)
    deploy    — On-robot deployment backend (ROS2 node, configs)

Examples and tests import the same objects through ``cadenza.go1`` /
``cadenza.g1`` as before — those modules now delegate here.
"""

from cadenza.robots.go1 import Go1Robot
from cadenza.robots.g1 import G1Robot

_ROBOTS = {
    "go1": Go1Robot,
    "go2": Go1Robot,   # Go2 shares Go1 structure with adjusted gains
    "g1":  G1Robot,
}


def get_robot(name: str):
    """Return the robot descriptor for a given platform name."""
    if name not in _ROBOTS:
        raise ValueError(f"Unknown robot '{name}'. Available: {list(_ROBOTS.keys())}")
    return _ROBOTS[name](name)
