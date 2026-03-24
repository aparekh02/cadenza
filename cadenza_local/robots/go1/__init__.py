"""cadenza.robots.go1 — Go1 quadruped primitive library.

This is the package that the Cadenza OS integration layer treats as the
"existing primitive library" (CLAUDE.md: cadenza/robots/go1/).

It re-exports the action definitions, MuJoCo model, robot spec, controller,
and on-robot deployment backend so every consumer — examples, tests, gym3d,
and the C++ OS wrapper — accesses the same objects from one canonical location.
"""

from __future__ import annotations

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
_PKG_DIR = Path(__file__).resolve().parent
_CADENZA_DIR = _PKG_DIR.parent.parent          # cadenza/
_MODELS_DIR = _CADENZA_DIR / "models" / "go1"
_LIBRARY_DIR = _CADENZA_DIR / "library" / "go1"

# MuJoCo model
MODEL_XML = _MODELS_DIR / "scene.xml"

# On-robot deployment assets
DEPLOY_DIR = _LIBRARY_DIR
DEPLOY_ROS2_NODE = _LIBRARY_DIR / "ros2_node.py"
DEPLOY_CONTROLLER = _LIBRARY_DIR / "run_controller.py"
DEPLOY_CONFIG = _LIBRARY_DIR / "config" / "go1.yaml"
TERRAIN_XML = _LIBRARY_DIR / "terrain.xml"


# ── Lazy imports (avoid pulling mujoco/numpy at import time) ─────────────────

def get_actions():
    """Return the ActionLibrary for Go1."""
    from cadenza_local.actions import get_library
    return get_library("go1")


def get_spec():
    """Return the RobotSpec for Go1 (kinematics, limits, gaits)."""
    from cadenza_local.locomotion.robot_spec import get_spec as _get_spec
    return _get_spec("go1")


def get_controller(**kwargs):
    """Return a Go1 controller instance."""
    from cadenza_local.go1 import Go1
    return Go1(**kwargs)


def list_actions() -> list[str]:
    """List all available Go1 action names."""
    return get_actions().list_actions()


def get_action(name: str):
    """Get a specific ActionSpec by name."""
    return get_actions().get(name)


class Go1Robot:
    """Descriptor object used by cadenza.robots.get_robot()."""

    def __init__(self, variant: str = "go1"):
        self.name = variant
        self.model_xml = MODEL_XML
        self.deploy_dir = DEPLOY_DIR
        self.n_joints = 12
        self.dof = 12
        self.robot_type = "quadruped"

    @property
    def actions(self):
        return get_actions()

    @property
    def spec(self):
        return get_spec()

    def controller(self, **kwargs):
        return get_controller(**kwargs)

    def action_names(self) -> list[str]:
        return list_actions()

    def __repr__(self):
        return f"Go1Robot(name={self.name!r}, joints={self.n_joints}, actions={len(self.action_names())})"
