"""cadenza.robots.g1 — G1 humanoid primitive library.

Re-exports action definitions, MuJoCo model, robot spec, controller,
and gait engine from one canonical location.
"""

from __future__ import annotations

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
_PKG_DIR = Path(__file__).resolve().parent
_CADENZA_DIR = _PKG_DIR.parent.parent          # cadenza/
_MODELS_DIR = _CADENZA_DIR / "models" / "g1"

# MuJoCo model
MODEL_XML = _MODELS_DIR / "scene.xml"

# Gait data (walk splines, stand pose, etc.)
GAIT_DATA_DIR = _MODELS_DIR


# ── Lazy imports ─────────────────────────────────────────────────────────────

def get_actions():
    """Return the ActionLibrary for G1."""
    from cadenza.actions import get_library
    return get_library("g1")


def get_spec():
    """Return the RobotSpec for G1 (kinematics, limits, gaits)."""
    from cadenza.locomotion.robot_spec import get_spec as _get_spec
    return _get_spec("g1")


def get_controller(**kwargs):
    """Return a G1 controller instance."""
    from cadenza.g1 import G1
    return G1(**kwargs)


def get_gait_engine():
    """Return the G1 gait execution module."""
    import cadenza.g1_gait as gait
    return gait


def list_actions() -> list[str]:
    """List all available G1 action names."""
    return get_actions().list_actions()


def get_action(name: str):
    """Get a specific ActionSpec by name."""
    return get_actions().get(name)


class G1Robot:
    """Descriptor object used by cadenza.robots.get_robot()."""

    def __init__(self, variant: str = "g1"):
        self.name = variant
        self.model_xml = MODEL_XML
        self.gait_data_dir = GAIT_DATA_DIR
        self.n_joints = 29       # full G1 body (legs + waist + arms + hands)
        self.n_leg_joints = 12   # leg DOF only
        self.n_arm_joints = 14   # arm DOF
        self.dof = 29
        self.robot_type = "humanoid"

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
        return f"G1Robot(name={self.name!r}, joints={self.n_joints}, actions={len(self.action_names())})"
