"""cadenza.robots.go1.primitives — Enumerated primitive table for Go1.

This module provides the flat primitive table that the Cadenza OS integration
layer (cadenza/integration/cadenza_scheduler.cpp) scans at compile time.

Each primitive is a dict with:
    name        — action name (matches ActionLibrary key)
    task_class  — DSK task class (CONTROL for all motion primitives)
    deadline_us — DSK deadline in microseconds (20ms for CONTROL)
    core        — core affinity (ISOLATED_1 for CONTROL)
    n_joints    — number of joints this primitive commands
    type        — "phase" (multi-phase trajectory) or "gait" (gait-engine)

This is also the authoritative list used by ``cadenza rl history``,
``cadenza kernel``, and the package registry to know what the robot can do.
"""

from cadenza.actions import get_library


def get_primitive_table(variant: str = "go1") -> list[dict]:
    """Return the flat primitive table for OS integration."""
    lib = get_library(variant)
    table = []
    for name in lib.list_actions():
        spec = lib.get(name)
        table.append({
            "name": name,
            "task_class": "CONTROL",
            "deadline_us": 20_000,
            "period_us": 5_000,
            "core": "ISOLATED_1",
            "priority": 80,
            "n_joints": 12,
            "type": "gait" if spec.gait is not None else "phase",
            "robot": variant,
        })
    return table


def get_joint_names() -> list[str]:
    """Return ordered joint names matching the ActionSpec joint ordering."""
    return [
        "FL_hip", "FL_thigh", "FL_calf",
        "FR_hip", "FR_thigh", "FR_calf",
        "RL_hip", "RL_thigh", "RL_calf",
        "RR_hip", "RR_thigh", "RR_calf",
    ]


def get_sensor_layout() -> dict:
    """Return sensor layout for RLAK delta_model input vector.

    The 22-float input vector is: joints[12] + imu[6] + terrain[4]
    """
    return {
        "joints": {"offset": 0, "size": 12, "names": get_joint_names()},
        "imu": {
            "offset": 12, "size": 6,
            "names": ["accel_x", "accel_y", "accel_z",
                       "gyro_x", "gyro_y", "gyro_z"],
        },
        "terrain": {
            "offset": 18, "size": 4,
            "names": ["slope", "roughness", "friction", "height_var"],
        },
    }
