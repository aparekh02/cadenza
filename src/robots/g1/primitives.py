"""cadenza.robots.g1.primitives — Enumerated primitive table for G1.

Same structure as go1/primitives.py but for the G1 humanoid.
"""

from cadenza.actions import get_library


def get_primitive_table(variant: str = "g1") -> list[dict]:
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
            "n_joints": 29,       # full body
            "n_leg_joints": 12,   # legs only (what RLAK corrects)
            "type": "gait" if spec.gait is not None else "phase",
            "robot": variant,
        })
    return table


def get_joint_names() -> list[str]:
    """Return ordered leg joint names (what RLAK corrects)."""
    return [
        "left_hip_yaw", "left_hip_roll", "left_hip_pitch",
        "left_knee", "left_ankle_pitch", "left_ankle_roll",
        "right_hip_yaw", "right_hip_roll", "right_hip_pitch",
        "right_knee", "right_ankle_pitch", "right_ankle_roll",
    ]


def get_sensor_layout() -> dict:
    """Return sensor layout for RLAK delta_model input vector."""
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
