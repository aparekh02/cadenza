"""FK/IK for Unitree Go1/Go2. Body frame: x=forward, y=left, z=up. All angles in radians."""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cadenza_local.locomotion.robot_spec import RobotSpec, Kinematics


def _rotation_x(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _rotation_y(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


_HIP_SIGN = {"FL": +1, "FR": -1, "RL": +1, "RR": -1}
_HIP_LAT_SIGN = {"FL": +1, "FR": -1, "RL": +1, "RR": -1}
_HIP_LON_SIGN = {"FL": +1, "FR": +1, "RL": -1, "RR": -1}


def foot_position_body(
    leg: str,
    q: np.ndarray,
    kin: "Kinematics",
) -> np.ndarray:
    """FK: joint angles -> foot position in body frame."""
    q_hip, q_thigh, q_calf = float(q[0]), float(q[1]), float(q[2])
    hs = _HIP_SIGN[leg]
    llon = _HIP_LON_SIGN[leg]
    llat = _HIP_LAT_SIGN[leg]

    hip_pos = np.array([
        llon * kin.hip_offset_lon,
        llat * kin.hip_offset_lat,
        0.0,
    ], dtype=np.float64)

    hip_rot = _rotation_x(hs * q_hip)

    thigh_vec = np.array([0.0, hs * kin.hip_offset_lat, 0.0]) + \
                hip_rot @ np.array([0.0, 0.0, -kin.thigh_length])

    thigh_dir = hip_rot @ _rotation_y(-q_thigh) @ np.array([0.0, 0.0, -1.0])
    knee_pos  = hip_pos + hip_rot @ np.array([0.0, hs * kin.hip_offset_lat, 0.0]) + \
                thigh_dir * kin.thigh_length

    calf_dir = hip_rot @ _rotation_y(-(q_thigh + q_calf)) @ np.array([0.0, 0.0, -1.0])
    foot_pos = knee_pos + calf_dir * kin.calf_length

    return foot_pos.astype(np.float32)


def ik_leg(
    leg: str,
    foot_target_body: np.ndarray,
    kin: "Kinematics",
) -> np.ndarray | None:
    """Analytical 3-DOF IK for one leg. Returns (3,) [hip, thigh, calf] or None if unreachable."""
    hs   = _HIP_SIGN[leg]
    llat = _HIP_LAT_SIGN[leg]
    llon = _HIP_LON_SIGN[leg]

    hip_origin = np.array([
        llon * kin.hip_offset_lon,
        llat * kin.hip_offset_lat,
        0.0,
    ], dtype=np.float64)
    p = np.array(foot_target_body, dtype=np.float64) - hip_origin

    lateral_offset = hs * kin.hip_offset_lat
    d_yz = np.sqrt(p[1]**2 + p[2]**2)

    if d_yz < abs(lateral_offset):
        return None

    # Solve p[1]*cos(t) + p[2]*sin(t) = lateral_offset via R*cos(t-psi) form
    psi   = np.arctan2(p[2], p[1])
    beta  = np.arccos(np.clip(lateral_offset / d_yz, -1.0, 1.0))
    q_hip = (psi + beta) * hs

    R_hip_inv = _rotation_x(-hs * q_hip)
    p_hip = R_hip_inv @ p - np.array([0.0, lateral_offset, 0.0])

    # 2D IK in sagittal plane
    lx, lz = p_hip[0], p_hip[2]
    L = np.sqrt(lx**2 + lz**2)
    l1, l2 = kin.thigh_length, kin.calf_length

    if L > l1 + l2 or L < abs(l1 - l2):
        return None

    cos_knee = (L**2 - l1**2 - l2**2) / (2 * l1 * l2)
    cos_knee = np.clip(cos_knee, -1.0, 1.0)
    q_calf   = -np.arccos(cos_knee)

    alpha = np.arctan2(lx, -lz)
    delta = np.arccos(np.clip((L**2 + l1**2 - l2**2) / (2 * L * l1), -1.0, 1.0))
    q_thigh = alpha + delta

    return np.array([q_hip, q_thigh, q_calf], dtype=np.float32)


def nominal_foot_positions(
    kin: "Kinematics",
    body_height: float,
    stance_width_extra: float = 0.025,
) -> dict[str, np.ndarray]:
    """Default foot positions under the body at a given height."""
    positions = {}
    for leg in ("FL", "FR", "RL", "RR"):
        x = _HIP_LON_SIGN[leg] * kin.hip_offset_lon
        # 2 * hip_offset_lat matches FK with hip=0 (trunk->hip + hip->thigh)
        y = _HIP_LAT_SIGN[leg] * (2 * kin.hip_offset_lat + stance_width_extra)
        z = -body_height
        positions[leg] = np.array([x, y, z], dtype=np.float32)
    return positions


def joint_vector_to_legs(q12: np.ndarray) -> dict[str, np.ndarray]:
    """Split 12-dim joint vector into per-leg (3,) arrays."""
    return {
        "FL": q12[0:3],
        "FR": q12[3:6],
        "RL": q12[6:9],
        "RR": q12[9:12],
    }


def legs_to_joint_vector(legs: dict[str, np.ndarray]) -> np.ndarray:
    """Merge per-leg (3,) arrays into 12-dim joint vector."""
    return np.concatenate([legs["FL"], legs["FR"], legs["RL"], legs["RR"]])


def clip_joints(q12: np.ndarray, spec: "RobotSpec") -> np.ndarray:
    """Hard-clip all 12 joints to the robot's limits."""
    q = q12.copy().astype(np.float32)
    jl = spec.joints
    for i in range(12):
        jtype = i % 3
        if jtype == 0:
            q[i] = np.clip(q[i], jl.hip_min, jl.hip_max)
        elif jtype == 1:
            q[i] = np.clip(q[i], jl.thigh_min, jl.thigh_max)
        else:
            q[i] = np.clip(q[i], jl.knee_min, jl.knee_max)
    return q


def check_joint_margins(q12: np.ndarray, spec: "RobotSpec", margin: float | None = None) -> list[str]:
    """Return joint names dangerously close to their limits."""
    if margin is None:
        margin = spec.safety.joint_pos_margin
    jl = spec.joints
    warnings = []
    for i, name in enumerate([
        "FL_hip","FL_thigh","FL_calf",
        "FR_hip","FR_thigh","FR_calf",
        "RL_hip","RL_thigh","RL_calf",
        "RR_hip","RR_thigh","RR_calf",
    ]):
        jtype = i % 3
        v = float(q12[i])
        if jtype == 0:
            lo, hi = jl.hip_min, jl.hip_max
        elif jtype == 1:
            lo, hi = jl.thigh_min, jl.thigh_max
        else:
            lo, hi = jl.knee_min, jl.knee_max
        if v < lo + margin or v > hi - margin:
            warnings.append(name)
    return warnings
