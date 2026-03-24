"""Robot profile extraction — normalizes RobotSpec into preset-ready format.

Converts the full MuJoCo-extracted RobotSpec into a lightweight RobotProfile
suitable for basis storage and preset reasoning.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from cadenza_local.robot_setup import RobotSpec, RobotHints
from cadenza_local.presets.schemas import RobotProfile


def extract_robot_profile(
    spec: RobotSpec,
    hints: Optional[RobotHints] = None,
) -> RobotProfile:
    """Convert a RobotSpec into a RobotProfile for basis storage.

    Args:
        spec: Full MuJoCo robot specification.
        hints: Robot-specific hints (actuator type, segment map, etc.).

    Returns:
        Normalized RobotProfile.
    """
    joint_limits_deg = {}
    max_torques = {}
    gripper_force = 0.0

    for jname in spec.joint_names:
        jnt = spec.joints.get(jname)
        act = spec.actuators.get(jname)
        if jnt:
            joint_limits_deg[jname] = jnt.range_deg
        if act:
            max_torques[jname] = act.max_torque
            # Heuristic: last actuator is often the gripper
            if "grip" in jname.lower() or "finger" in jname.lower():
                gripper_force = act.max_torque

    # Estimate workspace reach from body chain lengths
    workspace_reach = _estimate_reach(spec)

    actuator_type = "position"
    segment_map = {}
    if hints:
        actuator_type = hints.actuator_type
        segment_map = hints.segment_map or {}

    return RobotProfile(
        name=spec.name,
        n_actuators=spec.n_actuators,
        total_mass_kg=spec.total_mass,
        joint_names=list(spec.joint_names),
        joint_limits_deg=joint_limits_deg,
        max_torques_nm=max_torques,
        actuator_type=actuator_type,
        segment_map=segment_map,
        workspace_reach_m=workspace_reach,
        gripper_max_force_n=gripper_force,
    )


def profile_to_basis_records(
    profile: RobotProfile,
    user_id: str,
    preset_id: str,
) -> list[dict]:
    """Convert a RobotProfile into basis records for batch storage.

    Returns a list of records ready for BasisMemory.add_batch().
    """
    records = []

    # Overall capability summary
    records.append({
        "user_id": user_id,
        "category": "robot_capability",
        "content": (
            f"Robot {profile.name} has {profile.n_actuators} actuated joints "
            f"({profile.actuator_type} control), total mass {profile.total_mass_kg:.2f} kg, "
            f"estimated reach {profile.workspace_reach_m:.3f} m. "
            f"Joints: {', '.join(profile.joint_names)}."
        ),
        "data": {
            "name": profile.name,
            "n_actuators": profile.n_actuators,
            "total_mass_kg": profile.total_mass_kg,
            "actuator_type": profile.actuator_type,
            "joint_names": profile.joint_names,
            "workspace_reach_m": profile.workspace_reach_m,
            "gripper_max_force_n": profile.gripper_max_force_n,
        },
        "source": "robot_spec",
        "confidence": 1.0,
        "preset_id": preset_id,
    })

    # Per-joint capability records
    for jname in profile.joint_names:
        limits = profile.joint_limits_deg.get(jname, (-180, 180))
        torque = profile.max_torques_nm.get(jname, 0.0)
        records.append({
            "user_id": user_id,
            "category": "robot_capability",
            "content": (
                f"Joint {jname}: range [{limits[0]:+.0f}, {limits[1]:+.0f}] degrees, "
                f"max torque {torque:.1f} Nm."
            ),
            "data": {
                "joint_name": jname,
                "range_deg": list(limits),
                "max_torque_nm": torque,
            },
            "source": "robot_spec",
            "confidence": 1.0,
            "preset_id": preset_id,
        })

    # Segment mapping (if available)
    if profile.segment_map:
        for segment, mappings in profile.segment_map.items():
            joint_str = ", ".join(f"{jn} (scale={s})" for jn, s in mappings)
            records.append({
                "user_id": user_id,
                "category": "robot_capability",
                "content": (
                    f"Skeleton segment '{segment}' maps to robot joints: {joint_str}."
                ),
                "data": {
                    "segment": segment,
                    "joint_mappings": [
                        {"joint": jn, "scale": s} for jn, s in mappings
                    ],
                },
                "source": "robot_spec",
                "confidence": 1.0,
                "preset_id": preset_id,
            })

    return records


def _estimate_reach(spec: RobotSpec) -> float:
    """Estimate workspace reach from body positions.

    Sums the distances between consecutive bodies in the kinematic chain.
    Rough but useful for spatial reasoning.
    """
    if not spec.joint_names:
        return 0.0

    body_names = []
    for jname in spec.joint_names:
        jnt = spec.joints.get(jname)
        if jnt and jnt.body_name not in body_names:
            body_names.append(jnt.body_name)

    # Without actual body positions (need MuJoCo data), use a heuristic
    # based on number of joints and typical arm segment lengths
    n_joints = len(spec.joint_names)
    # Typical SO-101 segment: ~0.08m per link
    return n_joints * 0.08
