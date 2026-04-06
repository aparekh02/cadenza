"""cadenza.locomotion — Robot specs, gait generation, and kinematics.

Core support for the action library: robot physical parameters,
analytical gait generation, and forward/inverse kinematics.
"""

from cadenza.locomotion.robot_spec import (
    GO1, GO2, get_spec, RobotSpec,
    GAITS, GaitParams,
    JointLimits, MotorLimits, Kinematics,
    SafetyThresholds, Benchmarks,
    TerrainCapability,
    JOINT_NAMES, LEG_INDICES, FOOT_ORDER,
)
from cadenza.locomotion.kinematics import (
    foot_position_body, ik_leg,
    nominal_foot_positions,
    clip_joints, check_joint_margins,
    joint_vector_to_legs, legs_to_joint_vector,
)
from cadenza.locomotion.gait_engine import GaitEngine

__all__ = [
    "GO1", "GO2", "get_spec", "RobotSpec",
    "GAITS", "GaitParams",
    "JointLimits", "MotorLimits", "Kinematics",
    "SafetyThresholds", "Benchmarks", "TerrainCapability",
    "JOINT_NAMES", "LEG_INDICES", "FOOT_ORDER",
    "foot_position_body", "ik_leg",
    "nominal_foot_positions", "clip_joints", "check_joint_margins",
    "joint_vector_to_legs", "legs_to_joint_vector",
    "GaitEngine",
]
