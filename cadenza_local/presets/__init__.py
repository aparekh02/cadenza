"""Presets module — extract foundational knowledge from demonstrations.

Takes motion images, robot specifications, objects, and task text as input.
Analyzes everything with a lightweight vision model and returns structured
knowledge as a BasisPreset.

Usage:
    from cadenza_local.presets import PresetBuilder

    builder = PresetBuilder(user_id="my-agent")
    builder.add_motion_images("path/to/images/")
    builder.set_robot_spec(spec, hints)
    builder.add_object("bottle", shape="cylinder", mass=0.8,
                        interactions=["grasping", "pouring"])
    builder.set_task(description="Make a cocktail", actions="pick up bottle, pour")

    preset = await builder.build()
    builder.export_annotated("output/annotated/")
    builder.export_summary("output/preset_output.txt")
"""

from cadenza_local.presets.builder import PresetBuilder
from cadenza_local.presets.schemas import (
    BasisPreset,
    MotionBlueprint,
    RobotProfile,
    ObjectProfile,
    TaskDirective,
    SpatialRelation,
    FrameAnalysis,
    LimbKeypoint,
    LimbSegment,
    ObjectDetection,
    SegmentKinematics,
    JointTorqueEstimate,
    COMTrajectory,
    BalanceProfile,
    PhaseForceProfile,
    DynamicsProfile,
    WaypointPrediction,
    SpeedRecommendation,
    SpatialSnapshot,
    MotorProfile,
    Microstep,
    MicrostepSequence,
)
from cadenza_local.presets.motion_intake import (
    analyze_motion_images,
    annotate_images,
)
from cadenza_local.presets.robot_profile import extract_robot_profile
from cadenza_local.presets.analyzer import (
    analyze_task_text,
    analyze_spatial_relations,
)
from cadenza_local.presets.dynamics import compute_dynamics, dynamics_to_basis_records
from cadenza_local.presets.motor_profile import compute_motor_profile, motor_profile_to_basis_records
from cadenza_local.presets.microsteps import generate_microsteps, microsteps_to_basis_records

__all__ = [
    # Builder
    "PresetBuilder",
    # Schemas
    "BasisPreset",
    "MotionBlueprint",
    "RobotProfile",
    "ObjectProfile",
    "TaskDirective",
    "SpatialRelation",
    "FrameAnalysis",
    "LimbKeypoint",
    "LimbSegment",
    "ObjectDetection",
    # Dynamics schemas
    "SegmentKinematics",
    "JointTorqueEstimate",
    "COMTrajectory",
    "BalanceProfile",
    "PhaseForceProfile",
    "DynamicsProfile",
    # Motor profile schemas
    "WaypointPrediction",
    "SpeedRecommendation",
    "SpatialSnapshot",
    "MotorProfile",
    # Microstep schemas
    "Microstep",
    "MicrostepSequence",
    # Functions
    "analyze_motion_images",
    "annotate_images",
    "extract_robot_profile",
    "analyze_task_text",
    "analyze_spatial_relations",
    "compute_dynamics",
    "dynamics_to_basis_records",
    "compute_motor_profile",
    "motor_profile_to_basis_records",
    "generate_microsteps",
    "microsteps_to_basis_records",
]
