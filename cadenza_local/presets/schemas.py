"""Data models for the presets module.

These represent the structured knowledge extracted from motion images,
robot specifications, objects, and task text — before storage into the
mem_alpha basis tier.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LimbKeypoint:
    """A single detected limb joint in an image frame."""
    name: str                           # "shoulder", "elbow", "wrist", etc.
    x: float                            # normalized image x (0.0-1.0)
    y: float                            # normalized image y (0.0-1.0)
    confidence: float = 0.8


@dataclass
class LimbSegment:
    """A limb segment between two keypoints."""
    name: str                           # "upper_arm", "forearm", "hand", etc.
    angle_deg: float                    # angle in degrees
    start_joint: str                    # starting keypoint name
    end_joint: str                      # ending keypoint name


@dataclass
class FrameAnalysis:
    """Complete analysis of a single motion image frame."""
    frame_number: int
    location: str                       # camera angle
    keypoints: list[LimbKeypoint] = field(default_factory=list)
    segments: list[LimbSegment] = field(default_factory=list)
    objects_detected: list[ObjectDetection] = field(default_factory=list)
    interactions: list[str] = field(default_factory=list)
    phase_label: str = ""               # e.g., "reaching", "grasping", "pouring"
    description: str = ""               # text description of what's happening


@dataclass
class ObjectDetection:
    """A detected object in a frame."""
    name: str
    shape: str                          # "cylinder", "box", "sphere"
    center_x: float                     # normalized (0.0-1.0)
    center_y: float
    width_frac: float                   # fraction of image width
    height_frac: float                  # fraction of image height
    interaction: str = "none"           # "grasping", "reaching", "holding", etc.


@dataclass
class MotionBlueprint:
    """Complete motion analysis extracted from an image sequence."""
    frames: list[FrameAnalysis] = field(default_factory=list)
    total_frames: int = 0
    angle_deltas: dict[str, list[float]] = field(default_factory=dict)
    key_moments: list[dict] = field(default_factory=list)
    task_phases: list[str] = field(default_factory=list)
    duration_estimate_sec: float = 0.0

    def summary(self) -> str:
        lines = [
            f"Motion Blueprint: {self.total_frames} frames",
            f"Phases: {' -> '.join(self.task_phases)}",
            f"Key moments: {len(self.key_moments)}",
        ]
        for seg, deltas in self.angle_deltas.items():
            total = sum(deltas)
            lines.append(f"  {seg}: total delta={total:+.1f} deg")
        return "\n".join(lines)


@dataclass
class RobotProfile:
    """Normalized robot capabilities for basis storage."""
    name: str
    n_actuators: int
    total_mass_kg: float
    joint_names: list[str] = field(default_factory=list)
    joint_limits_deg: dict[str, tuple[float, float]] = field(default_factory=dict)
    max_torques_nm: dict[str, float] = field(default_factory=dict)
    actuator_type: str = "position"     # "position" or "torque"
    segment_map: dict[str, list[tuple[str, float]]] = field(default_factory=dict)
    workspace_reach_m: float = 0.0
    gripper_max_force_n: float = 0.0

    def summary(self) -> str:
        lines = [
            f"Robot: {self.name}",
            f"Actuators: {self.n_actuators}, Mass: {self.total_mass_kg:.2f} kg",
            f"Type: {self.actuator_type}",
            f"Joints: {', '.join(self.joint_names)}",
        ]
        return "\n".join(lines)


@dataclass
class ObjectProfile:
    """An object the robot interacts with."""
    name: str
    shape: str
    estimated_size: tuple[float, ...] = (0.05, 0.15)
    estimated_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    estimated_mass_kg: float = 0.1
    interaction_types: list[str] = field(default_factory=list)
    properties: dict[str, str] = field(default_factory=dict)


@dataclass
class TaskDirective:
    """Structured task definition from text input."""
    description: str
    action_sequence: list[str] = field(default_factory=list)
    required_objects: list[str] = field(default_factory=list)
    required_interactions: list[str] = field(default_factory=list)
    speed_preference: float = 0.3       # 0.1=slow, 1.0=fast
    behavioral_notes: str = ""


@dataclass
class SpatialRelation:
    """Spatial relationship between objects and the robot."""
    object_a: str
    object_b: str
    relation: str                       # "left_of", "right_of", "in_front", "behind", "on_top", "near"
    estimated_distance_m: float = 0.0


# ── Dynamics schemas ──

@dataclass
class SegmentKinematics:
    """Angular kinematics for a single segment across all frames."""
    segment_name: str
    angles_deg: list[float] = field(default_factory=list)           # per-frame angle
    angular_velocity_dps: list[float] = field(default_factory=list) # deg/s per transition
    angular_accel_dps2: list[float] = field(default_factory=list)   # deg/s² per transition
    peak_velocity_dps: float = 0.0
    peak_accel_dps2: float = 0.0


@dataclass
class JointTorqueEstimate:
    """Estimated torque for a single joint across all frames."""
    joint_name: str
    torques_nm: list[float] = field(default_factory=list)   # per-frame torque
    peak_torque_nm: float = 0.0
    mean_torque_nm: float = 0.0
    capacity_fraction: float = 0.0      # peak / max_torque (0.0-1.0)
    peak_phase: str = ""                # phase where peak occurs


@dataclass
class COMTrajectory:
    """Center of mass trajectory across frames."""
    positions_xy: list[tuple[float, float]] = field(default_factory=list)
    velocities_xy: list[tuple[float, float]] = field(default_factory=list)
    base_offsets: list[float] = field(default_factory=list)  # distance from base per frame
    max_offset: float = 0.0


@dataclass
class BalanceProfile:
    """Per-frame stability scores mirroring ForceVectorSpace.balance_urgency."""
    scores: list[float] = field(default_factory=list)       # 0.0=stable, 1.0=unstable
    critical_frames: list[int] = field(default_factory=list) # frames where score > 0.6
    mean_score: float = 0.0
    peak_score: float = 0.0


@dataclass
class PhaseForceProfile:
    """Force/torque profile for a single task phase."""
    phase_name: str
    frame_start: int = 0
    frame_end: int = 0
    dominant_segments: list[str] = field(default_factory=list)
    peak_torque_nm: float = 0.0
    mean_torque_nm: float = 0.0
    payload_name: str = ""
    payload_mass_kg: float = 0.0
    energy_joules: float = 0.0
    effort_level: str = "low"           # "low", "medium", "high"


@dataclass
class DynamicsProfile:
    """Complete dynamics analysis for the motion sequence."""
    kinematics: list[SegmentKinematics] = field(default_factory=list)
    torques: list[JointTorqueEstimate] = field(default_factory=list)
    com: Optional[COMTrajectory] = None
    balance: Optional[BalanceProfile] = None
    phase_forces: list[PhaseForceProfile] = field(default_factory=list)
    total_energy_joules: float = 0.0
    peak_torque_nm: float = 0.0
    hardest_phase: str = ""
    n_segments: int = 0
    n_frames: int = 0

    def summary(self) -> str:
        lines = ["=== Dynamics Profile ==="]
        lines.append(
            f"Peak torque: {self.peak_torque_nm:.2f} Nm, "
            f"total energy: {self.total_energy_joules:.2f} J"
        )
        lines.append(
            f"Hardest phase: '{self.hardest_phase}'. "
            f"{self.n_segments} segments across {self.n_frames} frames."
        )
        if self.kinematics:
            lines.append("\nSegment kinematics:")
            for sk in self.kinematics:
                lines.append(
                    f"  {sk.segment_name}: peak vel={sk.peak_velocity_dps:.1f} deg/s, "
                    f"peak accel={sk.peak_accel_dps2:.1f} deg/s²"
                )
        if self.torques:
            lines.append("\nJoint torques:")
            for jt in self.torques:
                cap_str = f"{jt.capacity_fraction*100:.0f}%" if jt.capacity_fraction > 0 else "N/A"
                lines.append(
                    f"  {jt.joint_name}: peak={jt.peak_torque_nm:.2f} Nm ({cap_str} capacity), "
                    f"mean={jt.mean_torque_nm:.2f} Nm"
                )
        if self.balance:
            lines.append(
                f"\nBalance: mean={self.balance.mean_score:.2f}, "
                f"peak={self.balance.peak_score:.2f}, "
                f"critical frames={self.balance.critical_frames}"
            )
        if self.com:
            lines.append(f"\nCOM max offset from base: {self.com.max_offset:.3f}")
        if self.phase_forces:
            lines.append("\nPhase force profiles:")
            for pf in self.phase_forces:
                payload = f", payload: {pf.payload_name} {pf.payload_mass_kg:.1f}kg" if pf.payload_name else ""
                lines.append(
                    f"  '{pf.phase_name}' (frames {pf.frame_start}-{pf.frame_end}): "
                    f"{pf.effort_level} effort. Peak τ={pf.peak_torque_nm:.2f} Nm, "
                    f"energy={pf.energy_joules:.2f} J{payload}"
                )
        return "\n".join(lines)


# ── Motor profile schemas ──

@dataclass
class WaypointPrediction:
    """Predicted motor state at a single waypoint/frame.

    Maps skeleton angles through the robot's segment_map to produce
    predicted joint angles in the robot's actuator space — the same
    format that MotionMapper produces at runtime and that SmolVLA outputs.
    """
    frame_index: int
    phase: str = ""
    joint_angles_rad: dict[str, float] = field(default_factory=dict)  # joint → predicted angle
    joint_velocities_rads: dict[str, float] = field(default_factory=dict)  # joint → velocity
    gravity_compensation: dict[str, float] = field(default_factory=dict)  # joint → holding torque as ctrl fraction
    gripper_state: float = 0.0          # 0.0=open, 1.0=closed
    payload_mass_kg: float = 0.0
    payload_name: str = ""
    ctrl_array: list[float] = field(default_factory=list)  # full ctrl in actuator order


@dataclass
class SpeedRecommendation:
    """Speed guidance for a task phase.

    Derived from torque headroom, payload weight, phase type, and
    the user's speed preference. Feeds into waypoint duration and
    velocity scaling at runtime.
    """
    phase: str
    velocity_factor: float = 0.3        # 0.0-1.0 fraction of max speed
    max_joint_velocity_rads: dict[str, float] = field(default_factory=dict)
    recommended_duration_sec: float = 1.0
    torque_headroom: float = 1.0        # 1.0 = full headroom, 0.0 = at limit
    reason: str = ""                    # human-readable explanation


@dataclass
class SpatialSnapshot:
    """Joint + object positions at a single frame.

    Mirrors what ForceVectorSpace.object_distances provides at runtime,
    but predicted from the skeleton pose + known object positions. This
    is the spatial context the VLA model needs to understand the scene.
    """
    frame_index: int
    joint_positions: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    object_positions: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    joint_to_object_distances: dict[str, dict[str, float]] = field(default_factory=dict)
    gripper_to_target_distance: float = float('inf')
    nearest_object: str = ""


@dataclass
class MotorProfile:
    """Complete motor command and spatial profile for the task.

    Bridges presets → runtime: predicted ctrl arrays, speed guidance,
    and spatial context in the format consumed by the cadenza controller
    layer (ForceStateEncoder) and VLA models (SmolVLA-450M).
    """
    waypoints: list[WaypointPrediction] = field(default_factory=list)
    speed_profile: list[SpeedRecommendation] = field(default_factory=list)
    spatial_snapshots: list[SpatialSnapshot] = field(default_factory=list)
    joint_names: list[str] = field(default_factory=list)    # ordered actuator names
    n_actuators: int = 0
    ctrl_range: list[tuple[float, float]] = field(default_factory=list)
    action_dimension: int = 0           # output dim for VLA (n_actuators + gripper)
    robot_name: str = ""
    actuator_type: str = "position"     # "position" or "torque"

    def summary(self) -> str:
        lines = ["=== Motor Profile ==="]
        lines.append(
            f"Robot: {self.robot_name}, {self.n_actuators} actuators ({self.actuator_type})"
        )
        lines.append(
            f"Action dimension: {self.action_dimension} "
            f"({self.n_actuators} joints + {'gripper' if self.action_dimension > self.n_actuators else 'no gripper'})"
        )
        lines.append(f"Joints: {', '.join(self.joint_names)}")
        if self.ctrl_range:
            lines.append(f"Ctrl range: {self.ctrl_range[0]} (first joint)")
        lines.append(f"Waypoints: {len(self.waypoints)}")

        if self.waypoints:
            lines.append("\nPredicted waypoints:")
            for wp in self.waypoints:
                angles_str = ", ".join(
                    f"{j}={a:+.3f}" for j, a in wp.joint_angles_rad.items()
                )
                grip = f", gripper={'closed' if wp.gripper_state > 0.5 else 'open'}"
                payload = f", holding {wp.payload_name}" if wp.payload_name else ""
                lines.append(
                    f"  F{wp.frame_index} [{wp.phase}]: {angles_str}{grip}{payload}"
                )

        if self.speed_profile:
            lines.append("\nSpeed profile:")
            for sp in self.speed_profile:
                lines.append(
                    f"  '{sp.phase}': {sp.velocity_factor:.0%} max speed, "
                    f"~{sp.recommended_duration_sec:.1f}s. {sp.reason}"
                )

        if self.spatial_snapshots:
            lines.append(f"\nSpatial snapshots: {len(self.spatial_snapshots)} frames")
            # Show first snapshot as example
            ss = self.spatial_snapshots[0]
            if ss.joint_positions:
                lines.append(f"  F{ss.frame_index} joints: {len(ss.joint_positions)} tracked")
            if ss.object_positions:
                for name, pos in ss.object_positions.items():
                    lines.append(f"  F{ss.frame_index} {name}: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
            if ss.nearest_object:
                lines.append(
                    f"  F{ss.frame_index} nearest to gripper: "
                    f"{ss.nearest_object} ({ss.gripper_to_target_distance:.3f}m)"
                )

        return "\n".join(lines)


# ── Force sandbox schemas ──

@dataclass
class JointForceCheck:
    """Per-joint force check at a single frame."""
    joint_name: str
    torque_required_nm: float = 0.0
    torque_capacity_nm: float = 0.0
    capacity_fraction: float = 0.0         # required / capacity (>1.0 = overloaded)
    gravity_component_nm: float = 0.0
    dynamic_component_nm: float = 0.0
    payload_component_nm: float = 0.0
    passed: bool = True
    margin: float = 1.0                    # 1.0 - capacity_fraction


@dataclass
class FrameForceCheck:
    """Complete force balance check for a single frame."""
    frame_index: int
    phase: str = ""
    joint_checks: list[JointForceCheck] = field(default_factory=list)
    balance_score: float = 0.0             # 0.0=stable, 1.0=tipping
    balance_passed: bool = True
    com_offset: float = 0.0
    speed_safe: bool = True
    max_velocity_dps: float = 0.0
    velocity_limit_dps: float = 0.0
    payload_mass_kg: float = 0.0
    payload_name: str = ""
    all_passed: bool = True
    failure_reasons: list[str] = field(default_factory=list)


@dataclass
class PhaseCheck:
    """Aggregate safety check for a task phase."""
    phase_name: str
    frame_start: int = 0
    frame_end: int = 0
    feasible: bool = True
    peak_capacity_fraction: float = 0.0
    critical_joint: str = ""
    transition_safe: bool = True
    transition_torque_jump_nm: float = 0.0
    payload_check_passed: bool = True
    n_failed_frames: int = 0
    n_warning_frames: int = 0
    failure_reasons: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


@dataclass
class SandboxResult:
    """Overall force sandbox test result — the go/no-go verdict."""
    verdict: str = "SAFE"                 # "SAFE", "WARNING", "UNSAFE"
    frame_checks: list[FrameForceCheck] = field(default_factory=list)
    phase_checks: list[PhaseCheck] = field(default_factory=list)
    n_frames_tested: int = 0
    n_frames_passed: int = 0
    n_frames_warning: int = 0
    n_frames_failed: int = 0
    peak_capacity_fraction: float = 0.0
    peak_capacity_joint: str = ""
    peak_capacity_frame: int = 0
    peak_balance_score: float = 0.0
    peak_balance_frame: int = 0
    recommendations: list[str] = field(default_factory=list)

    def summary_text(self) -> str:
        lines = [f"=== Force Sandbox: {self.verdict} ==="]
        lines.append(
            f"Tested {self.n_frames_tested} frames: "
            f"{self.n_frames_passed} passed, "
            f"{self.n_frames_warning} warnings, "
            f"{self.n_frames_failed} failed."
        )
        lines.append(
            f"Peak capacity: {self.peak_capacity_fraction:.0%} "
            f"at {self.peak_capacity_joint} (frame {self.peak_capacity_frame})."
        )
        lines.append(
            f"Peak balance score: {self.peak_balance_score:.2f} "
            f"(frame {self.peak_balance_frame})."
        )
        if self.phase_checks:
            lines.append("\nPhase verdicts:")
            for pc in self.phase_checks:
                status = "PASS" if pc.feasible else "FAIL"
                lines.append(
                    f"  '{pc.phase_name}': {status} "
                    f"(peak capacity {pc.peak_capacity_fraction:.0%})"
                )
                for r in pc.recommendations:
                    lines.append(f"    -> {r}")
        if self.recommendations:
            lines.append("\nRecommendations:")
            for r in self.recommendations:
                lines.append(f"  - {r}")
        return "\n".join(lines)


# ── Microstep schemas ──

@dataclass
class Microstep:
    """Single control-timestep guidance at 30Hz.

    An 18D feature vector that tells SmolVLA what to do at this instant:
    6D target joint deltas, 6D torque load fractions, and 6 scalar features
    (speed, gripper, phase progress, balance, payload, proximity).
    """
    timestep: int
    phase: str = ""
    target_delta_rad: list[float] = field(default_factory=list)   # 6D: per-joint delta
    torque_fraction: list[float] = field(default_factory=list)    # 6D: per-joint load (0-1)
    speed_factor: float = 0.3          # 0-1 velocity multiplier
    gripper_command: float = 0.0       # 0=open, 1=closed
    phase_progress: float = 0.0        # 0-1 within current phase
    balance_score: float = 0.0         # 0=stable, 1=unstable
    payload_mass_norm: float = 0.0     # payload_kg / 2.0 (normalized)
    gripper_proximity: float = 0.0     # 0-1 (1=touching target object)

    def to_vector(self) -> list[float]:
        """Return 18D feature vector for state injection."""
        return (
            list(self.target_delta_rad)
            + list(self.torque_fraction)
            + [self.speed_factor,
               self.gripper_command,
               self.phase_progress,
               self.balance_score,
               self.payload_mass_norm,
               self.gripper_proximity]
        )


@dataclass
class MicrostepSequence:
    """Dense microstep sequence for a full task."""
    steps: list[Microstep] = field(default_factory=list)
    hz: int = 30
    total_duration_sec: float = 0.0
    n_phases: int = 0
    phase_boundaries: list[tuple[int, int, str]] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"=== Microstep Sequence ==="]
        lines.append(f"{len(self.steps)} steps at {self.hz}Hz, {self.total_duration_sec:.1f}s total")
        lines.append(f"{self.n_phases} phases:")
        for start, end, name in self.phase_boundaries:
            lines.append(f"  '{name}': steps {start}-{end} ({end - start + 1} steps)")
        return "\n".join(lines)


@dataclass
class BasisPreset:
    """Complete preset — all extracted knowledge before basis storage."""
    preset_id: str
    motion: Optional[MotionBlueprint] = None
    robot: Optional[RobotProfile] = None
    objects: list[ObjectProfile] = field(default_factory=list)
    task: Optional[TaskDirective] = None
    spatial_relations: list[SpatialRelation] = field(default_factory=list)
    dynamics: Optional[DynamicsProfile] = None
    motor: Optional[MotorProfile] = None
    sandbox: Optional[SandboxResult] = None

    def summary(self) -> str:
        lines = [f"=== Basis Preset: {self.preset_id} ===\n"]
        if self.task:
            lines.append(f"Task: {self.task.description}")
            if self.task.action_sequence:
                lines.append(f"Actions: {' -> '.join(self.task.action_sequence)}")
            lines.append("")
        if self.robot:
            lines.append(self.robot.summary())
            lines.append("")
        if self.motion:
            lines.append(self.motion.summary())
            lines.append("")
        if self.objects:
            lines.append(f"Objects ({len(self.objects)}):")
            for obj in self.objects:
                lines.append(
                    f"  {obj.name} ({obj.shape}) — "
                    f"interactions: {', '.join(obj.interaction_types) if obj.interaction_types else 'none'}"
                )
            lines.append("")
        if self.spatial_relations:
            lines.append(f"Spatial Relations ({len(self.spatial_relations)}):")
            for rel in self.spatial_relations:
                lines.append(
                    f"  {rel.object_a} {rel.relation} {rel.object_b} "
                    f"(~{rel.estimated_distance_m:.2f}m)"
                )
            lines.append("")
        if self.dynamics:
            lines.append(self.dynamics.summary())
            lines.append("")
        if self.motor:
            lines.append(self.motor.summary())
            lines.append("")
        if self.sandbox:
            lines.append(self.sandbox.summary_text())
        return "\n".join(lines)
