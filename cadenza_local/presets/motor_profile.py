"""Motor profile computation — predicted actions, speed guidance, spatial context.

Bridges the gap between preset analysis (what the robot needs to do) and
runtime execution (what ctrl commands the model should produce). This is
the data that primes a VLA model like SmolVLA-450M before execution:

1. **Waypoint predictions** — predicted joint angles in the robot's
   actuator space at each frame, mapped through segment_map (same logic
   as MotionMapper). Includes gravity compensation estimates and gripper state.

2. **Speed profile** — per-phase velocity recommendations based on torque
   headroom, payload, phase type, and the user's speed preference.

3. **Spatial context** — per-frame joint positions + object positions +
   joint-to-object distances. Mirrors what ForceVectorSpace.object_distances
   provides at runtime, but predicted from presets.

The motor profile output matches what the cadenza controller layer
(ForceStateEncoder) and VLA models consume:
    - ctrl_array: [n_actuators] float values in actuator ctrl range
    - action_dimension: n_actuators + 1 (gripper) for VLA output head
    - spatial distances: per-joint × per-object distance matrix
"""

from __future__ import annotations

import math
from typing import Optional

from cadenza_local.presets.schemas import (
    MotionBlueprint,
    RobotProfile,
    ObjectProfile,
    FrameAnalysis,
    DynamicsProfile,
    PhaseForceProfile,
    WaypointPrediction,
    SpeedRecommendation,
    SpatialSnapshot,
    MotorProfile,
)


# ── Default segment → joint mapping (matches MotionMapper.DEFAULT_SEGMENT_MAP) ──

DEFAULT_SEGMENT_MAP: dict[str, list[tuple[str, float]]] = {
    "upper_arm": [("shoulder_lift", 1.0)],
    "forearm":   [("elbow_flex", 1.0)],
    "hand":      [("wrist_flex", 1.0)],
}

# Default joint ctrl ranges when no robot spec available.
DEFAULT_CTRL_RANGE = (-2.0, 2.0)

# Default max joint velocity (rad/s) — conservative for small arms.
DEFAULT_MAX_VELOCITY_RADS = 2.0

# Phase types that typically require a closed gripper.
GRIP_PHASES = {"grasping", "lifting", "holding", "pouring", "stirring", "carrying"}

# Phase types that are inherently slow (precision work).
SLOW_PHASES = {"pouring", "stirring", "placing", "inserting"}

# Phase types that can be faster (coarse positioning).
FAST_PHASES = {"reaching", "returning", "approaching", "retracting"}


def compute_motor_profile(
    blueprint: MotionBlueprint,
    frames: list[FrameAnalysis],
    dynamics: Optional[DynamicsProfile] = None,
    robot: Optional[RobotProfile] = None,
    objects: Optional[list[ObjectProfile]] = None,
    speed_preference: float = 0.3,
) -> MotorProfile:
    """Compute the full motor profile from presets data.

    Args:
        blueprint: Motion blueprint with angle deltas, phases, duration.
        frames: Per-frame analysis with segments, keypoints, phases.
        dynamics: Dynamics profile (torques, phase forces) if computed.
        robot: Robot profile for joint names, limits, segment_map.
        objects: Scene objects with positions and interaction types.
        speed_preference: User's speed setting (0.1=slow, 1.0=fast).

    Returns:
        MotorProfile with predicted waypoints, speed guidance, spatial context.
    """
    objects = objects or []

    # Resolve segment map and joint info from robot profile.
    segment_map = DEFAULT_SEGMENT_MAP
    joint_names: list[str] = []
    joint_limits_rad: dict[str, tuple[float, float]] = {}
    ctrl_ranges: list[tuple[float, float]] = []
    max_torques: dict[str, float] = {}
    robot_name = "unknown"
    actuator_type = "position"
    n_actuators = 0

    if robot:
        robot_name = robot.name
        actuator_type = robot.actuator_type
        joint_names = list(robot.joint_names)
        n_actuators = robot.n_actuators
        max_torques = dict(robot.max_torques_nm)
        if robot.segment_map:
            segment_map = robot.segment_map
        for jn in joint_names:
            lim_deg = robot.joint_limits_deg.get(jn, (-180.0, 180.0))
            joint_limits_rad[jn] = (math.radians(lim_deg[0]), math.radians(lim_deg[1]))
            ctrl_ranges.append(joint_limits_rad[jn])
    else:
        # Infer joint names from segment map.
        seen = set()
        for mappings in segment_map.values():
            for jn, _ in mappings:
                if jn not in seen:
                    joint_names.append(jn)
                    seen.add(jn)
        n_actuators = len(joint_names)
        for jn in joint_names:
            joint_limits_rad[jn] = (math.radians(-180), math.radians(180))
            ctrl_ranges.append(DEFAULT_CTRL_RANGE)

    # Gripper adds +1 to action dimension.
    has_gripper = any("gripper" in jn.lower() for jn in joint_names)
    action_dim = n_actuators + (0 if has_gripper else 1)

    # Build joint name → index for ctrl_array ordering.
    jname_to_idx = {jn: i for i, jn in enumerate(joint_names)}

    # Phase force lookup from dynamics.
    phase_force_map: dict[str, PhaseForceProfile] = {}
    if dynamics and dynamics.phase_forces:
        for pf in dynamics.phase_forces:
            phase_force_map[pf.phase_name] = pf

    # ── 1. Waypoint predictions ──
    waypoints = _compute_waypoints(
        frames, blueprint, segment_map, joint_names, jname_to_idx,
        joint_limits_rad, max_torques, objects,
    )

    # ── 2. Speed profile ──
    speed_recs = _compute_speed_profile(
        blueprint, frames, dynamics, robot, speed_preference,
        joint_names, max_torques,
    )

    # ── 3. Spatial context ──
    spatial = _compute_spatial_snapshots(
        frames, joint_names, robot, objects,
    )

    return MotorProfile(
        waypoints=waypoints,
        speed_profile=speed_recs,
        spatial_snapshots=spatial,
        joint_names=joint_names,
        n_actuators=n_actuators,
        ctrl_range=ctrl_ranges,
        action_dimension=action_dim,
        robot_name=robot_name,
        actuator_type=actuator_type,
    )


# ── Waypoint computation ──

def _compute_waypoints(
    frames: list[FrameAnalysis],
    blueprint: MotionBlueprint,
    segment_map: dict[str, list[tuple[str, float]]],
    joint_names: list[str],
    jname_to_idx: dict[str, int],
    joint_limits_rad: dict[str, tuple[float, float]],
    max_torques: dict[str, float],
    objects: list[ObjectProfile],
) -> list[WaypointPrediction]:
    """Map skeleton angles to predicted joint angles at each frame.

    Uses the same segment_map logic as MotionMapper.map_to_trajectory().
    """
    results: list[WaypointPrediction] = []
    n_joints = len(joint_names)

    # Build cumulative angles from deltas (same as MotionMapper).
    current_angles = [0.0] * n_joints

    for fi, frame in enumerate(frames):
        # Accumulate angle deltas from blueprint for this transition.
        if fi > 0:
            for segment, mappings in segment_map.items():
                deltas = blueprint.angle_deltas.get(segment, [])
                delta_idx = fi - 1
                if delta_idx < len(deltas):
                    delta_deg = deltas[delta_idx]
                    for jn, scale in mappings:
                        if jn in jname_to_idx:
                            idx = jname_to_idx[jn]
                            current_angles[idx] += math.radians(delta_deg) * scale

        # Clamp to joint limits.
        clamped = {}
        for jn in joint_names:
            idx = jname_to_idx[jn]
            lo, hi = joint_limits_rad.get(jn, (-math.pi, math.pi))
            clamped[jn] = max(lo, min(hi, current_angles[idx]))
            current_angles[idx] = clamped[jn]

        # Velocities: finite difference from previous frame.
        velocities = {}
        if fi > 0 and len(results) > 0:
            prev = results[-1].joint_angles_rad
            dt = blueprint.duration_estimate_sec / max(blueprint.total_frames, 1) if blueprint.duration_estimate_sec > 0 else 0.5
            for jn in joint_names:
                velocities[jn] = (clamped[jn] - prev.get(jn, 0.0)) / dt if dt > 0 else 0.0
        else:
            velocities = {jn: 0.0 for jn in joint_names}

        # Gravity compensation estimate: torque_needed / max_torque → ctrl fraction.
        grav_comp = {}
        for jn in joint_names:
            angle = clamped[jn]
            # Simple: gravity torque ~ cos(angle), normalized to ctrl range.
            max_t = max_torques.get(jn, 5.0)
            # Rough estimate: mass * g * length * cos(angle) for a mid-chain joint.
            grav_torque_est = 0.5 * 9.81 * 0.1 * math.cos(angle)
            grav_comp[jn] = grav_torque_est / max_t if max_t > 0 else 0.0

        # Gripper state from phase and interactions.
        gripper = 0.0
        phase = frame.phase_label.lower() if frame.phase_label else ""
        if phase in GRIP_PHASES:
            gripper = 1.0
        for inter in frame.interactions:
            if any(g in inter.lower() for g in GRIP_PHASES):
                gripper = 1.0
                break

        # Payload.
        payload_mass = 0.0
        payload_name = ""
        held_interactions = {"grasping", "lifting", "holding", "pouring", "stirring"}
        for inter in frame.interactions:
            inter_lower = inter.lower()
            for obj in objects:
                if obj.name.lower() in inter_lower:
                    for itype in obj.interaction_types:
                        if itype in held_interactions:
                            payload_mass = obj.estimated_mass_kg
                            payload_name = obj.name
                            break
                    if payload_name:
                        break
            if payload_name:
                break
        if not payload_name:
            for det in frame.objects_detected:
                if det.interaction in held_interactions:
                    for obj in objects:
                        if obj.name == det.name:
                            payload_mass = obj.estimated_mass_kg
                            payload_name = obj.name
                            break
                if payload_name:
                    break

        # Build ctrl_array: joint angles in actuator order + gripper.
        ctrl = [clamped[jn] for jn in joint_names]
        ctrl.append(gripper)

        results.append(WaypointPrediction(
            frame_index=fi,
            phase=frame.phase_label,
            joint_angles_rad=dict(clamped),
            joint_velocities_rads=velocities,
            gravity_compensation=grav_comp,
            gripper_state=gripper,
            payload_mass_kg=payload_mass,
            payload_name=payload_name,
            ctrl_array=ctrl,
        ))

    return results


# ── Speed profile ──

def _compute_speed_profile(
    blueprint: MotionBlueprint,
    frames: list[FrameAnalysis],
    dynamics: Optional[DynamicsProfile],
    robot: Optional[RobotProfile],
    speed_preference: float,
    joint_names: list[str],
    max_torques: dict[str, float],
) -> list[SpeedRecommendation]:
    """Compute per-phase speed recommendations.

    Speed is governed by:
    1. Torque headroom — less headroom means slower to stay safe.
    2. Payload — heavier objects need slower, more controlled movement.
    3. Phase type — pouring/stirring are inherently slow; reaching is fast.
    4. User preference — overall speed scaling factor.
    """
    if not blueprint.task_phases:
        return []

    # Map frames to phases.
    phase_frames: dict[str, list[int]] = {}
    for i, frame in enumerate(frames):
        phase = frame.phase_label or "unknown"
        phase_frames.setdefault(phase, []).append(i)

    for phase in blueprint.task_phases:
        if phase not in phase_frames:
            phase_frames[phase] = []

    # Phase force profiles from dynamics.
    phase_forces: dict[str, PhaseForceProfile] = {}
    if dynamics and dynamics.phase_forces:
        for pf in dynamics.phase_forces:
            phase_forces[pf.phase_name] = pf

    total_duration = blueprint.duration_estimate_sec or (len(frames) * 0.5)
    n_phases = max(len(phase_frames), 1)
    base_phase_duration = total_duration / n_phases

    results: list[SpeedRecommendation] = []

    for phase_name, frame_indices in phase_frames.items():
        reasons = []
        phase_lower = phase_name.lower()

        # Start with user preference as baseline.
        vel_factor = speed_preference

        # Phase type adjustment.
        if phase_lower in SLOW_PHASES:
            vel_factor *= 0.5
            reasons.append(f"precision phase ({phase_lower})")
        elif phase_lower in FAST_PHASES:
            vel_factor *= 1.5
            reasons.append(f"coarse positioning ({phase_lower})")

        # Torque headroom adjustment.
        pf = phase_forces.get(phase_name)
        headroom = 1.0
        if pf:
            # Find max capacity fraction across joints in this phase.
            if dynamics and dynamics.torques:
                capacities = []
                for t in dynamics.torques:
                    for fi in frame_indices:
                        if fi < len(t.torques_nm):
                            mt = max_torques.get(t.joint_name, 5.0)
                            cap = t.torques_nm[fi] / mt if mt > 0 else 0.0
                            capacities.append(cap)
                if capacities:
                    max_cap = max(capacities)
                    headroom = max(1.0 - max_cap, 0.05)
                    if max_cap > 0.7:
                        vel_factor *= 0.6
                        reasons.append(f"high torque load ({max_cap:.0%} capacity)")
                    elif max_cap > 0.4:
                        vel_factor *= 0.8
                        reasons.append(f"moderate torque ({max_cap:.0%} capacity)")

            # Payload adjustment.
            if pf.payload_mass_kg > 0.5:
                vel_factor *= 0.6
                reasons.append(f"heavy payload ({pf.payload_mass_kg:.1f}kg)")
            elif pf.payload_mass_kg > 0.1:
                vel_factor *= 0.8
                reasons.append(f"payload ({pf.payload_mass_kg:.1f}kg)")

        # Clamp velocity factor.
        vel_factor = max(0.05, min(1.0, vel_factor))

        # Recommended duration: inversely proportional to velocity factor.
        n_frames_in_phase = max(len(frame_indices), 1)
        # Base duration proportional to frame count, scaled by 1/vel_factor.
        duration = (n_frames_in_phase / max(len(frames), 1)) * total_duration / vel_factor

        # Per-joint max velocities.
        max_vels = {}
        for jn in joint_names:
            base_vel = DEFAULT_MAX_VELOCITY_RADS
            max_vels[jn] = base_vel * vel_factor

        reason_str = "; ".join(reasons) if reasons else "default speed"

        results.append(SpeedRecommendation(
            phase=phase_name,
            velocity_factor=vel_factor,
            max_joint_velocity_rads=max_vels,
            recommended_duration_sec=round(duration, 2),
            torque_headroom=headroom,
            reason=reason_str,
        ))

    return results


# ── Spatial context ──

def _compute_spatial_snapshots(
    frames: list[FrameAnalysis],
    joint_names: list[str],
    robot: Optional[RobotProfile],
    objects: list[ObjectProfile],
) -> list[SpatialSnapshot]:
    """Compute joint + object positions and distances per frame.

    Joint positions are estimated from keypoints projected into
    approximate 3D using segment lengths. Object positions come
    from ObjectProfile.estimated_position.
    """
    results: list[SpatialSnapshot] = []

    # Object positions are constant (from profiles).
    obj_positions: dict[str, tuple[float, float, float]] = {}
    for obj in objects:
        obj_positions[obj.name] = tuple(obj.estimated_position)

    # Segment lengths for 3D projection.
    segment_lengths: dict[str, float] = {}
    if robot and robot.segment_map:
        for seg_name, mappings in robot.segment_map.items():
            segment_lengths[seg_name] = sum(abs(c) for _, c in mappings)

    for fi, frame in enumerate(frames):
        # Estimate joint positions from keypoints.
        joint_pos: dict[str, tuple[float, float, float]] = {}

        if frame.keypoints:
            # Map keypoints to approximate 3D positions.
            # x,y from image coords (0-1), z estimated from segment chain.
            # Base is at origin; each subsequent joint adds segment length in z.
            workspace = robot.workspace_reach_m if robot else 0.3
            z_cursor = 0.0

            for kp in frame.keypoints:
                # Scale normalized image coords to approximate workspace coords.
                x_m = (kp.x - 0.5) * workspace * 2
                y_m = (0.9 - kp.y) * workspace * 2  # flip y: image top → robot forward
                z_m = z_cursor

                # Advance z for chain joints.
                for seg_name, seg_len in segment_lengths.items():
                    if kp.name in seg_name or seg_name.endswith(kp.name):
                        z_cursor += seg_len
                        break

                joint_pos[kp.name] = (round(x_m, 4), round(y_m, 4), round(z_m, 4))

        # Compute distances: each joint to each object.
        j2o_distances: dict[str, dict[str, float]] = {}
        min_gripper_dist = float('inf')
        nearest_obj = ""

        for jp_name, jp_xyz in joint_pos.items():
            j2o_distances[jp_name] = {}
            for obj_name, obj_xyz in obj_positions.items():
                dist = math.sqrt(
                    (jp_xyz[0] - obj_xyz[0]) ** 2
                    + (jp_xyz[1] - obj_xyz[1]) ** 2
                    + (jp_xyz[2] - obj_xyz[2]) ** 2
                )
                j2o_distances[jp_name][obj_name] = round(dist, 4)

                # Track gripper proximity.
                is_gripper = any(
                    g in jp_name.lower()
                    for g in ("wrist", "hand", "fingertip", "gripper")
                )
                if is_gripper and dist < min_gripper_dist:
                    min_gripper_dist = dist
                    nearest_obj = obj_name

        results.append(SpatialSnapshot(
            frame_index=fi,
            joint_positions=joint_pos,
            object_positions=dict(obj_positions),
            joint_to_object_distances=j2o_distances,
            gripper_to_target_distance=min_gripper_dist if min_gripper_dist < float('inf') else 0.0,
            nearest_object=nearest_obj,
        ))

    return results


# ── Basis record conversion ──

def motor_profile_to_basis_records(
    profile: MotorProfile,
    user_id: str,
    preset_id: str,
) -> list[dict]:
    """Convert a MotorProfile into basis records for batch storage.

    Categories: motor_waypoints, motor_speed, motor_spatial, motor_summary.
    """
    records: list[dict] = []

    # ── Waypoint records (per-frame predicted ctrl) ──
    for wp in profile.waypoints:
        angles_str = ", ".join(f"{j}={a:+.3f}rad" for j, a in wp.joint_angles_rad.items())
        grip_str = "closed" if wp.gripper_state > 0.5 else "open"
        payload_str = f" Holding {wp.payload_name} ({wp.payload_mass_kg:.1f}kg)." if wp.payload_name else ""
        records.append({
            "user_id": user_id,
            "category": "motor_waypoints",
            "content": (
                f"Frame {wp.frame_index} ({wp.phase}): predicted joint angles "
                f"[{angles_str}], gripper {grip_str}.{payload_str} "
                f"Ctrl: [{', '.join(f'{c:.3f}' for c in wp.ctrl_array)}]."
            ),
            "data": {
                "frame_index": wp.frame_index,
                "phase": wp.phase,
                "joint_angles_rad": wp.joint_angles_rad,
                "joint_velocities_rads": wp.joint_velocities_rads,
                "gravity_compensation": wp.gravity_compensation,
                "gripper_state": wp.gripper_state,
                "payload_mass_kg": wp.payload_mass_kg,
                "payload_name": wp.payload_name,
                "ctrl_array": wp.ctrl_array,
            },
            "source": "analysis",
            "confidence": 0.7,
            "preset_id": preset_id,
        })

    # ── Speed records (per-phase) ──
    for sp in profile.speed_profile:
        records.append({
            "user_id": user_id,
            "category": "motor_speed",
            "content": (
                f"Phase '{sp.phase}': recommended {sp.velocity_factor:.0%} max velocity. "
                f"Duration ~{sp.recommended_duration_sec:.1f}s. "
                f"Torque headroom: {sp.torque_headroom:.0%}. {sp.reason}."
            ),
            "data": {
                "phase": sp.phase,
                "velocity_factor": sp.velocity_factor,
                "max_joint_velocity_rads": sp.max_joint_velocity_rads,
                "recommended_duration_sec": sp.recommended_duration_sec,
                "torque_headroom": sp.torque_headroom,
                "reason": sp.reason,
            },
            "source": "analysis",
            "confidence": 0.7,
            "preset_id": preset_id,
        })

    # ── Spatial records (per-frame, but only store key frames to avoid bloat) ──
    # Store first, last, and any frame where nearest object changes.
    key_spatial_indices = set()
    if profile.spatial_snapshots:
        key_spatial_indices.add(0)
        key_spatial_indices.add(len(profile.spatial_snapshots) - 1)
        prev_nearest = ""
        for ss in profile.spatial_snapshots:
            if ss.nearest_object != prev_nearest:
                key_spatial_indices.add(ss.frame_index)
                prev_nearest = ss.nearest_object

    for ss in profile.spatial_snapshots:
        if ss.frame_index not in key_spatial_indices:
            continue

        obj_strs = []
        for name, pos in ss.object_positions.items():
            obj_strs.append(f"{name} at ({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f})")
        nearest_str = f" Nearest to gripper: {ss.nearest_object} ({ss.gripper_to_target_distance:.3f}m)." if ss.nearest_object else ""

        records.append({
            "user_id": user_id,
            "category": "motor_spatial",
            "content": (
                f"Frame {ss.frame_index} spatial: "
                f"{len(ss.joint_positions)} joint positions, "
                f"{len(ss.object_positions)} objects [{'; '.join(obj_strs)}].{nearest_str}"
            ),
            "data": {
                "frame_index": ss.frame_index,
                "joint_positions": {k: list(v) for k, v in ss.joint_positions.items()},
                "object_positions": {k: list(v) for k, v in ss.object_positions.items()},
                "joint_to_object_distances": ss.joint_to_object_distances,
                "gripper_to_target_distance": ss.gripper_to_target_distance,
                "nearest_object": ss.nearest_object,
            },
            "source": "analysis",
            "confidence": 0.65,
            "preset_id": preset_id,
        })

    # ── Summary record ──
    records.append({
        "user_id": user_id,
        "category": "motor_summary",
        "content": (
            f"Motor profile: robot '{profile.robot_name}', "
            f"{profile.n_actuators} actuators ({profile.actuator_type} control), "
            f"action dimension {profile.action_dimension}. "
            f"{len(profile.waypoints)} predicted waypoints, "
            f"{len(profile.speed_profile)} phase speed settings, "
            f"{len(profile.spatial_snapshots)} spatial snapshots. "
            f"Joints: {', '.join(profile.joint_names)}."
        ),
        "data": {
            "robot_name": profile.robot_name,
            "n_actuators": profile.n_actuators,
            "actuator_type": profile.actuator_type,
            "action_dimension": profile.action_dimension,
            "joint_names": profile.joint_names,
            "ctrl_range": profile.ctrl_range,
            "n_waypoints": len(profile.waypoints),
            "n_phases": len(profile.speed_profile),
        },
        "source": "analysis",
        "confidence": 0.8,
        "preset_id": preset_id,
    })

    return records
