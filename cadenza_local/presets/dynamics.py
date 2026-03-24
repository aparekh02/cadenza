"""Physics dynamics extraction from pose sequences.

Computes predicted physics profiles from the skeleton pose data extracted
during the presets pipeline. These predictions let the ForceVectorSpace
controller know what to expect before the robot moves.

Two modes:
    - Geometry-only (no RobotProfile): uses normalized segment lengths/masses,
      produces approximate but useful physics.
    - Robot-spec mode: uses actual masses, torques, segment lengths from
      the RobotProfile for accurate estimates.

Torque estimation model (serial manipulator chain):
    τ_i = I_i · α_i + m_i · g · (L_i/2) · cos(θ_i) + Σ(downstream loads) + payload
    where I_i = (1/3) · m · L²
"""

from __future__ import annotations

import math
from typing import Optional

from cadenza_local.presets.schemas import (
    MotionBlueprint,
    RobotProfile,
    ObjectProfile,
    FrameAnalysis,
    SegmentKinematics,
    JointTorqueEstimate,
    COMTrajectory,
    BalanceProfile,
    PhaseForceProfile,
    DynamicsProfile,
)

# ── Constants ──

GRAVITY = 9.81  # m/s²

# Default segment properties when no robot spec is available.
# (length_m, mass_kg) — approximate values for a small manipulator arm.
DEFAULT_SEGMENT_PROPS: dict[str, tuple[float, float]] = {
    "base_rotation":  (0.05, 0.3),
    "upper_arm":      (0.10, 0.2),
    "forearm":        (0.08, 0.15),
    "hand":           (0.06, 0.08),
}

# Chain order: base → tip (used for downstream load accumulation).
CHAIN_ORDER = ["base_rotation", "upper_arm", "forearm", "hand"]

# Dual-arm prefixes.
DUAL_PREFIXES = ["left_", "right_"]


# ── Helpers ──

def _strip_arm_prefix(name: str) -> str:
    """Remove left_/right_ prefix to get the canonical segment name."""
    for prefix in DUAL_PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def _segment_props(
    segment_name: str,
    robot: Optional[RobotProfile],
) -> tuple[float, float]:
    """Return (length_m, mass_kg) for a segment."""
    canonical = _strip_arm_prefix(segment_name)

    if robot and robot.segment_map:
        # segment_map: {segment_name: [(joint_name, contribution), ...]}
        # Try exact match first, then canonical.
        seg_map = robot.segment_map.get(segment_name) or robot.segment_map.get(canonical)
        if seg_map:
            # Estimate length from joint contributions, mass from total / n_segments.
            length = sum(abs(c) for _, c in seg_map) if seg_map else 0.1
            n_segs = max(len(robot.segment_map), 1)
            mass = robot.total_mass_kg / n_segs
            return (length, mass)

    return DEFAULT_SEGMENT_PROPS.get(canonical, (0.08, 0.15))


def _max_torque_for_segment(
    segment_name: str,
    robot: Optional[RobotProfile],
) -> float:
    """Return max torque (Nm) for the joint driving this segment."""
    if not robot or not robot.max_torques_nm:
        return 5.0  # default

    canonical = _strip_arm_prefix(segment_name)

    # Try segment_map → joint names.
    if robot.segment_map:
        seg_map = robot.segment_map.get(segment_name) or robot.segment_map.get(canonical)
        if seg_map:
            joint_name = seg_map[0][0]
            return robot.max_torques_nm.get(joint_name, 5.0)

    # Fallback: try matching joint names containing the canonical name.
    for jname, torque in robot.max_torques_nm.items():
        if canonical.replace("_", "") in jname.replace("_", ""):
            return torque
    return 5.0


def _get_chain(segment_names: list[str]) -> list[list[str]]:
    """Group segments into kinematic chains (one per arm or single chain).

    Returns a list of chains. Each chain is ordered base → tip.
    """
    left = [s for s in segment_names if s.startswith("left_")]
    right = [s for s in segment_names if s.startswith("right_")]
    unprefixed = [s for s in segment_names if not s.startswith("left_") and not s.startswith("right_")]

    chains: list[list[str]] = []

    for group in [unprefixed, left, right]:
        if not group:
            continue
        # Sort by chain order.
        order_map = {c: i for i, c in enumerate(CHAIN_ORDER)}
        group.sort(key=lambda s: order_map.get(_strip_arm_prefix(s), 99))
        chains.append(group)

    return chains


def _payload_at_frame(
    frame_idx: int,
    frames: list[FrameAnalysis],
    objects: list[ObjectProfile],
) -> tuple[float, str]:
    """Return (mass_kg, object_name) of any held payload at this frame."""
    if frame_idx >= len(frames):
        return (0.0, "")

    frame = frames[frame_idx]
    held_interactions = {"grasping", "lifting", "holding", "pouring", "stirring"}

    for inter in frame.interactions:
        inter_lower = inter.lower()
        for obj in objects:
            if obj.name.lower() in inter_lower:
                for itype in obj.interaction_types:
                    if itype in held_interactions:
                        return (obj.estimated_mass_kg, obj.name)

    # Check object detections.
    for det in frame.objects_detected:
        if det.interaction in held_interactions:
            for obj in objects:
                if obj.name == det.name:
                    return (obj.estimated_mass_kg, obj.name)

    return (0.0, "")


# ── Core computation ──

def compute_dynamics(
    blueprint: MotionBlueprint,
    frames: list[FrameAnalysis],
    robot: Optional[RobotProfile] = None,
    objects: Optional[list[ObjectProfile]] = None,
    dt: float = 0.0,
) -> DynamicsProfile:
    """Compute full dynamics profile from a motion blueprint.

    Args:
        blueprint: Motion blueprint with angle deltas and phase info.
        frames: Per-frame analysis with segments, keypoints, phases.
        robot: Optional robot profile for accurate mass/torque values.
        objects: Optional object list for payload estimation.
        dt: Time step between frames in seconds. If 0, estimated from
            blueprint.duration_estimate_sec / n_frames.

    Returns:
        Complete DynamicsProfile.
    """
    objects = objects or []
    n_frames = blueprint.total_frames or len(frames)
    if n_frames < 1:
        return DynamicsProfile()

    if dt <= 0:
        duration = blueprint.duration_estimate_sec or (n_frames * 0.5)
        dt = duration / max(n_frames, 1)

    # Collect all segment names from the first frame that has segments.
    segment_names: list[str] = []
    for frame in frames:
        if frame.segments:
            segment_names = [s.name for s in frame.segments]
            break

    if not segment_names:
        # Fall back to angle_deltas keys.
        segment_names = list(blueprint.angle_deltas.keys())

    if not segment_names:
        return DynamicsProfile(n_frames=n_frames)

    # ── 1. Segment kinematics ──
    kinematics = _compute_kinematics(segment_names, frames, blueprint, dt)

    # ── 2. Joint torque estimates ──
    chains = _get_chain(segment_names)
    torques = _compute_torques(
        chains, kinematics, frames, robot, objects, dt,
    )

    # ── 3. COM trajectory ──
    com = _compute_com(segment_names, frames, robot)

    # ── 4. Balance profile ──
    balance = _compute_balance(com, n_frames)

    # ── 5. Phase force profiles ──
    phase_forces = _compute_phase_forces(
        blueprint, frames, kinematics, torques, objects, dt,
    )

    # ── Summary metrics ──
    peak_torque = max((t.peak_torque_nm for t in torques), default=0.0)
    total_energy = sum(pf.energy_joules for pf in phase_forces)
    hardest = max(phase_forces, key=lambda p: p.peak_torque_nm) if phase_forces else None

    return DynamicsProfile(
        kinematics=kinematics,
        torques=torques,
        com=com,
        balance=balance,
        phase_forces=phase_forces,
        total_energy_joules=total_energy,
        peak_torque_nm=peak_torque,
        hardest_phase=hardest.phase_name if hardest else "",
        n_segments=len(segment_names),
        n_frames=n_frames,
    )


# ── Sub-computations ──

def _compute_kinematics(
    segment_names: list[str],
    frames: list[FrameAnalysis],
    blueprint: MotionBlueprint,
    dt: float,
) -> list[SegmentKinematics]:
    """Compute angular velocity and acceleration for each segment."""
    results: list[SegmentKinematics] = []

    for seg_name in segment_names:
        # Build angle time-series from frames.
        angles: list[float] = []
        for frame in frames:
            angle = 0.0
            for s in frame.segments:
                if s.name == seg_name:
                    angle = s.angle_deg
                    break
            angles.append(angle)

        # Velocities: finite difference.
        velocities: list[float] = []
        for i in range(1, len(angles)):
            vel = (angles[i] - angles[i - 1]) / dt if dt > 0 else 0.0
            velocities.append(vel)

        # Accelerations: finite difference of velocities.
        accels: list[float] = []
        for i in range(1, len(velocities)):
            acc = (velocities[i] - velocities[i - 1]) / dt if dt > 0 else 0.0
            accels.append(acc)

        peak_vel = max((abs(v) for v in velocities), default=0.0)
        peak_acc = max((abs(a) for a in accels), default=0.0)

        results.append(SegmentKinematics(
            segment_name=seg_name,
            angles_deg=angles,
            angular_velocity_dps=velocities,
            angular_accel_dps2=accels,
            peak_velocity_dps=peak_vel,
            peak_accel_dps2=peak_acc,
        ))

    return results


def _compute_torques(
    chains: list[list[str]],
    kinematics: list[SegmentKinematics],
    frames: list[FrameAnalysis],
    robot: Optional[RobotProfile],
    objects: list[ObjectProfile],
    dt: float,
) -> list[JointTorqueEstimate]:
    """Estimate torques using the serial manipulator chain model.

    τ_i = I_i·α_i + m_i·g·(L_i/2)·cos(θ_i) + Σ(downstream gravity) + payload
    """
    kin_map = {sk.segment_name: sk for sk in kinematics}
    results: list[JointTorqueEstimate] = []
    n_frames = max(len(frames), 1)

    for chain in chains:
        for idx, seg_name in enumerate(chain):
            sk = kin_map.get(seg_name)
            if not sk:
                continue

            length, mass = _segment_props(seg_name, robot)
            inertia = (1.0 / 3.0) * mass * length * length  # I = (1/3)mL²

            # Downstream segments: everything after this in the chain.
            downstream = chain[idx + 1:]
            downstream_mass = sum(_segment_props(ds, robot)[1] for ds in downstream)
            downstream_length_sum = sum(_segment_props(ds, robot)[0] for ds in downstream)

            torques_per_frame: list[float] = []
            max_torque = _max_torque_for_segment(seg_name, robot)

            for fi in range(n_frames):
                # Angle at this frame.
                angle_rad = math.radians(sk.angles_deg[fi]) if fi < len(sk.angles_deg) else 0.0

                # Angular acceleration (use frame fi-1 transition).
                alpha = 0.0
                accel_idx = fi - 2  # accels are offset by 2 from frame index
                if 0 <= accel_idx < len(sk.angular_accel_dps2):
                    alpha = math.radians(sk.angular_accel_dps2[accel_idx])

                # Payload at this frame.
                payload_mass, _ = _payload_at_frame(fi, frames, objects)

                # Torque: inertial + gravity (self) + gravity (downstream) + payload.
                tau_inertial = inertia * alpha
                tau_gravity_self = mass * GRAVITY * (length / 2.0) * math.cos(angle_rad)
                tau_gravity_downstream = downstream_mass * GRAVITY * (length + downstream_length_sum / 2.0) * math.cos(angle_rad)
                tau_payload = payload_mass * GRAVITY * (length + downstream_length_sum) * math.cos(angle_rad)

                tau = abs(tau_inertial) + abs(tau_gravity_self) + abs(tau_gravity_downstream) + abs(tau_payload)
                torques_per_frame.append(tau)

            peak = max(torques_per_frame) if torques_per_frame else 0.0
            mean = sum(torques_per_frame) / len(torques_per_frame) if torques_per_frame else 0.0
            capacity = peak / max_torque if max_torque > 0 else 0.0

            # Find which phase the peak occurs in.
            peak_frame = torques_per_frame.index(peak) if torques_per_frame else 0
            peak_phase = ""
            if peak_frame < len(frames):
                peak_phase = frames[peak_frame].phase_label

            results.append(JointTorqueEstimate(
                joint_name=seg_name,
                torques_nm=torques_per_frame,
                peak_torque_nm=peak,
                mean_torque_nm=mean,
                capacity_fraction=min(capacity, 1.0),
                peak_phase=peak_phase,
            ))

    return results


def _compute_com(
    segment_names: list[str],
    frames: list[FrameAnalysis],
    robot: Optional[RobotProfile],
) -> COMTrajectory:
    """Compute mass-weighted center of mass from keypoint positions."""
    positions: list[tuple[float, float]] = []
    velocities: list[tuple[float, float]] = []
    offsets: list[float] = []

    # Base position: first keypoint named "base" or (0.5, 0.9) default.
    base_x, base_y = 0.5, 0.9

    for frame in frames:
        if not frame.keypoints:
            positions.append((base_x, base_y))
            offsets.append(0.0)
            continue

        # Find base.
        for kp in frame.keypoints:
            if kp.name == "base":
                base_x, base_y = kp.x, kp.y
                break

        # Mass-weighted centroid.
        total_mass = 0.0
        cx, cy = 0.0, 0.0
        for kp in frame.keypoints:
            # Approximate: use segment mass for its end joint.
            m = 0.15  # default mass per keypoint
            if robot:
                n = max(len(frame.keypoints), 1)
                m = robot.total_mass_kg / n
            cx += kp.x * m
            cy += kp.y * m
            total_mass += m

        if total_mass > 0:
            cx /= total_mass
            cy /= total_mass

        positions.append((cx, cy))
        dist = math.sqrt((cx - base_x) ** 2 + (cy - base_y) ** 2)
        offsets.append(dist)

    # Velocities between frames (normalized coords / frame).
    for i in range(1, len(positions)):
        vx = positions[i][0] - positions[i - 1][0]
        vy = positions[i][1] - positions[i - 1][1]
        velocities.append((vx, vy))

    max_offset = max(offsets) if offsets else 0.0

    return COMTrajectory(
        positions_xy=positions,
        velocities_xy=velocities,
        base_offsets=offsets,
        max_offset=max_offset,
    )


def _compute_balance(
    com: COMTrajectory,
    n_frames: int,
) -> BalanceProfile:
    """Compute balance stability scores from COM trajectory.

    Mirrors ForceVectorSpace.balance_urgency: higher offset from base
    means less stable. Score range 0.0 (stable) to 1.0 (unstable).
    """
    scores: list[float] = []
    max_offset = com.max_offset if com.max_offset > 0 else 1.0

    for offset in com.base_offsets:
        # Normalize offset to 0-1 range, with a nonlinear curve.
        normalized = min(offset / max(max_offset * 1.5, 0.01), 1.0)
        # Apply sigmoid-like scaling: small offsets → low urgency, large → high.
        score = normalized ** 2 * 1.5
        score = min(score, 1.0)
        scores.append(score)

    critical = [i for i, s in enumerate(scores) if s > 0.6]
    mean_score = sum(scores) / len(scores) if scores else 0.0
    peak_score = max(scores) if scores else 0.0

    return BalanceProfile(
        scores=scores,
        critical_frames=critical,
        mean_score=mean_score,
        peak_score=peak_score,
    )


def _compute_phase_forces(
    blueprint: MotionBlueprint,
    frames: list[FrameAnalysis],
    kinematics: list[SegmentKinematics],
    torques: list[JointTorqueEstimate],
    objects: list[ObjectProfile],
    dt: float,
) -> list[PhaseForceProfile]:
    """Compute force/torque profiles per task phase."""
    if not blueprint.task_phases:
        return []

    # Map frames to phases.
    phase_frames: dict[str, list[int]] = {}
    for i, frame in enumerate(frames):
        phase = frame.phase_label or "unknown"
        phase_frames.setdefault(phase, []).append(i)

    # Also include phases from blueprint that may not appear in frames.
    for phase in blueprint.task_phases:
        if phase not in phase_frames:
            phase_frames[phase] = []

    torque_map = {t.joint_name: t for t in torques}
    results: list[PhaseForceProfile] = []

    for phase_name, frame_indices in phase_frames.items():
        if not frame_indices:
            results.append(PhaseForceProfile(phase_name=phase_name))
            continue

        f_start = min(frame_indices)
        f_end = max(frame_indices)

        # Collect torques in this phase range.
        phase_torques: dict[str, list[float]] = {}
        for seg_name, te in torque_map.items():
            seg_torques = []
            for fi in frame_indices:
                if fi < len(te.torques_nm):
                    seg_torques.append(te.torques_nm[fi])
            if seg_torques:
                phase_torques[seg_name] = seg_torques

        # Dominant segments: top 3 by peak torque in this phase.
        seg_peaks = {s: max(ts) for s, ts in phase_torques.items() if ts}
        dominant = sorted(seg_peaks, key=seg_peaks.get, reverse=True)[:3]

        # Peak and mean torques across all segments in this phase.
        all_torques = [t for ts in phase_torques.values() for t in ts]
        peak_t = max(all_torques) if all_torques else 0.0
        mean_t = sum(all_torques) / len(all_torques) if all_torques else 0.0

        # Payload: check first frame of phase.
        payload_mass, payload_name = _payload_at_frame(f_start, frames, objects)

        # Energy estimate: Σ(τ · Δθ) across segments in phase.
        energy = 0.0
        kin_map = {sk.segment_name: sk for sk in kinematics}
        for seg_name in phase_torques:
            sk = kin_map.get(seg_name)
            if not sk:
                continue
            te = torque_map.get(seg_name)
            if not te:
                continue
            for fi in frame_indices:
                if fi < len(te.torques_nm) and fi < len(sk.angles_deg) and fi > 0 and fi - 1 < len(sk.angles_deg):
                    delta_rad = abs(math.radians(sk.angles_deg[fi] - sk.angles_deg[fi - 1]))
                    energy += te.torques_nm[fi] * delta_rad

        # Effort level.
        if peak_t > 3.0 or payload_mass > 0.5:
            effort = "high"
        elif peak_t > 1.0 or payload_mass > 0.2:
            effort = "medium"
        else:
            effort = "low"

        results.append(PhaseForceProfile(
            phase_name=phase_name,
            frame_start=f_start,
            frame_end=f_end,
            dominant_segments=dominant,
            peak_torque_nm=peak_t,
            mean_torque_nm=mean_t,
            payload_name=payload_name,
            payload_mass_kg=payload_mass,
            energy_joules=energy,
            effort_level=effort,
        ))

    return results


# ── Basis record conversion ──

def dynamics_to_basis_records(
    dynamics: DynamicsProfile,
    user_id: str,
    preset_id: str,
) -> list[dict]:
    """Convert a DynamicsProfile into basis records for batch storage.

    Produces searchable text records in 6 categories:
        dynamics_kinematics, dynamics_torque, dynamics_com,
        dynamics_balance, dynamics_phase_forces, dynamics_summary.
    """
    records: list[dict] = []

    # ── Kinematics records ──
    for sk in dynamics.kinematics:
        records.append({
            "user_id": user_id,
            "category": "dynamics_kinematics",
            "content": (
                f"Segment '{sk.segment_name}' kinematics: "
                f"peak angular velocity {sk.peak_velocity_dps:.1f} deg/s, "
                f"peak angular acceleration {sk.peak_accel_dps2:.1f} deg/s². "
                f"{len(sk.angles_deg)} frames tracked."
            ),
            "data": {
                "segment_name": sk.segment_name,
                "peak_velocity_dps": sk.peak_velocity_dps,
                "peak_accel_dps2": sk.peak_accel_dps2,
                "angles_deg": sk.angles_deg,
                "angular_velocity_dps": sk.angular_velocity_dps,
                "angular_accel_dps2": sk.angular_accel_dps2,
            },
            "source": "analysis",
            "confidence": 0.7,
            "preset_id": preset_id,
        })

    # ── Torque records ──
    for jt in dynamics.torques:
        cap_str = f"{jt.capacity_fraction*100:.0f}% capacity" if jt.capacity_fraction > 0 else "capacity unknown"
        phase_str = f" Highest during {jt.peak_phase} phase." if jt.peak_phase else ""
        records.append({
            "user_id": user_id,
            "category": "dynamics_torque",
            "content": (
                f"Joint '{jt.joint_name}' estimated torque: "
                f"peak {jt.peak_torque_nm:.2f} Nm ({cap_str}), "
                f"mean {jt.mean_torque_nm:.2f} Nm.{phase_str}"
            ),
            "data": {
                "joint_name": jt.joint_name,
                "peak_torque_nm": jt.peak_torque_nm,
                "mean_torque_nm": jt.mean_torque_nm,
                "capacity_fraction": jt.capacity_fraction,
                "peak_phase": jt.peak_phase,
                "torques_nm": jt.torques_nm,
            },
            "source": "analysis",
            "confidence": 0.7,
            "preset_id": preset_id,
        })

    # ── COM record ──
    if dynamics.com:
        records.append({
            "user_id": user_id,
            "category": "dynamics_com",
            "content": (
                f"Center of mass trajectory: {len(dynamics.com.positions_xy)} frames. "
                f"Max offset from base: {dynamics.com.max_offset:.3f}. "
                f"Tracks mass-weighted centroid of all joints per frame."
            ),
            "data": {
                "positions_xy": dynamics.com.positions_xy,
                "velocities_xy": dynamics.com.velocities_xy,
                "base_offsets": dynamics.com.base_offsets,
                "max_offset": dynamics.com.max_offset,
            },
            "source": "analysis",
            "confidence": 0.65,
            "preset_id": preset_id,
        })

    # ── Balance record ──
    if dynamics.balance:
        crit_str = f" Critical frames: {dynamics.balance.critical_frames}." if dynamics.balance.critical_frames else " No critical frames."
        records.append({
            "user_id": user_id,
            "category": "dynamics_balance",
            "content": (
                f"Balance profile: mean stability score {dynamics.balance.mean_score:.2f}, "
                f"peak {dynamics.balance.peak_score:.2f} (0=stable, 1=unstable).{crit_str}"
            ),
            "data": {
                "scores": dynamics.balance.scores,
                "critical_frames": dynamics.balance.critical_frames,
                "mean_score": dynamics.balance.mean_score,
                "peak_score": dynamics.balance.peak_score,
            },
            "source": "analysis",
            "confidence": 0.65,
            "preset_id": preset_id,
        })

    # ── Phase force records ──
    for pf in dynamics.phase_forces:
        payload_str = f" Expected payload: {pf.payload_name} {pf.payload_mass_kg:.1f}kg." if pf.payload_name else ""
        records.append({
            "user_id": user_id,
            "category": "dynamics_phase_forces",
            "content": (
                f"Phase '{pf.phase_name}' (frames {pf.frame_start}-{pf.frame_end}): "
                f"{pf.effort_level} effort. "
                f"Dominant segments: {', '.join(pf.dominant_segments)}. "
                f"Peak torque {pf.peak_torque_nm:.2f} Nm. "
                f"Energy: {pf.energy_joules:.2f} J.{payload_str}"
            ),
            "data": {
                "phase_name": pf.phase_name,
                "frame_start": pf.frame_start,
                "frame_end": pf.frame_end,
                "dominant_segments": pf.dominant_segments,
                "peak_torque_nm": pf.peak_torque_nm,
                "mean_torque_nm": pf.mean_torque_nm,
                "payload_name": pf.payload_name,
                "payload_mass_kg": pf.payload_mass_kg,
                "energy_joules": pf.energy_joules,
                "effort_level": pf.effort_level,
            },
            "source": "analysis",
            "confidence": 0.7,
            "preset_id": preset_id,
        })

    # ── Summary record ──
    records.append({
        "user_id": user_id,
        "category": "dynamics_summary",
        "content": (
            f"Dynamics summary: peak torque {dynamics.peak_torque_nm:.2f} Nm, "
            f"total energy {dynamics.total_energy_joules:.2f} J. "
            f"Hardest phase: '{dynamics.hardest_phase}'. "
            f"{dynamics.n_segments} segments across {dynamics.n_frames} frames."
        ),
        "data": {
            "peak_torque_nm": dynamics.peak_torque_nm,
            "total_energy_joules": dynamics.total_energy_joules,
            "hardest_phase": dynamics.hardest_phase,
            "n_segments": dynamics.n_segments,
            "n_frames": dynamics.n_frames,
        },
        "source": "analysis",
        "confidence": 0.75,
        "preset_id": preset_id,
    })

    return records
