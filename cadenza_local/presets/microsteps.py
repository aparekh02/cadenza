"""Microstep generator — dense 30Hz control guidance from preset data.

Takes a BasisPreset (coarse per-frame waypoints at ~2Hz) and interpolates
to produce dense microsteps at the VLA control rate (30Hz). Each microstep
is an 18D feature vector injected into SmolVLA's state input.

The 18 dimensions:
    [0:6]   target_delta_rad   — per-joint position delta to reach
    [6:12]  torque_fraction    — per-joint load as fraction of capacity
    [12]    speed_factor       — 0-1 velocity multiplier
    [13]    gripper_command    — 0=open, 1=closed
    [14]    phase_progress     — 0-1 within current phase
    [15]    balance_score      — 0=stable, 1=unstable
    [16]    payload_mass_norm  — payload_kg / 2.0
    [17]    gripper_proximity  — 0-1 closeness to target object

Usage:
    from cadenza_local.presets.microsteps import generate_microsteps

    seq = generate_microsteps(preset, hz=30)
    for step in seq.steps:
        features = step.to_vector()  # 18D list
"""

from __future__ import annotations

import math
from typing import Optional

from cadenza_local.presets.schemas import (
    BasisPreset,
    Microstep,
    MicrostepSequence,
    MotorProfile,
    DynamicsProfile,
    RobotProfile,
)

# Default number of joints for SO-101.
DEFAULT_N_JOINTS = 6

# Maximum proximity distance (meters) — beyond this, proximity = 0.
MAX_PROXIMITY_DISTANCE = 0.5

# Payload normalization factor (kg).
PAYLOAD_NORM_KG = 2.0


def generate_microsteps(
    preset: BasisPreset,
    hz: int = 30,
) -> MicrostepSequence:
    """Generate dense microstep guidance from a BasisPreset.

    Interpolates between coarse per-frame waypoints to produce control
    guidance at the specified Hz rate.

    Args:
        preset: Complete preset with motor profile, dynamics, robot.
        hz: Control frequency in Hz (default 30).

    Returns:
        MicrostepSequence with interpolated steps.
    """
    motor = preset.motor
    dynamics = preset.dynamics
    robot = preset.robot

    if not motor or not motor.waypoints:
        return MicrostepSequence(hz=hz)

    n_joints = motor.n_actuators or DEFAULT_N_JOINTS
    joint_names = motor.joint_names or [f"joint_{i}" for i in range(n_joints)]

    # Build torque capacity lookup.
    max_torques: dict[str, float] = {}
    if robot and robot.max_torques_nm:
        max_torques = dict(robot.max_torques_nm)

    # Build per-frame data arrays for interpolation.
    n_frames = len(motor.waypoints)
    frame_angles = _extract_frame_angles(motor, joint_names, n_frames)
    frame_torque_frac = _extract_torque_fractions(dynamics, joint_names, n_frames, max_torques)
    frame_balance = _extract_balance_scores(dynamics, n_frames)
    frame_proximity = _extract_proximity(motor, n_frames)
    frame_payload = _extract_payload(motor, n_frames)
    frame_gripper = _extract_gripper(motor, n_frames)

    # Build phase schedule: list of (phase_name, n_steps, frame_start, frame_end).
    phase_schedule = _build_phase_schedule(motor, n_frames, hz)

    # Generate microsteps by interpolating across phases.
    all_steps: list[Microstep] = []
    phase_boundaries: list[tuple[int, int, str]] = []
    global_step = 0

    for phase_name, n_steps, f_start, f_end, vel_factor in phase_schedule:
        phase_start_step = global_step
        n_phase_frames = max(f_end - f_start, 1)

        for local_step in range(n_steps):
            # Map local step to fractional frame index.
            t = local_step / max(n_steps - 1, 1)  # 0.0 to 1.0
            frame_frac = f_start + t * n_phase_frames

            fi_lo = min(int(frame_frac), n_frames - 1)
            fi_hi = min(fi_lo + 1, n_frames - 1)
            alpha = frame_frac - fi_lo

            # Interpolate joint angles and compute deltas.
            current_angles = _lerp_list(frame_angles[fi_lo], frame_angles[fi_hi], alpha)
            if fi_hi < n_frames:
                next_angles = frame_angles[fi_hi]
            else:
                next_angles = current_angles
            target_delta = [next_angles[j] - current_angles[j] for j in range(n_joints)]

            # Interpolate torque fractions.
            torque_frac = _lerp_list(frame_torque_frac[fi_lo], frame_torque_frac[fi_hi], alpha)

            # Interpolate balance.
            balance = _lerp_scalar(frame_balance[fi_lo], frame_balance[fi_hi], alpha)

            # Interpolate proximity.
            proximity = _lerp_scalar(frame_proximity[fi_lo], frame_proximity[fi_hi], alpha)

            # Gripper: snap to nearest frame (no interpolation — binary).
            gripper = frame_gripper[fi_lo] if alpha < 0.5 else frame_gripper[fi_hi]

            # Payload: snap to nearest frame.
            payload = frame_payload[fi_lo] if alpha < 0.5 else frame_payload[fi_hi]

            # Phase progress.
            phase_progress = local_step / max(n_steps - 1, 1)

            all_steps.append(Microstep(
                timestep=global_step,
                phase=phase_name,
                target_delta_rad=target_delta,
                torque_fraction=torque_frac,
                speed_factor=vel_factor,
                gripper_command=gripper,
                phase_progress=phase_progress,
                balance_score=balance,
                payload_mass_norm=payload / PAYLOAD_NORM_KG,
                gripper_proximity=proximity,
            ))
            global_step += 1

        phase_boundaries.append((phase_start_step, global_step - 1, phase_name))

    total_duration = global_step / hz if hz > 0 else 0.0

    return MicrostepSequence(
        steps=all_steps,
        hz=hz,
        total_duration_sec=total_duration,
        n_phases=len(phase_schedule),
        phase_boundaries=phase_boundaries,
    )


# ── Data extraction helpers ──

def _extract_frame_angles(
    motor: MotorProfile,
    joint_names: list[str],
    n_frames: int,
) -> list[list[float]]:
    """Extract per-frame joint angle arrays from waypoints."""
    result: list[list[float]] = []
    for fi in range(n_frames):
        if fi < len(motor.waypoints):
            wp = motor.waypoints[fi]
            angles = [wp.joint_angles_rad.get(jn, 0.0) for jn in joint_names]
        else:
            angles = [0.0] * len(joint_names)
        result.append(angles)
    return result


def _extract_torque_fractions(
    dynamics: Optional[DynamicsProfile],
    joint_names: list[str],
    n_frames: int,
    max_torques: dict[str, float],
) -> list[list[float]]:
    """Extract per-frame torque fractions (torque / capacity) per joint."""
    n_joints = len(joint_names)
    result: list[list[float]] = []

    if not dynamics or not dynamics.torques:
        return [[0.0] * n_joints for _ in range(n_frames)]

    # Build torque lookup by joint name.
    torque_lookup = {t.joint_name: t for t in dynamics.torques}

    for fi in range(n_frames):
        fracs = []
        for jn in joint_names:
            te = torque_lookup.get(jn)
            if te and fi < len(te.torques_nm):
                cap = max_torques.get(jn, 5.0)
                frac = te.torques_nm[fi] / cap if cap > 0 else 0.0
                fracs.append(min(frac, 1.0))
            else:
                fracs.append(0.0)
        result.append(fracs)
    return result


def _extract_balance_scores(
    dynamics: Optional[DynamicsProfile],
    n_frames: int,
) -> list[float]:
    """Extract per-frame balance scores."""
    if not dynamics or not dynamics.balance or not dynamics.balance.scores:
        return [0.0] * n_frames
    scores = dynamics.balance.scores
    return [scores[i] if i < len(scores) else 0.0 for i in range(n_frames)]


def _extract_proximity(
    motor: MotorProfile,
    n_frames: int,
) -> list[float]:
    """Extract per-frame gripper proximity (0-1, inverted from distance)."""
    result: list[float] = []
    for fi in range(n_frames):
        if fi < len(motor.spatial_snapshots):
            dist = motor.spatial_snapshots[fi].gripper_to_target_distance
            if dist <= 0 or dist == float('inf'):
                result.append(0.0)
            else:
                prox = max(0.0, 1.0 - dist / MAX_PROXIMITY_DISTANCE)
                result.append(prox)
        else:
            result.append(0.0)
    return result


def _extract_payload(
    motor: MotorProfile,
    n_frames: int,
) -> list[float]:
    """Extract per-frame payload mass in kg."""
    result: list[float] = []
    for fi in range(n_frames):
        if fi < len(motor.waypoints):
            result.append(motor.waypoints[fi].payload_mass_kg)
        else:
            result.append(0.0)
    return result


def _extract_gripper(
    motor: MotorProfile,
    n_frames: int,
) -> list[float]:
    """Extract per-frame gripper state (0=open, 1=closed)."""
    result: list[float] = []
    for fi in range(n_frames):
        if fi < len(motor.waypoints):
            result.append(motor.waypoints[fi].gripper_state)
        else:
            result.append(0.0)
    return result


# ── Phase schedule ──

def _build_phase_schedule(
    motor: MotorProfile,
    n_frames: int,
    hz: int,
) -> list[tuple[str, int, int, int, float]]:
    """Build phase schedule: (phase_name, n_steps, frame_start, frame_end, vel_factor).

    Maps speed profile phases to frame ranges and computes step counts.
    """
    if not motor.speed_profile:
        # Single phase covering all frames.
        n_steps = max(int(n_frames * 0.5 * hz), hz)  # at least 1 second
        return [("default", n_steps, 0, n_frames - 1, 0.3)]

    # Map phases to frame ranges from waypoints.
    phase_frames: dict[str, list[int]] = {}
    for wp in motor.waypoints:
        phase = wp.phase or "unknown"
        phase_frames.setdefault(phase, []).append(wp.frame_index)

    schedule: list[tuple[str, int, int, int, float]] = []

    for sp in motor.speed_profile:
        phase_name = sp.phase
        vel_factor = sp.velocity_factor
        duration = sp.recommended_duration_sec

        # Frame range for this phase.
        f_indices = phase_frames.get(phase_name, [])
        if f_indices:
            f_start = min(f_indices)
            f_end = max(f_indices)
        else:
            # Fallback: distribute evenly.
            idx = len(schedule)
            total = len(motor.speed_profile)
            f_start = int(idx / total * n_frames)
            f_end = int((idx + 1) / total * n_frames) - 1

        n_steps = max(int(duration * hz), 1)
        schedule.append((phase_name, n_steps, f_start, f_end, vel_factor))

    return schedule


# ── Interpolation helpers ──

def _lerp_scalar(a: float, b: float, t: float) -> float:
    """Linear interpolation between two scalars."""
    return a + (b - a) * t


def _lerp_list(a: list[float], b: list[float], t: float) -> list[float]:
    """Linear interpolation between two lists element-wise."""
    return [a[i] + (b[i] - a[i]) * t for i in range(len(a))]


# ── Basis record conversion ──

def microsteps_to_basis_records(
    seq: MicrostepSequence,
    user_id: str,
    preset_id: str,
) -> list[dict]:
    """Convert a MicrostepSequence into basis records for batch storage.

    Stores per-phase summaries (not every step) to avoid record bloat.
    Category: microstep_phases.
    """
    records: list[dict] = []

    # Summary record.
    records.append({
        "user_id": user_id,
        "category": "microstep_summary",
        "content": (
            f"Microstep sequence: {len(seq.steps)} steps at {seq.hz}Hz, "
            f"{seq.total_duration_sec:.1f}s total, {seq.n_phases} phases."
        ),
        "data": {
            "n_steps": len(seq.steps),
            "hz": seq.hz,
            "total_duration_sec": seq.total_duration_sec,
            "n_phases": seq.n_phases,
            "phase_boundaries": seq.phase_boundaries,
        },
        "source": "analysis",
        "confidence": 0.8,
        "preset_id": preset_id,
    })

    # Per-phase records.
    for start, end, phase_name in seq.phase_boundaries:
        phase_steps = seq.steps[start:end + 1]
        if not phase_steps:
            continue

        # Aggregate stats for the phase.
        avg_speed = sum(s.speed_factor for s in phase_steps) / len(phase_steps)
        avg_balance = sum(s.balance_score for s in phase_steps) / len(phase_steps)
        max_torque = max(
            max(s.torque_fraction) if s.torque_fraction else 0.0
            for s in phase_steps
        )
        gripper = phase_steps[0].gripper_command
        payload = phase_steps[0].payload_mass_norm * PAYLOAD_NORM_KG

        duration = (end - start + 1) / seq.hz if seq.hz > 0 else 0.0
        payload_str = f" Payload: {payload:.1f}kg." if payload > 0 else ""

        records.append({
            "user_id": user_id,
            "category": "microstep_phases",
            "content": (
                f"Phase '{phase_name}': {end - start + 1} steps ({duration:.1f}s). "
                f"Speed: {avg_speed:.0%}, gripper: {'closed' if gripper > 0.5 else 'open'}, "
                f"max torque load: {max_torque:.0%}, balance: {avg_balance:.2f}.{payload_str}"
            ),
            "data": {
                "phase_name": phase_name,
                "step_start": start,
                "step_end": end,
                "n_steps": end - start + 1,
                "duration_sec": duration,
                "avg_speed_factor": avg_speed,
                "avg_balance": avg_balance,
                "max_torque_fraction": max_torque,
                "gripper_command": gripper,
                "payload_kg": payload,
            },
            "source": "analysis",
            "confidence": 0.75,
            "preset_id": preset_id,
        })

    return records
