"""Motion Mapper — converts skeleton angle changes to robot joint trajectories.

Maps the extracted skeleton segment angles from vision analysis onto any
robot's joints, producing a trajectory of joint angle waypoints that the
force space and controller can execute.

The mapping from skeleton segments to robot joints is configurable via
segment_map. Pass a custom map for your robot's joint names.

Usage:
    mapper = MotionMapper(spec)
    trajectory = mapper.map_to_trajectory(motion)
    force_states = mapper.get_waypoint_forces(trajectory, fvs, model, data)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

from cadenza_local.robot_setup import RobotSpec, RobotHints
from cadenza_local.vision import MotionSequence, SKELETON_SEGMENTS


@dataclass
class JointTrajectory:
    """A sequence of joint angle waypoints for the robot to follow."""
    joint_names: list[str]              # ordered joint names
    waypoints: list[np.ndarray]         # list of joint angle arrays (radians)
    durations: list[float]              # time between waypoints (seconds)
    n_joints: int = 0

    def __post_init__(self):
        self.n_joints = len(self.joint_names)


# ── Skeleton segment → robot joint mapping ──
#
# The vision module extracts angles for body segments (upper_arm, forearm, hand).
# This mapping connects those segments to the robot's actual joint names.
# Override DEFAULT_SEGMENT_MAP by passing a custom segment_map to MotionMapper.
#
# Example mapping for a 6-DOF arm:
#   Skeleton segment  →  Robot joint(s)
#   upper_arm         →  shoulder_lift (pitch)
#   forearm           →  elbow_flex
#   hand              →  wrist_flex

# Default mapping: skeleton segment name → list of (joint_name, scale_factor)
# These joint names match the SO-101 arm. Override for other robots.
DEFAULT_SEGMENT_MAP: dict[str, list[tuple[str, float]]] = {
    "upper_arm": [("shoulder_lift", 1.0)],
    "forearm":   [("elbow_flex", 1.0)],
    "hand":      [("wrist_flex", 1.0)],
}


class MotionMapper:
    """Maps skeleton angle changes to robot joint angle trajectories."""

    def __init__(
        self,
        spec: RobotSpec,
        hints: RobotHints | None = None,
        segment_map: dict[str, list[tuple[str, float]]] | None = None,
        waypoint_duration: float = 0.5,
    ):
        """Initialize the motion mapper.

        Args:
            spec: Robot specification with joint limits.
            hints: Robot hints (segment_map is pulled from here if not passed).
            segment_map: Explicit mapping from skeleton segments to joints.
                         Overrides hints.segment_map if both provided.
            waypoint_duration: Default time between waypoints in seconds.
        """
        self.spec = spec
        # Priority: explicit segment_map > hints.segment_map > DEFAULT_SEGMENT_MAP
        if segment_map:
            self.segment_map = segment_map
        elif hints and hints.segment_map:
            self.segment_map = hints.segment_map
        else:
            self.segment_map = DEFAULT_SEGMENT_MAP
        self.waypoint_duration = waypoint_duration

        # Build joint limit arrays for clamping
        self._joint_names = spec.joint_names
        self._joint_lo = np.array([
            spec.joints[jn].range_rad[0] if jn in spec.joints else -np.pi
            for jn in spec.joint_names
        ])
        self._joint_hi = np.array([
            spec.joints[jn].range_rad[1] if jn in spec.joints else np.pi
            for jn in spec.joint_names
        ])

    def map_to_trajectory(
        self,
        motion: MotionSequence,
        start_angles: np.ndarray | None = None,
    ) -> JointTrajectory:
        """Convert a motion sequence to a joint angle trajectory.

        Takes the angle deltas from the skeleton extraction and converts
        them to absolute joint angles for the robot arm.

        Args:
            motion: MotionSequence with per-segment angle deltas.
            start_angles: Initial joint angles (radians). If None, starts at zeros.

        Returns:
            JointTrajectory with one waypoint per motion frame.
        """
        n_joints = len(self._joint_names)

        # Start from current or zero position
        if start_angles is not None:
            current = start_angles.copy()
        else:
            current = np.zeros(n_joints)

        waypoints: list[np.ndarray] = [current.copy()]
        durations: list[float] = []

        # Number of transitions = number of delta entries
        n_transitions = 0
        for deltas in motion.angle_deltas.values():
            n_transitions = max(n_transitions, len(deltas))

        if n_transitions == 0:
            print("Warning: No motion deltas found, returning single-waypoint trajectory")
            return JointTrajectory(
                joint_names=list(self._joint_names),
                waypoints=waypoints,
                durations=[],
            )

        # Build joint name → index lookup
        jname_to_idx = {jn: i for i, jn in enumerate(self._joint_names)}

        for step_i in range(n_transitions):
            delta_angles = np.zeros(n_joints)

            # Accumulate deltas from each skeleton segment
            for segment, joint_mappings in self.segment_map.items():
                seg_deltas = motion.angle_deltas.get(segment, [])
                if step_i >= len(seg_deltas):
                    continue

                delta_deg = seg_deltas[step_i]

                for joint_name, scale in joint_mappings:
                    if joint_name in jname_to_idx:
                        idx = jname_to_idx[joint_name]
                        # Convert degrees to radians, apply scale
                        delta_angles[idx] += math.radians(delta_deg) * scale

            # Apply delta and clamp to joint limits
            current = current + delta_angles
            current = np.clip(current, self._joint_lo, self._joint_hi)

            waypoints.append(current.copy())
            durations.append(self.waypoint_duration)

        trajectory = JointTrajectory(
            joint_names=list(self._joint_names),
            waypoints=waypoints,
            durations=durations,
        )

        print(f"\nMapped trajectory: {len(waypoints)} waypoints, "
              f"{n_joints} joints")
        for i, wp in enumerate(waypoints):
            angles_deg = [f"{math.degrees(a):+6.1f}" for a in wp]
            print(f"  WP {i}: [{', '.join(angles_deg)}] deg")

        return trajectory

    def get_waypoint_forces(
        self,
        trajectory: JointTrajectory,
        fvs,   # ForceVectorSpace — avoid circular import
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> list:
        """Compute the force state at each trajectory waypoint.

        Sets the robot to each waypoint pose in simulation and calls
        fvs.compute() to get the opposing forces. This tells the
        controller what forces exist at each target pose.

        Args:
            trajectory: Joint angle trajectory to evaluate.
            fvs: ForceVectorSpace instance.
            model: MuJoCo model.
            data: MuJoCo data.

        Returns:
            List of ForceState objects, one per waypoint.
        """
        force_states = []

        for i, waypoint in enumerate(trajectory.waypoints):
            # Set all actuated joints to the waypoint angles
            for j, jname in enumerate(trajectory.joint_names):
                if jname in fvs.spec.joints:
                    ji = fvs.spec.joints[jname]
                    data.qpos[ji.qpos_index] = waypoint[j]

            # Set ctrl to match (for position actuators)
            data.ctrl[:len(waypoint)] = waypoint

            # Forward kinematics to update positions
            mujoco.mj_forward(model, data)

            # Compute force state
            state = fvs.compute(data)
            force_states.append(state)

        print(f"Computed force states for {len(force_states)} waypoints")
        return force_states

    def interpolate_waypoints(
        self,
        trajectory: JointTrajectory,
        steps_per_segment: int = 10,
    ) -> JointTrajectory:
        """Interpolate between waypoints for smoother motion.

        Args:
            trajectory: Original trajectory with coarse waypoints.
            steps_per_segment: Number of interpolation steps between each pair.

        Returns:
            New JointTrajectory with finer-grained waypoints.
        """
        if len(trajectory.waypoints) < 2:
            return trajectory

        interp_waypoints: list[np.ndarray] = [trajectory.waypoints[0].copy()]
        interp_durations: list[float] = []

        for i in range(len(trajectory.waypoints) - 1):
            start = trajectory.waypoints[i]
            end = trajectory.waypoints[i + 1]
            duration = trajectory.durations[i] if i < len(trajectory.durations) else self.waypoint_duration
            dt = duration / steps_per_segment

            for step in range(1, steps_per_segment + 1):
                t = step / steps_per_segment
                interp = start + t * (end - start)
                interp = np.clip(interp, self._joint_lo, self._joint_hi)
                interp_waypoints.append(interp)
                interp_durations.append(dt)

        return JointTrajectory(
            joint_names=trajectory.joint_names,
            waypoints=interp_waypoints,
            durations=interp_durations,
        )
