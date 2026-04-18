"""Analytical gait generator for Unitree Go1/Go2."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from cadenza.locomotion.robot_spec import GAITS, GaitParams, FOOT_ORDER, LEG_INDICES
from cadenza.locomotion.kinematics import (
    ik_leg, clip_joints, nominal_foot_positions,
    joint_vector_to_legs, legs_to_joint_vector,
)

if TYPE_CHECKING:
    from cadenza.locomotion.robot_spec import RobotSpec


def _quintic(t: float) -> float:
    """Normalised quintic: 0->1 with zero velocity/accel at endpoints."""
    t = np.clip(t, 0.0, 1.0)
    return 10*t**3 - 15*t**4 + 6*t**5


def _swing_foot_pos(
    t_norm: float,
    start:  np.ndarray,
    end:    np.ndarray,
    height: float,
) -> np.ndarray:
    """Compute 3D foot position along quintic swing arc."""
    q   = _quintic(t_norm)
    pos = (1 - q) * start + q * end
    h   = height * 4 * t_norm * (1 - t_norm)
    pos = pos.copy()
    pos[2] += h
    return pos


@dataclass
class LegState:
    name:          str
    in_stance:     bool        = True
    phase:         float       = 0.0
    foot_pos_body: np.ndarray  = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    swing_start:   np.ndarray  = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    swing_end:     np.ndarray  = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    q_target:      np.ndarray  = field(default_factory=lambda: np.zeros(3, dtype=np.float32))


class GaitEngine:
    """Generates 12-dim joint angle targets at each timestep."""

    def __init__(
        self,
        spec:        "RobotSpec",
        gait_name:   str   = "trot",
        body_height: float = 0.32,
    ):
        self._spec        = spec
        self._body_height = body_height
        self._t0          = time.monotonic()
        self._gait        = GAITS[gait_name]
        self._set_gait(gait_name)

        self._swing_height_override: float | None = None
        self._nominal = nominal_foot_positions(spec.kin, body_height)
        self._legs: dict[str, LegState] = {}
        for leg, phase_off in zip(FOOT_ORDER, self._gait.phase_offsets):
            ls = LegState(name=leg)
            ls.phase = phase_off
            ls.foot_pos_body = self._nominal[leg].copy()
            ls.swing_start   = ls.foot_pos_body.copy()
            ls.swing_end     = ls.foot_pos_body.copy()
            q = ik_leg(leg, ls.foot_pos_body, spec.kin)
            ls.q_target = q if q is not None else np.array(
                spec.poses.stand[LEG_INDICES[leg][0]:LEG_INDICES[leg][2]+1], dtype=np.float32
            )
            self._legs[leg] = ls

        self._cycle_time: float = 1.0 / self._gait.freq_hz if self._gait.freq_hz > 0 else 1.0

    def step(
        self,
        dt:        float,
        cmd_vel:   np.ndarray,
        body_rpy:  np.ndarray,
    ) -> np.ndarray:
        """Advance gait by dt seconds, return (12,) joint position targets."""
        if self._gait.freq_hz == 0:
            return self._standing_pose(body_rpy)

        self._cycle_time = 1.0 / self._gait.freq_hz

        for leg in FOOT_ORDER:
            ls = self._legs[leg]
            ls.phase = (ls.phase + dt / self._cycle_time) % 1.0

            was_stance = ls.in_stance
            ls.in_stance = (ls.phase < self._gait.duty_cycle)

            if was_stance and not ls.in_stance:
                ls.swing_start = ls.foot_pos_body.copy()
                ls.swing_end   = self._swing_target(leg, cmd_vel)

            if ls.in_stance:
                body_velocity_ground = np.array([cmd_vel[0], cmd_vel[1], 0.0], dtype=np.float32)
                ls.foot_pos_body = ls.foot_pos_body - body_velocity_ground * dt
                yaw_rate = float(cmd_vel[2]) if len(cmd_vel) > 2 else 0.0
                if abs(yaw_rate) > 0.01:
                    angle = -yaw_rate * dt
                    cx, cy = math.cos(angle), math.sin(angle)
                    px, py = ls.foot_pos_body[0], ls.foot_pos_body[1]
                    ls.foot_pos_body[0] = cx * px - cy * py
                    ls.foot_pos_body[1] = cy * px + cx * py
                ls.foot_pos_body[2] = -self._body_height
            else:
                swing_dur   = (1.0 - self._gait.duty_cycle) * self._cycle_time
                elapsed_swing = (ls.phase - self._gait.duty_cycle) * self._cycle_time
                t_norm = np.clip(elapsed_swing / swing_dur, 0.0, 1.0)
                sh = self._swing_height_override if self._swing_height_override is not None else self._gait.swing_height
                ls.foot_pos_body = _swing_foot_pos(
                    t_norm, ls.swing_start, ls.swing_end, sh
                )

            q = ik_leg(leg, ls.foot_pos_body, self._spec.kin)
            if q is not None:
                ls.q_target = q

        leg_qs = {leg: self._legs[leg].q_target for leg in FOOT_ORDER}
        q12 = legs_to_joint_vector(leg_qs)
        return clip_joints(q12, self._spec)

    def set_gait(self, gait_name: str) -> None:
        """Transition to a new gait with seamless phase preservation."""
        if gait_name not in GAITS:
            return
        self._set_gait(gait_name)
        self._cycle_time = 1.0 / self._gait.freq_hz if self._gait.freq_hz > 0 else 1.0

    def set_body_height(self, h: float) -> None:
        """Adjust nominal body height."""
        h = float(np.clip(h, 0.15, self._spec.kin.max_body_height))
        self._body_height = h
        self._nominal = nominal_foot_positions(self._spec.kin, h)

    def set_swing_height(self, h: float | None) -> None:
        """Override foot lift height. None = use gait default."""
        self._swing_height_override = h

    @property
    def stance_mask(self) -> np.ndarray:
        """(4,) float32 mask: 1.0 for stance, 0.0 for swing."""
        mask = np.zeros(4, dtype=np.float32)
        for i, leg in enumerate(FOOT_ORDER):
            if self._legs[leg].in_stance:
                mask[i] = 1.0
        return mask

    @property
    def gait_name(self) -> str:
        return self._gait.name

    @property
    def body_height(self) -> float:
        return self._body_height

    def _set_gait(self, name: str) -> None:
        self._gait = GAITS[name]

    def _swing_target(self, leg: str, cmd_vel: np.ndarray) -> np.ndarray:
        """Compute foot placement at end of swing (Raibert heuristic)."""
        nom   = self._nominal[leg].copy()
        freq  = self._gait.freq_hz if self._gait.freq_hz > 0 else 1.0
        stance_dur = self._gait.duty_cycle / freq
        nom[0] += float(cmd_vel[0]) * stance_dur * 0.5
        nom[1] += float(cmd_vel[1]) * stance_dur * 0.5
        yaw_rate = float(cmd_vel[2]) if len(cmd_vel) > 2 else 0.0
        if abs(yaw_rate) > 0.01:
            swing_dur = (1.0 - self._gait.duty_cycle) / freq
            angle = -yaw_rate * swing_dur * 0.5
            cx, cy = math.cos(angle), math.sin(angle)
            px, py = nom[0], nom[1]
            nom[0] = cx * px - cy * py
            nom[1] = cy * px + cx * py
        nom[2]  = -self._body_height
        return nom

    def _standing_pose(self, body_rpy: np.ndarray) -> np.ndarray:
        """Return standing joint targets with pitch compensation."""
        q = np.array(self._spec.poses.stand, dtype=np.float32)
        pitch = float(body_rpy[1])
        for leg in FOOT_ORDER:
            i = LEG_INDICES[leg]
            q[i[1]] = np.clip(q[i[1]] + pitch * 0.3,
                              self._spec.joints.thigh_min,
                              self._spec.joints.thigh_max)
        return q
