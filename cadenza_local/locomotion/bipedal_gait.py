"""Bipedal gait generator for Unitree G1 humanoid.

Proper stance/swing walking gait with hip/knee/ankle coordination,
modeled after the Go1 quadruped gait engine but adapted for bipedal
kinematics.  Uses position actuators — ctrl = joint target angles.

Gait cycle per leg:
  [0, duty)  STANCE — foot on ground, hip extends backward as body passes over
  [duty, 1)  SWING  — foot lifts, knee bends, hip flexes forward to place foot ahead

Left and right legs are 180 deg out of phase.

Key stability features:
  - Constant knee bend during stance lowers CoM
  - Ankle dorsiflex bias tilts body forward (prevents backward fall)
  - Hip pitch forward bias keeps CoM ahead of support foot
  - Aggressive ankle/hip balance feedback
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from cadenza_local.locomotion.robot_spec import RobotSpec


# Joint indices for G1 (16-DOF: 12 legs + 4 arms)
_L_HIP_YAW = 0
_L_HIP_ROLL = 1
_L_HIP_PITCH = 2
_L_KNEE = 3
_L_ANKLE_PITCH = 4
_L_ANKLE_ROLL = 5
_R_HIP_YAW = 6
_R_HIP_ROLL = 7
_R_HIP_PITCH = 8
_R_KNEE = 9
_R_ANKLE_PITCH = 10
_R_ANKLE_ROLL = 11
_L_SHOULDER_PITCH = 12
_L_ELBOW = 13
_R_SHOULDER_PITCH = 14
_R_ELBOW = 15

_N_JOINTS = 16


_BIPEDAL_GAITS = {
    "walk": {
        "freq_hz": 1.2,
        "duty_cycle": 0.60,
        "hip_pitch_amp": 0.25,      # rad — hip swing amplitude
        "hip_pitch_bias": 0.08,     # rad — constant forward lean via hip
        "knee_stance_bend": 0.12,   # rad — slight bent-knee stance (lowers CoM)
        "knee_swing_amp": 0.60,     # rad — knee bend for foot clearance
        "ankle_lean_fwd": 0.06,     # rad — ankle dorsiflex = forward body lean
        "ankle_swing_dorsi": 0.12,  # rad — dorsiflex during swing (clearance)
        "ankle_pushoff": 0.15,      # rad — plantarflex at end of stance
        "hip_roll_amp": 0.04,       # rad — lateral weight shift
        "arm_swing_amp": 0.30,      # rad — counter-phase arm swing
        "elbow_base": 0.20,         # rad — resting elbow bend
        "elbow_swing": 0.15,        # rad — additional bend during forward swing
    },
    "slow_walk": {
        "freq_hz": 0.8,
        "duty_cycle": 0.65,
        "hip_pitch_amp": 0.18,
        "hip_pitch_bias": 0.06,
        "knee_stance_bend": 0.10,
        "knee_swing_amp": 0.45,
        "ankle_lean_fwd": 0.05,
        "ankle_swing_dorsi": 0.08,
        "ankle_pushoff": 0.10,
        "hip_roll_amp": 0.03,
        "arm_swing_amp": 0.20,
        "elbow_base": 0.15,
        "elbow_swing": 0.10,
    },
    "stand": {
        "freq_hz": 0.0,
        "duty_cycle": 1.0,
        "hip_pitch_amp": 0.0,
        "hip_pitch_bias": 0.0,
        "knee_stance_bend": 0.0,
        "knee_swing_amp": 0.0,
        "ankle_lean_fwd": 0.0,
        "ankle_swing_dorsi": 0.0,
        "ankle_pushoff": 0.0,
        "hip_roll_amp": 0.0,
        "arm_swing_amp": 0.0,
        "elbow_base": 0.0,
        "elbow_swing": 0.0,
    },
}


def _quintic(t: float) -> float:
    """0 -> 1 with zero velocity/acceleration at endpoints."""
    t = max(0.0, min(1.0, t))
    return 10 * t**3 - 15 * t**4 + 6 * t**5


class BipedalGaitEngine:
    """Generates 16-DOF joint targets for bipedal humanoid walking.

    Proper stance/swing gait with:
    - Hip pitch for leg swing (primary locomotion driver)
    - Constant forward lean (hip bias + ankle dorsiflex + knee bend)
    - Knee flexion during swing for foot clearance
    - Ankle push-off during late stance for propulsion
    - Counter-phase arm swing for balance
    - Aggressive ankle/hip balance feedback
    """

    def __init__(
        self,
        spec: "RobotSpec",
        gait_name: str = "walk",
        body_height: float = 0.75,
    ):
        self._spec = spec
        self._body_height = body_height
        self._gait = _BIPEDAL_GAITS.get(gait_name, _BIPEDAL_GAITS["walk"])
        self._stand = np.array(spec.poses.stand, dtype=np.float64)
        self._elapsed = 0.0
        self._phase = 0.0  # Left leg phase [0, 1). Right = (phase + 0.5) % 1.0

    def step(
        self,
        dt: float,
        cmd_vel: np.ndarray,
        body_rpy: np.ndarray,
    ) -> np.ndarray:
        """Advance gait by *dt* seconds, return (16,) joint angle targets."""
        if self._gait["freq_hz"] == 0:
            return self._standing_targets(body_rpy)

        self._elapsed += dt
        freq = self._gait["freq_hz"]
        self._phase = (self._phase + dt * freq) % 1.0

        vx = float(cmd_vel[0])
        vy = float(cmd_vel[1]) if len(cmd_vel) > 1 else 0.0
        vyaw = float(cmd_vel[2]) if len(cmd_vel) > 2 else 0.0

        # Ramp up over 1.0 s to avoid impulse at start
        ramp = min(1.0, self._elapsed / 1.0)
        speed = min(abs(vx) / 0.3, 1.5) * ramp
        fwd = 1.0 if vx >= 0 else -1.0

        # Ensure minimum leg motion when turning in place
        if abs(vyaw) > 0.01 and speed < 0.3 * ramp:
            speed = 0.3 * ramp

        g = self._gait
        duty = g["duty_cycle"]
        hip_amp = g["hip_pitch_amp"] * speed * fwd
        hip_bias = g["hip_pitch_bias"] * speed * fwd     # constant forward lean
        knee_stance = g["knee_stance_bend"] * speed       # bent-knee stance
        knee_amp = g["knee_swing_amp"] * speed
        ankle_lean = g["ankle_lean_fwd"] * speed          # forward lean offset
        ankle_dorsi = g["ankle_swing_dorsi"] * speed
        pushoff = g["ankle_pushoff"] * speed * fwd
        roll_amp = g["hip_roll_amp"] * speed
        arm_amp = g["arm_swing_amp"] * speed * fwd

        q = np.zeros(_N_JOINTS, dtype=np.float64)

        # Phase per leg — 180 deg offset
        phase_L = self._phase
        phase_R = (self._phase + 0.5) % 1.0

        # Leg joint targets
        self._leg(q, 0, phase_L, duty,
                  hip_amp, hip_bias, knee_stance, knee_amp,
                  ankle_lean, ankle_dorsi, pushoff, speed)
        self._leg(q, 6, phase_R, duty,
                  hip_amp, hip_bias, knee_stance, knee_amp,
                  ankle_lean, ankle_dorsi, pushoff, speed)

        # ── Hip roll: lateral weight transfer ──
        phi = self._phase * 2.0 * math.pi
        q[_L_HIP_ROLL] = -roll_amp * math.sin(phi)
        q[_R_HIP_ROLL] = -roll_amp * math.sin(phi + math.pi)

        # ── Hip yaw: turning ──
        if abs(vyaw) > 0.01:
            yaw_scale = vyaw * 0.15 * ramp
            q[_L_HIP_YAW] = -yaw_scale * math.sin(phi)
            q[_R_HIP_YAW] = -yaw_scale * math.sin(phi + math.pi)

        # ── Lateral movement ──
        if abs(vy) > 0.01:
            lat = vy * 0.15 * ramp
            q[_L_HIP_ROLL] += lat
            q[_R_HIP_ROLL] += lat

        # ── Arm swing: counter-phase to legs ──
        q[_L_SHOULDER_PITCH] = -arm_amp * math.cos(phi)
        q[_R_SHOULDER_PITCH] = arm_amp * math.cos(phi)
        q[_L_ELBOW] = g["elbow_base"] + g["elbow_swing"] * max(0.0, -math.cos(phi))
        q[_R_ELBOW] = g["elbow_base"] + g["elbow_swing"] * max(0.0, math.cos(phi))

        # ── Balance feedback (negative feedback for stability) ──
        # Sign: body pitches forward (pitch>0) → REDUCE ankle dorsiflex → push back
        #        body pitches backward (pitch<0) → INCREASE dorsiflex → push forward
        roll_err = float(body_rpy[0])
        pitch_err = float(body_rpy[1])

        # Ankle pitch: primary pitch balance actuator
        q[_L_ANKLE_PITCH] -= pitch_err * 2.5
        q[_R_ANKLE_PITCH] -= pitch_err * 2.5
        # Ankle roll: primary roll balance
        q[_L_ANKLE_ROLL] -= roll_err * 1.2
        q[_R_ANKLE_ROLL] -= roll_err * 1.2
        # Hip roll: secondary roll compensation
        q[_L_HIP_ROLL] -= roll_err * 0.5
        q[_R_HIP_ROLL] -= roll_err * 0.5

        return self._clamp(q)

    # ── per-leg computation ──

    @staticmethod
    def _leg(
        q: np.ndarray,
        off: int,
        phase: float,
        duty: float,
        hip_amp: float,
        hip_bias: float,
        knee_stance: float,
        knee_amp: float,
        ankle_lean: float,
        ankle_dorsi: float,
        pushoff: float,
        speed: float,
    ) -> None:
        """Fill hip_pitch, knee, ankle_pitch for one leg at joint offset *off*."""
        if phase < duty:
            # ── STANCE: foot on ground, body passes over ──
            t = phase / duty                              # 0 -> 1 through stance

            # Hip pitch: +amp (leg ahead) -> -amp (leg behind) + forward bias
            q[off + 2] = hip_amp * (1.0 - 2.0 * t) + hip_bias

            # Knee: constant slight bend for stability / lower CoM
            q[off + 3] = knee_stance

            # Ankle: forward lean base + push-off in late stance
            if t < 0.5:
                q[off + 4] = ankle_lean
            else:
                push_t = (t - 0.5) / 0.5                 # 0 -> 1 in second half
                q[off + 4] = ankle_lean - pushoff * _quintic(push_t)
        else:
            # ── SWING: foot in air, leg swings forward ──
            t = (phase - duty) / (1.0 - duty)            # 0 -> 1 through swing

            # Hip pitch: -amp (behind) -> +amp (ahead) via quintic + forward bias
            q[off + 2] = hip_amp * (2.0 * _quintic(t) - 1.0) + hip_bias

            # Knee: bell-curve bend, peak at mid-swing for foot clearance
            q[off + 3] = knee_amp * math.sin(math.pi * t)

            # Ankle: dorsiflexion for ground clearance
            q[off + 4] = ankle_dorsi * math.sin(math.pi * t)

    # ── standing ──

    def _standing_targets(self, body_rpy: np.ndarray) -> np.ndarray:
        """Standing with active ankle/hip balance feedback. Returns 16-DOF."""
        q = np.zeros(_N_JOINTS, dtype=np.float64)
        q[: len(self._stand)] = self._stand
        roll_err = float(body_rpy[0])
        pitch_err = float(body_rpy[1])
        q[_L_HIP_ROLL] -= roll_err * 0.5
        q[_R_HIP_ROLL] -= roll_err * 0.5
        q[_L_ANKLE_PITCH] -= pitch_err * 2.0
        q[_R_ANKLE_PITCH] -= pitch_err * 2.0
        q[_L_ANKLE_ROLL] -= roll_err * 1.0
        q[_R_ANKLE_ROLL] -= roll_err * 1.0
        return self._clamp(q)

    # ── joint clamping ──

    def _clamp(self, q: np.ndarray) -> np.ndarray:
        """Clamp joint angles to G1 limits (16-DOF: 12 legs + 4 arms)."""
        limits = [
            # Left leg
            (-2.87, 2.87), (-0.52, 0.52), (-2.53, 2.53),
            (-0.26, 2.05), (-0.87, 0.52), (-0.26, 0.26),
            # Right leg
            (-2.87, 2.87), (-0.52, 0.52), (-2.53, 2.53),
            (-0.26, 2.05), (-0.87, 0.52), (-0.26, 0.26),
            # Left arm: shoulder_pitch, elbow
            (-3.09, 2.67), (-0.10, 2.09),
            # Right arm: shoulder_pitch, elbow
            (-3.09, 2.67), (-0.10, 2.09),
        ]
        for i, (lo, hi) in enumerate(limits):
            q[i] = max(lo, min(hi, q[i]))
        return q

    # ── properties / setters expected by sim.py ──

    @property
    def gait_name(self) -> str:
        for name, params in _BIPEDAL_GAITS.items():
            if params is self._gait:
                return name
        return "walk"

    @property
    def body_height(self) -> float:
        return self._body_height

    def set_body_height(self, h: float) -> None:
        self._body_height = h

    def set_swing_height(self, h: float | None) -> None:
        """No-op for bipedal — foot clearance is driven by knee amplitude."""
        pass
