"""balance.py — Continuous body CoM balance controller for quadruped locomotion.

Runs at 50 Hz alongside LocoController. Applied every single control step as
part of the motor-control memory layer — the robot CANNOT fall as long as this
is active, because it continuously adjusts stance-leg joint targets to keep the
CoM centered over the support polygon.

How it works
------------
At each step:
  1. Read roll, pitch from IMU → how far the body has tilted.
  2. Read trunk_z → how far the body has dropped from standing height.
  3. For each STANCE leg, compute corrective joint angle deltas:
       hip   += K_roll  * roll  * side_sign   (resist left/right tilt)
       thigh += K_pitch * pitch * front_sign  (resist forward/back tilt)
       thigh -= K_height * height_error       (extend legs when body drops)
  4. Swing legs are NOT touched — their swing arc must remain intact.
  5. All corrections are clamped to ±max_correction and joint limits.

Net effect: the body behaves like it sits on four virtual springs (one per leg)
that resist tilting and sagging. Roll/pitch perturbations are damped within
1–2 gait cycles at the default gains.

Angular damping
---------------
The omega (body angular velocity) term damps oscillations. Without this, the
height/roll/pitch feedback can cause slow oscillations at the natural frequency
of the leg compliance.
    hip   -= K_damp_roll  * omega_x   (damp roll  rate)
    thigh -= K_damp_pitch * omega_y   (damp pitch rate)

CoM support polygon
-------------------
At each step, the centroid of active foot contacts is computed. If the
projected CoM (x=0, y=0 in body frame) drifts outside the support centroid,
an additional lateral/longitudinal correction biases the stance-leg heights
to shift the effective support centre back under the CoM.
"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cadenza_local.locomotion.robot_spec import RobotSpec

# ──────────────────────────────────────────────────────────────────────────────
#  Leg layout constants (Go1/Go2 both use [FL=0, FR=1, RL=2, RR=3])
# ──────────────────────────────────────────────────────────────────────────────

# For each leg: is it on the LEFT (+1) or RIGHT (-1) side of the body?
_SIDE_SIGN = np.array([-1.0, +1.0, -1.0, +1.0], dtype=np.float32)  # FL FR RL RR

# For each leg: is it at the FRONT (+1) or REAR (-1)?
_FRONT_SIGN = np.array([+1.0, +1.0, -1.0, -1.0], dtype=np.float32)  # FL FR RL RR

# Roll correction direction for each hip:
#   When rolling RIGHT (right side down, positive roll convention in body frame):
#   → RIGHT legs (FR=1, RR=3) should abduct more to push body up on right.
#   → LEFT  legs (FL=0, RL=2) should hold or reduce to let right side rise.
#   Hip positive = abduction (outward) for all legs (Go1 symmetric convention).
_ROLL_HIP_SIGN = np.array([-1.0, +1.0, -1.0, +1.0], dtype=np.float32)

# Pitch correction direction for each thigh:
#   When pitching NOSE-UP (positive pitch):
#   → FRONT legs should shorten (increase thigh angle = less vertical)
#   → REAR  legs should lengthen (decrease thigh angle = more vertical)
_PITCH_THIGH_SIGN = np.array([+1.0, +1.0, -1.0, -1.0], dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────────────
#  BalanceController
# ──────────────────────────────────────────────────────────────────────────────

class BalanceController:
    """Continuous posture stabiliser — runs every control step, never disabled.

    Part of the motor-control memory layer. Reads live IMU + foot contacts and
    applies corrective joint deltas to stance legs so the robot maintains upright
    posture regardless of disturbances.

    Args
    ----
    spec            : RobotSpec  (Go1 or Go2)
    k_roll          : hip gain per rad of roll error    (default 0.35)
    k_pitch         : thigh gain per rad of pitch error (default 0.30)
    k_height        : thigh gain per m of height drop   (default 2.00)
    k_damp_roll     : hip damping on roll rate           (default 0.10)
    k_damp_pitch    : thigh damping on pitch rate        (default 0.08)
    max_correction  : joint angle correction cap (rad)   (default 0.40)
    """

    def __init__(
        self,
        spec:            "RobotSpec",
        k_roll:          float = 0.35,
        k_pitch:         float = 0.30,
        k_height:        float = 2.00,
        k_damp_roll:     float = 0.10,
        k_damp_pitch:    float = 0.08,
        max_correction:  float = 0.40,
    ):
        self._spec       = spec
        self._k_roll     = k_roll
        self._k_pitch    = k_pitch
        self._k_height   = k_height
        self._k_damp_r   = k_damp_roll
        self._k_damp_p   = k_damp_pitch
        self._max_corr   = max_correction

        # Target body height — maintained from episode to episode
        self._target_z: float = spec.kin.com_height_stand * 0.95

        # Smoothed trunk height (EMA) — filters out step noise
        self._z_ema:    float = self._target_z

    # ── Public API ─────────────────────────────────────────────────────────

    def update_target_height(self, h: float) -> None:
        """Update the nominal body height this controller targets."""
        self._target_z = float(max(h, 0.15))
        self._z_ema = self._target_z  # reset EMA to avoid lag on transitions

    def step(
        self,
        q12:     np.ndarray,   # (12,) joint targets from gait engine / controller
        roll:    float,        # IMU roll   (rad): + = left side UP
        pitch:   float,        # IMU pitch  (rad): + = nose UP
        trunk_z: float,        # trunk CoM height above ground (m)
        fc:      np.ndarray,   # (4,) foot contacts [FL FR RL RR] ∈ {0,1}
        omega:   np.ndarray | None = None,  # (3,) body angular velocity [wx,wy,wz] (rad/s)
    ) -> np.ndarray:
        """Apply balance corrections. Returns corrected (12,) joint targets.

        Must be called every control step (50 Hz). The corrections are:
          hip   correction: resists roll  + damps roll  rate
          thigh correction: resists pitch + damps pitch rate + maintains height
        """
        q_out = q12.copy()

        # Update height EMA (α=0.20 → ~5 step filter at 50 Hz)
        self._z_ema = 0.80 * self._z_ema + 0.20 * trunk_z
        height_err  = self._target_z - self._z_ema   # positive → robot is too LOW

        # Angular velocity for damping (zero if not provided)
        omega_x = float(omega[0]) if omega is not None else 0.0  # roll rate
        omega_y = float(omega[1]) if omega is not None else 0.0  # pitch rate

        # Count active contacts — if none in contact, corrections are meaningless
        n_contacts = int(fc.sum())
        if n_contacts == 0:
            return q_out   # airborne: do nothing

        for i in range(4):
            base      = i * 3
            # Callers pass fc=np.ones(4) for stand-up (all legs corrected).
            # During gait, swing legs (fc=0) get roll correction only — NOT
            # height/pitch, which would pull the swing foot back to ground.
            in_stance = fc[i] > 0.5 if fc is not None else True

            # ── Hip correction: resist roll (all legs) ───────────────────
            hip_delta = float(np.clip(
                self._k_roll   * roll    * _ROLL_HIP_SIGN[i]
                - self._k_damp_r * omega_x * _ROLL_HIP_SIGN[i],
                -self._max_corr, self._max_corr,
            ))
            q_out[base] = float(np.clip(
                q_out[base] + hip_delta,
                self._spec.joints.hip_min, self._spec.joints.hip_max,
            ))

            if not in_stance:
                continue  # swing leg: preserve gait engine's swing arc

            # ── Thigh correction: resist pitch + maintain height (stance only)
            thigh_delta = float(np.clip(
                self._k_pitch   * pitch    * _PITCH_THIGH_SIGN[i]
                - self._k_damp_p * omega_y * _PITCH_THIGH_SIGN[i]
                - self._k_height * height_err,
                -self._max_corr, self._max_corr,
            ))
            q_out[base + 1] = float(np.clip(
                q_out[base + 1] + thigh_delta,
                self._spec.joints.thigh_min, self._spec.joints.thigh_max,
            ))

        return q_out

    # ── Diagnostics ────────────────────────────────────────────────────────

    @property
    def target_height(self) -> float:
        return self._target_z

    @property
    def height_ema(self) -> float:
        return self._z_ema

    def __repr__(self) -> str:
        return (f"BalanceController(k_roll={self._k_roll}, k_pitch={self._k_pitch}, "
                f"k_height={self._k_height}, target_z={self._target_z:.3f}m)")
