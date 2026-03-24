"""Action library for Unitree Go1, Go2, and G1."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


_HIP_IDX   = (0, 3, 6, 9)
_THIGH_IDX = (1, 4, 7, 10)
_CALF_IDX  = (2, 5, 8, 11)

_FL = (0, 1, 2)
_FR = (3, 4, 5)
_RL = (6, 7, 8)
_RR = (9, 10, 11)
_FRONT = _FL + _FR
_REAR  = _RL + _RR


_GO1_HIP_RANGE   = (-0.863, 0.863)
_GO1_THIGH_RANGE = (-0.686, 4.501)
_GO1_KNEE_RANGE  = (-2.818, -0.888)

_GO1_TORQUE_HIP   = 23.7
_GO1_TORQUE_THIGH = 23.7
_GO1_TORQUE_KNEE  = 35.55

_GO1_VEL_HIP   = 30.1
_GO1_VEL_THIGH = 30.1
_GO1_VEL_KNEE  = 20.06

_GO1_MAX_TORQUE_12 = (
    _GO1_TORQUE_HIP, _GO1_TORQUE_THIGH, _GO1_TORQUE_KNEE,
    _GO1_TORQUE_HIP, _GO1_TORQUE_THIGH, _GO1_TORQUE_KNEE,
    _GO1_TORQUE_HIP, _GO1_TORQUE_THIGH, _GO1_TORQUE_KNEE,
    _GO1_TORQUE_HIP, _GO1_TORQUE_THIGH, _GO1_TORQUE_KNEE,
)

_GO1_MAX_VEL_12 = (
    _GO1_VEL_HIP, _GO1_VEL_THIGH, _GO1_VEL_KNEE,
    _GO1_VEL_HIP, _GO1_VEL_THIGH, _GO1_VEL_KNEE,
    _GO1_VEL_HIP, _GO1_VEL_THIGH, _GO1_VEL_KNEE,
    _GO1_VEL_HIP, _GO1_VEL_THIGH, _GO1_VEL_KNEE,
)

_GO2_HIP_RANGE   = (-1.047, 1.047)
_GO2_THIGH_RANGE = (-0.663, 3.927)
_GO2_KNEE_RANGE  = (-2.721, -0.837)

_GO2_TORQUE_HIP   = 23.7
_GO2_TORQUE_THIGH = 23.7
_GO2_TORQUE_KNEE  = 45.43

_GO2_VEL_HIP   = 30.1
_GO2_VEL_THIGH = 30.1
_GO2_VEL_KNEE  = 15.70

_GO2_MAX_TORQUE_12 = (
    _GO2_TORQUE_HIP, _GO2_TORQUE_THIGH, _GO2_TORQUE_KNEE,
    _GO2_TORQUE_HIP, _GO2_TORQUE_THIGH, _GO2_TORQUE_KNEE,
    _GO2_TORQUE_HIP, _GO2_TORQUE_THIGH, _GO2_TORQUE_KNEE,
    _GO2_TORQUE_HIP, _GO2_TORQUE_THIGH, _GO2_TORQUE_KNEE,
)

_GO2_MAX_VEL_12 = (
    _GO2_VEL_HIP, _GO2_VEL_THIGH, _GO2_VEL_KNEE,
    _GO2_VEL_HIP, _GO2_VEL_THIGH, _GO2_VEL_KNEE,
    _GO2_VEL_HIP, _GO2_VEL_THIGH, _GO2_VEL_KNEE,
    _GO2_VEL_HIP, _GO2_VEL_THIGH, _GO2_VEL_KNEE,
)


def _kp12(hip: float, thigh: float, calf: float) -> tuple:
    return (hip, thigh, calf) * 4

def _kd12(hip: float, thigh: float, calf: float) -> tuple:
    return (hip, thigh, calf) * 4

_GO1_KP_HOLD  = _kp12(100.0, 100.0, 100.0)
_GO1_KD_HOLD  = _kd12(1.0,   1.0,   1.0)
_GO1_KP_RISE  = _kp12(100.0, 100.0, 100.0)
_GO1_KD_RISE  = _kd12(1.0,   1.0,   1.0)
_GO1_KP_STANCE = _kp12(100.0, 100.0, 100.0)
_GO1_KD_STANCE = _kd12(0.6,   0.6,   0.6)
_GO1_KP_SWING  = _kp12(20.0,  20.0,  20.0)
_GO1_KD_SWING  = _kd12(0.5,   0.5,   0.5)
_GO1_KP_JUMP   = _kp12(100.0, 100.0, 100.0)
_GO1_KD_JUMP   = _kd12(1.0,   1.0,   1.0)
_GO1_KP_LAND   = _kp12(40.0,  40.0,  40.0)
_GO1_KD_LAND   = _kd12(1.0,   1.0,   1.0)

_GO2_KP_HOLD  = _kp12(120.0, 120.0, 120.0)
_GO2_KD_HOLD  = _kd12(1.2,   1.2,   1.2)
_GO2_KP_RISE  = _kp12(120.0, 120.0, 120.0)
_GO2_KD_RISE  = _kd12(1.2,   1.2,   1.2)
_GO2_KP_STANCE = _kp12(120.0, 120.0, 120.0)
_GO2_KD_STANCE = _kd12(0.8,   0.8,   0.8)
_GO2_KP_SWING  = _kp12(25.0,  25.0,  25.0)
_GO2_KD_SWING  = _kd12(0.5,   0.5,   0.5)
_GO2_KP_JUMP   = _kp12(120.0, 120.0, 120.0)
_GO2_KD_JUMP   = _kd12(1.2,   1.2,   1.2)
_GO2_KP_LAND   = _kp12(50.0,  50.0,  50.0)
_GO2_KD_LAND   = _kd12(1.0,   1.0,   1.0)


_STAND    = (0.0, 0.9, -1.8) * 4
_SIT      = (0.0, 1.2, -2.5,
             0.0, 1.2, -2.5,
             0.0, 0.5, -1.2,
             0.0, 0.5, -1.2)
_PRONE    = (0.0, 1.5, -2.8) * 4
_CROUCH   = (0.0, 1.1, -2.2) * 4
_JUMP_EXT = (0.0, 0.7, -1.4) * 4
_LAND     = (0.0, 0.8, -1.6) * 4


@dataclass(frozen=True)
class MotorSchedule:
    """Per-joint timing and force control for one phase."""
    max_velocity: tuple[float, ...]
    max_torque: tuple[float, ...]
    delay_s: tuple[float, ...]
    max_pos_error: float = 0.08
    sync_arrival: bool = True


@dataclass(frozen=True)
class JointTarget:
    """12-DOF target with per-joint PD gains."""
    q12: tuple[float, ...]
    kp:  tuple[float, ...]
    kd:  tuple[float, ...]

    def as_array(self) -> np.ndarray:
        return np.array(self.q12, dtype=np.float32)

    def kp_array(self) -> np.ndarray:
        return np.array(self.kp, dtype=np.float32)

    def kd_array(self) -> np.ndarray:
        return np.array(self.kd, dtype=np.float32)


@dataclass(frozen=True)
class ActionPhase:
    """One phase of a multi-phase action with full motor scheduling."""
    name:           str
    duration_s:     float
    target:         JointTarget
    motor_schedule: MotorSchedule
    interpolation:  str = "quintic"


@dataclass(frozen=True)
class GaitAction:
    """Gait-based action using the gait engine."""
    gait_name:    str
    cmd_vx:       float
    cmd_vy:       float = 0.0
    cmd_yaw:      float = 0.0
    body_height:  float = 0.265
    step_height:  float = 0.08


@dataclass(frozen=True)
class ActionSpec:
    """Complete specification of one robot action."""
    name:           str
    description:    str
    robot:          str

    phases:         tuple[ActionPhase, ...] = ()
    gait:           GaitAction | None = None

    distance_m:     float = 0.0
    rotation_rad:   float = 0.0
    duration_s:     float = 0.0
    speed_ms:       float = 0.0

    max_pitch_rad:    float = 0.55
    max_roll_rad:     float = 0.60
    min_feet_contact: int   = 2

    hip_range:   tuple[float, float] = (-0.863, 0.863)
    thigh_range: tuple[float, float] = (-0.686, 4.501)
    knee_range:  tuple[float, float] = (-2.818, -0.888)

    @property
    def is_gait(self) -> bool:
        return self.gait is not None

    @property
    def is_phase(self) -> bool:
        return len(self.phases) > 0

    def total_duration(self) -> float:
        if self.duration_s > 0:
            return self.duration_s
        return sum(p.duration_s for p in self.phases)

    def clamp_joints(self, q12: np.ndarray) -> np.ndarray:
        q = q12.copy().astype(np.float32)
        n_legs = min(len(q), 12)
        for i in range(n_legs):
            jtype = i % 3
            if jtype == 0:
                q[i] = np.clip(q[i], self.hip_range[0], self.hip_range[1])
            elif jtype == 1:
                q[i] = np.clip(q[i], self.thigh_range[0], self.thigh_range[1])
            else:
                q[i] = np.clip(q[i], self.knee_range[0], self.knee_range[1])
        # Arm joints (indices 12+) are clamped by MuJoCo actuator ctrlrange
        return q


def _vel12(hip: float, thigh: float, calf: float) -> tuple:
    return (hip, thigh, calf) * 4

def _delay12(hip: float, thigh: float, calf: float) -> tuple:
    return (hip, thigh, calf) * 4

def _delay_by_leg(fl: tuple, fr: tuple, rl: tuple, rr: tuple) -> tuple:
    return fl + fr + rl + rr

_ZERO_DELAY = (0.0,) * 12


def _go1_actions() -> dict[str, ActionSpec]:
    actions: dict[str, ActionSpec] = {}

    actions["stand"] = ActionSpec(
        name="stand",
        description="Stand upright. All joints sync to arrive together.",
        robot="go1",
        phases=(
            ActionPhase(
                name="stand",
                duration_s=2.0,
                target=JointTarget(q12=_STAND, kp=_GO1_KP_HOLD, kd=_GO1_KD_HOLD),
                motor_schedule=MotorSchedule(
                    max_velocity=_vel12(hip=0.5, thigh=0.5, calf=0.5),
                    max_torque=_GO1_MAX_TORQUE_12,
                    delay_s=_ZERO_DELAY,
                    max_pos_error=1.0,
                    sync_arrival=True,
                ),
            ),
        ),
        duration_s=2.0,
        max_pitch_rad=0.55, max_roll_rad=0.60, min_feet_contact=3,
        hip_range=_GO1_HIP_RANGE, thigh_range=_GO1_THIGH_RANGE, knee_range=_GO1_KNEE_RANGE,
    )

    actions["stand_up"] = ActionSpec(
        name="stand_up",
        description="Rise from prone. Calves tuck first, then thighs push.",
        robot="go1",
        phases=(
            ActionPhase(
                name="tuck",
                duration_s=1.5,
                target=JointTarget(
                    q12=(0.0, 1.3, -2.6) * 4,
                    kp=_GO1_KP_RISE, kd=_GO1_KD_RISE,
                ),
                motor_schedule=MotorSchedule(
                    max_velocity=_vel12(hip=0.3, thigh=0.4, calf=0.5),
                    max_torque=_GO1_MAX_TORQUE_12,
                    delay_s=_delay12(hip=0.3, thigh=0.2, calf=0.0),
                    max_pos_error=1.0,
                    sync_arrival=False,
                ),
            ),
            ActionPhase(
                name="half_rise",
                duration_s=2.5,
                target=JointTarget(
                    q12=(0.0, 1.1, -2.2) * 4,
                    kp=_GO1_KP_RISE, kd=_GO1_KD_RISE,
                ),
                motor_schedule=MotorSchedule(
                    max_velocity=_vel12(hip=0.3, thigh=0.4, calf=0.5),
                    max_torque=_GO1_MAX_TORQUE_12,
                    delay_s=_ZERO_DELAY,
                    max_pos_error=1.0,
                    sync_arrival=True,
                ),
            ),
            ActionPhase(
                name="full_stand",
                duration_s=2.5,
                target=JointTarget(q12=_STAND, kp=_GO1_KP_HOLD, kd=_GO1_KD_HOLD),
                motor_schedule=MotorSchedule(
                    max_velocity=_vel12(hip=0.3, thigh=0.5, calf=0.6),
                    max_torque=_GO1_MAX_TORQUE_12,
                    delay_s=_ZERO_DELAY,
                    max_pos_error=1.0,
                    sync_arrival=True,
                ),
            ),
        ),
        duration_s=6.5,
        max_pitch_rad=0.55, max_roll_rad=0.60, min_feet_contact=2,
        hip_range=_GO1_HIP_RANGE, thigh_range=_GO1_THIGH_RANGE, knee_range=_GO1_KNEE_RANGE,
    )

    actions["sit"] = ActionSpec(
        name="sit",
        description="Sit down. Rear legs fold first, then front extend.",
        robot="go1",
        phases=(
            ActionPhase(
                name="rear_fold",
                duration_s=1.5,
                target=JointTarget(
                    q12=(0.0, 0.9, -1.8,
                         0.0, 0.9, -1.8,
                         0.0, 0.5, -1.2,
                         0.0, 0.5, -1.2),
                    kp=_GO1_KP_HOLD, kd=_GO1_KD_HOLD,
                ),
                motor_schedule=MotorSchedule(
                    max_velocity=_vel12(hip=0.3, thigh=0.6, calf=0.8),
                    max_torque=_GO1_MAX_TORQUE_12,
                    delay_s=_delay_by_leg(
                        fl=(0.8, 0.8, 0.8),
                        fr=(0.8, 0.8, 0.8),
                        rl=(0.0, 0.0, 0.0),
                        rr=(0.0, 0.0, 0.0),
                    ),
                    max_pos_error=1.0,
                    sync_arrival=False,
                ),
            ),
            ActionPhase(
                name="front_extend",
                duration_s=1.5,
                target=JointTarget(q12=_SIT, kp=_GO1_KP_HOLD, kd=_GO1_KD_HOLD),
                motor_schedule=MotorSchedule(
                    max_velocity=_vel12(hip=0.3, thigh=0.5, calf=0.6),
                    max_torque=_GO1_MAX_TORQUE_12,
                    delay_s=_ZERO_DELAY,
                    max_pos_error=1.0,
                    sync_arrival=True,
                ),
            ),
        ),
        duration_s=3.0,
        max_pitch_rad=0.70, max_roll_rad=0.60, min_feet_contact=2,
        hip_range=_GO1_HIP_RANGE, thigh_range=_GO1_THIGH_RANGE, knee_range=_GO1_KNEE_RANGE,
    )

    actions["lie_down"] = ActionSpec(
        name="lie_down",
        description="Lower to prone. Knees fold first, then thighs.",
        robot="go1",
        phases=(
            ActionPhase(
                name="lower",
                duration_s=2.0,
                target=JointTarget(
                    q12=(0.0, 1.2, -2.4) * 4,
                    kp=_GO1_KP_HOLD, kd=_GO1_KD_HOLD,
                ),
                motor_schedule=MotorSchedule(
                    max_velocity=_vel12(hip=0.3, thigh=0.5, calf=0.7),
                    max_torque=_GO1_MAX_TORQUE_12,
                    delay_s=_delay12(hip=0.3, thigh=0.15, calf=0.0),
                    max_pos_error=1.0,
                    sync_arrival=False,
                ),
            ),
            ActionPhase(
                name="prone",
                duration_s=1.5,
                target=JointTarget(q12=_PRONE, kp=_GO1_KP_HOLD, kd=_GO1_KD_HOLD),
                motor_schedule=MotorSchedule(
                    max_velocity=_vel12(hip=0.2, thigh=0.4, calf=0.5),
                    max_torque=_GO1_MAX_TORQUE_12,
                    delay_s=_ZERO_DELAY,
                    max_pos_error=1.0,
                    sync_arrival=True,
                ),
            ),
        ),
        duration_s=3.5,
        max_pitch_rad=0.70, max_roll_rad=0.70, min_feet_contact=0,
        hip_range=_GO1_HIP_RANGE, thigh_range=_GO1_THIGH_RANGE, knee_range=_GO1_KNEE_RANGE,
    )

    actions["jump"] = ActionSpec(
        name="jump",
        description="Small hop. Controlled crouch, gentle launch, stable landing.",
        robot="go1",
        phases=(
            ActionPhase(
                name="crouch",
                duration_s=0.6,
                target=JointTarget(q12=_CROUCH, kp=_GO1_KP_HOLD, kd=_GO1_KD_HOLD),
                motor_schedule=MotorSchedule(
                    max_velocity=_vel12(hip=1.0, thigh=2.0, calf=2.5),
                    max_torque=_GO1_MAX_TORQUE_12,
                    delay_s=_ZERO_DELAY,
                    max_pos_error=1.0,
                    sync_arrival=True,
                ),
            ),
            ActionPhase(
                name="hold_crouch",
                duration_s=0.3,
                target=JointTarget(q12=_CROUCH, kp=_GO1_KP_HOLD, kd=_GO1_KD_HOLD),
                motor_schedule=MotorSchedule(
                    max_velocity=(0.1,) * 12,
                    max_torque=_GO1_MAX_TORQUE_12,
                    delay_s=_ZERO_DELAY,
                    max_pos_error=1.0,
                    sync_arrival=True,
                ),
                interpolation="hold",
            ),
            # Front legs fire first, rear delayed 0.06s to prevent nose-up pitch
            ActionPhase(
                name="launch",
                duration_s=0.3,
                target=JointTarget(q12=_JUMP_EXT, kp=_GO1_KP_JUMP, kd=_GO1_KD_JUMP),
                motor_schedule=MotorSchedule(
                    max_velocity=_vel12(hip=4.0, thigh=5.0, calf=5.0),
                    max_torque=_GO1_MAX_TORQUE_12,
                    delay_s=(0.0,0.0,0.0, 0.0,0.0,0.0, 0.06,0.06,0.06, 0.06,0.06,0.06),
                    max_pos_error=1.0,
                    sync_arrival=True,
                ),
                interpolation="linear",
            ),
            # Red+blue legs (FL/FR) fire again after initial launch
            ActionPhase(
                name="rear_kick",
                duration_s=0.3,
                target=JointTarget(q12=_JUMP_EXT, kp=_GO1_KP_JUMP, kd=_GO1_KD_JUMP),
                motor_schedule=MotorSchedule(
                    max_velocity=(
                        8.0, 10.0, 10.0,  # FL (red+blue) — max push
                        8.0, 10.0, 10.0,  # FR (red+blue) — max push
                        0.1, 0.1, 0.1,    # RL (orange+aqua) — hold
                        0.1, 0.1, 0.1,    # RR (orange+aqua) — hold
                    ),
                    max_torque=_GO1_MAX_TORQUE_12,
                    delay_s=_ZERO_DELAY,
                    max_pos_error=1.0,
                    sync_arrival=True,
                ),
                interpolation="linear",
            ),
            ActionPhase(
                name="flight",
                duration_s=0.3,
                target=JointTarget(q12=_LAND, kp=_GO1_KP_HOLD, kd=_GO1_KD_HOLD),
                motor_schedule=MotorSchedule(
                    max_velocity=_vel12(hip=5.0, thigh=6.0, calf=6.0),
                    max_torque=_GO1_MAX_TORQUE_12,
                    delay_s=_ZERO_DELAY,
                    max_pos_error=1.0,
                    sync_arrival=True,
                ),
            ),
            ActionPhase(
                name="land",
                duration_s=0.8,
                target=JointTarget(q12=_STAND, kp=_GO1_KP_HOLD, kd=_GO1_KD_HOLD),
                motor_schedule=MotorSchedule(
                    max_velocity=_vel12(hip=2.0, thigh=3.0, calf=3.0),
                    max_torque=_GO1_MAX_TORQUE_12,
                    delay_s=_ZERO_DELAY,
                    max_pos_error=1.0,
                    sync_arrival=True,
                ),
            ),
        ),
        duration_s=2.55,
        max_pitch_rad=1.20, max_roll_rad=1.20, min_feet_contact=0,
        hip_range=_GO1_HIP_RANGE, thigh_range=_GO1_THIGH_RANGE, knee_range=_GO1_KNEE_RANGE,
    )

    # actions["flip"] = ActionSpec(
    #     name="flip",
    #     description="Front flip. Front legs launch while back legs compress, then back legs explode.",
    #     robot="go1",
    #     phases=(
    #         ActionPhase(
    #             name="crouch",
    #             duration_s=0.6,
    #             target=JointTarget(q12=_CROUCH, kp=_GO1_KP_HOLD, kd=_GO1_KD_HOLD),
    #             motor_schedule=MotorSchedule(
    #                 max_velocity=_vel12(hip=1.0, thigh=2.0, calf=2.5),
    #                 max_torque=_GO1_MAX_TORQUE_12,
    #                 delay_s=_ZERO_DELAY,
    #                 max_pos_error=1.0,
    #                 sync_arrival=True,
    #             ),
    #         ),
    #         ActionPhase(
    #             name="hold_crouch",
    #             duration_s=0.3,
    #             target=JointTarget(q12=_CROUCH, kp=_GO1_KP_HOLD, kd=_GO1_KD_HOLD),
    #             motor_schedule=MotorSchedule(
    #                 max_velocity=(0.1,) * 12,
    #                 max_torque=_GO1_MAX_TORQUE_12,
    #                 delay_s=_ZERO_DELAY,
    #                 max_pos_error=1.0,
    #                 sync_arrival=True,
    #             ),
    #             interpolation="hold",
    #         ),
    #         # MASSIVE launch — all four legs, max extension, max speed
    #         ActionPhase(
    #             name="launch",
    #             duration_s=0.2,
    #             target=JointTarget(
    #                 q12=(0.0, 0.2, -0.4) * 4,  # near full extension
    #                 kp=_GO1_KP_JUMP, kd=_GO1_KD_JUMP,
    #             ),
    #             motor_schedule=MotorSchedule(
    #                 max_velocity=_vel12(hip=12.0, thigh=15.0, calf=15.0),
    #                 max_torque=_GO1_MAX_TORQUE_12,
    #                 delay_s=_ZERO_DELAY,
    #                 max_pos_error=1.0,
    #                 sync_arrival=True,
    #             ),
    #             interpolation="linear",
    #         ),
    #         # ALL legs buckle in completely — hold for 2 seconds
    #         ActionPhase(
    #             name="tuck",
    #             duration_s=2.0,
    #             target=JointTarget(
    #                 q12=(0.0, 1.3, -2.6) * 4,  # fully buckled
    #                 kp=_GO1_KP_JUMP, kd=_GO1_KD_JUMP,
    #             ),
    #             motor_schedule=MotorSchedule(
    #                 max_velocity=_vel12(hip=12.0, thigh=15.0, calf=15.0),
    #                 max_torque=_GO1_MAX_TORQUE_12,
    #                 delay_s=_ZERO_DELAY,
    #                 max_pos_error=1.0,
    #                 sync_arrival=True,
    #             ),
    #             interpolation="linear",
    #         ),
    #         ActionPhase(
    #             name="land",
    #             duration_s=0.8,
    #             target=JointTarget(q12=_STAND, kp=_GO1_KP_HOLD, kd=_GO1_KD_HOLD),
    #             motor_schedule=MotorSchedule(
    #                 max_velocity=_vel12(hip=2.0, thigh=3.0, calf=3.0),
    #                 max_torque=_GO1_MAX_TORQUE_12,
    #                 delay_s=_ZERO_DELAY,
    #                 max_pos_error=1.0,
    #                 sync_arrival=True,
    #             ),
    #         ),
    #     ),
    #     duration_s=4.1,
    #     max_pitch_rad=3.20, max_roll_rad=3.20, min_feet_contact=0,
    #     hip_range=_GO1_HIP_RANGE, thigh_range=_GO1_THIGH_RANGE, knee_range=_GO1_KNEE_RANGE,
    # )

    actions["walk_forward"] = ActionSpec(
        name="walk_forward",
        description="Walk forward 1m. 4-beat gait, >=3 feet on ground.",
        robot="go1",
        gait=GaitAction(gait_name="walk", cmd_vx=-0.25, body_height=0.26, step_height=0.10),
        distance_m=1.0, speed_ms=0.25, duration_s=4.0,
        min_feet_contact=3,
        hip_range=_GO1_HIP_RANGE, thigh_range=_GO1_THIGH_RANGE, knee_range=_GO1_KNEE_RANGE,
    )

    actions["walk_backward"] = ActionSpec(
        name="walk_backward",
        description="Walk backward 0.5m. Slow, stable.",
        robot="go1",
        gait=GaitAction(gait_name="walk", cmd_vx=0.10, body_height=0.26, step_height=0.06),
        distance_m=0.5, speed_ms=0.10, duration_s=5.0,
        min_feet_contact=3,
        hip_range=_GO1_HIP_RANGE, thigh_range=_GO1_THIGH_RANGE, knee_range=_GO1_KNEE_RANGE,
    )

    actions["trot_forward"] = ActionSpec(
        name="trot_forward",
        description="Trot forward 1m. Diagonal pairs, fast.",
        robot="go1",
        gait=GaitAction(gait_name="trot", cmd_vx=-0.3, body_height=0.26, step_height=0.08),
        distance_m=1.0, speed_ms=0.3, duration_s=3.33,
        min_feet_contact=2,
        hip_range=_GO1_HIP_RANGE, thigh_range=_GO1_THIGH_RANGE, knee_range=_GO1_KNEE_RANGE,
    )

    actions["crawl_forward"] = ActionSpec(
        name="crawl_forward",
        description="Crawl forward 0.5m. Ultra-stable, >=3 feet on ground.",
        robot="go1",
        gait=GaitAction(gait_name="crawl", cmd_vx=-0.08, body_height=0.24, step_height=0.10),
        distance_m=0.5, speed_ms=0.08, duration_s=6.25,
        min_feet_contact=3,
        hip_range=_GO1_HIP_RANGE, thigh_range=_GO1_THIGH_RANGE, knee_range=_GO1_KNEE_RANGE,
    )

    actions["turn_left"] = ActionSpec(
        name="turn_left",
        description="Turn left in place.",
        robot="go1",
        gait=GaitAction(gait_name="walk", cmd_vx=-0.03, cmd_yaw=0.5, body_height=0.26, step_height=0.06),
        rotation_rad=math.pi / 2, duration_s=3.1,
        min_feet_contact=3,
        hip_range=_GO1_HIP_RANGE, thigh_range=_GO1_THIGH_RANGE, knee_range=_GO1_KNEE_RANGE,
    )

    actions["turn_right"] = ActionSpec(
        name="turn_right",
        description="Turn right in place.",
        robot="go1",
        gait=GaitAction(gait_name="walk", cmd_vx=-0.03, cmd_yaw=-0.5, body_height=0.26, step_height=0.06),
        rotation_rad=math.pi / 2, duration_s=3.1,
        min_feet_contact=3,
        hip_range=_GO1_HIP_RANGE, thigh_range=_GO1_THIGH_RANGE, knee_range=_GO1_KNEE_RANGE,
    )

    actions["precision_turn_left"] = ActionSpec(
        name="precision_turn_left",
        description="Turn left — compressed body, tiny precise steps for tight spaces.",
        robot="go1",
        gait=GaitAction(gait_name="precision_turn", cmd_vx=-0.02, cmd_yaw=0.35, body_height=0.20, step_height=0.03),
        rotation_rad=math.pi / 2, duration_s=4.0,
        min_feet_contact=3,
        hip_range=_GO1_HIP_RANGE, thigh_range=_GO1_THIGH_RANGE, knee_range=_GO1_KNEE_RANGE,
    )

    actions["precision_turn_right"] = ActionSpec(
        name="precision_turn_right",
        description="Turn right — compressed body, tiny precise steps for tight spaces.",
        robot="go1",
        gait=GaitAction(gait_name="precision_turn", cmd_vx=-0.02, cmd_yaw=-0.35, body_height=0.20, step_height=0.03),
        rotation_rad=math.pi / 2, duration_s=4.0,
        min_feet_contact=3,
        hip_range=_GO1_HIP_RANGE, thigh_range=_GO1_THIGH_RANGE, knee_range=_GO1_KNEE_RANGE,
    )

    actions["climb_step"] = ActionSpec(
        name="climb_step",
        description="Climb single stair step (max 20cm rise).",
        robot="go1",
        gait=GaitAction(gait_name="stair_crawl", cmd_vx=-0.06, body_height=0.24, step_height=0.15),
        distance_m=0.3, speed_ms=0.06, duration_s=5.0,
        min_feet_contact=3,
        hip_range=_GO1_HIP_RANGE, thigh_range=_GO1_THIGH_RANGE, knee_range=_GO1_KNEE_RANGE,
    )

    actions["side_step_left"] = ActionSpec(
        name="side_step_left",
        description="Step left 0.3m.",
        robot="go1",
        gait=GaitAction(gait_name="walk", cmd_vx=0.0, cmd_vy=-0.10, body_height=0.28, step_height=0.12),
        distance_m=0.3, speed_ms=0.10, duration_s=3.0,
        min_feet_contact=3,
        hip_range=_GO1_HIP_RANGE, thigh_range=_GO1_THIGH_RANGE, knee_range=_GO1_KNEE_RANGE,
    )

    actions["side_step_right"] = ActionSpec(
        name="side_step_right",
        description="Step right 0.3m.",
        robot="go1",
        gait=GaitAction(gait_name="walk", cmd_vx=0.0, cmd_vy=0.10, body_height=0.28, step_height=0.12),
        distance_m=0.3, speed_ms=0.10, duration_s=3.0,
        min_feet_contact=3,
        hip_range=_GO1_HIP_RANGE, thigh_range=_GO1_THIGH_RANGE, knee_range=_GO1_KNEE_RANGE,
    )

    actions["pace_forward"] = ActionSpec(
        name="pace_forward",
        description="Pace forward 1m. Lateral pairs, fast.",
        robot="go1",
        gait=GaitAction(gait_name="pace", cmd_vx=-0.4, body_height=0.28, step_height=0.10),
        distance_m=1.0, speed_ms=0.4, duration_s=2.5,
        min_feet_contact=2,
        hip_range=_GO1_HIP_RANGE, thigh_range=_GO1_THIGH_RANGE, knee_range=_GO1_KNEE_RANGE,
    )

    actions["bound_forward"] = ActionSpec(
        name="bound_forward",
        description="Bound forward 1m. Front/rear pairs, flat ground only.",
        robot="go1",
        gait=GaitAction(gait_name="bound", cmd_vx=-0.5, body_height=0.28, step_height=0.08),
        distance_m=1.0, speed_ms=0.5, duration_s=2.0,
        min_feet_contact=2,
        hip_range=_GO1_HIP_RANGE, thigh_range=_GO1_THIGH_RANGE, knee_range=_GO1_KNEE_RANGE,
    )

    # ── Rear up (stand on hind legs) ──
    # Leg mapping: FL/FR(0-5) = BACK at +x, RL/RR(6-11) = FRONT at -x (head).
    # Deep crouch → explosive push → stand on back legs with head high.
    actions["rear_up"] = ActionSpec(
        name="rear_up", description="Stand on hind legs, front paws in air.", robot="go1",
        phases=(
            ActionPhase(
                name="crouch", duration_s=0.6,
                target=JointTarget(
                    # Deep crouch to load spring energy in back legs
                    q12=(0.0, 1.3, -2.6,  0.0, 1.3, -2.6,
                         0.0, 1.1, -2.2,  0.0, 1.1, -2.2),
                    kp=_GO1_KP_HOLD, kd=_GO1_KD_HOLD,
                ),
                motor_schedule=MotorSchedule(
                    max_velocity=_vel12(2.0, 3.0, 3.0),
                    max_torque=_GO1_MAX_TORQUE_12,
                    delay_s=_ZERO_DELAY, max_pos_error=1.0, sync_arrival=True,
                ),
            ),
            ActionPhase(
                name="hold_crouch", duration_s=0.2,
                target=JointTarget(
                    q12=(0.0, 1.3, -2.6,  0.0, 1.3, -2.6,
                         0.0, 1.1, -2.2,  0.0, 1.1, -2.2),
                    kp=_GO1_KP_HOLD, kd=_GO1_KD_HOLD,
                ),
                motor_schedule=MotorSchedule(
                    max_velocity=(0.1,) * 12,
                    max_torque=_GO1_MAX_TORQUE_12,
                    delay_s=_ZERO_DELAY, max_pos_error=1.0, sync_arrival=True,
                ),
                interpolation="hold",
            ),
            ActionPhase(
                name="push_up", duration_s=0.4,
                target=JointTarget(
                    # FL/FR(back): extend hard — pushes rear up
                    # RL/RR(front): fold — lifts head
                    q12=(0.0, 0.0, -0.89,  0.0, 0.0, -0.89,
                         0.0, -0.5, -2.5,  0.0, -0.5, -2.5),
                    kp=_GO1_KP_JUMP, kd=_GO1_KD_JUMP,
                ),
                motor_schedule=MotorSchedule(
                    max_velocity=_vel12(5.0, 12.0, 12.0),
                    max_torque=_GO1_MAX_TORQUE_12,
                    # Back legs fire first to tilt body
                    delay_s=_delay_by_leg((0.0,)*3, (0.0,)*3, (0.08,)*3, (0.08,)*3),
                    max_pos_error=1.0, sync_arrival=False,
                ),
                interpolation="linear",
            ),
            ActionPhase(
                name="hold_rear", duration_s=2.0,
                target=JointTarget(
                    q12=(0.0, 0.1, -0.89,  0.0, 0.1, -0.89,
                         0.0, -0.6, -2.7,  0.0, -0.6, -2.7),
                    kp=_GO1_KP_HOLD, kd=_GO1_KD_HOLD,
                ),
                motor_schedule=MotorSchedule(
                    max_velocity=_vel12(0.5, 0.5, 0.5),
                    max_torque=_GO1_MAX_TORQUE_12,
                    delay_s=_ZERO_DELAY, max_pos_error=1.0, sync_arrival=True,
                ),
            ),
            ActionPhase(
                name="tuck_front", duration_s=0.8,
                target=JointTarget(
                    q12=(0.0, 0.1, -0.89,  0.0, 0.1, -0.89,
                         0.0, 1.5, -2.818,  0.0, 1.5, -2.818),
                    kp=_GO1_KP_HOLD, kd=_GO1_KD_HOLD,
                ),
                motor_schedule=MotorSchedule(
                    max_velocity=_vel12(1.5, 2.0, 2.0),
                    max_torque=_GO1_MAX_TORQUE_12,
                    delay_s=_ZERO_DELAY, max_pos_error=1.0, sync_arrival=True,
                ),
            ),
            ActionPhase(
                name="lower_body", duration_s=1.5,
                target=JointTarget(
                    q12=(0.0, 1.3, -2.6,  0.0, 1.3, -2.6,
                         0.0, 1.3, -2.6,  0.0, 1.3, -2.6),
                    kp=_GO1_KP_HOLD, kd=_GO1_KD_HOLD,
                ),
                motor_schedule=MotorSchedule(
                    max_velocity=_vel12(0.5, 0.6, 0.6),
                    max_torque=_GO1_MAX_TORQUE_12,
                    delay_s=_ZERO_DELAY, max_pos_error=1.0, sync_arrival=True,
                ),
            ),
            ActionPhase(
                name="return_stand", duration_s=2.0,
                target=JointTarget(q12=_STAND, kp=_GO1_KP_HOLD, kd=_GO1_KD_HOLD),
                motor_schedule=MotorSchedule(
                    max_velocity=_vel12(0.5, 0.6, 0.6),
                    max_torque=_GO1_MAX_TORQUE_12,
                    delay_s=_ZERO_DELAY, max_pos_error=1.0, sync_arrival=True,
                ),
            ),
        ),
        duration_s=7.6, max_pitch_rad=1.5, max_roll_rad=0.60, min_feet_contact=0,
        hip_range=_GO1_HIP_RANGE, thigh_range=_GO1_THIGH_RANGE, knee_range=_GO1_KNEE_RANGE,
    )

    # ── Shake hand (wave front-left paw) ──
    # Leg mapping: FL/FR(0-5) = BACK at +x, RL/RR(6-11) = FRONT at -x (head).
    # Lift RL (front-left at -x), lean toward RR (front-right) for stability.
    actions["shake_hand"] = ActionSpec(
        name="shake_hand", description="Lift and wave front-left paw.", robot="go1",
        phases=(
            ActionPhase(
                name="lean_right", duration_s=0.8,
                target=JointTarget(
                    # FL/FR(back): stable support
                    # RL(front-L): hip out to prep lift; RR(front-R): hip in, lower for stability
                    q12=(0.1, 0.9, -1.8,  -0.1, 0.9, -1.8,
                         0.3, 0.9, -1.8,  -0.2, 0.8, -1.7),
                    kp=_GO1_KP_HOLD, kd=_GO1_KD_HOLD,
                ),
                motor_schedule=MotorSchedule(
                    max_velocity=_vel12(1.0, 1.0, 1.0),
                    max_torque=_GO1_MAX_TORQUE_12,
                    delay_s=_ZERO_DELAY, max_pos_error=1.0, sync_arrival=True,
                ),
            ),
            ActionPhase(
                name="lift_paw", duration_s=0.6,
                target=JointTarget(
                    # RL(front-L): lift! thigh forward, calf folded
                    q12=(0.1, 0.9, -1.8,  -0.1, 0.9, -1.8,
                         0.4, -0.2, -2.0,  -0.2, 0.8, -1.7),
                    kp=_GO1_KP_HOLD, kd=_GO1_KD_HOLD,
                ),
                motor_schedule=MotorSchedule(
                    max_velocity=_vel12(2.0, 3.0, 3.0),
                    max_torque=_GO1_MAX_TORQUE_12,
                    delay_s=_ZERO_DELAY, max_pos_error=1.0, sync_arrival=True,
                ),
            ),
            ActionPhase(
                name="wave_down", duration_s=0.4,
                target=JointTarget(
                    q12=(0.1, 0.9, -1.8,  -0.1, 0.9, -1.8,
                         0.4, 0.3, -1.8,  -0.2, 0.8, -1.7),
                    kp=_GO1_KP_HOLD, kd=_GO1_KD_HOLD,
                ),
                motor_schedule=MotorSchedule(
                    max_velocity=_vel12(2.0, 4.0, 4.0),
                    max_torque=_GO1_MAX_TORQUE_12,
                    delay_s=_ZERO_DELAY, max_pos_error=1.0, sync_arrival=True,
                ),
            ),
            ActionPhase(
                name="wave_up", duration_s=0.4,
                target=JointTarget(
                    q12=(0.1, 0.9, -1.8,  -0.1, 0.9, -1.8,
                         0.4, -0.3, -2.2,  -0.2, 0.8, -1.7),
                    kp=_GO1_KP_HOLD, kd=_GO1_KD_HOLD,
                ),
                motor_schedule=MotorSchedule(
                    max_velocity=_vel12(2.0, 4.0, 4.0),
                    max_torque=_GO1_MAX_TORQUE_12,
                    delay_s=_ZERO_DELAY, max_pos_error=1.0, sync_arrival=True,
                ),
            ),
            ActionPhase(
                name="wave_down_2", duration_s=0.4,
                target=JointTarget(
                    q12=(0.1, 0.9, -1.8,  -0.1, 0.9, -1.8,
                         0.4, 0.3, -1.8,  -0.2, 0.8, -1.7),
                    kp=_GO1_KP_HOLD, kd=_GO1_KD_HOLD,
                ),
                motor_schedule=MotorSchedule(
                    max_velocity=_vel12(2.0, 4.0, 4.0),
                    max_torque=_GO1_MAX_TORQUE_12,
                    delay_s=_ZERO_DELAY, max_pos_error=1.0, sync_arrival=True,
                ),
            ),
            ActionPhase(
                name="return_stand", duration_s=0.8,
                target=JointTarget(q12=_STAND, kp=_GO1_KP_HOLD, kd=_GO1_KD_HOLD),
                motor_schedule=MotorSchedule(
                    max_velocity=_vel12(2.0, 2.0, 2.0),
                    max_torque=_GO1_MAX_TORQUE_12,
                    delay_s=_ZERO_DELAY, max_pos_error=1.0, sync_arrival=True,
                ),
            ),
        ),
        duration_s=3.4, max_pitch_rad=0.55, max_roll_rad=0.80, min_feet_contact=2,
        hip_range=_GO1_HIP_RANGE, thigh_range=_GO1_THIGH_RANGE, knee_range=_GO1_KNEE_RANGE,
    )

    # ── Rear kick (horse-style buck) ──
    # Leg mapping: FL/FR(0-5) = BACK at +x, RL/RR(6-11) = FRONT at -x (head).
    # Strategy: shift weight forward, front legs brace as pivot, back legs
    # explosively extend to kick rear up, then retract and return.
    actions["rear_kick"] = ActionSpec(
        name="rear_kick", description="Dramatic rear kick / buck.", robot="go1",
        phases=(
            ActionPhase(
                name="weight_forward", duration_s=0.5,
                target=JointTarget(
                    # FL/FR(back): deep crouch — coiled to kick
                    # RL/RR(front): extend slightly — brace and pivot
                    q12=(0.0, 1.5, -2.75,  0.0, 1.5, -2.75,
                         0.0, 0.6, -1.3,  0.0, 0.6, -1.3),
                    kp=_GO1_KP_HOLD, kd=_GO1_KD_HOLD,
                ),
                motor_schedule=MotorSchedule(
                    max_velocity=_vel12(2.0, 3.0, 3.0),
                    max_torque=_GO1_MAX_TORQUE_12,
                    delay_s=_ZERO_DELAY, max_pos_error=1.0, sync_arrival=True,
                ),
            ),
            ActionPhase(
                name="hold_load", duration_s=0.2,
                target=JointTarget(
                    q12=(0.0, 1.5, -2.75,  0.0, 1.5, -2.75,
                         0.0, 0.6, -1.3,  0.0, 0.6, -1.3),
                    kp=_GO1_KP_HOLD, kd=_GO1_KD_HOLD,
                ),
                motor_schedule=MotorSchedule(
                    max_velocity=(0.1,) * 12,
                    max_torque=_GO1_MAX_TORQUE_12,
                    delay_s=_ZERO_DELAY, max_pos_error=1.0, sync_arrival=True,
                ),
                interpolation="hold",
            ),
            ActionPhase(
                name="kick_up", duration_s=0.25,
                target=JointTarget(
                    # FL/FR(back): strong extension — kick rear up
                    # RL/RR(front): crouched — weight on front, stay grounded
                    q12=(0.0, 0.0, -0.95,  0.0, 0.0, -0.95,
                         0.0, 0.9, -1.8,  0.0, 0.9, -1.8),
                    kp=_GO1_KP_JUMP, kd=_GO1_KD_JUMP,
                ),
                motor_schedule=MotorSchedule(
                    max_velocity=_vel12(5.0, 20.0, 20.0),
                    max_torque=_GO1_MAX_TORQUE_12,
                    delay_s=_ZERO_DELAY, max_pos_error=1.0, sync_arrival=False,
                ),
                interpolation="linear",
            ),
            ActionPhase(
                name="hold_kick", duration_s=0.6,
                target=JointTarget(
                    q12=(0.0, 0.0, -0.95,  0.0, 0.0, -0.95,
                         0.0, 0.9, -1.8,  0.0, 0.9, -1.8),
                    kp=_GO1_KP_HOLD, kd=_GO1_KD_HOLD,
                ),
                motor_schedule=MotorSchedule(
                    max_velocity=_vel12(0.5, 0.5, 0.5),
                    max_torque=_GO1_MAX_TORQUE_12,
                    delay_s=_ZERO_DELAY, max_pos_error=1.0, sync_arrival=True,
                ),
            ),
            ActionPhase(
                name="retract", duration_s=1.5,
                target=JointTarget(
                    # Slowly lower rear — front stays braced
                    q12=(0.0, 1.1, -2.2,  0.0, 1.1, -2.2,
                         0.0, 0.9, -1.8,  0.0, 0.9, -1.8),
                    kp=_GO1_KP_HOLD, kd=_GO1_KD_HOLD,
                ),
                motor_schedule=MotorSchedule(
                    max_velocity=_vel12(1.0, 1.5, 1.5),
                    max_torque=_GO1_MAX_TORQUE_12,
                    delay_s=_ZERO_DELAY, max_pos_error=1.0, sync_arrival=True,
                ),
            ),
            ActionPhase(
                name="return_stand", duration_s=1.0,
                target=JointTarget(q12=_STAND, kp=_GO1_KP_HOLD, kd=_GO1_KD_HOLD),
                motor_schedule=MotorSchedule(
                    max_velocity=_vel12(1.0, 1.5, 1.5),
                    max_torque=_GO1_MAX_TORQUE_12,
                    delay_s=_ZERO_DELAY, max_pos_error=1.0, sync_arrival=True,
                ),
            ),
        ),
        duration_s=4.05, max_pitch_rad=1.5, max_roll_rad=0.60, min_feet_contact=0,
        hip_range=_GO1_HIP_RANGE, thigh_range=_GO1_THIGH_RANGE, knee_range=_GO1_KNEE_RANGE,
    )

    return actions


_GAIN_MAP = {
    _GO1_KP_HOLD:   (_GO2_KP_HOLD,   _GO2_KD_HOLD),
    _GO1_KP_RISE:   (_GO2_KP_RISE,   _GO2_KD_RISE),
    _GO1_KP_STANCE: (_GO2_KP_STANCE, _GO2_KD_STANCE),
    _GO1_KP_SWING:  (_GO2_KP_SWING,  _GO2_KD_SWING),
    _GO1_KP_JUMP:   (_GO2_KP_JUMP,   _GO2_KD_JUMP),
    _GO1_KP_LAND:   (_GO2_KP_LAND,   _GO2_KD_LAND),
}

def _go2_actions() -> dict[str, ActionSpec]:
    go1 = _go1_actions()
    go2: dict[str, ActionSpec] = {}

    for name, spec in go1.items():
        fields = {
            "robot": "go2",
            "hip_range": _GO2_HIP_RANGE,
            "thigh_range": _GO2_THIGH_RANGE,
            "knee_range": _GO2_KNEE_RANGE,
            "max_pitch_rad": min(spec.max_pitch_rad + 0.05, 0.85),
            "max_roll_rad": min(spec.max_roll_rad + 0.05, 0.85),
        }

        if spec.phases:
            new_phases = []
            for phase in spec.phases:
                new_kp, new_kd = _GAIN_MAP.get(phase.target.kp,
                                                (phase.target.kp, phase.target.kd))
                new_phases.append(ActionPhase(
                    name=phase.name,
                    duration_s=phase.duration_s,
                    target=JointTarget(q12=phase.target.q12, kp=new_kp, kd=new_kd),
                    motor_schedule=MotorSchedule(
                        max_velocity=phase.motor_schedule.max_velocity,
                        max_torque=_GO2_MAX_TORQUE_12,
                        delay_s=phase.motor_schedule.delay_s,
                        max_pos_error=phase.motor_schedule.max_pos_error,
                        sync_arrival=phase.motor_schedule.sync_arrival,
                    ),
                    interpolation=phase.interpolation,
                ))
            fields["phases"] = tuple(new_phases)

        if spec.gait:
            fields["gait"] = GaitAction(
                gait_name=spec.gait.gait_name,
                cmd_vx=spec.gait.cmd_vx, cmd_vy=spec.gait.cmd_vy,
                cmd_yaw=spec.gait.cmd_yaw,
                body_height=spec.gait.body_height,
                step_height=spec.gait.step_height,
            )

        speed_factor = 1.2
        new_speed = spec.speed_ms * speed_factor if spec.speed_ms > 0 else 0.0
        new_dur = spec.distance_m / new_speed if (new_speed > 0 and spec.distance_m > 0) else spec.duration_s
        fields["speed_ms"] = new_speed
        if spec.distance_m > 0 and new_speed > 0:
            fields["duration_s"] = new_dur

        go2[name] = ActionSpec(
            name=spec.name, description=spec.description,
            phases=fields.get("phases", spec.phases),
            gait=fields.get("gait", spec.gait),
            distance_m=spec.distance_m, rotation_rad=spec.rotation_rad,
            duration_s=fields.get("duration_s", spec.duration_s),
            speed_ms=fields.get("speed_ms", spec.speed_ms),
            **{k: v for k, v in fields.items()
               if k not in ("phases", "gait", "duration_s", "speed_ms")},
        )

    return go2


# ═══════════════════════════════════════════════════════════════════════════════
#  Unitree G1 — Humanoid actions (16-DOF: 12 legs + 4 arms)
# ═══════════════════════════════════════════════════════════════════════════════

# G1 joint indices (16 DOF)
_G1_L_HIP_YAW = 0
_G1_L_HIP_ROLL = 1
_G1_L_HIP_PITCH = 2
_G1_L_KNEE = 3
_G1_L_ANKLE_PITCH = 4
_G1_L_ANKLE_ROLL = 5
_G1_R_HIP_YAW = 6
_G1_R_HIP_ROLL = 7
_G1_R_HIP_PITCH = 8
_G1_R_KNEE = 9
_G1_R_ANKLE_PITCH = 10
_G1_R_ANKLE_ROLL = 11
_G1_L_SHOULDER_PITCH = 12
_G1_L_ELBOW = 13
_G1_R_SHOULDER_PITCH = 14
_G1_R_ELBOW = 15

_G1_N_JOINTS = 16

# G1 joint ranges — wide enough for ActionSpec.clamp_joints (MuJoCo enforces real limits)
_G1_HIP_RANGE   = (-3.14, 3.14)
_G1_THIGH_RANGE = (-3.14, 3.14)
_G1_KNEE_RANGE  = (-3.14, 3.14)

# G1 motor torques per joint type
_G1_TORQUE_HIP    = 88.0    # M107 motor
_G1_TORQUE_KNEE   = 139.0   # M139 motor
_G1_TORQUE_ANKLE  = 50.0    # M050 motor
_G1_TORQUE_ARM    = 25.0    # M025 motor (shoulder/elbow)

_G1_MAX_TORQUE_16 = (
    _G1_TORQUE_HIP, _G1_TORQUE_HIP, _G1_TORQUE_HIP,
    _G1_TORQUE_KNEE, _G1_TORQUE_ANKLE, _G1_TORQUE_ANKLE,
    _G1_TORQUE_HIP, _G1_TORQUE_HIP, _G1_TORQUE_HIP,
    _G1_TORQUE_KNEE, _G1_TORQUE_ANKLE, _G1_TORQUE_ANKLE,
    _G1_TORQUE_ARM, _G1_TORQUE_ARM, _G1_TORQUE_ARM, _G1_TORQUE_ARM,
)

_G1_MAX_VEL_16 = (
    30.1, 30.1, 30.1, 20.0, 37.0, 37.0,
    30.1, 30.1, 30.1, 20.0, 37.0, 37.0,
    30.0, 30.0, 30.0, 30.0,
)

# PD gains: 16 DOF (legs kp=100/120, arms kp=80 — lighter limbs need less stiffness)
_G1_KP_HOLD  = (100.0,) * 12 + (80.0,) * 4
_G1_KD_HOLD  = (2.0,) * 12 + (1.5,) * 4
_G1_KP_RISE  = (120.0,) * 12 + (80.0,) * 4
_G1_KD_RISE  = (2.5,) * 12 + (1.5,) * 4
_G1_KP_WALK  = (80.0,) * 12 + (60.0,) * 4
_G1_KD_WALK  = (1.5,) * 12 + (1.0,) * 4

# G1 poses (16 DOF): legs (12) + arms (4: L_shoulder, L_elbow, R_shoulder, R_elbow)
# Leg order: hip_yaw, hip_roll, hip_pitch, knee, ankle_pitch, ankle_roll (×2)
# Arm order: L_shoulder_pitch, L_elbow, R_shoulder_pitch, R_elbow
# Knee axis +y: positive=flexion. Ankle_pitch axis +y: positive=dorsiflexion.
# Shoulder_pitch axis +y: positive=arm forward/up. Elbow axis +y: positive=bend.
# Foot flat: ankle_pitch = hip_pitch - knee
_G1_ARM_NEUTRAL = (0.0, 0.0, 0.0, 0.0)

_G1_STAND = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0) + _G1_ARM_NEUTRAL

_G1_SIT   = (0.0, 0.0, -0.60, 1.20, -0.60, 0.0,
             0.0, 0.0, -0.60, 1.20, -0.60, 0.0) + _G1_ARM_NEUTRAL

# Crouch: gentle squat within ankle balance range.
# With knee at 0.20 rad (~11°), robot stays stable.
# Ankle compensates: ankle_pitch = -knee to keep feet flat.
_G1_CROUCH = (0.0, 0.0, 0.0, 0.20, -0.20, 0.0,
              0.0, 0.0, 0.0, 0.20, -0.20, 0.0) + _G1_ARM_NEUTRAL

_G1_DEEP_CROUCH = (0.0, 0.0, 0.0, 0.25, -0.25, 0.0,
                   0.0, 0.0, 0.0, 0.25, -0.25, 0.0) + _G1_ARM_NEUTRAL

_G1_PRONE = (0.0, 0.0, -0.40, 0.80, -0.40, 0.0,
             0.0, 0.0, -0.40, 0.80, -0.40, 0.0) + _G1_ARM_NEUTRAL


def _g1_16(legs_12, arms_4=_G1_ARM_NEUTRAL):
    """Build 16-DOF tuple from 12-DOF legs + 4-DOF arms."""
    return tuple(legs_12) + tuple(arms_4)


_G1_ZERO_DELAY = (0.0,) * _G1_N_JOINTS


def _g1_actions() -> dict[str, ActionSpec]:
    """Build G1 humanoid action library (16-DOF: 12 legs + 4 arms)."""
    actions: dict[str, ActionSpec] = {}

    actions["stand"] = ActionSpec(
        name="stand",
        description="Stand upright. Balanced bipedal stance, arms at sides.",
        robot="g1",
        phases=(
            ActionPhase(
                name="stand",
                duration_s=2.0,
                target=JointTarget(q12=_G1_STAND, kp=_G1_KP_HOLD, kd=_G1_KD_HOLD),
                motor_schedule=MotorSchedule(
                    max_velocity=(0.5,) * _G1_N_JOINTS,
                    max_torque=_G1_MAX_TORQUE_16,
                    delay_s=_G1_ZERO_DELAY,
                    max_pos_error=1.0,
                    sync_arrival=True,
                ),
            ),
        ),
        duration_s=2.0,
        max_pitch_rad=0.45, max_roll_rad=0.45, min_feet_contact=1,
        hip_range=_G1_HIP_RANGE, thigh_range=_G1_THIGH_RANGE, knee_range=_G1_KNEE_RANGE,
    )

    actions["stand_up"] = ActionSpec(
        name="stand_up",
        description="Rise from crouched or prone to standing.",
        robot="g1",
        phases=(
            ActionPhase(
                name="crouch",
                duration_s=2.0,
                target=JointTarget(q12=_G1_CROUCH, kp=_G1_KP_RISE, kd=_G1_KD_RISE),
                motor_schedule=MotorSchedule(
                    max_velocity=(0.4,) * _G1_N_JOINTS,
                    max_torque=_G1_MAX_TORQUE_16,
                    delay_s=_G1_ZERO_DELAY,
                    max_pos_error=1.0,
                    sync_arrival=True,
                ),
            ),
            ActionPhase(
                name="rise",
                duration_s=3.0,
                target=JointTarget(q12=_G1_STAND, kp=_G1_KP_RISE, kd=_G1_KD_RISE),
                motor_schedule=MotorSchedule(
                    max_velocity=(0.3,) * _G1_N_JOINTS,
                    max_torque=_G1_MAX_TORQUE_16,
                    delay_s=_G1_ZERO_DELAY,
                    max_pos_error=1.0,
                    sync_arrival=True,
                ),
            ),
        ),
        duration_s=5.0,
        max_pitch_rad=0.60, max_roll_rad=0.50, min_feet_contact=1,
        hip_range=_G1_HIP_RANGE, thigh_range=_G1_THIGH_RANGE, knee_range=_G1_KNEE_RANGE,
    )

    actions["sit"] = ActionSpec(
        name="sit",
        description="Squat down to sitting position.",
        robot="g1",
        phases=(
            ActionPhase(
                name="crouch",
                duration_s=1.5,
                target=JointTarget(q12=_G1_CROUCH, kp=_G1_KP_HOLD, kd=_G1_KD_HOLD),
                motor_schedule=MotorSchedule(
                    max_velocity=(0.5,) * _G1_N_JOINTS,
                    max_torque=_G1_MAX_TORQUE_16,
                    delay_s=_G1_ZERO_DELAY,
                    max_pos_error=1.0,
                    sync_arrival=True,
                ),
            ),
            ActionPhase(
                name="sit",
                duration_s=2.0,
                target=JointTarget(q12=_G1_SIT, kp=_G1_KP_HOLD, kd=_G1_KD_HOLD),
                motor_schedule=MotorSchedule(
                    max_velocity=(0.4,) * _G1_N_JOINTS,
                    max_torque=_G1_MAX_TORQUE_16,
                    delay_s=_G1_ZERO_DELAY,
                    max_pos_error=1.0,
                    sync_arrival=True,
                ),
            ),
        ),
        duration_s=3.5,
        max_pitch_rad=0.80, max_roll_rad=0.50, min_feet_contact=1,
        hip_range=_G1_HIP_RANGE, thigh_range=_G1_THIGH_RANGE, knee_range=_G1_KNEE_RANGE,
    )

    actions["lie_down"] = ActionSpec(
        name="lie_down",
        description="Lower to prone kneeling position.",
        robot="g1",
        phases=(
            ActionPhase(
                name="crouch",
                duration_s=1.5,
                target=JointTarget(q12=_G1_CROUCH, kp=_G1_KP_HOLD, kd=_G1_KD_HOLD),
                motor_schedule=MotorSchedule(
                    max_velocity=(0.5,) * _G1_N_JOINTS,
                    max_torque=_G1_MAX_TORQUE_16,
                    delay_s=_G1_ZERO_DELAY,
                    max_pos_error=1.0,
                    sync_arrival=True,
                ),
            ),
            ActionPhase(
                name="prone",
                duration_s=2.0,
                target=JointTarget(q12=_G1_PRONE, kp=_G1_KP_HOLD, kd=_G1_KD_HOLD),
                motor_schedule=MotorSchedule(
                    max_velocity=(0.3,) * _G1_N_JOINTS,
                    max_torque=_G1_MAX_TORQUE_16,
                    delay_s=_G1_ZERO_DELAY,
                    max_pos_error=1.0,
                    sync_arrival=True,
                ),
            ),
        ),
        duration_s=3.5,
        max_pitch_rad=0.80, max_roll_rad=0.60, min_feet_contact=0,
        hip_range=_G1_HIP_RANGE, thigh_range=_G1_THIGH_RANGE, knee_range=_G1_KNEE_RANGE,
    )

    # ── Crouch: gait-engine based for active balance ──
    # Uses BipedalGaitEngine in "stand" mode at reduced body height.
    # The engine's ankle balance keeps the robot stable throughout.
    actions["crouch"] = ActionSpec(
        name="crouch",
        description="Lower body to crouched stance. Actively balanced.",
        robot="g1",
        gait=GaitAction(gait_name="stand", cmd_vx=0.0, body_height=0.65, step_height=0.0),
        duration_s=4.0,
        min_feet_contact=1,
        hip_range=_G1_HIP_RANGE, thigh_range=_G1_THIGH_RANGE, knee_range=_G1_KNEE_RANGE,
    )

    # ── Deep crouch: even lower ──
    actions["deep_crouch"] = ActionSpec(
        name="deep_crouch",
        description="Deep crouch. Very low stance, actively balanced.",
        robot="g1",
        gait=GaitAction(gait_name="stand", cmd_vx=0.0, body_height=0.55, step_height=0.0),
        duration_s=5.0,
        min_feet_contact=1,
        hip_range=_G1_HIP_RANGE, thigh_range=_G1_THIGH_RANGE, knee_range=_G1_KNEE_RANGE,
    )

    # ── Lift left hand: raise left arm overhead ──
    # Strategy: slight weight shift to right leg for CoM stability,
    # then raise L_shoulder_pitch to ~2.5 rad (arm nearly vertical above head),
    # slight elbow bend for natural pose. Hold, then return.
    # CoM shifts laterally — hip roll compensates.
    actions["lift_left_hand"] = ActionSpec(
        name="lift_left_hand",
        description="Raise left hand overhead. Stable single-arm lift.",
        robot="g1",
        phases=(
            ActionPhase(
                name="weight_shift",
                duration_s=1.0,
                target=JointTarget(
                    # Keep legs straight — the balance engine handles stability.
                    # Start raising the arm slowly.
                    q12=_g1_16(
                        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                        (0.5, 0.1, 0.0, 0.0),  # L arm starts rising
                    ),
                    kp=_G1_KP_HOLD, kd=_G1_KD_HOLD,
                ),
                motor_schedule=MotorSchedule(
                    max_velocity=(0.8,) * _G1_N_JOINTS,
                    max_torque=_G1_MAX_TORQUE_16,
                    delay_s=_G1_ZERO_DELAY,
                    max_pos_error=1.0, sync_arrival=True,
                ),
            ),
            ActionPhase(
                name="raise_arm",
                duration_s=2.5,
                target=JointTarget(
                    q12=_g1_16(
                        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                        (2.50, 0.20, 0.0, 0.0),  # L arm raised high
                    ),
                    kp=_G1_KP_HOLD, kd=_G1_KD_HOLD,
                ),
                motor_schedule=MotorSchedule(
                    max_velocity=(0.8,) * _G1_N_JOINTS,
                    max_torque=_G1_MAX_TORQUE_16,
                    delay_s=_G1_ZERO_DELAY,
                    max_pos_error=1.0, sync_arrival=True,
                ),
            ),
            ActionPhase(
                name="hold_raised",
                duration_s=2.0,
                target=JointTarget(
                    q12=_g1_16(
                        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                        (2.50, 0.20, 0.0, 0.0),
                    ),
                    kp=_G1_KP_HOLD, kd=_G1_KD_HOLD,
                ),
                motor_schedule=MotorSchedule(
                    max_velocity=(0.2,) * _G1_N_JOINTS,
                    max_torque=_G1_MAX_TORQUE_16,
                    delay_s=_G1_ZERO_DELAY,
                    max_pos_error=1.0, sync_arrival=True,
                ),
                interpolation="hold",
            ),
            ActionPhase(
                name="lower_arm",
                duration_s=1.5,
                target=JointTarget(q12=_G1_STAND, kp=_G1_KP_HOLD, kd=_G1_KD_HOLD),
                motor_schedule=MotorSchedule(
                    max_velocity=(1.5,) * _G1_N_JOINTS,
                    max_torque=_G1_MAX_TORQUE_16,
                    delay_s=_G1_ZERO_DELAY,
                    max_pos_error=1.0, sync_arrival=True,
                ),
            ),
        ),
        duration_s=5.5,
        max_pitch_rad=0.50, max_roll_rad=0.55, min_feet_contact=1,
        hip_range=_G1_HIP_RANGE, thigh_range=_G1_THIGH_RANGE, knee_range=_G1_KNEE_RANGE,
    )

    # ── Lift right hand: raise right arm overhead ──
    # Mirror of lift_left_hand. Weight shifts to left leg.
    actions["lift_right_hand"] = ActionSpec(
        name="lift_right_hand",
        description="Raise right hand overhead. Stable single-arm lift.",
        robot="g1",
        phases=(
            ActionPhase(
                name="weight_shift",
                duration_s=1.0,
                target=JointTarget(
                    q12=_g1_16(
                        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                        (0.0, 0.0, 0.5, 0.1),  # R arm starts rising
                    ),
                    kp=_G1_KP_HOLD, kd=_G1_KD_HOLD,
                ),
                motor_schedule=MotorSchedule(
                    max_velocity=(0.8,) * _G1_N_JOINTS,
                    max_torque=_G1_MAX_TORQUE_16,
                    delay_s=_G1_ZERO_DELAY,
                    max_pos_error=1.0, sync_arrival=True,
                ),
            ),
            ActionPhase(
                name="raise_arm",
                duration_s=2.5,
                target=JointTarget(
                    q12=_g1_16(
                        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                        (0.0, 0.0, 2.50, 0.20),  # R arm raised high
                    ),
                    kp=_G1_KP_HOLD, kd=_G1_KD_HOLD,
                ),
                motor_schedule=MotorSchedule(
                    max_velocity=(0.8,) * _G1_N_JOINTS,
                    max_torque=_G1_MAX_TORQUE_16,
                    delay_s=_G1_ZERO_DELAY,
                    max_pos_error=1.0, sync_arrival=True,
                ),
            ),
            ActionPhase(
                name="hold_raised",
                duration_s=2.0,
                target=JointTarget(
                    q12=_g1_16(
                        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                        (0.0, 0.0, 2.50, 0.20),
                    ),
                    kp=_G1_KP_HOLD, kd=_G1_KD_HOLD,
                ),
                motor_schedule=MotorSchedule(
                    max_velocity=(0.2,) * _G1_N_JOINTS,
                    max_torque=_G1_MAX_TORQUE_16,
                    delay_s=_G1_ZERO_DELAY,
                    max_pos_error=1.0, sync_arrival=True,
                ),
                interpolation="hold",
            ),
            ActionPhase(
                name="lower_arm",
                duration_s=1.5,
                target=JointTarget(q12=_G1_STAND, kp=_G1_KP_HOLD, kd=_G1_KD_HOLD),
                motor_schedule=MotorSchedule(
                    max_velocity=(1.5,) * _G1_N_JOINTS,
                    max_torque=_G1_MAX_TORQUE_16,
                    delay_s=_G1_ZERO_DELAY,
                    max_pos_error=1.0, sync_arrival=True,
                ),
            ),
        ),
        duration_s=5.5,
        max_pitch_rad=0.50, max_roll_rad=0.55, min_feet_contact=1,
        hip_range=_G1_HIP_RANGE, thigh_range=_G1_THIGH_RANGE, knee_range=_G1_KNEE_RANGE,
    )

    # Gait-based actions — use BipedalGaitEngine via GaitAction
    actions["walk_forward"] = ActionSpec(
        name="walk_forward",
        description="Walk forward 1m. Bipedal gait with arm swing.",
        robot="g1",
        gait=GaitAction(gait_name="walk", cmd_vx=0.3, body_height=0.75, step_height=0.05),
        distance_m=1.0, speed_ms=0.3, duration_s=3.33,
        min_feet_contact=1,
        hip_range=_G1_HIP_RANGE, thigh_range=_G1_THIGH_RANGE, knee_range=_G1_KNEE_RANGE,
    )

    actions["walk_backward"] = ActionSpec(
        name="walk_backward",
        description="Walk backward 0.5m. Slow, stable.",
        robot="g1",
        gait=GaitAction(gait_name="walk", cmd_vx=-0.15, body_height=0.75, step_height=0.04),
        distance_m=0.5, speed_ms=0.15, duration_s=3.33,
        min_feet_contact=1,
        hip_range=_G1_HIP_RANGE, thigh_range=_G1_THIGH_RANGE, knee_range=_G1_KNEE_RANGE,
    )

    actions["trot_forward"] = ActionSpec(
        name="trot_forward",
        description="Walk forward briskly 1m.",
        robot="g1",
        gait=GaitAction(gait_name="walk", cmd_vx=0.5, body_height=0.75, step_height=0.06),
        distance_m=1.0, speed_ms=0.5, duration_s=2.0,
        min_feet_contact=1,
        hip_range=_G1_HIP_RANGE, thigh_range=_G1_THIGH_RANGE, knee_range=_G1_KNEE_RANGE,
    )

    actions["crawl_forward"] = ActionSpec(
        name="crawl_forward",
        description="Walk forward slowly 0.5m. Crouched, cautious.",
        robot="g1",
        gait=GaitAction(gait_name="slow_walk", cmd_vx=0.10, body_height=0.65, step_height=0.04),
        distance_m=0.5, speed_ms=0.10, duration_s=5.0,
        min_feet_contact=1,
        hip_range=_G1_HIP_RANGE, thigh_range=_G1_THIGH_RANGE, knee_range=_G1_KNEE_RANGE,
    )

    actions["turn_left"] = ActionSpec(
        name="turn_left",
        description="Turn left 90 degrees in place.",
        robot="g1",
        gait=GaitAction(gait_name="walk", cmd_vx=0.02, cmd_yaw=0.5, body_height=0.75, step_height=0.04),
        rotation_rad=math.pi / 2, duration_s=3.14,
        min_feet_contact=1,
        hip_range=_G1_HIP_RANGE, thigh_range=_G1_THIGH_RANGE, knee_range=_G1_KNEE_RANGE,
    )

    actions["turn_right"] = ActionSpec(
        name="turn_right",
        description="Turn right 90 degrees in place.",
        robot="g1",
        gait=GaitAction(gait_name="walk", cmd_vx=0.02, cmd_yaw=-0.5, body_height=0.75, step_height=0.04),
        rotation_rad=math.pi / 2, duration_s=3.14,
        min_feet_contact=1,
        hip_range=_G1_HIP_RANGE, thigh_range=_G1_THIGH_RANGE, knee_range=_G1_KNEE_RANGE,
    )

    actions["side_step_left"] = ActionSpec(
        name="side_step_left",
        description="Step left 0.3m.",
        robot="g1",
        gait=GaitAction(gait_name="slow_walk", cmd_vx=0.0, cmd_vy=-0.10, body_height=0.75, step_height=0.04),
        distance_m=0.3, speed_ms=0.10, duration_s=3.0,
        min_feet_contact=1,
        hip_range=_G1_HIP_RANGE, thigh_range=_G1_THIGH_RANGE, knee_range=_G1_KNEE_RANGE,
    )

    actions["side_step_right"] = ActionSpec(
        name="side_step_right",
        description="Step right 0.3m.",
        robot="g1",
        gait=GaitAction(gait_name="slow_walk", cmd_vx=0.0, cmd_vy=0.10, body_height=0.75, step_height=0.04),
        distance_m=0.3, speed_ms=0.10, duration_s=3.0,
        min_feet_contact=1,
        hip_range=_G1_HIP_RANGE, thigh_range=_G1_THIGH_RANGE, knee_range=_G1_KNEE_RANGE,
    )

    # Jump for humanoid — crouch then extend (arms swing up for momentum)
    actions["jump"] = ActionSpec(
        name="jump",
        description="Small hop. Controlled crouch, push off, land stable.",
        robot="g1",
        phases=(
            ActionPhase(
                name="crouch",
                duration_s=0.8,
                target=JointTarget(
                    q12=_g1_16(
                        (0.0, 0.0, 0.50, 1.00, -0.50, 0.0,
                         0.0, 0.0, 0.50, 1.00, -0.50, 0.0),
                        (-0.3, 0.3, -0.3, 0.3),  # arms back, elbows bent (wind-up)
                    ),
                    kp=_G1_KP_HOLD, kd=_G1_KD_HOLD,
                ),
                motor_schedule=MotorSchedule(
                    max_velocity=(2.0,) * _G1_N_JOINTS,
                    max_torque=_G1_MAX_TORQUE_16,
                    delay_s=_G1_ZERO_DELAY,
                    max_pos_error=1.0, sync_arrival=True,
                ),
            ),
            ActionPhase(
                name="hold",
                duration_s=0.3,
                target=JointTarget(
                    q12=_g1_16(
                        (0.0, 0.0, 0.50, 1.00, -0.50, 0.0,
                         0.0, 0.0, 0.50, 1.00, -0.50, 0.0),
                        (-0.3, 0.3, -0.3, 0.3),
                    ),
                    kp=_G1_KP_HOLD, kd=_G1_KD_HOLD,
                ),
                motor_schedule=MotorSchedule(
                    max_velocity=(0.1,) * _G1_N_JOINTS,
                    max_torque=_G1_MAX_TORQUE_16,
                    delay_s=_G1_ZERO_DELAY,
                    max_pos_error=1.0, sync_arrival=True,
                ),
                interpolation="hold",
            ),
            ActionPhase(
                name="launch",
                duration_s=0.2,
                target=JointTarget(
                    q12=_g1_16(
                        (0.0, 0.0, 0.05, 0.10, -0.05, 0.0,
                         0.0, 0.0, 0.05, 0.10, -0.05, 0.0),
                        (1.5, 0.0, 1.5, 0.0),  # arms swing up for momentum
                    ),
                    kp=(150.0,) * 12 + (100.0,) * 4,
                    kd=(3.0,) * 12 + (2.0,) * 4,
                ),
                motor_schedule=MotorSchedule(
                    max_velocity=(6.0,) * _G1_N_JOINTS,
                    max_torque=_G1_MAX_TORQUE_16,
                    delay_s=_G1_ZERO_DELAY,
                    max_pos_error=1.0, sync_arrival=True,
                ),
                interpolation="linear",
            ),
            ActionPhase(
                name="flight",
                duration_s=0.3,
                target=JointTarget(
                    q12=_g1_16(
                        (0.0, 0.0, 0.20, 0.40, -0.20, 0.0,
                         0.0, 0.0, 0.20, 0.40, -0.20, 0.0),
                        (0.5, 0.1, 0.5, 0.1),  # arms come down
                    ),
                    kp=_G1_KP_HOLD, kd=_G1_KD_HOLD,
                ),
                motor_schedule=MotorSchedule(
                    max_velocity=(5.0,) * _G1_N_JOINTS,
                    max_torque=_G1_MAX_TORQUE_16,
                    delay_s=_G1_ZERO_DELAY,
                    max_pos_error=1.0, sync_arrival=True,
                ),
            ),
            ActionPhase(
                name="land",
                duration_s=1.0,
                target=JointTarget(q12=_G1_STAND, kp=_G1_KP_HOLD, kd=_G1_KD_HOLD),
                motor_schedule=MotorSchedule(
                    max_velocity=(2.0,) * _G1_N_JOINTS,
                    max_torque=_G1_MAX_TORQUE_16,
                    delay_s=_G1_ZERO_DELAY,
                    max_pos_error=1.0, sync_arrival=True,
                ),
            ),
        ),
        duration_s=2.6,
        max_pitch_rad=1.0, max_roll_rad=0.80, min_feet_contact=0,
        hip_range=_G1_HIP_RANGE, thigh_range=_G1_THIGH_RANGE, knee_range=_G1_KNEE_RANGE,
    )

    # Wave — now uses actual arm instead of lifting a leg
    actions["shake_hand"] = ActionSpec(
        name="shake_hand",
        description="Wave: raise right arm and wave.",
        robot="g1",
        phases=(
            ActionPhase(
                name="raise_arm",
                duration_s=0.8,
                target=JointTarget(
                    q12=_g1_16(
                        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                        (0.0, 0.0, 1.80, 1.20),  # R arm forward + elbow bent (wave position)
                    ),
                    kp=_G1_KP_HOLD, kd=_G1_KD_HOLD,
                ),
                motor_schedule=MotorSchedule(
                    max_velocity=(2.0,) * _G1_N_JOINTS,
                    max_torque=_G1_MAX_TORQUE_16,
                    delay_s=_G1_ZERO_DELAY,
                    max_pos_error=1.0, sync_arrival=True,
                ),
            ),
            ActionPhase(
                name="wave_down",
                duration_s=0.5,
                target=JointTarget(
                    q12=_g1_16(
                        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                        (0.0, 0.0, 1.50, 1.60),  # wave down: lower shoulder, bend elbow more
                    ),
                    kp=_G1_KP_HOLD, kd=_G1_KD_HOLD,
                ),
                motor_schedule=MotorSchedule(
                    max_velocity=(3.0,) * _G1_N_JOINTS,
                    max_torque=_G1_MAX_TORQUE_16,
                    delay_s=_G1_ZERO_DELAY,
                    max_pos_error=1.0, sync_arrival=True,
                ),
            ),
            ActionPhase(
                name="wave_up",
                duration_s=0.5,
                target=JointTarget(
                    q12=_g1_16(
                        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                        (0.0, 0.0, 2.00, 0.80),  # wave up: raise shoulder, extend elbow
                    ),
                    kp=_G1_KP_HOLD, kd=_G1_KD_HOLD,
                ),
                motor_schedule=MotorSchedule(
                    max_velocity=(3.0,) * _G1_N_JOINTS,
                    max_torque=_G1_MAX_TORQUE_16,
                    delay_s=_G1_ZERO_DELAY,
                    max_pos_error=1.0, sync_arrival=True,
                ),
            ),
            ActionPhase(
                name="wave_down_2",
                duration_s=0.5,
                target=JointTarget(
                    q12=_g1_16(
                        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                        (0.0, 0.0, 1.50, 1.60),
                    ),
                    kp=_G1_KP_HOLD, kd=_G1_KD_HOLD,
                ),
                motor_schedule=MotorSchedule(
                    max_velocity=(3.0,) * _G1_N_JOINTS,
                    max_torque=_G1_MAX_TORQUE_16,
                    delay_s=_G1_ZERO_DELAY,
                    max_pos_error=1.0, sync_arrival=True,
                ),
            ),
            ActionPhase(
                name="return_stand",
                duration_s=1.0,
                target=JointTarget(q12=_G1_STAND, kp=_G1_KP_HOLD, kd=_G1_KD_HOLD),
                motor_schedule=MotorSchedule(
                    max_velocity=(1.5,) * _G1_N_JOINTS,
                    max_torque=_G1_MAX_TORQUE_16,
                    delay_s=_G1_ZERO_DELAY,
                    max_pos_error=1.0, sync_arrival=True,
                ),
            ),
        ),
        duration_s=3.3, max_pitch_rad=0.45, max_roll_rad=0.50, min_feet_contact=1,
        hip_range=_G1_HIP_RANGE, thigh_range=_G1_THIGH_RANGE, knee_range=_G1_KNEE_RANGE,
    )

    # Rear kick — kick right leg backward
    actions["rear_kick"] = ActionSpec(
        name="rear_kick",
        description="Rear kick: lean forward, kick right leg back.",
        robot="g1",
        phases=(
            ActionPhase(
                name="lean_forward",
                duration_s=0.6,
                target=JointTarget(
                    q12=_g1_16(
                        (0.0, 0.0, 0.30, 0.50, -0.20, 0.0,
                         0.0, 0.0, 0.30, 0.50, -0.20, 0.0),
                        (0.5, 0.3, 0.5, 0.3),  # arms forward for counterbalance
                    ),
                    kp=_G1_KP_HOLD, kd=_G1_KD_HOLD,
                ),
                motor_schedule=MotorSchedule(
                    max_velocity=(2.0,) * _G1_N_JOINTS,
                    max_torque=_G1_MAX_TORQUE_16,
                    delay_s=_G1_ZERO_DELAY, max_pos_error=1.0, sync_arrival=True,
                ),
            ),
            ActionPhase(
                name="kick",
                duration_s=0.3,
                target=JointTarget(
                    q12=_g1_16(
                        (0.0, -0.05, 0.40, 0.60, -0.20, 0.0,
                         0.0, 0.0, -0.80, 0.30, -0.10, 0.0),
                        (0.8, 0.2, 0.8, 0.2),
                    ),
                    kp=(150.0,) * 12 + (100.0,) * 4,
                    kd=(3.0,) * 12 + (2.0,) * 4,
                ),
                motor_schedule=MotorSchedule(
                    max_velocity=(8.0,) * _G1_N_JOINTS,
                    max_torque=_G1_MAX_TORQUE_16,
                    delay_s=_G1_ZERO_DELAY, max_pos_error=1.0, sync_arrival=False,
                ),
                interpolation="linear",
            ),
            ActionPhase(
                name="hold_kick",
                duration_s=0.5,
                target=JointTarget(
                    q12=_g1_16(
                        (0.0, -0.05, 0.40, 0.60, -0.20, 0.0,
                         0.0, 0.0, -0.80, 0.30, -0.10, 0.0),
                        (0.8, 0.2, 0.8, 0.2),
                    ),
                    kp=_G1_KP_HOLD, kd=_G1_KD_HOLD,
                ),
                motor_schedule=MotorSchedule(
                    max_velocity=(0.5,) * _G1_N_JOINTS,
                    max_torque=_G1_MAX_TORQUE_16,
                    delay_s=_G1_ZERO_DELAY, max_pos_error=1.0, sync_arrival=True,
                ),
            ),
            ActionPhase(
                name="return_stand",
                duration_s=1.0,
                target=JointTarget(q12=_G1_STAND, kp=_G1_KP_HOLD, kd=_G1_KD_HOLD),
                motor_schedule=MotorSchedule(
                    max_velocity=(1.5,) * _G1_N_JOINTS,
                    max_torque=_G1_MAX_TORQUE_16,
                    delay_s=_G1_ZERO_DELAY, max_pos_error=1.0, sync_arrival=True,
                ),
            ),
        ),
        duration_s=2.4, max_pitch_rad=1.0, max_roll_rad=0.60, min_feet_contact=0,
        hip_range=_G1_HIP_RANGE, thigh_range=_G1_THIGH_RANGE, knee_range=_G1_KNEE_RANGE,
    )

    # Rear up — balance on one leg with arms out for balance
    actions["rear_up"] = ActionSpec(
        name="rear_up",
        description="Balance on left leg, right leg extended, arms out.",
        robot="g1",
        phases=(
            ActionPhase(
                name="shift_weight",
                duration_s=1.0,
                target=JointTarget(
                    q12=_g1_16(
                        (0.0, -0.10, 0.15, 0.30, -0.15, -0.05,
                         0.0, -0.10, 0.15, 0.30, -0.15, 0.05),
                        (0.5, 0.2, 0.5, 0.2),  # arms slightly forward for balance
                    ),
                    kp=_G1_KP_HOLD, kd=_G1_KD_HOLD,
                ),
                motor_schedule=MotorSchedule(
                    max_velocity=(0.8,) * _G1_N_JOINTS,
                    max_torque=_G1_MAX_TORQUE_16,
                    delay_s=_G1_ZERO_DELAY, max_pos_error=1.0, sync_arrival=True,
                ),
            ),
            ActionPhase(
                name="lift_right",
                duration_s=1.0,
                target=JointTarget(
                    q12=_g1_16(
                        (0.0, -0.15, 0.20, 0.40, -0.20, -0.05,
                         0.0, 0.0, -0.60, 0.20, -0.10, 0.0),
                        (1.0, 0.3, 1.0, 0.3),  # arms forward for counterbalance
                    ),
                    kp=_G1_KP_HOLD, kd=_G1_KD_HOLD,
                ),
                motor_schedule=MotorSchedule(
                    max_velocity=(1.5,) * _G1_N_JOINTS,
                    max_torque=_G1_MAX_TORQUE_16,
                    delay_s=_G1_ZERO_DELAY, max_pos_error=1.0, sync_arrival=True,
                ),
            ),
            ActionPhase(
                name="hold",
                duration_s=2.0,
                target=JointTarget(
                    q12=_g1_16(
                        (0.0, -0.15, 0.20, 0.40, -0.20, -0.05,
                         0.0, 0.0, -0.60, 0.20, -0.10, 0.0),
                        (1.0, 0.3, 1.0, 0.3),
                    ),
                    kp=_G1_KP_HOLD, kd=_G1_KD_HOLD,
                ),
                motor_schedule=MotorSchedule(
                    max_velocity=(0.3,) * _G1_N_JOINTS,
                    max_torque=_G1_MAX_TORQUE_16,
                    delay_s=_G1_ZERO_DELAY, max_pos_error=1.0, sync_arrival=True,
                ),
            ),
            ActionPhase(
                name="return_stand",
                duration_s=1.5,
                target=JointTarget(q12=_G1_STAND, kp=_G1_KP_HOLD, kd=_G1_KD_HOLD),
                motor_schedule=MotorSchedule(
                    max_velocity=(1.0,) * _G1_N_JOINTS,
                    max_torque=_G1_MAX_TORQUE_16,
                    delay_s=_G1_ZERO_DELAY, max_pos_error=1.0, sync_arrival=True,
                ),
            ),
        ),
        duration_s=5.5, max_pitch_rad=0.60, max_roll_rad=0.60, min_feet_contact=0,
        hip_range=_G1_HIP_RANGE, thigh_range=_G1_THIGH_RANGE, knee_range=_G1_KNEE_RANGE,
    )

    return actions


class ActionLibrary:
    def __init__(self, robot: str):
        robot = robot.lower()
        if robot == "go1":
            self._actions = _go1_actions()
        elif robot == "go2":
            self._actions = _go2_actions()
        elif robot == "g1":
            self._actions = _g1_actions()
        else:
            raise ValueError(f"Unknown robot '{robot}'. Available: go1, go2, g1")
        self._robot = robot

    @property
    def robot(self) -> str:
        return self._robot

    def get(self, name: str) -> ActionSpec:
        if name not in self._actions:
            raise KeyError(f"Action '{name}' not found for {self._robot}. "
                           f"Available: {list(self._actions.keys())}")
        return self._actions[name]

    def list_actions(self) -> list[str]:
        return list(self._actions.keys())

    def describe(self) -> str:
        lines = [f"Action Library — {self._robot.upper()}", ""]
        for name, spec in self._actions.items():
            kind = "gait" if spec.is_gait else "phase"
            dist = f"{spec.distance_m}m" if spec.distance_m > 0 else ""
            rot = f"{math.degrees(spec.rotation_rad):.0f}°" if spec.rotation_rad > 0 else ""
            info = dist or rot or "static"
            lines.append(f"  {name:<20s}  [{kind:>5s}]  {info:<8s}  {spec.description}")
        return "\n".join(lines)

    def __contains__(self, name: str) -> bool:
        return name in self._actions

    def __len__(self) -> int:
        return len(self._actions)

    def __iter__(self):
        return iter(self._actions.values())


_LIBRARIES: dict[str, ActionLibrary] = {}

def get_library(robot: str) -> ActionLibrary:
    robot = robot.lower()
    if robot not in _LIBRARIES:
        _LIBRARIES[robot] = ActionLibrary(robot)
    return _LIBRARIES[robot]

def get_action(robot: str, name: str) -> ActionSpec:
    return get_library(robot).get(name)

def list_actions(robot: str) -> list[str]:
    return get_library(robot).list_actions()
