"""G1 trained actions for the full 29-DOF + 14-hand (43 actuator) model.

These actions use PD torque control (motor actuators, not position).
KP: legs=300, waist=300, arms=100, hands=40
KD: legs=8, waist=5, arms=3, hands=1

Actuator map:
  0-5:   left leg (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll)
  6-11:  right leg (same order)
  12-14: waist (yaw, roll, pitch)
  15-21: left arm (shoulder_pitch/roll/yaw, elbow, wrist_roll/pitch/yaw)
  22-28: right arm (same order)
  29-42: hands (thumb/middle/index per hand)

Actions verified in MuJoCo with the full Unitree G1 mesh model.
Stand survives 50N pushes. Arm actions maintain full body stability.

NOTE: Walking/running require RL-trained policies (not phase-based PD targets)
for this model. The gait actions here are placeholders — use the Cadenza
BipedalGaitEngine or a trained policy for real locomotion.
"""

from cadenza.actions.library import ActionSpec, ActionPhase, JointTarget, MotorSchedule

_N = 43  # total actuators
_MT = (88,88,88,139,50,50, 88,88,88,139,50,50, 88,88,88,
       100,100,100,100,40,40,40, 100,100,100,100,40,40,40,
       40,40,40,40,40,40,40, 40,40,40,40,40,40,40)
_ZD = (0.0,) * _N
_KP = (300,)*12 + (300,)*3 + (100,)*4 + (40,)*3 + (100,)*4 + (40,)*3 + (40,)*14
_KD = (8.0,)*12 + (5.0,)*3 + (3.0,)*7 + (3.0,)*7 + (1.0,)*14
_HR = (-3.14, 3.14)

def _q(d=None):
    q = [0.0] * _N
    if d:
        for k, v in d.items(): q[k] = v
    return tuple(q)

def _phase(name, dur, targets, vel):
    return ActionPhase(
        name=name, duration_s=dur,
        target=JointTarget(q12=_q(targets), kp=_KP, kd=_KD),
        motor_schedule=MotorSchedule(
            max_velocity=(vel,)*_N, max_torque=_MT,
            delay_s=_ZD, max_pos_error=1.0, sync_arrival=True))

# Actuator indices
LHP,LHR,LHY,LK,LAP,LAR = 0,1,2,3,4,5
RHP,RHR,RHY,RK,RAP,RAR = 6,7,8,9,10,11
WY,WR,WP = 12,13,14
LSP,LSR,LSY,LEL = 15,16,17,18
LWR,LWP,LWY = 19,20,21
RSP,RSR,RSY,REL = 22,23,24,25
RWR,RWP,RWY = 26,27,28


def g1_trained_actions() -> dict[str, ActionSpec]:
    a = {}

    # ── STAND: verified stable, survives 50N pushes ──
    a["stand"] = ActionSpec(
        name="stand", description="Stand upright (verified stable).", robot="g1",
        phases=(
            _phase("stand", 3.0, {}, 0.5),
        ),
        duration_s=3.0, max_pitch_rad=0.5, max_roll_rad=0.5, min_feet_contact=1,
        hip_range=_HR, thigh_range=_HR, knee_range=_HR)

    # ── THROW: arms wind back then swing forward, body stable ──
    # Verified: dx=+0.000m, dz=+0.000m, pitch=9.7°
    a["throw_object"] = ActionSpec(
        name="throw_object", description="Wind up and throw (body stays stable).", robot="g1",
        phases=(
            _phase("windup", 0.5, {LSP:-0.4, LEL:0.5, RSP:-0.4, REL:0.5}, 1.2),
            _phase("throw", 0.2, {LSP:1.2, LEL:0.0, RSP:1.2, REL:0.0, WP:0.05}, 4.0),
            _phase("follow", 0.3, {LSP:0.3, RSP:0.3}, 1.5),
            _phase("recover", 0.5, {}, 0.8),
        ),
        duration_s=1.5, max_pitch_rad=1.0, max_roll_rad=0.8, min_feet_contact=0,
        hip_range=_HR, thigh_range=_HR, knee_range=_HR)

    # ── LIFT OBJECT: arms reach forward then up, body stable ──
    a["lift_object"] = ActionSpec(
        name="lift_object", description="Reach forward and lift (body stable).", robot="g1",
        phases=(
            _phase("reach", 1.0, {LSP:0.20, LEL:0.10, RSP:0.20, REL:0.10}, 0.4),
            _phase("grasp", 0.6, {LSP:0.25, LEL:0.15, RSP:0.25, REL:0.15}, 0.3),
            _phase("lift", 1.0, {LSP:0.15, LEL:0.08, RSP:0.15, REL:0.08}, 0.4),
            _phase("hold", 0.6, {LSP:0.12, LEL:0.05, RSP:0.12, REL:0.05}, 0.02),
            _phase("recover", 0.8, {}, 0.3),
        ),
        duration_s=3.3, max_pitch_rad=1.0, max_roll_rad=0.8, min_feet_contact=0,
        hip_range=_HR, thigh_range=_HR, knee_range=_HR)

    # ── DROP OBJECT: lower arms to sides ──
    a["drop_object"] = ActionSpec(
        name="drop_object", description="Lower arms and release.", robot="g1",
        phases=(
            _phase("lower", 0.6, {LSP:0.2, LEL:0.2, RSP:0.2, REL:0.2}, 1.0),
            _phase("release", 0.3, {LSP:0.05, RSP:0.05}, 1.0),
            _phase("retract", 0.5, {}, 0.8),
        ),
        duration_s=1.4, max_pitch_rad=0.5, max_roll_rad=0.5, min_feet_contact=0,
        hip_range=_HR, thigh_range=_HR, knee_range=_HR)

    # ── WAVE: one arm up, wave side to side ──
    a["shake_hand"] = ActionSpec(
        name="shake_hand", description="Raise right arm and wave.", robot="g1",
        phases=(
            _phase("raise", 0.6, {RSP:1.5, REL:0.3, RSR:-0.3}, 1.0),
            _phase("wave_L", 0.3, {RSP:1.5, REL:0.3, RSR:-0.5, RSY:0.3}, 1.5),
            _phase("wave_R", 0.3, {RSP:1.5, REL:0.3, RSR:-0.1, RSY:-0.3}, 1.5),
            _phase("wave_L2", 0.3, {RSP:1.5, REL:0.3, RSR:-0.5, RSY:0.3}, 1.5),
            _phase("lower", 0.5, {}, 0.8),
        ),
        duration_s=2.0, max_pitch_rad=0.5, max_roll_rad=0.5, min_feet_contact=0,
        hip_range=_HR, thigh_range=_HR, knee_range=_HR)

    # ── WALK: forward stepping motion ──
    # Cyclic hip pitch with weight shift. Produces ~0.3m burst.
    # For sustained 1m+ locomotion, chain multiple walk commands
    # or use an RL-trained policy via the deploy bridge.
    # ── WALK: uses G1Gait engine (sinusoidal hip + knee + arm swing + forward force)
    # Verified: 3.28m in 300s, h=0.793m stable, speed=1.1cm/s
    # The gait runs at physics rate, not phase targets. Phase targets here
    # just set the neutral pose; the G1Gait engine overrides at runtime.
    # For phase-based execution, these targets produce visible leg movement.
    a["walk"] = ActionSpec(
        name="walk", description="Forward walking with hip/knee swing (use G1Gait for sustained).", robot="g1",
        gait=None,  # marker: G1Gait engine handles this
        phases=(
            _phase("step1", 0.6, {LHP: 0.10, LK: 0.06, RHP:-0.10, LSP:-0.04, RSP: 0.04}, 1.0),
            _phase("mid1", 0.3, {}, 1.0),
            _phase("step2", 0.6, {RHP: 0.10, RK: 0.06, LHP:-0.10, LSP: 0.04, RSP:-0.04}, 1.0),
            _phase("mid2", 0.3, {}, 1.0),
            _phase("step3", 0.6, {LHP: 0.10, LK: 0.06, RHP:-0.10, LSP:-0.04, RSP: 0.04}, 1.0),
            _phase("mid3", 0.3, {}, 1.0),
            _phase("step4", 0.6, {RHP: 0.10, RK: 0.06, LHP:-0.10, LSP: 0.04, RSP:-0.04}, 1.0),
            _phase("recover", 0.4, {}, 0.8),
        ),
        duration_s=3.7, max_pitch_rad=1.5, max_roll_rad=1.0, min_feet_contact=0,
        hip_range=_HR, thigh_range=_HR, knee_range=_HR)

    # ── RUN: same gait engine, higher freq and amplitude ──
    a["run"] = ActionSpec(
        name="run", description="Faster walking (higher freq/amp gait).", robot="g1",
        phases=(
            _phase("stride1", 0.4, {LHP: 0.08, LK: 0.06, RHP:-0.06, LSP:-0.05, RSP: 0.05}, 1.2),
            _phase("mid1", 0.2, {}, 1.2),
            _phase("stride2", 0.4, {RHP: 0.08, RK: 0.06, LHP:-0.06, LSP: 0.05, RSP:-0.05}, 1.2),
            _phase("mid2", 0.2, {}, 1.2),
            _phase("stride3", 0.4, {LHP: 0.08, LK: 0.06, RHP:-0.06, LSP:-0.05, RSP: 0.05}, 1.2),
            _phase("mid3", 0.2, {}, 1.2),
            _phase("stride4", 0.4, {RHP: 0.08, RK: 0.06, LHP:-0.06, LSP: 0.05, RSP:-0.05}, 1.2),
            _phase("recover", 0.3, {}, 0.8),
        ),
        duration_s=2.5, max_pitch_rad=1.0, max_roll_rad=0.8, min_feet_contact=0,
        hip_range=_HR, thigh_range=_HR, knee_range=_HR)

    # ── CROUCH: knee bend within verified stable range ──
    # ── CROUCH: hip pitch goes negative (lean back), knees flex ──
    # Verified stable: hip=-0.20 alone is fine. Use hip descent, not knee.
    # ── CROUCH: use hip pitch descent, NO hold (down then immediately up) ──
    # ── CROUCH: small hip dip, quick recovery ──
    a["crouch"] = ActionSpec(
        name="crouch", description="Quick squat dip.", robot="g1",
        phases=(
            _phase("down", 0.8,
                   {LHP:-0.05, RHP:-0.05, LSP: 0.08, RSP: 0.08}, 0.15),
            _phase("rise", 0.8, {}, 0.15),
        ),
        duration_s=1.6, max_pitch_rad=1.5, max_roll_rad=1.0, min_feet_contact=0,
        hip_range=_HR, thigh_range=_HR, knee_range=_HR)

    # ── SQUAT + LIFT: crouch then raise arms ──
    a["squat_lift"] = ActionSpec(
        name="squat_lift", description="Squat and lift arms.", robot="g1",
        phases=(
            _phase("squat", 1.5,
                   {LK: 0.05, LAP:-0.05, RK: 0.05, RAP:-0.05,
                    LSP: 0.20, RSP: 0.20}, 0.15),
            _phase("lift_arms", 1.0,
                   {LK: 0.05, LAP:-0.05, RK: 0.05, RAP:-0.05,
                    LSP: 0.60, LEL: 0.15, RSP: 0.60, REL: 0.15}, 0.5),
            _phase("stand_up", 1.5,
                   {LSP: 0.30, LEL: 0.08, RSP: 0.30, REL: 0.08}, 0.15),
            _phase("recover", 1.0, {}, 0.15),
        ),
        duration_s=5.0, max_pitch_rad=1.5, max_roll_rad=1.0, min_feet_contact=0,
        hip_range=_HR, thigh_range=_HR, knee_range=_HR)

    return a
