"""Hardcoded physical specifications for Unitree Go1 and Go2."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


JOINT_NAMES = [
    "FL_hip", "FL_thigh", "FL_calf",
    "FR_hip", "FR_thigh", "FR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
]

LEG_INDICES = {
    "FL": [0, 1, 2],
    "FR": [3, 4, 5],
    "RL": [6, 7, 8],
    "RR": [9, 10, 11],
}

FOOT_ORDER = ["FL", "FR", "RL", "RR"]


@dataclass(frozen=True)
class JointLimits:
    hip_min:   float
    hip_max:   float
    thigh_min: float
    thigh_max: float
    knee_min:  float
    knee_max:  float


GO1_JOINT_LIMITS = JointLimits(
    hip_min   = -0.863,
    hip_max   =  0.863,
    thigh_min = -0.686,
    thigh_max =  4.501,
    knee_min  = -2.818,
    knee_max  = -0.888,
)

GO2_JOINT_LIMITS = JointLimits(
    hip_min   = -1.047,
    hip_max   =  1.047,
    thigh_min = -0.663,
    thigh_max =  3.927,
    knee_min  = -2.721,
    knee_max  = -0.837,
)


@dataclass(frozen=True)
class MotorLimits:
    max_torque_Nm:    float
    max_velocity_rads: float
    kp_default:       float
    kd_default:       float
    kp_stance:        float
    kd_stance:        float
    kp_swing:         float
    kd_swing:         float


GO1_MOTOR = MotorLimits(
    max_torque_Nm     = 33.5,
    max_velocity_rads = 21.0,
    kp_default        = 20.0,
    kd_default        =  0.5,
    kp_stance         = 35.0,
    kd_stance         =  0.6,
    kp_swing          = 20.0,
    kd_swing          =  0.5,
)

GO2_MOTOR = MotorLimits(
    max_torque_Nm     = 45.0,
    max_velocity_rads = 30.0,
    kp_default        = 25.0,
    kd_default        =  0.6,
    kp_stance         = 50.0,
    kd_stance         =  1.2,
    kp_swing          = 15.0,
    kd_swing          =  0.6,
)


@dataclass(frozen=True)
class Kinematics:
    body_length:      float
    body_width:       float
    hip_offset_lat:   float
    hip_offset_lon:   float
    thigh_length:     float
    calf_length:      float
    foot_radius:      float
    body_mass_kg:     float
    total_mass_kg:    float
    com_height_stand: float
    max_body_height:  float


GO1_KIN = Kinematics(
    body_length      = 0.3762,
    body_width       = 0.0945,
    hip_offset_lat   = 0.0838,
    hip_offset_lon   = 0.1881,
    thigh_length     = 0.2130,
    calf_length      = 0.2130,
    foot_radius      = 0.0165,
    body_mass_kg     = 9.041,
    total_mass_kg    = 12.0,
    com_height_stand = 0.280,
    max_body_height  = 0.320,
)

GO2_KIN = Kinematics(
    body_length      = 0.3762,
    body_width       = 0.0945,
    hip_offset_lat   = 0.0838,
    hip_offset_lon   = 0.1881,
    thigh_length     = 0.2130,
    calf_length      = 0.2130,
    foot_radius      = 0.0165,
    body_mass_kg     = 10.2,
    total_mass_kg    = 15.0,
    com_height_stand = 0.280,
    max_body_height  = 0.320,
)


@dataclass(frozen=True)
class DefaultPoses:
    stand:       tuple
    sit:         tuple
    prone:       tuple


def _pose12(fl, fr, rl, rr):
    """Build 12-dim pose from 4 leg (hip, thigh, calf) tuples."""
    return fl + fr + rl + rr


GO1_POSES = DefaultPoses(
    stand = _pose12(
        fl=(0.0,  0.9, -1.8), fr=( 0.0,  0.9, -1.8),
        rl=(0.0,  0.9, -1.8), rr=( 0.0,  0.9, -1.8),
    ),
    sit = _pose12(
        fl=(0.0,  1.2, -2.5), fr=( 0.0,  1.2, -2.5),
        rl=(0.0,  0.5, -1.2), rr=( 0.0,  0.5, -1.2),
    ),
    prone = _pose12(
        fl=(0.0,  1.5, -2.8), fr=( 0.0,  1.5, -2.8),
        rl=(0.0,  1.5, -2.8), rr=( 0.0,  1.5, -2.8),
    ),
)

GO2_POSES = GO1_POSES


@dataclass(frozen=True)
class GaitParams:
    name:          str
    freq_hz:       float
    duty_cycle:    float
    phase_offsets: tuple
    swing_height:  float
    max_speed:     float
    max_yaw:       float
    description:   str


GAITS = {
    "stand": GaitParams(
        name="stand", freq_hz=0.0, duty_cycle=1.0,
        phase_offsets=(0.0, 0.0, 0.0, 0.0),
        swing_height=0.0, max_speed=0.0, max_yaw=0.3,
        description="All feet on ground. Body rotation only.",
    ),
    "trot": GaitParams(
        name="trot", freq_hz=2.0, duty_cycle=0.50,
        phase_offsets=(0.0, 0.5, 0.5, 0.0),
        swing_height=0.12, max_speed=1.5, max_yaw=0.8,
        description="Diagonal pairs move together. Stable, efficient on flat terrain.",
    ),
    "walk": GaitParams(
        name="walk", freq_hz=1.3, duty_cycle=0.85,
        phase_offsets=(0.0, 0.5, 0.25, 0.75),
        swing_height=0.10, max_speed=0.6, max_yaw=0.8,
        description="Four-beat walk. Higher stepping, moderate ground contact.",
    ),
    "crawl": GaitParams(
        name="crawl", freq_hz=0.8, duty_cycle=0.90,
        phase_offsets=(0.0, 0.5, 0.25, 0.75),
        swing_height=0.08, max_speed=0.3, max_yaw=0.3,
        description="Ultra-stable crawl. ≥3 feet always. For rough terrain / slopes.",
    ),
    "pace": GaitParams(
        name="pace", freq_hz=3.5, duty_cycle=0.60,
        phase_offsets=(0.0, 0.5, 0.0, 0.5),
        swing_height=0.10, max_speed=1.8, max_yaw=0.5,
        description="Lateral pairs move together. Higher speed on firm ground.",
    ),
    "bound": GaitParams(
        name="bound", freq_hz=4.0, duty_cycle=0.50,
        phase_offsets=(0.0, 0.0, 0.5, 0.5),
        swing_height=0.12, max_speed=2.5, max_yaw=0.2,
        description="Pronking gait. High speed, needs flat ground.",
    ),
    "pronk": GaitParams(
        name="pronk", freq_hz=2.5, duty_cycle=0.40,
        phase_offsets=(0.0, 0.0, 0.0, 0.0),
        swing_height=0.15, max_speed=0.0, max_yaw=0.0,
        description="All feet leave ground simultaneously. Jumping in place.",
    ),
    "stair_crawl": GaitParams(
        name="stair_crawl", freq_hz=0.5, duty_cycle=0.90,
        phase_offsets=(0.0, 0.5, 0.25, 0.75),
        swing_height=0.18, max_speed=0.15, max_yaw=0.1,
        description="Maximum foot clearance crawl for stair climbing.",
    ),
    "precision_turn": GaitParams(
        name="precision_turn", freq_hz=0.6, duty_cycle=0.92,
        phase_offsets=(0.0, 0.5, 0.25, 0.75),
        swing_height=0.03, max_speed=0.05, max_yaw=0.4,
        description="Compressed, ultra-slow turn. Tiny steps, body low, ≥3 feet always.",
    ),
}


@dataclass(frozen=True)
class TerrainCapability:
    terrain:         str
    max_speed:       float
    recommended_gait: str
    max_slope_deg:   float
    max_step_height: float
    max_gap_width:   float
    foot_clearance:  float
    notes:           str


GO1_TERRAIN = [
    TerrainCapability("flat_concrete", 1.5, "trot",       0.0, 0.05, 0.20, 0.08, "Full speed trot. Home terrain."),
    TerrainCapability("flat_grass",    1.0, "trot",       5.0, 0.08, 0.15, 0.10, "Slight compliance. Watch for holes."),
    TerrainCapability("gravel",        0.6, "trot",       8.0, 0.06, 0.10, 0.10, "Loose surface. Reduce speed. Trot OK."),
    TerrainCapability("mud",           0.3, "crawl",      5.0, 0.05, 0.10, 0.12, "High resistance. Slow crawl. Monitor slip."),
    TerrainCapability("sand",          0.4, "walk",       8.0, 0.05, 0.10, 0.12, "High sinkage. Walk gait. Reduce load."),
    TerrainCapability("slope_gentle",  0.8, "trot",      15.0, 0.08, 0.15, 0.10, "≤15°. Lower body, tilt CoM uphill."),
    TerrainCapability("slope_steep",   0.3, "crawl",     25.0, 0.06, 0.10, 0.08, "15-25°. Crawl. Watch pitch > 0.35 rad."),
    TerrainCapability("stairs_up",     0.2, "stair_crawl",0.0, 0.20, 0.00, 0.18, "Max 20cm rise, 30cm tread. Crawl only."),
    TerrainCapability("stairs_down",   0.15,"stair_crawl",0.0, 0.20, 0.00, 0.15, "Descend: pitch body forward slightly."),
    TerrainCapability("rubble",        0.2, "crawl",     10.0, 0.15, 0.20, 0.15, "Uneven rocks. Very slow. Max clearance."),
    TerrainCapability("ice",           0.2, "walk",       5.0, 0.03, 0.10, 0.06, "Low clearance, slow walk. Watch lateral slip."),
    TerrainCapability("snow_shallow",  0.4, "trot",       5.0, 0.08, 0.12, 0.15, "< 15cm snow. High stepping trot."),
    TerrainCapability("snow_deep",     0.15,"crawl",      5.0, 0.05, 0.10, 0.20, "> 15cm snow. Crawl, max clearance."),
    TerrainCapability("wood_floor",    1.2, "trot",       0.0, 0.05, 0.20, 0.08, "Slippery. Reduce speed vs concrete."),
    TerrainCapability("carpet",        1.0, "trot",       0.0, 0.05, 0.20, 0.08, "High friction. Normal trot."),
]

GO2_TERRAIN = [t for t in GO1_TERRAIN]


@dataclass(frozen=True)
class SafetyThresholds:
    roll_warn_rad:     float
    roll_stop_rad:     float
    pitch_warn_rad:    float
    pitch_stop_rad:    float
    omega_warn_rads:   float
    omega_stop_rads:   float
    joint_pos_margin:  float
    joint_vel_warn:    float
    joint_torque_warn: float
    min_feet_contact:  int
    battery_warn_pct:  float
    battery_stop_pct:  float
    slip_velocity_threshold: float
    fall_height_threshold:   float


GO1_SAFETY = SafetyThresholds(
    roll_warn_rad    = 0.40,
    roll_stop_rad    = 0.60,
    pitch_warn_rad   = 0.35,
    pitch_stop_rad   = 0.55,
    omega_warn_rads  = 2.0,
    omega_stop_rads  = 4.0,
    joint_pos_margin = 0.05,
    joint_vel_warn   = 18.0,
    joint_torque_warn = 28.0,
    min_feet_contact = 2,
    battery_warn_pct = 20.0,
    battery_stop_pct = 10.0,
    slip_velocity_threshold = 0.3,
    fall_height_threshold   = 0.20,
)

GO2_SAFETY = SafetyThresholds(
    roll_warn_rad    = 0.45,
    roll_stop_rad    = 0.65,
    pitch_warn_rad   = 0.40,
    pitch_stop_rad   = 0.60,
    omega_warn_rads  = 2.5,
    omega_stop_rads  = 5.0,
    joint_pos_margin = 0.05,
    joint_vel_warn   = 25.0,
    joint_torque_warn = 38.0,
    min_feet_contact = 2,
    battery_warn_pct = 20.0,
    battery_stop_pct = 10.0,
    slip_velocity_threshold = 0.4,
    fall_height_threshold   = 0.20,
)


@dataclass(frozen=True)
class SensorSpec:
    imu_rate_hz:          float
    imu_noise_rpy_rad:    float
    joint_state_rate_hz:  float
    foot_force_threshold_N: float
    depth_camera:         bool
    lidar:                bool
    gps:                  bool
    camera_fov_deg:       float


GO1_SENSORS = SensorSpec(
    imu_rate_hz          = 500.0,
    imu_noise_rpy_rad    = 0.002,
    joint_state_rate_hz  = 500.0,
    foot_force_threshold_N = 20.0,
    depth_camera         = True,
    lidar                = False,
    gps                  = False,
    camera_fov_deg       = 86.0,
)

GO2_SENSORS = SensorSpec(
    imu_rate_hz          = 500.0,
    imu_noise_rpy_rad    = 0.001,
    joint_state_rate_hz  = 500.0,
    foot_force_threshold_N = 20.0,
    depth_camera         = True,
    lidar                = True,
    gps                  = False,
    camera_fov_deg       = 100.0,
)


@dataclass(frozen=True)
class Benchmarks:
    """Reference performance numbers for evaluation."""
    flat_speed_ms:         float
    slope_15deg_speed_ms:  float
    stair_speed_ms:        float
    trot_stability_roll:   float
    energy_flat_Wh_per_m:  float
    recovery_time_s:       float
    position_drift_m_per_m: float


GO1_BENCH = Benchmarks(
    flat_speed_ms          = 1.2,
    slope_15deg_speed_ms   = 0.6,
    stair_speed_ms         = 0.2,
    trot_stability_roll    = 0.15,
    energy_flat_Wh_per_m   = 0.25,
    recovery_time_s        = 2.0,
    position_drift_m_per_m = 0.05,
)

GO2_BENCH = Benchmarks(
    flat_speed_ms          = 1.8,
    slope_15deg_speed_ms   = 0.9,
    stair_speed_ms         = 0.3,
    trot_stability_roll    = 0.12,
    energy_flat_Wh_per_m   = 0.22,
    recovery_time_s        = 1.5,
    position_drift_m_per_m = 0.03,
)


@dataclass(frozen=True)
class RobotSpec:
    model:      str
    joints:     JointLimits
    motor:      MotorLimits
    kin:        Kinematics
    poses:      DefaultPoses
    safety:     SafetyThresholds
    sensors:    SensorSpec
    benchmarks: Benchmarks
    terrain:    list
    n_joints:   int = 12
    control_rate_hz: float = 50.0
    lowlevel_rate_hz: float = 500.0

    def terrain_for(self, label: str) -> "TerrainCapability | None":
        for t in self.terrain:
            if t.terrain == label:
                return t
        return None

    def gait(self, name: str):
        return GAITS.get(name)

    def summary(self) -> str:
        return (
            f"[{self.model}] mass={self.kin.total_mass_kg}kg "
            f"thigh={self.kin.thigh_length}m calf={self.kin.calf_length}m "
            f"max_torque={self.motor.max_torque_Nm}Nm "
            f"hip={self.joints.hip_min:.2f}..{self.joints.hip_max:.2f}rad "
            f"thigh={self.joints.thigh_min:.2f}..{self.joints.thigh_max:.2f}rad "
            f"knee={self.joints.knee_min:.2f}..{self.joints.knee_max:.2f}rad"
        )


GO1 = RobotSpec(
    model="go1", joints=GO1_JOINT_LIMITS, motor=GO1_MOTOR, kin=GO1_KIN,
    poses=GO1_POSES, safety=GO1_SAFETY, sensors=GO1_SENSORS,
    benchmarks=GO1_BENCH, terrain=GO1_TERRAIN,
)

GO2 = RobotSpec(
    model="go2", joints=GO2_JOINT_LIMITS, motor=GO2_MOTOR, kin=GO2_KIN,
    poses=GO2_POSES, safety=GO2_SAFETY, sensors=GO2_SENSORS,
    benchmarks=GO2_BENCH, terrain=GO2_TERRAIN,
)

# ═══════════════════════════════════════════════════════════════════════════════
#  Unitree G1 — Humanoid (12-DOF legs: 6 per leg)
# ═══════════════════════════════════════════════════════════════════════════════

G1_JOINT_NAMES = [
    "L_hip_yaw", "L_hip_roll", "L_hip_pitch", "L_knee", "L_ankle_pitch", "L_ankle_roll",
    "R_hip_yaw", "R_hip_roll", "R_hip_pitch", "R_knee", "R_ankle_pitch", "R_ankle_roll",
    "L_shoulder_pitch", "L_elbow", "R_shoulder_pitch", "R_elbow",
]

G1_LEG_INDICES = {
    "L": [0, 1, 2, 3, 4, 5],
    "R": [6, 7, 8, 9, 10, 11],
}

G1_ARM_INDICES = {
    "L": [12, 13],
    "R": [14, 15],
}

G1_FOOT_ORDER = ["L", "R"]

# Joint limits — reuse the quadruped dataclass with G1's widest ranges
# (actual per-joint limits are enforced in the MuJoCo model and gait engine)
G1_JOINT_LIMITS = JointLimits(
    hip_min=-2.87, hip_max=2.87,       # widest (hip yaw)
    thigh_min=-2.53, thigh_max=2.53,   # hip pitch range
    knee_min=-0.87, knee_max=2.05,     # covers knee + ankle pitch
)

G1_MOTOR = MotorLimits(
    max_torque_Nm=139.0,       # knee motor (strongest)
    max_velocity_rads=37.0,    # ankle motor (fastest)
    kp_default=50.0,
    kd_default=1.0,
    kp_stance=80.0,
    kd_stance=2.0,
    kp_swing=30.0,
    kd_swing=0.8,
)

G1_KIN = Kinematics(
    body_length=0.20,          # pelvis front-back
    body_width=0.175,          # hip-to-hip distance
    hip_offset_lat=0.0875,     # half hip width
    hip_offset_lon=0.0,        # humanoid: hips are centered
    thigh_length=0.30,
    calf_length=0.30,
    foot_radius=0.015,         # foot height (box)
    body_mass_kg=20.0,         # pelvis + torso
    total_mass_kg=35.0,
    com_height_stand=0.75,
    max_body_height=0.85,
)

# Standing pose: hip_yaw, hip_roll, hip_pitch, knee, ankle_pitch, ankle_roll (×2 legs)
#   + L_shoulder_pitch, L_elbow, R_shoulder_pitch, R_elbow
# Knee axis is +y: positive = flexion (backward bend)
# Ankle pitch axis is +y: positive = dorsiflexion (toes up)
# Shoulder pitch axis is +y: positive = arm forward/up
# Elbow axis is +y: positive = forearm flexion (bend)
# Foot flat: ankle_pitch = hip_pitch - knee
_G1_STAND_LEG = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
_G1_SIT_LEG   = (0.0, 0.0, 0.0, 0.20, -0.20, 0.0)
_G1_PRONE_LEG = (0.0, 0.0, 0.0, 0.20, -0.20, 0.0)
_G1_ARM_NEUTRAL = (0.0, 0.0, 0.0, 0.0)  # arms at sides, straight

G1_POSES = DefaultPoses(
    stand=_G1_STAND_LEG + _G1_STAND_LEG + _G1_ARM_NEUTRAL,
    sit=_G1_SIT_LEG + _G1_SIT_LEG + _G1_ARM_NEUTRAL,
    prone=_G1_PRONE_LEG + _G1_PRONE_LEG + _G1_ARM_NEUTRAL,
)

G1_SAFETY = SafetyThresholds(
    roll_warn_rad=0.25,
    roll_stop_rad=0.45,
    pitch_warn_rad=0.25,
    pitch_stop_rad=0.45,
    omega_warn_rads=1.5,
    omega_stop_rads=3.0,
    joint_pos_margin=0.05,
    joint_vel_warn=25.0,
    joint_torque_warn=100.0,
    min_feet_contact=1,
    battery_warn_pct=20.0,
    battery_stop_pct=10.0,
    slip_velocity_threshold=0.3,
    fall_height_threshold=0.50,
)

G1_SENSORS = SensorSpec(
    imu_rate_hz=500.0,
    imu_noise_rpy_rad=0.001,
    joint_state_rate_hz=500.0,
    foot_force_threshold_N=30.0,
    depth_camera=True,
    lidar=True,
    gps=False,
    camera_fov_deg=120.0,
)

G1_BENCH = Benchmarks(
    flat_speed_ms=1.5,
    slope_15deg_speed_ms=0.5,
    stair_speed_ms=0.3,
    trot_stability_roll=0.10,
    energy_flat_Wh_per_m=0.35,
    recovery_time_s=3.0,
    position_drift_m_per_m=0.04,
)

G1_TERRAIN = [
    TerrainCapability("flat_concrete", 1.5, "walk",  0.0, 0.10, 0.30, 0.10, "Full speed walk. Home terrain."),
    TerrainCapability("flat_grass",    1.0, "walk",  5.0, 0.10, 0.20, 0.12, "Slight compliance. Walk gait."),
    TerrainCapability("gravel",        0.5, "walk",  8.0, 0.08, 0.15, 0.12, "Loose surface. Slow walk."),
    TerrainCapability("stairs_up",     0.3, "walk",  0.0, 0.22, 0.00, 0.20, "Stairs. Slow deliberate walk."),
    TerrainCapability("stairs_down",   0.2, "walk",  0.0, 0.22, 0.00, 0.18, "Descend stairs. Very slow."),
    TerrainCapability("slope_gentle",  0.8, "walk", 15.0, 0.10, 0.20, 0.12, "Gentle slope. Walk gait."),
    TerrainCapability("slope_steep",   0.3, "walk", 25.0, 0.08, 0.15, 0.10, "Steep slope. Very slow walk."),
]

G1 = RobotSpec(
    model="g1", joints=G1_JOINT_LIMITS, motor=G1_MOTOR, kin=G1_KIN,
    poses=G1_POSES, safety=G1_SAFETY, sensors=G1_SENSORS,
    benchmarks=G1_BENCH, terrain=G1_TERRAIN,
    n_joints=16, control_rate_hz=50.0, lowlevel_rate_hz=500.0,
)

ROBOT_SPECS: dict[str, RobotSpec] = {"go1": GO1, "go2": GO2, "g1": G1}


def get_spec(model: str) -> RobotSpec:
    model = model.lower()
    if model not in ROBOT_SPECS:
        raise ValueError(f"Unknown robot model '{model}'. Available: {list(ROBOT_SPECS)}")
    return ROBOT_SPECS[model]
