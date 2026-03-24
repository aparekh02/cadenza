"""terrain_classifier.py — Online terrain classifier using IMU + foot contact only.

No camera, no LiDAR required. Classifies terrain from the statistics of
the current Short-Term Memory window (joint torques, IMU variance, contact patterns).

Returns a TerrainEstimate with a confidence score and recommended parameters.
The classifier also computes real-time benchmarks against the spec values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from cadenza_local.locomotion.robot_spec import TerrainCapability, GAITS

if TYPE_CHECKING:
    from cadenza_local.locomotion.robot_spec import RobotSpec
    from cadenza_local.locomotion.memory.stm import STM


# ──────────────────────────────────────────────────────────────────────────────
#  Output
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TerrainEstimate:
    label:           str     # matched terrain label
    confidence:      float   # [0, 1]
    recommended_gait: str
    max_speed:       float   # m/s
    step_height:     float   # m
    slope_deg:       float   # estimated slope (degrees)
    slip_risk:       float   # [0, 1] — 0=stable, 1=very slippery
    hardness:        float   # [0, 1] — 0=soft, 1=hard (inferred from contact bounce)
    features:        dict    # raw feature values for debugging


@dataclass
class BenchmarkResult:
    """Comparison of current performance vs spec benchmarks."""
    actual_speed:     float
    expected_speed:   float
    speed_ratio:      float   # actual / expected
    roll_stability:   float   # current RMS roll vs benchmark
    energy_estimate:  float   # Wh/m (estimated from torque)
    on_benchmark:     bool    # True if meeting spec
    notes:            list[str]


# ──────────────────────────────────────────────────────────────────────────────
#  Feature extraction from STM
# ──────────────────────────────────────────────────────────────────────────────

def _extract_features(stm: "STM") -> dict:
    """Extract classification features from the STM rolling window."""
    arr = stm.as_array()   # (T, D)
    if arr.ndim < 2 or arr.shape[0] < 2:
        return {}

    T = arr.shape[0]
    # Slice layout matches STMFrame.to_vector():
    # joint_pos (12), joint_vel (12), imu_rpy (3), imu_omega (3), foot_contact (4), cmd_vel (3)
    #   0-11         12-23           24-26         27-29          30-33            34-36
    jp   = arr[:, 0:12]    # joint positions
    jv   = arr[:, 12:24]   # joint velocities
    rpy  = arr[:, 24:27]   # roll, pitch, yaw
    omega = arr[:, 27:30]  # angular velocity
    fc   = arr[:, 30:34]   # foot contact (0/1)
    cv   = arr[:, 34:37]   # cmd_vel

    roll, pitch = rpy[:, 0], rpy[:, 1]
    speed       = np.abs(cv[:, 0])   # commanded forward speed

    # Stability metrics
    roll_var  = float(np.var(roll))
    pitch_var = float(np.var(pitch))
    slope_deg = float(np.degrees(np.mean(np.abs(pitch))))

    # Contact pattern
    n_contacts    = fc.sum(axis=1)  # (T,) number of feet in contact
    contact_mean  = float(n_contacts.mean())
    contact_var   = float(n_contacts.var())

    # Joint velocity variance (proxy for terrain roughness)
    jv_var = float(np.var(jv))

    # Slip estimate: if foot is commanded stationary but still moving
    # (approximate: high joint velocity variance when speed is low)
    speed_mean = float(speed.mean()) if speed.size > 0 else 0.0
    slip_risk = float(np.clip(jv_var / (speed_mean + 0.1) * 0.1, 0.0, 1.0))

    # Contact "bounce" = variance in contact signal → terrain hardness
    # Hard surfaces → binary contact (low variance per foot)
    # Soft surfaces → gradual contact (higher variance)
    fc_var_per_foot = float(fc.var(axis=0).mean())
    hardness = float(np.clip(1.0 - fc_var_per_foot * 5.0, 0.0, 1.0))

    return {
        "roll_var":     roll_var,
        "pitch_var":    pitch_var,
        "slope_deg":    slope_deg,
        "contact_mean": contact_mean,
        "contact_var":  contact_var,
        "jv_var":       jv_var,
        "slip_risk":    slip_risk,
        "hardness":     hardness,
        "speed_mean":   speed_mean,
        "roll_mean":    float(np.mean(np.abs(roll))),
    }


# ──────────────────────────────────────────────────────────────────────────────
#  Classifier
# ──────────────────────────────────────────────────────────────────────────────

# Feature-space rules for each terrain type
# Each entry: (terrain_label, min_slope, max_slope, max_jv_var, max_slip, min_hard, max_roll_var)
_RULES = [
    # label               sl_min sl_max jv_max slip_max hard_min roll_var_max
    ("flat_concrete",        0,    2,    0.05,   0.10,    0.80,    0.005),
    ("flat_grass",           0,    5,    0.10,   0.20,    0.60,    0.010),
    ("carpet",               0,    2,    0.06,   0.10,    0.70,    0.006),
    ("wood_floor",           0,    2,    0.07,   0.15,    0.85,    0.006),
    ("gravel",               0,    8,    0.20,   0.30,    0.50,    0.020),
    ("sand",                 0,    8,    0.25,   0.40,    0.20,    0.025),
    ("mud",                  0,    5,    0.30,   0.55,    0.10,    0.030),
    ("ice",                  0,    5,    0.20,   0.70,    0.90,    0.015),
    ("snow_shallow",         0,    5,    0.20,   0.30,    0.30,    0.020),
    ("snow_deep",            0,    5,    0.35,   0.40,    0.15,    0.035),
    ("slope_gentle",        10,   15,    0.12,   0.20,    0.60,    0.015),
    ("slope_steep",         15,   25,    0.18,   0.30,    0.50,    0.025),
    ("rubble",               5,   12,    0.40,   0.35,    0.40,    0.040),
    ("stairs_up",            0,   45,    0.15,   0.10,    0.80,    0.010),
    ("stairs_down",          0,   45,    0.15,   0.10,    0.80,    0.012),
]


class TerrainClassifier:
    """Online terrain classifier using IMU + contact features.

    Args:
        spec: RobotSpec — used for benchmark comparison
    """

    def __init__(self, spec: "RobotSpec"):
        self._spec = spec
        # Build fast lookup: terrain_label → TerrainCapability
        self._cap_map: dict[str, TerrainCapability] = {
            t.terrain: t for t in spec.terrain
        }

    def classify(self, stm: "STM", actual_speed: float = 0.0) -> TerrainEstimate:
        """Classify terrain from current STM window.

        Args:
            stm:          Short-term memory with recent frames
            actual_speed: actual forward speed (m/s) for benchmark

        Returns:
            TerrainEstimate
        """
        feats = _extract_features(stm)
        if not feats:
            return TerrainEstimate(
                label="unknown", confidence=0.0, recommended_gait="trot",
                max_speed=0.5, step_height=0.08, slope_deg=0.0,
                slip_risk=0.0, hardness=0.5, features={},
            )

        slope   = feats["slope_deg"]
        jv_var  = feats["jv_var"]
        slip    = feats["slip_risk"]
        hard    = feats["hardness"]
        rv      = feats["roll_var"]

        # Score each terrain rule
        scores: list[tuple[float, str]] = []
        for (label, sl_min, sl_max, jv_max, slip_max, hard_min, rv_max) in _RULES:
            score = 1.0
            # Slope match
            if not (sl_min <= slope <= sl_max + 5):
                score *= max(0.0, 1.0 - abs(slope - (sl_min + sl_max) / 2) / 10.0)
            # JV variance
            score *= max(0.0, 1.0 - max(0.0, jv_var - jv_max) / (jv_max + 0.01))
            # Slip
            score *= max(0.0, 1.0 - max(0.0, slip - slip_max) / (slip_max + 0.1))
            # Hardness
            score *= max(0.0, 1.0 - max(0.0, hard_min - hard) / (hard_min + 0.1))
            # Roll var
            score *= max(0.0, 1.0 - max(0.0, rv - rv_max) / (rv_max + 0.001))
            scores.append((score, label))

        scores.sort(reverse=True)
        best_score, best_label = scores[0]
        confidence = float(np.clip(best_score, 0.0, 1.0))

        cap = self._cap_map.get(best_label)
        if cap is None:
            cap = self._cap_map.get("flat_concrete")

        return TerrainEstimate(
            label            = best_label,
            confidence       = confidence,
            recommended_gait = cap.recommended_gait if cap else "trot",
            max_speed        = cap.max_speed if cap else 0.5,
            step_height      = cap.foot_clearance if cap else 0.08,
            slope_deg        = slope,
            slip_risk        = slip,
            hardness         = hard,
            features         = feats,
        )

    def benchmark(self, stm: "STM", actual_speed: float, gait_name: str) -> BenchmarkResult:
        """Compare current performance against spec benchmarks.

        Returns:
            BenchmarkResult with ratio and notes.
        """
        bench  = self._spec.benchmarks
        est    = self.classify(stm, actual_speed)
        notes: list[str] = []

        # Expected speed from terrain
        cap = self._cap_map.get(est.label)
        expected = cap.max_speed if cap else bench.flat_speed_ms

        ratio = actual_speed / expected if expected > 0 else 0.0

        feats = est.features
        roll_rms = float(np.sqrt(feats.get("roll_var", 0.0)))
        on_bench = (ratio > 0.6 and roll_rms < bench.trot_stability_roll * 1.5)

        if ratio < 0.5:
            notes.append(f"Speed {actual_speed:.2f}m/s is {ratio*100:.0f}% of expected {expected:.2f}m/s")
        if roll_rms > bench.trot_stability_roll:
            notes.append(f"Roll instability: {roll_rms:.3f}rad (benchmark {bench.trot_stability_roll:.3f}rad)")
        if est.slip_risk > 0.5:
            notes.append(f"High slip risk: {est.slip_risk:.2f} — reduce speed or change gait")
        if est.confidence < 0.4:
            notes.append(f"Low terrain confidence ({est.confidence:.2f}) — using conservative params")

        return BenchmarkResult(
            actual_speed   = actual_speed,
            expected_speed = expected,
            speed_ratio    = ratio,
            roll_stability = roll_rms,
            energy_estimate = 0.0,   # placeholder
            on_benchmark   = on_bench,
            notes          = notes,
        )
