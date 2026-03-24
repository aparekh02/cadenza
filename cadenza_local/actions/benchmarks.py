"""benchmarks.py — Action effectiveness measurement and memory.

Every action execution is scored on whether it actually achieved what it
was supposed to. This catches problems like:
  - "walk" but the body was dragging on its belly (shoulders not supporting)
  - "stand" but trunk height never reached target
  - "turn" but the robot tipped over mid-rotation

Benchmark data is accumulated across runs and persisted to disk. The LoRA
optimizer and future action library tuning can read these to adjust params.

Measured per action execution:
  1. body_height    — avg trunk z during action vs expected height
  2. stability      — avg |roll| + |pitch| during action (lower = better)
  3. upright_ratio  — fraction of steps where trunk_z > 80% of expected height
  4. joint_error    — avg |target - actual| across all joints (tracking quality)
  5. foot_contacts  — avg number of feet on ground
  6. distance_err   — actual distance vs expected (for gait actions)
  7. energy         — total |torque| applied (lower = more efficient)
  8. completion     — did the action finish without safety abort? (bool)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np


@dataclass
class ActionBenchmark:
    """Measurements from a single action execution."""
    action_name:    str
    robot:          str
    timestamp:      float = 0.0

    # Body height metrics
    avg_body_height:    float = 0.0   # m — average trunk z during action
    min_body_height:    float = 0.0   # m — lowest trunk z
    max_body_height:    float = 0.0   # m — highest trunk z
    expected_height:    float = 0.265 # m — what height the action expects
    upright_ratio:      float = 0.0   # fraction of steps where z > 80% expected

    # Stability metrics
    avg_roll:           float = 0.0   # rad — average |roll|
    avg_pitch:          float = 0.0   # rad — average |pitch|
    max_roll:           float = 0.0   # rad — peak |roll|
    max_pitch:          float = 0.0   # rad — peak |pitch|
    stability_score:    float = 0.0   # 0..1 — 1 = perfectly stable

    # Joint tracking
    avg_joint_error:    float = 0.0   # rad — avg |target - actual| per joint
    max_joint_error:    float = 0.0   # rad — worst single-joint error
    shoulder_error:     float = 0.0   # rad — avg error on thigh joints specifically
    knee_error:         float = 0.0   # rad — avg error on calf joints

    # Foot contacts
    avg_feet_on_ground: float = 0.0   # average number of feet in contact
    min_feet_on_ground: float = 0.0   # lowest contact count

    # Distance / rotation
    distance_actual:    float = 0.0   # m — how far the robot actually moved
    distance_expected:  float = 0.0   # m — how far it was supposed to move
    distance_error:     float = 0.0   # ratio: |actual - expected| / expected

    # Energy
    total_energy:       float = 0.0   # sum of |torque| across all steps and joints
    avg_torque:         float = 0.0   # average |torque| per joint per step

    # Result
    completed:          bool  = True  # did the action finish without abort?
    n_steps:            int   = 0     # total control steps executed
    duration_s:         float = 0.0   # wall-clock time

    @property
    def grade(self) -> str:
        """Letter grade based on overall quality."""
        score = self.overall_score
        if score >= 0.9:
            return "A"
        if score >= 0.75:
            return "B"
        if score >= 0.5:
            return "C"
        if score >= 0.25:
            return "D"
        return "F"

    @property
    def overall_score(self) -> float:
        """0..1 composite score."""
        if not self.completed:
            return 0.0
        # Weighted combination
        w_upright   = 0.30   # most important: was the body actually up?
        w_stability = 0.25   # was it stable?
        w_tracking  = 0.20   # did joints reach targets?
        w_distance  = 0.15   # did it go the right distance?
        w_energy    = 0.10   # was it efficient?

        s_upright   = self.upright_ratio
        s_stability = self.stability_score
        s_tracking  = max(0.0, 1.0 - self.avg_joint_error / 0.5)  # 0.5 rad = zero score
        s_distance  = max(0.0, 1.0 - self.distance_error) if self.distance_expected > 0 else 1.0
        s_energy    = max(0.0, 1.0 - self.avg_torque / 20.0)  # 20 Nm avg = zero score

        return (w_upright * s_upright +
                w_stability * s_stability +
                w_tracking * s_tracking +
                w_distance * s_distance +
                w_energy * s_energy)

    def summary(self) -> str:
        """One-line summary."""
        return (
            f"[{self.grade}] {self.action_name}: "
            f"height={self.avg_body_height:.3f}m "
            f"(upright {self.upright_ratio:.0%}), "
            f"stability={self.stability_score:.2f}, "
            f"joint_err={self.avg_joint_error:.3f}rad, "
            f"shoulder_err={self.shoulder_error:.3f}rad, "
            f"score={self.overall_score:.2f}"
        )

    def problems(self) -> list[str]:
        """List specific problems detected."""
        issues = []
        if self.upright_ratio < 0.5:
            issues.append(
                f"Body dragging: only upright {self.upright_ratio:.0%} of the time "
                f"(avg height {self.avg_body_height:.3f}m vs expected {self.expected_height:.3f}m)"
            )
        if self.shoulder_error > 0.15:
            issues.append(
                f"Shoulders not supporting: thigh joint error {self.shoulder_error:.3f}rad "
                f"(thighs not reaching target → body sags)"
            )
        if self.knee_error > 0.15:
            issues.append(
                f"Knees not tracking: calf joint error {self.knee_error:.3f}rad"
            )
        if self.max_roll > 0.40:
            issues.append(f"Excessive roll: peak {self.max_roll:.2f}rad")
        if self.max_pitch > 0.40:
            issues.append(f"Excessive pitch: peak {self.max_pitch:.2f}rad")
        if self.distance_expected > 0 and self.distance_error > 0.5:
            issues.append(
                f"Distance off: moved {self.distance_actual:.2f}m "
                f"vs expected {self.distance_expected:.2f}m"
            )
        if not self.completed:
            issues.append("Action aborted by safety check")
        return issues


class BenchmarkRecorder:
    """Records sensor data during action execution to produce an ActionBenchmark."""

    def __init__(self, action_name: str, robot: str, expected_height: float = 0.265,
                 expected_distance: float = 0.0):
        self._action = action_name
        self._robot = robot
        self._expected_height = expected_height
        self._expected_distance = expected_distance
        self._start_time = time.monotonic()
        self._start_pos: np.ndarray | None = None

        # Accumulation buffers
        self._heights: list[float] = []
        self._rolls: list[float] = []
        self._pitches: list[float] = []
        self._joint_errors: list[float] = []
        self._shoulder_errors: list[float] = []
        self._knee_errors: list[float] = []
        self._foot_counts: list[float] = []
        self._torques: list[float] = []

    def set_start_position(self, pos_xy: np.ndarray) -> None:
        self._start_pos = pos_xy.copy()

    def record_step(
        self,
        trunk_z: float,
        roll: float,
        pitch: float,
        q_target: np.ndarray,
        q_actual: np.ndarray,
        foot_contacts: np.ndarray,
        torques: np.ndarray,
    ) -> None:
        """Record one control step's measurements."""
        self._heights.append(trunk_z)
        self._rolls.append(abs(roll))
        self._pitches.append(abs(pitch))

        # Per-joint tracking error
        err = np.abs(q_target - q_actual)
        self._joint_errors.append(float(np.mean(err)))

        # Shoulder (thigh) error specifically — indices 1,4,7,10
        thigh_err = err[[1, 4, 7, 10]]
        self._shoulder_errors.append(float(np.mean(thigh_err)))

        # Knee (calf) error — indices 2,5,8,11
        calf_err = err[[2, 5, 8, 11]]
        self._knee_errors.append(float(np.mean(calf_err)))

        self._foot_counts.append(float(np.sum(foot_contacts)))
        self._torques.append(float(np.sum(np.abs(torques))))

    def finish(self, end_pos_xy: np.ndarray, completed: bool = True) -> ActionBenchmark:
        """Compute final benchmark from accumulated data."""
        n = max(len(self._heights), 1)
        heights = np.array(self._heights) if self._heights else np.array([0.0])
        rolls = np.array(self._rolls) if self._rolls else np.array([0.0])
        pitches = np.array(self._pitches) if self._pitches else np.array([0.0])

        # Upright ratio: fraction of steps where body was at ≥80% expected height
        upright_threshold = self._expected_height * 0.80
        upright_ratio = float(np.mean(heights >= upright_threshold)) if len(heights) > 0 else 0.0

        # Stability score: 1.0 when roll+pitch = 0, 0.0 when avg > 0.5 rad
        avg_tilt = float(np.mean(rolls + pitches))
        stability = max(0.0, 1.0 - avg_tilt / 0.5)

        # Distance
        dist_actual = 0.0
        if self._start_pos is not None:
            dist_actual = float(np.linalg.norm(end_pos_xy - self._start_pos))
        dist_err = (abs(dist_actual - self._expected_distance) /
                    max(self._expected_distance, 0.01)) if self._expected_distance > 0 else 0.0

        return ActionBenchmark(
            action_name=self._action,
            robot=self._robot,
            timestamp=time.time(),
            avg_body_height=float(np.mean(heights)),
            min_body_height=float(np.min(heights)),
            max_body_height=float(np.max(heights)),
            expected_height=self._expected_height,
            upright_ratio=upright_ratio,
            avg_roll=float(np.mean(rolls)),
            avg_pitch=float(np.mean(pitches)),
            max_roll=float(np.max(rolls)),
            max_pitch=float(np.max(pitches)),
            stability_score=stability,
            avg_joint_error=float(np.mean(self._joint_errors)) if self._joint_errors else 0.0,
            max_joint_error=float(np.max(self._joint_errors)) if self._joint_errors else 0.0,
            shoulder_error=float(np.mean(self._shoulder_errors)) if self._shoulder_errors else 0.0,
            knee_error=float(np.mean(self._knee_errors)) if self._knee_errors else 0.0,
            avg_feet_on_ground=float(np.mean(self._foot_counts)) if self._foot_counts else 0.0,
            min_feet_on_ground=float(np.min(self._foot_counts)) if self._foot_counts else 0.0,
            distance_actual=dist_actual,
            distance_expected=self._expected_distance,
            distance_error=dist_err,
            total_energy=float(np.sum(self._torques)) if self._torques else 0.0,
            avg_torque=(float(np.mean(self._torques)) / 12.0) if self._torques else 0.0,
            completed=completed,
            n_steps=n,
            duration_s=time.monotonic() - self._start_time,
        )


class BenchmarkMemory:
    """Persistent memory of action benchmarks across runs.

    Stores the last N benchmarks per action and computes aggregate stats.
    Saved to disk as JSON so future runs can read historical performance.
    """

    def __init__(self, max_per_action: int = 20):
        self._max = max_per_action
        self._history: dict[str, list[dict]] = {}   # action_name → list of benchmark dicts
        self._path: Path | None = None

    def record(self, bench: ActionBenchmark) -> None:
        """Add a benchmark result."""
        name = bench.action_name
        if name not in self._history:
            self._history[name] = []
        self._history[name].append(asdict(bench))
        # Keep only last N
        if len(self._history[name]) > self._max:
            self._history[name] = self._history[name][-self._max:]

    def get_history(self, action_name: str) -> list[ActionBenchmark]:
        """Get all benchmarks for an action."""
        entries = self._history.get(action_name, [])
        results = []
        for d in entries:
            b = ActionBenchmark(**{k: v for k, v in d.items()
                                   if k in ActionBenchmark.__dataclass_fields__})
            results.append(b)
        return results

    def avg_score(self, action_name: str) -> float:
        """Average overall score for an action across history."""
        history = self.get_history(action_name)
        if not history:
            return 0.0
        return sum(b.overall_score for b in history) / len(history)

    def recurring_problems(self, action_name: str) -> list[str]:
        """Problems that appear in >50% of recent executions."""
        history = self.get_history(action_name)
        if len(history) < 2:
            return []
        # Count problem occurrences
        problem_counts: dict[str, int] = {}
        for b in history[-10:]:  # last 10
            for p in b.problems():
                # Normalize problem text to first few words for grouping
                key = " ".join(p.split()[:3])
                problem_counts[key] = problem_counts.get(key, 0) + 1
        threshold = len(history[-10:]) * 0.5
        return [k for k, v in problem_counts.items() if v >= threshold]

    def report(self) -> str:
        """Full report of all action benchmarks."""
        lines = ["Action Benchmark Report", "=" * 50]
        for name, entries in sorted(self._history.items()):
            history = self.get_history(name)
            if not history:
                continue
            latest = history[-1]
            avg = self.avg_score(name)
            lines.append(f"\n{name} ({len(history)} runs, avg score={avg:.2f}):")
            lines.append(f"  Latest: {latest.summary()}")
            problems = latest.problems()
            if problems:
                for p in problems:
                    lines.append(f"  !! {p}")
            recurring = self.recurring_problems(name)
            if recurring:
                lines.append(f"  Recurring issues: {', '.join(recurring)}")
        return "\n".join(lines)

    def save(self, path: Path) -> None:
        """Save to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._history, indent=2))
        self._path = path

    def load(self, path: Path) -> None:
        """Load from JSON."""
        if path.exists():
            self._history = json.loads(path.read_text())
            self._path = path
