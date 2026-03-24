"""STM — Short-Term Memory for the locomotion layer.

Rolling window of recent sensor frames. Oldest frames are evicted automatically.
Each frame captures a snapshot of the robot's sensor state at one timestep.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque

import numpy as np


@dataclass
class STMFrame:
    """One timestep of sensor data.

    Fields
    ------
    timestamp   : float        wall-clock seconds (monotonic)
    joint_pos   : (N,) float32 measured joint positions (rad)
    joint_vel   : (N,) float32 measured joint velocities (rad/s)
    imu_rpy     : (3,) float32 roll, pitch, yaw (rad)
    imu_omega   : (3,) float32 angular velocity (rad/s)
    foot_contact: (4,) bool    FL FR RL RR contact flags
    cmd_vel     : (3,) float32 commanded vx, vy, yaw_rate
    extra       : dict         any additional fields (e.g. terrain depth)
    """

    timestamp:    float
    joint_pos:    np.ndarray
    joint_vel:    np.ndarray
    imu_rpy:      np.ndarray
    imu_omega:    np.ndarray
    foot_contact: np.ndarray
    cmd_vel:      np.ndarray
    extra:        dict = field(default_factory=dict)

    def to_vector(self) -> np.ndarray:
        """Flatten all numeric fields to a 1-D float32 vector."""
        parts = [
            self.joint_pos.astype(np.float32),
            self.joint_vel.astype(np.float32),
            self.imu_rpy.astype(np.float32),
            self.imu_omega.astype(np.float32),
            self.foot_contact.astype(np.float32),
            self.cmd_vel.astype(np.float32),
        ]
        return np.concatenate(parts, dtype=np.float32)


class STM:
    """Rolling window of recent STMFrames.

    Args:
        window: max number of frames to retain (FIFO eviction)
    """

    def __init__(self, window: int = 50):
        self._window: int = window
        self._frames: Deque[STMFrame] = deque(maxlen=window)

    # ── Write ──────────────────────────────────────────────────────────────

    def push(self, frame: STMFrame) -> None:
        """Add a new frame; oldest is evicted when window is full."""
        self._frames.append(frame)

    def clear(self) -> None:
        self._frames.clear()

    # ── Read ───────────────────────────────────────────────────────────────

    def latest(self) -> STMFrame | None:
        """Most recent frame, or None if empty."""
        return self._frames[-1] if self._frames else None

    def as_array(self) -> np.ndarray:
        """Stack all frames into (T, D) float32 array. Empty → shape (0,)."""
        if not self._frames:
            return np.empty((0,), dtype=np.float32)
        rows = [f.to_vector() for f in self._frames]
        return np.stack(rows, axis=0).astype(np.float32)   # (T, D)

    def mean_embedding(self) -> np.ndarray:
        """Time-averaged frame vector — shape (D,) float32. Zeros if empty."""
        arr = self.as_array()
        if arr.ndim < 2 or arr.shape[0] == 0:
            # Return zeros of the expected dimensionality (unknown until first push)
            return np.zeros(1, dtype=np.float32)
        return arr.mean(axis=0).astype(np.float32)

    # ── Dunder ─────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._frames)

    def __repr__(self) -> str:
        return f"STM(window={self._window}, filled={len(self._frames)})"
