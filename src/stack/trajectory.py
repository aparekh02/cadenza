"""Trajectory progress monitor.

Tracks robot position over time relative to a target and flags when the
robot is *stuck* — distance to the target hasn't dropped by at least
``min_progress_m`` over the last ``window`` ticks. The stack uses this to
decide when to invoke the vision-based recovery navigator.

Lightweight and adapter-agnostic: any adapter can carry one of these.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field


@dataclass
class TrajectoryMonitor:
    """Sliding-window distance tracker. ``update`` once per tick."""
    target_xy: tuple[float, float] | None = None
    window: int = 4                      # ticks of history compared for progress
    min_progress_m: float = 0.10         # require this much closer over the window
    arrival_distance_m: float = 0.45     # within this we count as "arrived"
    history: deque[float] = field(default_factory=lambda: deque(maxlen=12))

    def __post_init__(self) -> None:
        if self.target_xy is not None:
            self.target_xy = (float(self.target_xy[0]), float(self.target_xy[1]))

    def update(self, robot_xy: tuple[float, float]) -> None:
        if self.target_xy is None:
            return
        dx = self.target_xy[0] - float(robot_xy[0])
        dy = self.target_xy[1] - float(robot_xy[1])
        self.history.append(math.hypot(dx, dy))

    @property
    def distance_m(self) -> float | None:
        return self.history[-1] if self.history else None

    @property
    def at_target(self) -> bool:
        d = self.distance_m
        return d is not None and d <= self.arrival_distance_m

    @property
    def is_stuck(self) -> bool:
        if self.target_xy is None or len(self.history) < self.window + 1:
            return False
        recent = self.history[-1]
        baseline = self.history[-1 - self.window]
        return (baseline - recent) < self.min_progress_m

    def progress_summary(self) -> str:
        if not self.history:
            return "no progress data"
        d = self.history[-1]
        if len(self.history) > self.window:
            delta = self.history[-1 - self.window] - d
            return f"dist={d:.2f}m  Δ{self.window}={delta:+.2f}m"
        return f"dist={d:.2f}m  (warming up)"

    def reset_after_recovery(self) -> None:
        """Call this after acting on a recovery decision so we don't fire again
        for a few ticks while the new action is in flight."""
        if self.history:
            last = self.history[-1]
            self.history.clear()
            self.history.append(last)
