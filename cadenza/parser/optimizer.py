"""Action optimizer — sensor-driven parameter adjustment (stub).

Pass-through implementation. The full sensor-driven optimizer
(terrain adaptation, speed adjustment, gait swapping) is in Cadenza Pro.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SensorSnapshot:
    """Sensor data for the optimizer."""
    slope: float = 0.0
    roughness: float = 0.0
    friction: float = 1.0
    roll: float = 0.0
    pitch: float = 0.0
    height: float = 0.0
    stability: float = 1.0


class ActionOptimizer:
    """Pass-through optimizer (Community Edition).

    For sensor-driven gait adaptation, upgrade to Cadenza Pro.
    """

    def __init__(self, robot: str = "go1"):
        self.robot = robot

    def classify(self, sensors: SensorSnapshot | dict | None = None) -> str:
        """Classify terrain/environment. Returns 'normal' in community edition."""
        return "normal"

    def optimize(self, action_name: str, params: dict,
                 sensors: SensorSnapshot | None = None) -> dict:
        """Optimize action parameters. Pass-through in community edition."""
        return params
