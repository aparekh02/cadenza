"""cadenza.stack.vision — vision sensors used for trajectory recovery.

When the trajectory monitor flags the robot as stuck, the stack switches from
its planned action queue to a perception-driven recovery cycle:

    forward camera (RGB)  ->  DepthEstimator (DepthAnything-V2-Small)
                              ----------------+
                                              v
    target bearing/distance  ->  VisionNavigator (SmolVLM-256M-Instruct)
                                              |
                                              v
                                  one ProposedAction toward the goal

These models are integrated directly into the cadenza stack so any adapter
can use them — the SmolVLA adapter does so for the stairs course, but a new
adapter can call ``VisionNavigator.decide(...)`` the same way.
"""

from cadenza.stack.vision.depth import DepthEstimator
from cadenza.stack.vision.navigator import (
    NavigationDecision,
    VisionNavigator,
)

__all__ = [
    "DepthEstimator",
    "VisionNavigator",
    "NavigationDecision",
]
