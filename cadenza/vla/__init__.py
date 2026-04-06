"""cadenza.vla — Vision-Language-Action guardian for obstacle avoidance.

Monitors the robot's camera feed during action execution using SmolVLM.
Only activates when obstacles are detected. Interjects avoidance actions
from the existing action library.

Usage::

    import cadenza

    go1 = cadenza.go1()
    go1.run([go1.walk_forward(distance_m=5.0)], vla=True)
"""

from cadenza.vla.guardian import VLAGuardian, ObstacleResult
