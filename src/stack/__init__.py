"""cadenza.stack — full stack for world-model-driven robot control.

The stack lets a robotics world model (e.g. pi_0.5, OpenVLA, Octo) drive a
Cadenza-supported robot using the action library as its action space, instead
of low-level motor torques. The world model reasons about *which* action and
*how much* to move; the stack handles motor execution.

Client usage::

    import cadenza

    # Auto-detects a world model at the project root and runs the loop.
    cadenza.stack.run(robot="go1", goal="walk to the chair and sit")

Pipeline (matches src/stack architecture):

    Client
       |
       v
    Model detector  ->  finds world model, loads adapter
       |
       v
    World-model bridge  <-->  Cadenza action library
       |  (model reasons over obs + task using action vocabulary)
       v
    Action sequence builder  ->  validated, timed ActionCall series
       |
       v
    Gym adapter  ->  Robot gym (sim/real)
       ^
       |  observations
       +---------------------------------------------------------+
"""

from __future__ import annotations

from cadenza.stack.detector import (
    WorldModelHandle,
    detect_world_model,
    register_world_model,
)
from cadenza.stack.vocabulary import ActionVocabulary, build_vocabulary
from cadenza.stack.bridge import WorldModelBridge
from cadenza.stack.builder import ActionSequenceBuilder, BuiltSequence
from cadenza.stack.gym_adapter import GymAdapter, Observation
from cadenza.stack.runtime import Stack, run
from cadenza.stack.trajectory import TrajectoryMonitor
from cadenza.stack.modalities import (
    Modality,
    ModalityResult,
    register_modality,
    get_modality,
    list_modalities,
)

__all__ = [
    "run",
    "Stack",
    "WorldModelHandle",
    "detect_world_model",
    "register_world_model",
    "ActionVocabulary",
    "build_vocabulary",
    "WorldModelBridge",
    "ActionSequenceBuilder",
    "BuiltSequence",
    "GymAdapter",
    "Observation",
    "TrajectoryMonitor",
    "Modality",
    "ModalityResult",
    "register_modality",
    "get_modality",
    "list_modalities",
]
