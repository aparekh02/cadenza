"""Stack runtime — wires the detector, vocabulary, bridge, builder, and gym
adapter into one perceive-reason-act loop.

This is the top of the stack: a single ``run(...)`` function that lets the
client say "drive this robot toward this goal" without knowing anything about
which world model is loaded, which adapter to use, or how Cadenza talks to
MuJoCo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cadenza.actions.library import get_library
from cadenza.stack.adapters.base import (
    AdapterReply,
    ProposedAction,
    WorldModelAdapter,
)
from cadenza.stack.bridge import WorldModelBridge
from cadenza.stack.builder import ActionSequenceBuilder, BuiltSequence
from cadenza.stack.detector import (
    WorldModelHandle,
    detect_world_model,
)
from cadenza.stack.gym_adapter import GymAdapter, Observation
from cadenza.stack.vocabulary import ActionVocabulary, build_vocabulary


@dataclass
class StackResult:
    """Returned by Stack.run / cadenza.stack.run."""
    handle: WorldModelHandle
    vocabulary: ActionVocabulary
    executed: list[BuiltSequence] = field(default_factory=list)
    final_observation: Observation | None = None
    notes: list[str] = field(default_factory=list)
    done: bool = False

    @property
    def total_actions(self) -> int:
        return sum(len(seq) for seq in self.executed)


class Stack:
    """End-to-end controller assembled from stack components.

    Typical usage is via the module-level ``run`` helper, but you can build a
    Stack directly to keep state across calls or to swap parts (e.g. plug in a
    custom adapter for testing).

    Example::

        stack = Stack(robot="go1", goal="walk forward 2 meters then sit")
        result = stack.run()
    """

    def __init__(
        self,
        robot: str = "go1",
        *,
        goal: str = "",
        target: tuple[float, float] | None = None,
        world_model: str | type[WorldModelAdapter] | WorldModelHandle | None = None,
        root: str | Path = ".",
        max_iterations: int = 30,
        headless: bool = False,
        render_camera: bool = True,
        xml_path: str | None = None,
        verbose: bool = True,
    ):
        self.robot = robot
        self.goal = goal
        self.target = (
            (float(target[0]), float(target[1])) if target is not None else None
        )
        self.root = root
        self.max_iterations = max_iterations
        self.headless = headless
        self.render_camera = render_camera
        self.xml_path = xml_path
        self.verbose = verbose

        # 1. Detect (or accept) the world model.
        self.handle = self._resolve_handle(world_model, root)
        self.adapter: WorldModelAdapter = self.handle.build()

        # 2. Build action vocabulary.
        self.library = get_library(robot)
        self.vocabulary = build_vocabulary(robot, library=self.library)

        # 3. Bridge + builder.
        self.bridge = WorldModelBridge(self.adapter, self.vocabulary, goal=goal)
        self.builder = ActionSequenceBuilder(self.vocabulary, library=self.library)

        # 4. Gym adapter is constructed lazily on .run() so .__init__ stays cheap.
        self._gym: GymAdapter | None = None

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _resolve_handle(
        self,
        world_model,
        root: str | Path,
    ) -> WorldModelHandle:
        if isinstance(world_model, WorldModelHandle):
            return world_model
        if world_model is None:
            return detect_world_model(root)
        # A string adapter name or class — wrap into a handle directly.
        from cadenza.stack.adapters.base import get_adapter
        cls = world_model if isinstance(world_model, type) else get_adapter(world_model)
        return WorldModelHandle(adapter_cls=cls, checkpoint=None, source="explicit")

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"  [stack] {msg}")

    # ── Public ───────────────────────────────────────────────────────────────

    def set_goal(self, goal: str) -> None:
        self.goal = goal
        self.bridge.set_goal(goal)

    def run(self) -> StackResult:
        """Run the perceive-reason-act loop until the adapter says done."""
        result = StackResult(handle=self.handle, vocabulary=self.vocabulary)

        self._log(
            f"world model: {self.handle.name} (source={self.handle.source}, "
            f"checkpoint={self.handle.checkpoint})"
        )
        self._log(
            f"vocabulary: {len(self.vocabulary)} actions for {self.robot}; "
            f"goal=\"{self.goal}\""
        )

        self._gym = GymAdapter(
            robot=self.robot,
            xml_path=self.xml_path,
            headless=self.headless,
            render_camera=self.render_camera,
        )
        observation = self._gym.reset()

        def _obs_dict():
            d = observation.to_dict()
            if self.target is not None:
                d["target_xy"] = self.target
            return d

        try:
            for it in range(self.max_iterations):
                reply: AdapterReply = self.bridge.tick(_obs_dict())
                if reply.note:
                    result.notes.append(reply.note)
                self._log(
                    f"iter {it + 1}/{self.max_iterations}: "
                    f"{len(reply.actions)} proposed | done={reply.done} "
                    f"| {reply.note}"
                )

                if not reply.actions:
                    if reply.done:
                        result.done = True
                        break
                    self._log("no actions proposed; breaking")
                    break

                plan = self.builder.build(reply.actions)
                if plan.rejected:
                    for name, why in plan.rejected:
                        self._log(f"rejected '{name}': {why}")

                for built in plan.steps:
                    self._log(
                        f"  -> {built.call.action_name} "
                        f"(~{built.estimated_duration_s:.1f}s)"
                    )
                    observation, info = self._gym.step(built.call)
                    if not info.get("ok", True):
                        result.notes.append(
                            f"{built.call.action_name} aborted at step {info.get('step')}"
                        )

                result.executed.append(plan)
                if reply.done:
                    result.done = True
                    break
        finally:
            result.final_observation = self._gym._observe() if self._gym.is_open else None
            self._gym.close()

        self._log(
            f"finished: {result.total_actions} actions executed, "
            f"done={result.done}"
        )
        return result


# ── Module-level convenience entry point ─────────────────────────────────────

def run(
    robot: str = "go1",
    goal: str = "",
    *,
    target: tuple[float, float] | None = None,
    world_model: str | type[WorldModelAdapter] | WorldModelHandle | None = None,
    root: str | Path = ".",
    max_iterations: int = 30,
    headless: bool = False,
    render_camera: bool = True,
    xml_path: str | None = None,
    verbose: bool = True,
) -> StackResult:
    """Run the Cadenza stack end-to-end.

    Example::

        import cadenza
        cadenza.stack.run(
            robot="go1",
            goal="walk forward 2 meters then sit",
            target=(-5.5, 0.0),     # optional — enables vision recovery
        )
    """
    stack = Stack(
        robot=robot,
        goal=goal,
        target=target,
        world_model=world_model,
        root=root,
        max_iterations=max_iterations,
        headless=headless,
        render_camera=render_camera,
        xml_path=xml_path,
        verbose=verbose,
    )
    return stack.run()


__all__ = ["Stack", "StackResult", "run"]
