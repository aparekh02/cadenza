"""WorldModelBridge — sits between the gym observation stream and the world
model adapter.

The bridge owns no model state of its own. It composes (observation + task +
vocabulary + history) into the format the adapter expects, calls the adapter,
and returns the adapter's reply unchanged. Keeping the bridge stateless makes
it easy to swap adapters without touching the orchestrator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cadenza.stack.adapters.base import (
    AdapterReply,
    ProposedAction,
    WorldModelAdapter,
)
from cadenza.stack.vocabulary import ActionVocabulary


@dataclass
class BridgeContext:
    """Snapshot of everything passed to the adapter on one reasoning step."""
    step: int
    observation: dict[str, Any]
    goal: str
    vocabulary: ActionVocabulary
    history: list[ProposedAction] = field(default_factory=list)


class WorldModelBridge:
    """Feeds (observation, task, vocabulary) into a world model.

    Usage::

        bridge = WorldModelBridge(adapter, vocabulary)
        bridge.set_goal("walk to the chair")
        reply = bridge.tick(observation)
    """

    def __init__(
        self,
        adapter: WorldModelAdapter,
        vocabulary: ActionVocabulary,
        *,
        goal: str = "",
        history_limit: int = 32,
    ):
        self.adapter = adapter
        self.vocabulary = vocabulary
        self._goal = goal
        self._history: list[ProposedAction] = []
        self._step = 0
        self._history_limit = history_limit

    # ── Configuration ────────────────────────────────────────────────────────

    @property
    def goal(self) -> str:
        return self._goal

    def set_goal(self, goal: str) -> None:
        self._goal = goal

    @property
    def history(self) -> list[ProposedAction]:
        return list(self._history)

    def reset(self) -> None:
        self._history.clear()
        self._step = 0

    # ── Driving the model ────────────────────────────────────────────────────

    def tick(self, observation: dict[str, Any]) -> AdapterReply:
        """One reasoning step. Pass the latest observation, get a reply."""
        if not self.adapter.is_loaded:
            self.adapter.load()
        ctx = BridgeContext(
            step=self._step,
            observation=observation,
            goal=self._goal,
            vocabulary=self.vocabulary,
            history=list(self._history),
        )
        reply = self.adapter.propose_actions(
            observation=ctx.observation,
            goal=ctx.goal,
            vocabulary=ctx.vocabulary,
            history=ctx.history,
        )
        self._record(reply.actions)
        self._step += 1
        return reply

    def _record(self, actions: list[ProposedAction]) -> None:
        for a in actions:
            self._history.append(a)
        if len(self._history) > self._history_limit:
            self._history = self._history[-self._history_limit:]
