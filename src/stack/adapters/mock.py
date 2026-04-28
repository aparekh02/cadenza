"""MockAdapter — deterministic fallback adapter.

Used when no real world model is found at the project root or HF cache, and
for stack tests. It parses the natural-language goal with the existing
``CommandParser`` and emits one ProposedAction per parsed call. After
executing, it sets ``done=True``.

This lets ``cadenza.stack.run(robot=..., goal="walk forward 2 meters")`` work
end-to-end with no model installed — the goal text *is* the plan.
"""

from __future__ import annotations

from pathlib import Path

from cadenza.actions.library import ActionCall
from cadenza.parser.translator import CommandParser
from cadenza.stack.adapters.base import (
    AdapterReply,
    ProposedAction,
    WorldModelAdapter,
    register_adapter,
)
from cadenza.stack.vocabulary import ActionVocabulary


@register_adapter
class MockAdapter(WorldModelAdapter):
    """Heuristic adapter — no neural net. Treats `goal` as a Cadenza command."""

    name = "mock"
    description = "Heuristic adapter — parses the goal string with CommandParser."

    @classmethod
    def detect(cls, root: Path) -> Path | None:
        # Mock never auto-detects; it's only used as the configured fallback.
        return None

    def _load_impl(self) -> None:
        # Nothing to load.
        return

    def propose_actions(
        self,
        observation: dict,
        goal: str,
        vocabulary: ActionVocabulary,
        history: list[ProposedAction] | None = None,
    ) -> AdapterReply:
        # If we already executed something, we're done — single-shot planner.
        if history:
            return AdapterReply(actions=[], done=True, note="mock: plan exhausted")

        if not goal or not goal.strip():
            # No goal: stand still, then exit.
            return AdapterReply(
                actions=[ProposedAction(name="stand", params={})],
                done=True,
                note="mock: no goal supplied",
            )

        parser = CommandParser(vocabulary.robot)
        calls: list[ActionCall] = parser.parse(goal)
        actions: list[ProposedAction] = []
        for call in calls:
            if call.action_name not in vocabulary:
                # Skip unknown actions silently — keeps the mock robust.
                continue
            params: dict = {"speed": call.speed, "repeat": call.repeat}
            if call.distance_m > 0:
                params["distance_m"] = call.distance_m
            if call.rotation_rad != 0:
                params["rotation_rad"] = call.rotation_rad
            if call.duration_s > 0:
                params["duration_s"] = call.duration_s
            actions.append(ProposedAction(
                name=call.action_name,
                params=params,
                rationale="parsed from goal",
            ))

        if not actions:
            actions = [ProposedAction(name="stand", params={},
                                      rationale="goal unparseable; standing")]

        return AdapterReply(
            actions=actions,
            done=True,
            note=f"mock: parsed {len(actions)} actions from goal",
        )
