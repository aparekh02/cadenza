"""ActionSequenceBuilder — turns ProposedAction lists into validated, timed
ActionCall sequences.

The builder is the bouncer: it rejects unknown action names, clamps numeric
params into the schema-declared ranges, fills in defaults from the underlying
ActionSpec, and stamps each call with a per-step duration estimate so the
runtime can budget the loop.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from cadenza.actions.library import ActionCall, ActionLibrary, get_library
from cadenza.stack.adapters.base import ProposedAction
from cadenza.stack.vocabulary import ActionDescriptor, ActionVocabulary


@dataclass
class BuiltStep:
    """One validated step ready for the gym adapter."""
    call: ActionCall
    descriptor: ActionDescriptor
    estimated_duration_s: float
    rejected_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class BuiltSequence:
    """The full validated plan."""
    steps: list[BuiltStep] = field(default_factory=list)
    rejected: list[tuple[str, str]] = field(default_factory=list)
    total_estimated_s: float = 0.0

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self):
        return iter(self.steps)

    def calls(self) -> list[ActionCall]:
        return [s.call for s in self.steps]


class ActionSequenceBuilder:
    """Validate ProposedAction lists against the vocabulary and build calls."""

    def __init__(self, vocabulary: ActionVocabulary,
                 library: ActionLibrary | None = None,
                 *,
                 strict: bool = False):
        """
        Args:
            vocabulary: Vocabulary the world model spoke against.
            library: Optional pre-built ActionLibrary; defaults to the global one.
            strict: If True, raises on any rejected action; otherwise drops.
        """
        self.vocabulary = vocabulary
        self.library = library if library is not None else get_library(vocabulary.robot)
        self.strict = strict

    # ── Public API ───────────────────────────────────────────────────────────

    def build(self, proposals: list[ProposedAction]) -> BuiltSequence:
        out = BuiltSequence()
        for prop in proposals:
            try:
                step = self._build_one(prop)
            except ValueError as e:
                if self.strict:
                    raise
                out.rejected.append((prop.name, str(e)))
                continue
            out.steps.append(step)
            out.total_estimated_s += step.estimated_duration_s
        return out

    # ── Internal ─────────────────────────────────────────────────────────────

    def _build_one(self, prop: ProposedAction) -> BuiltStep:
        if prop.name not in self.vocabulary:
            raise ValueError(
                f"action '{prop.name}' not in vocabulary "
                f"({len(self.vocabulary)} known)"
            )
        descriptor = self.vocabulary.get(prop.name)
        spec = self.library.get(prop.name)

        # Validate + clamp params per the descriptor schema.
        params, rejected = self._validate(descriptor, prop.params)

        # Apply spec-level defaults if the model didn't choose a quantity.
        distance_m = float(params.get("distance_m", 0.0)) or float(spec.distance_m)
        rotation_rad = float(params.get("rotation_rad", 0.0)) or float(spec.rotation_rad)
        duration_s = float(params.get("duration_s", 0.0))

        call = ActionCall(
            action_name=prop.name,
            speed=float(params.get("speed", 1.0)),
            extension=float(params.get("extension", 1.0)),
            repeat=int(params.get("repeat", 1)),
            distance_m=distance_m,
            rotation_rad=rotation_rad,
            duration_s=duration_s,
        )

        est = self._estimate_duration(spec, call)
        return BuiltStep(
            call=call,
            descriptor=descriptor,
            estimated_duration_s=est,
            rejected_params=rejected,
        )

    def _validate(
        self,
        descriptor: ActionDescriptor,
        params: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Clamp known params; collect unknown/invalid ones."""
        clean: dict[str, Any] = {}
        rejected: dict[str, Any] = {}
        schemas = {p.name: p for p in descriptor.params}
        for k, v in params.items():
            if k not in schemas:
                rejected[k] = v
                continue
            schema = schemas[k]
            try:
                if schema.type == "float":
                    val = float(v)
                elif schema.type == "int":
                    val = int(v)
                elif schema.type == "bool":
                    val = bool(v)
                else:
                    val = v
            except (TypeError, ValueError):
                rejected[k] = v
                continue
            if schema.min is not None and isinstance(val, (int, float)):
                val = max(val, schema.min)
            if schema.max is not None and isinstance(val, (int, float)):
                val = min(val, schema.max)
            clean[k] = val
        # Apply schema defaults for any params the model didn't specify.
        for k, schema in schemas.items():
            clean.setdefault(k, schema.default)
        return clean, rejected

    def _estimate_duration(self, spec, call: ActionCall) -> float:
        """Rough wall-clock estimate so the runtime can budget steps."""
        if call.duration_s > 0:
            return call.duration_s * max(call.repeat, 1)
        if spec.is_gait:
            # Gait: distance / speed if both known, else fall back to spec.
            speed = float(spec.speed_ms) if spec.speed_ms > 0 else 0.5
            speed *= max(call.speed, 0.1)
            if call.distance_m > 0:
                return (call.distance_m / max(speed, 0.05)) * max(call.repeat, 1)
            if call.rotation_rad > 0:
                yaw_speed = 0.8 * max(call.speed, 0.1)
                return (call.rotation_rad / max(yaw_speed, 0.05)) * max(call.repeat, 1)
            return 2.0 * max(call.repeat, 1)
        # Phase action: sum of phase durations, scaled by speed.
        base = float(spec.total_duration() or 1.5)
        scaled = base / max(call.speed, 0.1)
        return scaled * max(call.repeat, 1)


__all__ = ["ActionSequenceBuilder", "BuiltSequence", "BuiltStep"]
