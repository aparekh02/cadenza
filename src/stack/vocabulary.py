"""Action vocabulary — Cadenza's action library exposed as a structured tool
schema for the world model.

The vocabulary is the language the world model speaks back to the stack.
Each action is a named entry with a description and a typed parameter schema.
The world model picks an action and fills in *how much* (distance, rotation,
speed, duration) using its onboard reasoning — it never sees raw motors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from cadenza.actions.library import ActionLibrary, ActionSpec, get_library


# ── Parameter schema ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ParamSchema:
    """One parameter of an action."""
    name: str
    type: str            # "float" | "int" | "bool"
    description: str
    default: Any
    min: float | None = None
    max: float | None = None
    unit: str = ""


@dataclass(frozen=True)
class ActionDescriptor:
    """One entry in the vocabulary the world model sees."""
    name: str
    description: str
    kind: str            # "phase" | "gait"
    params: tuple[ParamSchema, ...]
    default_distance_m: float = 0.0
    default_rotation_rad: float = 0.0
    default_duration_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "kind": self.kind,
            "params": [
                {
                    "name": p.name, "type": p.type, "description": p.description,
                    "default": p.default, "min": p.min, "max": p.max, "unit": p.unit,
                }
                for p in self.params
            ],
            "defaults": {
                "distance_m": self.default_distance_m,
                "rotation_rad": self.default_rotation_rad,
                "duration_s": self.default_duration_s,
            },
        }


# ── Vocabulary ────────────────────────────────────────────────────────────────

@dataclass
class ActionVocabulary:
    """The full set of actions a world model may invoke for a robot."""
    robot: str
    actions: dict[str, ActionDescriptor] = field(default_factory=dict)

    def names(self) -> list[str]:
        return list(self.actions.keys())

    def get(self, name: str) -> ActionDescriptor:
        if name not in self.actions:
            raise KeyError(
                f"Action '{name}' not in vocabulary for {self.robot}. "
                f"Available: {self.names()}"
            )
        return self.actions[name]

    def __contains__(self, name: str) -> bool:
        return name in self.actions

    def __len__(self) -> int:
        return len(self.actions)

    # Serialization formats for different model interfaces ─────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Plain JSON-serializable dict — the canonical model-facing format."""
        return {
            "robot": self.robot,
            "actions": [a.to_dict() for a in self.actions.values()],
        }

    def to_tool_schema(self) -> list[dict[str, Any]]:
        """OpenAI/Anthropic-style function-calling schema.

        Each action becomes one tool the world model can pick. Useful for
        adapters wrapping LLM-based VLAs that already speak tool-use.
        """
        tools = []
        for a in self.actions.values():
            properties: dict[str, Any] = {}
            for p in a.params:
                spec: dict[str, Any] = {
                    "type": "number" if p.type == "float"
                            else "integer" if p.type == "int"
                            else "boolean" if p.type == "bool"
                            else "string",
                    "description": p.description + (f" ({p.unit})" if p.unit else ""),
                }
                if p.min is not None:
                    spec["minimum"] = p.min
                if p.max is not None:
                    spec["maximum"] = p.max
                properties[p.name] = spec
            tools.append({
                "name": a.name,
                "description": a.description,
                "input_schema": {
                    "type": "object",
                    "properties": properties,
                    "required": [],
                },
            })
        return tools

    def to_prompt(self) -> str:
        """Compact human-readable text format for plain-text LLM contexts."""
        lines = [f"Cadenza action vocabulary — robot: {self.robot}"]
        for a in self.actions.values():
            params = ", ".join(
                f"{p.name}: {p.type}={p.default}{p.unit}" for p in a.params
            )
            lines.append(f"  {a.name}({params}) — {a.description}")
        return "\n".join(lines)


# ── Builder ───────────────────────────────────────────────────────────────────

# Common parameter schemas shared across actions.
_SPEED = ParamSchema(
    name="speed", type="float",
    description="Speed multiplier (1.0 = nominal).",
    default=1.0, min=0.1, max=2.5,
)
_EXTENSION = ParamSchema(
    name="extension", type="float",
    description="Joint-displacement multiplier from the stand pose.",
    default=1.0, min=0.5, max=1.5,
)
_REPEAT = ParamSchema(
    name="repeat", type="int",
    description="How many times to repeat this action back-to-back.",
    default=1, min=1, max=10,
)
_DISTANCE = ParamSchema(
    name="distance_m", type="float",
    description="Distance to travel.",
    default=0.0, min=0.0, max=20.0, unit="m",
)
_ROTATION = ParamSchema(
    name="rotation_rad", type="float",
    description="Rotation angle.",
    default=0.0, min=0.0, max=2 * math.pi, unit="rad",
)
_DURATION = ParamSchema(
    name="duration_s", type="float",
    description="Duration override; 0 uses the action default.",
    default=0.0, min=0.0, max=60.0, unit="s",
)


def _params_for(spec: ActionSpec) -> tuple[ParamSchema, ...]:
    """Pick which parameters are meaningful for this action."""
    base: list[ParamSchema] = [_SPEED, _EXTENSION, _REPEAT]
    if spec.is_gait:
        # gaits are time/distance-driven
        base.append(_DISTANCE)
        if spec.rotation_rad > 0 or "turn" in spec.name:
            base.append(_ROTATION)
        base.append(_DURATION)
    else:
        # phase actions have intrinsic durations; expose duration override only
        base.append(_DURATION)
    return tuple(base)


def _describe(spec: ActionSpec) -> str:
    """Augment the raw description with hints about defaults."""
    pieces = [spec.description.strip()]
    if spec.distance_m > 0:
        pieces.append(f"default distance ~{spec.distance_m} m")
    if spec.rotation_rad > 0:
        pieces.append(f"default rotation ~{math.degrees(spec.rotation_rad):.0f}°")
    if spec.duration_s > 0 and not spec.is_gait:
        pieces.append(f"runs ~{spec.duration_s:.1f}s")
    return " ".join(pieces)


def build_vocabulary(robot: str, library: ActionLibrary | None = None) -> ActionVocabulary:
    """Build the action vocabulary for `robot`.

    Args:
        robot: "go1" | "go2" | "g1".
        library: Optional pre-built library; if None, fetched via get_library.
    """
    lib = library if library is not None else get_library(robot)
    descriptors: dict[str, ActionDescriptor] = {}
    for name in lib.list_actions():
        spec = lib.get(name)
        descriptors[name] = ActionDescriptor(
            name=name,
            description=_describe(spec),
            kind="gait" if spec.is_gait else "phase",
            params=_params_for(spec),
            default_distance_m=spec.distance_m,
            default_rotation_rad=spec.rotation_rad,
            default_duration_s=spec.duration_s,
        )
    return ActionVocabulary(robot=robot, actions=descriptors)
