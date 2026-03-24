"""Minimal LoRA translator — handles exact action names.

The full natural language parser (e.g. "walk forward 2 meters then turn left")
is available in Cadenza Pro. This community version handles exact action names
and basic modifiers.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ActionCall:
    """A single action call with parameters."""
    action_name: str
    speed: float = 1.0
    extension: float = 1.0
    repeat: int = 1
    distance_m: float = 0.0
    rotation_rad: float = 0.0
    duration_s: float = 0.0

    def __repr__(self):
        parts = [self.action_name]
        if self.speed != 1.0:
            parts.append(f"speed={self.speed}")
        if self.distance_m > 0:
            parts.append(f"{self.distance_m}m")
        if self.rotation_rad != 0:
            parts.append(f"rot={self.rotation_rad:.2f}")
        return f"ActionCall({', '.join(parts)})"


class LoRATranslator:
    """Minimal translator: maps action names to ActionCall objects.

    For natural language parsing ("walk forward 2 meters then turn left"),
    upgrade to Cadenza Pro.
    """

    def __init__(self, robot: str = "go1"):
        self.robot = robot
        from cadenza_local.actions import get_library
        self._lib = get_library(robot)

    def translate(self, command: str) -> list[ActionCall]:
        """Translate a command string to ActionCall list.

        Handles:
          - Exact action names: "walk_forward", "jump", "stand"
          - Basic "then"/"and" splitting: "stand then walk_forward then sit"
          - Simple distance: "walk forward 2 meters" → walk_forward(distance_m=2.0)

        For full NL parsing, upgrade to Cadenza Pro.
        """
        import re

        # Split on "then" / "and"
        parts = [p.strip() for p in re.split(r'\s+(?:then|and)\s+', command) if p.strip()]
        calls = []

        for part in parts:
            call = self._parse_single(part)
            if call:
                calls.append(call)

        return calls

    def _parse_single(self, text: str) -> ActionCall | None:
        """Parse a single command fragment."""
        text = text.strip().lower()

        # Try exact match first
        normalized = text.replace(" ", "_").replace("-", "_")
        if normalized in self._lib._actions:
            return ActionCall(action_name=normalized)

        # Try common aliases
        aliases = {
            "stand up": "stand_up",
            "sit down": "sit",
            "lie down": "lie_down",
            "walk forward": "walk_forward",
            "walk backward": "walk_backward",
            "walk backwards": "walk_backward",
            "turn left": "turn_left",
            "turn right": "turn_right",
            "trot forward": "trot_forward",
            "crawl forward": "crawl_forward",
            "side step left": "side_step_left",
            "side step right": "side_step_right",
        }

        import re

        # Check for distance modifier: "walk forward 2 meters"
        dist_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:m(?:eter)?s?)', text)
        distance = float(dist_match.group(1)) if dist_match else 0.0

        # Strip the distance part for matching
        clean = re.sub(r'\d+(?:\.\d+)?\s*(?:m(?:eter)?s?)', '', text).strip()

        for alias, action in aliases.items():
            if clean.startswith(alias):
                return ActionCall(action_name=action, distance_m=distance)

        # Last resort: try as-is
        if clean.replace(" ", "_") in self._lib._actions:
            return ActionCall(action_name=clean.replace(" ", "_"), distance_m=distance)

        return None
