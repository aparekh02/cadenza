"""Command parser — maps natural language to action sequences.

Handles exact action names and basic modifiers like distance.
"""

from __future__ import annotations

import re

from cadenza.actions.library import ActionCall


class CommandParser:
    """Parse command strings into ActionCall sequences.

    Usage::

        parser = CommandParser("go1")
        calls = parser.parse("walk forward 2 meters then turn left then jump")
    """

    def __init__(self, robot: str = "go1"):
        self.robot = robot
        from cadenza.actions import get_library
        self._lib = get_library(robot)

    def parse(self, command: str) -> list[ActionCall]:
        """Parse a command string into a list of ActionCall objects.

        Supports:
          - Exact action names: "walk_forward", "jump", "stand"
          - "then"/"and" splitting: "stand then walk_forward then sit"
          - Distance modifier: "walk forward 2 meters"
        """
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

        normalized = text.replace(" ", "_").replace("-", "_")
        if normalized in self._lib._actions:
            return ActionCall(action_name=normalized)

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

        dist_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:m(?:eter)?s?)', text)
        distance = float(dist_match.group(1)) if dist_match else 0.0

        clean = re.sub(r'\d+(?:\.\d+)?\s*(?:m(?:eter)?s?)', '', text).strip()

        for alias, action in aliases.items():
            if clean.startswith(alias):
                return ActionCall(action_name=action, distance_m=distance)

        if clean.replace(" ", "_") in self._lib._actions:
            return ActionCall(action_name=clean.replace(" ", "_"), distance_m=distance)

        return None


# Backward-compatible alias
LoRATranslator = CommandParser
