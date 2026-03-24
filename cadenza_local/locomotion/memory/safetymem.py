"""SafetyMem — Safety constraint memory.

A set of declarative rules derived from past failures. Each rule defines
a triggering condition on the current state and the corrective action to
apply (override a field of the LocoCommand).

Loaded read-only from snapshot.json at runtime.
Built offline by examples/unitree_go1/backend/build_safetymem.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SafetyRule:
    """One safety constraint.

    Trigger condition is expressed as axis-aligned bounds on a named
    sensor field (e.g. imu_pitch > 0.4 rad).  If triggered, `override`
    specifies the corrective action as a dict of LocoCommand field names
    to new values (e.g. {"cmd_vx": 0.0, "cmd_vy": 0.0}).

    Fields
    ------
    name      : str   human-readable rule name
    field     : str   sensor field name (STMFrame attribute)
    axis      : int   index within the field's array (-1 for scalar)
    min_val   : float lower bound; -inf to ignore
    max_val   : float upper bound; +inf to ignore
    override  : dict  LocoCommand field → corrective value
    priority  : int   higher priority rules win conflicts (default 0)
    """

    name:     str
    field:    str
    axis:     int
    min_val:  float = -np.inf
    max_val:  float =  np.inf
    override: dict  = field(default_factory=dict)
    priority: int   = 0

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SafetyRule":
        return cls(
            name     = str(d["name"]),
            field    = str(d["field"]),
            axis     = int(d.get("axis", -1)),
            min_val  = float(d.get("min_val", -np.inf)),
            max_val  = float(d.get("max_val",  np.inf)),
            override = dict(d.get("override", {})),
            priority = int(d.get("priority", 0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name":     self.name,
            "field":    self.field,
            "axis":     self.axis,
            "min_val":  self.min_val  if not np.isinf(self.min_val) else None,
            "max_val":  self.max_val  if not np.isinf(self.max_val) else None,
            "override": self.override,
            "priority": self.priority,
        }

    def is_triggered(self, frame_value: Any) -> bool:
        """Check whether this rule fires for a given sensor value."""
        if frame_value is None:
            return False
        try:
            arr = np.asarray(frame_value, dtype=np.float32)
            val = float(arr) if arr.ndim == 0 else float(arr[self.axis])
        except (IndexError, TypeError, ValueError):
            return False
        return val < self.min_val or val > self.max_val


@dataclass
class SafetyCheckResult:
    """Result of SafetyMem.check().

    Fields
    ------
    triggered   : bool         True if at least one rule fired
    active_rules: list[str]    names of fired rules
    overrides   : dict         merged corrective LocoCommand overrides
    """

    triggered:    bool
    active_rules: list[str]
    overrides:    dict


class SafetyMem:
    """Apply safety rules to protect the robot.

    Args:
        rules: list of SafetyRule (read-only at runtime)
    """

    def __init__(self, rules: list[SafetyRule] | None = None):
        # Sorted by descending priority so high-priority rules win conflicts
        self._rules: list[SafetyRule] = sorted(
            rules or [], key=lambda r: r.priority, reverse=True
        )

    # ── Check ──────────────────────────────────────────────────────────────

    def check(self, frame) -> SafetyCheckResult:
        """Evaluate all rules against the latest STMFrame.

        Args:
            frame: STMFrame (or any object with named attributes)

        Returns:
            SafetyCheckResult with merged overrides.
        """
        active: list[str] = []
        overrides: dict = {}

        for rule in self._rules:
            val = getattr(frame, rule.field, None)
            if rule.is_triggered(val):
                active.append(rule.name)
                # Lower-priority rules do not overwrite higher-priority ones
                for k, v in rule.override.items():
                    if k not in overrides:
                        overrides[k] = v

        return SafetyCheckResult(
            triggered    = len(active) > 0,
            active_rules = active,
            overrides    = overrides,
        )

    # ── Persistence ────────────────────────────────────────────────────────

    @classmethod
    def from_list(cls, records: list[dict]) -> "SafetyMem":
        rules = [SafetyRule.from_dict(r) for r in records]
        return cls(rules)

    def to_list(self) -> list[dict]:
        return [r.to_dict() for r in self._rules]

    # ── Dunder ─────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._rules)

    def __repr__(self) -> str:
        return f"SafetyMem(rules={len(self._rules)})"
