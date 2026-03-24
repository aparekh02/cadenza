"""UserMem — User preference memory.

Stores named scalar or flag preferences set by the operator (e.g. max
speed cap, preferred gait, noise mode). At runtime these preferences are
applied on top of the controller's baseline command before it is emitted.

Loaded read-only from snapshot.json at runtime (or updated via ROS2 topic).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class UserPreference:
    """One operator preference entry.

    Fields
    ------
    key      : str  preference name (e.g. "max_speed", "gait_override")
    value    : Any  current value
    default  : Any  fallback when value is absent
    doc      : str  human description
    """

    key:     str
    value:   Any
    default: Any  = None
    doc:     str  = ""

    @classmethod
    def from_dict(cls, d: dict) -> "UserPreference":
        return cls(
            key     = str(d["key"]),
            value   = d["value"],
            default = d.get("default"),
            doc     = str(d.get("doc", "")),
        )

    def to_dict(self) -> dict:
        return {
            "key":     self.key,
            "value":   self.value,
            "default": self.default,
            "doc":     self.doc,
        }


class UserMem:
    """Key-value preference store.

    Args:
        prefs: initial list of UserPreference objects
    """

    def __init__(self, prefs: list[UserPreference] | None = None):
        self._store: dict[str, UserPreference] = {}
        for p in (prefs or []):
            self._store[p.key] = p

    # ── Read / Write ───────────────────────────────────────────────────────

    def get(self, key: str, fallback: Any = None) -> Any:
        """Return the current value for key, or fallback if not set."""
        if key in self._store:
            p = self._store[key]
            return p.value if p.value is not None else p.default
        return fallback

    def set(self, key: str, value: Any) -> None:
        """Update or insert a preference value."""
        if key in self._store:
            self._store[key].value = value
        else:
            self._store[key] = UserPreference(key=key, value=value)

    def apply(self, cmd: dict) -> dict:
        """Apply preferences to a LocoCommand dict.

        Rules (applied in order):
        - "max_speed"    → cap cmd["cmd_vx"] and cmd["cmd_vy"]
        - "gait_override"→ overwrite cmd["gait"] if not None/""
        - any other key that matches a cmd key → overwrite directly
        """
        cmd = dict(cmd)   # shallow copy so we don't mutate caller's dict

        max_speed = self.get("max_speed")
        if max_speed is not None:
            import numpy as np
            for k in ("cmd_vx", "cmd_vy"):
                if k in cmd:
                    cmd[k] = float(np.clip(cmd[k], -max_speed, max_speed))

        gait_override = self.get("gait_override")
        if gait_override:
            cmd["gait"] = str(gait_override)

        # Generic passthrough: any pref key that matches a cmd field
        for key, pref in self._store.items():
            if key in ("max_speed", "gait_override"):
                continue
            if key in cmd and pref.value is not None:
                cmd[key] = pref.value

        return cmd

    # ── Persistence ────────────────────────────────────────────────────────

    @classmethod
    def from_list(cls, records: list[dict]) -> "UserMem":
        prefs = [UserPreference.from_dict(r) for r in records]
        return cls(prefs)

    def to_list(self) -> list[dict]:
        return [p.to_dict() for p in self._store.values()]

    # ── Dunder ─────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._store)

    def __repr__(self) -> str:
        return f"UserMem(keys={list(self._store)})"
