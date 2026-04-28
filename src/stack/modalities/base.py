"""Modality base class + registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, TYPE_CHECKING

if TYPE_CHECKING:
    from cadenza.stack.gym_adapter import Observation


@dataclass
class ModalityResult:
    """Container for the keys a modality contributes to the observation dict.

    A modality may also expose a short one-line summary the runtime prints
    each tick, so the user can see at a glance that the modality is alive
    and what it's seeing.
    """
    keys: dict[str, Any] = field(default_factory=dict)
    summary: str = ""


class Modality(ABC):
    """Pluggable input-modality base class.

    Lifecycle::

        m = MyModality()
        m.setup()                # optional, called once before the loop
        result = m.compute(obs)  # called every tick
        m.teardown()             # optional, called once after the loop
    """

    name: ClassVar[str] = "base"
    description: ClassVar[str] = ""

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def setup(self) -> None:
        """Optional one-time init before the first tick."""
        return

    def teardown(self) -> None:
        """Optional cleanup at the end of the run."""
        return

    @property
    def is_loaded(self) -> bool:
        """Override if the modality holds heavy state (e.g. a model)."""
        return True

    # ── Per-tick compute ─────────────────────────────────────────────────────

    @abstractmethod
    def compute(self, observation: "Observation") -> ModalityResult:
        """Inspect the observation and return keys to merge into the obs dict."""
        raise NotImplementedError


# ─── Registry ─────────────────────────────────────────────────────────────────

_MODALITIES: dict[str, type[Modality]] = {}


def register_modality(cls: type[Modality]) -> type[Modality]:
    """Register a Modality subclass so it can be looked up by name."""
    if not issubclass(cls, Modality):
        raise TypeError(f"{cls!r} must subclass Modality")
    _MODALITIES[cls.name] = cls
    return cls


def get_modality(name: str) -> type[Modality]:
    if name not in _MODALITIES:
        raise KeyError(
            f"Unknown modality '{name}'. Registered: {sorted(_MODALITIES)}"
        )
    return _MODALITIES[name]


def list_modalities() -> list[type[Modality]]:
    return list(_MODALITIES.values())
