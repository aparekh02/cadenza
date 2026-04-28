"""WorldModelAdapter — base ABC every adapter implements.

An adapter does two things:

  1. Tell the stack whether a model of its kind is present (`detect`).
  2. Translate (observation, goal, vocabulary) into a list of ProposedAction.

The adapter does NOT execute motors — it just speaks the action vocabulary
back to the stack, which validates and runs it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar


@dataclass
class ProposedAction:
    """One action proposed by the world model.

    `name` must match an action in the Cadenza action library for the active
    robot. `params` is a free-form dict of action parameters — the builder
    validates and clamps them.
    """
    name: str
    params: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""


@dataclass
class AdapterReply:
    """Reply from a world model after one reasoning step."""
    actions: list[ProposedAction] = field(default_factory=list)
    done: bool = False
    note: str = ""


class WorldModelAdapter(ABC):
    """Base class for world-model adapters."""

    # Subclasses set this to a stable short id (e.g. "pi_zero", "openvla").
    name: ClassVar[str] = "base"

    # Human-readable description used by detector logs.
    description: ClassVar[str] = ""

    def __init__(self, checkpoint: str | Path | None = None,
                 model: Any = None, **kwargs):
        """Adapter holds a handle to the underlying model.

        Args:
            checkpoint: Path or HuggingFace id of the model checkpoint.
            model: Optional already-loaded model object (overrides `checkpoint`).
            **kwargs: Adapter-specific options.
        """
        self.checkpoint = Path(checkpoint) if isinstance(checkpoint, str) and Path(checkpoint).exists() else checkpoint
        self.model = model
        self.options = kwargs
        self._loaded = model is not None

    # ── Detection ─────────────────────────────────────────────────────────────

    @classmethod
    @abstractmethod
    def detect(cls, root: Path) -> Path | None:
        """Look for a checkpoint of this model at `root`.

        Returns the checkpoint path if found, otherwise None. Detection should
        be cheap (filesystem checks only — no model loading).
        """
        raise NotImplementedError

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load the underlying model into memory. Idempotent."""
        if self._loaded:
            return
        self._load_impl()
        self._loaded = True

    def _load_impl(self) -> None:
        """Subclass hook for actual model loading. Default: no-op."""
        return

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ── Inference ─────────────────────────────────────────────────────────────

    @abstractmethod
    def propose_actions(
        self,
        observation: dict,
        goal: str,
        vocabulary: "ActionVocabulary",  # noqa: F821
        history: list[ProposedAction] | None = None,
    ) -> AdapterReply:
        """Ask the world model what to do next.

        Args:
            observation: dict from the gym adapter (pose, velocity, camera, ...).
            goal: natural-language task description.
            vocabulary: structured tool schema describing available actions.
            history: actions already executed this episode (for context).

        Returns:
            AdapterReply with a list of ProposedAction and a `done` flag.
        """
        raise NotImplementedError


# ─── Adapter registry ─────────────────────────────────────────────────────────

_ADAPTERS: dict[str, type[WorldModelAdapter]] = {}


def register_adapter(cls: type[WorldModelAdapter]) -> type[WorldModelAdapter]:
    """Register an adapter class so the detector can find it. Usable as a
    decorator."""
    if not issubclass(cls, WorldModelAdapter):
        raise TypeError(f"{cls!r} must subclass WorldModelAdapter")
    _ADAPTERS[cls.name] = cls
    return cls


def get_adapter(name: str) -> type[WorldModelAdapter]:
    if name not in _ADAPTERS:
        raise KeyError(
            f"Unknown adapter '{name}'. Registered: {sorted(_ADAPTERS)}"
        )
    return _ADAPTERS[name]


def list_adapters() -> list[type[WorldModelAdapter]]:
    return list(_ADAPTERS.values())
