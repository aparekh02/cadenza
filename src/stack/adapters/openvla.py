"""OpenVLAAdapter — adapter for the OpenVLA family.

Detection: looks for an ``openvla`` checkpoint folder or HF cache hit. The
runtime hookup is a structural placeholder, like PiZeroAdapter — the real
inference call is left to the user's installed runtime.
"""

from __future__ import annotations

from pathlib import Path

from cadenza.stack.adapters.base import (
    AdapterReply,
    ProposedAction,
    WorldModelAdapter,
    register_adapter,
)
from cadenza.stack.vocabulary import ActionVocabulary


_NAME_HINTS = ("openvla", "open_vla", "open-vla", "prismatic")
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".ckpt")


def _has_weights(path: Path) -> bool:
    if not path.is_dir():
        return False
    for entry in path.iterdir():
        if entry.is_file() and entry.suffix.lower() in _WEIGHT_SUFFIXES:
            return True
    return False


@register_adapter
class OpenVLAAdapter(WorldModelAdapter):
    """Adapter for the OpenVLA / Prismatic VLA family."""

    name = "openvla"
    description = "OpenVLA-7B and related Prismatic VLAs"

    @classmethod
    def detect(cls, root: Path) -> Path | None:
        if not root.exists() or not root.is_dir():
            return None

        def _qualifies(p: Path) -> Path | None:
            if not p.is_dir():
                return None
            lname = p.name.lower()
            name_hit = any(h in lname for h in _NAME_HINTS)
            cfg = p / "config.json"
            cfg_hit = False
            if cfg.is_file():
                try:
                    text = cfg.read_text(errors="ignore").lower()
                except Exception:
                    text = ""
                cfg_hit = any(h in text for h in _NAME_HINTS)
            if cfg_hit:
                return p
            if name_hit and _has_weights(p):
                return p
            return None

        hit = _qualifies(root)
        if hit:
            return hit
        for sub in root.iterdir():
            hit = _qualifies(sub)
            if hit:
                return hit
        return None

    def _load_impl(self) -> None:
        try:
            from transformers import AutoModelForVision2Seq, AutoProcessor  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "OpenVLAAdapter requires `transformers`. "
                "pip install transformers torch pillow"
            ) from e
        if not self.checkpoint:
            raise RuntimeError("OpenVLAAdapter has no checkpoint to load.")
        self.model = AutoModelForVision2Seq.from_pretrained(
            str(self.checkpoint), trust_remote_code=True,
        )
        self.options["processor"] = AutoProcessor.from_pretrained(
            str(self.checkpoint), trust_remote_code=True,
        )

    def propose_actions(
        self,
        observation: dict,
        goal: str,
        vocabulary: ActionVocabulary,
        history: list[ProposedAction] | None = None,
    ) -> AdapterReply:
        if not self.is_loaded:
            self.load()

        # OpenVLA returns 7-DoF end-effector deltas natively; projecting to
        # quadruped/humanoid Cadenza actions requires task-specific glue.
        # This placeholder keeps the stack runnable; users override
        # `_project_to_vocabulary` for real deployments.
        return AdapterReply(
            actions=[ProposedAction(name="stand", params={},
                                    rationale="openvla runtime stub")],
            done=True,
            note="openvla adapter loaded but projection not implemented",
        )
