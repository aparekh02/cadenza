"""PiZeroAdapter — adapter for Physical Intelligence's pi_0.5 (a.k.a. pi05).

The pi_0 family is a flow-matching VLA. We don't ship a runtime here — the
adapter only knows how to detect a checkpoint and how to translate the model's
output into ProposedAction objects. If the user does not have the model
weights or the ``openpi`` runtime installed, ``load()`` will raise.

Detection heuristics:
  * directory name contains "pi_0", "pi05", "pi_zero", or "pi-0.5"
  * folder contains a config.json that mentions "pi_0", "pi05", or "openpi"
  * HF cache entry under ``models--physical-intelligence--*``
"""

from __future__ import annotations

import json
from pathlib import Path

from cadenza.stack.adapters.base import (
    AdapterReply,
    ProposedAction,
    WorldModelAdapter,
    register_adapter,
)
from cadenza.stack.vocabulary import ActionVocabulary


_NAME_HINTS = ("pi_0", "pi05", "pi_zero", "pi-0.5", "pi0.5", "physical-intelligence")
_CONFIG_HINTS = ("pi_0", "pi05", "openpi", "pi_zero")
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".msgpack")


def _has_weights(path: Path) -> bool:
    """True if `path` contains any file that looks like model weights."""
    if not path.is_dir():
        return False
    for entry in path.iterdir():
        if entry.is_file() and entry.suffix.lower() in _WEIGHT_SUFFIXES:
            return True
    return False


@register_adapter
class PiZeroAdapter(WorldModelAdapter):
    """Adapter for Physical Intelligence's pi_0 / pi_0.5 family."""

    name = "pi_zero"
    description = "Physical Intelligence pi_0 / pi_0.5 VLA"

    @classmethod
    def detect(cls, root: Path) -> Path | None:
        """A directory qualifies only if it BOTH name-matches pi_0 AND contains
        actual model artifacts (config.json with pi_0 hints, or weight files).
        Empty stub folders named "pi_zero" do not count."""
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
                cfg_hit = any(h in text for h in _CONFIG_HINTS)
            weights_hit = _has_weights(p)
            # Strong: config.json explicitly identifies pi_0.
            if cfg_hit:
                return p
            # Otherwise require name match AND model weights.
            if name_hit and weights_hit:
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
        """Load the policy via openpi if a checkpoint path is set.

        Mirrors openpi's documented loading flow:
            from openpi.training import config as _config
            from openpi.policies import policy_config
            policy = policy_config.create_trained_policy(config, ckpt)
        """
        if self.model is not None:
            return  # caller passed an already-loaded policy
        if not self.checkpoint:
            raise RuntimeError(
                "PiZeroAdapter has no model to load. Pass `model=` with a "
                "loaded openpi policy, or set `checkpoint=` to a config name."
            )
        try:
            from openpi.training import config as _config           # type: ignore
            from openpi.policies import policy_config               # type: ignore
            from openpi.shared import download                      # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "PiZeroAdapter requires `openpi`. Install it from "
                "https://github.com/Physical-Intelligence/openpi"
            ) from e
        config_name = self.options.get("config_name", "pi05_droid")
        cfg = _config.get_config(config_name)
        ckpt = download.maybe_download(str(self.checkpoint))
        self.model = policy_config.create_trained_policy(cfg, ckpt)

    def propose_actions(
        self,
        observation: dict,
        goal: str,
        vocabulary: ActionVocabulary,
        history: list[ProposedAction] | None = None,
    ) -> AdapterReply:
        if not self.is_loaded:
            self.load()
        if self.model is None:
            return AdapterReply(
                actions=[ProposedAction(name="stand", params={})],
                done=True, note="pi_zero: no model loaded",
            )

        # openpi policies expose `.infer(example)["actions"]`. Build the
        # minimal example dict expected by the policy: the goal becomes the
        # prompt, and any image keys the caller stashed in `observation` are
        # forwarded as-is.
        example = {k: v for k, v in observation.items() if k.startswith("observation/")}
        example["prompt"] = goal

        try:
            out = self.model.infer(example)
        except Exception as e:
            return AdapterReply(
                actions=[ProposedAction(name="stand", params={})],
                done=True, note=f"pi_zero infer error: {e}",
            )

        return self._project_to_vocabulary(out, goal, vocabulary, history)

    # ── Projection helper ────────────────────────────────────────────────────

    def _project_to_vocabulary(
        self,
        raw: dict,
        goal: str,
        vocabulary: ActionVocabulary,
        history: list[ProposedAction] | None,
    ) -> AdapterReply:
        """Project raw model output onto Cadenza's action vocabulary.

        Expected shapes from a wrapped pi_0 runtime:
          * ``raw["actions"]`` — list of {name, params, rationale}
          * ``raw["done"]`` — bool
          * (alternative) ``raw["plan"]`` — text plan parsed via the goal channel.
        """
        actions: list[ProposedAction] = []
        for entry in raw.get("actions", []):
            name = entry.get("name")
            if not name or name not in vocabulary:
                continue
            actions.append(ProposedAction(
                name=name,
                params=entry.get("params", {}),
                rationale=entry.get("rationale", ""),
            ))
        return AdapterReply(
            actions=actions,
            done=bool(raw.get("done", False)),
            note=raw.get("note", "pi_zero"),
        )
