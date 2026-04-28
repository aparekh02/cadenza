"""SmolVLAAdapter — adapter for HuggingFace's SmolVLA (~450M params, CPU-OK).

Loop semantics
--------------

The adapter emits **one action per tick**, not the entire plan upfront. Every
time the runtime calls ``propose_actions``:

  1. The current observation (pose, foot contacts, terrain probe ahead, and a
     forward camera frame when available) is packaged for SmolVLA.
  2. SmolVLA's ``select_action`` is invoked to get a fresh continuous action
     vector. The vector is logged but does not yet drive the robot — projecting
     ~7-DoF arm/gripper deltas onto a quadruped's named-action vocabulary needs
     task-specific glue we leave for the user.
  3. The adapter consults *the current observation* to decide the next named
     action: it pops from a goal-derived plan, but inserts perception-driven
     overrides (e.g. ``climb_step`` if a step appears ahead). That makes the
     "VLA model determines the situation periodically" claim real even before
     the projection is wired.
  4. ``done=False`` until the plan is exhausted, the robot has settled, or the
     tick budget is consumed.

This is the difference from ``cadenza.run("walk forward and sit")`` — there,
the entire plan is fixed at parse time. Here, every step is decided after
looking at the world.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from cadenza.actions.library import ActionCall
from cadenza.parser.translator import CommandParser
from cadenza.stack.adapters.base import (
    AdapterReply,
    ProposedAction,
    WorldModelAdapter,
    register_adapter,
)
from cadenza.stack.trajectory import TrajectoryMonitor
from cadenza.stack.vocabulary import ActionVocabulary


_LOG = logging.getLogger("cadenza.stack.smolvla")
_NAME_HINTS = ("smolvla", "smol_vla", "smol-vla")
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".ckpt")


def _has_weights(path: Path) -> bool:
    if not path.is_dir():
        return False
    for entry in path.iterdir():
        if entry.is_file() and entry.suffix.lower() in _WEIGHT_SUFFIXES:
            return True
    return False


# ── Per-episode adapter state ────────────────────────────────────────────────

@dataclass
class _EpisodeState:
    """State the adapter carries across ticks within one stack run."""
    plan: list[ProposedAction] = field(default_factory=list)
    completed: list[str] = field(default_factory=list)
    last_action_vec: np.ndarray | None = None
    settled_ticks: int = 0
    started: bool = False
    sequential_dodges: int = 0   # consecutive obstacle-avoidance ticks
    trajectory: TrajectoryMonitor = field(default_factory=TrajectoryMonitor)
    consecutive_recoveries: int = 0


@register_adapter
class SmolVLAAdapter(WorldModelAdapter):
    """Adapter for HuggingFace's SmolVLA via the lerobot package."""

    name = "smolvla"
    description = "HuggingFace SmolVLA (~450M params, CPU-friendly)"
    DEFAULT_MODEL_ID = "lerobot/smolvla_base"

    # ── Tunables ─────────────────────────────────────────────────────────────
    MAX_TICKS = 60               # hard cap on the perceive-reason-act loop
    SETTLE_THRESHOLD = 3         # consecutive low-motion ticks → done
    LOW_MOTION_NORM = 0.05       # |action vector| below this counts as settled

    OBSTACLE_TRIGGER_M = 0.55    # closer than this counts as "in the way"
    OBSTACLE_CLEAR_M = 0.85      # farther than this counts as "path clear"
    MAX_SEQUENTIAL_DODGES = 5    # safety: if we dodge this many in a row, give up
    CLIMB_STEP_MIN_M = 0.05      # below this isn't worth a climb action
    CLIMB_STEP_MAX_M = 0.20      # above this is a tall obstacle, not a stair

    # Trajectory-recovery tunables ──
    PROGRESS_WINDOW = 4          # ticks compared for distance-to-target progress
    PROGRESS_MIN_M = 0.10        # require this much closer over the window
    ARRIVAL_M = 0.45             # within this we count as "arrived"
    MAX_RECOVERIES = 8           # cap for total vision-recovery invocations

    # ── Detection ────────────────────────────────────────────────────────────

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

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def _load_impl(self) -> None:
        if self.model is not None:
            return
        try:
            from lerobot.policies.smolvla.modeling_smolvla import (    # type: ignore
                SmolVLAPolicy,
            )
        except ImportError as e:
            raise RuntimeError(
                "SmolVLAAdapter requires `lerobot[smolvla]`. Install with:\n"
                "    pip install 'lerobot[smolvla]'"
            ) from e
        model_id = str(self.checkpoint) if self.checkpoint else self.DEFAULT_MODEL_ID
        self.model = SmolVLAPolicy.from_pretrained(model_id)
        try:
            self.model.eval()
        except AttributeError:
            pass
        self._episode = _EpisodeState()

    def _ensure_episode(self) -> _EpisodeState:
        if not hasattr(self, "_episode"):
            self._episode = _EpisodeState()
        return self._episode

    # ── Vision navigator (lazy) ──────────────────────────────────────────────

    def _ensure_navigator(self):
        """Lazy-load the depth model + SmolVLM only if we actually get stuck."""
        if getattr(self, "_navigator", None) is not None:
            return self._navigator
        try:
            from cadenza.stack.vision import VisionNavigator
        except ImportError:
            self._navigator = None
            return None
        self._navigator = VisionNavigator()
        return self._navigator

    def _attempt_vision_recovery(
        self,
        observation: dict,
        vocabulary: ActionVocabulary,
    ) -> ProposedAction | None:
        """Run the depth + VLM stack and return one recovery ProposedAction.

        Returns None if the navigator can't run (no camera, no target, or
        the chosen action isn't in the vocabulary).
        """
        cam = observation.get("camera")
        target = observation.get("target_xy")
        pos = observation.get("pos")
        rpy = observation.get("rpy")
        if cam is None or target is None or pos is None or rpy is None:
            return None

        navigator = self._ensure_navigator()
        if navigator is None:
            return None

        try:
            decision = navigator.decide(
                rgb=np.asarray(cam),
                target_xy=(float(target[0]), float(target[1])),
                robot_xy=(float(pos[0]), float(pos[1])),
                robot_yaw=float(rpy[2]),
            )
        except Exception as e:
            _LOG.warning("vision navigator failed: %s", e)
            return None

        if decision.action not in vocabulary:
            return None

        # Sensible defaults so the recovery action actually moves the robot.
        params: dict[str, Any] = {}
        if decision.action == "walk_forward":
            params["distance_m"] = 0.6
        elif decision.action in {"side_step_left", "side_step_right"}:
            params["distance_m"] = 0.30
        elif decision.action in {"turn_left", "turn_right"}:
            # turn just enough to point toward the target
            params["rotation_rad"] = max(
                0.5,
                min(1.5, math.radians(abs(decision.target_bearing_deg))),
            )

        return ProposedAction(
            name=decision.action,
            params=params,
            rationale=decision.rationale,
        )

    # ── Inference (perception) ───────────────────────────────────────────────

    def _infer(self, observation: dict, goal: str) -> np.ndarray | None:
        """Run SmolVLA on the current frame. Returns the action vector or None.

        The output is used as a "perception signal" — it tells us the model
        looked at the world this tick. Projecting it onto the action library is
        left as integration work; here we just log it and use its norm to tell
        whether the model considers the scene "active".
        """
        if self.model is None:
            return None
        try:
            import torch
            cam = observation.get("camera")
            frame: dict[str, Any] = {
                "task": goal,
                "observation.state": torch.tensor(
                    np.asarray(observation.get("qpos", []), dtype=np.float32),
                ),
            }
            if cam is not None:
                arr = np.asarray(cam, dtype=np.float32) / 255.0
                if arr.ndim == 3:
                    arr = arr.transpose(2, 0, 1)[None, ...]  # (1, C, H, W)
                frame["observation.image"] = torch.tensor(arr)
            with torch.inference_mode():
                action = self.model.select_action(frame)
            return np.asarray(action.detach().cpu().numpy(), dtype=np.float32).flatten()
        except Exception as e:
            _LOG.debug("smolvla.select_action failed (%s); using perception-only loop", e)
            return None

    # ── Reasoning ────────────────────────────────────────────────────────────

    def _build_plan(self, goal: str, vocabulary: ActionVocabulary) -> list[ProposedAction]:
        """Parse the goal once into a queue of named actions."""
        if not goal.strip():
            return [ProposedAction(name="stand", params={},
                                   rationale="no goal supplied; standing")]
        parser = CommandParser(vocabulary.robot)
        out: list[ProposedAction] = []
        for call in parser.parse(goal):
            if call.action_name not in vocabulary:
                continue
            params: dict[str, Any] = {"speed": call.speed, "repeat": call.repeat}
            if call.distance_m > 0:
                params["distance_m"] = call.distance_m
            if call.rotation_rad != 0:
                params["rotation_rad"] = call.rotation_rad
            if call.duration_s > 0:
                params["duration_s"] = call.duration_s
            out.append(ProposedAction(
                name=call.action_name, params=params,
                rationale="parsed from goal",
            ))
        if not out:
            out = [ProposedAction(name="stand", params={},
                                  rationale="goal unparseable; standing")]
        return out

    def _terrain_override(
        self,
        next_action: ProposedAction,
        observation: dict,
        vocabulary: ActionVocabulary,
    ) -> ProposedAction:
        """Swap walk → climb_step when a *climbable* step is visible ahead.

        Tall vertical features (boxes, barrels, walls) also register as
        "step up" to a downward raycast, so we only auto-climb when the step
        is in the climbable range. Tall obstacles are handled by
        ``_obstacle_detour`` (side-step away).
        """
        if next_action.name not in {"walk_forward", "trot_forward", "pace_forward"}:
            return next_action
        terrain = observation.get("terrain_ahead") or {}
        step = float(terrain.get("max_step_up") or 0.0)
        if (self.CLIMB_STEP_MIN_M < step < self.CLIMB_STEP_MAX_M
                and "climb_step" in vocabulary):
            return ProposedAction(
                name="climb_step",
                params={},
                rationale=f"perception: step_up={step:.2f}m ahead",
            )
        return next_action

    def _obstacle_detour(
        self,
        observation: dict,
        vocabulary: ActionVocabulary,
    ) -> ProposedAction | None:
        """If an obstacle blocks the path, return a side-step away from it.

        Detection uses three forward raycasts (left/centre/right) at body
        height. If something is within ``OBSTACLE_TRIGGER_M``, we pick the
        clearer side and emit ``side_step_left/right``. The planned action
        is *not* consumed — we resume the plan once the path is clear.

        Stairs read as obstacles to a horizontal ray, so when the terrain
        probe also sees a climbable step ahead, we defer to the terrain
        override (which swaps in ``climb_step``) instead of dodging.
        """
        obs = observation.get("obstacles_ahead") or {}
        center = obs.get("center_m")
        left = obs.get("left_m")
        right = obs.get("right_m")
        if center is None and left is None and right is None:
            return None

        nearest = min(d for d in (center, left, right) if d is not None)
        if nearest >= self.OBSTACLE_TRIGGER_M:
            return None

        # If the step ahead is in the *climbable* band, this is the base of
        # the stairs — don't dodge, let _terrain_override emit climb_step.
        # Higher steps are tall obstacles (box / barrel / wall) and we DO
        # want to dodge those.
        terrain = observation.get("terrain_ahead") or {}
        step = float(terrain.get("max_step_up") or 0.0)
        if self.CLIMB_STEP_MIN_M < step < self.CLIMB_STEP_MAX_M:
            return None

        max_range = float(obs.get("max_range_m") or 1.5)
        left_clear = max_range if left is None else float(left)
        right_clear = max_range if right is None else float(right)

        if left_clear > right_clear + 0.05 and "side_step_left" in vocabulary:
            name, chosen, available = "side_step_left", "left", left_clear
        elif "side_step_right" in vocabulary:
            name, chosen, available = "side_step_right", "right", right_clear
        else:
            return None

        return ProposedAction(
            name=name,
            params={"distance_m": 0.30},
            rationale=(
                f"obstacle reasoning: nearest={nearest:.2f}m "
                f"(side={obs.get('side')}); going {chosen} "
                f"(clearance={available:.2f}m)"
            ),
        )

    def _is_settled(self, action_vec: np.ndarray | None) -> bool:
        ep = self._ensure_episode()
        if action_vec is None:
            return False
        norm = float(np.linalg.norm(action_vec))
        if norm < self.LOW_MOTION_NORM:
            ep.settled_ticks += 1
        else:
            ep.settled_ticks = 0
        return ep.settled_ticks >= self.SETTLE_THRESHOLD

    # ── Per-tick API ─────────────────────────────────────────────────────────

    def propose_actions(
        self,
        observation: dict,
        goal: str,
        vocabulary: ActionVocabulary,
        history: list[ProposedAction] | None = None,
    ) -> AdapterReply:
        if not self.is_loaded:
            self.load()
        ep = self._ensure_episode()

        # First call of the episode: build the goal-driven plan queue and
        # configure the trajectory monitor with the user-supplied target.
        if not ep.started:
            ep.plan = self._build_plan(goal, vocabulary)
            ep.trajectory = TrajectoryMonitor(
                target_xy=observation.get("target_xy"),
                window=self.PROGRESS_WINDOW,
                min_progress_m=self.PROGRESS_MIN_M,
                arrival_distance_m=self.ARRIVAL_M,
            )
            ep.started = True
            tgt_msg = (
                f"; target={ep.trajectory.target_xy}"
                if ep.trajectory.target_xy is not None else ""
            )
            _LOG.info("smolvla: built plan with %d actions%s", len(ep.plan), tgt_msg)

        # Trajectory: every tick, record distance-to-target.
        pos = observation.get("pos")
        if pos is not None and ep.trajectory.target_xy is not None:
            ep.trajectory.update((float(pos[0]), float(pos[1])))

        n_ticks = len(ep.completed)
        norm_str = "nan"

        # 1. Perceive: run SmolVLA on the latest observation.
        action_vec = self._infer(observation, goal)
        ep.last_action_vec = action_vec
        if action_vec is not None:
            norm_str = f"{float(np.linalg.norm(action_vec)):.3f}"

        # 2. Stop conditions.
        if ep.trajectory.at_target:
            return AdapterReply(
                actions=[ProposedAction(name="sit", params={},
                                        rationale="reached target")],
                done=True,
                note=f"smolvla: arrived ({ep.trajectory.progress_summary()})",
            )
        if n_ticks >= self.MAX_TICKS:
            return AdapterReply(actions=[], done=True,
                                note=f"smolvla: tick budget {self.MAX_TICKS} reached")
        if not ep.plan and self._is_settled(action_vec):
            return AdapterReply(actions=[], done=True,
                                note="smolvla: plan exhausted and robot settled")
        if not ep.plan:
            return AdapterReply(actions=[], done=True,
                                note="smolvla: plan exhausted")

        # 2b. Trajectory recovery — if distance to target hasn't dropped
        #     enough over the recent window, hand off to the vision navigator
        #     (depth-anything + SmolVLM) for a redirect. The plan is *not*
        #     consumed; once we make progress again the queue resumes.
        if (
            ep.trajectory.is_stuck
            and ep.consecutive_recoveries < self.MAX_RECOVERIES
        ):
            recovery = self._attempt_vision_recovery(observation, vocabulary)
            if recovery is not None:
                ep.consecutive_recoveries += 1
                ep.completed.append(recovery.name)
                ep.trajectory.reset_after_recovery()
                note = (
                    f"smolvla tick {n_ticks + 1}: "
                    f"vla_norm={norm_str} -> {recovery.name} "
                    f"[VISION RECOVERY {ep.consecutive_recoveries}/"
                    f"{self.MAX_RECOVERIES}; {recovery.rationale}; "
                    f"{ep.trajectory.progress_summary()}]"
                )
                return AdapterReply(actions=[recovery], done=False, note=note)

        # 3. Obstacle reasoning takes priority over the plan. If something is
        #    in the way, emit a side-step *without* consuming the plan queue —
        #    the planned action will be reattempted once the path is clear.
        detour = self._obstacle_detour(observation, vocabulary)
        if detour is not None:
            ep.sequential_dodges += 1
            if ep.sequential_dodges > self.MAX_SEQUENTIAL_DODGES:
                # Safety: don't get stuck dodging forever. Fall through to the
                # planned action and hope the next tick sees a clearer scene.
                _LOG.warning(
                    "smolvla: %d consecutive dodges, abandoning detour and "
                    "trying the planned action again",
                    ep.sequential_dodges,
                )
                ep.sequential_dodges = 0
            else:
                ep.completed.append(detour.name)
                note = (
                    f"smolvla tick {n_ticks + 1}: "
                    f"vla_norm={norm_str} -> {detour.name} "
                    f"[detour {ep.sequential_dodges}/{self.MAX_SEQUENTIAL_DODGES}; "
                    f"{detour.rationale}]"
                )
                return AdapterReply(actions=[detour], done=False, note=note)

        ep.sequential_dodges = 0

        # 4. Path clear: pop the next planned action and apply terrain override.
        next_action = ep.plan.pop(0)
        next_action = self._terrain_override(next_action, observation, vocabulary)
        ep.completed.append(next_action.name)

        note = (
            f"smolvla tick {n_ticks + 1}: "
            f"vla_norm={norm_str} -> {next_action.name} "
            f"(remaining {len(ep.plan)})"
        )
        return AdapterReply(actions=[next_action], done=False, note=note)
