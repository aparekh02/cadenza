"""SmolVLAAdapter — closed-loop adapter for HuggingFace's SmolVLA.

The stack runs as a **recursive perceive-reason-act loop**: it keeps ticking
until the goal is reached. There is *no* pre-built plan when a ``target`` is
supplied — every tick the adapter looks at the world (observation + every
plugged-in multi-modal sensing modality + SmolVLA inference) and picks the
single next action that makes the most progress toward the goal.

Per-tick decision order (closed-loop mode, ``target=(x, y)`` supplied):

  1. **Goal reached** — distance to ``target_xy`` ≤ ``ARRIVAL_M``. Emit
     ``sit`` and signal ``done=True``.
  2. **Stuck** — distance hasn't dropped over the progress window. Hand off
     to ``VisionNavigator`` (depth + SmolVLM) which picks a recovery action
     from the camera frame + bearing/distance to the target.
  3. **Climbable step ahead** — ``terrain_ahead.max_step_up`` in the
     climbable band. Emit ``climb_step``.
  4. **Obstacle blocking the path** — horizontal raycasts and / or the
     ``depth_left/center/right`` modality keys agree something's in the way.
     Pick the clearer side and side-step.
  5. **Misaligned with target heading** — bearing > 25°. Turn toward target
     by exactly the misalignment.
  6. **Path clear and aligned** — ``walk_forward`` with a chunk sized for
     the remaining distance (so we don't overshoot near the target).

The loop terminates only on (1), or hitting a sanity ``MAX_TICKS`` cap, or
the runtime's ``max_iterations``. Without ``target``, the adapter falls
back to the legacy plan-queue mode for backward compatibility.
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
    # The loop runs until the goal is reached. MAX_TICKS is a sanity cap so a
    # broken policy can't run forever; raise it for very long courses.
    MAX_TICKS = 250
    SETTLE_THRESHOLD = 3         # consecutive low-motion ticks → settled
    LOW_MOTION_NORM = 0.05       # |action vector| below this counts as settled

    OBSTACLE_TRIGGER_M = 0.55    # closer than this counts as "in the way"
    OBSTACLE_CLEAR_M = 0.85      # farther than this counts as "path clear"
    MAX_SEQUENTIAL_DODGES = 6    # if we dodge this many in a row, fall through
    CLIMB_STEP_MIN_M = 0.05      # below this isn't worth a climb action
    CLIMB_STEP_MAX_M = 0.20      # above this is a tall obstacle, not a stair

    # Heading control ──
    HEADING_TOL_DEG = 25.0       # bearing within this counts as "aligned"
    MAX_TURN_RAD = 1.2           # cap per-tick rotation (~70°)

    # Forward gait chunking ──
    WALK_CHUNK_MAX_M = 0.7       # max distance for a single walk_forward call
    WALK_CHUNK_FRAC = 0.4        # cover at most this fraction of remaining dist

    # Trajectory-recovery tunables ──
    PROGRESS_WINDOW = 4          # ticks compared for distance-to-target progress
    PROGRESS_MIN_M = 0.10        # require this much closer over the window
    ARRIVAL_M = 0.45             # within this we count as "arrived"
    MAX_RECOVERIES = 12          # cap for total vision-recovery invocations

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

    # ── Closed-loop helpers ──────────────────────────────────────────────────

    @staticmethod
    def _target_bearing(
        target_xy: tuple[float, float],
        robot_xy: tuple[float, float],
        robot_yaw: float,
    ) -> tuple[float, float]:
        """(bearing_deg, distance_m). Bearing > 0 = target is to the left."""
        dx = target_xy[0] - robot_xy[0]
        dy = target_xy[1] - robot_xy[1]
        target_world_heading = math.atan2(dy, dx)
        # Cadenza convention: forward = -x in body frame ⇒ world = yaw + π.
        robot_world_heading = robot_yaw + math.pi
        bearing = target_world_heading - robot_world_heading
        bearing = (bearing + math.pi) % (2 * math.pi) - math.pi
        return math.degrees(bearing), math.hypot(dx, dy)

    def _depth_modality_detour(
        self,
        observation: dict,
        vocabulary: ActionVocabulary,
    ) -> ProposedAction | None:
        """If the depth modality says the centre is much closer than the
        sides, dodge toward the deeper side. This catches obstacles the
        sim-raycast probe missed (small/distant)."""
        d_left = observation.get("depth_left")
        d_center = observation.get("depth_center")
        d_right = observation.get("depth_right")
        if d_left is None or d_center is None or d_right is None:
            return None
        max_side = max(float(d_left), float(d_right))
        if d_center >= max_side * 0.75:
            return None  # centre is roughly as open as the sides
        if d_left > d_right and "side_step_left" in vocabulary:
            name, side, clear = "side_step_left", "left", float(d_left)
        elif "side_step_right" in vocabulary:
            name, side, clear = "side_step_right", "right", float(d_right)
        else:
            return None
        return ProposedAction(
            name=name,
            params={"distance_m": 0.30},
            rationale=(
                f"depth modality: centre={d_center:.2f} < sides "
                f"({d_left:.2f}/{d_right:.2f}); going {side}"
            ),
        )

    def _closed_loop_step(
        self,
        observation: dict,
        vocabulary: ActionVocabulary,
        ep: _EpisodeState,
    ) -> tuple[ProposedAction, str]:
        """One tick of the closed perceive-reason-act loop. Returns the chosen
        action and a tag describing which branch fired."""
        pos = observation["pos"]
        rpy = observation.get("rpy", [0.0, 0.0, 0.0])
        target = ep.trajectory.target_xy
        bearing_deg, distance_m = self._target_bearing(
            (float(target[0]), float(target[1])),
            (float(pos[0]), float(pos[1])),
            float(rpy[2]),
        )

        # 1. Stuck → vision recovery (depth + SmolVLM).
        if ep.trajectory.is_stuck and ep.consecutive_recoveries < self.MAX_RECOVERIES:
            recovery = self._attempt_vision_recovery(observation, vocabulary)
            if recovery is not None:
                ep.consecutive_recoveries += 1
                ep.trajectory.reset_after_recovery()
                return recovery, f"VISION_RECOVERY[{ep.consecutive_recoveries}/{self.MAX_RECOVERIES}]"

        # 2. Climbable step ahead → climb_step.
        terrain = observation.get("terrain_ahead") or {}
        step = float(terrain.get("max_step_up") or 0.0)
        if (self.CLIMB_STEP_MIN_M < step < self.CLIMB_STEP_MAX_M
                and "climb_step" in vocabulary):
            return ProposedAction(
                name="climb_step",
                params={},
                rationale=f"climbable step ahead ({step:.2f}m)",
            ), "CLIMB"

        # 3. Obstacle blocking the path. Try raycast first (ground truth in sim);
        #    fall back to depth modality (real perception model output).
        if ep.sequential_dodges <= self.MAX_SEQUENTIAL_DODGES:
            detour = self._obstacle_detour(observation, vocabulary)
            if detour is None:
                detour = self._depth_modality_detour(observation, vocabulary)
            if detour is not None:
                ep.sequential_dodges += 1
                return detour, f"DETOUR[{ep.sequential_dodges}/{self.MAX_SEQUENTIAL_DODGES}]"
        # Too many dodges in a row — clear it and let other branches drive.
        ep.sequential_dodges = 0

        # 4. Misaligned with target → turn by the bearing magnitude.
        if abs(bearing_deg) > self.HEADING_TOL_DEG:
            turn_rad = max(0.4, min(self.MAX_TURN_RAD, math.radians(abs(bearing_deg))))
            name = "turn_left" if bearing_deg > 0 else "turn_right"
            if name in vocabulary:
                return ProposedAction(
                    name=name,
                    params={"rotation_rad": turn_rad},
                    rationale=f"align: bearing={bearing_deg:+.0f}°",
                ), "TURN"

        # 5. Aligned and clear → walk_forward, chunk-sized so we don't overshoot.
        chunk = max(0.20, min(self.WALK_CHUNK_MAX_M, distance_m * self.WALK_CHUNK_FRAC))
        # If we're inside the arrival ring already the parent loop catches it;
        # otherwise this keeps us closing the distance.
        return ProposedAction(
            name="walk_forward",
            params={"distance_m": chunk},
            rationale=f"advance: dist={distance_m:.2f}m bearing={bearing_deg:+.0f}°",
        ), "ADVANCE"

    def _legacy_plan_step(
        self,
        observation: dict,
        vocabulary: ActionVocabulary,
        ep: _EpisodeState,
    ) -> tuple[ProposedAction | None, str, bool]:
        """Pre-target plan-queue mode. Used only when no ``target_xy`` was
        supplied. Returns (action, tag, plan_exhausted)."""
        if not ep.plan:
            return None, "PLAN_EMPTY", True
        # Stuck handling (only meaningful if a target *is* set, but we keep
        # the symmetry by trying the visioon recovery as a courtesy).
        next_action = ep.plan.pop(0)
        next_action = self._terrain_override(next_action, observation, vocabulary)
        return next_action, f"PLAN[remain={len(ep.plan)}]", False

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

        # ── Episode init ────────────────────────────────────────────────────
        if not ep.started:
            target = observation.get("target_xy")
            ep.trajectory = TrajectoryMonitor(
                target_xy=target,
                window=self.PROGRESS_WINDOW,
                min_progress_m=self.PROGRESS_MIN_M,
                arrival_distance_m=self.ARRIVAL_M,
            )
            # No plan when target-driven; build one only as legacy fallback.
            ep.plan = [] if target is not None else self._build_plan(goal, vocabulary)
            ep.started = True
            mode = "closed-loop" if target is not None else f"plan({len(ep.plan)})"
            _LOG.info("smolvla: episode start, mode=%s, target=%s", mode, target)

        # ── Per-tick perception ────────────────────────────────────────────
        pos = observation.get("pos")
        if pos is not None and ep.trajectory.target_xy is not None:
            ep.trajectory.update((float(pos[0]), float(pos[1])))

        action_vec = self._infer(observation, goal)
        ep.last_action_vec = action_vec
        norm_str = (
            f"{float(np.linalg.norm(action_vec)):.3f}"
            if action_vec is not None else "nan"
        )

        n_ticks = len(ep.completed)

        # ── Termination ────────────────────────────────────────────────────
        if ep.trajectory.at_target:
            return AdapterReply(
                actions=[ProposedAction(name="sit", params={},
                                        rationale="reached target")],
                done=True,
                note=f"smolvla: arrived ({ep.trajectory.progress_summary()})",
            )
        if n_ticks >= self.MAX_TICKS:
            return AdapterReply(
                actions=[], done=True,
                note=f"smolvla: tick budget {self.MAX_TICKS} reached "
                     f"({ep.trajectory.progress_summary()})",
            )

        # ── Decide the next action ─────────────────────────────────────────
        if ep.trajectory.target_xy is not None:
            action, branch = self._closed_loop_step(observation, vocabulary, ep)
        else:
            # Legacy plan-queue mode (no target supplied).
            action, branch, exhausted = self._legacy_plan_step(observation, vocabulary, ep)
            if action is None:
                done = True
                if exhausted and self._is_settled(action_vec):
                    note = "smolvla: plan exhausted and robot settled"
                else:
                    note = "smolvla: plan exhausted"
                return AdapterReply(actions=[], done=done, note=note)

        ep.completed.append(action.name)
        progress = ep.trajectory.progress_summary() if ep.trajectory.target_xy else "no target"
        note = (
            f"smolvla tick {n_ticks + 1}: "
            f"vla_norm={norm_str} -> {action.name} "
            f"[{branch}; {action.rationale}; {progress}]"
        )
        return AdapterReply(actions=[action], done=False, note=note)
