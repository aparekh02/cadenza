"""episode_guide.py — Continuous memory steering for goal-directed locomotion.

This module is the bridge between the five memory compartments and the robot's
moment-to-moment behaviour. It runs at the same 50 Hz as LocoController.

How it replaces RL
------------------
Instead of training weights/biases offline, this system:
  1. Scores the robot's stability every step (IMU + foot contacts + height).
  2. Smoothly adapts speed and gait through UserMem — LocoController reads
     these values automatically on every call to ctrl.step().
  3. Accumulates successful joint trajectories in SkillMem — LocoController
     blends them in (skill_alpha) every step, so past good motions guide
     current joint targets.
  4. Records fall-pattern rules in SafetyMem — LocoController enforces them
     before issuing any command.
  5. Stores terrain clusters in MapMem — TerrainClassifier recalls them to
     select recommended gaits.

Net effect: the robot gets slower and safer when unstable, faster when
confident, and inherits joint motion from its own successful past — all
without changing any model weights.

Continuous steering loop (called every control step from visual_gym.py)
------------------------------------------------------------------------
    guide.step(t, x, trunk_z, rpy, fc, q12) → SteeringOutput
        • writes max_speed / gait_override / body_height to UserMem
        • LocoController reads them immediately on the same ctrl.step() call
        • returns { cmd_vx, gait, stability, phase } for the gym to use

Episode lifecycle
-----------------
    begin_episode(n)   → params dict (recall from memory + Groq suggestion)
    step(...)          → SteeringOutput (continuous, every control step)
    end_episode(...)   → EpisodeRecord  (write back to all memory compartments)

Curriculum
----------
    Stage 0 — Walk:     goal x = 3 m   (no hurdle)
    Stage 1 — Approach: goal x = 5 m   (step over hurdle with trot)
    Stage 2 — Full:     goal x = 8 m
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from cadenza_local.locomotion.groq_agent import EpisodeSteeringAgent

# ── Motor-primitive embedding constants (must match skillmem._STEP_HEIGHT_*) ──
_STEP_HEIGHT_BASELINE: float = 0.10   # m
_STEP_HEIGHT_SCALE:    float = 8.0


def _skill_embedding(cmd_vel: np.ndarray, step_height: float) -> np.ndarray:
    """4-dim goal embedding for motor-primitive skills.

    Identical math to skillmem.goal_to_embedding_4d — duplicated here to
    avoid importing skillmem at module level from episode_guide.
    """
    v = np.asarray(cmd_vel, dtype=np.float32).copy()
    n = float(np.linalg.norm(v))
    if n > 1e-8:
        v = v / n
    step_dev = (float(step_height) - _STEP_HEIGHT_BASELINE) * _STEP_HEIGHT_SCALE
    v4 = np.array([v[0], v[1], v[2], step_dev], dtype=np.float32)
    n4 = float(np.linalg.norm(v4))
    if n4 > 1e-8:
        v4 = v4 / n4
    return v4

# ─────────────────────────────────────────────────────────────────────────────
#  Phases
# ─────────────────────────────────────────────────────────────────────────────

PHASE_APPROACH = "approach"
PHASE_SPRINT   = "sprint"
PHASE_BOUND    = "bound"      # high-step trot near hurdle (NOT bound gait)
PHASE_RECOVERY = "recovery"
PHASE_GOAL     = "goal"

# ─────────────────────────────────────────────────────────────────────────────
#  Curriculum stages
# ─────────────────────────────────────────────────────────────────────────────

STAGE_WALK     = 0
STAGE_APPROACH = 1
STAGE_FULL     = 2

# ─────────────────────────────────────────────────────────────────────────────
#  Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SteeringOutput:
    """Return value of EpisodeGuide.step() — used by the gym each control step."""
    cmd_vx:    float   # stability-adapted forward speed (m/s)
    gait:      str     # "trot" / "walk" / "stand"
    stability: float   # 0 = unstable, 1 = perfect
    phase:     str


@dataclass
class EpisodeRecord:
    episode:          int
    stage:            int
    params:           dict
    distance_reached: float
    max_phase:        str
    success:          bool
    fell:             bool
    fall_x:           Optional[float]
    fall_speed:       Optional[float]
    jump_cleared:     bool
    mean_stability:   float
    groq_reasoning:   str = ""


# ─────────────────────────────────────────────────────────────────────────────
#  EpisodeGuide
# ─────────────────────────────────────────────────────────────────────────────

class EpisodeGuide:
    """Continuous memory-driven locomotion steering.

    Args
    ----
    spec      : RobotSpec
    mapmem    : MapMem
    skillmem  : SkillMem
    safetymem : SafetyMem
    usermem   : UserMem
    hurdle_x  : x-position of hurdle (default 4.0 m)
    goal_x    : x-position of goal (default 8.0 m)
    steering  : EpisodeSteeringAgent (Groq — optional)
    """

    # Stability EMA time constants (in control steps at 50 Hz)
    _STABILITY_ALPHA = 0.10   # ≈10-step time constant — responsive but not jumpy
    _SPEED_ALPHA     = 0.08   # speed ramp — ~12 steps to reach 63% of target

    # Stability thresholds for gait/speed decisions
    _STAB_TROT  = 0.60   # above: full trot at phase speed
    _STAB_WALK  = 0.35   # above: cautious walk at half speed
    # Below _STAB_WALK: stand

    def __init__(
        self,
        spec,
        mapmem,
        skillmem,
        safetymem,
        usermem,
        hurdle_x:  float = 4.0,
        goal_x:    float = 8.0,
        steering:  "EpisodeSteeringAgent | None" = None,
    ):
        self._spec      = spec
        self._mapmem    = mapmem
        self._skillmem  = skillmem
        self._safetymem = safetymem
        self._usermem   = usermem
        self._hurdle_x  = hurdle_x
        self._goal_x    = goal_x
        self._steering  = steering

        # Curriculum
        self._stage:       int = STAGE_WALK
        self._walk_wins:   int = 0
        self._hurdle_wins: int = 0

        # Cross-episode memory
        self._history:       list[EpisodeRecord] = []
        self._best_dist:     float = 0.0
        self._best_q12_buf:  list  = []

        # Per-episode state (reset by begin_episode)
        self._params:        dict  = {}
        self._pace_log:      list  = []  # [(t, x, phase)]
        self._stab_log:      list  = []  # [stability_score per step]
        self._max_x:         float = 0.0
        self._jump_cleared:  bool  = False
        self._q12_buf:       list  = []   # joints near hurdle for SkillMem
        self._current_phase: str   = PHASE_APPROACH

        # Continuous steering state (reset by begin_episode)
        self._stability_ema: float = 0.70  # neutral start — robot just settled
        self._speed_ema:     float = 0.0   # always start from stopped

        # Groq suggestion cache
        self._groq_suggestion: dict | None = None

    # ── Stage helpers ─────────────────────────────────────────────────────

    @property
    def stage(self) -> int:
        return self._stage

    @property
    def stage_goal_x(self) -> float:
        return {STAGE_WALK: 3.0, STAGE_APPROACH: 5.0, STAGE_FULL: self._goal_x}[self._stage]

    @property
    def stability_ema(self) -> float:
        return self._stability_ema

    # ── Phase ─────────────────────────────────────────────────────────────

    def current_phase(self, x: float) -> str:
        bx = self._params.get("bound_x",  self._hurdle_x - 0.5)
        sx = self._params.get("sprint_x", self._hurdle_x - 1.5)
        if x >= self._goal_x:
            return PHASE_GOAL
        elif self._stage == STAGE_WALK:
            return PHASE_APPROACH
        elif x >= self._hurdle_x + 1.5:
            return PHASE_RECOVERY
        elif x >= bx:
            return PHASE_BOUND
        elif x >= sx:
            return PHASE_SPRINT
        else:
            return PHASE_APPROACH

    # ── Episode lifecycle ─────────────────────────────────────────────────

    def begin_episode(self, episode_n: int) -> dict:
        """Recall memory → compute params → seed UserMem. Returns params dict."""
        if self._groq_suggestion is not None:
            params = self._merge_groq(self._groq_suggestion)
            self._groq_suggestion = None
        else:
            params = self._recall(episode_n)

        if episode_n == 0 and len(self._skillmem) == 0:
            self._seed_motor_primitives()

        self._params        = params
        self._pace_log      = []
        self._stab_log      = []
        self._max_x         = 0.0
        self._jump_cleared  = False
        self._q12_buf       = []
        self._current_phase = PHASE_APPROACH
        self._stability_ema = 0.70  # neutral — robot just finished settling
        self._speed_ema     = 0.0   # always start from stopped

        # Seed UserMem with safe initial values
        self._usermem.set("max_speed",    0.0)      # zero until first step
        self._usermem.set("gait_override", "trot")
        self._usermem.set("body_height",
                          self._spec.kin.com_height_stand * params["body_height_factor"])
        return params

    # ── Continuous steering (called every control step) ───────────────────

    def step(
        self,
        t:       float,
        x:       float,
        trunk_z: float,
        rpy:     np.ndarray,
        fc:      np.ndarray,
        q12:     np.ndarray,
    ) -> SteeringOutput:
        """Core continuous steering method — must be called every control step.

        Computes a stability score from live sensor data, adapts speed and
        gait, and writes them to UserMem so LocoController applies them on
        the same step.

        Returns SteeringOutput with the adapted cmd_vx, gait, stability and
        current phase for the gym's logging and PD control.
        """
        # 1. Update position tracking
        if x > self._max_x:
            self._max_x = x

        # 2. Compute instantaneous stability score (0=bad, 1=perfect)
        raw_stability = self._stability_score(trunk_z, rpy, fc)

        # 3. Smooth it — slow EMA so robot doesn't jerk from one wobble
        self._stability_ema = (
            (1 - self._STABILITY_ALPHA) * self._stability_ema
            + self._STABILITY_ALPHA * raw_stability
        )
        self._stab_log.append(self._stability_ema)

        # 4. Determine phase and base speed for this phase
        phase = self.current_phase(x)
        if phase != self._current_phase:
            self._current_phase = phase
        # Write phase so LocoController selects the matching motor primitive
        self._usermem.set("current_phase", phase)

        base_speed = self._phase_base_speed(phase)

        # 5. Stability-gated target speed and gait
        #    Three zones: stand | walk-half | trot-full
        if self._stability_ema < self._STAB_WALK:
            target_vx = 0.0
            gait      = "stand"
        elif self._stability_ema < self._STAB_TROT:
            # Marginal zone: walk at half speed — linear ramp from 0 to 0.5
            frac      = (self._stability_ema - self._STAB_WALK) / (self._STAB_TROT - self._STAB_WALK)
            target_vx = base_speed * 0.5 * frac
            gait      = "walk"
        else:
            # Stable: full phase speed — trot
            target_vx = base_speed
            gait      = "trot"

        # 6. Smooth the speed — prevents abrupt lurch changes
        self._speed_ema = (
            (1 - self._SPEED_ALPHA) * self._speed_ema
            + self._SPEED_ALPHA * target_vx
        )
        adapted_vx = float(np.clip(self._speed_ema, 0.0, base_speed))

        # 7. Body height and step height — raise near hurdle for clearance
        bh_factor  = self._params.get("body_height_factor", 0.95)
        step_h_now = self._params.get("step_height", 0.12)
        if phase in (PHASE_SPRINT, PHASE_BOUND):
            bh_factor  = min(bh_factor + 0.04, 1.00)   # cap at 1.0 (max_body_height)
            step_h_now = max(step_h_now, 0.14)          # guarantee >10cm hurdle clearance

        # 8. Write ALL steering params to UserMem — LocoController reads on ctrl.step()
        self._usermem.set("max_speed",     adapted_vx)
        self._usermem.set("gait_override", gait)
        self._usermem.set("body_height",   self._spec.kin.com_height_stand * bh_factor)
        self._usermem.set("step_height",   step_h_now)

        # 10. Record for post-episode analysis
        self._pace_log.append((t, x, phase))
        if (self._hurdle_x - 1.2) < x < (self._hurdle_x + 1.5):
            if len(self._q12_buf) < 40:
                self._q12_buf.append(q12.astype(np.float32).copy())
        if x >= self._hurdle_x + 1.5:
            self._jump_cleared = True

        return SteeringOutput(
            cmd_vx    = adapted_vx,
            gait      = gait,
            stability = self._stability_ema,
            phase     = phase,
        )

    def _stability_score(self, trunk_z: float, rpy: np.ndarray, fc: np.ndarray) -> float:
        """Compute 0–1 stability score from raw sensor values.

        Uses a weighted-danger approach instead of multiplicative components.
        Multiplicative scoring tanks to ~0.30 during normal trot (2 contacts,
        roll≈0.10rad) and prevents the robot from ever reaching full speed.

        Danger components (each 0–1, where 1 = at hard limit):
          orient_danger — max(roll, pitch) fraction of their stop limits
          height_danger — how close the trunk is to the fallen threshold
          contact_danger — no feet on ground at all

        Normal healthy trot: roll≈0.10, pitch≈0.08, z≈0.28, 2 contacts
          → danger≈0.14 → stability≈0.86  (trot at full speed ✓)
        """
        sf = self._spec.safety
        roll  = abs(float(rpy[0]))
        pitch = abs(float(rpy[1]))

        # Orientation danger: fraction of hard stop limits
        orient_danger = max(roll / sf.roll_stop_rad, pitch / sf.pitch_stop_rad)

        # Height danger: 0 at z≥0.30m (good trot height), 1 at z≤0.15m (fallen)
        height_danger = float(max(0.0, 1.0 - (trunk_z - 0.15) / 0.15))

        # Contact danger: only fire if truly airborne (trot always has ≥1 contact)
        contact_danger = 0.0 if fc.sum() >= 1.0 else 1.0

        # Weighted sum — orientation is the dominant signal
        danger = 0.60 * orient_danger + 0.30 * height_danger + 0.10 * contact_danger
        return float(max(0.0, 1.0 - danger))

    def _phase_base_speed(self, phase: str) -> float:
        """Base (maximum) speed for the current phase, from episode params."""
        p = self._params
        if phase == PHASE_APPROACH:
            return p.get("approach_speed", 0.15)
        elif phase == PHASE_SPRINT:
            return p.get("sprint_speed", 0.25)
        elif phase == PHASE_BOUND:
            return p.get("bound_speed", p.get("sprint_speed", 0.25))
        elif phase == PHASE_RECOVERY:
            return p.get("approach_speed", 0.15)
        return 0.0

    # ── Episode end ───────────────────────────────────────────────────────

    def end_episode(
        self,
        success:    bool,
        fell:       bool,
        fall_x:     float | None = None,
        fall_speed: float | None = None,
    ) -> EpisodeRecord:
        """Record outcome → update all memory compartments → trigger Groq."""
        max_phase = self._furthest_phase()
        mean_stab = float(np.mean(self._stab_log)) if self._stab_log else 0.0

        record = EpisodeRecord(
            episode          = len(self._history),
            stage            = self._stage,
            params           = dict(self._params),
            distance_reached = self._max_x,
            max_phase        = max_phase,
            success          = success,
            fell             = fell,
            fall_x           = fall_x,
            fall_speed       = fall_speed,
            jump_cleared     = self._jump_cleared,
            mean_stability   = mean_stab,
        )
        self._history.append(record)

        if self._max_x > self._best_dist:
            self._best_dist  = self._max_x
            self._best_q12_buf = list(self._q12_buf)

        # Memory writes
        if self._jump_cleared and len(self._q12_buf) >= 10:
            self._update_skillmem(record)
        if fell and (fall_speed or 0) > 0.9 and (fall_x or 99) < 2.0:
            self._update_safetymem(record)
        if self._jump_cleared:
            self._update_mapmem(record)

        # Curriculum advancement
        if self._stage == STAGE_WALK:
            if success:
                self._walk_wins += 1
                if self._walk_wins >= 2:
                    self._stage = STAGE_APPROACH
                    print("  [Guide] Stage 1 — Approach hurdle unlocked")
            else:
                self._walk_wins = 0
        elif self._stage == STAGE_APPROACH and self._jump_cleared:
            self._hurdle_wins += 1
            if self._hurdle_wins >= 1:
                self._stage = STAGE_FULL
                print("  [Guide] Stage 2 — Full task unlocked")

        # Groq steering (background)
        if self._steering is not None:
            import threading
            threading.Thread(target=self._run_steering, args=(record,), daemon=True).start()

        return record

    # ── Statistical recall ────────────────────────────────────────────────

    def _recall(self, episode_n: int) -> dict:
        """Compute episode params from memory history.

        Speeds start very conservative (walk pace) and increase only as
        the robot demonstrates consistent stability.
        """
        hx = self._hurdle_x

        # Conservative defaults — tuned for stability-first locomotion in sim
        defaults = {
            "approach_speed":     0.15,   # slow walk pace — ramp up with stability
            "sprint_speed":       0.25,   # hurdle approach speed (gentle)
            "bound_speed":        0.25,   # same — high-step trot near hurdle
            "sprint_x":           hx - 1.5,
            "bound_x":            hx - 0.5,
            "body_height_factor": 0.95,   # FK-verified: 0.95*0.280=0.266m, feet ~1cm above ground
            "step_height":        0.12,   # 12cm clears 10cm hurdle with margin
        }

        if not self._history:
            return defaults

        rng  = np.random.default_rng(episode_n + 13)
        recent = self._history[-min(5, len(self._history)):]
        mean_stab = float(np.mean([r.mean_stability for r in recent]))

        successful  = [r for r in self._history if r.success]
        cleared     = [r for r in self._history if r.jump_cleared]
        fell_early  = [r for r in self._history
                       if r.fell and (r.fall_x or 99) < hx - 1.0]
        fell_hurdle = [r for r in self._history
                       if r.fell and hx - 1.0 <= (r.fall_x or -99) < hx + 2.0]

        params = dict(defaults)

        if successful:
            w = np.array([max(r.distance_reached, 0.01) for r in successful], dtype=np.float32)
            w = w / w.sum()
            params["approach_speed"] = float(np.clip(
                sum(wi * r.params["approach_speed"] for wi, r in zip(w, successful))
                + rng.normal(0, 0.02), 0.10, 0.60))
            params["sprint_speed"] = float(np.clip(
                sum(wi * r.params["sprint_speed"] for wi, r in zip(w, successful))
                + rng.normal(0, 0.02), 0.15, 0.80))
            params["step_height"] = float(np.clip(
                sum(wi * r.params["step_height"] for wi, r in zip(w, successful))
                + rng.normal(0, 0.005), 0.08, 0.20))

        elif cleared:
            best = max(cleared, key=lambda r: r.distance_reached)
            params["approach_speed"] = min(best.params["approach_speed"] * 1.05, 0.60)
            params["sprint_speed"]   = min(best.params["sprint_speed"]   * 1.05, 0.80)
            params["step_height"]    = best.params["step_height"]

        elif fell_hurdle:
            last = fell_hurdle[-1]
            params["approach_speed"] = max(last.params["approach_speed"] * 0.90, 0.10)
            params["sprint_speed"]   = max(last.params["sprint_speed"]   * 0.90, 0.15)
            params["step_height"]    = min(last.params["step_height"] + 0.015, 0.20)

        elif fell_early:
            last = self._history[-1]
            params["approach_speed"] = max(last.params["approach_speed"] * 0.85, 0.10)
            params["sprint_speed"]   = max(last.params["sprint_speed"]   * 0.90, 0.15)

        else:
            last = self._history[-1]
            # Increase speed only if mean stability was high
            speed_factor = 1.0 + 0.06 * max(mean_stab - 0.5, 0.0)
            params["approach_speed"] = min(last.params["approach_speed"] * speed_factor, 0.60)
            params["sprint_speed"]   = min(last.params["sprint_speed"]   * speed_factor, 0.80)

        return params

    def _merge_groq(self, suggestion: dict) -> dict:
        base = self._recall(len(self._history))
        for key, lo, hi in [
            ("approach_speed",     0.10, 0.60),
            ("sprint_speed",       0.15, 0.80),
            ("step_height",        0.06, 0.22),
            ("body_height_factor", 0.85, 1.00),
        ]:
            if key in suggestion:
                base[key] = float(np.clip(suggestion[key], lo, hi))
        return base

    # ── Memory writes ─────────────────────────────────────────────────────

    def _seed_motor_primitives(self) -> None:
        """Seed SkillMem with two analytically-generated motor primitives.

        trot_forward — 60 steps at approach_speed, normal step height (0.08 m).
                       Used during APPROACH and RECOVERY phases.
        trot_hurdle  — 60 steps at sprint_speed,   high step height (0.15 m).
                       Used during SPRINT and BOUND phases (near the hurdle).

        Both use the actual GaitEngine so waypoints contain real kinematic
        joint angles that reinforce correct 4-leg coordination — not the
        static stand pose which destroyed swing-leg motion.
        """
        from cadenza_local.locomotion.memory.skillmem import Skill
        from cadenza_local.locomotion.gait_engine import GaitEngine
        from cadenza_local.locomotion.robot_spec import GAITS, GaitParams

        DT       = 0.02                   # 50 Hz control period
        N_STEPS  = 60                     # 1.2 s ≈ 2.4 trot cycles at 2 Hz (real Go1 data)
        rpy_zero = np.zeros(3, dtype=np.float32)

        def _make_trot_waypoints(speed: float, step_h: float) -> np.ndarray:
            bh  = self._spec.kin.com_height_stand * 0.95   # FK-verified height
            eng = GaitEngine(self._spec, "trot", bh)
            g   = GAITS["trot"]
            eng._gait = GaitParams(
                name=g.name, freq_hz=g.freq_hz, duty_cycle=g.duty_cycle,
                phase_offsets=g.phase_offsets, swing_height=step_h,
                max_speed=g.max_speed, max_yaw=g.max_yaw, description=g.description,
            )
            cmd = np.array([speed, 0.0, 0.0], dtype=np.float32)
            wps = np.zeros((N_STEPS, 12), dtype=np.float32)
            for i in range(N_STEPS):
                wps[i] = eng.step(DT, cmd, rpy_zero)
            return wps, cmd

        approach_spd = self._params.get("approach_speed", 0.15)
        sprint_spd   = self._params.get("sprint_speed",   0.25)

        # ── Primitive 1: trot_forward (approach / recovery) ──────────────
        step_h_fwd = 0.08
        wps_fwd, cmd_fwd = _make_trot_waypoints(approach_spd, step_h_fwd)
        self._skillmem._skills.append(Skill(
            name      = "trot_forward",
            goal_emb  = _skill_embedding(cmd_fwd, step_h_fwd),
            waypoints = wps_fwd,
            tags      = ["trot", "forward", "approach"],
            extra     = {"step_height": step_h_fwd, "seeded": True},
        ))

        # ── Primitive 2: trot_hurdle (sprint / bound near obstacle) ──────
        step_h_hur = 0.15
        wps_hur, cmd_hur = _make_trot_waypoints(sprint_spd, step_h_hur)
        self._skillmem._skills.append(Skill(
            name      = "trot_hurdle",
            goal_emb  = _skill_embedding(cmd_hur, step_h_hur),
            waypoints = wps_hur,
            tags      = ["trot", "hurdle", "high_step"],
            extra     = {"step_height": step_h_hur, "seeded": True},
        ))

        self._skillmem._build_index()
        print(f"  [Guide] Motor primitives seeded: trot_forward ({approach_spd:.2f}m/s, "
              f"h={step_h_fwd:.2f}m)  trot_hurdle ({sprint_spd:.2f}m/s, h={step_h_hur:.2f}m)")

    def _update_skillmem(self, record: EpisodeRecord) -> None:
        from cadenza_local.locomotion.memory.skillmem import Skill, goal_to_embedding
        q12_arr   = np.stack(self._q12_buf, axis=0).astype(np.float32)
        existing  = [s for s in self._skillmem._skills
                     if "hurdle" in s.name or "jump" in s.name]
        if existing:
            best_dist = max(s.extra.get("distance_reached", 0.0) for s in existing)
            if record.distance_reached <= best_dist:
                return
        cmd  = np.array([record.params["sprint_speed"], 0.0, 0.0], dtype=np.float32)
        skill = Skill(
            name      = f"hurdle_ep{record.episode}",
            goal_emb  = goal_to_embedding(cmd, "step over hurdle"),
            waypoints = q12_arr,
            tags      = ["hurdle", "trot"],
            extra     = {"distance_reached": record.distance_reached,
                         "episode": record.episode},
        )
        self._skillmem._skills = [
            s for s in self._skillmem._skills
            if "hurdle" not in s.name and "jump" not in s.name
        ]
        self._skillmem._skills.append(skill)
        self._skillmem._build_index()

    def _update_safetymem(self, record: EpisodeRecord) -> None:
        from cadenza_local.locomotion.memory.safetymem import SafetyRule
        # Only add rule at 90% of hard limit — won't fire during normal walking
        pitch_thresh = float(self._spec.safety.pitch_stop_rad * 0.90)
        safe_vx      = float((record.fall_speed or 0.5) * 0.75)
        rule_name    = "early_fall_speed_cap"
        self._safetymem._rules = [r for r in self._safetymem._rules
                                   if r.name != rule_name]
        self._safetymem._rules.append(SafetyRule(
            name="early_fall_speed_cap", field="imu_rpy", axis=1,
            min_val=-float("inf"), max_val=pitch_thresh,
            override={"cmd_vx": safe_vx}, priority=3,
        ))
        self._safetymem._rules.sort(key=lambda r: r.priority, reverse=True)

    def _update_mapmem(self, record: EpisodeRecord) -> None:
        from cadenza_local.locomotion.memory.mapmem import TerrainCluster
        centroid = np.array([
            record.params["approach_speed"] / max(self._spec.benchmarks.flat_speed_ms, 0.1),
            record.params["step_height"],
            float(record.jump_cleared),
        ], dtype=np.float32)
        n = float(np.linalg.norm(centroid))
        if n > 1e-8:
            centroid = centroid / n
        self._mapmem._clusters = [c for c in self._mapmem._clusters
                                   if c.label != "hurdle_approach"]
        self._mapmem._clusters.append(TerrainCluster(
            label="hurdle_approach", centroid=centroid,
            speed_limit=record.params["sprint_speed"],
            step_height=record.params["step_height"],
            gait="trot",
            extra={"source": "episode_guide", "episode": record.episode},
        ))
        if self._mapmem._clusters:
            self._mapmem._build_index()

    # ── Groq steering ─────────────────────────────────────────────────────

    def _run_steering(self, record: EpisodeRecord) -> None:
        try:
            suggestion = self._steering.suggest(
                episode=record.episode,
                distance_reached=record.distance_reached,
                fell=record.fell,
                fall_x=record.fall_x,
                fall_speed=record.fall_speed,
                max_phase=record.max_phase,
                jump_cleared=record.jump_cleared,
                approach_pace=float(np.mean(
                    [x for _, x, p in self._pace_log if p == PHASE_APPROACH]
                    or [0]
                )),
                prev_params=record.params,
                history_summary=self.history_summary(),
                max_speed=self._spec.benchmarks.flat_speed_ms,
            )
            if suggestion:
                self._groq_suggestion = suggestion
                reasoning = suggestion.get("reasoning", "")
                if reasoning:
                    agent_tag = getattr(self._steering, "model",
                                        type(self._steering).__name__)
                    print(f"\n  [Steer/{agent_tag}] ep{record.episode+1}: {reasoning}\n")
        except Exception as e:
            print(f"  [Steer] {e}")

    # ── Helpers ───────────────────────────────────────────────────────────

    def _furthest_phase(self) -> str:
        order   = [PHASE_APPROACH, PHASE_SPRINT, PHASE_BOUND, PHASE_RECOVERY, PHASE_GOAL]
        reached = {p for _, _, p in self._pace_log}
        for ph in reversed(order):
            if ph in reached:
                return ph
        return PHASE_APPROACH

    def print_summary(self, record: EpisodeRecord) -> None:
        outcome    = "SUCCESS" if record.success else ("FELL" if record.fell else "timeout")
        stage_name = {0: "Walk", 1: "Approach", 2: "Full"}[record.stage]
        print(
            f"\n  ── ep{record.episode:>3} [{stage_name}]  {outcome:<8}  "
            f"dist={record.distance_reached:.2f}m  "
            f"hurdle={'✓' if record.jump_cleared else '✗'}  "
            f"stab={record.mean_stability:.2f}  phase={record.max_phase}"
        )
        print(
            f"           params   approach={record.params['approach_speed']:.2f}  "
            f"sprint={record.params['sprint_speed']:.2f}  "
            f"step_h={record.params['step_height']:.2f}"
        )
        if record.fell and record.fall_x is not None:
            print(f"           fell     x={record.fall_x:.2f}m  "
                  f"speed={record.fall_speed:.2f}m/s")
        print(
            f"           memory   SkillMem={len(self._skillmem)}  "
            f"SafetyMem={len(self._safetymem)}  MapMem={len(self._mapmem)}\n"
        )

    def history_summary(self) -> str:
        if not self._history:
            return "no episodes yet"
        n_succ = sum(1 for r in self._history if r.success)
        n_clr  = sum(1 for r in self._history if r.jump_cleared)
        n_fell = sum(1 for r in self._history if r.fell)
        return (f"{len(self._history)} eps: {n_succ} success / "
                f"{n_clr} cleared / {n_fell} fell | best={self._best_dist:.2f}m "
                f"stage={self._stage}")
