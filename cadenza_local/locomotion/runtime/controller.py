"""LocoController — robot-specific memory-augmented locomotion controller.

Integrates:
  - Complete Unitree Go1/Go2 physics spec (robot_spec.py)
  - Analytical forward/inverse kinematics (kinematics.py)
  - Precise gait generation — no RL (gait_engine.py)
  - Online terrain classification (terrain_classifier.py)
  - Groq-powered navigation/memory/benchmark agents (groq_agent.py)
  - Five memory compartments (stm, mapmem, skillmem, safetymem, usermem)

step() pipeline (runs at cfg.control.rate_hz — default 50 Hz):
    1.  STM.push(frame)
    2.  terrain_est  = TerrainClassifier.classify(stm)
    3.  safety_check = SafetyMem.check(frame) ← hard rules first
    4.  If safety triggered → override cmd, flag
    5.  gait_engine.step(dt, cmd_vel, rpy) → q12_target
    6.  Blend q12_target with SkillMem waypoint (skill_alpha)
    7.  Apply UserMem preferences (speed cap, gait override)
    8.  Apply GroqAdvisor latest decision (non-blocking)
    9.  Log step
    10. Return LocoCommand
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from cadenza_local.locomotion.memory.stm       import STM, STMFrame
from cadenza_local.locomotion.memory.mapmem    import MapMem, stm_to_embedding
from cadenza_local.locomotion.memory.skillmem  import (
    SkillMem, goal_to_embedding, goal_to_embedding_4d,
)
from cadenza_local.locomotion.memory.safetymem import SafetyMem
from cadenza_local.locomotion.memory.usermem   import UserMem
from cadenza_local.locomotion.robot_spec       import get_spec, RobotSpec, GAITS
from cadenza_local.locomotion.gait_engine      import GaitEngine
from cadenza_local.locomotion.terrain_classifier import TerrainClassifier, TerrainEstimate
from cadenza_local.locomotion.kinematics       import clip_joints, check_joint_margins

if TYPE_CHECKING:
    from cadenza_local.locomotion.config           import UnitreeConfig
    from cadenza_local.locomotion.runtime.logger   import ExperienceLogger
    from cadenza_local.locomotion.groq_agent       import GroqAdvisor


# ──────────────────────────────────────────────────────────────────────────────
#  LocoCommand — rich output with all diagnostic info
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class LocoCommand:
    """Full output of LocoController.step().

    Consumers:
      - ROS2 publisher (ros2_node.py) — serialise to JSON
      - Direct joint controller — use q12_target + pd_gains
      - Logging (ExperienceLogger)
    """
    # Velocity commands (m/s, rad/s)
    cmd_vx:      float = 0.0
    cmd_vy:      float = 0.0
    cmd_yaw:     float = 0.0

    # Gait
    gait:         str   = "trot"
    step_height:  float = 0.08
    body_height:  float = 0.32

    # Joint targets (12-dim) + PD gains
    q12_target:   np.ndarray = field(default_factory=lambda: np.zeros(12, dtype=np.float32))
    kp:           float = 20.0
    kd:           float =  0.5

    # Terrain
    terrain:      str   = "unknown"
    terrain_conf: float = 0.0
    slope_deg:    float = 0.0
    slip_risk:    float = 0.0

    # Status flags
    safety_active: bool = False
    active_safety_rules: list = field(default_factory=list)
    joint_warnings: list = field(default_factory=list)
    skill_name:    str  = ""
    groq_reasoning: str = ""
    on_benchmark:  bool = True

    # Timing
    timestamp:    float = field(default_factory=time.monotonic)

    def to_dict(self) -> dict:
        return {
            "cmd_vx":          self.cmd_vx,
            "cmd_vy":          self.cmd_vy,
            "cmd_yaw":         self.cmd_yaw,
            "gait":            self.gait,
            "step_height":     self.step_height,
            "body_height":     self.body_height,
            "q12_target":      self.q12_target.tolist(),
            "kp":              self.kp,
            "kd":              self.kd,
            "terrain":         self.terrain,
            "terrain_conf":    self.terrain_conf,
            "slope_deg":       self.slope_deg,
            "slip_risk":       self.slip_risk,
            "safety_active":   self.safety_active,
            "safety_rules":    self.active_safety_rules,
            "joint_warnings":  self.joint_warnings,
            "skill_name":      self.skill_name,
            "groq_reasoning":  self.groq_reasoning,
            "on_benchmark":    self.on_benchmark,
            "timestamp":       self.timestamp,
        }


# ──────────────────────────────────────────────────────────────────────────────
#  Controller
# ──────────────────────────────────────────────────────────────────────────────

class LocoController:
    """Memory-augmented locomotion controller for Unitree Go1/Go2.

    No RL. No fine-tuning. Robot-specific physics baked in.

    Args:
        cfg:           UnitreeConfig (from YAML)
        stm:           STM instance
        mapmem:        MapMem (terrain clusters)
        skillmem:      SkillMem (motion library)
        safetymem:     SafetyMem (safety rules)
        usermem:       UserMem (operator preferences)
        logger:        ExperienceLogger (optional)
        groq_advisor:  GroqAdvisor (optional — requires GROQ_API_KEY)
        spec:          RobotSpec (auto-loaded from cfg.robot_model if None)
    """

    def __init__(
        self,
        cfg:          "UnitreeConfig",
        stm:          STM,
        mapmem:       MapMem,
        skillmem:     SkillMem,
        safetymem:    SafetyMem,
        usermem:      UserMem,
        logger:       "ExperienceLogger | None" = None,
        groq_advisor: "GroqAdvisor | None"      = None,
        spec:         "RobotSpec | None"        = None,
    ):
        self._cfg       = cfg
        self._stm       = stm
        self._mapmem    = mapmem
        self._skillmem  = skillmem
        self._safetymem = safetymem
        self._usermem   = usermem
        self._logger    = logger
        self._groq      = groq_advisor

        # Load robot spec
        self._spec: RobotSpec = spec or get_spec(cfg.robot_model)

        # Sub-systems
        body_h = float(self._usermem.get("body_height", self._spec.kin.com_height_stand * 0.8))
        init_gait = str(self._usermem.get("gait_override", "") or "trot")
        self._gait_engine   = GaitEngine(self._spec, init_gait, body_h)
        self._terrain_cls   = TerrainClassifier(self._spec)

        # Skill tracking
        self._active_skill   = None
        self._skill_step     = 0

        # Timing
        self._last_t   = time.monotonic()
        self._step_n   = 0

        # Terrain history for Groq
        self._terrains_seen: list[str] = []

    # ── Public API ──────────────────────────────────────────────────────────

    def step(
        self,
        frame:    STMFrame,
        cmd_vel:  np.ndarray,    # (3,) vx, vy, yaw_rate
        task_text: str = "",
    ) -> LocoCommand:
        """Compute one LocoCommand.

        Args:
            frame:     latest STMFrame
            cmd_vel:   (3,) velocity command
            task_text: natural-language task (fed to Groq navigator)

        Returns:
            LocoCommand
        """
        now = time.monotonic()
        dt  = float(np.clip(now - self._last_t, 0.001, 0.1))
        self._last_t = now
        self._step_n += 1

        # 1. Update STM
        self._stm.push(frame)

        # 2. Terrain classification
        actual_speed = float(np.linalg.norm(cmd_vel[:2]))
        terrain_est  = self._terrain_cls.classify(self._stm, actual_speed)
        if terrain_est.label not in self._terrains_seen:
            self._terrains_seen.append(terrain_est.label)

        # 3. Safety check (hard rules — runs before everything else)
        safety_result = self._safetymem.check(frame)
        safety_active = safety_result.triggered

        # Also apply spec-level hard safety
        roll  = float(frame.imu_rpy[0])
        pitch = float(frame.imu_rpy[1])
        spec_violations: list[str] = []
        sf = self._spec.safety
        if abs(roll)  > sf.roll_stop_rad:
            spec_violations.append(f"roll={roll:.3f}rad exceeds limit {sf.roll_stop_rad}")
        if abs(pitch) > sf.pitch_stop_rad:
            spec_violations.append(f"pitch={pitch:.3f}rad exceeds limit {sf.pitch_stop_rad}")
        if spec_violations:
            safety_active = True

        # 4. Determine gait and speed
        gait_name   = terrain_est.recommended_gait
        speed_limit = terrain_est.max_speed
        step_h      = terrain_est.step_height

        # Apply user preferences — all four values are written every step by EpisodeGuide.
        # These MUST override terrain estimates: EpisodeGuide knows the task context better.
        user_max_speed = float(self._usermem.get("max_speed", speed_limit))
        gait_override  = str(self._usermem.get("gait_override", "") or "")
        if gait_override and gait_override in GAITS:
            gait_name = gait_override
        # Body height — EpisodeGuide raises this near the hurdle
        user_body_height = self._usermem.get("body_height", None)
        if user_body_height is not None:
            self._gait_engine.set_body_height(float(user_body_height))
        # Step height — EpisodeGuide raises this to ≥0.14m near hurdle for clearance
        user_step_height = self._usermem.get("step_height", None)
        if user_step_height is not None:
            step_h = float(user_step_height)

        # 5. Apply Groq navigator decision (non-blocking — uses last result)
        groq_reasoning = ""
        if self._groq is not None:
            n_contacts = int(frame.foot_contact.sum())
            nav = self._groq.update_state(
                terrain_label  = terrain_est.label,
                terrain_conf   = terrain_est.confidence,
                slope_deg      = terrain_est.slope_deg,
                slip_risk      = terrain_est.slip_risk,
                roll_rad       = roll,
                pitch_rad      = pitch,
                n_contacts     = n_contacts,
                cmd_vx         = float(cmd_vel[0]),
                current_gait   = gait_name,
                actual_speed   = actual_speed,
                task_text      = task_text,
                step_dict      = {"frame": {
                    "joint_pos":    frame.joint_pos.tolist(),
                    "joint_vel":    frame.joint_vel.tolist(),
                    "imu_rpy":      frame.imu_rpy.tolist(),
                    "imu_omega":    frame.imu_omega.tolist(),
                    "foot_contact": frame.foot_contact.tolist(),
                    "cmd_vel":      frame.cmd_vel.tolist(),
                }, "t": frame.timestamp},
            )
            if nav is not None:
                gait_name  = nav.gait if nav.gait in GAITS else gait_name
                speed_limit = min(speed_limit, nav.speed_limit)
                step_h      = nav.step_height
                self._gait_engine.set_body_height(nav.body_height)
                groq_reasoning = nav.reasoning
                if nav.caution_flags:
                    spec_violations.extend(nav.caution_flags)

        # 6. Safety override: cut speed to 0 if triggered
        if safety_active or spec_violations:
            cmd_vel_safe = np.zeros(3, dtype=np.float32)
            gait_name    = "stand"
        else:
            # Effective limit = min(terrain cap, UserMem cap from EpisodeGuide)
            effective_limit = min(speed_limit, user_max_speed)
            # If speed cap is essentially zero, hold stand pose (don't trot in place)
            if effective_limit < 0.02:
                cmd_vel_safe = np.zeros(3, dtype=np.float32)
                gait_name    = "stand"
            else:
                vx_clamped  = float(np.clip(cmd_vel[0], -effective_limit, effective_limit))
                vy_clamped  = float(np.clip(cmd_vel[1], -effective_limit * 0.5, effective_limit * 0.5))
                yaw_clamped = float(np.clip(cmd_vel[2], -GAITS[gait_name].max_yaw,
                                            GAITS[gait_name].max_yaw))
                cmd_vel_safe = np.array([vx_clamped, vy_clamped, yaw_clamped], dtype=np.float32)

        # 7. Update gait engine and get joint targets
        self._gait_engine.set_gait(gait_name)
        self._gait_engine._gait = GAITS[gait_name]  # update swing height
        # Propagate step height to gait engine
        import cadenza_local.locomotion.robot_spec as _rs
        if gait_name in _rs.GAITS:
            g = _rs.GAITS[gait_name]
            # Create modified gait with current step height
            self._gait_engine._gait = _rs.GaitParams(
                name=g.name, freq_hz=g.freq_hz, duty_cycle=g.duty_cycle,
                phase_offsets=g.phase_offsets, swing_height=step_h,
                max_speed=g.max_speed, max_yaw=g.max_yaw, description=g.description,
            )

        q12 = self._gait_engine.step(dt, cmd_vel_safe, frame.imu_rpy)

        # 8. Blend with skill waypoint if a good match exists.
        #
        # Phase-aware skill selection: near the hurdle (bound/sprint) we want the
        # trot_hurdle primitive; elsewhere trot_forward.  The 4-dim embedding encodes
        # step_height so the correct primitive is selected automatically.
        #
        # If SkillMem is empty or best cosine < 0.50, skill=None and no blending
        # happens — the pure gait engine output is used.  This prevents the old
        # stand_baseline skill from corrupting swing-leg motion.
        skill_name = ""
        if len(self._skillmem) > 0:
            # Determine step height query from UserMem + phase
            current_phase = str(self._usermem.get("current_phase", "approach") or "approach")
            _HURDLE_PHASES = ("bound", "sprint")
            if current_phase in _HURDLE_PHASES:
                query_step_h = float(np.clip(
                    self._usermem.get("step_height", 0.15) or 0.15, 0.12, 0.20))
            else:
                query_step_h = float(np.clip(
                    self._usermem.get("step_height", 0.08) or 0.08, 0.06, 0.12))

            # Use 4-dim embedding if skills were seeded by _seed_motor_primitives;
            # fall back to 3-dim for legacy 3-dim skill snapshots.
            stored_dim = len(self._skillmem._skills[0].goal_emb) if self._skillmem._skills else 3
            if stored_dim == 4:
                query_emb = goal_to_embedding_4d(cmd_vel_safe, query_step_h)
            else:
                query_emb = goal_to_embedding(cmd_vel_safe)

            skill = self._skillmem.best_skill(query_emb, min_similarity=0.50)
            if skill is not None and len(skill.waypoints) > 0:
                if skill is not self._active_skill:
                    self._active_skill = skill
                    self._skill_step   = 0
                skill_name = skill.name
                wp_idx     = self._skill_step % len(skill.waypoints)
                wp         = skill.waypoints[wp_idx].astype(np.float32)
                alpha      = self._cfg.control.skill_alpha   # 0.15 — gentle nudge
                if len(wp) == 12:
                    q12 = (1 - alpha) * q12 + alpha * clip_joints(wp, self._spec)
                self._skill_step += 1

        # 9. Final joint clamp + margin check
        q12 = clip_joints(q12, self._spec)
        joint_warns = check_joint_margins(q12, self._spec)

        # 10. PD gains — switch to stance/swing values
        motor = self._spec.motor
        n_stance = int(frame.foot_contact.sum())
        kp = motor.kp_stance if n_stance >= 3 else motor.kp_swing
        kd = motor.kd_stance if n_stance >= 3 else motor.kd_swing

        # 11. Benchmark evaluation
        bench = self._terrain_cls.benchmark(self._stm, actual_speed, gait_name)

        # 12. Build command
        loco_cmd = LocoCommand(
            cmd_vx           = float(cmd_vel_safe[0]),
            cmd_vy           = float(cmd_vel_safe[1]),
            cmd_yaw          = float(cmd_vel_safe[2]),
            gait             = gait_name,
            step_height      = step_h,
            body_height      = self._gait_engine.body_height,
            q12_target       = q12,
            kp               = kp,
            kd               = kd,
            terrain          = terrain_est.label,
            terrain_conf     = terrain_est.confidence,
            slope_deg        = terrain_est.slope_deg,
            slip_risk        = terrain_est.slip_risk,
            safety_active    = safety_active or bool(spec_violations),
            active_safety_rules = safety_result.active_rules + spec_violations,
            joint_warnings   = joint_warns,
            skill_name       = skill_name,
            groq_reasoning   = groq_reasoning,
            on_benchmark     = bench.on_benchmark,
            timestamp        = frame.timestamp,
        )

        # 13. Log
        if self._logger is not None:
            self._logger.log(frame, loco_cmd)

        # 14. Groq fall detection
        if self._groq and safety_active and "roll" in str(spec_violations):
            self._groq.notify_fall()

        return loco_cmd

    def reset(self) -> None:
        """Clear STM and skill tracking (call at episode start)."""
        self._stm.clear()
        self._active_skill = None
        self._skill_step   = 0
        self._terrains_seen.clear()
        self._last_t = time.monotonic()

    def notify_episode_outcome(self, success: bool, note: str = "") -> None:
        """Record episode outcome in logger and trigger Groq memory update."""
        if self._logger:
            self._logger.episode_outcome(success=success, note=note)
        if self._groq:
            self._groq.notify_episode_end(
                success=success,
                failure_reason="" if success else note,
            )

    @property
    def spec(self) -> RobotSpec:
        return self._spec

    @property
    def gait_engine(self) -> GaitEngine:
        return self._gait_engine

    @property
    def terrain_classifier(self) -> TerrainClassifier:
        return self._terrain_cls
