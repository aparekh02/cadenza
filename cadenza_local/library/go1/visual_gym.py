"""visual_gym.py — Stand-up task for Unitree Go1.

The robot starts collapsed / crumpled on the ground and must stand up.
That is the only task: JUST STAND.

Memory system
-------------
Five compartments work together every 50 Hz step:

  UserMem   — stores ramp_rate (how fast to raise body height target).
               The memory layer adjusts this based on past outcomes.
  SkillMem  — on successful stand, records the joint trajectory so future
               episodes can blend it in as a reference.
  SafetyMem — if the robot rolls over, a safety rule is stored to prevent
               that speed / configuration in future episodes.
  STM       — rolling window of recent sensor frames for terrain estimation.
  BalanceController — always-active CoM stabiliser: resists roll/pitch/sag
               so the robot holds position once upright.

Episode lifecycle
-----------------
  1. Drop robot to ground in prone pose (300 physics steps, no PD).
  2. Ramp the IK body-height target from collapsed (~0.12 m) → nominal (0.266 m)
     at ramp_rate m/step.  Joint targets computed analytically via kinematics.
  3. Apply BalanceController corrections to stance-leg joints.
  4. PD torques sent to MuJoCo (kp_stance=60, kd=2; kp_swing=20, kd=0.5).
  5. Success = trunk_z > 0.24 m sustained for 100 consecutive steps (2 s).
  6. Cross-episode LLM steering (LocalSteeringAgent / Groq) tunes ramp_rate.

Usage
-----
    .venv/bin/mjpython examples/unitree_go1/visual_gym.py
    .venv/bin/mjpython examples/unitree_go1/visual_gym.py --episodes 20
    .venv/bin/mjpython examples/unitree_go1/visual_gym.py --local-model llama3.2:3b
    .venv/bin/mjpython examples/unitree_go1/visual_gym.py --groq
    .venv/bin/mjpython examples/unitree_go1/visual_gym.py --resume go1-xxxx
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

# ── Repo root + .env ──────────────────────────────────────────────────────────
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_env_file = _REPO / ".env"
if _env_file.exists():
    for _l in _env_file.read_text().splitlines():
        _l = _l.strip()
        if _l and not _l.startswith("#") and "=" in _l:
            k, v = _l.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

import mujoco
import mujoco.viewer

from cadenza_local.locomotion import (
    SkillMem, SafetyMem, UserMem,
    ExperienceLogger,
    LocalSteeringAgent,
    get_spec, load_config,
    FOOT_ORDER, LEG_INDICES,
)
from cadenza_local.locomotion.kinematics import (
    ik_leg, nominal_foot_positions, legs_to_joint_vector, clip_joints,
)
from cadenza_local.locomotion.balance import BalanceController

# ──────────────────────────────────────────────────────────────────────────────
_XML     = Path(__file__).parent / "go1.xml"
_MEM_DIR = Path.home() / ".cadenza" / "runs"

# Stand-up task thresholds
_STAND_Z          = 0.23    # trunk height to consider "standing" (m)
_STAND_HOLD_STEPS = 80      # must hold _STAND_Z for this many steps (1.6 s at 50 Hz)
_COLLAPSE_Z       = 0.09    # if trunk drops below this after initial rise → fail
_EPISODE_TIMEOUT_S = 90.0   # generous: initial speed gives ~40s per episode

# Starting configuration
_PRONE_Z    = 0.12          # initial body placement height (m) before drop
_DROP_STEPS = 400           # long settle: robot must be fully at rest before we start

# ── Velocity-limited joint control ────────────────────────────────────────────
# Instead of a cosine ramp (which has a peak slope at t=0.5 that creates joint-lag
# energy bursts), each control step the commanded joint target is incremented by
# AT MOST max_dq_per_step radians per joint.  This is a hard velocity cap:
# joints cannot physically move faster regardless of PD gains.
#
# max_dq_per_step lives in UserMem so the AI agent controls it.
# At 50 Hz:  0.0010 rad/step = 0.05 rad/s  (very slow — ~30s for 1.5 rad)
#            0.0020 rad/step = 0.10 rad/s  (slow    — ~15s)
#            0.0050 rad/step = 0.25 rad/s  (medium  — ~6s)
#            0.0100 rad/step = 0.50 rad/s  (fast    — ~3s)
#
_DQ_INITIAL = 0.0008   # start VERY slow — agent unlocks more after proving stability
_DQ_MIN     = 0.0005   # minimum (safety floor — agent cannot go below this)
_DQ_MAX     = 0.0100   # maximum the agent may ever set

# Foot bodies for contact detection
_FOOT_BODY_NAMES = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
_FOOT_RADIUS     = 0.0165

# PD gains + force limiter.
# _MAX_ERROR caps how large a position error can be BEFORE multiplying by kp.
# This hard-limits max torque per joint = kp × MAX_ERROR regardless of lag.
# At kp=45, MAX_ERROR=0.10: max torque = 4.5 Nm/joint — enough to lift against
# gravity (~3-4 Nm per thigh) but NOT enough to generate a jump impulse.
# The velocity cap (max_dq) limits TARGET speed; MAX_ERROR limits FORCE.
# Together they make the robot slow AND gentle — neither fast nor violent.
_KP        = 45.0   # tracking gain
_KD        =  3.0   # velocity damping
_MAX_ERROR =  0.08  # rad — force cap: max torque per joint = 45 × 0.08 = 3.6 Nm
_MAX_TORQUE = 33.5


# ──────────────────────────────────────────────────────────────────────────────
#  Sensor helpers
# ──────────────────────────────────────────────────────────────────────────────

def _quat_to_rpy(q) -> np.ndarray:
    w, x, y, z = q
    roll  = math.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    pitch = math.asin(max(-1., min(1., 2*(w*y - z*x))))
    yaw   = math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return np.array([roll, pitch, yaw], dtype=np.float32)


def _foot_contacts(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    fc = np.zeros(4, dtype=np.float32)
    for i, name in enumerate(_FOOT_BODY_NAMES):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0 and data.xpos[bid, 2] < _FOOT_RADIUS * 2.5:
            fc[i] = 1.0
    return fc


# ──────────────────────────────────────────────────────────────────────────────
#  Stand-up IK: compute joint targets for a given body height
# ──────────────────────────────────────────────────────────────────────────────

def _standup_q12(spec, body_height: float) -> np.ndarray:
    """Compute 12-DOF symmetric joint targets for a given body height via IK.

    Go1 FK (verified against MuJoCo Menagerie):
        foot_z = -(L·cos(thigh) + L·cos(thigh + calf))
    The stand pose uses thigh = θ, calf = -2θ (symmetric about knee):
        foot_z = -(L·cos(θ) + L·cos(θ - 2θ)) = -(L·cos(θ) + L·cos(-θ)) = -2L·cos(θ)
    Invert:  θ = arccos(h / (2L))    calf = -2θ

    Verified: at h=0.264m, θ=arccos(0.264/0.426)=0.902≈0.9 rad,
    calf=-1.804≈-1.8 rad — matches spec stand pose exactly.
    """
    L = spec.kin.thigh_length    # 0.213 m
    h = float(np.clip(body_height, 0.10, spec.kin.max_body_height))

    # Correct Go1 IK:
    q_thigh = float(np.arccos(np.clip(h / (2.0 * L), 0.0, 1.0)))
    q_calf  = -2.0 * q_thigh    # always negative, symmetric knee

    jl  = spec.joints
    q_h = float(np.clip(0.0,     jl.hip_min,   jl.hip_max))
    q_t = float(np.clip(q_thigh, jl.thigh_min, jl.thigh_max))
    q_c = float(np.clip(q_calf,  jl.knee_min,  jl.knee_max))
    return np.tile([q_h, q_t, q_c], 4).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
#  Pre-seed SkillMem with analytical stand-up reference trajectory
# ──────────────────────────────────────────────────────────────────────────────

def _seed_standup_memory(spec, skillmem, usermem,
                         max_dq: float = _DQ_INITIAL) -> None:
    """Bake a velocity-limited stand-up trajectory into SkillMem before episode 0.

    Uses the same velocity-limited increment that the controller uses — so the
    seeded reference trajectory is physically identical to what a perfectly
    executing controller would produce.  This gives the skill blender an exact
    reference at every step.

    max_dq matches the initial speed the agent will use, so the reference is
    always the right 'shape' for the current operating speed.
    """
    from cadenza_local.locomotion.memory.skillmem import Skill

    h_end    = spec.kin.com_height_stand * 0.95   # ~0.266 m
    q_cur    = _standup_q12(spec, 0.12).copy()
    q_target = _standup_q12(spec, h_end)

    waypoints: list[np.ndarray] = []
    # Simulate velocity-limited motion until convergence, then hold for 80 frames
    max_frames = 2000   # safety cap
    for _ in range(max_frames):
        waypoints.append(q_cur.copy())
        delta = q_target - q_cur
        if np.max(np.abs(delta)) < 1e-4:
            break
        q_cur = q_cur + np.clip(delta, -max_dq, max_dq)
    # Hold at top for 80 frames so the reference includes the "hold upright" phase
    for _ in range(80):
        waypoints.append(q_target.copy())

    waypoints_arr = np.stack(waypoints, axis=0).astype(np.float32)
    n_frames      = len(waypoints_arr)

    goal_emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    skill = Skill(
        name      = "standup_reference",
        goal_emb  = goal_emb,
        waypoints = waypoints_arr,
        tags      = ["standup", "reference", "analytical"],
        extra     = {
            "source":   "velocity_limited_ik",
            "max_dq":   max_dq,
            "h_end":    float(h_end),
            "n_frames": n_frames,
            "pre_seeded": True,
        },
    )

    skillmem._skills = [s for s in skillmem._skills if "standup" not in s.name]
    skillmem._skills.insert(0, skill)
    skillmem._build_index()

    usermem.set("skill_alpha",     0.55)
    usermem.set("max_dq_per_step", max_dq)
    usermem.set("perf_phase",      "initial")
    usermem.set("consecutive_successes", 0.0)
    usermem.set("consecutive_falls",     0.0)
    usermem.set("quality_score",         0.0)

    print(f"  [SkillMem] seeded 'standup_reference': {n_frames} frames "
          f"@ max_dq={max_dq:.4f} rad/step  ({n_frames/50:.0f}s to convergence)")
    print(f"  [UserMem]  speed=VERY_SLOW ({max_dq:.4f} rad/step)  "
          f"phase=initial  — agent controls all progression\n")


# ──────────────────────────────────────────────────────────────────────────────
#  Persistence helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_resume(mem_id: str):
    path = _MEM_DIR / f"{mem_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"No memory ID '{mem_id}' at {path}")
    snap = json.loads(path.read_text())
    return (
        SkillMem.from_list(snap.get("skillmem",  [])),
        SafetyMem.from_list(snap.get("safetymem", [])),
        UserMem.from_list(snap.get("usermem",    [])),
    )


def _save_snapshot(robot, episodes, skillmem, safetymem, usermem) -> str:
    _MEM_DIR.mkdir(parents=True, exist_ok=True)
    ts     = datetime.utcnow().isoformat()
    uid    = hashlib.sha1(f"{robot}-{ts}".encode()).hexdigest()[:4]
    mem_id = f"{robot}-standup-{uid}"
    (_MEM_DIR / f"{mem_id}.json").write_text(json.dumps({
        "meta":      {"robot": robot, "task": "standup",
                      "episodes": episodes, "timestamp": ts, "id": mem_id},
        "skillmem":  skillmem.to_list()  if hasattr(skillmem,  "to_list") else [],
        "safetymem": safetymem.to_list() if hasattr(safetymem, "to_list") else [],
        "usermem":   usermem.to_list()   if hasattr(usermem,   "to_list") else [],
    }, indent=2))
    return mem_id


# ──────────────────────────────────────────────────────────────────────────────
#  AI Agent system prompt
# ──────────────────────────────────────────────────────────────────────────────

_AGENT_SYSTEM_PROMPT = """\
You are the motor-control AI agent for a Unitree Go1 quadruped robot.
Your role: analyse each episode using the memory system, assess data quality,
and decide the exact parameters for the NEXT episode before it runs.

The robot starts crumpled on the ground.  Each joint is moved toward the
standing pose at a velocity cap of max_dq_per_step radians per 50Hz step.
This hard cap prevents zooming — the robot CANNOT move faster than this limit.

════════════════════════════════════════════════════════════════════════════════
MEMORY SYSTEM — read all compartments before deciding:

  UserMem  — current speed (max_dq_per_step), performance phase, quality score
  SkillMem — best trajectory recorded, how high robot got, at what speed
  SafetyMem— safety violations logged (roll-overs, tip events)
  History  — per-episode outcomes: max_z, fell, hold_steps, speed used

════════════════════════════════════════════════════════════════════════════════
DECISION PROTOCOL — follow this ORDER every episode:

  1. ASSESS DATA QUALITY
     - Was the motion smooth and controlled? (fell=False AND max_z rose steadily)
     - Did robot reach and hold standing height?
     - Are there safety violations in SafetyMem?
     - quality: "poor" | "acceptable" | "good" | "excellent"

  2. CATEGORISE OUTCOME
     - STABLE_SUCCESS  : success=True, hold_steps >= 60, fell=False
     - PARTIAL_RISE    : fell=False, max_z >= 0.18 but not success
     - FELL_EARLY      : fell=True, max_z < 0.18  (robot tipped before rising)
     - FELL_LATE       : fell=True, max_z >= 0.18 (robot rose then tipped)
     - TIMEOUT         : not fell, not success, time ran out

  3. DETERMINE PHASE TRANSITION
     - "initial"    → only advance to "improving" after 2+ STABLE_SUCCESS in a row
     - "improving"  → only advance to "stable"   after 3+ STABLE_SUCCESS in a row
     - "stable"     → maintain unless fell → revert to "improving"

  4. SET SPEED (max_dq_per_step)
     RULES — violating these causes the robot to fall:
     - FELL_EARLY / FELL_LATE  : multiply by 0.70 (slow down 30%)
     - TIMEOUT (no rise at all): multiply by 0.80
     - PARTIAL_RISE            : hold same speed, do NOT increase
     - STABLE_SUCCESS (1st)    : hold same speed, do NOT increase yet
     - STABLE_SUCCESS (2nd consecutive): multiply by 1.10 (only 10% faster)
     - STABLE_SUCCESS (3rd+ consecutive): multiply by 1.15 (only 15% faster)
     - Speed range: 0.0005 (min, very slow) to 0.0100 (max, fast)
     - DEFAULT starting speed: 0.0008 — do NOT exceed 0.0015 until phase="improving"

════════════════════════════════════════════════════════════════════════════════
OUTPUT FORMAT — respond with ONLY valid JSON, nothing outside it:
{
  "quality_assessment": "<poor|acceptable|good|excellent> — one sentence why",
  "outcome_category":   "<STABLE_SUCCESS|PARTIAL_RISE|FELL_EARLY|FELL_LATE|TIMEOUT>",
  "perf_phase":         "<initial|improving|stable>",
  "max_dq_per_step":    <float>,
  "target_height":      <float between 0.200 and 0.266>,
  "reasoning":          "<one sentence: what you observed and why you set this speed>"
}
"""


# ──────────────────────────────────────────────────────────────────────────────
#  StandupAgent — blocking AI agent, called synchronously between episodes
# ──────────────────────────────────────────────────────────────────────────────

class StandupAgent:
    """Blocking AI agent that analyses memory and decides next episode parameters.

    Called synchronously at episode end — the next episode does NOT start until
    the agent has finished thinking.  This is the agent being "in charge".
    """

    def __init__(self, model: str = "qwen2.5:7b", host: str = "http://localhost:11434"):
        self._agent = LocalSteeringAgent(model=model, host=host)
        self._agent._SYSTEM_PROMPT = _AGENT_SYSTEM_PROMPT   # type: ignore

    @property
    def available(self) -> bool:
        return self._agent.available

    def decide(
        self,
        ep:            int,
        result:        dict,        # this episode's outcome dict
        history:       list[dict],  # all past episodes
        usermem,                    # UserMem instance
        skillmem,                   # SkillMem instance
        safetymem,                  # SafetyMem instance
    ) -> dict | None:
        """Blocking call. Returns next-episode params or None on failure."""
        import json as _json

        # ── Build full memory context for the agent ────────────────────────
        phase    = usermem.get("perf_phase",      "initial")
        cur_dq   = float(usermem.get("max_dq_per_step", _DQ_INITIAL))
        q_score  = float(usermem.get("quality_score",   0.0))
        c_succ   = int(usermem.get("consecutive_successes", 0.0))
        c_fail   = int(usermem.get("consecutive_falls",     0.0))

        best_skill = None
        if skillmem._skills:
            best_skill = max(skillmem._skills,
                             key=lambda s: s.extra.get("max_z", 0.0), default=None)

        safety_log = [
            f"ep{r.name}: roll-over rule added"
            for r in getattr(safetymem, "_rules", [])
            if "standup" in r.name
        ]

        recent = history[-5:] if len(history) >= 5 else history
        recent_summary = "; ".join(
            f"ep{r['ep']}={'OK' if r['success'] else 'FAIL'} "
            f"z={r['max_z']:.2f} dq={r.get('max_dq_per_step', _DQ_INITIAL):.4f}"
            for r in recent
        )

        user_msg = (
            f"EPISODE {ep} JUST ENDED\n"
            f"  outcome     : {'SUCCESS' if result['success'] else 'FAIL'}\n"
            f"  fell        : {result['fell']}\n"
            f"  max_trunk_z : {result['max_z']:.3f} m  (goal {result['target_height']:.3f} m)\n"
            f"  hold_steps  : {result['hold_steps']}\n"
            f"  max_dq_used : {result.get('max_dq_per_step', cur_dq):.4f} rad/step "
            f"({result.get('max_dq_per_step', cur_dq)*50:.3f} rad/s)\n"
            f"\n"
            f"CURRENT MEMORY STATE\n"
            f"  perf_phase          : {phase}\n"
            f"  quality_score       : {q_score:.2f}\n"
            f"  consecutive_success : {c_succ}\n"
            f"  consecutive_falls   : {c_fail}\n"
            f"  best_skill_max_z    : "
            f"{best_skill.extra.get('max_z', 0.0):.3f}m" if best_skill else "none yet"
            f"\n"
            f"  safety_violations   : {len(safety_log)}  {'; '.join(safety_log[:3])}\n"
            f"\n"
            f"RECENT HISTORY (last 5 episodes)\n"
            f"  {recent_summary}\n"
            f"\n"
            f"Analyse the above, then set parameters for episode {ep + 1}."
        )

        print(f"\n  [Agent] thinking about episode {ep} → {ep+1}...", flush=True)
        raw = self._agent._call(user_msg)
        if raw is None:
            print(f"  [Agent] no response — keeping current parameters")
            return None

        try:
            cleaned = raw.strip()
            if "```" in cleaned:
                lines  = cleaned.split("\n")
                cleaned = "\n".join(
                    l for l in lines if not l.strip().startswith("```")
                )
            # Find JSON object in response
            start = cleaned.find("{")
            end   = cleaned.rfind("}") + 1
            if start < 0 or end <= start:
                return None
            data = _json.loads(cleaned[start:end])

            dq_raw = float(data.get("max_dq_per_step", cur_dq))
            return {
                "max_dq_per_step":    float(np.clip(dq_raw, _DQ_MIN, _DQ_MAX)),
                "target_height":      float(np.clip(
                                          data.get("target_height", 0.250),
                                          0.200, 0.266)),
                "perf_phase":         str(data.get("perf_phase", phase)),
                "quality_assessment": str(data.get("quality_assessment", "")),
                "reasoning":          str(data.get("reasoning", "")),
            }
        except Exception as exc:
            print(f"  [Agent] parse error: {exc}")
            return None


# ──────────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────────

def main(
    robot:       str,
    episodes:    int,
    cfg_path:    Path,
    resume:      str | None,
    local_model: str,
    use_groq:    bool,
) -> None:
    spec = get_spec(robot)
    cfg  = load_config(cfg_path)

    _NOMINAL_H = spec.kin.com_height_stand * 0.95   # FK-verified stand height ~0.266 m

    print(f"\n  Cadenza  |  {robot.upper()}  |  standup task  |  {episodes} episodes")
    print(f"  Goal: stand up from prone and hold trunk_z > {_STAND_Z} m "
          f"for {_STAND_HOLD_STEPS} steps ({_STAND_HOLD_STEPS/50:.1f} s)")
    print(f"  Nominal stand height: {_NOMINAL_H:.3f} m  |  PD: kp={_KP} kd={_KD}  (uniform, all joints)\n")

    # ── Memory ────────────────────────────────────────────────────────────
    if resume:
        skillmem, safetymem, usermem = _load_resume(resume)
        print(f"  Resumed: {resume}  (Skill={len(skillmem)} Safety={len(safetymem)})\n")
    else:
        skillmem, safetymem, usermem = SkillMem(), SafetyMem(), UserMem()

    # Initial UserMem — agent sets all speed/phase values, we only set height here
    usermem.set("target_height", 0.250)
    usermem.set("body_height",   _NOMINAL_H)

    # ── Pre-seed SkillMem with analytical stand-up reference data ─────────
    if not resume:
        _seed_standup_memory(spec, skillmem, usermem, max_dq=_DQ_INITIAL)

    # ── AI Agent (blocking — decides parameters before each episode) ───────
    agent: Optional[StandupAgent] = None
    if use_groq:
        print("  [Info] --groq flag set; using local LLM agent (Groq not wired here)\n")
    _sa = StandupAgent(model=local_model)
    if _sa.available:
        agent = _sa
        print(f"  [Agent] {local_model} READY — blocks between episodes to decide speed\n")
    else:
        print(f"  [Agent] Ollama not available — statistical fallback only\n")
        print(f"  To enable: brew install ollama && ollama pull {local_model} && ollama serve\n")

    # ── Balance controller ─────────────────────────────────────────────────
    # k_height=0: IK ramp already handles height. Only roll/pitch corrections.
    # Applied to ALL joints (not just stance) — when crumpled, all legs need
    # equal correction. The stance/swing distinction only makes sense for walking.
    balance = BalanceController(
        spec,
        k_roll  = 0.15, k_pitch = 0.12, k_height = 0.00,
        k_damp_roll  = 0.05, k_damp_pitch = 0.04,
        max_correction = 0.12,   # small nudges only — large corrections add energy
    )
    print(f"  {balance}  (roll/pitch only, applied to all legs)\n")

    # ── Experience logger ──────────────────────────────────────────────────
    logger = ExperienceLogger(log_dir=cfg.memory.log_dir, run_id=f"{robot}_standup")

    # ── MuJoCo model ──────────────────────────────────────────────────────
    model = mujoco.MjModel.from_xml_path(str(_XML))
    data  = mujoco.MjData(model)

    prone_pose = np.array(spec.poses.prone, dtype=np.float64)
    stand_pose = np.array(spec.poses.stand, dtype=np.float64)

    def _drop_to_ground() -> None:
        """Place robot in prone pose and let it fall under gravity — no PD."""
        data.qpos[:] = 0.0
        data.qvel[:] = 0.0
        data.ctrl[:] = 0.0
        data.qpos[2] = _PRONE_Z
        data.qpos[3] = 1.0          # upright quaternion — robot falls from here
        data.qpos[7:] = prone_pose
        mujoco.mj_forward(model, data)
        # Let gravity drop the robot with only gentle holding force (no full PD)
        for _ in range(_DROP_STEPS):
            q  = data.qpos[7:19].astype(np.float64)
            dq = data.qvel[6:18].astype(np.float64)
            # Very light damping only — just prevents wild joint oscillation during fall
            data.ctrl[:] = np.clip(-1.5 * dq, -_MAX_TORQUE, _MAX_TORQUE)
            mujoco.mj_step(model, data)

    def _has_failed(trunk_z: float, roll: float, pitch: float, risen: bool) -> bool:
        sf = spec.safety
        severe_tilt = (abs(roll) > sf.roll_stop_rad or abs(pitch) > sf.pitch_stop_rad)
        collapsed   = risen and trunk_z < _COLLAPSE_Z
        return severe_tilt or collapsed

    # Timing
    cadenza_hz   = cfg.control.rate_hz
    phys_per_cmd = int(1.0 / (model.opt.timestep * cadenza_hz))
    dt_ctrl      = 1.0 / cadenza_hz

    # Episode history for cross-episode learning
    history: list[dict] = []

    # Header
    print(f"  {'ep':>3}  {'step':>5}  {'dq':>7}  {'trunk_z':>7}  "
          f"{'hold':>5}  {'roll':>5}  status")
    print(f"  {'─'*3}  {'─'*5}  {'─'*7}  {'─'*7}  {'─'*5}  {'─'*5}  {'─'*8}")

    # ── Episode loop ──────────────────────────────────────────────────────
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance  = 2.5
        viewer.cam.elevation = -15
        viewer.cam.azimuth   = 30
        viewer.cam.lookat[:] = [0.0, 0.0, 0.20]

        for ep in range(episodes):
            if not viewer.is_running():
                break

            # ── Reset ─────────────────────────────────────────────────────
            _drop_to_ground()
            viewer.sync()

            # Per-episode parameters — ALL come from UserMem (set by agent)
            target_height  = float(usermem.get("target_height",    0.250))
            sk_alpha       = float(usermem.get("skill_alpha",       0.55))
            max_dq         = float(usermem.get("max_dq_per_step",   _DQ_INITIAL))

            # Read ACTUAL settled joint angles — velocity limit starts FROM here
            q_start = data.qpos[7:19].astype(np.float32).copy()
            q_stand = _standup_q12(spec, target_height)

            # ── Synchronized per-joint speed scaling ─────────────────────
            # Each leg settles differently after the drop (body weight
            # distribution means front ≠ rear actual angles).  If all joints
            # move at the same max_dq, the ones with LESS distance to travel
            # reach q_stand first and keep pushing → pitch imbalance → jump.
            # Fix: scale each joint's speed so ALL joints arrive simultaneously.
            # The joint with maximum remaining distance uses full max_dq;
            # all others are proportionally slower.
            dist_to_go    = np.abs(q_stand - q_start)           # (12,)
            max_dist      = float(np.max(dist_to_go)) + 1e-6
            joint_dq_scale = dist_to_go / max_dist              # (12,) in [0, 1]

            # q_commanded: the velocity-limited target that the PD tracks each step
            # Starts at q_start so there is zero initial error, then inches forward
            q_commanded = q_start.copy()

            # Stand-up state machine
            ep_step    = 0
            hold_steps = 0
            max_z      = 0.0
            fell       = False
            success    = False
            risen      = False
            q12_buf: list[np.ndarray] = []

            t_ep_start   = time.monotonic()
            t_step_start = time.monotonic()

            while viewer.is_running():
                elapsed_ep = time.monotonic() - t_ep_start
                if elapsed_ep > _EPISODE_TIMEOUT_S:
                    break

                # ── Sensors ───────────────────────────────────────────────
                q12     = data.qpos[7:19].astype(np.float32)
                dq12    = data.qvel[6:18].astype(np.float64)
                rpy     = _quat_to_rpy(data.qpos[3:7]).astype(np.float32)
                omega   = data.qvel[3:6].astype(np.float32)
                fc      = _foot_contacts(model, data)
                trunk_z = float(data.qpos[2])
                roll    = float(rpy[0])
                pitch   = float(rpy[1])

                if trunk_z > max_z:
                    max_z = trunk_z
                if trunk_z > 0.15:
                    risen = True

                # ── Fail check ────────────────────────────────────────────
                if _has_failed(trunk_z, roll, pitch, risen):
                    fell = True
                    break

                # ── Success check ─────────────────────────────────────────
                if trunk_z >= _STAND_Z:
                    hold_steps += 1
                else:
                    hold_steps = 0
                if hold_steps >= _STAND_HOLD_STEPS:
                    success = True
                    break

                # ── Velocity-limited + synchronized target update ─────────
                # Each joint advances by at most (max_dq × its scale factor).
                # Scale = 1.0 for the joint with most travel, proportionally
                # less for all others → every joint arrives at q_stand together.
                q_delta     = q_stand - q_commanded
                per_joint_dq = max_dq * joint_dq_scale     # (12,) per-joint cap
                q_commanded  = q_commanded + np.sign(q_delta) * np.minimum(
                    np.abs(q_delta), per_joint_dq
                )

                # ── SkillMem reference blend ───────────────────────────────
                q12_target = q_commanded.copy()
                if len(skillmem) > 0:
                    skill = skillmem._skills[0]
                    n_ref = len(skill.waypoints)
                    total_dist = float(np.linalg.norm(q_stand - q_start)) + 1e-6
                    done_dist  = float(np.linalg.norm(q_commanded - q_start))
                    progress   = float(np.clip(done_dist / total_dist, 0.0, 1.0))
                    sk_idx     = int(progress * (n_ref - 1))
                    ref        = skill.waypoints[sk_idx].astype(np.float32)
                    q12_target = (1.0 - sk_alpha) * q_commanded + sk_alpha * ref

                # Clip to joint limits
                jl = spec.joints
                q12_target = np.clip(
                    q12_target.astype(np.float32),
                    [jl.hip_min,   jl.thigh_min, jl.knee_min] * 4,
                    [jl.hip_max,   jl.thigh_max, jl.knee_max] * 4,
                )

                # Balance: tiny roll/pitch nudges only
                q_balanced = balance.step(
                    q12=q12_target, roll=roll, pitch=pitch,
                    trunk_z=trunk_z, fc=np.ones(4, dtype=np.float32), omega=omega,
                )

                # ── Force-limited PD torques ───────────────────────────────
                # CLAMP the position error to ±_MAX_ERROR before multiplying
                # by kp.  This is the key anti-jump mechanism:
                #   max torque/joint = kp × MAX_ERROR = 45 × 0.08 = 3.6 Nm
                # The robot cannot generate more than this per joint regardless
                # of how much lag has accumulated — no sudden energy bursts.
                q_tgt   = q_balanced.astype(np.float64)
                torques = np.zeros(12)
                for j in range(12):
                    pos_err    = float(np.clip(
                        q_tgt[j] - float(q12[j]), -_MAX_ERROR, _MAX_ERROR))
                    torques[j] = _KP * pos_err - _KD * float(dq12[j])
                data.ctrl[:] = np.clip(torques, -_MAX_TORQUE, _MAX_TORQUE)

                # Record trajectory during rise (for SkillMem)
                if risen and len(q12_buf) < 200:
                    q12_buf.append(q12.copy())

                # ── Physics ───────────────────────────────────────────────
                for _ in range(phys_per_cmd):
                    mujoco.mj_step(model, data)

                # ── Camera ────────────────────────────────────────────────
                viewer.cam.lookat[2] = max(trunk_z * 0.8, 0.10)
                viewer.sync()

                # ── Status line (every 25 steps) ──────────────────────────
                if ep_step % 25 == 0:
                    status = ("STANDING!" if hold_steps >= _STAND_HOLD_STEPS // 2
                              else ("rising" if risen else "collapsed"))
                    print(
                        f"  {ep:>3}  {ep_step:>5}  {max_dq:>7.4f}  "
                        f"{trunk_z:>7.3f}  {hold_steps:>5}  {roll:>5.2f}  {status}"
                    )

                # ── Real-time pacing ──────────────────────────────────────
                ep_step += 1
                deadline = t_step_start + ep_step * dt_ctrl
                sleep_t  = deadline - time.monotonic()
                if sleep_t > 0.0:
                    time.sleep(sleep_t)

            # ── Episode end ───────────────────────────────────────────────
            outcome = "SUCCESS" if success else ("FELL" if fell else "TIMEOUT")
            print(f"\n  ep={ep}  {outcome}  max_z={max_z:.3f}m  "
                  f"hold={hold_steps}  dq={max_dq:.4f} ({max_dq*50:.3f} rad/s)\n")

            # Update consecutive streak counters in UserMem
            c_succ = int(usermem.get("consecutive_successes", 0.0))
            c_fail = int(usermem.get("consecutive_falls",     0.0))
            if success:
                c_succ += 1; c_fail = 0
            elif fell:
                c_fail += 1; c_succ = 0
            else:
                c_succ = 0   # timeout resets success streak but not fall counter
            usermem.set("consecutive_successes", float(c_succ))
            usermem.set("consecutive_falls",     float(c_fail))

            # ── SkillMem: record observed trajectory if useful ────────────
            if max_z >= 0.20 and len(q12_buf) >= 20:
                from cadenza_local.locomotion.memory.skillmem import Skill, goal_to_embedding
                existing = [s for s in skillmem._skills if "standup_ep" in s.name]
                if not existing or max_z > max(
                        s.extra.get("max_z", 0.0) for s in existing):
                    new_skill = Skill(
                        name      = f"standup_ep{ep}",
                        goal_emb  = goal_to_embedding(np.zeros(3, np.float32), "stand up"),
                        waypoints = np.stack(q12_buf, axis=0).astype(np.float32),
                        tags      = ["standup"],
                        extra     = {"max_z": max_z, "episode": ep,
                                     "max_dq_per_step": max_dq},
                    )
                    skillmem._skills = [s for s in skillmem._skills
                                        if "standup_ep" not in s.name]
                    skillmem._skills.append(new_skill)
                    skillmem._build_index()
                    print(f"  [SkillMem] saved standup_ep{ep} "
                          f"({len(q12_buf)} waypoints, max_z={max_z:.3f}m)")

            # ── SafetyMem: log roll-over events ──────────────────────────
            if fell:
                from cadenza_local.locomotion.memory.safetymem import SafetyRule
                safetymem._rules.append(SafetyRule(
                    name     = f"standup_rollover_ep{ep}",
                    field    = "imu_rpy", axis = 0,
                    min_val  = -spec.safety.roll_stop_rad,
                    max_val  =  spec.safety.roll_stop_rad,
                    override = {"cmd_vx": 0.0}, priority = 5,
                ))

            # Record full episode data for agent
            ep_result = {
                "ep": ep, "success": success, "fell": fell,
                "max_z": max_z, "hold_steps": hold_steps,
                "target_height": target_height, "max_dq_per_step": max_dq,
            }
            history.append(ep_result)

            # ── AI Agent: blocking decision for next episode ──────────────
            # The agent reads ALL memory compartments and decides the next
            # episode's parameters BEFORE the episode starts.
            if agent is not None:
                decision = agent.decide(
                    ep       = ep,
                    result   = ep_result,
                    history  = history,
                    usermem  = usermem,
                    skillmem = skillmem,
                    safetymem= safetymem,
                )
                if decision is not None:
                    usermem.set("max_dq_per_step", decision["max_dq_per_step"])
                    usermem.set("target_height",   decision["target_height"])
                    usermem.set("perf_phase",       decision["perf_phase"])
                    qa_words = decision["quality_assessment"].split()
                    qa_key   = qa_words[0].lower() if qa_words else ""
                    q_score  = {"poor": 0.1, "acceptable": 0.4,
                                "good": 0.7, "excellent": 1.0}.get(qa_key, 0.3)
                    usermem.set("quality_score", q_score)
                    print(f"  [Agent→ep{ep+1}]")
                    print(f"    quality  : {decision['quality_assessment']}")
                    print(f"    speed    : {decision['max_dq_per_step']:.4f} rad/step "
                          f"({decision['max_dq_per_step']*50:.3f} rad/s)")
                    print(f"    phase    : {decision['perf_phase']}")
                    print(f"    reason   : {decision['reasoning']}\n")
                else:
                    # Agent failed to parse — conservative fallback
                    if fell:
                        usermem.set("max_dq_per_step", max(max_dq * 0.75, _DQ_MIN))
                        print(f"  [Agent fallback] fell → dq {max_dq:.4f}→"
                              f"{max(max_dq*0.75, _DQ_MIN):.4f}")
            else:
                # No agent: pure statistical fallback
                if fell:
                    new_dq = max(max_dq * 0.75, _DQ_MIN)
                    usermem.set("max_dq_per_step", new_dq)
                    print(f"  [Stat] fell → dq {max_dq:.4f}→{new_dq:.4f}")
                elif success and c_succ >= 2:
                    new_dq = min(max_dq * 1.10, _DQ_MAX)
                    usermem.set("max_dq_per_step", new_dq)
                    print(f"  [Stat] {c_succ}× success → dq {max_dq:.4f}→{new_dq:.4f}")

            if not viewer.is_running():
                break

    # ── Wrap up ───────────────────────────────────────────────────────────
    logger.close()
    n_success = sum(1 for r in history if r["success"])
    best_z    = max((r["max_z"] for r in history), default=0.0)
    print(f"\n  {episodes} episodes  |  {n_success} successes  |  best_z={best_z:.3f}m")
    mem_id = _save_snapshot(robot, episodes, skillmem, safetymem, usermem)
    print(f"  Memory ID: {mem_id}  →  --resume {mem_id}")
    print(f"  Log: {logger.path}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Cadenza stand-up task for Unitree Go1/Go2")
    p.add_argument("--robot",    default="go1", choices=["go1", "go2"])
    p.add_argument("--episodes", type=int, default=15,
                   help="Number of episodes to run")
    p.add_argument("--config",   type=Path,
                   default=Path(__file__).parent / "config" / "go1.yaml")
    p.add_argument("--resume",   type=str, default=None, metavar="MEMORY_ID",
                   help="Resume from a saved memory ID (e.g. go1-standup-a3f2)")
    p.add_argument("--local-model", type=str, default="qwen2.5:7b", metavar="MODEL",
                   help="Ollama model for local LLM steering")
    p.add_argument("--groq", action="store_true",
                   help="(reserved — currently uses local LLM only for this task)")
    a = p.parse_args()

    cfg_path = a.config
    if a.robot == "go2" and a.config == p.get_default("config"):
        cfg_path = Path(__file__).parent / "config" / "go2.yaml"
        if not cfg_path.exists():
            cfg_path = a.config

    main(
        robot       = a.robot,
        episodes    = a.episodes,
        cfg_path    = cfg_path,
        resume      = a.resume,
        local_model = a.local_model,
        use_groq    = a.groq,
    )
