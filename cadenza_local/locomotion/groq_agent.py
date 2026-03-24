"""groq_agent.py — Groq-powered locomotion intelligence agent.

Provides three agent roles:
    1. NavigatorAgent   — GPS-like path decisions, gait selection, obstacle avoidance
    2. MemoryAgent      — Summarises episodic memory, extracts safety rules, updates UserMem
    3. BenchmarkAgent   — Evaluates performance, suggests improvements

All agents use llama-3.1-8b-instant via Groq for fast inference.
They are called asynchronously in a background thread so they never block
the 50Hz control loop.

System prompt encodes the complete robot spec so the model knows exactly
what the robot can and cannot do — acting like an expert operator.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    from groq import Groq
    _GROQ_AVAILABLE = True
except ImportError:
    _GROQ_AVAILABLE = False
    Groq = None  # type: ignore


# ──────────────────────────────────────────────────────────────────────────────
#  Agent output types
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class NavigatorDecision:
    """Output of the NavigatorAgent."""
    gait:          str    = "trot"
    speed_limit:   float  = 1.0
    yaw_rate:      float  = 0.0
    body_height:   float  = 0.32
    step_height:   float  = 0.08
    reasoning:     str    = ""
    caution_flags: list[str] = field(default_factory=list)
    timestamp:     float     = field(default_factory=time.monotonic)


@dataclass
class MemorySummary:
    """Output of the MemoryAgent."""
    new_safety_rules:   list[dict] = field(default_factory=list)
    user_pref_updates:  dict       = field(default_factory=dict)
    terrain_updates:    list[dict] = field(default_factory=list)
    episode_summary:    str        = ""
    timestamp:          float      = field(default_factory=time.monotonic)


@dataclass
class BenchmarkAdvice:
    """Output of the BenchmarkAgent."""
    performance_rating:  str   = "unknown"   # "excellent"|"good"|"degraded"|"poor"
    suggested_gait:      str   = "trot"
    suggested_speed:     float = 1.0
    issues:              list[str] = field(default_factory=list)
    suggestions:         list[str] = field(default_factory=list)
    timestamp:           float     = field(default_factory=time.monotonic)


# ──────────────────────────────────────────────────────────────────────────────
#  Robot spec prompt encoder
# ──────────────────────────────────────────────────────────────────────────────

def _robot_system_prompt(spec_summary: str, model: str) -> str:
    """Build the system prompt encoding complete robot knowledge."""
    return f"""You are an expert locomotion controller for the Unitree {model.upper()} quadruped robot.

## Robot Specifications
{spec_summary}

## Joint Limits (CRITICAL — never exceed these)
- Hip (abduction): ±0.863 rad (Go1) / ±1.047 rad (Go2)
- Thigh (flexion): -0.686 to +4.501 rad
- Knee (calf): -2.818 to -0.888 rad (ALWAYS NEGATIVE)
- Max torque: 33.5 Nm (Go1) / 45.0 Nm (Go2)
- Max joint velocity: 21 rad/s (Go1) / 30 rad/s (Go2)

## Gait Library
- stand:       All feet planted. Used for precise tasks or waiting.
- trot:        Diagonal pairs. 3.0 Hz, 65% duty. Standard locomotion ≤1.5 m/s.
- walk:        4-beat, 1.0 Hz, 75% duty. ≤0.5 m/s. Always 3+ feet down.
- crawl:       4-beat, 0.8 Hz, 85% duty. ≤0.3 m/s. Maximum stability.
- pace:        Lateral pairs. 3.5 Hz. ≤1.8 m/s. For firm straight ground.
- stair_crawl: 0.5 Hz, 90% duty, 18cm swing. ≤0.15 m/s. Stairs only.
- bound:       All-in-one-group. 4.0 Hz. High speed straight runs.

## Terrain Rules
- Flat concrete/carpet: trot at full speed (1.5 m/s)
- Grass/gravel: trot, reduce to 0.6-1.0 m/s
- Mud/sand: walk or crawl, ≤0.3-0.4 m/s
- Slope ≤15°: trot, lower body, reduce speed proportionally
- Slope 15-25°: crawl ONLY
- Slope >25°: STOP — exceeds capability
- Stairs: stair_crawl ONLY, ≤20cm step rise, ≤30cm tread depth
- Ice/wet: walk, ≤0.2 m/s, minimal lateral movement

## Safety Hard Limits (EMERGENCY STOP if exceeded)
- |Roll| > 0.60 rad: STOP immediately
- |Pitch| > 0.55 rad: STOP immediately
- Body angular rate > 4 rad/s: STOP immediately
- Feet in contact < 2 during stance: STOP immediately
- CoM height < 0.20 m: FALLEN — initiate recovery

## Control Interface
You receive sensor state and return JSON decisions. Be specific and conservative.
Always prioritize safety over speed. Reason step by step before deciding.
"""


# ──────────────────────────────────────────────────────────────────────────────
#  Base agent
# ──────────────────────────────────────────────────────────────────────────────

class _GroqAgentBase:
    MODEL = "llama-3.1-8b-instant"
    TEMPERATURE = 0.1   # low for deterministic control decisions

    def __init__(self, api_key: str | None, spec_summary: str, robot_model: str):
        if not _GROQ_AVAILABLE:
            raise ImportError("groq package not installed: pip install groq")
        key = api_key or os.environ.get("GROQ_API_KEY", "")
        if not key:
            raise ValueError("GROQ_API_KEY not set. Set env var or pass api_key=")
        self._client = Groq(api_key=key)
        self._system = _robot_system_prompt(spec_summary, robot_model)
        self._history: list[dict] = []

    def _call(self, user_msg: str, max_tokens: int = 512) -> str:
        """Send a message and return the assistant response text."""
        messages = [
            {"role": "system", "content": self._system},
            *self._history[-6:],    # last 3 turns of context
            {"role": "user",   "content": user_msg},
        ]
        resp = self._client.chat.completions.create(
            model       = self.MODEL,
            messages    = messages,
            temperature = self.TEMPERATURE,
            max_tokens  = max_tokens,
            response_format={"type": "json_object"},
        )
        text = resp.choices[0].message.content or "{}"
        # Update rolling history
        self._history.append({"role": "user",      "content": user_msg})
        self._history.append({"role": "assistant", "content": text})
        return text

    @staticmethod
    def _safe_json(text: str) -> dict:
        try:
            return json.loads(text)
        except Exception:
            return {}


# ──────────────────────────────────────────────────────────────────────────────
#  Navigator Agent — real-time GPS-like guidance
# ──────────────────────────────────────────────────────────────────────────────

class NavigatorAgent(_GroqAgentBase):
    """Decides gait, speed, and body configuration based on current sensor state.

    Called every N seconds (not every control step) from a background thread.
    """

    def decide(
        self,
        terrain_label:  str,
        terrain_conf:   float,
        slope_deg:      float,
        slip_risk:      float,
        roll_rad:       float,
        pitch_rad:      float,
        n_contacts:     int,
        cmd_vx:         float,
        current_gait:   str,
        task_text:      str = "",
        benchmark_ratio: float = 1.0,
    ) -> NavigatorDecision:
        """Make a navigation decision given current state.

        Args:
            terrain_label:   classified terrain
            terrain_conf:    classifier confidence [0,1]
            slope_deg:       estimated slope (degrees)
            slip_risk:       estimated slip probability [0,1]
            roll_rad:        current body roll (rad)
            pitch_rad:       current body pitch (rad)
            n_contacts:      feet currently in contact
            cmd_vx:          requested forward speed (m/s)
            current_gait:    gait currently executing
            task_text:       operator task description (e.g. "navigate to dock")
            benchmark_ratio: actual/expected speed ratio

        Returns:
            NavigatorDecision — safe to apply immediately
        """
        msg = json.dumps({
            "task": task_text or "general locomotion",
            "terrain": terrain_label,
            "terrain_confidence": round(terrain_conf, 2),
            "slope_deg": round(slope_deg, 1),
            "slip_risk": round(slip_risk, 2),
            "roll_rad": round(roll_rad, 3),
            "pitch_rad": round(pitch_rad, 3),
            "feet_in_contact": n_contacts,
            "requested_speed_ms": round(cmd_vx, 2),
            "current_gait": current_gait,
            "performance_ratio": round(benchmark_ratio, 2),
            "instruction": (
                "Decide the safest gait and speed. "
                "Return JSON: {gait, speed_limit, yaw_rate_limit, body_height, "
                "step_height, reasoning, caution_flags}"
            ),
        })
        raw = self._call(msg, max_tokens=400)
        d   = self._safe_json(raw)
        return NavigatorDecision(
            gait          = str(d.get("gait", current_gait)),
            speed_limit   = float(d.get("speed_limit", cmd_vx)),
            yaw_rate      = float(d.get("yaw_rate_limit", 0.8)),
            body_height   = float(d.get("body_height", 0.32)),
            step_height   = float(d.get("step_height", 0.08)),
            reasoning     = str(d.get("reasoning", "")),
            caution_flags = list(d.get("caution_flags", [])),
        )


# ──────────────────────────────────────────────────────────────────────────────
#  Memory Agent — episodic memory maintenance
# ──────────────────────────────────────────────────────────────────────────────

class MemoryAgent(_GroqAgentBase):
    """Analyses episode logs and updates memory compartments intelligently."""

    def update_memory(
        self,
        episode_steps:  list[dict],   # recent JSONL log entries
        success:        bool,
        terrain_seen:   list[str],
        failure_reason: str = "",
    ) -> MemorySummary:
        """Analyse an episode and extract memory updates.

        Args:
            episode_steps:  list of step dicts from ExperienceLogger
            success:        whether the episode succeeded
            terrain_seen:   list of terrain labels encountered
            failure_reason: description of failure if any

        Returns:
            MemorySummary with new rules and preference updates
        """
        # Summarise key stats before sending
        if episode_steps:
            rolls  = [s["frame"]["imu_rpy"][0] for s in episode_steps if "frame" in s]
            speeds = [abs(s["frame"]["cmd_vel"][0]) for s in episode_steps if "frame" in s]
            avg_roll  = float(np.mean(np.abs(rolls)))  if rolls  else 0.0
            avg_speed = float(np.mean(speeds)) if speeds else 0.0
        else:
            avg_roll = avg_speed = 0.0

        msg = json.dumps({
            "episode_success": success,
            "failure_reason": failure_reason,
            "terrains_encountered": terrain_seen,
            "steps": len(episode_steps),
            "avg_roll_rad": round(avg_roll, 3),
            "avg_speed_ms": round(avg_speed, 2),
            "instruction": (
                "Analyse this episode. Extract: "
                "(1) new safety rules if the episode failed or showed instability, "
                "(2) user preference updates (max_speed, gait_override), "
                "(3) terrain parameter updates. "
                "Return JSON: {new_safety_rules: [...], user_pref_updates: {...}, "
                "terrain_updates: [...], episode_summary: str}"
            ),
        })
        raw = self._call(msg, max_tokens=600)
        d   = self._safe_json(raw)
        return MemorySummary(
            new_safety_rules  = list(d.get("new_safety_rules",  [])),
            user_pref_updates = dict(d.get("user_pref_updates", {})),
            terrain_updates   = list(d.get("terrain_updates",   [])),
            episode_summary   = str(d.get("episode_summary",    "")),
        )


# ──────────────────────────────────────────────────────────────────────────────
#  Benchmark Agent — performance evaluation and coaching
# ──────────────────────────────────────────────────────────────────────────────

class BenchmarkAgent(_GroqAgentBase):
    """Evaluates robot performance and suggests improvements."""

    def evaluate(
        self,
        terrain_label:   str,
        actual_speed:    float,
        expected_speed:  float,
        roll_rms:        float,
        slip_risk:       float,
        n_falls:         int,
        gait_used:       str,
        n_steps:         int,
    ) -> BenchmarkAdvice:
        """Evaluate current performance and suggest improvements.

        Args:
            terrain_label:  current terrain
            actual_speed:   measured speed (m/s)
            expected_speed: benchmark speed for this terrain (m/s)
            roll_rms:       RMS roll angle (rad) — stability proxy
            slip_risk:      current slip risk estimate [0,1]
            n_falls:        number of fall events in recent window
            gait_used:      current gait name
            n_steps:        steps since last evaluation

        Returns:
            BenchmarkAdvice with rating and suggestions
        """
        ratio = actual_speed / expected_speed if expected_speed > 0 else 0.0
        msg = json.dumps({
            "terrain": terrain_label,
            "actual_speed_ms":    round(actual_speed,  2),
            "expected_speed_ms":  round(expected_speed, 2),
            "speed_ratio":        round(ratio, 2),
            "roll_rms_rad":       round(roll_rms,  3),
            "slip_risk":          round(slip_risk, 2),
            "fall_events":        n_falls,
            "gait":               gait_used,
            "steps_evaluated":    n_steps,
            "instruction": (
                "Rate performance and suggest specific improvements. "
                "Return JSON: {performance_rating, suggested_gait, suggested_speed, "
                "issues: [...], suggestions: [...]}"
            ),
        })
        raw = self._call(msg, max_tokens=500)
        d   = self._safe_json(raw)
        return BenchmarkAdvice(
            performance_rating = str(d.get("performance_rating", "unknown")),
            suggested_gait     = str(d.get("suggested_gait",  gait_used)),
            suggested_speed    = float(d.get("suggested_speed", actual_speed)),
            issues             = list(d.get("issues",       [])),
            suggestions        = list(d.get("suggestions",  [])),
        )


# ──────────────────────────────────────────────────────────────────────────────
#  GroqAdvisor — orchestrates all three agents in background threads
# ──────────────────────────────────────────────────────────────────────────────

class GroqAdvisor:
    """Non-blocking wrapper around all three Groq agents.

    Runs agent calls in background threads. The control loop reads the
    latest results via `.latest_nav`, `.latest_memory`, `.latest_bench`.

    Args:
        robot_model:    "go1" or "go2"
        api_key:        Groq API key (or None → reads GROQ_API_KEY env var)
        nav_interval_s: how often to call NavigatorAgent (seconds)
        bench_interval_s: how often to call BenchmarkAgent (seconds)
    """

    def __init__(
        self,
        robot_model:      str   = "go1",
        api_key:          str | None = None,
        nav_interval_s:   float = 3.0,
        bench_interval_s: float = 10.0,
    ):
        from cadenza_local.locomotion.robot_spec import get_spec
        spec = get_spec(robot_model)
        spec_summary = spec.summary()

        self._nav_interval   = nav_interval_s
        self._bench_interval = bench_interval_s
        self._running        = False

        if not _GROQ_AVAILABLE:
            print("[GroqAdvisor] groq package not available — advisor disabled")
            self._enabled = False
            return

        if not (api_key or os.environ.get("GROQ_API_KEY", "")):
            print("[GroqAdvisor] GROQ_API_KEY not set — advisor disabled")
            self._enabled = False
            return

        self._enabled = True
        self._nav   = NavigatorAgent(api_key, spec_summary, robot_model)
        self._mem   = MemoryAgent(api_key,   spec_summary, robot_model)
        self._bench = BenchmarkAgent(api_key, spec_summary, robot_model)

        # Latest results (updated by background threads, read by control loop)
        self.latest_nav:    NavigatorDecision | None = None
        self.latest_memory: MemorySummary    | None = None
        self.latest_bench:  BenchmarkAdvice  | None = None

        # State queue for background thread
        self._nav_q:   queue.Queue = queue.Queue(maxsize=1)
        self._mem_q:   queue.Queue = queue.Queue(maxsize=1)
        self._bench_q: queue.Queue = queue.Queue(maxsize=1)

        # Episode log buffer
        self._episode_steps:   list[dict] = []
        self._terrains_seen:   list[str]  = []
        self._fall_count:      int        = 0
        self._last_nav_t:      float      = 0.0
        self._last_bench_t:    float      = 0.0

        self._lock = threading.Lock()

    # ── Control loop interface (called at 50Hz — must be fast) ─────────────

    def update_state(
        self,
        terrain_label:  str,
        terrain_conf:   float,
        slope_deg:      float,
        slip_risk:      float,
        roll_rad:       float,
        pitch_rad:      float,
        n_contacts:     int,
        cmd_vx:         float,
        current_gait:   str,
        actual_speed:   float,
        task_text:      str = "",
        step_dict:      dict | None = None,   # raw log dict for memory
    ) -> NavigatorDecision | None:
        """Feed current state; returns latest nav decision (non-blocking).

        Kicks off background agent calls when intervals expire.
        """
        if not self._enabled:
            return None

        now = time.monotonic()

        # Buffer episode step
        if step_dict:
            with self._lock:
                self._episode_steps.append(step_dict)
                if terrain_label not in self._terrains_seen:
                    self._terrains_seen.append(terrain_label)

        # Kick navigator if interval elapsed and queue is empty
        if now - self._last_nav_t > self._nav_interval:
            self._last_nav_t = now
            bench_ratio = 1.0
            if self.latest_bench:
                bench_ratio = self.latest_bench.suggested_speed / (cmd_vx + 0.01)
            kwargs = dict(
                terrain_label=terrain_label, terrain_conf=terrain_conf,
                slope_deg=slope_deg, slip_risk=slip_risk, roll_rad=roll_rad,
                pitch_rad=pitch_rad, n_contacts=n_contacts, cmd_vx=cmd_vx,
                current_gait=current_gait, task_text=task_text,
                benchmark_ratio=bench_ratio,
            )
            t = threading.Thread(target=self._run_nav, kwargs=kwargs, daemon=True)
            t.start()

        # Kick benchmark if interval elapsed
        if now - self._last_bench_t > self._bench_interval and self.latest_nav:
            self._last_bench_t = now
            t = threading.Thread(
                target=self._run_bench,
                kwargs=dict(
                    terrain_label=terrain_label, actual_speed=actual_speed,
                    expected_speed=self.latest_nav.speed_limit,
                    roll_rms=abs(roll_rad), slip_risk=slip_risk,
                    n_falls=self._fall_count, gait_used=current_gait,
                    n_steps=len(self._episode_steps),
                ),
                daemon=True,
            )
            t.start()

        return self.latest_nav

    def notify_episode_end(self, success: bool, failure_reason: str = "") -> None:
        """Call at end of episode — triggers memory agent in background."""
        if not self._enabled:
            return
        with self._lock:
            steps  = list(self._episode_steps)
            terrains = list(self._terrains_seen)
            self._episode_steps.clear()
            self._terrains_seen.clear()

        t = threading.Thread(
            target=self._run_memory,
            kwargs=dict(
                episode_steps=steps, success=success,
                terrain_seen=terrains, failure_reason=failure_reason,
            ),
            daemon=True,
        )
        t.start()

    def notify_fall(self) -> None:
        with self._lock:
            self._fall_count += 1

    # ── Background thread runners ──────────────────────────────────────────

    def _run_nav(self, **kwargs) -> None:
        try:
            result = self._nav.decide(**kwargs)
            self.latest_nav = result
        except Exception as e:
            print(f"[GroqAdvisor] NavigatorAgent error: {e}")

    def _run_memory(self, **kwargs) -> None:
        try:
            result = self._mem.update_memory(**kwargs)
            self.latest_memory = result
        except Exception as e:
            print(f"[GroqAdvisor] MemoryAgent error: {e}")

    def _run_bench(self, **kwargs) -> None:
        try:
            result = self._bench.evaluate(**kwargs)
            self.latest_bench = result
            self._fall_count  = 0   # reset after evaluation
        except Exception as e:
            print(f"[GroqAdvisor] BenchmarkAgent error: {e}")

    @property
    def enabled(self) -> bool:
        return self._enabled


# ──────────────────────────────────────────────────────────────────────────────
#  EpisodeSteeringAgent — the actual cross-episode steering brain
# ──────────────────────────────────────────────────────────────────────────────

class EpisodeSteeringAgent(_GroqAgentBase):
    """Analyzes episode outcomes and suggests concrete parameter adjustments.

    This is the Groq steering layer — it replaces RL by reasoning about
    failures and outputting actionable parameter changes for the next episode.

    Called once per episode (after the robot finishes), not every step.
    Response is a JSON dict with specific float values the EpisodeGuide applies.
    """

    SYSTEM_SUFFIX = """
You are the steering brain for a quadruped locomotion memory system.
Your job: analyze each episode and output SPECIFIC parameter values for the next run.

TASK: The robot starts at x=0, must reach a GOAL at x=8m.
      At x=4m there is a HURDLE (10cm tall) it must step over with high foot clearance.

CURRICULUM (follow strictly):
  Stage 0 — Walk: robot must reach x=3m reliably before attempting hurdle.
  Stage 1 — Approach: robot must reach x=5m (past hurdle) using trot only.
  Stage 2 — Full: robot reaches x=8m goal.

GAIT RULES:
  - Only use "trot" or "walk". NEVER suggest "bound" or "crawl" — they cause falls.
  - step_height 0.08–0.12 = normal trot, 0.14–0.18 = for clearing the hurdle.
  - Slow down (approach_speed ≤ 0.6) if the robot keeps falling before x=3m.
  - The robot CAN walk at 0.4–0.8 m/s stably with trot gait.

OUTPUT FORMAT (JSON, all fields required):
{
  "approach_speed": <float 0.2–0.8>,    // m/s for approach phase
  "sprint_speed":   <float 0.4–1.0>,    // m/s for sprint phase
  "step_height":    <float 0.08–0.20>,  // foot clearance at hurdle
  "body_height_factor": <float 0.75–0.95>,  // fraction of standing height
  "gait_to_use":    "trot",             // ALWAYS trot
  "reasoning":      "<one sentence>"    // why these values
}
"""

    def __init__(self, api_key: str | None, spec_summary: str, robot_model: str):
        super().__init__(api_key, spec_summary, robot_model)
        self._system = self._system + self.SYSTEM_SUFFIX

    def suggest(
        self,
        episode:          int,
        distance_reached: float,
        fell:             bool,
        fall_x:           float | None,
        fall_speed:       float | None,
        max_phase:        str,
        jump_cleared:     bool,
        approach_pace:    float,
        prev_params:      dict,
        history_summary:  str,
        max_speed:        float,
    ) -> dict | None:
        """Return suggested params dict, or None on error."""
        fall_str = (f"fell at x={fall_x:.1f}m going {fall_speed:.2f} m/s"
                    if fell and fall_x is not None else "did not fall")
        msg = (
            f"Episode {episode} result:\n"
            f"  distance_reached={distance_reached:.2f}m  hurdle_cleared={jump_cleared}\n"
            f"  furthest_phase={max_phase}  {fall_str}\n"
            f"  approach_pace={approach_pace:.2f} m/s\n"
            f"  params_used: approach={prev_params.get('approach_speed',0):.2f} "
            f"sprint={prev_params.get('sprint_speed',0):.2f} "
            f"step_height={prev_params.get('step_height',0):.2f}\n"
            f"  robot_max_speed={max_speed:.2f} m/s\n"
            f"  history: {history_summary}\n"
            f"Suggest parameters for episode {episode+1}."
        )
        try:
            raw = self._call(msg, max_tokens=256)
            import json as _json
            data = _json.loads(raw)
            # Clamp to safe ranges
            return {
                "approach_speed":     float(min(max(data.get("approach_speed", 0.5), 0.2), 0.8)),
                "sprint_speed":       float(min(max(data.get("sprint_speed", 0.7), 0.3), 1.0)),
                "step_height":        float(min(max(data.get("step_height", 0.10), 0.06), 0.22)),
                "body_height_factor": float(min(max(data.get("body_height_factor", 0.82), 0.70), 0.95)),
                "gait_to_use":        "trot",  # always trot regardless of suggestion
                "reasoning":          str(data.get("reasoning", "")),
            }
        except Exception as e:
            return None


# ──────────────────────────────────────────────────────────────────────────────
#  GroqMemoryLayer — memory organisation only, no per-step control
# ──────────────────────────────────────────────────────────────────────────────

class GroqMemoryLayer:
    """Periodic Groq memory organisation, completely separate from the robot
    control loop.

    The local analytical controller (GaitEngine + TerrainClassifier) runs at
    full rate with no API calls.  This class fires MemoryAgent and
    BenchmarkAgent on a configurable timer to keep memory up to date.

    Args:
        robot_model:       "go1" or "go2"
        memory_interval_s: seconds between MemoryAgent calls (default 30).
                           Set to 0 to disable periodic calls (only fires
                           on episode end / fall).
        api_key:           Groq API key (or None → reads GROQ_API_KEY env var)
    """

    def __init__(
        self,
        robot_model:       str         = "go1",
        memory_interval_s: float       = 30.0,
        api_key:           str | None  = None,
    ):
        self._interval = memory_interval_s   # 0 = episode-end only
        self._enabled  = False

        if not _GROQ_AVAILABLE:
            print("[GroqMemory] groq package not installed — disabled")
            return
        key = api_key or os.environ.get("GROQ_API_KEY", "")
        if not key:
            print("[GroqMemory] GROQ_API_KEY not set — disabled")
            return

        from cadenza_local.locomotion.robot_spec import get_spec
        spec_sum = get_spec(robot_model).summary()
        self._mem   = MemoryAgent(key,   spec_sum, robot_model)
        self._bench = BenchmarkAgent(key, spec_sum, robot_model)

        self._enabled    = True
        self._lock       = threading.Lock()
        self._steps:     list[dict] = []
        self._terrains:  list[str]  = []
        self._falls:     int        = 0
        self._last_mem_t:   float   = 0.0
        self._last_bench_t: float   = 0.0
        self._last_bench_loco       = None   # last LocoCommand for bench

        self.latest_summary: "MemorySummary | None"  = None
        self.latest_bench:   "BenchmarkAdvice | None" = None

    # ── Called every control step (cheap — just buffers data + checks timer) ─

    def push(self, frame, loco) -> None:
        """Buffer one step.  Call after every ctrl.step()."""
        if not self._enabled:
            return

        with self._lock:
            self._steps.append({
                "frame": {
                    "imu_rpy":      frame.imu_rpy.tolist(),
                    "foot_contact": frame.foot_contact.tolist(),
                    "cmd_vel":      frame.cmd_vel.tolist(),
                },
                "gait":    loco.gait,
                "terrain": loco.terrain,
                "vx":      loco.cmd_vx,
            })
            if loco.terrain not in self._terrains:
                self._terrains.append(loco.terrain)
            self._last_bench_loco = loco

        now = time.monotonic()
        # Periodic memory org
        if self._interval > 0 and now - self._last_mem_t > self._interval:
            self._last_mem_t = now
            self._fire_memory(success=True, reason="periodic checkpoint")
        # Periodic benchmark (2× interval)
        bench_iv = max(self._interval * 2.0, 60.0)
        if self._interval > 0 and now - self._last_bench_t > bench_iv:
            self._last_bench_t = now
            self._fire_bench()

    # ── Episode / fall events ─────────────────────────────────────────────────

    def on_fall(self) -> None:
        if not self._enabled:
            return
        with self._lock:
            self._falls += 1
        self._fire_memory(success=False, reason="fall event")

    def on_episode_end(self, success: bool) -> None:
        if not self._enabled:
            return
        self._fire_memory(success=success, reason="" if success else "episode failed")
        self._fire_bench()

    # ── Background runners ────────────────────────────────────────────────────

    def _fire_memory(self, success: bool, reason: str = "") -> None:
        with self._lock:
            steps    = list(self._steps)
            terrains = list(self._terrains)
            self._steps.clear()
            self._terrains.clear()
        threading.Thread(
            target=self._run_mem,
            args=(steps, success, terrains, reason),
            daemon=True,
        ).start()

    def _fire_bench(self) -> None:
        with self._lock:
            loco = self._last_bench_loco
            falls = self._falls
            self._falls = 0
        if loco is None:
            return
        threading.Thread(
            target=self._run_bench,
            kwargs=dict(
                terrain_label  = loco.terrain,
                actual_speed   = loco.cmd_vx,
                expected_speed = max(loco.cmd_vx, 0.5),
                roll_rms       = 0.0,
                slip_risk      = loco.slip_risk,
                n_falls        = falls,
                gait_used      = loco.gait,
                n_steps        = 0,
            ),
            daemon=True,
        ).start()

    def _run_mem(self, steps, success, terrains, reason):
        try:
            result = self._mem.update_memory(steps, success, terrains, reason)
            self.latest_summary = result
            if result.episode_summary:
                print(f"\n  [Groq Memory] {result.episode_summary[:90]}\n")
        except Exception as e:
            print(f"  [Groq Memory] error: {e}")

    def _run_bench(self, **kwargs):
        try:
            result = self._bench.evaluate(**kwargs)
            self.latest_bench = result
            if result.suggestions:
                print(f"\n  [Groq Bench]  {result.performance_rating}"
                      f" — {result.suggestions[0][:80]}\n")
        except Exception as e:
            print(f"  [Groq Bench] error: {e}")

    @property
    def enabled(self) -> bool:
        return self._enabled
