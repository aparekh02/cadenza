"""Tier 2 — Action Planner.

Takes a scene description (from Tier 1) + body state + goal and produces
a structured action plan.

Model: Phi-3.5-mini (3.8B, int4) or Gemma-2-2B (int4) via Ollama.
Runs at 2Hz. Falls back to heuristics if model unavailable.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from cadenza.aicore.engine import WorldState


PLANNER_SYSTEM_PROMPT = """You are the action planner for a quadruped robot (Unitree Go1).

You receive:
1. A scene description from the robot's camera
2. The robot's body state (orientation, height, joint angles, foot contact)
3. A high-level goal from the operator

You output a JSON action plan. Each step is an action the robot can execute.

Available actions:
- stand: Stand in place
- walk_forward: Walk forward (params: speed 0.1-2.0, distance_m)
- walk_backward: Walk backward
- trot_forward: Fast trot (params: speed, distance_m)
- crawl_forward: Low, stable crawl for rough terrain (params: speed, distance_m)
- turn_left: Turn left (params: speed)
- turn_right: Turn right (params: speed)
- climb_step: Climb a step or obstacle
- jump: Jump forward or over gap
- sit: Sit down
- side_step_left / side_step_right: Lateral movement

Rules:
- ALWAYS output valid JSON array
- Each step: {"action": "...", "speed": 0.1-2.0, "distance_m": 0.0+, "reasoning": "..."}
- Use crawl_forward on rough terrain, climb_step for stairs
- Reduce speed on slopes and near obstacles
- Maximum 5 steps per plan
- If unsure, output [{"action": "stand", "speed": 1.0, "distance_m": 0, "reasoning": "uncertain, holding position"}]
"""


@dataclass
class ActionStep:
    action: str = "stand"
    speed: float = 1.0
    distance_m: float = 0.0
    reasoning: str = ""


@dataclass
class ActionPlan:
    steps: list[ActionStep] = field(default_factory=list)
    confidence: float = 0.0
    latency_ms: float = 0.0
    model_used: str = ""
    timestamp: float = 0.0

    @property
    def current_step(self) -> ActionStep | None:
        return self.steps[0] if self.steps else None


class ActionPlanner:
    def __init__(self, model: str = "phi3.5:3.8b-mini-instruct-q4_K_M",
                 provider: str = "ollama",
                 base_url: str = "http://localhost:11434",
                 timeout_ms: int = 500):
        self.model = model
        self.provider = provider
        self.base_url = base_url
        self.timeout_ms = timeout_ms
        self._last_plan: ActionPlan | None = None

    def plan(self, scene: str, world_state: WorldState, goal: str) -> ActionPlan:
        start = time.monotonic()
        prompt = self._build_prompt(scene, world_state, goal)

        try:
            response = self._query(prompt)
            steps = self._parse_plan(response)
            confidence = 0.8 if len(steps) > 0 else 0.0
        except Exception:
            steps = self._heuristic_plan(scene, world_state, goal)
            confidence = 0.5

        elapsed_ms = (time.monotonic() - start) * 1000
        plan = ActionPlan(
            steps=steps, confidence=confidence, latency_ms=elapsed_ms,
            model_used=self.model, timestamp=time.time())
        self._last_plan = plan
        return plan

    def _build_prompt(self, scene: str, ws: WorldState, goal: str) -> str:
        import math
        body_desc = (
            f"roll={math.degrees(ws.roll):.1f}deg, pitch={math.degrees(ws.pitch):.1f}deg, "
            f"height={ws.height:.2f}m, foot_contact={ws.foot_contact.tolist()}")
        terrain_names = {0: "flat", 1: "rough", 2: "stairs", 3: "slope", 4: "gap"}
        terrain_desc = (
            f"{terrain_names.get(ws.terrain_class, 'unknown')}, "
            f"roughness={ws.roughness:.2f}, slope={math.degrees(ws.slope):.1f}deg")
        return (
            f"Scene: {scene}\nBody: {body_desc}\nTerrain: {terrain_desc}\n"
            f"Obstacle: {ws.obstacle_distance:.1f}m away, {ws.obstacle_height:.2f}m tall\n"
            f"Goal: {goal}\n\nOutput your action plan as a JSON array:")

    def _query(self, prompt: str) -> str:
        import urllib.request
        import json as json_mod
        url = f"{self.base_url}/api/generate"
        payload = json_mod.dumps({
            "model": self.model, "system": PLANNER_SYSTEM_PROMPT,
            "prompt": prompt, "stream": False, "format": "json",
            "options": {"temperature": 0.1, "num_predict": 300},
        }).encode()
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"}, method="POST")
        timeout_s = self.timeout_ms / 1000.0
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = json_mod.loads(resp.read())
            return data.get("response", "")

    def _parse_plan(self, text: str) -> list[ActionStep]:
        text = text.strip()
        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            raw = json.loads(text[start:end])
        else:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                raw = [json.loads(text[start:end])]
            else:
                return []
        steps = []
        for item in raw[:5]:
            steps.append(ActionStep(
                action=str(item.get("action", "stand")),
                speed=float(item.get("speed", 1.0)),
                distance_m=float(item.get("distance_m", 0.0)),
                reasoning=str(item.get("reasoning", ""))))
        return steps

    def _heuristic_plan(self, scene: str, ws: WorldState, goal: str) -> list[ActionStep]:
        goal_lower = goal.lower()
        steps = []
        if "sit" in goal_lower:
            steps.append(ActionStep("walk_forward", 0.8, 1.0, "approach target"))
            steps.append(ActionStep("sit", 1.0, 0, "sit as requested"))
        elif "walk" in goal_lower or "go" in goal_lower:
            speed = 0.6 if ws.roughness > 0.3 else 1.0
            gait = "crawl_forward" if ws.terrain_class in (1, 3) else "walk_forward"
            steps.append(ActionStep(gait, speed, 2.0, "walk toward goal"))
        elif "turn" in goal_lower:
            direction = "turn_left" if "left" in goal_lower else "turn_right"
            steps.append(ActionStep(direction, 0.6, 0, "turn as requested"))
        elif "explore" in goal_lower:
            steps.append(ActionStep("walk_forward", 0.5, 1.0, "explore forward"))
            steps.append(ActionStep("turn_left", 0.5, 0, "look around"))
            steps.append(ActionStep("walk_forward", 0.5, 1.0, "continue exploring"))
        else:
            steps.append(ActionStep("stand", 1.0, 0, "uncertain goal, holding"))
        return steps

    @property
    def last_plan(self) -> ActionPlan | None:
        return self._last_plan
