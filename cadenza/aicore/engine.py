"""Behavior Engine — Python implementation of the AICore decision loop.

Same priority cascade as the C++ reasoning_engine.cpp:
    safety > terrain > goal > strategic (planner/SLM)

Can optionally use Tier 0 (vision), Tier 2 (planner), Tier 3 (SLM).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class WorldState:
    timestamp: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    angular_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    linear_accel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    height: float = 0.28
    joint_q: np.ndarray = field(default_factory=lambda: np.zeros(12))
    joint_dq: np.ndarray = field(default_factory=lambda: np.zeros(12))
    foot_contact: np.ndarray = field(default_factory=lambda: np.ones(4))
    slope: float = 0.0
    roughness: float = 0.0
    friction: float = 0.6
    obstacle_distance: float = 100.0
    obstacle_height: float = 0.0
    terrain_class: int = 0
    goal_text: str = ""
    target_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    urgency: float = 0.5
    goal_type: int = 3
    confidence: float = 1.0


@dataclass
class ActionDecision:
    action: str = "stand"
    speed: float = 1.0
    heading_rad: float = 0.0
    distance_m: float = 0.0
    confidence: float = 1.0
    priority: int = 0
    reasoning: str = ""
    layer: str = ""


class BehaviorEngine:
    def __init__(self, robot: str = "go1", slm=None, vision=None,
                 planner=None, hardware: str = "dev"):
        self.robot = robot
        self.slm = slm
        self.vision = vision
        self.planner = planner
        self.hardware = hardware
        self._goal = WorldState()
        self._decision_count = 0
        self._history: list[ActionDecision] = []
        self._last_scene: str = ""
        self._last_plan: list = []
        self._plan_index: int = 0

    def set_goal(self, text: str = "", target: np.ndarray | None = None,
                 urgency: float = 0.5, goal_type: int = 0):
        self._goal.goal_text = text
        if target is not None:
            self._goal.target_position = target
        self._goal.urgency = urgency
        self._goal.goal_type = goal_type

    def observe(self, qpos: np.ndarray, qvel: np.ndarray,
                sensor_data: dict | None = None) -> WorldState:
        ws = WorldState()
        ws.timestamp = time.time()
        if len(qpos) >= 7:
            ws.height = qpos[2]
            w, x, y, z = qpos[3:7]
            ws.roll = math.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
            ws.pitch = math.asin(max(-1, min(1, 2*(w*y - z*x))))
            ws.yaw = math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        if len(qpos) >= 19:
            ws.joint_q = qpos[7:19].copy()
        if len(qvel) >= 18:
            ws.angular_vel = qvel[0:3].copy()
            ws.linear_accel = qvel[3:6].copy()
            ws.joint_dq = qvel[6:18].copy()
        if sensor_data:
            ws.slope = sensor_data.get("slope", 0.0)
            ws.roughness = sensor_data.get("roughness", 0.0)
            ws.friction = sensor_data.get("friction", 0.6)
            ws.obstacle_distance = sensor_data.get("obstacle_distance", 100.0)
            ws.obstacle_height = sensor_data.get("obstacle_height", 0.0)
            ws.terrain_class = sensor_data.get("terrain_class", 0)
        ws.goal_text = self._goal.goal_text
        ws.target_position = self._goal.target_position.copy()
        ws.urgency = self._goal.urgency
        ws.goal_type = self._goal.goal_type
        ws.confidence = self._estimate_confidence(ws)
        return ws

    def observe_with_camera(self, qpos: np.ndarray, qvel: np.ndarray,
                            camera_frame: np.ndarray | None = None,
                            sensor_data: dict | None = None) -> WorldState:
        ws = self.observe(qpos, qvel, sensor_data)
        if camera_frame is not None and self.vision is not None:
            self.vision.encode(camera_frame)
            if self.slm and self.slm.is_available():
                try:
                    result = self.slm._query(
                        f"Describe what a quadruped robot sees. "
                        f"Terrain roughness={ws.roughness:.1f}, slope={ws.slope:.1f}rad. "
                        f"Obstacle at {ws.obstacle_distance:.1f}m. Brief (1-2 sentences).")
                    self._last_scene = result.strip()[:200]
                except Exception:
                    self._last_scene = "scene description failed"
            else:
                self._last_scene = "camera frame captured, no scene model"
        return ws

    def decide(self, ws: WorldState) -> ActionDecision:
        decision = ActionDecision()
        if self._check_safety(ws, decision):
            decision.layer = "SAFETY"
        elif self._check_terrain(ws, decision):
            decision.layer = "REACTIVE"
        elif self._check_goal(ws, decision):
            decision.layer = "TACTICAL"
        else:
            self._strategic(ws, decision)
            decision.layer = "STRATEGIC"
        self._decision_count += 1
        self._history.append(decision)
        if len(self._history) > 100:
            self._history.pop(0)
        return decision

    def _check_safety(self, ws: WorldState, d: ActionDecision) -> bool:
        tilt = math.sqrt(ws.roll**2 + ws.pitch**2)
        if tilt > 0.45:
            d.action = "stand"; d.speed = 0.5; d.priority = 3; d.confidence = 0.95
            d.reasoning = f"Critical tilt {math.degrees(tilt):.0f} deg, stabilizing"
            return True
        if 0 < ws.obstacle_distance < 0.3:
            d.action = "stand"; d.speed = 1.0; d.priority = 3; d.confidence = 0.9
            d.reasoning = f"Obstacle at {ws.obstacle_distance:.2f}m, halting"
            return True
        contacts = int(np.sum(ws.foot_contact))
        if contacts <= 1 and ws.height > 0.1:
            d.action = "stand"; d.speed = 0.3; d.priority = 3; d.confidence = 0.85
            d.reasoning = f"Only {contacts} feet in contact, stabilizing"
            return True
        return False

    def _check_terrain(self, ws: WorldState, d: ActionDecision) -> bool:
        if ws.goal_type == 3:
            return False
        terrain_gaits = {
            0: ("walk_forward", 1.0, "flat ground"),
            1: ("crawl_forward", 0.6, f"rough surface (roughness={ws.roughness:.2f})"),
            2: ("climb_step", 0.4, f"stairs (height={ws.obstacle_height:.2f}m)"),
            3: ("crawl_forward" if ws.slope > 0 else "walk_forward",
                0.5 if ws.slope > 0 else 0.7,
                f"slope {math.degrees(ws.slope):.0f} deg"),
            4: ("jump", 1.5, "gap detected"),
        }
        gait, speed, reason = terrain_gaits.get(ws.terrain_class, (None, 1.0, ""))
        if gait is None:
            return False
        if ws.terrain_class == 0 and ws.roughness < 0.1:
            return False
        d.action = gait; d.speed = speed; d.priority = 1
        d.confidence = ws.confidence * 0.8
        d.reasoning = f"Terrain: {reason}"
        return True

    def _check_goal(self, ws: WorldState, d: ActionDecision) -> bool:
        if ws.goal_type == 3:
            return False
        tx, ty = ws.target_position[0], ws.target_position[1]
        if tx == 0 and ty == 0 and ws.goal_text:
            return False
        dist = math.sqrt(tx**2 + ty**2)
        if dist < 0.01:
            return False
        if dist < 0.2:
            d.action = "stand"; d.speed = 1.0; d.priority = 1; d.confidence = 0.9
            d.reasoning = f"Arrived at target ({dist:.2f}m away)"
            return True
        heading = math.atan2(ty, tx)
        if abs(heading) > 0.3:
            d.action = "turn_left" if heading > 0 else "turn_right"
            d.speed = 0.6; d.heading_rad = heading; d.priority = 1; d.confidence = 0.8
            d.reasoning = f"Turning {math.degrees(heading):.0f} deg toward target ({dist:.1f}m)"
            return True
        d.action = "walk_forward"; d.speed = min(1.0, ws.urgency + 0.5)
        d.distance_m = dist; d.priority = 1; d.confidence = 0.8
        d.reasoning = f"Walking toward target ({dist:.1f}m away)"
        return True

    def _strategic(self, ws: WorldState, d: ActionDecision):
        if ws.goal_text and self.planner:
            if self._last_plan and self._plan_index < len(self._last_plan):
                step = self._last_plan[self._plan_index]
                d.action = step.action; d.speed = step.speed
                d.distance_m = step.distance_m; d.confidence = 0.75
                d.reasoning = f"PLANNER step {self._plan_index+1}/{len(self._last_plan)}: {step.reasoning}"
                self._plan_index += 1
                return
            scene = self._last_scene or "no camera data"
            plan = self.planner.plan(scene, ws, ws.goal_text)
            if plan.steps:
                self._last_plan = plan.steps; self._plan_index = 0
                step = self._last_plan[0]
                d.action = step.action; d.speed = step.speed
                d.distance_m = step.distance_m; d.confidence = plan.confidence
                d.reasoning = f"PLANNER ({plan.model_used}, {plan.latency_ms:.0f}ms): {step.reasoning}"
                self._plan_index = 1
                return
        if ws.goal_text and self.slm:
            slm_decision = self.slm.reason(ws)
            if slm_decision:
                d.action = slm_decision.get("action", "stand")
                d.speed = slm_decision.get("speed", 0.5)
                d.confidence = slm_decision.get("confidence", 0.6)
                d.reasoning = f"SLM: {slm_decision.get('reasoning', 'model decision')}"
                return
        if ws.goal_type == 2:
            d.action = "walk_forward"; d.speed = 0.5; d.confidence = 0.5
            d.reasoning = "Exploring"
        else:
            d.action = "stand"; d.speed = 1.0; d.confidence = 1.0
            d.reasoning = "No goal, standing by"

    def _estimate_confidence(self, ws: WorldState) -> float:
        conf = 1.0
        tilt = math.sqrt(ws.roll**2 + ws.pitch**2)
        if tilt > 0.3: conf *= 0.6
        if tilt > 0.5: conf *= 0.2
        contacts = int(np.sum(ws.foot_contact))
        if contacts < 3: conf *= 0.7
        return conf

    @property
    def history(self) -> list[ActionDecision]:
        return self._history

    @property
    def decision_count(self) -> int:
        return self._decision_count
