"""Gym adapter — translates named ActionCalls into raw motor commands by
driving the Cadenza Sim, then reads back observations.

This is the boundary between the symbolic action vocabulary the world model
speaks and the joint-space MuJoCo physics. It's a thin gym-style env:

    obs = gym.reset()
    obs, info = gym.step(action_call)
    gym.close()
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from cadenza.actions.library import ActionCall


# ── Observation ───────────────────────────────────────────────────────────────

@dataclass
class Observation:
    """Snapshot of the robot's state after one step. Passed to the bridge."""
    pos: np.ndarray                  # world-frame xyz
    rpy: np.ndarray                  # roll, pitch, yaw (rad)
    body_height: float
    qpos: np.ndarray
    qvel: np.ndarray
    foot_contacts: tuple[bool, ...]
    terrain_ahead: dict[str, Any] = field(default_factory=dict)
    obstacles_ahead: dict[str, Any] = field(default_factory=dict)
    camera: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "pos": self.pos.tolist() if isinstance(self.pos, np.ndarray) else self.pos,
            "rpy": self.rpy.tolist() if isinstance(self.rpy, np.ndarray) else self.rpy,
            "body_height": self.body_height,
            "foot_contacts": list(self.foot_contacts),
            "terrain_ahead": self.terrain_ahead,
            "obstacles_ahead": self.obstacles_ahead,
            "metadata": self.metadata,
            # raw arrays kept as-is for adapters that need them
            "qpos": self.qpos,
            "qvel": self.qvel,
        }
        if self.camera is not None:
            d["camera"] = self.camera
            d["camera_shape"] = list(self.camera.shape)
        return d


# ── Gym adapter ───────────────────────────────────────────────────────────────

class GymAdapter:
    """Wraps cadenza.Sim into a step-based env keyed on ActionCall.

    Lifecycle::

        gym = GymAdapter(robot="go1")
        obs = gym.reset()
        for call in plan.calls():
            obs, info = gym.step(call)
        gym.close()
    """

    def __init__(
        self,
        robot: str,
        *,
        xml_path: str | None = None,
        headless: bool = False,
        render_camera: bool = False,
        cam_distance: float = 0.0,
        cam_elevation: float = -15.0,
        cam_azimuth: float = 270.0,
        max_action_seconds: float = 30.0,
    ):
        self.robot = robot
        self.xml_path = xml_path
        self.headless = headless
        self.render_camera = render_camera
        self.max_action_seconds = max_action_seconds

        self._cam_cfg = (cam_distance, cam_elevation, cam_azimuth)

        self._sim = None
        self._viewer_cm = None      # context manager
        self._viewer = None
        self._renderer = None
        self._step_count = 0
        self._opened = False

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def reset(self) -> Observation:
        """Initialize sim + viewer; return the first observation."""
        self.close()
        from cadenza.sim import Sim  # late import — pulls in mujoco

        self._sim = Sim(self.robot, xml_path=self.xml_path)

        if not self.headless:
            import mujoco.viewer
            cam_d, cam_e, cam_a = self._cam_cfg
            if cam_d == 0:
                cam_d = 4.0 if self._sim._is_humanoid else 2.5
            self._viewer_cm = mujoco.viewer.launch_passive(
                self._sim.model, self._sim.data,
            )
            self._viewer = self._viewer_cm.__enter__()
            lookat_z = self._sim.spec.kin.com_height_stand * 0.5
            self._viewer.cam.distance = cam_d
            self._viewer.cam.elevation = cam_e
            self._viewer.cam.azimuth = cam_a
            self._viewer.cam.lookat[:] = [0, 0, lookat_z]

        if self.render_camera:
            import mujoco
            self._renderer = mujoco.Renderer(self._sim.model, height=224, width=224)

        self._step_count = 0
        self._opened = True
        return self._observe()

    def close(self) -> None:
        if self._renderer is not None:
            try:
                self._renderer.close()
            except Exception:
                pass
            self._renderer = None
        if self._viewer_cm is not None:
            try:
                self._viewer_cm.__exit__(None, None, None)
            except Exception:
                pass
            self._viewer_cm = None
            self._viewer = None
        self._sim = None
        self._opened = False

    @property
    def is_open(self) -> bool:
        return self._opened

    # ── Step ─────────────────────────────────────────────────────────────────

    def step(self, call: ActionCall) -> tuple[Observation, dict[str, Any]]:
        """Execute one ActionCall and return (observation, info)."""
        if not self._opened:
            raise RuntimeError("GymAdapter.step() called before reset()")

        sim = self._sim
        spec = sim.lib.get(call.action_name)

        # Apply speed/extension scaling using the same helpers as Go1/G1.
        spec = _apply_speed(spec, call.speed)
        spec = _apply_extension(spec, call.extension)

        # Build the inner ActionCall with overrides plumbed for the sim helpers.
        inner = ActionCall(
            action_name=call.action_name,
            speed=call.speed,
            extension=call.extension,
            repeat=max(call.repeat, 1),
            distance_m=call.distance_m or spec.distance_m,
            rotation_rad=call.rotation_rad or spec.rotation_rad,
            duration_s=call.duration_s,
            speed_override=spec.speed_ms * call.speed if spec.is_gait else 0.0,
        )

        start_pos = sim.data.qpos[0:3].copy()
        t0 = time.monotonic()

        viewer = self._viewer if self._viewer is not None else _NullViewer()
        if spec.is_gait:
            ok = sim._run_gait(spec, viewer, inner)
        elif spec.is_phase:
            ok = sim._run_phase(spec, viewer, inner)
        else:
            ok = True

        elapsed = time.monotonic() - t0
        end_pos = sim.data.qpos[0:3].copy()
        moved = float(np.linalg.norm(end_pos[:2] - start_pos[:2]))
        self._step_count += 1

        info = {
            "ok": bool(ok) if isinstance(ok, bool) else True,
            "elapsed_s": elapsed,
            "moved_m": moved,
            "step": self._step_count,
            "raw_result": ok,
        }
        if self._viewer is not None:
            self._viewer.cam.lookat[:] = sim.data.qpos[0:3]
            self._viewer.cam.lookat[2] = max(float(sim.data.qpos[2]) * 0.8, 0.15)

        return self._observe(), info

    # ── Observation extraction ───────────────────────────────────────────────

    def _observe(self) -> Observation:
        sim = self._sim
        state = sim.get_state()
        qpos = sim.data.qpos.copy()
        qvel = sim.data.qvel.copy()

        camera = None
        if self._renderer is not None:
            try:
                self._renderer.update_scene(sim.data, camera="forward")
                camera = self._renderer.render()
            except Exception:
                # camera "forward" may not exist on all robots
                try:
                    self._renderer.update_scene(sim.data)
                    camera = self._renderer.render()
                except Exception:
                    camera = None

        return Observation(
            pos=np.asarray(state["pos"], dtype=np.float32),
            rpy=np.array([state["roll"], state["pitch"], state["yaw"]], dtype=np.float32),
            body_height=float(state["body_height"]),
            qpos=qpos,
            qvel=qvel,
            foot_contacts=tuple(bool(c) for c in state.get("foot_contacts", ())),
            terrain_ahead=state.get("terrain_ahead", {}),
            obstacles_ahead=state.get("obstacles_ahead", {}),
            camera=camera,
            metadata={"robot": self.robot, "step": self._step_count},
        )


# ── Internal helpers (mirror Go1's _apply_speed/_apply_extension) ────────────

def _apply_speed(spec, speed: float):
    from dataclasses import replace
    if speed == 1.0 or not spec.is_phase:
        return spec
    new_phases = []
    for phase in spec.phases:
        new_vel = tuple(v * speed for v in phase.motor_schedule.max_velocity)
        new_sched = replace(phase.motor_schedule, max_velocity=new_vel)
        new_phases.append(replace(
            phase, motor_schedule=new_sched,
            duration_s=phase.duration_s / speed,
        ))
    return replace(spec, phases=tuple(new_phases),
                   duration_s=spec.duration_s / speed)


def _apply_extension(spec, ext: float):
    from dataclasses import replace
    if ext == 1.0:
        return spec
    if spec.is_gait:
        new_gait = replace(spec.gait, step_height=spec.gait.step_height * ext)
        return replace(spec, gait=new_gait)
    if not spec.is_phase:
        return spec
    new_phases = []
    _stand = np.array((0.0, 0.9, -1.8) * 4, dtype=np.float32)
    for phase in spec.phases:
        q = np.array(phase.target.q12, dtype=np.float32)
        nj = min(len(q), 12)
        q_new = q.copy()
        q_new[:nj] = _stand[:nj] + ext * (q[:nj] - _stand[:nj])
        for i in range(nj):
            jtype = i % 3
            if jtype == 0:
                q_new[i] = np.clip(q_new[i], spec.hip_range[0], spec.hip_range[1])
            elif jtype == 1:
                q_new[i] = np.clip(q_new[i], spec.thigh_range[0], spec.thigh_range[1])
            else:
                q_new[i] = np.clip(q_new[i], spec.knee_range[0], spec.knee_range[1])
        new_target = replace(phase.target, q12=tuple(q_new.tolist()))
        new_phases.append(replace(phase, target=new_target))
    return replace(spec, phases=tuple(new_phases))


class _NullViewer:
    """Stand-in for headless mode so sim helpers that probe `is_running`
    keep iterating."""
    cam = type("Cam", (), {
        "distance": 0.0, "elevation": 0.0, "azimuth": 0.0,
        "lookat": np.zeros(3, dtype=np.float64),
    })()

    def is_running(self) -> bool:
        return True

    def sync(self) -> None:
        return
