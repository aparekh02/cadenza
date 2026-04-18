"""cadenza.go1 — Clean developer API for the Unitree Go1.

Usage::

    import cadenza

    go1 = cadenza.go1()
    go1.run([
        go1.stand(),
        go1.jump(speed=2.0, extension=0.8),
        [go1.turn_left(), go1.walk_forward(speed=1.5)],  # concurrent
        go1.flip(),
    ])
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace

import numpy as np
import mujoco, mujoco.viewer

from cadenza.actions import get_library
from cadenza.actions.library import (
    ActionSpec, ActionPhase, JointTarget, MotorSchedule, GaitAction,
)
from cadenza.actions.library import ActionCall
from cadenza.locomotion.robot_spec import get_spec
from cadenza.locomotion.gait_engine import GaitEngine

from pathlib import Path

_HZ = 50.0
_STAND = (0.0, 0.9, -1.8) * 4
_STAND_NP = np.array(_STAND, dtype=np.float32)
_LIBRARY_DIR = Path(__file__).resolve().parent / "library" / "go1"


# ═══════════════════════════════════════════════════════════════════════════════
#  Action descriptor — lightweight, returned by go1.jump() etc.
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Step:
    """One action in a sequence. Returned by go1.jump(), go1.walk_forward(), etc."""
    name: str
    speed: float = 1.0           # multiplier on max_velocity (phase) or cmd speed (gait)
    extension: float = 1.0       # multiplier on joint displacement from stand pose
    repeat: int = 1
    distance_m: float = 0.0      # 0 = use action default
    rotation_rad: float = 0.0    # 0 = use action default

    def __repr__(self):
        parts = [self.name]
        if self.speed != 1.0:
            parts.append(f"speed={self.speed}")
        if self.extension != 1.0:
            parts.append(f"ext={self.extension}")
        if self.repeat > 1:
            parts.append(f"x{self.repeat}")
        if self.distance_m > 0:
            parts.append(f"{self.distance_m}m")
        return f"Step({', '.join(parts)})"


# ═══════════════════════════════════════════════════════════════════════════════
#  Go1 — the main developer-facing class
# ═══════════════════════════════════════════════════════════════════════════════

class Go1:
    """Unitree Go1 robot controller.

    Create with ``cadenza.go1()``, define actions, and call ``run()``.

    Example::

        import cadenza

        go1 = cadenza.go1()
        go1.run([
            go1.stand(),
            go1.walk_forward(speed=1.5, distance_m=2.0),
            [go1.turn_left(), go1.walk_forward()],   # concurrent
            go1.jump(speed=2.0, extension=1.2),
        ])
    """

    def __init__(self, cam_distance: float = 2.5, cam_elevation: float = -15,
                 cam_azimuth: float = 270, xml_path: str | None = None):
        self._xml_path = xml_path
        self._cam_distance = cam_distance
        self._cam_elevation = cam_elevation
        self._cam_azimuth = cam_azimuth

    # ── Action methods ────────────────────────────────────────────────────────
    # Each returns a Step descriptor. Nothing is executed until run().

    def stand(self, speed=1.0, extension=1.0):
        return Step("stand", speed=speed, extension=extension)

    def stand_up(self, speed=1.0, extension=1.0):
        return Step("stand_up", speed=speed, extension=extension)

    def sit(self, speed=1.0, extension=1.0):
        return Step("sit", speed=speed, extension=extension)

    def lie_down(self, speed=1.0, extension=1.0):
        return Step("lie_down", speed=speed, extension=extension)

    def jump(self, speed=1.0, extension=1.0):
        return Step("jump", speed=speed, extension=extension)

    # def flip(self, speed=1.0, extension=1.0):
    #     return Step("flip", speed=speed, extension=extension)

    def walk_forward(self, speed=1.0, extension=1.0, distance_m=0.0, repeat=1):
        return Step("walk_forward", speed=speed, extension=extension,
                    distance_m=distance_m, repeat=repeat)

    def walk_backward(self, speed=1.0, extension=1.0, distance_m=0.0, repeat=1):
        return Step("walk_backward", speed=speed, extension=extension,
                    distance_m=distance_m, repeat=repeat)

    def trot_forward(self, speed=1.0, extension=1.0, distance_m=0.0, repeat=1):
        return Step("trot_forward", speed=speed, extension=extension,
                    distance_m=distance_m, repeat=repeat)

    def crawl_forward(self, speed=1.0, extension=1.0, distance_m=0.0, repeat=1):
        return Step("crawl_forward", speed=speed, extension=extension,
                    distance_m=distance_m, repeat=repeat)

    def pace_forward(self, speed=1.0, extension=1.0, distance_m=0.0, repeat=1):
        return Step("pace_forward", speed=speed, extension=extension,
                    distance_m=distance_m, repeat=repeat)

    def bound_forward(self, speed=1.0, extension=1.0, distance_m=0.0, repeat=1):
        return Step("bound_forward", speed=speed, extension=extension,
                    distance_m=distance_m, repeat=repeat)

    def turn_left(self, speed=1.0, rotation_rad=0.0, repeat=1):
        return Step("turn_left", speed=speed, rotation_rad=rotation_rad, repeat=repeat)

    def turn_right(self, speed=1.0, rotation_rad=0.0, repeat=1):
        return Step("turn_right", speed=speed, rotation_rad=rotation_rad, repeat=repeat)

    def climb_step(self, speed=1.0, extension=1.0, repeat=1):
        return Step("climb_step", speed=speed, extension=extension, repeat=repeat)

    def side_step_left(self, speed=1.0, extension=1.0, distance_m=0.0, repeat=1):
        return Step("side_step_left", speed=speed, extension=extension,
                    distance_m=distance_m, repeat=repeat)

    def side_step_right(self, speed=1.0, extension=1.0, distance_m=0.0, repeat=1):
        return Step("side_step_right", speed=speed, extension=extension,
                    distance_m=distance_m, repeat=repeat)

    def rear_up(self, speed=1.0, extension=1.0):
        return Step("rear_up", speed=speed, extension=extension)

    def shake_hand(self, speed=1.0, extension=1.0):
        return Step("shake_hand", speed=speed, extension=extension)

    def rear_kick(self, speed=1.0, extension=1.0):
        return Step("rear_kick", speed=speed, extension=extension)

    def crouch(self, speed=1.0, extension=1.0):
        return Step("crouch", speed=speed, extension=extension)

    def deep_crouch(self, speed=1.0, extension=1.0):
        return Step("deep_crouch", speed=speed, extension=extension)

    def action(self, name: str, **kwargs) -> Step:
        """Generic action by name. Use for any action in the library."""
        return Step(name, **kwargs)

    def _call_to_step(self, call: ActionCall) -> Step:
        """Convert an ActionCall to a Step."""
        return Step(
            name=call.action_name,
            speed=call.speed,
            distance_m=call.distance_m,
            rotation_rad=call.rotation_rad,
            repeat=call.repeat,
        )

    # ── Run ───────────────────────────────────────────────────────────────────

    def run(self, sequence: list, vla: bool = False):
        """Execute a sequence of actions.

        Args:
            sequence: List of Step objects (or nested lists for concurrency).
                      Nested list = concurrent actions (gait commands are merged).
            vla: If True, enable VLA guardian for obstacle avoidance.
                 The VLA model (SmolVLM-256M) monitors the robot's camera
                 and interjects avoidance actions when obstacles are detected.

        Example::

            go1.run([
                go1.stand(),
                go1.walk_forward(speed=1.5),
                [go1.turn_left(), go1.walk_forward()],  # walk + turn = arc
                go1.jump(speed=2.0),
            ])

            # With VLA obstacle avoidance:
            go1.run([go1.walk_forward(distance_m=5.0)], vla=True)
        """
        from cadenza.sim import Sim

        sim = Sim("go1", xml_path=self._xml_path)
        lib = get_library("go1")
        spec = get_spec("go1")
        steps = self._normalize_sequence(sequence)

        guardian = None
        if vla:
            from cadenza.vla import VLAGuardian
            guardian = VLAGuardian("go1", show_camera=True)
            guardian.load()

        vla_label = "  VLA=ON" if vla else ""
        print(f"\n  Cadenza Go1  |  {len(steps)} steps{vla_label}\n")

        lookat_z = spec.kin.com_height_stand * 0.5

        with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
            viewer.cam.distance = self._cam_distance
            viewer.cam.elevation = self._cam_elevation
            viewer.cam.azimuth = self._cam_azimuth
            viewer.cam.lookat[:] = [0, 0, lookat_z]

            i = 0
            while i < len(steps):
                if not viewer.is_running():
                    break

                item = steps[i]

                if isinstance(item, list):
                    names = " + ".join(s.name for s in item)
                    print(f"  [{i+1}/{len(steps)}] [{names}]")
                    self._execute_concurrent(item, sim, lib, viewer)
                    i += 1
                else:
                    print(f"  [{i+1}/{len(steps)}] {item.name}", end="")
                    if item.speed != 1.0:
                        print(f"  speed={item.speed}", end="")
                    if item.distance_m > 0:
                        print(f"  {item.distance_m:.1f}m", end="")
                    print()

                    result = self._execute_single(item, sim, lib, viewer,
                                                  vla_guardian=guardian)

                    if result is not None and hasattr(result, 'detected') and result.detected:
                        # VLA interrupted — figure out remaining distance
                        completed_frac = 0.0
                        if hasattr(result, '_steps_completed') and hasattr(result, '_steps_total'):
                            completed_frac = result._steps_completed / max(result._steps_total, 1)
                        original_dist = item.distance_m or 0
                        distance_done = original_dist * completed_frac
                        distance_left = original_dist - distance_done

                        print(f"\n  VLA INTERRUPT: obstacle {result.position} ({result.size})")
                        print(f"       Completed {distance_done:.1f}m of {original_dist:.1f}m")
                        print(f"       Remaining: {distance_left:.1f}m")

                        # Execute avoidance — VLA OFF during this sequence
                        avoidance = guardian.get_avoidance_steps(result)
                        if avoidance:
                            print(f"       Avoidance: {[s.name for s in avoidance]}\n")
                            for av_step in avoidance:
                                if not viewer.is_running():
                                    break
                                print(f"    >> {av_step.name}", end="")
                                if av_step.distance_m > 0:
                                    print(f"  {av_step.distance_m:.1f}m", end="")
                                print()
                                self._execute_single(av_step, sim, lib, viewer)
                                viewer.cam.lookat[:] = sim.data.qpos[0:3]
                                viewer.cam.lookat[2] = max(
                                    float(sim.data.qpos[2]) * 0.8, 0.15)

                        # Resume the original action with remaining distance
                        if distance_left > 0.1:
                            resume = Step(
                                name=item.name,
                                speed=item.speed,
                                extension=item.extension,
                                distance_m=distance_left,
                            )
                            print(f"\n  RESUME: {item.name} {distance_left:.1f}m remaining\n")
                            # Don't advance i — we'll re-run this step with remaining distance
                            steps[i] = resume
                            continue  # re-enter the loop with the reduced step
                        else:
                            print(f"\n  VLA: action was nearly complete, moving on\n")

                    i += 1

                viewer.cam.lookat[:] = sim.data.qpos[0:3]
                viewer.cam.lookat[2] = max(float(sim.data.qpos[2]) * 0.8, 0.15)

            print("\n  Done. Close viewer to exit.")
            stand = np.array(spec.poses.stand, dtype=np.float64)
            while viewer.is_running():
                sim.data.ctrl[:] = stand
                for _ in range(sim._phys):
                    mujoco.mj_step(sim.model, sim.data)
                viewer.sync()
                time.sleep(0.02)

    # ── Reactive ──────────────────────────────────────────────────────────────

    def run_reactive(self, memory_fn, vla_fn, goal_fn, step_duration: float = 0.3):
        """Memory-driven locomotion with VLA monitoring.

        Args:
            memory_fn: callable(state) -> dict with "command", "sensors", "zone"
            vla_fn: callable(state) -> dict with "ok", "turn" (ActionCall), "log"
            goal_fn: callable(state) -> bool (True when done)
            step_duration: seconds per forward step

        Example::

            go1 = cadenza.go1(xml_path=cadenza.go1.terrain("terrain"))
            go1.run_reactive(memory_fn, vla_fn, goal_fn)
        """
        from cadenza.sim import Sim
        sim = Sim("go1", xml_path=self._xml_path)
        sim.run_reactive(
            memory_fn=memory_fn,
            vla_fn=vla_fn,
            goal_fn=goal_fn,
            cam_distance=self._cam_distance,
            cam_elevation=self._cam_elevation,
            cam_azimuth=self._cam_azimuth,
            step_duration=step_duration,
        )

    # ── Bundled assets ────────────────────────────────────────────────────────

    @staticmethod
    def terrain(name: str = "terrain") -> str:
        """Path to a bundled terrain XML. Use as xml_path.

        Example::

            go1 = cadenza.go1(xml_path=Go1.terrain("terrain"))
        """
        p = _LIBRARY_DIR / f"{name}.xml"
        if p.exists():
            return str(p)
        raise FileNotFoundError(f"No bundled terrain '{name}' at {p}")

    @staticmethod
    def model() -> str:
        """Path to the bundled Go1 scene XML."""
        p = _LIBRARY_DIR / "go1.xml"
        if p.exists():
            return str(p)
        raise FileNotFoundError(f"No bundled model at {p}")

    # ── Deploy to physical robot ────────────────────────────────────────────

    def deploy(self, sequence: list, domain_id: int = 0,
               network_interface: str | None = None):
        """Deploy actions to a physical Go1 robot over DDS.

        Args:
            sequence: List of Step objects (same as run()).
            domain_id: 0 for real robot, 1 for simulation.
            network_interface: Network interface for DDS (None = default).

        Example::

            go1 = cadenza.go1()
            go1.deploy([
                go1.stand(),
                go1.walk_forward(speed=0.5),
                go1.jump(),
            ])
        """
        from cadenza.deploy.go1_driver import Go1Driver
        steps = self._normalize_sequence(sequence)
        driver = Go1Driver(domain_id=domain_id, network_interface=network_interface)
        driver.connect()
        try:
            driver.deploy(steps)
        finally:
            driver.disconnect()

    # ── SSH deploy ────────────────────────────────────────────────────────────

    @staticmethod
    def ssh(host: str, user: str = "unitree", key: str | None = None,
            port: int = 22, password: str | None = None):
        """Create an SSH connection to a Go1 robot.

        Args:
            host: Robot IP (default Go1: 192.168.123.15)
            user: SSH user (default: "unitree")
            key: Path to SSH private key file
            password: SSH password (alternative to key)

        Returns:
            SSHDeploy instance.

        Example::

            go1 = cadenza.go1()
            conn = go1.ssh("192.168.123.15", key="~/.ssh/id_rsa")
            conn.deploy_and_run("my_demo.py")
        """
        from cadenza.deploy.ssh import SSHDeploy
        return SSHDeploy(host=host, user=user, key=key, port=port, password=password)

    def deploy_ssh(self, script: str, host: str = "192.168.123.15",
                   user: str = "unitree", key: str | None = None,
                   password: str | None = None, setup: bool = True,
                   background: bool = False):
        """Deploy and run a script on the physical Go1 over SSH.

        Args:
            script: Local .py file to upload and run on the robot.
            host: Robot IP (default: 192.168.123.15)
            key: Path to SSH private key
            password: SSH password (alternative to key)
            setup: Run full setup on first deploy (True by default)
            background: Run in background on the robot

        Example::

            go1 = cadenza.go1()
            go1.deploy_ssh("my_demo.py", host="192.168.123.15", key="~/.ssh/go1_rsa")
        """
        from cadenza.deploy.ssh import SSHDeploy
        conn = SSHDeploy(host=host, user=user, key=key, password=password)
        conn.deploy_and_run(script, setup=setup, background=background)

    def deploy_ssh_bridge(self, host: str = "192.168.123.15",
                          user: str = "unitree", key: str | None = None,
                          password: str | None = None, setup: bool = True):
        """Start a bridge on the robot and return a live control handle.

        Use this when your VLA/memory model runs on your host PC and you
        need real-time control of the physical robot.

        Returns:
            HostBridge — send actions, read telemetry, emergency stop.

        Example::

            go1 = cadenza.go1()
            bridge = go1.deploy_ssh_bridge(host="192.168.123.15", key="~/.ssh/go1_rsa")
            bridge.send_action("stand")
            bridge.send_action("walk_forward", speed=0.5)
            print(bridge.telemetry)  # live joint positions
            bridge.estop()           # emergency stop
        """
        from cadenza.deploy.ssh import SSHDeploy
        conn = SSHDeploy(host=host, user=user, key=key, password=password)
        if setup:
            conn.setup()
        return conn.start_bridge(robot="go1")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _normalize_sequence(self, sequence: list) -> list:
        """Validate and normalize: keep Steps and nested lists of Steps."""
        out = []
        for item in sequence:
            if isinstance(item, Step):
                out.append(item)
            elif isinstance(item, list):
                group = [s for s in item if isinstance(s, Step)]
                if group:
                    out.append(group)
            elif isinstance(item, str):
                out.append(Step(item))
            else:
                raise TypeError(f"Expected Step or list, got {type(item)}")
        return out

    def _execute_single(self, step: Step, sim, lib, viewer, vla_guardian=None):
        """Execute one Step. Returns ObstacleResult if VLA interrupted, else None."""
        action_spec = lib.get(step.name)
        action_spec = self._apply_speed(action_spec, step.speed)
        action_spec = self._apply_extension(action_spec, step.extension)

        call = ActionCall(
            action_name=step.name,
            repeat=step.repeat if step.repeat > 1 else 1,
            distance_m=step.distance_m or action_spec.distance_m,
            rotation_rad=step.rotation_rad or action_spec.rotation_rad,
            speed_override=action_spec.speed_ms * step.speed if action_spec.is_gait else 0.0,
        )

        start = sim.data.qpos[0:3].copy()

        if action_spec.is_gait:
            result = sim._run_gait(action_spec, viewer, call,
                                   vla_guardian=vla_guardian)
        elif action_spec.is_phase:
            result = sim._run_phase(action_spec, viewer, call)
        else:
            result = True

        end = sim.data.qpos[0:3].copy()
        moved = float(np.linalg.norm(end[:2] - start[:2]))

        # If VLA interrupted, result is an ObstacleResult, not a bool
        if hasattr(result, 'detected'):
            print(f"       STOPPED by VLA  moved={moved:.2f}m")
            return result

        print(f"       OK  moved={moved:.2f}m  z={end[2]:.3f}m")
        return None

    def _execute_concurrent(self, steps: list[Step], sim, lib, viewer):
        """Execute concurrent actions by merging gait commands."""
        gait_steps = []
        phase_steps = []

        for step in steps:
            spec = lib.get(step.name)
            if spec.is_gait:
                gait_steps.append((step, spec))
            else:
                phase_steps.append((step, spec))

        # Run any phase-based actions first (can't truly be concurrent)
        for step, spec in phase_steps:
            spec = self._apply_speed(spec, step.speed)
            spec = self._apply_extension(spec, step.extension)
            call = ActionCall(action_name=step.name, repeat=step.repeat)
            sim._run_phase(spec, viewer, call)

        if not gait_steps:
            return

        # Merge gait commands
        total_vx = 0.0
        total_vy = 0.0
        total_yaw = 0.0
        max_height = 0.0
        max_step_h = 0.0
        max_duration = 0.0

        for step, spec in gait_steps:
            g = spec.gait
            spd = step.speed
            total_vx += g.cmd_vx * spd
            total_vy += g.cmd_vy * spd
            total_yaw += g.cmd_yaw * spd
            max_height = max(max_height, g.body_height)
            sh = g.step_height * step.extension if step.extension != 1.0 else g.step_height
            max_step_h = max(max_step_h, sh)
            max_duration = max(max_duration, spec.duration_s)

        # Use first gait's name as base
        base_gait_name = gait_steps[0][1].gait.gait_name
        merged_gait = GaitAction(
            gait_name=base_gait_name,
            cmd_vx=total_vx,
            cmd_vy=total_vy,
            cmd_yaw=total_yaw,
            body_height=max_height,
            step_height=max_step_h,
        )

        merged_spec = replace(gait_steps[0][1],
                              gait=merged_gait,
                              duration_s=max_duration)

        call = ActionCall(
            action_name=gait_steps[0][0].name,
            repeat=1,
            distance_m=max_duration * abs(total_vx),
            speed_override=abs(total_vx),
        )

        start = sim.data.qpos[0:3].copy()
        ok = sim._run_gait(merged_spec, viewer, call)
        end = sim.data.qpos[0:3].copy()
        moved = float(np.linalg.norm(end[:2] - start[:2]))
        print(f"       {'OK' if ok else 'ABORT'}  moved={moved:.2f}m  z={end[2]:.3f}m")

    def _apply_speed(self, spec: ActionSpec, speed: float) -> ActionSpec:
        """Scale velocities and durations by speed multiplier."""
        if speed == 1.0 or not spec.is_phase:
            return spec
        new_phases = []
        for phase in spec.phases:
            new_vel = tuple(v * speed for v in phase.motor_schedule.max_velocity)
            new_sched = replace(phase.motor_schedule, max_velocity=new_vel)
            new_phase = replace(phase, motor_schedule=new_sched,
                                duration_s=phase.duration_s / speed)
            new_phases.append(new_phase)
        return replace(spec, phases=tuple(new_phases),
                       duration_s=spec.duration_s / speed)

    def _apply_extension(self, spec: ActionSpec, ext: float) -> ActionSpec:
        """Scale joint targets relative to stand pose."""
        if ext == 1.0:
            return spec
        if spec.is_gait:
            new_gait = replace(spec.gait, step_height=spec.gait.step_height * ext)
            return replace(spec, gait=new_gait)
        if not spec.is_phase:
            return spec

        new_phases = []
        for phase in spec.phases:
            q = np.array(phase.target.q12, dtype=np.float32)
            q_new = _STAND_NP + ext * (q - _STAND_NP)
            # Clamp to joint limits
            for i in range(min(len(q_new), 12)):
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
