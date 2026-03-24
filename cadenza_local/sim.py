"""cadenza.sim — MuJoCo simulator for quadruped and humanoid robots."""

from __future__ import annotations

import math, re, time
from pathlib import Path

import numpy as np
import mujoco, mujoco.viewer

from cadenza_local.actions import get_library
from cadenza_local.lora import LoRATranslator
from cadenza_local.lora.optimizer import LoRAOptimizer, SensorSnapshot
from cadenza_local.locomotion.robot_spec import get_spec
from cadenza_local.locomotion.gait_engine import GaitEngine

_HZ = 50.0
_HOLD_S = 1.0
_CHECK_INTERVAL = int(0.5 * _HZ)   # stability check every 0.5s during gaits

# Critical instability thresholds — only triggers on extreme cases
# (violent rocking leg-to-leg, feet off ground, about to topple)
_CRITICAL_TILT = 0.55       # rad (~32°) — body nearly sideways
_CRITICAL_OMEGA = 3.0       # rad/s — violent swinging

# Humanoid robots — different kinematic structure
_HUMANOID_ROBOTS = {"g1"}

# Bundled models — keyed by robot name
_MODELS_DIR = Path(__file__).resolve().parent / "models"
_BUNDLED = {
    "go1": _MODELS_DIR / "go1" / "scene.xml",
    "g1":  _MODELS_DIR / "g1"  / "scene.xml",
}


def _find_model(robot: str, xml_path: str | Path | None) -> Path:
    """Resolve the MuJoCo XML for a robot.

    Priority: explicit path > bundled model.
    """
    if xml_path is not None:
        p = Path(xml_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"Model not found: {xml_path}")
    if robot in _BUNDLED and _BUNDLED[robot].exists():
        return _BUNDLED[robot]
    raise FileNotFoundError(
        f"No model found for '{robot}'. Pass xml_path= or place it in {_MODELS_DIR}/{robot}/scene.xml"
    )


def _parse_commands(commands: str | list[str]) -> list[str]:
    """Accept a string or list. Strings are split on 'then' / 'and'."""
    if isinstance(commands, str):
        return [c.strip() for c in re.split(r'\s+(?:then|and)\s+', commands) if c.strip()]
    return list(commands)


def _rpy(q):
    w, x, y, z = q
    return np.array([
        math.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y)),
        math.asin(max(-1., min(1., 2*(w*y - z*x)))),
        math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z)),
    ], dtype=np.float32)


class Sim:
    """MuJoCo simulation for quadruped robots.

    Usage::

        import cadenza

        # One-liner
        cadenza.run("walk forward 2 meters then turn left then jump")

        # Or with more control
        sim = cadenza.Sim("go1")
        sim.run("walk forward then shake hand then rear kick")
    """

    def __init__(self, robot: str = "go1", xml_path: str | Path | None = None):
        xml = _find_model(robot, xml_path)
        self.spec = get_spec(robot)
        self.lib = get_library(robot)
        self.translator = LoRATranslator(robot)
        self.optimizer = LoRAOptimizer(robot)
        self.model = mujoco.MjModel.from_xml_path(str(xml))
        self.data = mujoco.MjData(self.model)
        self._stand = np.array(self.spec.poses.stand, dtype=np.float64)
        self._crouch = self._compute_crouch(robot, 0.20)
        self._is_humanoid = robot in _HUMANOID_ROBOTS
        self._n_joints = self.spec.n_joints
        self._phys = int(1.0 / (self.model.opt.timestep * _HZ))
        self._robot = robot
        self._foot_geom_ids: list[int] | None = None
        self._terrain_sensors: dict | None = None   # current terrain context for mid-action checks
        self._init_pose()

    def _compute_crouch(self, robot: str, height: float) -> np.ndarray:
        """Compute joint angles for a crouched pose at the given body height via IK."""
        if robot in _HUMANOID_ROBOTS:
            return self._stand.copy()  # humanoids don't crouch-turn
        from cadenza_local.locomotion.kinematics import nominal_foot_positions, ik_leg, legs_to_joint_vector
        feet = nominal_foot_positions(self.spec.kin, height)
        legs = {}
        for leg in ("FL", "FR", "RL", "RR"):
            q = ik_leg(leg, feet[leg], self.spec.kin)
            if q is None:
                return self._stand.copy()
            legs[leg] = q
        return legs_to_joint_vector(legs).astype(np.float64)

    def _init_pose(self):
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.qpos[2] = self.spec.kin.com_height_stand
        self.data.qpos[3] = 1.0
        self.data.qpos[7:7 + self._n_joints] = self._stand
        mujoco.mj_forward(self.model, self.data)
        if self._is_humanoid:
            # Humanoid init: active ankle balance at physics rate
            from cadenza_local.locomotion.bipedal_gait import BipedalGaitEngine
            engine = BipedalGaitEngine(self.spec, gait_name="stand")
            phys_dt = self.model.opt.timestep
            cmd = np.zeros(3, dtype=np.float32)
            for _ in range(1000):
                rpy = _rpy(self.data.qpos[3:7])
                self.data.ctrl[:] = engine.step(phys_dt, cmd, rpy)
                mujoco.mj_step(self.model, self.data)
        else:
            for _ in range(500):
                self.data.ctrl[:] = self._stand
                mujoco.mj_step(self.model, self.data)

    def run(self, commands: str | list[str], cam_distance: float = 0,
            cam_elevation: float = -15, cam_azimuth: float = 270,
            terrain_sensors: list[dict] | None = None,
            max_retries: int = 3):
        """Run commands in the MuJoCo viewer with closed-loop stability feedback.

        Args:
            commands: Natural language string or list of strings.
                      String is auto-split on "then" / "and".
            terrain_sensors: Optional per-command sensor overrides (slope, roughness, etc.).
                             List aligned with commands. Each entry is a dict of
                             SensorSnapshot field overrides from VLA perception.
            max_retries: Max recovery attempts per action before skipping.
        """
        if cam_distance == 0:
            cam_distance = 4.0 if self._is_humanoid else 2.5
        cmds = _parse_commands(commands)
        print(f"\n  Cadenza {self._robot}  |  {len(cmds)} commands  |  feedback=ON\n")

        lookat_z = self.spec.kin.com_height_stand * 0.5

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            viewer.cam.distance = cam_distance
            viewer.cam.elevation = cam_elevation
            viewer.cam.azimuth = cam_azimuth
            viewer.cam.lookat[:] = [0, 0, lookat_z]

            for i, command in enumerate(cmds):
                if not viewer.is_running():
                    break

                # Get terrain sensor overrides for this command (if provided)
                t_sens = None
                if terrain_sensors and i < len(terrain_sensors):
                    t_sens = terrain_sensors[i]
                self._terrain_sensors = t_sens  # store for mid-action checks

                # Read live sensors + overlay terrain data
                sensors = self._read_sensors(t_sens)
                env = self.optimizer.classify(sensors)

                print(f"  [{i+1}/{len(cmds)}] \"{command}\"")
                if env.needs_caution:
                    flags = []
                    if env.terrain != "flat": flags.append(f"terrain={env.terrain}")
                    if env.tightness != "open": flags.append(f"space={env.tightness}")
                    if env.stability != "stable": flags.append(f"stability={env.stability}")
                    if env.slip_risk not in ("none", "low"): flags.append(f"slip={env.slip_risk}")
                    if env.slope_category != "flat": flags.append(f"slope={env.slope_category}")
                    print(f"    env: {', '.join(flags)}")

                # Translate then optimize with current sensor data
                plan = self.translator.translate(command)
                plan = self.optimizer.optimize(plan, sensors)

                for call in plan.calls:
                    if not viewer.is_running():
                        break

                    attempt = 0
                    success = False

                    while attempt <= max_retries and not success:
                        if attempt > 0:
                            # Recovery: stabilize, re-read sensors, re-optimize
                            print(f"    !! Recovering (attempt {attempt}/{max_retries})...")
                            settled = self._stabilize(viewer)
                            if not settled:
                                print(f"    !! Could not stabilize — skipping")
                                break

                            # Re-read sensors after stabilizing
                            sensors = self._read_sensors(t_sens)
                            env = self.optimizer.classify(sensors)
                            print(f"    !! Re-assessed: stability={env.stability}, terrain={env.terrain}")

                            # Re-optimize the remaining action
                            retry_plan = self.optimizer.optimize(
                                plan.__class__(calls=[call], source_command=command),
                                sensors,
                            )
                            call = retry_plan.calls[0]

                        action = self.lib.get(call.action_name)
                        label = call.action_name
                        if call.speed_override > 0:
                            label += f" @{call.speed_override:.2f}m/s"
                        if call.height_override > 0:
                            label += f" h={call.height_override:.2f}m"
                        if call.distance_m > 0:
                            label += f" {call.distance_m:.1f}m"
                        if call.rotation_rad > 0:
                            label += f" {math.degrees(call.rotation_rad):.0f}deg"
                        prefix = "    ->" if attempt == 0 else "    ~>"
                        print(f"  {prefix} {label}")

                        start = self.data.qpos[0:3].copy()
                        if action.is_gait:
                            ok = self._run_gait(action, viewer, call)
                        elif action.is_phase:
                            ok = self._run_phase(action, viewer, call)
                        else:
                            ok = True
                        end = self.data.qpos[0:3].copy()
                        moved = float(np.linalg.norm(end[:2] - start[:2]))
                        print(f"       {'OK' if ok else 'ABORT'}  moved={moved:.2f}m  z={end[2]:.3f}m")

                        if ok:
                            success = True
                        else:
                            attempt += 1

                    if not success and attempt > max_retries:
                        print(f"    !! Skipped after {max_retries} retries")
                        # Stabilize before moving to next command
                        self._stabilize(viewer)

                    viewer.cam.lookat[:] = self.data.qpos[0:3]
                    viewer.cam.lookat[2] = max(float(self.data.qpos[2]) * 0.8, 0.15)
                print()

            print("  Done. Close viewer to exit.")
            while viewer.is_running():
                self.data.ctrl[:] = self._stand
                for _ in range(self._phys):
                    mujoco.mj_step(self.model, self.data)
                viewer.sync()
                time.sleep(0.02)

    # ── reactive loop ──

    def get_state(self) -> dict:
        """Current robot state — position, orientation, contacts, terrain ahead."""
        rpy = _rpy(self.data.qpos[3:7])
        terrain = self._probe_terrain_ahead()
        return {
            "pos": self.data.qpos[0:3].copy(),
            "yaw": float(rpy[2]),
            "roll": float(rpy[0]),
            "pitch": float(rpy[1]),
            "body_height": float(self.data.qpos[2]),
            "foot_contacts": self._foot_contacts(),
            "terrain_ahead": terrain,
        }

    def _probe_terrain_ahead(self) -> dict:
        """Raycast ahead of the robot to measure terrain height changes.

        Casts rays downward at several distances in front of the robot.
        Current ground height is estimated from foot z-positions.
        Returns max step-up height detected — the VLA's "vision".
        """
        pos = self.data.qpos[0:3].copy()
        yaw = float(_rpy(self.data.qpos[3:7])[2])

        # Direction the robot faces (head is -x in body frame)
        fwd = np.array([math.cos(yaw + math.pi), math.sin(yaw + math.pi), 0.0])
        # Lateral offset to cast rays beside the robot (avoid hitting legs)
        right = np.array([math.cos(yaw + math.pi - math.pi/2),
                          math.sin(yaw + math.pi - math.pi/2), 0.0])

        # Current ground z from foot contact positions
        ground_z_here = self._foot_ground_z()

        # Probe ahead: cast from well above, offset laterally to miss legs
        probe_dists = [0.15, 0.25, 0.35, 0.50]
        ray_start_z = float(pos[2]) + 0.8
        ray_dir = np.array([0.0, 0.0, -1.0], dtype=np.float64)

        heights = []
        max_step_up = 0.0

        for d in probe_dists:
            # Cast two rays (left and right of centerline) and take the max
            best_z = None
            for lat in [-0.12, 0.0, 0.12]:
                px = float(pos[0]) + fwd[0] * d + right[0] * lat
                py = float(pos[1]) + fwd[1] * d + right[1] * lat
                gz = self._raycast_terrain_z(px, py, ray_start_z, ray_dir)
                if gz is not None and (best_z is None or gz > best_z):
                    best_z = gz
            heights.append(best_z)
            if best_z is not None and ground_z_here is not None:
                step = best_z - ground_z_here
                if step > max_step_up:
                    max_step_up = step

        return {
            "ground_z_here": ground_z_here,
            "ground_z_ahead": heights,
            "max_step_up": max_step_up,
            "probe_dists": probe_dists,
        }

    def _foot_ground_z(self) -> float | None:
        """Estimate current ground z from the lowest foot contact."""
        foot_bodies = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        min_z = None
        for name in foot_bodies:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid >= 0:
                z = float(self.data.xpos[bid, 2])
                if min_z is None or z < min_z:
                    min_z = z
        return min_z

    def _raycast_terrain_z(self, x: float, y: float, z_start: float,
                           ray_dir: np.ndarray) -> float | None:
        """Cast a ray downward, skip robot geoms, return terrain z-hit."""
        if not hasattr(self, '_robot_body_ids'):
            self._robot_body_ids = set()
            for i in range(self.model.nbody):
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
                if name and name != "world":
                    self._robot_body_ids.add(i)

        ray_from = np.array([x, y, z_start], dtype=np.float64)
        geomid = np.array([-1], dtype=np.int32)
        dist = mujoco.mj_ray(
            self.model, self.data, ray_from, ray_dir,
            None, 1, -1, geomid,
        )
        if dist < 0:
            return None
        gid = int(geomid[0])
        if gid >= 0 and self.model.geom_bodyid[gid] in self._robot_body_ids:
            return None
        return z_start + ray_dir[2] * dist

    def run_reactive(self, memory_fn, vla_fn, goal_fn,
                     cam_distance: float = 2.5, cam_elevation: float = -15,
                     cam_azimuth: float = 270, step_duration: float = 0.3):
        """Memory-driven locomotion with VLA monitoring.

        The memory system drives the robot forward continuously in small steps.
        The VLA monitors every step but only interrupts when a correction is
        needed (off-track, heading drift). When the VLA flags an issue, the
        memory system switches to turn mode until aligned, then resumes.

        Args:
            memory_fn: callable(state) -> dict with keys:
                "command": str — zone-appropriate gait command
                "sensors": dict — terrain sensor overrides
                "zone": str — current zone name
            vla_fn: callable(state) -> dict with keys:
                "ok": bool — True if on track, False if correction needed
                "turn": ActionCall | None — turn action if correction needed
                "log": str — compact status line
            goal_fn: callable(state) -> bool — True when done
            step_duration: seconds per forward step (small = frequent VLA input)
        """
        print(f"\n  Cadenza {self._robot}  |  reactive mode\n")

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            viewer.cam.distance = cam_distance
            viewer.cam.elevation = cam_elevation
            viewer.cam.azimuth = cam_azimuth
            viewer.cam.lookat[:] = [0, 0, 0.25]

            step_n = 0
            step_steps = max(2, int(step_duration * _HZ))
            last_zone = None
            # Keep a gait engine running across steps for smooth motion
            cur_engine = None
            cur_cmd = None
            cur_action = None

            while viewer.is_running():
                state = self.get_state()
                if goal_fn(state):
                    print("  GOAL")
                    break

                # ── Memory decides the gait, VLA monitors ──
                mem = memory_fn(state)
                vla = vla_fn(state)
                step_n += 1

                zone = mem.get("zone", "")
                if zone != last_zone:
                    print(f"\n  ── {zone} ──")
                    last_zone = zone
                    cur_engine = None  # new zone = new gait

                print(f"  [{step_n:3d}] {vla['log']}")

                # ── VLA says off track → turn until aligned ──
                if not vla["ok"] and vla.get("turn") is not None:
                    cur_engine = None
                    narrow = vla.get("narrow", False)

                    if narrow:
                        # ── NARROW ZONE: compress → precision shuffle → extend ──
                        # Phase 1: Compress legs down
                        cur_q = self.data.qpos[7:7 + self._n_joints].astype(np.float64).copy()
                        self._smooth_blend(cur_q, self._crouch, 0.6, viewer)
                        self._hold(self._crouch, viewer, 0.25)

                        # Phase 2: Precision turn — tiny steps while crouched
                        safety = 20
                        while safety > 0 and viewer.is_running():
                            tc = vla["turn"]
                            # Remap to precision variant
                            pname = tc.action_name.replace("turn_", "precision_turn_")
                            pturn = ActionCall(action_name=pname, repeat=1,
                                               rotation_rad=tc.rotation_rad)
                            turn_action = self.lib.get(pturn.action_name)
                            engine, n_steps, cmd, _ = self._gait_setup(turn_action, pturn)
                            blend = self.data.qpos[7:7 + self._n_joints].astype(np.float64).copy()
                            done = self._run_gait_chunk(
                                engine, cmd, n_steps, turn_action, viewer, blend_from=blend,
                            )
                            if done < 0:
                                self._stabilize(viewer)
                                break
                            # Settle at crouched height, VLA re-checks
                            self._smooth_blend(
                                self.data.qpos[7:7 + self._n_joints].astype(np.float64).copy(),
                                self._crouch, 0.2, viewer,
                            )
                            self._hold(self._crouch, viewer, 0.15)
                            state = self.get_state()
                            vla = vla_fn(state)
                            safety -= 1
                            if vla["ok"] or vla.get("turn") is None:
                                step_n += 1
                                print(f"  [{step_n:3d}] {vla['log']}")
                                break
                            step_n += 1
                            print(f"  [{step_n:3d}] {vla['log']}")

                        # Phase 3: Extend legs back to walking height
                        cur_q = self.data.qpos[7:7 + self._n_joints].astype(np.float64).copy()
                        self._smooth_blend(cur_q, self._stand, 0.5, viewer)
                        self._hold(self._stand, viewer, 0.15)

                    else:
                        # ── WIDE ZONE: quick turn at normal height ──
                        safety = 20
                        while safety > 0 and viewer.is_running():
                            turn_call = vla["turn"]
                            turn_action = self.lib.get(turn_call.action_name)
                            engine, n_steps, cmd, _ = self._gait_setup(turn_action, turn_call)
                            blend = self.data.qpos[7:7 + self._n_joints].astype(np.float64).copy()
                            done = self._run_gait_chunk(
                                engine, cmd, n_steps, turn_action, viewer, blend_from=blend,
                            )
                            if done < 0:
                                self._stabilize(viewer)
                                break
                            self._hold(self._stand, viewer, 0.15)
                            state = self.get_state()
                            vla = vla_fn(state)
                            safety -= 1
                            if vla["ok"] or vla.get("turn") is None:
                                step_n += 1
                                print(f"  [{step_n:3d}] {vla['log']}")
                                break
                            step_n += 1
                            print(f"  [{step_n:3d}] {vla['log']}")

                    viewer.cam.lookat[:] = self.data.qpos[0:3]
                    viewer.cam.lookat[2] = max(float(self.data.qpos[2]) * 0.8, 0.15)
                    continue

                # ── On track → memory drives forward ──
                t_sens = mem.get("sensors", {})
                command = mem.get("command", "walk forward")

                # Only re-translate if zone/command changed
                if cur_engine is None:
                    sensors = self._read_sensors(t_sens)
                    plan = self.translator.translate(command)
                    plan = self.optimizer.optimize(plan, sensors)
                    if not plan.calls:
                        continue
                    call = plan.calls[0]
                    cur_action = self.lib.get(call.action_name)
                    if cur_action.is_gait:
                        cur_engine, _, cur_cmd, _ = self._gait_setup(cur_action, call)
                    else:
                        # Phase action (climb_step etc) — run fully then reset
                        self._run_phase(cur_action, viewer, call)
                        cur_engine = None
                        viewer.cam.lookat[:] = self.data.qpos[0:3]
                        viewer.cam.lookat[2] = max(float(self.data.qpos[2]) * 0.8, 0.15)
                        continue

                # Update swing height every step from VLA terrain probe
                sw = mem.get("swing_height")
                cur_engine.set_swing_height(sw)  # None = use gait default

                # Execute one small step of the current gait — no blend between
                # consecutive steps so motion is continuous
                self._run_gait_chunk(
                    cur_engine, cur_cmd, step_steps, cur_action, viewer,
                )

                viewer.cam.lookat[:] = self.data.qpos[0:3]
                viewer.cam.lookat[2] = max(float(self.data.qpos[2]) * 0.8, 0.15)

            print("\n  Done. Close viewer to exit.")
            while viewer.is_running():
                self.data.ctrl[:] = self._stand
                for _ in range(self._phys):
                    mujoco.mj_step(self.model, self.data)
                viewer.sync()
                time.sleep(0.02)

    # ── sensor feedback ──

    def _get_foot_geom_ids(self) -> list[int]:
        """Lazily find geom IDs for foot bodies."""
        if self._foot_geom_ids is not None:
            return self._foot_geom_ids
        if self._is_humanoid:
            foot_geom_names = ["L_foot_geom", "R_foot_geom"]
        else:
            foot_geom_names = ["FL_foot_geom", "FR_foot_geom", "RL_foot_geom", "RR_foot_geom"]
        ids = []
        for name in foot_geom_names:
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid >= 0:
                ids.append(gid)
        if not ids:
            # Fallback: try body-based search for older models
            foot_bodies = ["FL_calf", "FR_calf", "RL_calf", "RR_calf"]
            for name in foot_bodies:
                body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
                if body_id >= 0:
                    for g in range(self.model.ngeom):
                        if self.model.geom_bodyid[g] == body_id:
                            ids.append(g)
                            break
        n_feet = 2 if self._is_humanoid else 4
        self._foot_geom_ids = ids if len(ids) == n_feet else list(range(n_feet))
        return self._foot_geom_ids

    def _foot_contacts(self) -> np.ndarray:
        """Check which feet are in contact with the ground."""
        foot_ids = self._get_foot_geom_ids()
        n_feet = len(foot_ids)
        contacts = np.zeros(n_feet, dtype=np.float32)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            for fi, gid in enumerate(foot_ids):
                if c.geom1 == gid or c.geom2 == gid:
                    contacts[fi] = 1.0
        return contacts

    def _read_sensors(self, terrain_sensors: dict | None = None) -> SensorSnapshot:
        """Build a SensorSnapshot from current MuJoCo state + optional terrain info."""
        rpy = _rpy(self.data.qpos[3:7])
        omega = self.data.qvel[3:6].copy()  # angular velocity in world frame
        nj = self._n_joints

        snap = SensorSnapshot(
            roll=float(rpy[0]),
            pitch=float(rpy[1]),
            yaw=float(rpy[2]),
            omega_roll=float(omega[0]),
            omega_pitch=float(omega[1]),
            joint_pos=self.data.qpos[7:7 + nj].astype(np.float32).copy(),
            joint_vel=self.data.qvel[6:6 + nj].astype(np.float32).copy(),
            foot_contacts=self._foot_contacts(),
            body_height=float(self.data.qpos[2]),
        )

        # Overlay terrain-level sensor data (from VLA / scene description)
        if terrain_sensors:
            for k, v in terrain_sensors.items():
                if hasattr(snap, k):
                    setattr(snap, k, v)

        return snap

    def _stabilize(self, viewer, max_wait: float = 3.0) -> bool:
        """Hold standing pose until the robot settles or timeout.

        Returns True if robot reached a stable state, False if still unstable.
        """
        stand = np.array(self.spec.poses.stand, dtype=np.float64)
        dt = 1.0 / _HZ
        steps = int(max_wait * _HZ)
        stable_count = 0
        STABLE_THRESHOLD = int(0.5 * _HZ)  # 0.5s of consecutive stability
        # Humanoids use active balance
        balance_engine = None
        if self._is_humanoid:
            from cadenza_local.locomotion.bipedal_gait import BipedalGaitEngine
            balance_engine = BipedalGaitEngine(self.spec, gait_name="stand")

        for s in range(steps):
            if not viewer.is_running():
                return False
            if balance_engine is not None:
                rpy = _rpy(self.data.qpos[3:7])
                cmd = np.zeros(3, dtype=np.float32)
                self.data.ctrl[:] = balance_engine.step(dt, cmd, rpy)
            else:
                self.data.ctrl[:] = stand
            for _ in range(self._phys):
                mujoco.mj_step(self.model, self.data)
            viewer.sync()

            rpy = _rpy(self.data.qpos[3:7])
            omega = self.data.qvel[3:6]
            tilt = math.sqrt(float(rpy[0])**2 + float(rpy[1])**2)
            ang_speed = math.sqrt(float(omega[0])**2 + float(omega[1])**2)

            if tilt < 0.12 and ang_speed < 0.3:
                stable_count += 1
                if stable_count >= STABLE_THRESHOLD:
                    return True
            else:
                stable_count = 0

            time.sleep(max(0, 1.0 / _HZ - 0.001))

        return False

    def _is_critical(self) -> tuple[bool, float, float]:
        """Check if robot is in critical instability — about to fall over.

        Only returns True for EXTREME cases: violent rocking, legs off ground,
        body nearly sideways. Normal gait wobble is ignored completely.
        """
        rpy = _rpy(self.data.qpos[3:7])
        omega_vec = self.data.qvel[3:6]
        tilt = math.sqrt(float(rpy[0])**2 + float(rpy[1])**2)
        omega = math.sqrt(float(omega_vec[0])**2 + float(omega_vec[1])**2)
        feet = float(np.sum(self._foot_contacts()))

        critical = (tilt > _CRITICAL_TILT or omega > _CRITICAL_OMEGA or feet < 1)
        return critical, tilt, omega

    # ── internals ──

    def _smooth_blend(self, q_from, q_to, duration, viewer):
        """Smoothly interpolate joint targets from q_from to q_to over duration seconds."""
        steps = max(1, int(duration * _HZ))
        t0 = time.monotonic()
        for s in range(steps):
            if not viewer.is_running():
                return
            # Smooth ease-in-out via cosine interpolation
            a = 0.5 * (1.0 - math.cos(math.pi * (s + 1) / steps))
            self.data.ctrl[:] = (1 - a) * q_from + a * q_to
            for _ in range(self._phys):
                mujoco.mj_step(self.model, self.data)
            viewer.sync()
            wait = t0 + (s + 1) / _HZ - time.monotonic()
            if wait > 0:
                time.sleep(wait)

    def _hold(self, q_tgt, viewer, duration):
        steps = max(1, int(duration * _HZ))
        dt = 1.0 / _HZ
        # Humanoids need active balance during hold
        balance_engine = None
        if self._is_humanoid:
            from cadenza_local.locomotion.bipedal_gait import BipedalGaitEngine
            balance_engine = BipedalGaitEngine(self.spec, gait_name="stand")
        t0 = time.monotonic()
        for s in range(steps):
            if not viewer.is_running():
                return
            if balance_engine is not None:
                rpy = _rpy(self.data.qpos[3:7])
                cmd = np.zeros(3, dtype=np.float32)
                self.data.ctrl[:] = balance_engine.step(dt, cmd, rpy)
            else:
                self.data.ctrl[:] = q_tgt
            for _ in range(self._phys):
                mujoco.mj_step(self.model, self.data)
            viewer.sync()
            wait = t0 + (s + 1) / _HZ - time.monotonic()
            if wait > 0:
                time.sleep(wait)

    def _run_phase(self, action, viewer, call):
        dt = 1.0 / _HZ
        last_q = np.array(action.phases[-1].target.q12, dtype=np.float32)

        # Humanoids: use BipedalGaitEngine as active balance base at physics rate
        balance_engine = None
        if self._is_humanoid:
            from cadenza_local.locomotion.bipedal_gait import BipedalGaitEngine
            balance_engine = BipedalGaitEngine(self.spec, gait_name="stand")

        for _ in range(call.repeat):
            if not viewer.is_running():
                return False
            for phase in action.phases:
                if not viewer.is_running():
                    return False
                q_target = np.array(phase.target.q12, dtype=np.float32)
                max_vel = np.array(phase.motor_schedule.max_velocity, dtype=np.float32)
                delay = np.array(phase.motor_schedule.delay_s, dtype=np.float32)
                steps = max(1, int(phase.duration_s * _HZ))
                nj = self._n_joints
                q_cmd = self.data.qpos[7:7 + nj].astype(np.float32).copy()
                dq_max = max_vel * dt
                t0 = time.monotonic()

                for s in range(steps):
                    if not viewer.is_running():
                        return False
                    elapsed = (s + 1) * dt
                    delta = q_target - q_cmd
                    for j in range(nj):
                        if elapsed >= delay[j]:
                            q_cmd[j] += np.clip(delta[j], -dq_max[j], dq_max[j])

                    if balance_engine is not None:
                        # Humanoid: apply balance at PHYSICS rate for stability.
                        # Phase targets override the standing pose, but balance
                        # corrections (ankle/hip) are overlaid at every physics step.
                        phys_dt = self.model.opt.timestep
                        for _ in range(self._phys):
                            rpy = _rpy(self.data.qpos[3:7])
                            q_bal = balance_engine.step(phys_dt, np.zeros(3, dtype=np.float32), rpy)
                            # Start from phase target, overlay balance corrections (ankle + hip roll)
                            q_out = q_cmd.copy()
                            # Ankle pitch: use balance engine's output (contains pitch compensation)
                            q_out[4] = q_cmd[4] + q_bal[4]  # L_ankle: phase + balance delta
                            q_out[10] = q_cmd[10] + q_bal[10]
                            # Ankle roll: use balance engine's roll correction
                            q_out[5] = q_cmd[5] + q_bal[5]
                            q_out[11] = q_cmd[11] + q_bal[11]
                            # Hip roll: use balance engine's roll correction
                            q_out[1] = q_cmd[1] + q_bal[1]
                            q_out[7] = q_cmd[7] + q_bal[7]
                            self.data.ctrl[:] = action.clamp_joints(q_out)
                            mujoco.mj_step(self.model, self.data)
                    else:
                        self.data.ctrl[:] = action.clamp_joints(q_cmd)
                        for _ in range(self._phys):
                            mujoco.mj_step(self.model, self.data)

                    rpy = _rpy(self.data.qpos[3:7])
                    if abs(rpy[0]) > action.max_roll_rad or abs(rpy[1]) > action.max_pitch_rad:
                        return False
                    viewer.sync()
                    wait = t0 + (s + 1) / _HZ - time.monotonic()
                    if wait > 0:
                        time.sleep(wait)

            self._hold(last_q, viewer, _HOLD_S)
        return True

    def _gait_setup(self, action, call):
        """Shared setup for gait execution. Returns (engine, n_steps, cmd, speed)."""
        gait = action.gait
        speed = call.speed_override if call.speed_override > 0 else action.speed_ms
        height = call.height_override if call.height_override > 0 else gait.body_height
        if self._is_humanoid:
            from cadenza_local.locomotion.bipedal_gait import BipedalGaitEngine
            engine = BipedalGaitEngine(self.spec, gait_name=gait.gait_name, body_height=height)
        else:
            engine = GaitEngine(self.spec, gait_name=gait.gait_name, body_height=height)

        dist_total = call.distance_m * call.repeat if call.distance_m > 0 else 0
        rot_total = call.rotation_rad * call.repeat if call.rotation_rad > 0 else 0
        if dist_total > 0:
            duration = dist_total / max(speed, 0.05)
        elif rot_total > 0:
            duration = rot_total / max(abs(gait.cmd_yaw), 0.2)
        else:
            duration = action.duration_s * call.repeat

        n_steps = max(1, int(duration * _HZ))
        cmd = np.array([
            gait.cmd_vx * (speed / max(action.speed_ms, 0.01)) if action.speed_ms > 0 else gait.cmd_vx,
            gait.cmd_vy, gait.cmd_yaw,
        ], dtype=np.float32)
        return engine, n_steps, cmd, speed

    def _run_gait_chunk(self, engine, cmd, n_steps, action, viewer, blend_from=None):
        """Execute n_steps of gait with no stability monitoring. Returns steps completed."""
        dt = 1.0 / _HZ
        blend_n = int(0.5 * _HZ) if blend_from is not None else 0
        q_pre = blend_from
        t0 = time.monotonic()

        for s in range(n_steps):
            if not viewer.is_running():
                return s

            if self._is_humanoid:
                # Humanoid: run gait engine at PHYSICS rate (500Hz) for tight
                # ankle balance feedback — critical for bipedal stability.
                phys_dt = self.model.opt.timestep
                for _ in range(self._phys):
                    rpy = _rpy(self.data.qpos[3:7])
                    q_gait = engine.step(phys_dt, cmd, rpy)
                    if blend_from is not None and s < blend_n:
                        a = s / blend_n
                        q_gait = (1 - a) * q_pre + a * q_gait
                    self.data.ctrl[:] = q_gait
                    mujoco.mj_step(self.model, self.data)
                rpy = _rpy(self.data.qpos[3:7])
            else:
                rpy = _rpy(self.data.qpos[3:7])
                q_gait = engine.step(dt, cmd, rpy)
                if blend_from is not None and s < blend_n:
                    a = s / blend_n
                    q_gait = (1 - a) * q_pre + a * q_gait
                self.data.ctrl[:] = q_gait
                for _ in range(self._phys):
                    mujoco.mj_step(self.model, self.data)

            # Hard abort: way past recovery
            if abs(float(rpy[0])) > action.max_roll_rad or abs(float(rpy[1])) > action.max_pitch_rad:
                return -(s + 1)   # negative = aborted
            viewer.sync()
            wait = t0 + (s + 1) / _HZ - time.monotonic()
            if wait > 0:
                time.sleep(wait)
        return n_steps

    def _run_gait(self, action, viewer, call):
        """Execute a gait action in chunks, always monitoring for critical instability.

        Runs the full gait uninterrupted. Every 0.5s, checks if the robot is
        in a critical state (violent rocking, legs off ground). If so:
        1. Stop and hold standing pose until settled
        2. Resume the SAME gait with remaining distance — no direction changes,
           no speed changes, just pick up where we left off.

        Normal wobble is completely ignored — the robot pushes through it.
        """
        engine, total_steps, cmd, speed = self._gait_setup(action, call)
        remaining = total_steps
        need_blend = True
        max_pauses = 3  # don't get stuck in infinite pause loops

        while remaining > 0:
            chunk_size = min(_CHECK_INTERVAL, remaining)

            blend = self.data.qpos[7:7 + self._n_joints].astype(np.float64).copy() if need_blend else None
            need_blend = False

            done = self._run_gait_chunk(engine, cmd, chunk_size, action, viewer, blend_from=blend)

            if done < 0:
                return False  # hard abort — flipped over

            remaining -= chunk_size

            if remaining <= 0:
                break

            # Always monitoring — but only react to CRITICAL instability
            critical, tilt, omega = self._is_critical()

            if critical and max_pauses > 0:
                print(f"       !! Critical (tilt={math.degrees(tilt):.0f}° ω={omega:.1f}) — stabilizing then resuming")
                settled = self._stabilize(viewer)
                max_pauses -= 1
                if not settled:
                    return False
                # Resume: re-blend from current pose, same cmd, same direction
                need_blend = True
                engine, _, _, _ = self._gait_setup(action, call)

        self._hold(np.array(self.spec.poses.stand, dtype=np.float32), viewer, _HOLD_S)
        return True


# Keep backward compat
Go1Sim = Sim


def run(commands: str | list[str], robot: str = "go1", **kwargs):
    """One-liner: simulate commands in MuJoCo viewer.

    Usage::

        import cadenza
        cadenza.run("walk forward 2 meters then turn left then jump")
    """
    sim = Sim(robot=robot)
    sim.run(commands, **kwargs)
