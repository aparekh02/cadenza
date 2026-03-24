"""ros2_node.py — ROS2 node wrapping the Cadenza locomotion controller.

Works as a pure standalone script if ROS2 is not available (falls back
to printing commands instead of publishing).

ROS2 subscriptions
------------------
    /cmd_vel         geometry_msgs/Twist   → velocity command
    /joint_states    sensor_msgs/JointState → joint positions + velocities
    /imu/data        sensor_msgs/Imu       → orientation + angular velocity
    /foot_contact    std_msgs/Float32MultiArray → 4 contact flags

ROS2 publications
-----------------
    /cadenza/loco_command  std_msgs/String  → JSON-serialised LocoCommand

All topic names come from the YAML config (config/go1.yaml or config/go2.yaml).

Usage
-----
    # With ROS2:
    python ros2_node.py --config config/go1.yaml --snapshot snapshot.json

    # Without ROS2 (prints to stdout):
    python ros2_node.py --config config/go1.yaml --snapshot snapshot.json --no-ros
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np

# ── Repo on path ───────────────────────────────────────────────────────────
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from cadenza_local.locomotion import (
    STM, STMFrame,
    MapMem, SkillMem, SafetyMem, UserMem,
    LocoController, ExperienceLogger,
    load_config,
)

# ── Conditional ROS2 ───────────────────────────────────────────────────────
try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg     import Twist
    from sensor_msgs.msg       import JointState, Imu
    from std_msgs.msg          import String, Float32MultiArray
    _HAS_ROS2 = True
except ImportError:
    _HAS_ROS2 = False
    Node = object    # type: ignore[misc,assignment]


# ──────────────────────────────────────────────────────────────────────────
#  Snapshot loader (shared with run_controller.py)
# ──────────────────────────────────────────────────────────────────────────

def load_snapshot(path) -> tuple[MapMem, SkillMem, SafetyMem, UserMem]:
    if path and Path(path).exists():
        with open(path) as fh:
            snap = json.load(fh)
        return (
            MapMem.from_list(snap.get("mapmem",    [])),
            SkillMem.from_list(snap.get("skillmem",  [])),
            SafetyMem.from_list(snap.get("safetymem", [])),
            UserMem.from_list(snap.get("usermem",   [])),
        )
    return MapMem(), SkillMem(), SafetyMem(), UserMem()


# ──────────────────────────────────────────────────────────────────────────
#  ROS2 node
# ──────────────────────────────────────────────────────────────────────────

class CadenzaLocoNode(Node):  # type: ignore[misc]
    """ROS2 node: subscribes to sensors, publishes LocoCommand as JSON."""

    def __init__(self, cfg, ctrl: LocoController, logger: ExperienceLogger):
        if _HAS_ROS2:
            super().__init__("cadenza_loco")
        self._cfg    = cfg
        self._ctrl   = ctrl
        self._logger = logger

        # Latest sensor snapshots (updated by callbacks)
        self._lock        = threading.Lock()
        self._joint_pos   = np.zeros(cfg.robot.n_joints, dtype=np.float32)
        self._joint_vel   = np.zeros(cfg.robot.n_joints, dtype=np.float32)
        self._imu_rpy     = np.zeros(3, dtype=np.float32)
        self._imu_omega   = np.zeros(3, dtype=np.float32)
        self._foot_contact = np.ones(4, dtype=np.float32)
        self._cmd_vel     = np.zeros(3, dtype=np.float32)
        self._t0          = time.monotonic()

        if _HAS_ROS2:
            t = cfg.topics
            self.create_subscription(Twist,              t.cmd_vel,      self._cb_cmd_vel,    10)
            self.create_subscription(JointState,         t.joint_states, self._cb_joints,     10)
            self.create_subscription(Imu,                t.imu,          self._cb_imu,        10)
            self.create_subscription(Float32MultiArray,  t.foot_contact, self._cb_foot,       10)
            self._pub = self.create_publisher(String, t.loco_command, 10)
            dt = 1.0 / cfg.control.rate_hz
            self.create_timer(dt, self._control_loop)

    # ── Callbacks ──────────────────────────────────────────────────────────

    def _cb_cmd_vel(self, msg):
        with self._lock:
            self._cmd_vel[:] = [msg.linear.x, msg.linear.y, msg.angular.z]

    def _cb_joints(self, msg):
        with self._lock:
            n = min(self._cfg.robot.n_joints, len(msg.position))
            self._joint_pos[:n] = np.array(msg.position[:n], dtype=np.float32)
            if msg.velocity:
                self._joint_vel[:n] = np.array(msg.velocity[:n], dtype=np.float32)

    def _cb_imu(self, msg):
        from scipy.spatial.transform import Rotation
        q  = msg.orientation
        try:
            r = Rotation.from_quat([q.x, q.y, q.z, q.w])
            rpy = r.as_euler("xyz").astype(np.float32)
        except Exception:
            rpy = np.zeros(3, dtype=np.float32)
        with self._lock:
            self._imu_rpy[:] = rpy
            self._imu_omega[:] = np.array([
                msg.angular_velocity.x,
                msg.angular_velocity.y,
                msg.angular_velocity.z,
            ], dtype=np.float32)

    def _cb_foot(self, msg):
        with self._lock:
            vals = np.array(msg.data[:4], dtype=np.float32) if len(msg.data) >= 4 else np.ones(4, np.float32)
            self._foot_contact[:] = vals

    # ── Control loop ───────────────────────────────────────────────────────

    def _control_loop(self):
        """Called at cfg.control.rate_hz — build frame, step controller, publish."""
        with self._lock:
            frame = STMFrame(
                timestamp    = time.monotonic() - self._t0,
                joint_pos    = self._joint_pos.copy(),
                joint_vel    = self._joint_vel.copy(),
                imu_rpy      = self._imu_rpy.copy(),
                imu_omega    = self._imu_omega.copy(),
                foot_contact = self._foot_contact.copy(),
                cmd_vel      = self._cmd_vel.copy(),
            )
            cmd_vel = self._cmd_vel.copy()

        loco_cmd = self._ctrl.step(frame, cmd_vel)

        if _HAS_ROS2:
            out = String()
            out.data = json.dumps(loco_cmd.to_dict())
            self._pub.publish(out)
        else:
            print(f"[{loco_cmd.timestamp:.2f}]  "
                  f"vx={loco_cmd.cmd_vx:.3f}  vy={loco_cmd.cmd_vy:.3f}  "
                  f"yaw={loco_cmd.cmd_yaw:.3f}  gait={loco_cmd.gait}  "
                  f"terrain={loco_cmd.terrain}  skill={loco_cmd.skill_name}  "
                  f"safe={'Y' if loco_cmd.safety_active else 'N'}")


# ──────────────────────────────────────────────────────────────────────────
#  Entry points
# ──────────────────────────────────────────────────────────────────────────

def run_ros2(cfg, ctrl, logger) -> None:
    rclpy.init()
    node = CadenzaLocoNode(cfg, ctrl, logger)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        ctrl.notify_episode_outcome(success=True, note="ros2 shutdown")
        logger.close()
        node.destroy_node()
        rclpy.shutdown()


def run_standalone(cfg, ctrl, logger, n_steps: int = 300) -> None:
    """Simulate sensor input without ROS2."""
    rng = np.random.default_rng(7)
    dt  = 1.0 / cfg.control.rate_hz
    t0  = time.monotonic()
    joint_pos = np.zeros(cfg.robot.n_joints, dtype=np.float32)

    node = CadenzaLocoNode(cfg, ctrl, logger)   # no ROS2 init

    print(f"Running {n_steps} standalone steps (no ROS2) …\n")
    for i in range(n_steps):
        t = i * dt
        cmd_vel = np.array([
            0.5 * np.sin(t * 0.3),
            0.1 * np.cos(t * 0.5),
            0.2 * np.sin(t * 0.2),
        ], dtype=np.float32)

        # Fake sensor update
        joint_pos += 0.05 * (cmd_vel[0] - joint_pos) + rng.standard_normal(cfg.robot.n_joints).astype(np.float32) * 0.01
        with node._lock:
            node._cmd_vel[:]      = cmd_vel
            node._joint_pos[:]    = joint_pos
            node._joint_vel[:]    = rng.standard_normal(cfg.robot.n_joints).astype(np.float32) * 0.05
            node._imu_rpy[:]      = rng.standard_normal(3).astype(np.float32) * 0.05
            node._imu_omega[:]    = rng.standard_normal(3).astype(np.float32) * 0.02
            node._foot_contact[:] = (rng.random(4) > 0.3).astype(np.float32)

        node._control_loop()

        elapsed = time.monotonic() - t0
        sleep_for = (i + 1) * dt - elapsed
        if sleep_for > 0:
            time.sleep(sleep_for)

    ctrl.notify_episode_outcome(success=True, note="standalone demo")
    logger.close()
    print(f"\nDone. Log: {logger.path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cadenza locomotion ROS2 node")
    parser.add_argument("--config",   type=Path, default=Path("config/go1.yaml"))
    parser.add_argument("--snapshot", type=Path, default=None)
    parser.add_argument("--no-ros",   action="store_true",
                        help="run standalone loop instead of ROS2 spin")
    parser.add_argument("--steps",    type=int, default=300,
                        help="steps for standalone mode (ignored in ROS2 mode)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    snap = args.snapshot or (cfg.memory.snapshot or None)
    mapmem, skillmem, safetymem, usermem = load_snapshot(snap)

    usermem.set("max_speed", cfg.robot.max_speed_xy)

    logger = ExperienceLogger(log_dir=cfg.memory.log_dir, run_id=cfg.robot_model)
    stm    = STM(window=cfg.control.stm_window)
    ctrl   = LocoController(
        cfg=cfg, stm=stm,
        mapmem=mapmem, skillmem=skillmem,
        safetymem=safetymem, usermem=usermem,
        logger=logger,
    )

    if args.no_ros or not _HAS_ROS2:
        if not args.no_ros:
            print("ROS2 not available — falling back to standalone mode")
        run_standalone(cfg, ctrl, logger, n_steps=args.steps)
    else:
        run_ros2(cfg, ctrl, logger)


if __name__ == "__main__":
    main()
