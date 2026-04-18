"""cadenza.deploy.g1_driver — Physical deployment driver for Unitree G1.

Translates Cadenza actions into real-time motor commands for the G1 humanoid
over DDS. Supports debug mode and motion mode (with LocoClient for base movement).

Usage::

    import cadenza

    g1 = cadenza.g1()
    g1.deploy([
        g1.stand(),
        g1.walk_forward(speed=0.3),
    ])
"""

from __future__ import annotations

import time
import threading

import numpy as np

from cadenza.deploy.connection import RobotConnection

# G1: 35 motors total
# Legs: 0-11 (6 per leg), Waist: 12-14, Arms: 15-28, Hands: 29-34
_N_JOINTS = 35
_N_LEG_JOINTS = 12
_CONTROL_HZ = 250.0
_CONTROL_DT = 1.0 / _CONTROL_HZ
_VELOCITY_LIMIT = 20.0

# G1 PD gains (from xr_teleoperate)
_KP_HIGH = 300.0
_KD_HIGH = 3.0
_KP_LOW = 80.0
_KD_LOW = 3.0
_KP_WRIST = 40.0
_KD_WRIST = 1.5

# Weak motors (lower gains)
_WEAK_MOTORS = {12, 13, 14, 23, 24, 25, 26, 27, 28}  # waist + wrist joints
_WRIST_MOTORS = {23, 24, 25, 26, 27, 28}


def _pd_gains(joint_idx: int) -> tuple[float, float]:
    if joint_idx in _WRIST_MOTORS:
        return _KP_WRIST, _KD_WRIST
    if joint_idx in _WEAK_MOTORS:
        return _KP_LOW, _KD_LOW
    return _KP_HIGH, _KD_HIGH


class G1Driver:
    """Deploys Cadenza actions to a physical G1 humanoid robot.

    Handles:
    - DDS connection (unitree_hg protocol)
    - Debug mode / motion mode switching
    - Joint position control at 250Hz
    - LocoClient integration for base locomotion
    """

    def __init__(self, domain_id: int = 0, network_interface: str | None = None,
                 motion_mode: bool = False):
        self._conn = RobotConnection("g1", domain_id=domain_id,
                                     network_interface=network_interface)
        self._motion_mode = motion_mode
        self._running = False
        self._ctrl_thread = None
        self._target_q = np.zeros(_N_JOINTS, dtype=np.float32)
        self._ctrl_lock = threading.Lock()
        self._loco = None
        self._motion_switcher = None

    def connect(self) -> "G1Driver":
        """Connect to the physical G1 robot."""
        self._conn.connect()

        # Get current joint positions
        current_q = self._conn.read_q()[:_N_JOINTS]
        with self._ctrl_lock:
            self._target_q = current_q.copy()

        # Enter debug mode if needed
        if not self._motion_mode:
            self._enter_debug_mode()

        # Start control loop
        self._running = True
        self._ctrl_thread = threading.Thread(target=self._control_loop, daemon=True)
        self._ctrl_thread.start()

        # Initialize loco client for base movement (motion mode only)
        if self._motion_mode:
            self._init_loco()

        print(f"  G1 driver ready ({'motion' if self._motion_mode else 'debug'} mode)")
        return self

    def _enter_debug_mode(self):
        """Switch robot to debug mode (direct joint control)."""
        try:
            from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
            self._motion_switcher = MotionSwitcherClient()
            self._motion_switcher.SetTimeout(1.0)
            self._motion_switcher.Init()

            status, result = self._motion_switcher.CheckMode()
            while result.get("name"):
                self._motion_switcher.ReleaseMode()
                status, result = self._motion_switcher.CheckMode()
                time.sleep(1)
            print("  Debug mode entered.")
        except Exception as e:
            print(f"  Warning: Could not switch to debug mode: {e}")

    def _init_loco(self):
        """Initialize locomotion client for base movement."""
        try:
            from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
            self._loco = LocoClient()
            self._loco.SetTimeout(0.0001)
            self._loco.Init()
            print("  LocoClient initialized.")
        except Exception as e:
            print(f"  Warning: LocoClient unavailable: {e}")
            self._loco = None

    def disconnect(self):
        """Gracefully shut down."""
        print("  Shutting down G1 driver...")
        self._running = False
        if self._ctrl_thread:
            self._ctrl_thread.join(timeout=2.0)
        # Re-enable AI mode
        if self._motion_switcher:
            try:
                self._motion_switcher.SelectMode(nameOrAlias='ai')
            except Exception:
                pass

    def _control_loop(self):
        """250Hz control loop for all 35 joints."""
        while self._running:
            t0 = time.time()

            with self._ctrl_lock:
                target = self._target_q.copy()

            current_q = self._conn.read_q()
            if current_q is not None:
                current_q = current_q[:_N_JOINTS]
                delta = target - current_q
                max_step = _VELOCITY_LIMIT * _CONTROL_DT
                clipped = current_q + np.clip(delta, -max_step, max_step)
            else:
                clipped = target

            cmds = []
            for i in range(_N_JOINTS):
                kp, kd = _pd_gains(i)
                cmds.append({
                    "id": i,
                    "q": float(clipped[i]),
                    "dq": 0.0,
                    "kp": kp,
                    "kd": kd,
                    "tau": 0.0,
                    "mode": 1,
                })
            self._conn.send_cmd(cmds)

            elapsed = time.time() - t0
            time.sleep(max(0, _CONTROL_DT - elapsed))

    def set_target(self, q: np.ndarray):
        """Set joint position target."""
        with self._ctrl_lock:
            self._target_q[:len(q)] = np.array(q, dtype=np.float32)

    def move_base(self, vx: float = 0.0, vy: float = 0.0, vyaw: float = 0.0):
        """Move the robot base (motion mode only, uses LocoClient)."""
        if self._loco is None:
            print("  Warning: LocoClient not available. Use motion_mode=True.")
            return
        self._loco.Move(vx, vy, vyaw, continous_move=False)

    def deploy(self, sequence: list):
        """Deploy a sequence of Steps to the physical G1."""
        print(f"\n  Deploying {len(sequence)} actions to G1...\n")

        for i, item in enumerate(sequence):
            if isinstance(item, list):
                step = item[0]
                print(f"  [{i+1}] concurrent — running first: {step.name}")
            else:
                step = item
                print(f"  [{i+1}] {step.name}", end="")
                if step.speed != 1.0:
                    print(f" speed={step.speed}", end="")
                print()

            # For gait actions in motion mode, use LocoClient
            from cadenza.actions import get_library
            lib = get_library("g1")
            try:
                action = lib.get(step.name)
            except KeyError:
                print(f"       SKIP — action '{step.name}' not in g1 library")
                continue

            if action.is_gait and self._loco:
                gait = action.gait
                spd = step.speed
                duration = action.duration_s * (step.repeat or 1)
                print(f"       loco: vx={gait.cmd_vx*spd:.2f} dur={duration:.1f}s")
                t0 = time.time()
                while time.time() - t0 < duration:
                    self.move_base(gait.cmd_vx * spd, gait.cmd_vy * spd, gait.cmd_yaw * spd)
                    time.sleep(0.1)
                self.move_base(0, 0, 0)
            elif action.is_phase:
                # Direct joint control for phase-based actions
                stand = self._target_q.copy()
                for phase in action.phases:
                    q_target = np.array(phase.target.q12, dtype=np.float32)
                    duration = phase.duration_s / max(step.speed, 0.1)
                    steps = max(1, int(duration * 50))
                    dt = duration / steps
                    q_cmd = self._target_q[:len(q_target)].copy()

                    max_vel = np.array(phase.motor_schedule.max_velocity, dtype=np.float32)
                    if step.speed != 1.0:
                        max_vel *= step.speed
                    dq_max = max_vel * dt

                    for s in range(steps):
                        delta = q_target - q_cmd
                        for j in range(len(q_target)):
                            q_cmd[j] += np.clip(delta[j], -dq_max[j], dq_max[j])
                        self.set_target(q_cmd)
                        time.sleep(dt)
                time.sleep(0.5)

            print(f"       OK")

        print("\n  Deploy complete.")
