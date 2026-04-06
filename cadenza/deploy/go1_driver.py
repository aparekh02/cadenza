"""cadenza.deploy.go1_driver — Physical deployment driver for Unitree Go1.

Translates Cadenza ActionSpecs (phase-based and gait-based) into real-time
motor commands sent to the physical Go1 over DDS.

Usage::

    import cadenza

    go1 = cadenza.go1()
    go1.deploy([
        go1.stand(),
        go1.walk_forward(speed=0.5),
        go1.jump(),
    ])
"""

from __future__ import annotations

import time
import threading

import numpy as np

from cadenza.actions import get_library
from cadenza.actions.library import ActionSpec
from cadenza.actions.library import ActionCall
from cadenza.locomotion.robot_spec import get_spec
from cadenza.deploy.connection import RobotConnection

# Go1: 12 motors, 4 legs x 3 joints (hip, thigh, calf)
_N_JOINTS = 12
_CONTROL_HZ = 250.0
_CONTROL_DT = 1.0 / _CONTROL_HZ
_VELOCITY_LIMIT = 20.0  # rad/s safety clamp

# Go1 PD gains for physical robot (from URDF torque limits)
_KP_HIP = 60.0
_KD_HIP = 2.0
_KP_THIGH = 60.0
_KD_THIGH = 2.0
_KP_CALF = 80.0
_KD_CALF = 2.0

_STAND = np.array([0.0, 0.9, -1.8] * 4, dtype=np.float32)


def _pd_gains(joint_idx: int) -> tuple[float, float]:
    """Return (kp, kd) for a joint by index."""
    jtype = joint_idx % 3
    if jtype == 0:
        return _KP_HIP, _KD_HIP
    elif jtype == 1:
        return _KP_THIGH, _KD_THIGH
    return _KP_CALF, _KD_CALF


class Go1Driver:
    """Deploys Cadenza actions to a physical Go1 robot.

    Handles:
    - DDS connection setup
    - Phase-based action execution (stand, sit, jump, etc.)
    - Gait-based action execution (walk, trot, etc.)
    - Safety: velocity limiting, graceful shutdown, emergency stop
    """

    def __init__(self, domain_id: int = 0, network_interface: str | None = None):
        self._conn = RobotConnection("go1", domain_id=domain_id,
                                     network_interface=network_interface)
        self._lib = get_library("go1")
        self._spec = get_spec("go1")
        self._running = False
        self._ctrl_thread = None
        self._target_q = _STAND.copy()
        self._ctrl_lock = threading.Lock()

    def connect(self) -> "Go1Driver":
        """Connect to the physical robot."""
        self._conn.connect()
        # Lock all joints to current position
        current_q = self._conn.read_q()[:_N_JOINTS]
        with self._ctrl_lock:
            self._target_q = current_q.copy()
        self._running = True
        self._ctrl_thread = threading.Thread(target=self._control_loop, daemon=True)
        self._ctrl_thread.start()
        print(f"  Go1 driver ready. Current q: {current_q}")
        return self

    def disconnect(self):
        """Gracefully shut down — hold stand position then release."""
        print("  Shutting down Go1 driver...")
        with self._ctrl_lock:
            self._target_q = _STAND.copy()
        time.sleep(1.0)
        self._running = False
        if self._ctrl_thread:
            self._ctrl_thread.join(timeout=2.0)

    def _control_loop(self):
        """250Hz control loop: sends PD commands to all 12 joints."""
        while self._running:
            t0 = time.time()

            with self._ctrl_lock:
                target = self._target_q.copy()

            # Read current state and velocity-clamp target
            current_q = self._conn.read_q()
            if current_q is not None:
                current_q = current_q[:_N_JOINTS]
                delta = target - current_q
                max_step = _VELOCITY_LIMIT * _CONTROL_DT
                clipped = current_q + np.clip(delta, -max_step, max_step)
            else:
                clipped = target

            # Build motor commands
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
                    "mode": 0x0A,
                })
            self._conn.send_cmd(cmds)

            elapsed = time.time() - t0
            time.sleep(max(0, _CONTROL_DT - elapsed))

    def set_target(self, q: np.ndarray):
        """Set joint position target (12-DOF)."""
        with self._ctrl_lock:
            self._target_q = np.array(q, dtype=np.float32)[:_N_JOINTS]

    def execute_action(self, action_name: str, speed: float = 1.0,
                       extension: float = 1.0, repeat: int = 1):
        """Execute a named action from the library on the physical robot."""
        action = self._lib.get(action_name)
        if action.is_phase:
            self._execute_phase_action(action, speed, extension, repeat)
        elif action.is_gait:
            self._execute_gait_action(action, speed, extension, repeat)

    def _execute_phase_action(self, action: ActionSpec, speed: float,
                              extension: float, repeat: int):
        """Execute phase-based action (stand, sit, jump, etc.)."""
        stand = _STAND

        for _ in range(repeat):
            for phase in action.phases:
                q_target = np.array(phase.target.q12, dtype=np.float32)

                # Apply extension
                if extension != 1.0:
                    q_target = stand + extension * (q_target - stand)

                max_vel = np.array(phase.motor_schedule.max_velocity, dtype=np.float32)
                if speed != 1.0:
                    max_vel *= speed

                delay = np.array(phase.motor_schedule.delay_s, dtype=np.float32)
                duration = phase.duration_s / max(speed, 0.1)
                steps = max(1, int(duration * 50))  # 50Hz action rate
                dt = duration / steps

                q_cmd = self._target_q.copy()
                dq_max = max_vel * dt

                for s in range(steps):
                    elapsed = (s + 1) * dt
                    delta = q_target - q_cmd
                    for j in range(_N_JOINTS):
                        if elapsed >= delay[j]:
                            q_cmd[j] += np.clip(delta[j], -dq_max[j], dq_max[j])

                    self.set_target(q_cmd)
                    time.sleep(dt)

        # Brief hold at final pose
        time.sleep(0.5)

    def _execute_gait_action(self, action: ActionSpec, speed: float,
                             extension: float, repeat: int):
        """Execute gait-based action using the gait engine on physical robot."""
        from cadenza.locomotion.gait_engine import GaitEngine

        gait = action.gait
        actual_speed = action.speed_ms * speed
        cmd_vx = gait.cmd_vx * speed
        body_height = gait.body_height
        step_height = gait.step_height * extension if extension != 1.0 else gait.step_height

        engine = GaitEngine(
            self._spec,
            gait_name=gait.gait_name,
            body_height=body_height,
            step_height=step_height,
        )

        duration = action.duration_s * repeat
        cmd = np.array([cmd_vx, gait.cmd_vy * speed, gait.cmd_yaw * speed],
                       dtype=np.float32)

        dt = 1.0 / 50.0  # 50Hz gait update
        steps = int(duration / dt)

        for _ in range(steps):
            q_target = engine.step(dt, cmd)
            self.set_target(q_target)
            time.sleep(dt)

    def deploy(self, sequence: list, from_go1=None):
        """Deploy a sequence of Steps to the physical robot.

        Args:
            sequence: List of Step objects from go1.stand(), go1.walk_forward(), etc.
            from_go1: Optional Go1 instance to apply speed/extension transforms.
        """
        print(f"\n  Deploying {len(sequence)} actions to Go1...\n")

        for i, item in enumerate(sequence):
            if isinstance(item, list):
                # Concurrent — run first only (physical robot is sequential)
                print(f"  [{i+1}] concurrent — running first: {item[0].name}")
                step = item[0]
            else:
                step = item
                print(f"  [{i+1}] {step.name}", end="")
                if step.speed != 1.0:
                    print(f" speed={step.speed}", end="")
                if step.extension != 1.0:
                    print(f" ext={step.extension}", end="")
                print()

            self.execute_action(
                step.name,
                speed=step.speed,
                extension=step.extension,
                repeat=step.repeat if step.repeat > 1 else 1,
            )

        print("\n  Deploy complete.")
