"""cadenza.deploy.connection — DDS connection to physical Unitree robots.

Wraps unitree_sdk2py's ChannelFactory, publisher/subscriber, and CRC.
Handles both Go1/Go2 (unitree_go) and G1/H1 (unitree_hg) message types.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np


@dataclass
class MotorState:
    q: float = 0.0
    dq: float = 0.0
    tau: float = 0.0


class DataBuffer:
    """Thread-safe buffer for latest robot state."""

    def __init__(self):
        self._data = None
        self._lock = threading.Lock()

    def get(self):
        with self._lock:
            return self._data

    def set(self, data):
        with self._lock:
            self._data = data


class RobotConnection:
    """DDS connection to a physical Unitree robot.

    Args:
        robot: "go1", "go2", "g1", or "h1"
        domain_id: 0 for real robot, 1 for simulation
        network_interface: Network interface for DDS (None = default)
    """

    # DDS topics
    TOPIC_LOWCMD = "rt/lowcmd"
    TOPIC_LOWSTATE = "rt/lowstate"

    def __init__(self, robot: str, domain_id: int = 0,
                 network_interface: str | None = None):
        self.robot = robot.lower()
        self.domain_id = domain_id
        self._network_interface = network_interface
        self._initialized = False
        self._publisher = None
        self._subscriber = None
        self._state_buffer = DataBuffer()
        self._crc = None
        self._msg = None
        self._sub_thread = None

    def connect(self) -> "RobotConnection":
        """Initialize DDS and connect to the robot. Returns self for chaining."""
        from unitree_sdk2py.core.channel import (
            ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize,
        )
        from unitree_sdk2py.utils.crc import CRC

        ChannelFactoryInitialize(self.domain_id, self._network_interface)
        self._crc = CRC()

        if self.robot in ("go1", "go2"):
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import (
                LowCmd_ as LowCmd, LowState_ as LowState,
            )
            from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
            self._msg = unitree_go_msg_dds__LowCmd_()
            self._msg.head[0] = 0xFE
            self._msg.head[1] = 0xEF
            self._msg.level_flag = 0xFF
            self._msg.gpio = 0
        else:
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import (
                LowCmd_ as LowCmd, LowState_ as LowState,
            )
            from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
            self._msg = unitree_hg_msg_dds__LowCmd_()
            self._msg.mode_pr = 0
            self._msg.mode_machine = 0

        self._publisher = ChannelPublisher(self.TOPIC_LOWCMD, LowCmd)
        self._publisher.Init()
        self._subscriber = ChannelSubscriber(self.TOPIC_LOWSTATE, LowState)
        self._subscriber.Init()

        self._n_motors = self._detect_motor_count()

        # Start state subscription thread
        self._sub_thread = threading.Thread(target=self._subscribe_loop, daemon=True)
        self._sub_thread.start()

        # Wait for first state
        t0 = time.time()
        while self._state_buffer.get() is None:
            if time.time() - t0 > 10.0:
                raise TimeoutError(f"No response from {self.robot} after 10s. Check connection.")
            time.sleep(0.1)

        self._initialized = True
        print(f"  Connected to {self.robot} (domain={self.domain_id}, {self._n_motors} motors)")
        return self

    def _detect_motor_count(self) -> int:
        counts = {"go1": 12, "go2": 12, "g1": 35, "h1": 20}
        return counts.get(self.robot, 12)

    def _subscribe_loop(self):
        while True:
            msg = self._subscriber.Read()
            if msg is not None:
                states = []
                for i in range(self._n_motors):
                    ms = MotorState(
                        q=msg.motor_state[i].q,
                        dq=msg.motor_state[i].dq,
                        tau=getattr(msg.motor_state[i], 'tau_est', 0.0),
                    )
                    states.append(ms)
                self._state_buffer.set(states)
            time.sleep(0.002)

    def read_state(self) -> list[MotorState] | None:
        """Get latest motor states. Returns None if not connected."""
        return self._state_buffer.get()

    def read_q(self) -> np.ndarray | None:
        """Get current joint positions as array."""
        states = self.read_state()
        if states is None:
            return None
        return np.array([s.q for s in states], dtype=np.float32)

    def send_cmd(self, motor_cmds: list[dict]):
        """Send motor commands.

        Args:
            motor_cmds: List of dicts with keys: id, q, dq, kp, kd, tau, mode
        """
        if not self._initialized:
            raise RuntimeError("Not connected. Call connect() first.")

        for cmd in motor_cmds:
            i = cmd["id"]
            mc = self._msg.motor_cmd[i]
            mc.mode = cmd.get("mode", 0x0A)
            mc.q = cmd.get("q", 0.0)
            mc.dq = cmd.get("dq", 0.0)
            mc.kp = cmd.get("kp", 0.0)
            mc.kd = cmd.get("kd", 0.0)
            mc.tau = cmd.get("tau", 0.0)

        self._msg.crc = self._crc.Crc(self._msg)
        self._publisher.Write(self._msg)

    @property
    def n_motors(self) -> int:
        return self._n_motors

    @property
    def connected(self) -> bool:
        return self._initialized
