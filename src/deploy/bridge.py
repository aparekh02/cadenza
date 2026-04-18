"""cadenza.deploy.bridge — Host <-> Robot communication bridge.

Split architecture:
  HOST PC (your laptop)          ROBOT (Jetson/RK3588)
  ├─ VLA model                   ├─ cadenza action library
  ├─ Memory system               ├─ gait engine
  ├─ Heavy inference             ├─ DDS motor control
  └─ bridge.HostBridge ──TCP──── └─ bridge.RobotBridge

The host sends high-level commands (action names + params).
The robot executes them on hardware and streams back telemetry.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from dataclasses import dataclass, asdict

_BRIDGE_PORT = 9737
_BUFFER_SIZE = 4096


@dataclass
class RobotTelemetry:
    """Telemetry packet streamed from robot to host."""
    timestamp: float = 0.0
    joint_q: list[float] | None = None
    joint_dq: list[float] | None = None
    body_pos: list[float] | None = None    # [x, y, z]
    body_rpy: list[float] | None = None    # [roll, pitch, yaw]
    foot_contacts: list[bool] | None = None
    battery_pct: float = -1.0
    action_name: str = ""
    action_phase: str = ""
    action_progress: float = 0.0           # 0.0 to 1.0
    status: str = "idle"                   # idle, running, error, done
    error: str = ""
    log: str = ""


@dataclass
class HostCommand:
    """Command packet sent from host to robot."""
    type: str = "action"         # action, stop, estop, query, set_gains
    action_name: str = ""
    speed: float = 1.0
    extension: float = 1.0
    repeat: int = 1
    params: dict | None = None   # extra params (gains, etc.)


class RobotBridge:
    """Runs on the robot. Listens for commands, executes actions, streams telemetry.

    Usage (robot-side script)::

        from cadenza.deploy.bridge import RobotBridge
        bridge = RobotBridge("go1")
        bridge.serve()  # blocks, listens for host commands
    """

    def __init__(self, robot: str = "go1", port: int = _BRIDGE_PORT):
        self.robot = robot
        self.port = port
        self._driver = None
        self._server = None
        self._client = None
        self._running = False
        self._telemetry_thread = None

    def serve(self):
        """Start the bridge server. Blocks until stopped."""
        # Lazy import — only the driver needed on robot side
        if self.robot in ("go1", "go2"):
            from cadenza.deploy.go1_driver import Go1Driver
            self._driver = Go1Driver()
        else:
            from cadenza.deploy.g1_driver import G1Driver
            self._driver = G1Driver()

        self._driver.connect()
        self._running = True

        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind(("0.0.0.0", self.port))
        self._server.listen(1)
        print(f"  Bridge listening on port {self.port}...")

        try:
            while self._running:
                self._server.settimeout(1.0)
                try:
                    conn, addr = self._server.accept()
                except socket.timeout:
                    continue
                print(f"  Host connected: {addr}")
                self._client = conn
                self._handle_client(conn)
        finally:
            self._driver.disconnect()
            self._server.close()

    def _handle_client(self, conn: socket.socket):
        """Handle one host connection."""
        # Start telemetry streaming
        self._telemetry_thread = threading.Thread(
            target=self._stream_telemetry, args=(conn,), daemon=True)
        self._telemetry_thread.start()

        try:
            while self._running:
                data = conn.recv(_BUFFER_SIZE)
                if not data:
                    break
                # Commands can be batched with newlines
                for line in data.decode().strip().split("\n"):
                    if not line:
                        continue
                    cmd = HostCommand(**json.loads(line))
                    self._execute_command(cmd, conn)
        except (ConnectionResetError, BrokenPipeError):
            print("  Host disconnected.")
        finally:
            conn.close()
            self._client = None

    def _execute_command(self, cmd: HostCommand, conn: socket.socket):
        """Execute a command from the host."""
        if cmd.type == "estop":
            self._send_telemetry(conn, status="estop", log="EMERGENCY STOP")
            self._driver.disconnect()
            self._running = False
            return

        if cmd.type == "stop":
            self._driver.set_target(
                __import__("numpy").array([0.0, 0.9, -1.8] * 4, dtype="float32"))
            self._send_telemetry(conn, status="idle", log="Stopped")
            return

        if cmd.type == "query":
            self._send_telemetry(conn, status="idle")
            return

        if cmd.type == "action":
            self._send_telemetry(conn, status="running",
                                 action_name=cmd.action_name, log=f"Starting {cmd.action_name}")
            try:
                self._driver.execute_action(
                    cmd.action_name,
                    speed=cmd.speed,
                    extension=cmd.extension,
                    repeat=cmd.repeat,
                )
                self._send_telemetry(conn, status="done",
                                     action_name=cmd.action_name,
                                     action_progress=1.0,
                                     log=f"Completed {cmd.action_name}")
            except Exception as e:
                self._send_telemetry(conn, status="error",
                                     error=str(e), log=f"FAILED {cmd.action_name}: {e}")

    def _stream_telemetry(self, conn: socket.socket):
        """Stream telemetry at 10Hz."""
        while self._running and self._client:
            try:
                self._send_telemetry(conn)
            except (BrokenPipeError, ConnectionResetError, OSError):
                break
            time.sleep(0.1)

    def _send_telemetry(self, conn: socket.socket, **overrides):
        """Build and send a telemetry packet."""
        t = RobotTelemetry(timestamp=time.time())

        # Read live state from driver
        state = self._driver._conn.read_state() if self._driver._conn.connected else None
        if state:
            t.joint_q = [s.q for s in state[:12]]
            t.joint_dq = [s.dq for s in state[:12]]

        # Apply overrides
        for k, v in overrides.items():
            setattr(t, k, v)

        try:
            msg = json.dumps(asdict(t)) + "\n"
            conn.sendall(msg.encode())
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def stop(self):
        self._running = False


class HostBridge:
    """Runs on the host PC. Connects to the robot bridge, sends commands,
    receives telemetry.

    Usage::

        from cadenza.deploy.bridge import HostBridge

        bridge = HostBridge("192.168.123.15")
        bridge.connect()
        bridge.send_action("walk_forward", speed=0.5)
        bridge.send_action("jump")
        bridge.estop()  # emergency stop
    """

    def __init__(self, host: str, port: int = _BRIDGE_PORT):
        self.host = host
        self.port = port
        self._sock = None
        self._running = False
        self._recv_thread = None
        self._telemetry: RobotTelemetry | None = None
        self._telemetry_lock = threading.Lock()
        self._log_callback = None

    def connect(self, log_callback=None):
        """Connect to the robot bridge.

        Args:
            log_callback: Optional function(RobotTelemetry) called on each telemetry update.
                          Use for custom dashboards or logging.
        """
        self._log_callback = log_callback
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.settimeout(10.0)
        try:
            self._sock.connect((self.host, self.port))
        except (socket.timeout, ConnectionRefusedError) as e:
            raise ConnectionError(
                f"Cannot connect to bridge at {self.host}:{self.port}. "
                f"Is the robot bridge running?\n  Error: {e}"
            )
        self._sock.settimeout(None)
        self._running = True

        self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._recv_thread.start()
        print(f"  Connected to robot bridge at {self.host}:{self.port}")

    def _recv_loop(self):
        """Receive telemetry packets."""
        buf = ""
        while self._running:
            try:
                data = self._sock.recv(_BUFFER_SIZE)
                if not data:
                    break
                buf += data.decode()
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    if not line:
                        continue
                    t = RobotTelemetry(**json.loads(line))
                    with self._telemetry_lock:
                        self._telemetry = t

                    # Smart log output
                    if t.log:
                        self._print_log(t)

                    if self._log_callback:
                        self._log_callback(t)
            except (ConnectionResetError, BrokenPipeError, OSError):
                break
        print("  Bridge connection closed.")

    def _print_log(self, t: RobotTelemetry):
        """Smart formatted log output."""
        icon = {"idle": " ", "running": ">", "done": "+",
                "error": "!", "estop": "X"}.get(t.status, "?")

        line = f"  [{icon}]"
        if t.action_name:
            pct = f" {t.action_progress:.0%}" if t.action_progress > 0 else ""
            line += f" {t.action_name}{pct}"
        if t.joint_q:
            z = t.body_pos[2] if t.body_pos else 0
            line += f"  z={z:.3f}" if z else ""
        if t.log:
            line += f"  | {t.log}"
        if t.error:
            line += f"  !! {t.error}"

        print(line)

    def _send(self, cmd: HostCommand):
        msg = json.dumps(asdict(cmd)) + "\n"
        self._sock.sendall(msg.encode())

    def send_action(self, action_name: str, speed: float = 1.0,
                    extension: float = 1.0, repeat: int = 1):
        """Send an action command to the robot."""
        self._send(HostCommand(
            type="action", action_name=action_name,
            speed=speed, extension=extension, repeat=repeat,
        ))

    def stop(self):
        """Stop current action (graceful)."""
        self._send(HostCommand(type="stop"))

    def estop(self):
        """Emergency stop — kills motor control immediately."""
        self._send(HostCommand(type="estop"))
        print("  !! EMERGENCY STOP sent")

    def query(self) -> RobotTelemetry | None:
        """Get latest telemetry."""
        self._send(HostCommand(type="query"))
        time.sleep(0.15)
        with self._telemetry_lock:
            return self._telemetry

    @property
    def telemetry(self) -> RobotTelemetry | None:
        with self._telemetry_lock:
            return self._telemetry

    def disconnect(self):
        self._running = False
        if self._sock:
            self._sock.close()
