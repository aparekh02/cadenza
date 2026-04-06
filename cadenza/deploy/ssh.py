"""cadenza.deploy.ssh — Deploy Cadenza programs to robots over SSH.

Two deployment modes:

1. **Script mode** — upload and run a .py file directly on the robot.
   Good for self-contained scripts (action sequences, simple demos).

2. **Bridge mode** — start a bridge server on the robot, then control
   it from your host PC. Good for VLA/memory systems that need your GPU.

Usage::

    import cadenza

    go1 = cadenza.go1()

    # Script mode: upload and run
    go1.deploy_ssh("my_demo.py", host="192.168.123.15", key="~/.ssh/go1_rsa")

    # Bridge mode: host controls robot in real-time
    go1.deploy_ssh_bridge(host="192.168.123.15", key="~/.ssh/go1_rsa")
"""

from __future__ import annotations

import subprocess
import time
import threading
import sys
from pathlib import Path
from datetime import datetime

_CADENZA_ROOT = Path(__file__).resolve().parent.parent
_REMOTE_DIR = "/home/unitree/cadenza"
_REMOTE_VENV = f"{_REMOTE_DIR}/.venv"

# Only install what the robot actually needs (lightweight)
_ROBOT_DEPS = "numpy unitree_sdk2py"


class LogStream:
    """Smart log parser for robot output. Colorizes and structures output."""

    STATUS_ICONS = {
        "ok": "+", "abort": "!", "error": "!", "running": ">",
        "done": "+", "skip": "-", "warn": "~",
    }

    def __init__(self, prefix: str = "robot", callback=None):
        self.prefix = prefix
        self.callback = callback
        self._action_count = 0
        self._start_time = time.time()

    def feed(self, line: str):
        """Process one line of robot output."""
        line = line.rstrip()
        if not line:
            return

        elapsed = time.time() - self._start_time
        ts = f"{elapsed:6.1f}s"

        # Detect action lines
        if line.strip().startswith("[") and "/" in line[:20]:
            self._action_count += 1

        # Detect status
        status = ""
        lower = line.lower()
        if "ok" in lower and "moved=" in lower:
            status = "ok"
        elif "abort" in lower:
            status = "abort"
        elif "error" in lower or "traceback" in lower:
            status = "error"
        elif "done" in lower:
            status = "done"

        icon = self.STATUS_ICONS.get(status, " ")
        print(f"  [{ts}] [{icon}] {line}")

        if self.callback:
            self.callback(line, status, elapsed)


class SSHDeploy:
    """Deploy and run Cadenza programs on a robot over SSH."""

    def __init__(self, host: str, user: str = "unitree",
                 key: str | None = None, port: int = 22,
                 password: str | None = None):
        self.host = host
        self.user = user
        self.key = key
        self.port = port
        self.password = password
        self._remote_proc = None

    def _ssh_opts(self) -> list[str]:
        opts = ["-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                "-p", str(self.port)]
        if self.key:
            opts += ["-i", str(Path(self.key).expanduser())]
        return opts

    def _ssh_base(self) -> list[str]:
        ssh = ["ssh"] + self._ssh_opts() + [f"{self.user}@{self.host}"]
        if self.password:
            ssh = ["sshpass", "-p", self.password] + ssh
        return ssh

    def _ssh_cmd(self, cmd: str, timeout: int = 60) -> str:
        """Run a command on the robot. Returns stdout."""
        result = subprocess.run(
            self._ssh_base() + [cmd],
            capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(f"SSH failed: {cmd}\n  stderr: {result.stderr.strip()}")
        return result.stdout.strip()

    def _scp(self, local: str, remote: str, timeout: int = 120):
        """Copy file/dir to robot."""
        scp = ["scp", "-r"] + self._ssh_opts()
        if self.password:
            scp = ["sshpass", "-p", self.password] + scp
        scp += [local, f"{self.user}@{self.host}:{remote}"]
        result = subprocess.run(scp, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            raise RuntimeError(f"SCP failed: {local} -> {remote}\n  {result.stderr.strip()}")

    # ── Connection ────────────────────────────────────────────────────────────

    def test_connection(self) -> bool:
        try:
            return "ok" in self._ssh_cmd("echo ok", timeout=10)
        except Exception:
            return False

    def probe(self) -> dict:
        """Probe the robot for hardware info."""
        info = {}
        try:
            info["hostname"] = self._ssh_cmd("hostname")
            info["arch"] = self._ssh_cmd("uname -m")
            info["python"] = self._ssh_cmd("python3 --version 2>&1 || echo none")
            info["memory_mb"] = self._ssh_cmd("free -m | awk '/Mem:/{print $2}'")
            info["disk_free"] = self._ssh_cmd("df -h /home | tail -1 | awk '{print $4}'")
            info["gpu"] = self._ssh_cmd(
                "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo none"
            )
            info["cadenza_installed"] = "true" in self._ssh_cmd(
                f"test -d {_REMOTE_DIR} && echo true || echo false"
            )
        except Exception as e:
            info["error"] = str(e)
        return info

    # ── Setup ─────────────────────────────────────────────────────────────────

    def setup(self, full: bool = True):
        """Upload cadenza and install deps on the robot.

        Args:
            full: If True, install all deps. If False, only upload code (faster).
        """
        print(f"  Setting up cadenza on {self.user}@{self.host}...")

        self._ssh_cmd(f"mkdir -p {_REMOTE_DIR}")

        # Upload only the cadenza package (not the full repo)
        print("  Uploading cadenza package...")
        self._scp(str(_CADENZA_ROOT), _REMOTE_DIR)

        if not full:
            print("  Quick upload done (skip deps).")
            return

        # Lightweight venv — only what the robot needs
        print("  Installing robot-side dependencies...")
        self._ssh_cmd(
            f"cd {_REMOTE_DIR} && "
            f"python3 -m venv {_REMOTE_VENV} 2>/dev/null; "
            f"{_REMOTE_VENV}/bin/pip install -q --upgrade pip && "
            f"{_REMOTE_VENV}/bin/pip install -q {_ROBOT_DEPS}",
            timeout=300,
        )
        print("  Setup complete.")

    # ── Script deploy ─────────────────────────────────────────────────────────

    def upload_script(self, script: str | Path):
        script = Path(script)
        if not script.exists():
            raise FileNotFoundError(f"Script not found: {script}")
        print(f"  Uploading {script.name}...")
        self._scp(str(script), f"{_REMOTE_DIR}/{script.name}")

    def run_remote(self, script: str | Path, background: bool = False,
                   log_callback=None) -> str:
        """Run a script on the robot with smart log streaming.

        Args:
            script: Local script to upload and run.
            background: Run detached (returns immediately).
            log_callback: Optional fn(line, status, elapsed) for custom handling.
        """
        script = Path(script)
        if script.exists():
            self.upload_script(script)

        remote_script = f"{_REMOTE_DIR}/{script.name}"
        python = f"{_REMOTE_VENV}/bin/python"
        env = f"KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH={_REMOTE_DIR}"

        if background:
            log_file = f"/tmp/cadenza_{script.stem}.log"
            self._ssh_cmd(
                f"cd {_REMOTE_DIR} && "
                f"nohup {env} {python} {remote_script} > {log_file} 2>&1 &"
            )
            print(f"  Running {script.name} in background")
            print(f"  Stream logs: ssh {self.user}@{self.host} 'tail -f {log_file}'")
            return ""

        print(f"  Running {script.name} on {self.host}...\n")
        cmd = f"cd {_REMOTE_DIR} && {env} {python} -u {remote_script} 2>&1"
        proc = subprocess.Popen(
            self._ssh_base() + [cmd],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        self._remote_proc = proc

        log = LogStream(prefix=self.host, callback=log_callback)
        output = []
        try:
            for line in proc.stdout:
                log.feed(line)
                output.append(line)
        except KeyboardInterrupt:
            print("\n  Ctrl+C — stopping remote process...")
            self.stop_remote()
        finally:
            proc.wait()
            self._remote_proc = None

        return "".join(output)

    def stop_remote(self):
        """Kill cadenza processes on the robot."""
        try:
            self._ssh_cmd(f"pkill -f '{_REMOTE_DIR}' 2>/dev/null || true", timeout=5)
        except Exception:
            pass
        if self._remote_proc:
            self._remote_proc.terminate()
        print(f"  Stopped cadenza on {self.host}")

    def tail_logs(self, script_name: str = ""):
        """Stream logs from a background process."""
        log_file = f"/tmp/cadenza_{script_name}.log" if script_name else "/tmp/cadenza*.log"
        cmd = f"tail -f {log_file} 2>/dev/null"
        proc = subprocess.Popen(
            self._ssh_base() + [cmd],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        log = LogStream(prefix=self.host)
        try:
            for line in proc.stdout:
                log.feed(line)
        except KeyboardInterrupt:
            pass
        finally:
            proc.terminate()

    # ── Bridge deploy ─────────────────────────────────────────────────────────

    def start_bridge(self, robot: str = "go1") -> "HostBridge":
        """Start a bridge server on the robot and connect to it.

        This starts the lightweight bridge process on the robot, then
        returns a HostBridge you can use to send commands from your host
        (where the VLA model / heavy compute runs).

        Args:
            robot: "go1" or "g1"

        Returns:
            HostBridge connected to the robot.
        """
        from cadenza.deploy.bridge import HostBridge

        # Generate and upload bridge runner script
        bridge_script = (
            "import sys; sys.path.insert(0, '/home/unitree/cadenza')\n"
            "from cadenza.deploy.bridge import RobotBridge\n"
            f"bridge = RobotBridge('{robot}')\n"
            "bridge.serve()\n"
        )
        self._ssh_cmd(f"cat > {_REMOTE_DIR}/_bridge.py << 'PYEOF'\n{bridge_script}PYEOF")

        # Start bridge in background
        python = f"{_REMOTE_VENV}/bin/python"
        env = f"KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH={_REMOTE_DIR}"
        self._ssh_cmd(
            f"cd {_REMOTE_DIR} && "
            f"nohup {env} {python} -u {_REMOTE_DIR}/_bridge.py "
            f"> /tmp/cadenza_bridge.log 2>&1 &"
        )
        print(f"  Bridge started on {self.host}")

        # Wait for bridge to be ready
        time.sleep(2.0)

        # Connect from host side
        bridge = HostBridge(self.host)
        bridge.connect()
        return bridge

    # ── One-call deploy ───────────────────────────────────────────────────────

    def deploy_and_run(self, script: str | Path, setup: bool = True,
                       background: bool = False, log_callback=None) -> str:
        """Upload cadenza + script, install deps, run.

        Args:
            script: Local .py file.
            setup: Full setup on first deploy (True). Use False after first time.
            background: Run detached.
            log_callback: Optional fn(line, status, elapsed).
        """
        print(f"\n  Deploying to {self.user}@{self.host}...")
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"  Time: {ts}\n")

        if not self.test_connection():
            raise ConnectionError(
                f"Cannot reach {self.host}.\n"
                f"  - Robot powered on?\n"
                f"  - IP correct? (Go1: 192.168.123.15, G1: 192.168.123.164)\n"
                f"  - Key valid? {self.key or '(default)'}"
            )

        # Probe hardware
        info = self.probe()
        print(f"  Robot: {info.get('hostname', '?')} ({info.get('arch', '?')})")
        print(f"  RAM: {info.get('memory_mb', '?')}MB  Disk: {info.get('disk_free', '?')}")
        gpu = info.get("gpu", "none")
        if gpu != "none":
            print(f"  GPU: {gpu}")
        else:
            print(f"  GPU: none (VLA must run on host)")
        print()

        if setup:
            self.setup(full=not info.get("cadenza_installed", False))

        return self.run_remote(script, background=background, log_callback=log_callback)
