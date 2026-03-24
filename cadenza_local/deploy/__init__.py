"""cadenza.deploy — Deploy actions to physical Unitree robots.

Two deployment modes:

1. **Script mode** — upload a .py and run it on the robot.
2. **Bridge mode** — robot runs lightweight controller, host runs VLA/memory.

Script mode::

    go1 = cadenza.go1()
    go1.deploy_ssh("my_demo.py", host="192.168.123.15", key="~/.ssh/go1_rsa")

Bridge mode (VLA on host, actions on robot)::

    go1 = cadenza.go1()
    bridge = go1.deploy_ssh_bridge(host="192.168.123.15", key="~/.ssh/go1_rsa")
    bridge.send_action("walk_forward", speed=0.5)
    bridge.send_action("jump")
    print(bridge.telemetry)
    bridge.estop()
"""

from cadenza_local.deploy.connection import RobotConnection
from cadenza_local.deploy.go1_driver import Go1Driver
from cadenza_local.deploy.g1_driver import G1Driver
from cadenza_local.deploy.ssh import SSHDeploy
from cadenza_local.deploy.bridge import HostBridge, RobotBridge, RobotTelemetry, HostCommand
