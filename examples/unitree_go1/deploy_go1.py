"""Deploy Go1 — Real robot deployment examples.
   python examples/unitree_go1/deploy_go1.py
"""
import cadenza_local as cadenza

go1 = cadenza.go1()

# ── Option 1: Script mode ─────────────────────────────────────────────────
# Upload this file and run it directly on the robot.
# Everything executes on the robot's onboard computer.
#
#   go1.deploy_ssh(
#       "examples/unitree_go1/deploy_go1.py",
#       host="192.168.123.15",
#       key="~/.ssh/go1_rsa",
#   )

# ── Option 2: Direct deploy (DDS, same network) ──────────────────────────
# Run from a laptop connected to the robot's network.
# Sends motor commands directly over DDS — no SSH needed.
#
#   go1.deploy([
#       go1.stand(),
#       go1.walk_forward(speed=0.5),
#       go1.sit(),
#   ])

# ── Option 3: Bridge mode (VLA on laptop, actions on robot) ──────────────
# Best for real applications. Your laptop runs the heavy model,
# the robot runs the lightweight action engine.
#
#   bridge = go1.deploy_ssh_bridge(
#       host="192.168.123.15",
#       key="~/.ssh/go1_rsa",
#   )
#
#   # VLA inference loop on your laptop GPU
#   while True:
#       state = bridge.telemetry
#       if state and state.joint_q:
#           action = my_vla_model(state)
#           bridge.send_action(action, speed=0.5)
#
#   bridge.estop()

# ═══════════════════════════════════════════════════════════════════════════
#  Demo: action sequence for physical Go1
# ═══════════════════════════════════════════════════════════════════════════

ROBOT_IP = "192.168.123.15"
SSH_KEY = "~/.ssh/go1_rsa"  # change to your key path

# Actions to run on the real robot (conservative speeds for safety)
actions = [
    go1.stand(),
    go1.walk_forward(speed=0.4, distance_m=1.0),
    go1.turn_left(speed=0.5),
    go1.walk_forward(speed=0.4, distance_m=1.0),
    go1.turn_right(speed=0.5),
    go1.jump(speed=0.6, extension=0.7),
    go1.sit(),
]

if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "sim"

    if mode == "sim":
        # Test in simulation first
        print("Running in simulation (pass 'deploy' or 'bridge' for real robot)\n")
        go1.run(actions)

    elif mode == "deploy":
        # Deploy script to robot via SSH
        go1.deploy_ssh(__file__, host=ROBOT_IP, key=SSH_KEY)

    elif mode == "direct":
        # DDS direct (laptop on robot network)
        go1.deploy(actions)

    elif mode == "bridge":
        # Bridge: VLA on laptop, actions on robot
        bridge = go1.deploy_ssh_bridge(host=ROBOT_IP, key=SSH_KEY)

        for step in actions:
            print(f"  Sending: {step.name}")
            bridge.send_action(step.name, speed=step.speed, extension=step.extension)
            import time; time.sleep(3)

        bridge.stop()
        bridge.disconnect()

    else:
        print("Usage: python deploy_go1.py [sim|deploy|direct|bridge]")
