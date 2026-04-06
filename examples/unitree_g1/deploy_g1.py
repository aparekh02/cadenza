"""Deploy G1 — Real robot deployment examples.
   python examples/unitree_g1/deploy_g1.py
"""
import cadenza as cadenza

g1 = cadenza.g1()

# ═══════════════════════════════════════════════════════════════════════════
#  Demo: action sequence for physical G1
# ═══════════════════════════════════════════════════════════════════════════

ROBOT_IP = "" #change to your robot's IP address
SSH_KEY = ""  # change to your key path

# Actions (conservative for humanoid — slower speeds, less extension)
actions = [
    g1.stand(),
    g1.walk_forward(speed=0.3, distance_m=1.0),
    g1.turn_left(speed=0.3),
    g1.walk_forward(speed=0.3, distance_m=1.0),
    g1.lift_left_hand(speed=0.5),
    g1.shake_hand(speed=0.4),
    g1.stand(),
    g1.sit(),
]

if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "sim"

    if mode == "sim":
        # Test in simulation first
        print("Running in simulation (pass 'deploy' or 'bridge' for real robot)\n")
        g1.run(actions)

    elif mode == "deploy":
        # Deploy script to robot via SSH
        g1.deploy_ssh(__file__, host=ROBOT_IP, key=SSH_KEY)

    elif mode == "direct":
        # DDS direct (laptop on robot network)
        g1.deploy(actions)

    elif mode == "bridge":
        # Bridge: VLA on laptop, actions on robot
        bridge = g1.deploy_ssh_bridge(host=ROBOT_IP, key=SSH_KEY)

        for step in actions:
            print(f"  Sending: {step.name}")
            bridge.send_action(step.name, speed=step.speed, extension=step.extension)
            import time; time.sleep(3)

        # Query live telemetry
        t = bridge.query()
        if t:
            print(f"\n  Robot state:")
            print(f"    Joints: {t.joint_q[:6]}...")  # first 6 joints
            print(f"    Status: {t.status}")

        bridge.stop()
        bridge.disconnect()

    else:
        print("Usage: python deploy_g1.py [sim|deploy|direct|bridge]")
