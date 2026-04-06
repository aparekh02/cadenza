"""VLA Guardian — obstacle course demo + tests.

This IS the demo. Run it and the Go1 walks 9m through an obstacle course
with VLA active. It MUST navigate around all 3 obstacles and reach the goal.

Usage:
    mjpython tests/test_vla_obstacle.py
"""

import sys
import os
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

OBSTACLE_SCENE = str(
    Path(__file__).resolve().parent.parent / "cadenza" / "models" / "go1" / "obstacle_scene.xml"
)


def run_obstacle_course():
    """Go1 walks 9m through obstacle course with VLA guardian active.

    Obstacle layout (robot walks toward -x):
      - Obstacle 1: Cardboard box at x=-2.0 (center)
      - Obstacle 2: Concrete barrier at x=-4.5 (left of center)
      - Obstacle 3: Traffic cone at x=-7.0 (right of center)
      - Goal pad at x=-9.0

    The VLA guardian uses physics raycasts to detect obstacles and
    SmolVLM-256M to judge size and plan avoidance. The robot MUST
    navigate around all obstacles and reach the goal.
    """
    import cadenza

    print("\n" + "=" * 60)
    print("  CADENZA VLA OBSTACLE COURSE")
    print("  Go1 → 9m forward → 3 obstacles → goal")
    print("  Detection: MuJoCo raycasts (physics-accurate)")
    print("  Planning:  SmolVLM-256M-Instruct")
    print("=" * 60 + "\n")

    go1 = cadenza.go1(xml_path=OBSTACLE_SCENE)

    go1.run([
        go1.stand(),
        go1.walk_forward(speed=1.0, distance_m=90.0),
        go1.stand(),
    ], vla=True)


# ── Unit tests (run with --test flag) ────────────────────────────────────────

def test_guardian_import():
    from cadenza.vla import VLAGuardian, ObstacleResult
    guardian = VLAGuardian("go1")
    assert guardian.robot == "go1"
    print("  PASS: VLAGuardian imports")


def test_obstacle_result():
    from cadenza.vla.guardian import ObstacleResult
    result = ObstacleResult(detected=True, position="left", size="large")
    assert result.detected is True
    assert result.avoidance_actions == []
    print("  PASS: ObstacleResult dataclass")


def test_parse_plan():
    from cadenza.vla.guardian import VLAGuardian
    guardian = VLAGuardian("go1")

    result = guardian._parse_plan("SIZE: LARGE", "center", 0.5)
    assert result.detected is True
    assert len(result.avoidance_actions) == 7  # full U-shaped detour

    result = guardian._parse_plan("SIZE: SMALL", "center", 0.4)
    assert result.size == "small"
    assert result.avoidance_actions[0].name == "crawl_forward"

    result = guardian._parse_plan("SIZE: LARGE", "left", 0.6)
    assert result.avoidance_actions[0].name == "turn_right"  # left → go right

    result = guardian._parse_plan("SIZE: LARGE", "right", 0.6)
    assert result.avoidance_actions[0].name == "turn_left"  # right → go left

    print("  PASS: Plan parsing")


def test_raycast_detection():
    import mujoco
    from cadenza.vla.guardian import VLAGuardian

    model = mujoco.MjModel.from_xml_path(OBSTACLE_SCENE)
    data = mujoco.MjData(model)

    data.qpos[2] = 0.27
    data.qpos[3] = 1.0
    stand = np.array([0.0, 0.9, -1.8] * 4, dtype=np.float64)
    data.qpos[7:19] = stand
    mujoco.mj_forward(model, data)
    for _ in range(200):
        data.ctrl[:] = stand
        mujoco.mj_step(model, data)

    guardian = VLAGuardian("go1")

    # Far from obstacle — should NOT detect
    data.qpos[0] = -0.5
    mujoco.mj_forward(model, data)
    detected, dist, pos = guardian.check_raycast_only(model, data)
    assert not detected, f"False positive at x=-0.5 (dist={dist})"

    # Close to obstacle 1 (box at x=-2.0, front face ~x=-1.85) — MUST detect
    data.qpos[0] = -1.3
    mujoco.mj_forward(model, data)
    detected, dist, pos = guardian.check_raycast_only(model, data)
    assert detected, "Failed to detect obstacle at 0.55m"
    assert dist < 0.75
    assert pos == "center"

    print(f"  PASS: Raycast detection (dist={dist:.2f}m, pos={pos})")


def test_camera_rendering():
    import mujoco
    from cadenza.vla.guardian import VLAGuardian

    model = mujoco.MjModel.from_xml_path(OBSTACLE_SCENE)
    data = mujoco.MjData(model)

    data.qpos[2] = 0.27
    data.qpos[3] = 1.0
    stand = np.array([0.0, 0.9, -1.8] * 4, dtype=np.float64)
    data.qpos[7:19] = stand
    mujoco.mj_forward(model, data)
    for _ in range(200):
        data.ctrl[:] = stand
        mujoco.mj_step(model, data)

    guardian = VLAGuardian("go1")
    frame = guardian._render_camera(model, data)
    assert frame.shape == (384, 384, 3)
    assert frame.max() > 0
    print(f"  PASS: Camera rendering ({frame.shape})")


def run_tests():
    print("\n  VLA Guardian Unit Tests\n")
    test_guardian_import()
    test_obstacle_result()
    test_parse_plan()
    test_raycast_detection()
    test_camera_rendering()
    print("\n  All tests passed!\n")


if __name__ == "__main__":
    if "--test" in sys.argv:
        run_tests()
    else:
        run_obstacle_course()
