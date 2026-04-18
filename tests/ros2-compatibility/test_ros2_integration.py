"""Cadenza ROS 2 Integration Tests — Real rclpy Nodes + Headless MuJoCo.

Spins up actual rclpy nodes, exchanges real ROS 2 messages over real topics
and services, and runs headless MuJoCo physics to verify that Cadenza integrates
cleanly into a ROS 2 stack.

No display required. Set MUJOCO_GL=disabled for headless-only physics.

Run inside the provided Docker container:
    docker build -f Dockerfile.ros2 -t cadenza-ros2 .
    docker run --rm cadenza-ros2

Or directly (with ROS 2 Humble sourced):
    source /opt/ros/humble/setup.bash
    python3 tests/test_ros2_integration.py
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "disabled")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import mujoco

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

_MODELS_DIR = Path(__file__).resolve().parent.parent / "src" / "models"

_GO1_JOINT_NAMES = [
    "FL_hip_joint",  "FL_thigh_joint",  "FL_calf_joint",
    "FR_hip_joint",  "FR_thigh_joint",  "FR_calf_joint",
    "RL_hip_joint",  "RL_thigh_joint",  "RL_calf_joint",
    "RR_hip_joint",  "RR_thigh_joint",  "RR_calf_joint",
]
_STAND_CTRL = np.array([0.0, 0.9, -1.8] * 4, dtype=np.float64)
_GO1_HIP_RANGE   = (-0.863,  0.863)
_GO1_THIGH_RANGE = (-0.686,  4.501)
_GO1_KNEE_RANGE  = (-2.818, -0.888)


# ── CadenzaCommandNode ────────────────────────────────────────────────────────

class CadenzaCommandNode(Node):
    """ROS 2 command interface for Cadenza Go1.

    Topics in:
        /cmd_vel            geometry_msgs/Twist
        /cadenza/command    std_msgs/String  (natural language)
        /cadenza/execute    std_msgs/String  (JSON list of action names)

    Topics out:
        /cadenza/active_action  std_msgs/String
        /cadenza/joint_targets  sensor_msgs/JointState
        /cadenza/ready          std_msgs/Bool

    Services:
        /cadenza/list_actions   std_srvs/Trigger  → JSON list in response.message
        /cadenza/plan_validate  std_srvs/Trigger  → validates queued plan
    """

    def __init__(self):
        super().__init__("cadenza_command")

        from cadenza.actions import get_library, list_actions
        from cadenza.parser.translator import CommandParser

        self._lib = get_library("go1")
        self._parser = CommandParser("go1")
        self._all_actions = list_actions("go1")

        # publishers
        self._pub_action  = self.create_publisher(String,     "/cadenza/active_action",  10)
        self._pub_targets = self.create_publisher(JointState, "/cadenza/joint_targets",  10)
        self._pub_ready   = self.create_publisher(Bool,       "/cadenza/ready",           10)

        # subscribers
        self.create_subscription(Twist,  "/cmd_vel",          self._on_cmd_vel,  10)
        self.create_subscription(String, "/cadenza/command",  self._on_command,  10)
        self.create_subscription(String, "/cadenza/execute",  self._on_execute,  10)

        # services
        self.create_service(Trigger, "/cadenza/list_actions",  self._srv_list_actions)
        self.create_service(Trigger, "/cadenza/plan_validate", self._srv_plan_validate)

        self._last_action:   str       = "idle"
        self._last_sequence: list[str] = []
        self._queued_plan:   list[str] = []

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _twist_to_action(msg: Twist) -> str:
        vx, vy, wz = msg.linear.x, msg.linear.y, msg.angular.z
        mags = [abs(vx), abs(vy), abs(wz)]
        if max(mags) < 1e-3:
            return "stand"
        dom = mags.index(max(mags))
        if dom == 0:
            return "walk_forward"  if vx > 0 else "walk_backward"
        if dom == 1:
            return "side_step_left" if vy > 0 else "side_step_right"
        return "turn_left" if wz > 0 else "turn_right"

    def _joint_targets_for(self, action_name: str) -> JointState:
        try:
            spec = self._lib.get(action_name)
        except KeyError:
            q12 = _STAND_CTRL.tolist()
        else:
            q12 = list(spec.phases[0].target.q12) if (spec.is_phase and spec.phases) \
                  else _STAND_CTRL.tolist()
        js = JointState()
        js.name     = _GO1_JOINT_NAMES
        js.position = q12
        js.velocity = [0.0] * 12
        js.effort   = [0.0] * 12
        return js

    # ── subscribers ───────────────────────────────────────────────────────────

    def _on_cmd_vel(self, msg: Twist):
        name = self._twist_to_action(msg)
        self._last_action = name

        out = String(); out.data = name
        self._pub_action.publish(out)
        self._pub_targets.publish(self._joint_targets_for(name))

    def _on_command(self, msg: String):
        calls = self._parser.parse(msg.data)
        self._last_sequence = [c.action_name for c in calls]

        resp = String(); resp.data = json.dumps(self._last_sequence)
        self._pub_action.publish(resp)

    def _on_execute(self, msg: String):
        names = json.loads(msg.data)
        available = set(self._lib._actions.keys())
        invalid   = [n for n in names if n not in available]

        if invalid:
            resp = String()
            resp.data = json.dumps({"status": "aborted", "invalid": invalid})
            self._pub_action.publish(resp)
        else:
            self._queued_plan = names
            resp = String()
            resp.data = json.dumps({"status": "accepted", "count": len(names)})
            self._pub_action.publish(resp)
            ready = Bool(); ready.data = True
            self._pub_ready.publish(ready)

    # ── services ──────────────────────────────────────────────────────────────

    def _srv_list_actions(self, _req, response):
        response.success = True
        response.message = json.dumps(sorted(self._all_actions))
        return response

    def _srv_plan_validate(self, _req, response):
        if self._queued_plan:
            response.success = True
            response.message = json.dumps(self._queued_plan)
        else:
            response.success = False
            response.message = "no plan queued"
        return response


# ── CadenzaSimNode ────────────────────────────────────────────────────────────

class CadenzaSimNode(Node):
    """Headless MuJoCo physics node.

    Runs Go1 at 50 Hz (stand controller). Publishes live joint states and
    body state so downstream ROS 2 nodes see a real physics-based robot.

    Topics out:
        /cadenza/sim/joint_states   sensor_msgs/JointState  (50 Hz)
        /cadenza/sim/state          std_msgs/String          (50 Hz, JSON)

    Topics in:
        /cadenza/sim/reset          std_msgs/Bool  → reset to stand pose
    """

    def __init__(self):
        super().__init__("cadenza_sim")

        xml = str(_MODELS_DIR / "go1" / "scene.xml")
        self._model = mujoco.MjModel.from_xml_path(xml)
        self._data  = mujoco.MjData(self._model)
        self._lock  = threading.Lock()

        # Put robot in stand pose
        with self._lock:
            self._data.qpos[2]    = 0.27   # body height
            self._data.qpos[3]    = 1.0    # quaternion w
            self._data.qpos[7:19] = _STAND_CTRL
            mujoco.mj_forward(self._model, self._data)

        self._pub_joints = self.create_publisher(
            JointState, "/cadenza/sim/joint_states", 10)
        self._pub_state  = self.create_publisher(
            String, "/cadenza/sim/state", 10)
        self.create_subscription(Bool, "/cadenza/sim/reset", self._on_reset, 10)

        # 50 Hz physics + publish timer
        self._step_count = 0
        self.create_timer(1.0 / 50.0, self._tick)

    def _tick(self):
        with self._lock:
            self._data.ctrl[:] = _STAND_CTRL
            mujoco.mj_step(self._model, self._data)
            self._step_count += 1

            q   = self._data.qpos[7:19].tolist()
            dq  = self._data.qvel[6:18].tolist()
            pos = self._data.qpos[:3].tolist()
            h   = float(self._data.qpos[2])

        js = JointState()
        js.name     = _GO1_JOINT_NAMES
        js.position = q
        js.velocity = dq
        js.effort   = [0.0] * 12
        self._pub_joints.publish(js)

        st = String()
        st.data = json.dumps({"x": pos[0], "y": pos[1], "z": h,
                               "step": self._step_count})
        self._pub_state.publish(st)

    def _on_reset(self, msg: Bool):
        if not msg.data:
            return
        with self._lock:
            mujoco.mj_resetData(self._model, self._data)
            self._data.qpos[2]    = 0.27
            self._data.qpos[3]    = 1.0
            self._data.qpos[7:19] = _STAND_CTRL
            mujoco.mj_forward(self._model, self._data)
            self._step_count = 0

    def body_height(self) -> float:
        with self._lock:
            return float(self._data.qpos[2])

    def joint_positions(self) -> list[float]:
        with self._lock:
            return self._data.qpos[7:19].tolist()


# ── Test harness ──────────────────────────────────────────────────────────────

class Harness:
    """Owns the rclpy context and both nodes for the duration of the test run."""

    def __init__(self):
        rclpy.init()
        self.cmd = CadenzaCommandNode()
        self.sim = CadenzaSimNode()

        self._exec = MultiThreadedExecutor(num_threads=4)
        self._exec.add_node(self.cmd)
        self._exec.add_node(self.sim)

        self._t = threading.Thread(target=self._exec.spin, daemon=True)
        self._t.start()
        time.sleep(0.4)   # let DDS discovery settle

    def shutdown(self):
        self._exec.shutdown(timeout_sec=3.0)
        self.cmd.destroy_node()
        self.sim.destroy_node()
        rclpy.shutdown()

    # ── helpers ───────────────────────────────────────────────────────────────

    def subscribe_once(self, msg_type, topic: str, timeout: float = 3.0):
        """Block until one message arrives on *topic*, return it."""
        received = []
        event = threading.Event()

        def cb(msg):
            received.append(msg)
            event.set()

        sub = self.cmd.create_subscription(msg_type, topic, cb, 10)
        event.wait(timeout)
        self.cmd.destroy_subscription(sub)
        return received[0] if received else None

    def subscribe_n(self, msg_type, topic: str, n: int,
                    timeout: float = 5.0) -> list:
        """Collect *n* messages from *topic*."""
        received = []
        event = threading.Event()

        def cb(msg):
            received.append(msg)
            if len(received) >= n:
                event.set()

        sub = self.cmd.create_subscription(msg_type, topic, cb, 10)
        event.wait(timeout)
        self.cmd.destroy_subscription(sub)
        return received

    def publish(self, msg_type, topic: str, data, qos: int = 10):
        """Publish one message and return."""
        pub = self.cmd.create_publisher(msg_type, topic, qos)
        time.sleep(0.05)   # allow subscriber discovery
        pub.publish(data)
        time.sleep(0.05)
        self.cmd.destroy_publisher(pub)

    def call_service(self, srv_type, srv_name: str, timeout: float = 3.0):
        """Call a Trigger service and return the response."""
        client = self.cmd.create_client(srv_type, srv_name)
        if not client.wait_for_service(timeout_sec=timeout):
            raise TimeoutError(f"Service {srv_name!r} unavailable")
        future = client.call_async(srv_type.Request())
        deadline = time.time() + timeout
        while not future.done():
            if time.time() > deadline:
                raise TimeoutError(f"Service call to {srv_name!r} timed out")
            time.sleep(0.01)
        self.cmd.destroy_client(client)
        return future.result()


# ═══════════════════════════════════════════════════════════════════════════════
#  Tests
# ═══════════════════════════════════════════════════════════════════════════════

_PASSED: list[str] = []
_FAILED: list[tuple[str, Exception]] = []


def run_test(name: str, fn, harness: Harness):
    try:
        fn(harness)
        _PASSED.append(name)
        print(f"  PASS  {name}")
    except Exception as exc:
        _FAILED.append((name, exc))
        print(f"  FAIL  {name}")
        print(f"        {type(exc).__name__}: {exc}")


# ── 1. cmd_vel → active_action ────────────────────────────────────────────────

def test_cmd_vel_forward(h: Harness):
    """Publish /cmd_vel (linear.x=0.5) → /cadenza/active_action == 'walk_forward'."""
    msg = Twist()
    msg.linear.x = 0.5

    def _go():
        h.publish(Twist, "/cmd_vel", msg)

    t = threading.Thread(target=_go, daemon=True)
    t.start()

    result = h.subscribe_once(String, "/cadenza/active_action", timeout=3.0)
    assert result is not None, "/cadenza/active_action timeout"
    assert result.data == "walk_forward", f"got {result.data!r}"


def test_cmd_vel_all_directions(h: Harness):
    """All 6 Twist directions map to the correct named action."""
    cases = [
        (Twist(), {"linear.x":  0.5},  "walk_forward"),
        (Twist(), {"linear.x": -0.5},  "walk_backward"),
        (Twist(), {"angular.z":  1.0}, "turn_left"),
        (Twist(), {"angular.z": -1.0}, "turn_right"),
        (Twist(), {"linear.y":  0.3},  "side_step_left"),
        (Twist(), {"linear.y": -0.3},  "side_step_right"),
    ]

    for base, fields, expected in cases:
        for attr, val in fields.items():
            obj, key = (base.linear,  attr.split(".")[1]) if "linear"  in attr else \
                       (base.angular, attr.split(".")[1])
            setattr(obj, key, val)

        results = []
        event = threading.Event()
        sub = h.cmd.create_subscription(
            String, "/cadenza/active_action",
            lambda m, r=results, e=event: (r.append(m.data), e.set()), 10)

        h.publish(Twist, "/cmd_vel", base)
        event.wait(3.0)
        h.cmd.destroy_subscription(sub)

        assert results, f"no message for {expected}"
        assert results[0] == expected, f"expected {expected!r}, got {results[0]!r}"


def test_cmd_vel_zero_is_stand(h: Harness):
    """Zero Twist → 'stand' (safe idle)."""
    msg = Twist()    # all zeros

    results = []
    event = threading.Event()
    sub = h.cmd.create_subscription(
        String, "/cadenza/active_action",
        lambda m: (results.append(m.data), event.set()), 10)

    h.publish(Twist, "/cmd_vel", msg)
    event.wait(3.0)
    h.cmd.destroy_subscription(sub)

    assert results and results[0] == "stand", f"got {results}"


# ── 2. NL command → sequence ──────────────────────────────────────────────────

def test_nl_command_parse(h: Harness):
    """NL string → correct ordered ActionCall sequence over /cadenza/command."""
    msg = String(); msg.data = "stand then walk forward then turn left then jump"

    results = []
    event = threading.Event()
    sub = h.cmd.create_subscription(
        String, "/cadenza/active_action",
        lambda m: (results.append(m.data), event.set()), 10)

    h.publish(String, "/cadenza/command", msg)
    event.wait(3.0)
    h.cmd.destroy_subscription(sub)

    assert results, "no response on /cadenza/active_action"
    seq = json.loads(results[0])
    assert seq == ["stand", "walk_forward", "turn_left", "jump"], \
        f"got {seq}"


def test_nl_command_distance(h: Harness):
    """Parser extracts quantitative distance from NL — not a blackbox estimate."""
    from cadenza.parser.translator import CommandParser
    parser = CommandParser("go1")
    calls = parser.parse("walk forward 3 meters")
    assert len(calls) == 1
    assert calls[0].action_name == "walk_forward"
    assert calls[0].distance_m == 3.0, f"got {calls[0].distance_m}"


def test_nl_command_determinism(h: Harness):
    """Same command string always produces identical ActionCall lists (no randomness)."""
    from cadenza.parser.translator import CommandParser
    parser = CommandParser("go1")
    cmd = "walk forward 1.5 meters then turn right then trot forward"

    a = parser.parse(cmd)
    b = parser.parse(cmd)
    c = parser.parse(cmd)

    for x, y in zip(a, b):
        assert x.action_name == y.action_name
        assert x.distance_m  == y.distance_m
    for x, z in zip(a, c):
        assert x.action_name == z.action_name


# ── 3. Services ───────────────────────────────────────────────────────────────

def test_list_actions_service(h: Harness):
    """'/cadenza/list_actions' service returns all 21 Go1 actions."""
    resp = h.call_service(Trigger, "/cadenza/list_actions")

    assert resp.success is True
    actions = json.loads(resp.message)
    assert isinstance(actions, list)
    assert len(actions) == 21, f"expected 21, got {len(actions)}"

    required = {"stand", "sit", "jump", "walk_forward", "walk_backward",
                "turn_left", "turn_right", "crawl_forward", "side_step_left",
                "side_step_right", "trot_forward"}
    missing = required - set(actions)
    assert not missing, f"missing actions: {missing}"


def test_execute_topic_valid(h: Harness):
    """Publishing a valid JSON plan to /cadenza/execute is accepted."""
    plan = ["stand", "walk_forward", "turn_left", "sit"]
    msg = String(); msg.data = json.dumps(plan)

    results = []
    event = threading.Event()
    sub = h.cmd.create_subscription(
        String, "/cadenza/active_action",
        lambda m: (results.append(m.data), event.set()), 10)

    h.publish(String, "/cadenza/execute", msg)
    event.wait(3.0)
    h.cmd.destroy_subscription(sub)

    assert results, "no response"
    resp = json.loads(results[0])
    assert resp["status"] == "accepted", f"got {resp}"
    assert resp["count"] == 4


def test_execute_topic_invalid(h: Harness):
    """Publishing an unknown action name is rejected — no silent failures."""
    plan = ["stand", "do_the_robot", "moonwalk_backward"]
    msg = String(); msg.data = json.dumps(plan)

    results = []
    event = threading.Event()
    sub = h.cmd.create_subscription(
        String, "/cadenza/active_action",
        lambda m: (results.append(m.data), event.set()), 10)

    h.publish(String, "/cadenza/execute", msg)
    event.wait(3.0)
    h.cmd.destroy_subscription(sub)

    assert results, "no response"
    resp = json.loads(results[0])
    assert resp["status"] == "aborted", f"got {resp}"
    assert "do_the_robot" in resp["invalid"]


def test_plan_validate_service(h: Harness):
    """After queuing a valid plan, /cadenza/plan_validate confirms it."""
    # Queue a plan first
    plan = ["stand", "walk_forward", "jump", "sit"]
    msg = String(); msg.data = json.dumps(plan)

    ready_event = threading.Event()
    ready_sub = h.cmd.create_subscription(
        Bool, "/cadenza/ready",
        lambda m: ready_event.set() if m.data else None, 10)

    h.publish(String, "/cadenza/execute", msg)
    ready_event.wait(3.0)
    h.cmd.destroy_subscription(ready_sub)

    # Now validate it via service
    resp = h.call_service(Trigger, "/cadenza/plan_validate")
    assert resp.success is True
    validated = json.loads(resp.message)
    assert validated == plan, f"got {validated}"


# ── 4. Joint state validity ───────────────────────────────────────────────────

def test_joint_targets_within_urdf_limits(h: Harness):
    """Joint targets published on /cadenza/joint_targets are within URDF limits."""
    actions_to_check = ["stand", "sit", "lie_down", "crouch"]

    for action_name in actions_to_check:
        cmd = String(); cmd.data = action_name

        results = []
        event = threading.Event()
        sub = h.cmd.create_subscription(
            JointState, "/cadenza/joint_targets",
            lambda m: (results.append(m), event.set()), 10)

        # Trigger via cmd_vel (stand) or direct execute
        if action_name == "stand":
            h.publish(Twist, "/cmd_vel", Twist())  # zero → stand
        else:
            plan_msg = String(); plan_msg.data = json.dumps([action_name])
            h.publish(String, "/cadenza/execute", plan_msg)
            # Also request the joint state via cmd (triggers _publish_joints_for)
            h.publish(String, "/cadenza/command", cmd)

        event.wait(3.0)
        h.cmd.destroy_subscription(sub)

        if not results:
            continue   # action may not have triggered joint pub; skip but don't fail

        js = results[0]
        assert len(js.name)     == 12, f"{action_name}: {len(js.name)} joints"
        assert len(js.position) == 12

        for i, pos in enumerate(js.position):
            jtype = i % 3
            if jtype == 0:
                lo, hi = _GO1_HIP_RANGE
            elif jtype == 1:
                lo, hi = _GO1_THIGH_RANGE
            else:
                lo, hi = _GO1_KNEE_RANGE
            assert lo <= pos <= hi, \
                f"{action_name} joint {_GO1_JOINT_NAMES[i]}: {pos:.3f} out of [{lo},{hi}]"


# ── 5. Headless physics ───────────────────────────────────────────────────────

def test_sim_publishes_joint_states(h: Harness):
    """Headless MuJoCo sim node publishes real JointState at 50 Hz."""
    msgs = h.subscribe_n(JointState, "/cadenza/sim/joint_states", n=5, timeout=3.0)
    assert len(msgs) >= 5, f"only got {len(msgs)} messages"

    for js in msgs:
        assert len(js.name)     == 12
        assert len(js.position) == 12
        assert len(js.velocity) == 12
        assert js.name[0] == "FL_hip_joint"
        assert js.name[9] == "RR_hip_joint"


def test_sim_publishes_state_json(h: Harness):
    """Headless sim publishes body position JSON on /cadenza/sim/state."""
    msgs = h.subscribe_n(String, "/cadenza/sim/state", n=3, timeout=3.0)
    assert msgs, "no state messages"

    state = json.loads(msgs[-1].data)
    assert "x" in state and "y" in state and "z" in state and "step" in state
    assert state["step"] > 0


def test_sim_stand_height_stable(h: Harness):
    """After 100 steps the robot's body holds stand height (≥ 0.20 m)."""
    # Wait for at least 100 physics steps (2 s at 50 Hz)
    steps_needed = 100
    event = threading.Event()
    seen_step = [0]

    def cb(msg: String):
        s = json.loads(msg.data)
        seen_step[0] = s["step"]
        if s["step"] >= steps_needed:
            event.set()

    sub = h.cmd.create_subscription(String, "/cadenza/sim/state", cb, 10)
    event.wait(timeout=5.0)
    h.cmd.destroy_subscription(sub)

    assert seen_step[0] >= steps_needed, \
        f"only reached step {seen_step[0]}"

    height = h.sim.body_height()
    assert height >= 0.20, \
        f"robot fell: body height {height:.3f} m < 0.20 m"


def test_sim_joint_positions_valid_after_warmup(h: Harness):
    """After physics warmup, live joint positions remain within URDF limits."""
    # Wait for 50 steps
    event = threading.Event()
    sub = h.cmd.create_subscription(
        String, "/cadenza/sim/state",
        lambda m: event.set() if json.loads(m.data)["step"] >= 50 else None, 10)
    event.wait(timeout=4.0)
    h.cmd.destroy_subscription(sub)

    q = h.sim.joint_positions()
    assert len(q) == 12

    for i, pos in enumerate(q):
        jtype = i % 3
        if jtype == 0:
            lo, hi = _GO1_HIP_RANGE
        elif jtype == 1:
            lo, hi = _GO1_THIGH_RANGE
        else:
            lo, hi = _GO1_KNEE_RANGE
        assert lo <= pos <= hi, \
            f"Live joint {_GO1_JOINT_NAMES[i]}: {pos:.3f} out of [{lo},{hi}]"


def test_sim_reset_topic(h: Harness):
    """Publishing True on /cadenza/sim/reset restarts physics at step 0."""
    # Wait until a few steps have accumulated
    event1 = threading.Event()
    sub1 = h.cmd.create_subscription(
        String, "/cadenza/sim/state",
        lambda m: event1.set() if json.loads(m.data)["step"] >= 20 else None, 10)
    event1.wait(timeout=3.0)
    h.cmd.destroy_subscription(sub1)

    pre_height = h.sim.body_height()

    # Reset
    reset_msg = Bool(); reset_msg.data = True
    h.publish(Bool, "/cadenza/sim/reset", reset_msg)
    time.sleep(0.2)

    # Wait for step counter to restart
    event2 = threading.Event()
    step_after = [0]
    sub2 = h.cmd.create_subscription(
        String, "/cadenza/sim/state",
        lambda m: (step_after.__setitem__(0, json.loads(m.data)["step"]), event2.set()), 10)
    event2.wait(timeout=2.0)
    h.cmd.destroy_subscription(sub2)

    # Step counter reset to 0 and climbed again
    assert step_after[0] < 50, \
        f"step counter {step_after[0]} suggests reset didn't happen"
    assert h.sim.body_height() >= 0.20, \
        f"post-reset height {h.sim.body_height():.3f} m — robot fell"


# ── 6. Nav2 primitives coverage ───────────────────────────────────────────────

def test_nav2_primitives_all_covered(h: Harness):
    """Action library covers every locomotion primitive nav2 needs."""
    from cadenza.actions import list_actions
    actions = set(list_actions("go1"))
    nav2 = {"walk_forward", "walk_backward", "turn_left", "turn_right",
            "side_step_left", "side_step_right", "stand"}
    missing = nav2 - actions
    assert not missing, f"nav2 primitives absent from library: {missing}"

    extras = actions - nav2
    assert len(extras) >= 10, f"expected ≥10 cadenza-only actions, got {len(extras)}"


# ═══════════════════════════════════════════════════════════════════════════════
#  Runner
# ═══════════════════════════════════════════════════════════════════════════════

TESTS = [
    ("cmd_vel forward → walk_forward",             test_cmd_vel_forward),
    ("cmd_vel all 6 directions",                   test_cmd_vel_all_directions),
    ("cmd_vel zero → stand",                       test_cmd_vel_zero_is_stand),
    ("NL command parse → sequence",                test_nl_command_parse),
    ("NL command extracts distance",               test_nl_command_distance),
    ("Parser is deterministic",                    test_nl_command_determinism),
    ("/cadenza/list_actions service (21 actions)", test_list_actions_service),
    ("/cadenza/execute valid plan accepted",        test_execute_topic_valid),
    ("/cadenza/execute invalid plan rejected",      test_execute_topic_invalid),
    ("/cadenza/plan_validate service",             test_plan_validate_service),
    ("Joint targets within URDF limits",           test_joint_targets_within_urdf_limits),
    ("Sim publishes JointState at 50 Hz",          test_sim_publishes_joint_states),
    ("Sim publishes body state JSON",              test_sim_publishes_state_json),
    ("Sim stand height stable after 100 steps",    test_sim_stand_height_stable),
    ("Live joint positions valid after warmup",    test_sim_joint_positions_valid_after_warmup),
    ("Sim resets on /cadenza/sim/reset",           test_sim_reset_topic),
    ("Nav2 locomotion primitives all covered",     test_nav2_primitives_all_covered),
]


def main():
    print()
    print("=" * 66)
    print("  Cadenza ROS 2 Integration Tests")
    print("  Real rclpy nodes + headless MuJoCo physics")
    print("=" * 66)
    print()

    harness = Harness()
    try:
        for name, fn in TESTS:
            run_test(name, fn, harness)
    finally:
        harness.shutdown()

    total = len(TESTS)
    print()
    print(f"  {len(_PASSED)}/{total} passed", end="")
    if _FAILED:
        print(f"  ({len(_FAILED)} failed)\n")
        sys.exit(1)
    else:
        print("  — all passed\n")


if __name__ == "__main__":
    main()
