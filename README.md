<div align="center">

<img src="cadenza-logo.png" alt="Cadenza OS" width="350">

**The FIRST AI-native, developer-first OS for robot intelligence.**
<br>Deterministic scheduling. Easy custom kernel/package integration. On-edge multimodal AI.

</div>

<p align="center">
  <a href="#quick-start">Quick Start</a>
  <span>&nbsp;&nbsp;&bull;&nbsp;&nbsp;</span>
  <a href="#aicore">AICore</a>
  <span>&nbsp;&nbsp;&bull;&nbsp;&nbsp;</span>
  <a href="#action-library">Action Library</a>
  <span>&nbsp;&nbsp;&bull;&nbsp;&nbsp;</span>
  <a href="#kernel-sdk">Kernel SDK</a>
  <span>&nbsp;&nbsp;&bull;&nbsp;&nbsp;</span>
  <a href="#examples">Examples</a>
  <span>&nbsp;&nbsp;&bull;&nbsp;&nbsp;</span>
  <a href="#cadenza-pro">Pro</a>
</p>

<p align="center">
  <img alt="Version" src="https://img.shields.io/badge/version-1.1.2-blue?style=for-the-badge">
  <img alt="Platform" src="https://img.shields.io/badge/Platform-ARM64_Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black">
  <img alt="License" src="https://img.shields.io/badge/License-Apache_2.0-green?style=for-the-badge">
  <img alt="Repo Size" src="https://img.shields.io/badge/Repo_Size-233_MB-orange?style=for-the-badge">
</p>

<p align="center">
  <img alt="Jetson Orin" src="https://img.shields.io/badge/Jetson_Orin-Supported-76B900?style=flat-square&logo=nvidia&logoColor=white">
  <img alt="Unitree Go1" src="https://img.shields.io/badge/Unitree_Go1-21_Actions-E34F26?style=flat-square">
  <img alt="Unitree G1" src="https://img.shields.io/badge/Unitree_G1-20_Actions-E34F26?style=flat-square">
</p>

## What is Cadenza?
<div align="center">

<video src="demo.mov" width="600" controls autoplay loop muted>
  Your browser does not support the video tag.
</video>

</div>

| Problem | How Cadenza Solves It |
|---|---|
| Specializing robot actions for different situations/environments is REALLY brute force with RL and requires intense resources/planning. Along so, cloud inference adds 50–200ms latency to every decision. And the developer experience is often annoying when reflashing new features or implementing new packages. | Cadenza has a static action library, which has preset walking, running, lifting, and other actions stored, using on-board 4-tier SLM stack running entirely on-edge to specialize the action library to the situation. Hot-load packages and kernels at runtime allow zero reboot for developers adding their OWN features to the robot. |

---

## Features

| Category | Features | Benefits |
|----------|----------|----------|
| **On-Edge Intelligence** | 4-tier SLM stack (SigLIP → Moondream → Phi-3.5 → Llama-3.2)<br>Priority-cascaded behavior tree<br>Per-hardware model selection<br>Ollama integration for local LLMs | Sub-50ms decisions on Jetson Orin<br>Safety reflexes never blocked by inference<br>Works on Orin NX, Orin Nano, and dev machines<br>Swappable models without code changes |
| **Action Library** | 41 motor-level primitives (Go1 + G1)<br>URDF-sourced joint targets and torque limits<br>Phase-based and gait-engine actions<br>Concurrent action composition | Zero sim-to-real gap at the primitive level<br>Physically validated on real hardware<br>Walk, trot, crawl, jump, climb, turn<br>Walking arcs, complex maneuvers |
| **Real-Time OS** | Deterministic scheduling (SCHED_FIFO + SCHED_DEADLINE)<br>Zero-copy pub/sub over shared memory<br>Isolated core pinning per task class<br>Hot-load/unload packages without reboot | 1ms SAFETY, 5ms CONTROL guaranteed<br>Lock-free SPSC queues, no mutexes<br>AI inference never starves motor control<br>Deploy new behaviors to a running robot |
| **Developer Experience** | `cadenza` CLI for project lifecycle<br>MuJoCo simulation with closed-loop feedback<br>Kernel SDK for custom modules<br>Sim → SSH → DDS → Bridge deployment | One command to init, build, deploy<br>Test everything before touching hardware<br>Write perception/planning as a kernel<br>Laptop runs VLA, robot runs actions |


## <a name="quick-start"></a> Quick Start

### Install

```bash
git clone https://github.com/yourorg/cadenza.git
cd cadenza

python -m venv .venv
source .venv/bin/activate
pip install numpy mujoco scipy faiss-cpu
```

### First Simulation

```bash
mjpython example.py
```

Opens a MuJoCo viewer. The Go1 stands, walks 2m, arcs through a turn, jumps, and sits.

### Natural Language Commands

```bash
mjpython tests/test_go1_actions.py "walk forward then turn left then jump"
```

```python
import cadenza_local as cadenza
cadenza.run("stand then walk forward 2 meters then turn left then jump then sit down")
```

---

## <a name="aicore"></a> AICore — On-Edge Intelligence

Four tiers of SLMs, each running at the frequency its task demands. No tier waits for a slower tier. Safety never waits for any model.

| Tier | Model | Params | Latency | Frequency | Role |
|------|-------|--------|---------|-----------|------|
| 0 | SigLIP-SO400M | 400M | <25ms | 20 Hz | Camera frame → 768-dim scene embedding |
| 1 | Moondream2 | 1.6B | <50ms | 10 Hz | Image + prompt → structured scene description |
| 2 | Phi-3.5-mini | 3.8B | <500ms | 2 Hz | Scene + state + goal → JSON action plan |
| 3 | Llama-3.2-3B | 3B | <2s | 0.5 Hz | Complex goal decomposition via Ollama |

All models run quantized (INT4) on-device. Tier 3 is optional.

### Hardware-Specific Stacks

| Target | T0 Vision | T1 Scene | T2 Planner | T3 Strategic |
|--------|-----------|----------|------------|--------------|
| Jetson Orin NX | SigLIP 400M | Moondream 1.6B | Phi-3.5 3.8B | Llama-3.2 3B |
| Jetson Orin Nano | DINOv2 22M | Moondream 1.6B | Gemma-2 2B | Llama-3.2 3B |
| Dev machine | SigLIP 400M | Moondream 1.6B | Phi-3.5 3.8B | Llama-3.2 3B |
| Minimal | DINOv2 22M | — | Gemma-2 2B | — |

### Behavior Tree Guarantees

| Layer | Name | Latency | What It Does |
|-------|------|---------|--------------|
| 0 | SAFETY | <1ms | Tilt recovery, obstacle halt, contact loss detection |
| 1 | REACTIVE | <1ms | Terrain-adaptive gait selection (rough → crawl, stairs → climb) |
| 2 | TACTICAL | <1ms | Waypoint navigation, heading correction |
| 3 | STRATEGIC | <2s | SLM-powered planning (only when Layers 0–2 defer) |

### Usage

```python
from cadenza.aicore import BehaviorEngine, ActionPlanner, VisionEncoder, SLMBridge

# Full stack
engine = BehaviorEngine(
    "go1",
    vision=VisionEncoder("siglip-so400m"),
    planner=ActionPlanner("phi3.5:3.8b-mini-instruct-q4_K_M"),
    slm=SLMBridge(),
)

engine.set_goal("navigate to the red cone, avoid obstacles, sit when you arrive")

world = engine.observe_with_camera(qpos, qvel, camera_frame)
decision = engine.decide(world)

print(decision.action)     # "crawl_forward"
print(decision.layer)      # "REACTIVE"
print(decision.reasoning)  # "Terrain: rough surface (roughness=0.47), crawling"
```

### CLI

```bash
cadenza ai status      # model info, decision rate, confidence
cadenza ai goal "..."  # set a high-level goal
cadenza ai decide      # current decision with reasoning
cadenza ai history     # last 10 decisions
```

---

## <a name="action-library"></a> Action Library

41 motor-level action primitives for two robot platforms, sourced from URDF and MuJoCo Menagerie. Every action has exact joint targets, PD gains, and torque limits. No learned controller, no sim-to-real gap at the primitive level.

### Go1 Quadruped — 21 Actions

```python
import cadenza_local as cadenza

go1 = cadenza.go1()
go1.run([
    go1.stand(),
    go1.walk_forward(speed=1.5, distance_m=3.0),
    [go1.turn_left(), go1.walk_forward()],   # concurrent: walking arc
    go1.jump(speed=2.0, extension=1.2),
    go1.sit(),
])
```

<details>
<summary><strong>Full Go1 action table</strong></summary>

| Action | Type | Description |
|--------|------|-------------|
| `stand()` | phase | Stand at default height |
| `stand_up()` | phase | Stand up from lying down |
| `sit()` | phase | Sit down |
| `lie_down()` | phase | Lie flat |
| `jump()` | phase | Jump in place |
| `walk_forward()` | gait | Walk forward |
| `walk_backward()` | gait | Walk backward |
| `trot_forward()` | gait | Trot (diagonal gait) |
| `crawl_forward()` | gait | Crawl (low, stable) |
| `pace_forward()` | gait | Pace (lateral gait) |
| `bound_forward()` | gait | Bound (synchronous front-back) |
| `turn_left()` | gait | Turn left in place |
| `turn_right()` | gait | Turn right in place |
| `climb_step()` | gait | Climb a step |
| `side_step_left()` | gait | Lateral step left |
| `side_step_right()` | gait | Lateral step right |
| `rear_up()` | phase | Rear up on hind legs |
| `shake_hand()` | phase | Extend front paw |
| `rear_kick()` | phase | Kick with rear legs |

All actions accept `speed` and `extension` multipliers. Gait actions also accept `distance_m` and `repeat`.

</details>

### G1 Humanoid — 20 Actions

```python
g1 = cadenza.g1()
g1.run([
    g1.stand(),
    g1.walk_forward(speed=0.3, distance_m=1.0),
    g1.crouch(),
    g1.lift_left_hand(),
    g1.stand(),
])
```

### Direct Library Access

```python
from cadenza_local.actions import get_library

lib = get_library("go1")
spec = lib.get("walk_forward")
print(spec.gait)        # GaitAction with velocity commands
print(spec.phases)      # list of ActionPhase (for phase-based actions)
```

---

## Deploying to Hardware

```bash
mjpython examples/unitree_go1/deploy_go1.py           # sim (default)
mjpython examples/unitree_go1/deploy_go1.py deploy     # SSH to robot
mjpython examples/unitree_go1/deploy_go1.py bridge     # VLA on laptop, actions on robot
```

```python
# Bridge mode: heavy model on your laptop, lightweight actions on the robot
bridge = go1.deploy_ssh_bridge(host="192.168.123.15", key="~/.ssh/go1_rsa")

while True:
    state = bridge.telemetry
    action = my_model(state)
    bridge.send_action(action, speed=0.5)

bridge.estop()
```

---

## <a name="kernel-sdk"></a> Kernel SDK

Build custom perception, planning, or control modules that run on the OS's deterministic scheduler and communicate via zero-copy pub/sub.

```cpp
#include <cadenza/kernel_sdk/cadenza_kernel.h>

class MyPerceptionKernel : public cadenza::sdk::CadenzaKernel {
public:
    int on_init() override {
        subscribe("sensors/camera", [this](uint32_t, const void* data, size_t len) {
            // process camera frame
        });
        return 0;
    }

    int on_tick() override {
        float obstacles[10] = { /* ... */ };
        publish("perception/obstacles", obstacles, sizeof(obstacles));
        return 0;
    }

    void on_shutdown() override {}

    cadenza::sdk::KernelConfig config() const override {
        return {
            .name = "my_perception",
            .task_class = cadenza::dsk::TaskClass::INFERENCE,
            .period_us = 50000,
            .deadline_us = 50000,
            .core = cadenza::dsk::CoreAffinity::SHARED,
        };
    }
};

CADENZA_REGISTER_KERNEL(MyPerceptionKernel)
```

```bash
cadenza build && cadenza attach my_perception
```

### DSK Task Classes

| Class | Period | Policy | Priority | Core | Use For |
|-------|--------|--------|----------|------|---------|
| SAFETY | 1ms | SCHED_FIFO | 99 | Isolated | E-stop, collision avoidance |
| CONTROL | 5ms | SCHED_FIFO | 80 | Isolated | Joint control, balance |
| SENSOR | 10ms | SCHED_DEADLINE | 60 | Isolated | IMU, LiDAR, cameras |
| INFERENCE | 50ms | SCHED_DEADLINE | 40 | Shared | Perception, planning |
| COMMS | 100ms | SCHED_OTHER | 0 | Shared | Telemetry, fleet sync |

SAFETY and CONTROL are immutable. INFERENCE and COMMS adjustable at runtime.

---

## Building the OS

```bash
cd cadenza
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCADENZA_TEST_MODE=1
make -j$(nproc) && make test
```

### First-Time Setup (Ubuntu 22.04 ARM64)

```bash
sudo bash cadenza/scripts/setup.sh
```

### CLI

```bash
cadenza init my-project --robot go1     # scaffold a new project
cadenza build                           # cmake + make
cadenza deploy --target 192.168.1.10    # rsync + hot-swap, no reboot
cadenza attach my_package               # hot-load a package
cadenza kernel                          # task table: name, core, CPU%, miss rate
cadenza ai status                       # AICore health + model info
cadenza log --topic sensors/imu         # stream IKM topic as JSON
```

---

## Trajectory Generation

The `actions-gen/` module generates physically stable walking trajectories using ZMP preview control (Kajita et al. 2003):

```bash
cd actions-gen && python runMe.py
```

LIPM preview control → multi-task IK with null-space projections → MuJoCo physics validation → `simulation_video.gif`

---

## <a name="examples"></a> Examples

| Example | Robot | Description |
|---------|-------|-------------|
| `example.py` | Go1 | Stand, walk, arc turn, jump, sit |
| `examples/unitree_go1/deploy_go1.py` | Go1 | Sim / SSH deploy / DDS / bridge mode |
| `examples/unitree_g1/deploy_g1.py` | G1 | Humanoid sim and deployment |
| `examples/mountain_goat/mountain_goat.py` | Go1 | Terrain-aware navigation across 7 zones |
| `tests/test_go1_actions.py` | Go1 | Natural language command demo |

```bash
mjpython example.py
mjpython examples/mountain_goat/mountain_goat.py
```

---

## Project Structure

```
cadenza/              C++ OS kernel + AICore intelligence
  aicore/             Multimodal SLM stack (Python + C++)
  dsk/                Deterministic RT scheduling
  ikm/                Zero-copy pub/sub middleware
  packages/           Hot-loadable plugin system
  kernel_sdk/         SDK for custom kernels
  cli/                Developer CLI

cadenza_local/        Python action library + simulation
  actions/            41 motor-level primitives
  locomotion/         Kinematics, gait engine, balance
  robots/             Robot descriptors (Go1, G1)
  models/             MuJoCo XMLs + gait data
  sim.py              MuJoCo simulator
  go1.py, g1.py       Developer controllers

actions-gen/          ZMP preview control trajectory generator
examples/             Runnable demos
gym3d/                Shared MuJoCo 3D environment
tests/                Integration tests
```

---

## <a name="cadenza-pro"></a> Cadenza Pro

The open-source edition includes AICore, the full action library, simulation, deterministic scheduling, pub/sub, packages, kernel SDK, and CLI.

**Cadenza Pro** adds on-device learning and adaptive resource management:

| Feature | What It Does |
|---------|--------------|
| **RLAK** | Real-time RL that learns joint corrections from live experience, bounded to safe limits |
| **AIOS** | Adaptive OS governor — thermal management, precision switching, load prediction |
| **AIK** | Deadline-guaranteed inference with automatic caching and chained perception |
| **Hardware Acceleration** | Jetson DLA/GPU async inference, TensorRT INT8/INT4 quantization |
| **Full LoRA** | Natural language command parsing with sensor-driven parameter optimization |
| **Action Generation** | Physics-based synthesis of new action library entries from experience |

<p align="center">
  <a href="https://cadenza.dev/pro">
    <img alt="Cadenza Pro" src="https://img.shields.io/badge/Learn_More-Cadenza_Pro-0A0A0A?style=for-the-badge">
  </a>
</p>

---

## Specifications

| Spec | Value |
|------|-------|
| Version | 1.1.2 |
| OS target | Ubuntu 22.04 LTS (ARM64) |
| Language (OS) | C++20 |
| Language (local) | Python 3.10+ |
| RT scheduling | `SCHED_FIFO` (SAFETY/CONTROL), `SCHED_DEADLINE` (SENSOR/INFERENCE) |
| IKM throughput | 160K msg/s, zero-copy, lock-free SPSC |
| AICore latency | <1ms safety, <50ms perception, <500ms planning |
| Supported robots | Unitree Go1 (12 DOF), Unitree G1 (29 DOF) |
| Action primitives | 41 total (21 Go1 + 20 G1) |
| SLM tiers | 4 (SigLIP 400M → Moondream 1.6B → Phi-3.5 3.8B → Llama-3.2 3B) |
| Quantization | INT4 on-device (all tiers) |
| Hardware targets | Jetson Orin NX, Orin Nano, RK3588, any ARM64 dev board |
| Simulator | MuJoCo 3.x with closed-loop feedback |
| Deployment | SSH, DDS, Bridge (VLA on laptop + actions on robot) |
| Hot-reload | Packages and kernels, zero reboot |

---

## Contributing

Cadenza is built for the robotics community. Contributions welcome across the stack — action primitives, kernels, packages, examples, and documentation.

1. Browse [open issues](https://github.com/yourorg/cadenza/issues) for `good first issue` tags
2. Read the [contribution guide](CONTRIBUTING.md)
3. Open a PR — we review within 48 hours

---

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.
