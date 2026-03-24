"""run_controller.py — Cadenza locomotion demo for Unitree Go1/Go2.

Usage
-----
    python examples/unitree_go1/run_controller.py
    python examples/unitree_go1/run_controller.py --robot go2
    python examples/unitree_go1/run_controller.py --steps 500
    python examples/unitree_go1/run_controller.py --visualize
    python examples/unitree_go1/run_controller.py --resume go1-a3f2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# ── Load .env from repo root ──────────────────────────────────────────────────
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_env_file = _REPO / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from cadenza_local.locomotion import (
    STM, STMFrame,
    MapMem, SkillMem, SafetyMem, UserMem,
    LocoController, ExperienceLogger,
    GroqMemoryLayer, get_spec, load_config, clip_joints,
)
from cadenza_local.locomotion.visualizer import RobotVisualizer

# ── Memory store ──────────────────────────────────────────────────────────────
_MEM_DIR = Path.home() / ".cadenza" / "runs"


def _save_snapshot(robot: str, steps: int,
                   mapmem: MapMem, skillmem: SkillMem,
                   safetymem: SafetyMem, usermem: UserMem) -> str:
    _MEM_DIR.mkdir(parents=True, exist_ok=True)
    ts  = datetime.utcnow().isoformat()
    raw = f"{robot}-{ts}-{steps}"
    uid = hashlib.sha1(raw.encode()).hexdigest()[:4]
    mem_id = f"{robot}-{uid}"
    data = {
        "meta":      {"robot": robot, "steps": steps, "timestamp": ts, "id": mem_id},
        "mapmem":    mapmem.to_list()    if hasattr(mapmem,    "to_list") else [],
        "skillmem":  skillmem.to_list()  if hasattr(skillmem,  "to_list") else [],
        "safetymem": safetymem.to_list() if hasattr(safetymem, "to_list") else [],
        "usermem":   usermem.to_list()   if hasattr(usermem,   "to_list") else [],
    }
    path = _MEM_DIR / f"{mem_id}.json"
    path.write_text(json.dumps(data, indent=2))
    return mem_id


def _load_resume(mem_id: str) -> tuple[MapMem, SkillMem, SafetyMem, UserMem]:
    path = _MEM_DIR / f"{mem_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"No memory found for ID '{mem_id}' at {path}")
    snap = json.loads(path.read_text())
    return (
        MapMem.from_list(snap.get("mapmem",    [])),
        SkillMem.from_list(snap.get("skillmem",  [])),
        SafetyMem.from_list(snap.get("safetymem", [])),
        UserMem.from_list(snap.get("usermem",   [])),
    )


def _load_snapshot(path) -> tuple[MapMem, SkillMem, SafetyMem, UserMem]:
    if path and Path(path).exists():
        snap = json.loads(Path(path).read_text())
        return (
            MapMem.from_list(snap.get("mapmem",    [])),
            SkillMem.from_list(snap.get("skillmem",  [])),
            SafetyMem.from_list(snap.get("safetymem", [])),
            UserMem.from_list(snap.get("usermem",   [])),
        )
    return MapMem(), SkillMem(), SafetyMem(), UserMem()


# ──────────────────────────────────────────────────────────────────────────────

def _cmd_at(t: float) -> tuple[np.ndarray, str]:
    """Cycle through representative tasks for the demo."""
    if   t < 10: return np.array([0.8,  0.0,  0.0],  dtype=np.float32), "walk forward"
    elif t < 20: return np.array([0.4,  0.1,  0.2],  dtype=np.float32), "turn on gravel"
    elif t < 30: return np.array([0.3,  0.0,  0.0],  dtype=np.float32), "slope ahead"
    elif t < 40: return np.array([0.15, 0.0,  0.0],  dtype=np.float32), "climb stairs"
    elif t < 50: return np.array([1.0,  0.0,  0.0],  dtype=np.float32), "sprint on flat"
    else:        return np.array([0.5,  0.0, -0.1],  dtype=np.float32), "gentle curve"


class _Sim:
    """Generates realistic sensor data without real hardware."""

    def __init__(self, spec):
        self._spec = spec
        self._rng  = np.random.default_rng(0)
        self._t0   = time.monotonic()
        self._q    = np.array(spec.poses.stand, dtype=np.float32)
        self._dq   = np.zeros(12, dtype=np.float32)

    def read(self, cmd_vel: np.ndarray, t: float) -> STMFrame:
        spec  = self._spec
        phase = t * 3.0 * 2 * math.pi
        swing = 0.5 * (1 + math.sin(phase))
        tgt   = np.array(spec.poses.stand, dtype=np.float32)
        for i, a in [(1, swing), (4, 1-swing), (7, 1-swing), (10, swing)]:
            tgt[i] += 0.3 * a * abs(cmd_vel[0])
        self._dq = (tgt - self._q) * 0.5 + self._rng.normal(0, 0.05, 12).astype(np.float32)
        self._q  = clip_joints(self._q + self._dq * 0.02, spec)
        return STMFrame(
            timestamp    = time.monotonic() - self._t0,
            joint_pos    = self._q.copy(),
            joint_vel    = self._dq.copy(),
            imu_rpy      = np.array([0.03*math.sin(t*0.3), 0.02*math.sin(t*0.4), 0.0], dtype=np.float32),
            imu_omega    = self._rng.normal(0, 0.01, 3).astype(np.float32),
            foot_contact = np.array([swing<0.5, swing>=0.5, swing>=0.5, swing<0.5], dtype=np.float32),
            cmd_vel      = cmd_vel.copy(),
        )


# ──────────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────────

def main(robot: str, steps: int, snapshot, cfg_path: Path,
         visualize: bool, resume: str | None,
         groq_interval: float = 30.0) -> None:
    spec     = get_spec(robot)
    cfg      = load_config(cfg_path)
    has_groq = bool(os.environ.get("GROQ_API_KEY"))
    groq_desc = (f"memory every {int(groq_interval)}s" if groq_interval > 0
                 else "episode-end only") if has_groq else "off (no GROQ_API_KEY)"

    # ── Header ─────────────────────────────────────────────────────────────
    print(f"\n  {spec.model.upper()}  |  {spec.kin.total_mass_kg}kg  "
          f"|  thigh/calf {spec.kin.thigh_length}m  "
          f"|  max torque {spec.motor.max_torque_Nm}Nm  "
          f"|  Groq memory: {groq_desc}\n")

    # ── Setup ──────────────────────────────────────────────────────────────
    if resume:
        print(f"  Resuming from memory ID: {resume}")
        mapmem, skillmem, safetymem, usermem = _load_resume(resume)
    else:
        mapmem, skillmem, safetymem, usermem = _load_snapshot(snapshot)

    usermem.set("max_speed",   spec.benchmarks.flat_speed_ms)
    usermem.set("body_height", spec.kin.com_height_stand * 0.8)

    # Local analytical controller — no API calls in the control loop
    logger   = ExperienceLogger(log_dir=cfg.memory.log_dir, run_id=robot)
    stm      = STM(window=cfg.control.stm_window)
    ctrl     = LocoController(
        cfg=cfg, stm=stm,
        mapmem=mapmem, skillmem=skillmem,
        safetymem=safetymem, usermem=usermem,
        logger=logger, groq_advisor=None, spec=spec,
    )
    # Groq memory layer — fires periodically in background threads
    groq_mem = GroqMemoryLayer(
        robot_model=robot,
        memory_interval_s=groq_interval,
    ) if has_groq else None
    sim = _Sim(spec)
    dt  = 1.0 / cfg.control.rate_hz
    t0  = time.monotonic()

    # ── Visualizer ─────────────────────────────────────────────────────────
    viz = None
    if visualize:
        viz = RobotVisualizer(spec, refresh_hz=10.0)
        viz.start()
        print("  Visualizer: open\n")

    # ── Table ──────────────────────────────────────────────────────────────
    print(f"  {'step':>5}  {'task':<18}  {'terrain':<13}  {'gait':<10}  "
          f"{'vx':>5}  {'vy':>5}  {'status':<8}  {'bench'}")
    print(f"  {'─'*5}  {'─'*18}  {'─'*13}  {'─'*10}  "
          f"{'─'*5}  {'─'*5}  {'─'*8}  {'─'*5}")

    _done = threading.Event()

    def _control_loop():
        for i in range(steps):
            t          = i * dt
            cmd, task  = _cmd_at(t)
            frame      = sim.read(cmd, t)
            loco       = ctrl.step(frame, cmd, task_text=task)

            if groq_mem is not None:
                groq_mem.push(frame, loco)

            if viz is not None:
                viz.update(loco.q12_target, frame.imu_rpy, frame.foot_contact,
                           loco.gait, loco.terrain, loco.cmd_vx, i)

            if i % 25 == 0:
                status = "STOP" if loco.safety_active else "ok"
                bench  = "ok" if loco.on_benchmark else "low"
                print(f"  {i:>5}  {task:<18}  {loco.terrain:<13}  "
                      f"{loco.gait:<10}  {loco.cmd_vx:>5.2f}  {loco.cmd_vy:>5.2f}  "
                      f"{status:<8}  {bench}")
                if groq_mem is not None and groq_mem.latest_summary:
                    s = groq_mem.latest_summary
                    if s.episode_summary:
                        print(f"         [Groq Memory] {s.episode_summary[:72]}")

            # real-time pace
            elapsed = time.monotonic() - t0
            sleep   = (i + 1) * dt - elapsed
            if sleep > 0:
                time.sleep(sleep)

        ctrl.notify_episode_outcome(success=True)
        if groq_mem is not None:
            groq_mem.on_episode_end(success=True)
        logger.close()
        _done.set()

    if viz is not None:
        # Control loop in background; main thread drives GUI
        threading.Thread(target=_control_loop, daemon=True).start()
        while not _done.is_set():
            viz.tick()
            time.sleep(viz._refresh)
        viz.stop()
    else:
        _control_loop()

    # ── Save memory & print ID ─────────────────────────────────────────────
    mem_id = _save_snapshot(robot, steps, mapmem, skillmem, safetymem, usermem)
    print(f"\n  Done — {steps} steps  |  log: {logger.path}")
    print(f"  Memory ID: {mem_id}  (resume with --resume {mem_id})\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--robot",      default="go1",  choices=["go1", "go2"])
    p.add_argument("--steps",      type=int, default=300)
    p.add_argument("--snapshot",   type=Path, default=None)
    p.add_argument("--config",     type=Path,
                   default=Path(__file__).parent / "config" / "go1.yaml")
    p.add_argument("--visualize",  action="store_true",
                   help="Open live 3D stick-figure viewer")
    p.add_argument("--resume",     type=str, default=None, metavar="MEMORY_ID",
                   help="Resume from a saved memory ID (e.g. go1-a3f2)")
    p.add_argument("--groq-interval", type=float, default=30.0, metavar="SECONDS",
                   help="Seconds between Groq memory calls (0 = episode-end only, "
                        "requires GROQ_API_KEY)")
    a = p.parse_args()

    cfg_path = a.config
    if a.robot == "go2" and a.config == p.get_default("config"):
        cfg_path = Path(__file__).parent / "config" / "go2.yaml"

    main(a.robot, a.steps, a.snapshot, cfg_path, a.visualize, a.resume,
         groq_interval=a.groq_interval)
