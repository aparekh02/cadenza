"""UnitreeConfig — YAML-driven configuration for the locomotion layer.

All topic names, control rates, blend weights, memory paths, and robot
constants are centralised here. The YAML file is the single source of truth.

Typical usage
-------------
    cfg = load_config("examples/unitree_go1/config/go1.yaml")
    print(cfg.robot_model)   # "go1"
    print(cfg.topics.cmd_vel)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ──────────────────────────────────────────────────────────────────────────────
#  Sub-configs
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TopicConfig:
    """ROS2 topic names (or standalone socket IDs)."""
    cmd_vel:      str = "/cmd_vel"
    joint_states: str = "/joint_states"
    imu:          str = "/imu/data"
    foot_contact: str = "/foot_contact"
    loco_command: str = "/cadenza/loco_command"
    memory_update: str = "/cadenza/memory_update"


@dataclass
class ControlConfig:
    """Control loop and blend parameters."""
    rate_hz:         float = 50.0          # control loop frequency
    stm_window:      int   = 50            # STM rolling window size
    skill_alpha:     float = 0.15          # skill waypoint blend weight (gentle nudge)
    terrain_alpha:   float = 0.3           # terrain recommendation blend weight
    safety_priority: bool  = True          # safety overrides always win


@dataclass
class MemoryConfig:
    """Paths to memory snapshot and log output."""
    snapshot:   str = ""          # path to snapshot.json (required at runtime)
    log_dir:    str = "logs"      # directory for experience JSONL files


@dataclass
class RobotConfig:
    """Physical constants for the target robot."""
    n_joints:      int   = 12
    joint_names:   list[str] = field(default_factory=lambda: [
        "FL_hip", "FL_thigh", "FL_calf",
        "FR_hip", "FR_thigh", "FR_calf",
        "RL_hip", "RL_thigh", "RL_calf",
        "RR_hip", "RR_thigh", "RR_calf",
    ])
    max_speed_xy:  float = 1.5    # m/s
    max_yaw_rate:  float = 1.0    # rad/s


# ──────────────────────────────────────────────────────────────────────────────
#  Top-level config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class UnitreeConfig:
    robot_model: str          = "go1"
    topics:      TopicConfig  = field(default_factory=TopicConfig)
    control:     ControlConfig = field(default_factory=ControlConfig)
    memory:      MemoryConfig  = field(default_factory=MemoryConfig)
    robot:       RobotConfig   = field(default_factory=RobotConfig)


# ──────────────────────────────────────────────────────────────────────────────
#  Loader
# ──────────────────────────────────────────────────────────────────────────────

def load_config(path: str | Path) -> UnitreeConfig:
    """Load a YAML config file and return a UnitreeConfig.

    Args:
        path: path to the YAML config file

    Raises:
        FileNotFoundError: if the file does not exist
        ImportError:       if PyYAML is not installed
    """
    try:
        import yaml
    except ImportError as e:
        raise ImportError("PyYAML required: pip install pyyaml") from e

    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path) as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}

    cfg = UnitreeConfig()
    cfg.robot_model = str(raw.get("robot_model", cfg.robot_model))

    topics_raw = raw.get("topics", {})
    cfg.topics = TopicConfig(
        cmd_vel       = str(topics_raw.get("cmd_vel",       cfg.topics.cmd_vel)),
        joint_states  = str(topics_raw.get("joint_states",  cfg.topics.joint_states)),
        imu           = str(topics_raw.get("imu",           cfg.topics.imu)),
        foot_contact  = str(topics_raw.get("foot_contact",  cfg.topics.foot_contact)),
        loco_command  = str(topics_raw.get("loco_command",  cfg.topics.loco_command)),
        memory_update = str(topics_raw.get("memory_update", cfg.topics.memory_update)),
    )

    ctrl_raw = raw.get("control", {})
    cfg.control = ControlConfig(
        rate_hz         = float(ctrl_raw.get("rate_hz",         cfg.control.rate_hz)),
        stm_window      = int(ctrl_raw.get("stm_window",        cfg.control.stm_window)),
        skill_alpha     = float(ctrl_raw.get("skill_alpha",     cfg.control.skill_alpha)),
        terrain_alpha   = float(ctrl_raw.get("terrain_alpha",   cfg.control.terrain_alpha)),
        safety_priority = bool(ctrl_raw.get("safety_priority",  cfg.control.safety_priority)),
    )

    mem_raw = raw.get("memory", {})
    cfg.memory = MemoryConfig(
        snapshot = str(mem_raw.get("snapshot", cfg.memory.snapshot)),
        log_dir  = str(mem_raw.get("log_dir",  cfg.memory.log_dir)),
    )

    robot_raw = raw.get("robot", {})
    cfg.robot = RobotConfig(
        n_joints      = int(robot_raw.get("n_joints",     cfg.robot.n_joints)),
        joint_names   = list(robot_raw.get("joint_names", cfg.robot.joint_names)),
        max_speed_xy  = float(robot_raw.get("max_speed_xy", cfg.robot.max_speed_xy)),
        max_yaw_rate  = float(robot_raw.get("max_yaw_rate", cfg.robot.max_yaw_rate)),
    )

    return cfg
