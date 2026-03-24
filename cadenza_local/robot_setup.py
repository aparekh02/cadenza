"""Robot specification extraction from MuJoCo models.

Given a robot name or XML path, extracts every physical property:
bodies (masses, inertias, hierarchy), joints (axes, ranges, stiffness),
actuators (gear ratios, ctrl ranges, max torques), and tendons.

Robots are registered externally via register_robot() — cadenza/ contains
no robot-specific assets or registrations.

Usage:
    # Register from outside (e.g., in your example script):
    register_robot("so101", "/path/to/so101.xml", hints=RobotHints(...))

    # Then load by name:
    spec, model, data = RobotSetup.from_name("so101")

    # Or load directly from XML:
    spec, model, data = RobotSetup.from_xml("path/to/robot.xml")

    print(RobotSetup.describe(spec))  # LLM-readable summary
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import mujoco
import numpy as np


# ── Robot hints (robot-specific metadata for force computation) ──

@dataclass
class RobotHints:
    """Robot-specific hints for force computation and motion mapping.

    Removes the need to hardcode body/geom names in the force computation
    layer. Each robot registers its own hints, including how vision skeleton
    segments map to its joints.
    """
    torso_body: str = "torso"
    ground_contact_geoms: list[str] = field(default_factory=lambda: ["right_foot", "left_foot"])
    height_qpos_index: Optional[int] = 2     # qpos index for vertical height (None for arms)
    actuator_type: str = "torque"             # "torque" or "position"
    base_body: Optional[str] = None           # for arms: the fixed base body name
    segment_map: Optional[dict] = None        # skeleton segment → [(joint_name, scale), ...]


# ── Robot registry ──

_ROBOT_REGISTRY: dict[str, dict] = {}


def register_robot(name: str, xml_path: str, hints: Optional[RobotHints] = None):
    """Register a robot name -> XML path mapping.

    Args:
        name: Human-friendly robot name (e.g., "so101")
        xml_path: Path to the robot's MuJoCo XML file (absolute or relative).
        hints: Robot-specific hints for force computation.
    """
    _ROBOT_REGISTRY[name] = {
        "xml_path": str(Path(xml_path).resolve()),
        "hints": hints or RobotHints(),
    }


def list_robots() -> list[str]:
    """Return all registered robot names."""
    return list(_ROBOT_REGISTRY.keys())


def get_robot_hints(name: str) -> RobotHints:
    """Get the RobotHints for a registered robot.

    Raises:
        KeyError: If robot name is not registered.
    """
    if name not in _ROBOT_REGISTRY:
        available = ", ".join(_ROBOT_REGISTRY.keys())
        raise KeyError(f"Unknown robot '{name}'. Available: {available}")
    return _ROBOT_REGISTRY[name]["hints"]


@dataclass
class BodyInfo:
    name: str
    id: int
    mass: float                  # kg
    inertia: np.ndarray          # (3,) diagonal in body frame
    parent_id: int
    parent_name: str
    children: list[str] = field(default_factory=list)


@dataclass
class JointInfo:
    name: str
    id: int
    body_name: str
    axis: np.ndarray             # (3,) rotation axis in body frame
    range_rad: tuple[float, float]
    range_deg: tuple[float, float]
    damping: float
    stiffness: float
    armature: float
    qpos_index: int              # index into data.qpos
    qvel_index: int              # index into data.qvel


@dataclass
class ActuatorInfo:
    name: str
    id: int
    joint_name: str
    joint_id: int
    gear_ratio: float
    ctrl_range: tuple[float, float]
    max_torque: float            # gear * max(abs(ctrl_range))


@dataclass
class TendonInfo:
    name: str
    joints: list[tuple[str, float]]  # [(joint_name, coefficient), ...]


@dataclass
class RobotSpec:
    name: str
    xml_path: str
    gravity: np.ndarray                    # (3,)
    timestep: float
    bodies: dict[str, BodyInfo] = field(default_factory=dict)
    joints: dict[str, JointInfo] = field(default_factory=dict)
    actuators: dict[str, ActuatorInfo] = field(default_factory=dict)
    tendons: list[TendonInfo] = field(default_factory=list)
    n_actuators: int = 0
    total_mass: float = 0.0
    joint_names: list[str] = field(default_factory=list)  # ordered by actuator index
    joint_qpos_start: int = 0       # first hinge qpos index
    joint_qvel_start: int = 0       # first hinge qvel index


class RobotSetup:
    """Extract physical properties from a MuJoCo model."""

    @staticmethod
    def from_name(robot_name: str) -> tuple[RobotSpec, mujoco.MjModel, mujoco.MjData]:
        """Load a robot by registered name.

        Args:
            robot_name: Name from the registry (e.g., "so101")

        Raises:
            KeyError: If robot name is not registered.
            FileNotFoundError: If XML file doesn't exist at expected path.
        """
        if robot_name not in _ROBOT_REGISTRY:
            available = ", ".join(_ROBOT_REGISTRY.keys())
            raise KeyError(f"Unknown robot '{robot_name}'. Available: {available}")

        xml_path = Path(_ROBOT_REGISTRY[robot_name]["xml_path"])

        if not xml_path.exists():
            raise FileNotFoundError(f"Robot XML not found: {xml_path}")

        return RobotSetup.from_xml(str(xml_path))

    @staticmethod
    def from_xml(xml_path: str) -> tuple[RobotSpec, mujoco.MjModel, mujoco.MjData]:
        """Load model from XML, create data, extract spec."""
        xml_path = str(Path(xml_path).resolve())
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)

        # Step once to populate derived quantities
        mujoco.mj_forward(model, data)

        spec = RobotSetup.from_model(model, xml_path=xml_path)
        return spec, model, data

    @staticmethod
    def from_model(model: mujoco.MjModel, xml_path: str = "") -> RobotSpec:
        """Extract spec from an already-loaded model."""
        spec = RobotSpec(
            name=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MODEL, 0) or "unknown",
            xml_path=xml_path,
            gravity=model.opt.gravity.copy(),
            timestep=float(model.opt.timestep),
        )

        # ── Bodies ──
        for i in range(model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name is None:
                name = f"body_{i}"
            parent_id = int(model.body_parentid[i])
            parent_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, parent_id) or "world"

            spec.bodies[name] = BodyInfo(
                name=name,
                id=i,
                mass=float(model.body_mass[i]),
                inertia=model.body_inertia[i].copy(),
                parent_id=parent_id,
                parent_name=parent_name if i > 0 else "",
            )

        # Fill children lists
        for name, body in spec.bodies.items():
            if body.parent_id > 0:
                parent_name = body.parent_name
                if parent_name in spec.bodies:
                    spec.bodies[parent_name].children.append(name)

        spec.total_mass = float(np.sum(model.body_mass))

        # ── Joints (hinge only) ──
        for i in range(model.njnt):
            jnt_type = int(model.jnt_type[i])
            if jnt_type != mujoco.mjtJoint.mjJNT_HINGE:
                continue

            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name is None:
                continue

            body_id = int(model.jnt_bodyid[i])
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body_{body_id}"

            lo_rad, hi_rad = float(model.jnt_range[i, 0]), float(model.jnt_range[i, 1])

            spec.joints[name] = JointInfo(
                name=name,
                id=i,
                body_name=body_name,
                axis=model.jnt_axis[i].copy(),
                range_rad=(lo_rad, hi_rad),
                range_deg=(float(np.degrees(lo_rad)), float(np.degrees(hi_rad))),
                damping=float(model.dof_damping[model.jnt_dofadr[i]]),
                stiffness=float(model.jnt_stiffness[i]),
                armature=float(model.dof_armature[model.jnt_dofadr[i]]),
                qpos_index=int(model.jnt_qposadr[i]),
                qvel_index=int(model.jnt_dofadr[i]),
            )

        # ── Actuators ──
        joint_names_ordered = []
        qpos_start = None
        qvel_start = None

        for i in range(model.nu):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if name is None:
                name = f"actuator_{i}"

            jnt_id = int(model.actuator_trnid[i, 0])
            jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id) or f"joint_{jnt_id}"
            gear = float(model.actuator_gear[i, 0])
            ctrl_lo = float(model.actuator_ctrlrange[i, 0])
            ctrl_hi = float(model.actuator_ctrlrange[i, 1])
            max_torque = gear * max(abs(ctrl_lo), abs(ctrl_hi))

            spec.actuators[name] = ActuatorInfo(
                name=name,
                id=i,
                joint_name=jnt_name,
                joint_id=jnt_id,
                gear_ratio=gear,
                ctrl_range=(ctrl_lo, ctrl_hi),
                max_torque=max_torque,
            )
            joint_names_ordered.append(jnt_name)

            # Track qpos/qvel start indices
            if jnt_name in spec.joints:
                ji = spec.joints[jnt_name]
                if qpos_start is None or ji.qpos_index < qpos_start:
                    qpos_start = ji.qpos_index
                if qvel_start is None or ji.qvel_index < qvel_start:
                    qvel_start = ji.qvel_index

        spec.n_actuators = model.nu
        spec.joint_names = joint_names_ordered
        spec.joint_qpos_start = qpos_start or 0
        spec.joint_qvel_start = qvel_start or 0

        # ── Tendons ──
        for i in range(model.ntendon):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_TENDON, i)
            if name is None:
                name = f"tendon_{i}"

            # Get joint-tendon coupling coefficients
            adr = int(model.tendon_adr[i])
            n = int(model.tendon_num[i])
            joints_coefs = []
            for j in range(adr, adr + n):
                jnt_id = int(model.wrap_objid[j])
                coef = float(model.wrap_prm[j])
                jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id)
                if jnt_name:
                    joints_coefs.append((jnt_name, coef))

            if joints_coefs:
                spec.tendons.append(TendonInfo(name=name, joints=joints_coefs))

        return spec

    @staticmethod
    def describe(spec: RobotSpec) -> str:
        """Generate LLM-readable robot description for system prompts."""
        lines = [
            f"ROBOT: {spec.name}",
            f"Total mass: {spec.total_mass:.2f} kg",
            f"Gravity: [{spec.gravity[0]:.2f}, {spec.gravity[1]:.2f}, {spec.gravity[2]:.2f}] m/s^2",
            f"Timestep: {spec.timestep*1000:.1f} ms",
            f"Actuators: {spec.n_actuators}",
            "",
            "ACTUATED JOINTS (index: name, body, gear, ctrl_range, max_torque, joint_range):",
        ]

        for i, jname in enumerate(spec.joint_names):
            act = spec.actuators.get(jname)
            jnt = spec.joints.get(jname)
            if not act or not jnt:
                continue
            lines.append(
                f"  {i:2d}: {jname:<20s} body={jnt.body_name:<18s} "
                f"gear={act.gear_ratio:>5.0f}  ctrl=[{act.ctrl_range[0]:+.1f},{act.ctrl_range[1]:+.1f}]  "
                f"max={act.max_torque:>6.1f}Nm  "
                f"range=[{jnt.range_deg[0]:+.0f},{jnt.range_deg[1]:+.0f}]deg  "
                f"stiff={jnt.stiffness:.0f} damp={jnt.damping:.0f}"
            )

        # Body masses
        lines.append("")
        lines.append("BODY MASSES:")
        for name, body in spec.bodies.items():
            if body.mass > 0.001:
                lines.append(f"  {name:<20s} {body.mass:.3f} kg")

        # Tendons
        if spec.tendons:
            lines.append("")
            lines.append("TENDON COUPLINGS:")
            for t in spec.tendons:
                coupling_str = " + ".join(
                    f"{coef:+.1f}*{jn}" for jn, coef in t.joints
                )
                lines.append(f"  {t.name}: {coupling_str}")

        return "\n".join(lines)
