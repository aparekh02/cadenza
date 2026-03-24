"""Task preset and scene composition engine.

Converts 2D image-extracted scene information into 3D MuJoCo environments.
SceneBuilder composes a robot XML with detected scene objects (bottles, glasses,
etc.) into a single loadable MuJoCo model with freejoint objects and weld
constraints for grasping.

Usage:
    from cadenza_local.task import TaskPreset, SceneObject, SceneBuilder

    task = TaskPreset(
        task_description="bartending - picking up bottles and pouring drinks",
        expected_objects=[{"name": "bottle", "shape": "cylinder"}],
    )

    # After scene extraction from images:
    xml = SceneBuilder.build_scene_xml("robot.xml", scene_objects, grasp_pairs)
    model = mujoco.MjModel.from_xml_string(xml)
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import mujoco


# ── Data structures ──

@dataclass
class SceneObject:
    """A physical object to place in the MuJoCo scene."""
    name: str                           # unique id, e.g., "bottle_1"
    shape: str                          # "box", "cylinder", "sphere", "capsule"
    size: tuple[float, ...]             # MuJoCo geom size params (shape-dependent)
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)  # (x, y, z) — set from images if available
    rgba: tuple[float, float, float, float] = (0.8, 0.4, 0.2, 1.0)
    mass: float = 0.1                   # kg
    movable: bool = True                # True = freejoint (6DOF), False = static
    friction: tuple[float, float, float] = (1.0, 0.005, 0.0001)


@dataclass
class Interaction:
    """A detected interaction between the arm and a scene object."""
    frame: int                          # image frame number
    action: str                         # "reaching", "grasping", "holding", "lifting", "pouring", "placing"
    object_name: str                    # which SceneObject
    confidence: float = 0.0             # 0.0 - 1.0 from vision model


@dataclass
class SceneExtraction:
    """Complete scene understanding from the vision pipeline."""
    objects: list[SceneObject] = field(default_factory=list)
    interactions: list[Interaction] = field(default_factory=list)
    task_summary: str = ""              # LLM's summary of the task


@dataclass
class TaskPreset:
    """User-facing task configuration that defines the 3D scene and movement style.

    Defined outside cadenza (e.g., in examples/) and passed into run().
    The preset IS the source of truth for scene objects — not image extraction.
    Images are only used for arm movement (skeleton extraction).

    Args:
        task_description: What the robot is doing. Guides the learning loop.
        scene_objects: Physical objects to place in the MuJoCo scene.
            These must be sized for the robot's gripper.
        interaction_types: What kinds of contact happen (grasping, pouring, etc).
        speed: Movement speed multiplier. 0.1 = very slow/careful, 1.0 = fast.
            Lower values give more simulation steps per waypoint.
        behavior: Free-text instructions for HOW the robot should move.
            Injected into the controller's task prompt every step.
    """
    task_description: str
    scene_objects: list[SceneObject] | None = None
    interaction_types: list[str] | None = None
    speed: float = 0.3
    behavior: str = ""


# ── Scene XML builder ──

class SceneBuilder:
    """Generates a combined MuJoCo XML with robot + scene objects.

    Parses the robot XML, inserts scene object bodies (with freejoints
    for movable objects), and adds weld constraints for grasping.
    """

    @staticmethod
    def build_scene_xml(
        robot_xml_path: str,
        objects: list[SceneObject],
        grasp_pairs: list[tuple[str, str]] | None = None,
    ) -> str:
        """Generate combined MuJoCo XML string.

        Args:
            robot_xml_path: Path to the base robot XML file.
            objects: Scene objects to add to the world.
            grasp_pairs: List of (gripper_body_name, object_name) pairs
                         for weld constraints. Constraints start disabled.

        Returns:
            Complete XML string loadable with mujoco.MjModel.from_xml_string().
        """
        tree = ET.parse(robot_xml_path)
        root = tree.getroot()

        worldbody = root.find("worldbody")
        if worldbody is None:
            raise ValueError("Robot XML has no <worldbody>")

        # Insert scene objects into worldbody
        for obj in objects:
            body = ET.SubElement(worldbody, "body")
            body.set("name", obj.name)
            body.set("pos", f"{obj.position[0]} {obj.position[1]} {obj.position[2]}")

            if obj.movable:
                ET.SubElement(body, "freejoint", name=f"{obj.name}_free")

            geom = ET.SubElement(body, "geom")
            geom.set("name", f"{obj.name}_geom")
            geom.set("type", obj.shape)
            geom.set("size", " ".join(f"{s}" for s in obj.size))
            geom.set("rgba", " ".join(f"{c}" for c in obj.rgba))
            geom.set("mass", str(obj.mass))
            geom.set("friction", " ".join(f"{f}" for f in obj.friction))
            geom.set("condim", "4")  # full friction cone for grasping

            # Auto inertia from mass + shape
            inertial = ET.SubElement(body, "inertial")
            inertial.set("pos", "0 0 0")
            inertial.set("mass", str(obj.mass))
            i_val = obj.mass * 0.001  # rough approximation
            inertial.set("diaginertia", f"{i_val} {i_val} {i_val}")

        # Add weld constraints for grasping
        if grasp_pairs:
            equality = root.find("equality")
            if equality is None:
                equality = ET.SubElement(root, "equality")

            for gripper_body, obj_name in grasp_pairs:
                # Check that the object exists
                if not any(o.name == obj_name for o in objects):
                    continue
                weld = ET.SubElement(equality, "weld")
                weld.set("name", f"grasp_{obj_name}")
                weld.set("body1", gripper_body)
                weld.set("body2", obj_name)
                weld.set("active", "false")
                # Soft constraint parameters for stable grasping
                weld.set("solref", "0.02 1")
                weld.set("solimp", "0.9 0.95 0.001")

        return ET.tostring(root, encoding="unicode")


# ── Grasp manager ──

class GraspManager:
    """Runtime management of grasp weld constraints.

    Monitors contact between gripper geoms and object geoms.
    When sustained contact is detected, activates the weld constraint
    to simulate grasping. Deactivates when contact is lost.
    """

    CONTACT_THRESHOLD = 0.5     # N — minimum normal force to consider contact
    SUSTAIN_STEPS = 5           # consecutive contact steps before activating

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        grasp_pairs: list[tuple[str, str]],
        objects: list[SceneObject],
    ):
        self._model = model
        self._data = data
        self._grasp_map: dict[str, _GraspState] = {}

        for gripper_body, obj_name in grasp_pairs:
            # Find the weld constraint index
            eq_name = f"grasp_{obj_name}"
            eq_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, eq_name)
            if eq_id < 0:
                continue

            # Find geom IDs for contact detection
            obj_geom_name = f"{obj_name}_geom"
            obj_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, obj_geom_name)

            # Get all geoms belonging to the gripper body
            gripper_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, gripper_body)
            gripper_geom_ids = []
            for g in range(model.ngeom):
                if model.geom_bodyid[g] == gripper_body_id:
                    gripper_geom_ids.append(g)
            # Also include child body geoms (finger geoms)
            for b in range(model.nbody):
                if model.body_parentid[b] == gripper_body_id:
                    for g in range(model.ngeom):
                        if model.geom_bodyid[g] == b:
                            gripper_geom_ids.append(g)

            if obj_geom_id >= 0 and gripper_geom_ids:
                obj = next((o for o in objects if o.name == obj_name), None)
                self._grasp_map[obj_name] = _GraspState(
                    eq_id=eq_id,
                    obj_geom_id=obj_geom_id,
                    gripper_geom_ids=gripper_geom_ids,
                    obj_mass=obj.mass if obj else 0.1,
                )

            # Ensure constraint starts disabled
            if eq_id >= 0:
                data.eq_active[eq_id] = 0

    def update(self, data: mujoco.MjData) -> list[str]:
        """Check contacts and activate/deactivate grasps.

        Returns list of newly grasped object names.
        """
        newly_grasped = []

        for obj_name, state in self._grasp_map.items():
            has_contact = self._check_contact(data, state)

            if has_contact:
                state.contact_count += 1
                if not state.active and state.contact_count >= self.SUSTAIN_STEPS:
                    data.eq_active[state.eq_id] = 1
                    state.active = True
                    newly_grasped.append(obj_name)
            else:
                if state.active:
                    data.eq_active[state.eq_id] = 0
                    state.active = False
                state.contact_count = 0

        return newly_grasped

    def get_grasped_objects(self) -> list[tuple[str, float]]:
        """Return list of (object_name, mass) for currently grasped objects."""
        return [
            (name, s.obj_mass)
            for name, s in self._grasp_map.items()
            if s.active
        ]

    def _check_contact(self, data: mujoco.MjData, state: _GraspState) -> bool:
        """Check if any gripper geom is in contact with the object geom."""
        for i in range(data.ncon):
            c = data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)

            # Check if this contact involves the object and a gripper geom
            if g1 == state.obj_geom_id and g2 in state.gripper_geom_ids:
                return True
            if g2 == state.obj_geom_id and g1 in state.gripper_geom_ids:
                return True

        return False


@dataclass
class _GraspState:
    """Internal state for one grasp pair."""
    eq_id: int
    obj_geom_id: int
    gripper_geom_ids: list[int]
    obj_mass: float
    contact_count: int = 0
    active: bool = False
