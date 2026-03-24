"""PresetBuilder — main API for constructing and storing presets.

Orchestrates the full pipeline: motion images → skeleton extraction →
robot profile → object analysis → task decomposition → spatial reasoning →
basis storage in mem_alpha.

Usage:
    from cadenza_local.presets import PresetBuilder

    builder = PresetBuilder(user_id="bartender-agent")

    builder.add_motion_images("examples/motion_images/")
    builder.set_robot_spec(spec, hints)
    builder.add_object("campari_bottle", shape="cylinder", mass=0.8,
                        interactions=["grasping", "pouring"])
    builder.set_task(
        description="Make a Negroni cocktail",
        actions="Pick up Campari, pour into mixing glass, pick up gin, pour, "
                "pick up vermouth, pour, stir with bar spoon, pour into rocks glass",
    )

    # Analyze everything and store to basis
    preset = await builder.build()

    # Export annotated images
    builder.export_annotated("output/annotated/")

    # Export text summary
    builder.export_summary("output/preset_output.txt")
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional
from uuid import uuid4

from cadenza_local.robot_setup import RobotSpec, RobotHints
from cadenza_local.presets.schemas import (
    BasisPreset,
    MotionBlueprint,
    RobotProfile,
    ObjectProfile,
    TaskDirective,
    SpatialRelation,
    FrameAnalysis,
)
from cadenza_local.presets.motion_intake import (
    analyze_motion_images,
    annotate_images,
    blueprint_to_basis_records,
)
from cadenza_local.presets.robot_profile import (
    extract_robot_profile,
    profile_to_basis_records,
)
from cadenza_local.presets.analyzer import (
    analyze_task_text,
    analyze_spatial_relations,
    task_to_basis_records,
    objects_to_basis_records,
    spatial_to_basis_records,
)
from cadenza_local.presets.dynamics import compute_dynamics, dynamics_to_basis_records
from cadenza_local.presets.motor_profile import compute_motor_profile, motor_profile_to_basis_records


def parse_task_file(path: Path) -> dict:
    """Parse a task definition file into sections.

    Returns:
        {"config": {key: val}, "task": {key: val}, "objects": [dict, ...]}
    """
    text = path.read_text()
    sections: dict = {"config": {}, "task": {}, "objects": []}
    current_section = None

    # For multi-line values: key on first line, continuation indented.
    current_key = None
    current_val_lines: list[str] = []

    def _flush_kv():
        nonlocal current_key, current_val_lines
        if current_key and current_section in ("config", "task"):
            val = " ".join(line.strip() for line in current_val_lines).strip()
            sections[current_section][current_key] = val
        current_key = None
        current_val_lines = []

    for raw_line in text.splitlines():
        line = raw_line.strip()

        # Skip comments and blanks.
        if not line or line.startswith("#"):
            continue

        # Section headers.
        if line.startswith("[") and line.endswith("]"):
            _flush_kv()
            current_section = line[1:-1].strip().lower()
            continue

        # Object rows (pipe-delimited).
        if current_section == "objects" and "|" in line:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 5:
                continue
            name = parts[0]
            shape = parts[1] if len(parts) > 1 else "cylinder"
            size = tuple(float(x) for x in parts[2].split(",")) if len(parts) > 2 and parts[2] else (0.05, 0.15)
            position = tuple(float(x) for x in parts[3].split(",")) if len(parts) > 3 and parts[3] else (0.0, 0.0, 0.0)
            mass = float(parts[4]) if len(parts) > 4 and parts[4] else 0.1
            interactions = [x.strip() for x in parts[5].split(",")] if len(parts) > 5 and parts[5] else []
            properties = {}
            if len(parts) > 6 and parts[6]:
                for kv in parts[6].split(","):
                    if "=" in kv:
                        k, v = kv.split("=", 1)
                        properties[k.strip()] = v.strip()
            sections["objects"].append({
                "name": name, "shape": shape, "size": size,
                "position": position, "mass": mass,
                "interactions": interactions, "properties": properties,
            })
            continue

        # Key-value lines in config/task.
        if current_section in ("config", "task"):
            # Continuation line (starts with whitespace in the raw line).
            if raw_line[0] in (" ", "\t") and current_key:
                current_val_lines.append(line)
                continue
            # New key.
            _flush_kv()
            if ":" in line:
                key, val = line.split(":", 1)
                current_key = key.strip().lower()
                current_val_lines = [val.strip()]

    _flush_kv()
    return sections


class PresetBuilder:
    """Constructs a BasisPreset from raw inputs and stores to mem_alpha.

    Designed as a fluent builder — chain .add_*() calls, then await .build().
    """

    def __init__(self, user_id: str = "default"):
        self._user_id = user_id
        self._preset_id = str(uuid4())[:12]
        self._image_dir: Optional[str] = None
        self._robot_spec: Optional[RobotSpec] = None
        self._robot_hints: Optional[RobotHints] = None
        self._objects: list[ObjectProfile] = []
        self._task_description: str = ""
        self._task_actions: str = ""
        self._task_objects_context: str = ""
        self._speed: float = 0.3
        self._both_arms: bool = False

        # Populated after build()
        self._frames: list[FrameAnalysis] = []
        self._poses: list = []  # PoseResult list for annotation
        self._objects_per_frame: dict = {}  # frame_number → object dicts
        self._preset: Optional[BasisPreset] = None

    @property
    def preset_id(self) -> str:
        return self._preset_id

    # ── File-based construction ──

    @classmethod
    def from_file(cls, task_file: str) -> PresetBuilder:
        """Load a preset definition from a task file.

        File format (see examples/bartender/task.txt):
            [config]   — user_id, both_arms, images, speed
            [task]     — description, actions, context
            [objects]  — pipe-delimited rows: name|shape|size|pos|mass|interactions|properties

        Paths in the file are resolved relative to the file's directory.

        Args:
            task_file: Path to the .txt task definition file.

        Returns:
            Configured PresetBuilder ready for .build().
        """
        task_path = Path(task_file).resolve()
        if not task_path.is_file():
            raise FileNotFoundError(f"Task file not found: {task_path}")

        base_dir = task_path.parent
        sections = parse_task_file(task_path)

        # Config
        cfg = sections.get("config", {})
        builder = cls(user_id=cfg.get("user_id", "default"))

        if cfg.get("both_arms", "").lower() in ("true", "yes", "1"):
            builder.set_both_arms(True)

        images = cfg.get("images", "")
        if images:
            img_path = Path(images)
            if not img_path.is_absolute():
                img_path = base_dir / img_path
            if img_path.is_dir():
                builder.add_motion_images(str(img_path))

        speed = float(cfg.get("speed", "0.3"))

        # Task
        task = sections.get("task", {})
        if task.get("description"):
            builder.set_task(
                description=task["description"],
                actions=task.get("actions", ""),
                objects_context=task.get("context", ""),
                speed=speed,
            )

        # Objects
        for obj_dict in sections.get("objects", []):
            builder.add_object(**obj_dict)

        return builder

    # ── Input methods ──

    def add_motion_images(self, image_dir: str) -> PresetBuilder:
        """Set the motion images directory.

        Expects files named {location}_{timestamp}.png
        e.g., front_1.png, right_3.png
        """
        p = Path(image_dir).resolve()
        if not p.is_dir():
            raise FileNotFoundError(f"Image directory not found: {p}")
        self._image_dir = str(p)
        return self

    def set_both_arms(self, both: bool = True) -> PresetBuilder:
        """Enable dual-arm tracking (tracks both left and right arms)."""
        self._both_arms = both
        return self

    def set_robot_spec(
        self,
        spec: RobotSpec,
        hints: Optional[RobotHints] = None,
    ) -> PresetBuilder:
        """Set the robot specification (from RobotSetup.from_xml or from_name)."""
        self._robot_spec = spec
        self._robot_hints = hints
        return self

    def set_robot(
        self, robot_name: str,
    ) -> PresetBuilder:
        """Set the robot by registered name. Loads spec and hints."""
        from cadenza_local.robot_setup import RobotSetup, get_robot_hints
        spec, _, _ = RobotSetup.from_name(robot_name)
        hints = get_robot_hints(robot_name)
        return self.set_robot_spec(spec, hints)

    def add_object(
        self,
        name: str,
        shape: str = "cylinder",
        size: tuple[float, ...] = (0.05, 0.15),
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        mass: float = 0.1,
        interactions: Optional[list[str]] = None,
        properties: Optional[dict[str, str]] = None,
    ) -> PresetBuilder:
        """Add a scene object the robot interacts with."""
        self._objects.append(ObjectProfile(
            name=name,
            shape=shape,
            estimated_size=size,
            estimated_position=position,
            estimated_mass_kg=mass,
            interaction_types=interactions or [],
            properties=properties or {},
        ))
        return self

    def add_objects(self, objects: list[dict]) -> PresetBuilder:
        """Add multiple objects from dicts.

        Each dict: {name, shape, size?, position?, mass?, interactions?, properties?}
        """
        for obj in objects:
            self.add_object(
                name=obj["name"],
                shape=obj.get("shape", "cylinder"),
                size=tuple(obj.get("size", (0.05, 0.15))),
                position=tuple(obj.get("position", (0.0, 0.0, 0.0))),
                mass=obj.get("mass", 0.1),
                interactions=obj.get("interactions"),
                properties=obj.get("properties"),
            )
        return self

    def set_task(
        self,
        description: str,
        actions: str = "",
        objects_context: str = "",
        speed: float = 0.3,
    ) -> PresetBuilder:
        """Set the task description and context.

        Args:
            description: What the robot will be doing.
            actions: Specific actions the robot performs (comma/semicolon separated).
            objects_context: Additional context about objects and their roles.
            speed: Movement speed preference (0.1=slow, 1.0=fast).
        """
        self._task_description = description
        self._task_actions = actions
        self._task_objects_context = objects_context
        self._speed = speed
        return self

    # ── Build ──

    async def build(self) -> BasisPreset:
        """Run the full analysis pipeline and return the assembled preset.

        Returns:
            Complete BasisPreset with all extracted knowledge.
        """
        print(f"\n{'='*60}")
        print(f"Building preset: {self._preset_id}")
        print(f"{'='*60}\n")

        # 1. Analyze motion images
        blueprint = None
        if self._image_dir:
            print("[1/8] Analyzing motion images...")
            object_names = [o.name for o in self._objects] if self._objects else None
            self._frames, blueprint, self._poses = analyze_motion_images(
                self._image_dir,
                object_names=object_names,
                task_description=self._task_description,
                both_arms=self._both_arms,
            )
            # Store objects per frame for annotation
            for frame in self._frames:
                if frame.objects_detected:
                    self._objects_per_frame[frame.frame_number] = [
                        {
                            "name": o.name, "cx": o.center_x, "cy": o.center_y,
                            "w": o.width_frac, "h": o.height_frac,
                            "interaction": o.interaction, "shape": o.shape,
                        }
                        for o in frame.objects_detected
                    ]
            print(f"  {len(self._frames)} frames analyzed\n")
        else:
            print("[1/8] No motion images provided, skipping.\n")

        # 2. Extract robot profile
        robot_profile = None
        if self._robot_spec:
            print("[2/8] Extracting robot profile...")
            robot_profile = extract_robot_profile(
                self._robot_spec, self._robot_hints,
            )
            print(f"  {robot_profile.summary()}\n")
        else:
            print("[2/8] No robot spec provided, skipping.\n")

        # 3. Analyze task text
        print("[3/8] Analyzing task description...")
        task_directive = None
        if self._task_description:
            task_directive = analyze_task_text(
                self._task_description,
                self._objects,
                self._task_actions,
            )
            task_directive.speed_preference = self._speed
            actions_str = " -> ".join(task_directive.action_sequence)
            print(f"  Actions: {actions_str}")
            print(f"  Interactions: {task_directive.required_interactions}\n")
        else:
            print("  No task description provided.\n")

        # 4. Compute dynamics
        dynamics_profile = None
        if blueprint and self._frames:
            print("[4/8] Computing dynamics profile...")
            dynamics_profile = compute_dynamics(
                blueprint,
                self._frames,
                robot=robot_profile,
                objects=list(self._objects) if self._objects else None,
            )
            print(f"  Peak torque: {dynamics_profile.peak_torque_nm:.2f} Nm")
            print(f"  Total energy: {dynamics_profile.total_energy_joules:.2f} J")
            print(f"  Hardest phase: '{dynamics_profile.hardest_phase}'")
            print(f"  {dynamics_profile.n_segments} segments, {dynamics_profile.n_frames} frames\n")
        else:
            print("[4/8] No motion data for dynamics, skipping.\n")

        # 5. Analyze spatial relations
        print("[5/8] Computing spatial relations...")
        spatial = []
        if self._objects:
            robot_name = robot_profile.name if robot_profile else "robot"
            spatial = analyze_spatial_relations(self._objects, robot_name)
            print(f"  {len(spatial)} relations computed\n")
        else:
            print("  No objects to relate.\n")

        # 6. Compute motor profile
        motor = None
        if blueprint and self._frames:
            print("[6/8] Computing motor profile...")
            motor = compute_motor_profile(
                blueprint,
                self._frames,
                dynamics=dynamics_profile,
                robot=robot_profile,
                objects=list(self._objects) if self._objects else None,
                speed_preference=self._speed,
            )
            print(f"  Robot: {motor.robot_name}, {motor.n_actuators} actuators ({motor.actuator_type})")
            print(f"  Action dimension: {motor.action_dimension}")
            print(f"  Waypoints: {len(motor.waypoints)}, speed phases: {len(motor.speed_profile)}")
            print(f"  Spatial snapshots: {len(motor.spatial_snapshots)}\n")
        else:
            print("[6/8] No motion data for motor profile, skipping.\n")

        # 7. Assemble preset
        print("[7/7] Assembling preset...")
        self._preset = BasisPreset(
            preset_id=self._preset_id,
            motion=blueprint,
            robot=robot_profile,
            objects=list(self._objects),
            task=task_directive,
            spatial_relations=spatial,
            dynamics=dynamics_profile,
            motor=motor,
        )
        print(f"  Preset {self._preset_id} assembled.\n")

        print(f"{'='*60}")
        print(f"Preset build complete: {self._preset_id}")
        print(f"{'='*60}\n")

        return self._preset

    # ── Export methods ──

    def export_annotated(self, output_dir: str) -> list[str]:
        """Export annotated images with skeleton overlays.

        Must call build() first.

        Returns:
            List of saved file paths.
        """
        if not self._frames:
            print("No frames to annotate. Call build() first.")
            return []
        if not self._image_dir:
            print("No image directory set.")
            return []

        return annotate_images(
            self._image_dir, self._frames, output_dir,
            poses=self._poses if self._poses else None,
            objects_per_frame=self._objects_per_frame if self._objects_per_frame else None,
        )

    def export_summary(self, output_path: str) -> str:
        """Export a text summary of the preset to a file.

        Returns:
            The summary text.
        """
        if not self._preset:
            raise RuntimeError("Call build() before export_summary()")

        text = self._preset.summary()

        # Add dynamics profile
        if self._preset.dynamics:
            text += "\n\n" + self._preset.dynamics.summary()

        # Add motor profile
        if self._preset.motor:
            text += "\n\n" + self._preset.motor.summary()

        # Add detailed frame-by-frame analysis
        if self._frames:
            text += "\n\n=== Frame-by-Frame Analysis ===\n"
            for frame in self._frames:
                text += f"\nFrame {frame.frame_number} ({frame.location}):"
                text += f"\n  Phase: {frame.phase_label}"
                if frame.segments:
                    angles = ", ".join(
                        f"{s.name}={s.angle_deg:+.1f}deg" for s in frame.segments
                    )
                    text += f"\n  Angles: {angles}"
                if frame.keypoints:
                    joints = ", ".join(
                        f"{k.name}=({k.x:.2f},{k.y:.2f})" for k in frame.keypoints
                    )
                    text += f"\n  Joints: {joints}"
                if frame.interactions:
                    text += f"\n  Interactions: {', '.join(frame.interactions)}"
                if frame.objects_detected:
                    objs = ", ".join(o.name for o in frame.objects_detected)
                    text += f"\n  Objects: {objs}"

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(text)

        print(f"Preset summary saved to {output_path}")
        return text

    def get_preset(self) -> Optional[BasisPreset]:
        """Return the built preset (None if build() hasn't been called)."""
        return self._preset
