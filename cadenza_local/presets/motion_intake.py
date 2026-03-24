"""Motion image intake — loading, analysis, and annotation.

Uses MediaPipe PoseDetector for accurate skeleton tracking and optionally
Groq Vision for object detection and interaction analysis. Produces:
1. FrameAnalysis records with real joint positions and computed angles
2. Annotated images with skeleton overlays (via cadenza.annotate)
3. Motion blueprints for basis storage
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from cadenza_local.pose import PoseDetector, PoseResult
from cadenza_local.annotate import annotate_sequence as _annotate_seq
from cadenza_local.presets.schemas import (
    FrameAnalysis,
    LimbKeypoint,
    LimbSegment,
    ObjectDetection,
    MotionBlueprint,
)


# Skeleton segment → (start_joint, end_joint) for single arm
SEGMENT_CONNECTIONS = {
    "base_rotation": ("base", "shoulder"),
    "upper_arm":     ("shoulder", "elbow"),
    "forearm":       ("elbow", "wrist"),
    "hand":          ("wrist", "fingertip"),
}

# For dual arm, prefix + base connections
DUAL_SEGMENT_CONNECTIONS = {
    "left_base_rotation": ("base", "left_shoulder"),
    "left_upper_arm":     ("left_shoulder", "left_elbow"),
    "left_forearm":       ("left_elbow", "left_wrist"),
    "left_hand":          ("left_wrist", "left_fingertip"),
    "right_base_rotation": ("base", "right_shoulder"),
    "right_upper_arm":     ("right_shoulder", "right_elbow"),
    "right_forearm":       ("right_elbow", "right_wrist"),
    "right_hand":          ("right_wrist", "right_fingertip"),
}


def analyze_motion_images(
    image_dir: str,
    object_names: Optional[list[str]] = None,
    task_description: str = "",
    dominant_arm: Optional[str] = None,
    both_arms: bool = False,
    detect_objects: bool = True,
) -> tuple[list[FrameAnalysis], MotionBlueprint, list[PoseResult]]:
    """Load and analyze motion images with MediaPipe pose detection.

    Args:
        image_dir: Path to directory with {location}_{number}.png images.
        object_names: Names of objects to detect in frames.
        task_description: Text describing the task (improves phase inference + object detection).
        dominant_arm: Force arm selection ("left"/"right"). Ignored if both_arms=True.
        both_arms: Track both arms with prefixed joint names.
        detect_objects: Whether to run Groq Vision for object detection.

    Returns:
        Tuple of (frame analyses, motion blueprint, raw pose results).
    """
    # 1. Pose detection (MediaPipe)
    detector = PoseDetector()
    poses = detector.detect_sequence(
        image_dir, dominant_arm=dominant_arm, both_arms=both_arms,
    )
    detector.close()

    # 2. Object detection (Groq Vision) — optional
    objects_per_frame: dict[int, list[dict]] = {}
    interactions_per_frame: dict[int, list[str]] = {}
    if detect_objects:
        objects_per_frame, interactions_per_frame = _detect_objects_in_images(
            image_dir, task_description, object_names,
        )

    # 3. Build FrameAnalysis records
    frames = _poses_to_frames(
        poses, task_description, both_arms, objects_per_frame, interactions_per_frame,
    )
    blueprint = _build_blueprint(frames, poses, task_description, both_arms)

    return frames, blueprint, poses


def _detect_objects_in_images(
    image_dir: str,
    task_description: str,
    object_names: Optional[list[str]] = None,
) -> tuple[dict[int, list[dict]], dict[int, list[str]]]:
    """Run Groq Vision scene extraction for object detection.

    Returns:
        (objects_per_frame, interactions_per_frame)
        objects_per_frame: {frame_num: [{name, cx, cy, w, h, interaction, shape}]}
        interactions_per_frame: {frame_num: ["grasping bottle", ...]}
    """
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("  GROQ_API_KEY not set — skipping object detection.")
        return {}, {}

    try:
        from cadenza_local.vision import load_motion_images, extract_scene
    except ImportError as e:
        print(f"  Vision module unavailable — skipping object detection: {e}")
        return {}, {}

    print("\n  Running Groq Vision for object detection...")
    images = load_motion_images(image_dir)

    expected_objects = None
    if object_names:
        expected_objects = [{"name": n} for n in object_names]

    try:
        _, scene = extract_scene(
            images,
            task_description=task_description or "task demonstration",
            expected_objects=expected_objects,
        )
    except Exception as e:
        print(f"  Object detection failed: {e}")
        return {}, {}

    # Convert SceneExtraction to per-frame dicts
    objects_per_frame: dict[int, list[dict]] = {}
    interactions_per_frame: dict[int, list[str]] = {}

    # Build object lookup by name → average position
    obj_lookup = {}
    for sobj in scene.objects:
        # Convert world position back to approximate image fraction
        cx = sobj.position[0] / 0.8 + 0.5  # inverse of world mapping
        cy = 0.5 + sobj.position[1] / 0.12
        w_frac = sobj.size[0] / 0.4 if len(sobj.size) > 0 else 0.05
        h_frac = sobj.size[1] * 2 / 0.9 if len(sobj.size) > 1 else 0.15
        # Clamp to valid range
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        obj_lookup[sobj.name] = {
            "name": sobj.name,
            "cx": round(cx, 3),
            "cy": round(cy, 3),
            "w": round(max(0.02, min(0.3, w_frac)), 3),
            "h": round(max(0.02, min(0.5, h_frac)), 3),
            "shape": sobj.shape,
        }

    # Map interactions to frames
    for ix in scene.interactions:
        obj_info = obj_lookup.get(ix.object_name)
        if obj_info is None:
            continue

        if ix.frame not in objects_per_frame:
            objects_per_frame[ix.frame] = []
        if ix.frame not in interactions_per_frame:
            interactions_per_frame[ix.frame] = []

        # Add object with interaction
        obj_entry = dict(obj_info)
        obj_entry["interaction"] = ix.action
        objects_per_frame[ix.frame].append(obj_entry)
        interactions_per_frame[ix.frame].append(f"{ix.action} {ix.object_name}")

    # Also add objects with "none" interaction for frames without interactions
    for img in images:
        if img.number not in objects_per_frame:
            objects_per_frame[img.number] = [
                dict(obj, interaction="none") for obj in obj_lookup.values()
            ]

    n_objs = len(scene.objects)
    n_ix = len(scene.interactions)
    print(f"  Detected {n_objs} objects, {n_ix} interactions across frames.\n")

    return objects_per_frame, interactions_per_frame


def annotate_images(
    image_dir: str,
    frames: list[FrameAnalysis],
    output_dir: str,
    poses: Optional[list[PoseResult]] = None,
    objects_per_frame: Optional[dict[int, list[dict]]] = None,
) -> list[str]:
    """Draw skeleton overlays on motion images and save to output_dir.

    If poses (PoseResult list) are provided, uses cadenza.annotate for
    accurate MediaPipe-based drawing.

    Args:
        image_dir: Source image directory.
        frames: Analyzed frames with keypoints and segments.
        output_dir: Where to save annotated images.
        poses: Optional raw PoseResult list for accurate rendering.
        objects_per_frame: Optional {frame_number: [object dicts]}.

    Returns:
        List of saved annotated image paths.
    """
    if poses:
        # Build phase map from frames
        phases = {f.frame_number: f.phase_label for f in frames if f.phase_label}
        return _annotate_seq(
            poses, image_dir, output_dir,
            objects_per_frame=objects_per_frame,
            phases=phases,
        )

    # Fallback: draw from FrameAnalysis data
    return _annotate_from_frames(image_dir, frames, output_dir)


def _poses_to_frames(
    poses: list[PoseResult],
    task_description: str = "",
    both_arms: bool = False,
    objects_per_frame: Optional[dict[int, list[dict]]] = None,
    interactions_per_frame: Optional[dict[int, list[str]]] = None,
) -> list[FrameAnalysis]:
    """Convert PoseResult list to FrameAnalysis list."""
    seg_map = DUAL_SEGMENT_CONNECTIONS if both_arms else SEGMENT_CONNECTIONS
    frames = []

    for pose in poses:
        if not pose.joints:
            continue

        keypoints = []
        for jname, (x, y) in pose.joints.items():
            vis = pose.visibility.get(jname, 0.8)
            keypoints.append(LimbKeypoint(name=jname, x=x, y=y, confidence=vis))

        segments = []
        for sname, (start_j, end_j) in seg_map.items():
            angle = pose.angles.get(sname, 0.0)
            segments.append(LimbSegment(
                name=sname, angle_deg=angle,
                start_joint=start_j, end_joint=end_j,
            ))

        # Object detections from Groq Vision
        obj_dets = []
        frame_objs = (objects_per_frame or {}).get(pose.frame_number, [])
        for obj in frame_objs:
            obj_dets.append(ObjectDetection(
                name=obj.get("name", "object"),
                shape=obj.get("shape", "cylinder"),
                center_x=obj.get("cx", 0.5),
                center_y=obj.get("cy", 0.5),
                width_frac=obj.get("w", 0.05),
                height_frac=obj.get("h", 0.15),
                interaction=obj.get("interaction", "none"),
            ))

        interactions = (interactions_per_frame or {}).get(pose.frame_number, [])
        phase = _infer_phase(pose, both_arms, interactions)

        frames.append(FrameAnalysis(
            frame_number=pose.frame_number,
            location=pose.location,
            keypoints=keypoints,
            segments=segments,
            objects_detected=obj_dets,
            interactions=interactions,
            phase_label=phase,
            description=f"Frame {pose.frame_number}: {phase}",
        ))

    return frames


def _infer_phase(
    pose: PoseResult,
    both_arms: bool = False,
    interactions: Optional[list[str]] = None,
) -> str:
    """Infer the task phase from pose angles, gripper state, and interactions."""
    # If we have Groq Vision interactions, use those directly
    if interactions:
        for ix in interactions:
            for action in ("pouring", "lifting", "grasping", "placing", "holding"):
                if action in ix.lower():
                    return action
        return "interacting"

    if both_arms:
        # Use the arm with most activity
        phases = []
        for prefix in ("left_", "right_"):
            gripper = pose.angles.get(f"{prefix}gripper", 45)
            forearm = pose.angles.get(f"{prefix}forearm", 0)
            upper_arm = pose.angles.get(f"{prefix}upper_arm", 0)
            wrist_roll = pose.angles.get(f"{prefix}wrist_roll", 0)
            phases.append(_classify_arm_phase(gripper, forearm, upper_arm, wrist_roll))
        # Pick the more active phase
        priority = ["pouring", "lifting", "gripping", "reaching", "positioning", "idle"]
        for p in priority:
            if p in phases:
                return p
        return phases[0]
    else:
        gripper = pose.gripper_openness
        forearm = pose.angles.get("forearm", 0)
        upper_arm = pose.angles.get("upper_arm", 0)
        wrist_roll = pose.angles.get("wrist_roll", 0)
        return _classify_arm_phase(gripper, forearm, upper_arm, wrist_roll)


def _classify_arm_phase(gripper, forearm, upper_arm, wrist_roll) -> str:
    """Classify a single arm's phase from its angles."""
    if gripper < 20:
        if abs(wrist_roll) > 30:
            return "pouring"
        if abs(upper_arm) > 40:
            return "lifting"
        return "gripping"
    if gripper > 60:
        if abs(forearm) > 30:
            return "reaching"
        return "idle"
    return "positioning"


def _build_blueprint(
    frames: list[FrameAnalysis],
    poses: list[PoseResult],
    task_description: str,
    both_arms: bool = False,
) -> MotionBlueprint:
    """Construct a MotionBlueprint from analyzed frames."""
    seg_names = list(DUAL_SEGMENT_CONNECTIONS.keys()) if both_arms else list(SEGMENT_CONNECTIONS.keys())

    angle_deltas: dict[str, list[float]] = {}
    for seg_name in seg_names:
        deltas = []
        for i in range(1, len(poses)):
            prev = poses[i - 1].angles.get(seg_name, 0)
            curr = poses[i].angles.get(seg_name, 0)
            deltas.append(round(curr - prev, 1))
        if deltas:
            angle_deltas[seg_name] = deltas

    key_moments = []
    phases = []
    prev_phase = ""
    for frame in frames:
        if frame.phase_label != prev_phase:
            if frame.phase_label:
                phases.append(frame.phase_label)
            key_moments.append({
                "frame": frame.frame_number,
                "phase": frame.phase_label,
                "description": frame.description,
            })
            prev_phase = frame.phase_label

    duration = max(0, len(frames) - 1) * 0.5

    return MotionBlueprint(
        frames=frames,
        total_frames=len(frames),
        angle_deltas=angle_deltas,
        key_moments=key_moments,
        task_phases=phases,
        duration_estimate_sec=duration,
    )


def blueprint_to_basis_records(
    blueprint: MotionBlueprint,
    user_id: str,
    preset_id: str,
    task_description: str = "",
) -> list[dict]:
    """Convert a MotionBlueprint into basis records for batch storage."""
    records = []

    phase_str = " -> ".join(blueprint.task_phases) if blueprint.task_phases else "unknown"
    records.append({
        "user_id": user_id,
        "category": "motion_blueprint",
        "content": (
            f"Motion sequence: {blueprint.total_frames} frames, "
            f"{len(blueprint.key_moments)} key moments. "
            f"Phases: {phase_str}. "
            f"Estimated duration: {blueprint.duration_estimate_sec:.1f}s. "
            f"Task: {task_description}"
        ),
        "data": {
            "total_frames": blueprint.total_frames,
            "task_phases": blueprint.task_phases,
            "key_moments": blueprint.key_moments,
            "duration_estimate_sec": blueprint.duration_estimate_sec,
            "angle_deltas": dict(blueprint.angle_deltas),
        },
        "source": "motion_images",
        "confidence": 0.75,
        "preset_id": preset_id,
    })

    for moment in blueprint.key_moments:
        records.append({
            "user_id": user_id,
            "category": "motion_blueprint",
            "content": (
                f"Key moment at frame {moment['frame']}: "
                f"phase={moment['phase']}. {moment.get('description', '')}"
            ),
            "data": moment,
            "source": "motion_images",
            "confidence": 0.7,
            "preset_id": preset_id,
        })

    for seg, deltas in blueprint.angle_deltas.items():
        if not deltas:
            continue
        total = sum(deltas)
        max_delta = max(abs(d) for d in deltas) if deltas else 0
        records.append({
            "user_id": user_id,
            "category": "motion_blueprint",
            "content": (
                f"Segment '{seg}' motion: total delta {total:+.1f} deg over "
                f"{len(deltas)} transitions, max single delta {max_delta:.1f} deg."
            ),
            "data": {
                "segment": seg,
                "total_delta_deg": total,
                "max_delta_deg": max_delta,
                "deltas": deltas,
            },
            "source": "motion_images",
            "confidence": 0.75,
            "preset_id": preset_id,
        })

    return records


def _annotate_from_frames(
    image_dir: str,
    frames: list[FrameAnalysis],
    output_dir: str,
) -> list[str]:
    """Fallback annotation from FrameAnalysis data (no PoseResults)."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("Warning: Pillow not installed. Skipping annotation.")
        return []

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    src_path = Path(image_dir)
    saved = []

    for frame in frames:
        src_file = None
        for ext in ("png", "jpg", "jpeg"):
            p = src_path / f"{frame.location}_{frame.frame_number}.{ext}"
            if p.exists():
                src_file = p
                break
        if src_file is None:
            continue

        img = Image.open(src_file).convert("RGBA")
        w, h = img.size
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
            font_lg = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
        except (OSError, IOError):
            font = ImageFont.load_default()
            font_lg = font

        jpx = {}
        for kp in frame.keypoints:
            jpx[kp.name] = (int(kp.x * w), int(kp.y * h))

        for seg in frame.segments:
            sp = jpx.get(seg.start_joint)
            ep = jpx.get(seg.end_joint)
            if sp and ep:
                draw.line([sp, ep], fill=(255, 80, 80, 200), width=3)

        r = 6
        for kp in frame.keypoints:
            px, py = int(kp.x * w), int(kp.y * h)
            draw.ellipse(
                [px - r, py - r, px + r, py + r],
                fill=(0, 255, 100, 230),
                outline=(255, 255, 255, 255),
                width=2,
            )

        phase_text = f"Frame {frame.frame_number} | {frame.phase_label}"
        bbox = draw.textbbox((10, 10), phase_text, font=font_lg)
        draw.rectangle(
            [bbox[0] - 4, bbox[1] - 4, bbox[2] + 4, bbox[3] + 4],
            fill=(0, 0, 0, 180),
        )
        draw.text((10, 10), phase_text, fill=(255, 255, 255, 255), font=font_lg)

        result = Image.alpha_composite(img, overlay).convert("RGB")
        out_file = out_path / f"annotated_{frame.location}_{frame.frame_number}.png"
        result.save(out_file, quality=95)
        saved.append(str(out_file))

    return saved
