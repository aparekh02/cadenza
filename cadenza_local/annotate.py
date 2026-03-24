"""Image annotation tool — skeleton overlays from MediaPipe pose detection.

Core cadenza tool for visualizing detected poses on motion images. Draws
accurate joint positions, limb segments, angles, object detections, and
phase labels using PIL. Supports both single-arm and dual-arm modes.

Usage:
    from cadenza_local.pose import PoseDetector
    from cadenza_local.annotate import annotate_sequence

    detector = PoseDetector()

    # Both arms
    results = detector.detect_sequence("path/to/images/", both_arms=True)
    paths = annotate_sequence(results, "path/to/images/", "output/annotated/")
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from cadenza_local.pose import PoseResult


# Single-arm skeleton segments
SKELETON_SEGMENTS = [
    ("base", "shoulder"),
    ("shoulder", "elbow"),
    ("elbow", "wrist"),
    ("wrist", "fingertip"),
]

# Angle labels for single-arm segments
SEGMENT_ANGLE_MAP = {
    ("base", "shoulder"): "base_rotation",
    ("shoulder", "elbow"): "upper_arm",
    ("elbow", "wrist"): "forearm",
    ("wrist", "fingertip"): "hand",
}

# Dual-arm skeleton segments (prefix → segments)
DUAL_ARM_SEGMENTS = {
    "left_": [
        ("base", "left_shoulder"),
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"),
        ("left_wrist", "left_fingertip"),
    ],
    "right_": [
        ("base", "right_shoulder"),
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"),
        ("right_wrist", "right_fingertip"),
    ],
}

# Dual-arm angle labels
DUAL_ANGLE_MAP = {
    ("base", "left_shoulder"): "left_base_rotation",
    ("left_shoulder", "left_elbow"): "left_upper_arm",
    ("left_elbow", "left_wrist"): "left_forearm",
    ("left_wrist", "left_fingertip"): "left_hand",
    ("base", "right_shoulder"): "right_base_rotation",
    ("right_shoulder", "right_elbow"): "right_upper_arm",
    ("right_elbow", "right_wrist"): "right_forearm",
    ("right_wrist", "right_fingertip"): "right_hand",
}

# Colors (R, G, B)
LEFT_ARM_COLOR = (100, 180, 255)     # blue
RIGHT_ARM_COLOR = (255, 100, 80)     # red
SINGLE_ARM_COLOR = (255, 80, 80)     # red (same as right for single arm)
JOINT_COLOR = (0, 255, 100)          # green
ANGLE_COLOR = (255, 255, 0)          # yellow
OBJECT_COLOR = (80, 180, 255)        # cyan
GRASP_COLOR = (255, 120, 0)          # orange

# Finger landmark names (drawn smaller)
FINGER_JOINTS = {"index", "pinky", "thumb",
                 "left_index", "left_pinky", "left_thumb",
                 "right_index", "right_pinky", "right_thumb"}


def _is_both_arms(pose: PoseResult) -> bool:
    """Check if this pose has both arms tracked."""
    return pose.dominant_arm == "both" or "left_shoulder" in pose.joints


def annotate_frame(
    pose: PoseResult,
    output_path: str,
    objects: Optional[list[dict]] = None,
    phase_label: str = "",
) -> str:
    """Annotate a single image with pose overlay.

    Automatically detects dual-arm mode from the PoseResult joints.

    Args:
        pose: PoseResult from the pose detector.
        output_path: Where to save the annotated image.
        objects: Optional object detections [{name, cx, cy, w, h, interaction}].
        phase_label: Task phase label to display.

    Returns:
        Path to saved annotated image.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        raise ImportError("Pillow required for annotation: pip install Pillow")

    if not pose.image_path:
        raise ValueError("PoseResult has no image_path")

    img = Image.open(pose.image_path).convert("RGBA")
    w, h = img.size

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Load fonts
    try:
        font_size = max(12, h // 35)
        font_lg_size = max(16, h // 25)
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
        font_lg = ImageFont.truetype(
            "/System/Library/Fonts/Helvetica.ttc", font_lg_size
        )
    except (OSError, IOError):
        font = ImageFont.load_default()
        font_lg = font

    # Convert normalized joint positions to pixel coords
    jpx = {}
    for name, (jx, jy) in pose.joints.items():
        px = int(jx * w)
        py = int(jy * h)
        if 0 <= px <= w and 0 <= py <= h:
            jpx[name] = (px, py)

    dual = _is_both_arms(pose)

    if dual:
        _draw_dual_arm_skeleton(draw, jpx, pose, font)
    else:
        _draw_single_arm_skeleton(draw, jpx, pose, font)

    # Draw joint dots
    r = max(5, h // 70)
    for name, (px, py) in jpx.items():
        if name == "base":
            # Base gets a distinct marker
            draw.ellipse(
                [px - r, py - r, px + r, py + r],
                fill=(255, 255, 255, 200),
                outline=(100, 100, 100, 255),
                width=2,
            )
            _draw_label(draw, px + r + 3, py - r, "base", font, (200, 200, 200))
            continue

        if name in FINGER_JOINTS:
            sr = r // 2
            draw.ellipse(
                [px - sr, py - sr, px + sr, py + sr],
                fill=(200, 200, 100, 180),
            )
            continue

        # Pick color based on arm side
        if name.startswith("left_"):
            jc = LEFT_ARM_COLOR
        elif name.startswith("right_"):
            jc = RIGHT_ARM_COLOR
        else:
            jc = JOINT_COLOR

        draw.ellipse(
            [px - r, py - r, px + r, py + r],
            fill=jc + (230,),
            outline=(255, 255, 255, 255),
            width=2,
        )
        # Joint label (strip prefix for cleaner display)
        display_name = name
        for pfx in ("left_", "right_"):
            if name.startswith(pfx):
                display_name = name[len(pfx):]
                break
        _draw_label(draw, px + r + 3, py - r, display_name, font, (255, 255, 255))

    # Draw object boxes
    if objects:
        for obj in objects:
            _draw_object_box(draw, obj, w, h, font)

    # Header labels
    frame_label = f"Frame {pose.frame_number}"
    if phase_label:
        frame_label += f" | {phase_label.upper()}"
    arm_label = f"arm: {'both' if dual else pose.dominant_arm}"

    info_parts = [arm_label]
    if dual:
        lg = pose.angles.get("left_gripper", 0)
        rg = pose.angles.get("right_gripper", 0)
        info_parts.append(f"L grip: {lg:.0f}")
        info_parts.append(f"R grip: {rg:.0f}")
    else:
        info_parts.append(f"grip: {pose.gripper_openness:.0f}")
        wr = pose.angles.get("wrist_roll")
        if wr is not None:
            info_parts.append(f"wrist_roll: {wr:+.0f}")

    # Top-left header
    _draw_label(draw, 10, 10, frame_label, font_lg, (255, 255, 255), bg_alpha=200)

    # Bottom info
    info_text = "  ".join(info_parts)
    bbox = draw.textbbox((0, 0), info_text, font=font)
    tw = bbox[2] - bbox[0]
    _draw_label(draw, w - tw - 15, h - 30, info_text, font, ANGLE_COLOR, bg_alpha=180)

    # Composite
    result = Image.alpha_composite(img, overlay).convert("RGB")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path, quality=95)
    return output_path


def _draw_single_arm_skeleton(draw, jpx, pose, font):
    """Draw skeleton segments for single-arm mode."""
    for start_j, end_j in SKELETON_SEGMENTS:
        if start_j not in jpx or end_j not in jpx:
            continue
        sx, sy = jpx[start_j]
        ex, ey = jpx[end_j]
        draw.line([(sx, sy), (ex, ey)], fill=SINGLE_ARM_COLOR + (220,), width=3)

        angle_key = SEGMENT_ANGLE_MAP.get((start_j, end_j))
        if angle_key and angle_key in pose.angles:
            mx = (sx + ex) // 2
            my = (sy + ey) // 2
            label = f"{angle_key}: {pose.angles[angle_key]:+.0f}"
            _draw_label(draw, mx + 5, my - 8, label, font, ANGLE_COLOR)


def _draw_dual_arm_skeleton(draw, jpx, pose, font):
    """Draw skeleton segments for both arms with different colors."""
    for prefix, segments in DUAL_ARM_SEGMENTS.items():
        color = LEFT_ARM_COLOR if prefix == "left_" else RIGHT_ARM_COLOR
        arm_tag = "L" if prefix == "left_" else "R"

        for start_j, end_j in segments:
            if start_j not in jpx or end_j not in jpx:
                continue
            sx, sy = jpx[start_j]
            ex, ey = jpx[end_j]
            draw.line([(sx, sy), (ex, ey)], fill=color + (220,), width=3)

            angle_key = DUAL_ANGLE_MAP.get((start_j, end_j))
            if angle_key and angle_key in pose.angles:
                mx = (sx + ex) // 2
                my = (sy + ey) // 2
                # Strip prefix for compact label
                short_name = angle_key.replace(prefix, "")
                label = f"{arm_tag}:{short_name}: {pose.angles[angle_key]:+.0f}"
                _draw_label(draw, mx + 5, my - 8, label, font, color)


def annotate_sequence(
    poses: list[PoseResult],
    image_dir: str,
    output_dir: str,
    objects_per_frame: Optional[dict[int, list[dict]]] = None,
    phases: Optional[dict[int, str]] = None,
) -> list[str]:
    """Annotate a full sequence of motion images.

    Args:
        poses: List of PoseResult from detect_sequence().
        image_dir: Source image directory (for reference).
        output_dir: Where to save annotated images.
        objects_per_frame: Optional {frame_number: [object dicts]}.
        phases: Optional {frame_number: "phase_label"}.

    Returns:
        List of saved file paths.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    print(f"Annotating {len(poses)} frames...")

    for pose in poses:
        if not pose.joints:
            print(f"  Frame {pose.frame_number}: no pose detected, skipping")
            continue

        frame_objects = (objects_per_frame or {}).get(pose.frame_number)
        phase = (phases or {}).get(pose.frame_number, "")

        out_file = out_dir / (
            f"annotated_{pose.location}_{pose.frame_number}.png"
        )
        path = annotate_frame(pose, str(out_file), objects=frame_objects, phase_label=phase)
        saved.append(path)
        print(f"  Frame {pose.frame_number}: saved")

    print(f"Annotated {len(saved)} images → {out_dir}")
    return saved


def _draw_label(
    draw,
    x: int,
    y: int,
    text: str,
    font,
    color: tuple,
    bg_alpha: int = 160,
):
    """Draw text with a semi-transparent background for readability."""
    bbox = draw.textbbox((x, y), text, font=font)
    draw.rectangle(
        [bbox[0] - 2, bbox[1] - 1, bbox[2] + 2, bbox[3] + 1],
        fill=(0, 0, 0, bg_alpha),
    )
    draw.text((x, y), text, fill=color + (255,), font=font)


def _draw_object_box(
    draw,
    obj: dict,
    img_w: int,
    img_h: int,
    font,
):
    """Draw an object detection bounding box."""
    cx = int(obj.get("cx", 0.5) * img_w)
    cy = int(obj.get("cy", 0.5) * img_h)
    bw = int(obj.get("w", 0.08) * img_w)
    bh = int(obj.get("h", 0.15) * img_h)
    x1, y1 = cx - bw // 2, cy - bh // 2
    x2, y2 = cx + bw // 2, cy + bh // 2

    name = obj.get("name", "object")
    interaction = obj.get("interaction", "none")
    is_grasped = interaction in ("grasping", "holding", "lifting", "pouring")
    color = GRASP_COLOR if is_grasped else OBJECT_COLOR

    draw.rectangle([x1, y1, x2, y2], outline=color + (220,), width=2)
    label = name
    if interaction != "none":
        label += f" [{interaction}]"
    _draw_label(draw, x1, y1 - 18, label, font, color)
