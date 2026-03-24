"""Vision pipeline — image-based motion capture and scene extraction using Groq Vision.

Loads a sequence of images (location_number.png), sends them to Groq Vision
to extract stick-figure skeletons (lines for bones, joints at intersections),
and computes the angle changes between sequential frames.

When a task description is provided, extract_scene() also detects scene objects
(bottles, glasses, etc.) and subject-object interactions (grasping, lifting,
pouring) from the same images in a single API call per frame.

Usage:
    images = load_motion_images("path/to/images/")

    # Skeleton only (backward compatible):
    poses = extract_skeleton(images)
    motion = compute_motion_sequence(poses)

    # Full scene extraction with task context:
    poses, scene = extract_scene(images, "bartending - picking up bottles")
    motion = compute_motion_sequence(poses)
    # scene.objects → SceneObject list for SceneBuilder
    # scene.interactions → per-frame action detections
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq


# ── Vision model ──

VISION_MODEL = "meta-llama/llama-4-maverick-17b-128e-instruct"

# Maximum image dimension (longest edge) before resizing for the API.
# Keeps base64 payloads well within Groq's 4 MB limit.
MAX_IMAGE_DIM = 1024


# ── Data structures ──

@dataclass
class MotionImage:
    """A single image from the motion capture sequence."""
    location: str      # camera angle: "front", "right", "back", "left", etc.
    number: int        # frame order (1-10)
    path: str          # absolute file path
    image_b64: str     # base64 encoded image data


@dataclass
class SkeletonPose:
    """Extracted stick-figure pose from a single image."""
    frame: int                                # frame number
    location: str                             # camera angle
    joints: dict[str, tuple[float, float]]    # joint_name → (x, y) normalized coords
    angles: dict[str, float]                  # segment_name → angle in degrees
    objects: dict[str, tuple[float, float]] = field(default_factory=dict)  # obj_name → (x, y) image coords
    grasping: list[str] = field(default_factory=list)  # object names being held


@dataclass
class MotionSequence:
    """Complete motion sequence derived from a set of images."""
    poses: list[SkeletonPose] = field(default_factory=list)
    angle_deltas: dict[str, list[float]] = field(default_factory=dict)


# ── Skeleton segments we ask the vision model to identify ──

SKELETON_SEGMENTS = [
    "base_rotation", # horizontal rotation of the arm base (left-right)
    "upper_arm",     # shoulder to elbow (pitch)
    "forearm",       # elbow to wrist (pitch)
    "hand",          # wrist to fingertip (pitch)
    "wrist_roll",    # rotation of the wrist (tilt of held object)
    "gripper",       # hand opening: 0 = closed, 90 = fully open
]

SKELETON_JOINTS = [
    "base",
    "shoulder",
    "elbow",
    "wrist",
    "fingertip",
]


# ── Skeleton cache ──

_DEFAULT_CACHE_DIR = Path(".cache/cadenza_vision")


def _compute_directory_hash(images: list[MotionImage]) -> str:
    """Compute a stable content hash of the image set.

    Hashes sorted filenames + file sizes + mtimes. Changes when files
    are added, removed, or modified.
    """
    hasher = hashlib.sha256()
    for img in sorted(images, key=lambda i: i.path):
        p = Path(img.path)
        st = p.stat()
        hasher.update(p.name.encode())
        hasher.update(str(st.st_size).encode())
        hasher.update(str(int(st.st_mtime)).encode())
    return hasher.hexdigest()[:16]


def _load_skeleton_cache(
    cache_dir: Path, content_hash: str
) -> list[SkeletonPose] | None:
    """Load cached skeleton poses if they exist and match the content hash."""
    cache_file = cache_dir / f"skeletons_{content_hash}.json"
    if not cache_file.exists():
        return None

    try:
        with open(cache_file, "r") as f:
            data = json.load(f)

        poses = []
        for entry in data:
            joints = {k: tuple(v) for k, v in entry["joints"].items()}
            angles = {k: float(v) for k, v in entry["angles"].items()}
            objects = {k: tuple(v) for k, v in entry.get("objects", {}).items()}
            grasping = entry.get("grasping", [])
            poses.append(SkeletonPose(
                frame=entry["frame"],
                location=entry["location"],
                joints=joints,
                angles=angles,
                objects=objects,
                grasping=grasping,
            ))
        return poses
    except Exception:
        return None


def _save_skeleton_cache(
    cache_dir: Path, content_hash: str, poses: list[SkeletonPose]
) -> None:
    """Save skeleton poses to cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"skeletons_{content_hash}.json"

    data = []
    for pose in poses:
        data.append({
            "frame": pose.frame,
            "location": pose.location,
            "joints": {k: list(v) for k, v in pose.joints.items()},
            "angles": pose.angles,
            "objects": {k: list(v) for k, v in pose.objects.items()},
            "grasping": pose.grasping,
        })

    with open(cache_file, "w") as f:
        json.dump(data, f, indent=2)


# ── Image loading ──

def _resize_and_encode(path: str, max_dim: int = MAX_IMAGE_DIM) -> str:
    """Read an image, shrink if needed, convert RGBA→RGB, and return base64 JPEG.

    Keeps the longest edge ≤ *max_dim* so the base64 payload stays
    well within Groq's 4 MB limit.
    """
    from PIL import Image as PILImage

    img = PILImage.open(path)

    # Drop alpha channel — API doesn't need transparency.
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGB")

    # Down-scale large images.
    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), PILImage.LANCZOS)

    # Encode as JPEG for a much smaller payload.
    import io
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def load_motion_images(directory: str) -> list[MotionImage]:
    """Load motion images from a directory.

    Expects filenames in the format: {location}_{number}.png
    Examples: front_1.png, right_3.png, back_5.png

    Args:
        directory: Path to directory containing motion images.

    Returns:
        List of MotionImage objects, sorted by frame number.
    """
    dir_path = Path(directory).resolve()
    if not dir_path.is_dir():
        raise FileNotFoundError(f"Image directory not found: {dir_path}")

    pattern = re.compile(r"^(.+)_(\d+)\.(png|jpg|jpeg)$", re.IGNORECASE)
    images: list[MotionImage] = []

    for entry in sorted(dir_path.iterdir()):
        if not entry.is_file():
            continue
        match = pattern.match(entry.name)
        if not match:
            continue

        location = match.group(1).lower()
        number = int(match.group(2))

        # Resize, convert RGBA→RGB, and base64-encode as JPEG.
        b64 = _resize_and_encode(str(entry))

        images.append(MotionImage(
            location=location,
            number=number,
            path=str(entry),
            image_b64=b64,
        ))

    images.sort(key=lambda img: img.number)

    if not images:
        raise ValueError(
            f"No motion images found in {dir_path}. "
            f"Expected format: location_number.png (e.g., front_1.png)"
        )

    print(f"Loaded {len(images)} motion images from {dir_path}")
    for img in images:
        print(f"  {img.location}_{img.number} ({Path(img.path).name})")

    return images


# ── Groq Vision skeleton extraction ──

def _build_skeleton_prompt(object_names: list[str] | None = None) -> str:
    """Build the prompt that instructs Groq Vision to extract skeleton + objects."""
    objects_section = ""
    objects_json = ""
    if object_names:
        names_str = ", ".join(object_names)
        objects_section = f"""
Also locate these objects in the image: {names_str}
For each visible object, report its center (x, y) as image fractions (0.0-1.0).
If the subject is touching, holding, or grasping an object, note it.
"""
        obj_entries = ", ".join(f'"{n}": [x, y]' for n in object_names)
        objects_json = f""",
  "objects": {{
    {obj_entries}
  }},
  "grasping": ["list of object names being held or touched"]"""

    return f"""Analyze this image of a person's arm performing a task.

Extract the ARM skeleton with ALL 6 degrees of freedom:
- Joint positions: base (arm root/shoulder mount), shoulder, elbow, wrist, fingertip
- Report each joint's (x, y) as image fractions (0.0-1.0, top-left = 0,0)

Measure these 6 angles:
- base_rotation: how far the arm base rotates left/right from center (0° = centered, positive = right, negative = left)
- upper_arm: shoulder-to-elbow angle from horizontal (0° = right, 90° = up, -90° = down)
- forearm: elbow-to-wrist angle from horizontal
- hand: wrist-to-fingertip angle from horizontal
- wrist_roll: rotation of the wrist/hand about the forearm axis (0° = neutral, positive = clockwise when looking down the arm). If a cup or bottle is being tilted, this angle shows the tilt.
- gripper: how open the hand is (0 = fully closed/gripping, 45 = relaxed, 90 = fully open)
{objects_section}
Return ONLY a JSON object in this exact format:
{{
  "joints": {{
    "base": [x, y],
    "shoulder": [x, y],
    "elbow": [x, y],
    "wrist": [x, y],
    "fingertip": [x, y]
  }},
  "angles": {{
    "base_rotation": degrees,
    "upper_arm": degrees,
    "forearm": degrees,
    "hand": degrees,
    "wrist_roll": degrees,
    "gripper": degrees
  }}{objects_json}
}}

Return ONLY the JSON. No explanation."""


def _parse_skeleton_response(reply: str, frame: int, location: str) -> SkeletonPose:
    """Parse the Groq Vision JSON response into a SkeletonPose."""
    json_match = re.search(r"\{[\s\S]*\}", reply)
    if not json_match:
        raise ValueError(f"No JSON found in vision response for frame {frame}")

    data = json.loads(json_match.group())

    joints = {}
    for jname, coords in data.get("joints", {}).items():
        if isinstance(coords, list) and len(coords) == 2:
            try:
                joints[jname] = (float(coords[0]), float(coords[1]))
            except (TypeError, ValueError):
                pass

    angles = {}
    for sname, angle in data.get("angles", {}).items():
        if angle is not None:
            try:
                angles[sname] = float(angle)
            except (TypeError, ValueError):
                angles[sname] = 0.0

    objects = {}
    for oname, coords in data.get("objects", {}).items():
        if isinstance(coords, list) and len(coords) == 2:
            try:
                objects[oname] = (float(coords[0]), float(coords[1]))
            except (TypeError, ValueError):
                pass

    grasping = []
    raw_grasp = data.get("grasping", [])
    if isinstance(raw_grasp, list):
        grasping = [str(g) for g in raw_grasp]

    return SkeletonPose(
        frame=frame,
        location=location,
        joints=joints,
        angles=angles,
        objects=objects,
        grasping=grasping,
    )


def extract_skeleton(
    images: list[MotionImage],
    object_names: list[str] | None = None,
    cache_dir: str | Path | None = _DEFAULT_CACHE_DIR,
) -> list[SkeletonPose]:
    """Extract stick-figure skeletons from motion images using Groq Vision.

    Extracts all 6 joint angles. When object_names are provided, also extracts
    object positions and grasping state from each frame.

    Results are cached to disk keyed by a hash of the image directory contents.
    Subsequent runs with the same images skip the API entirely.

    Args:
        images: List of MotionImage objects.
        object_names: Names of objects to locate in images (from TaskPreset).
        cache_dir: Cache directory path. None disables caching.

    Returns:
        List of SkeletonPose objects, one per image, ordered by frame number.
    """
    # ── Check cache ──
    # Include object_names in cache key so different tasks get separate caches
    cache_extra = "_".join(sorted(object_names)) if object_names else ""
    content_hash = _compute_directory_hash(images)
    if cache_extra:
        combined = f"{content_hash}_{cache_extra}"
        content_hash = hashlib.md5(combined.encode()).hexdigest()[:16]

    if cache_dir is not None:
        cache_path = Path(cache_dir)
        cached = _load_skeleton_cache(cache_path, content_hash)
        if cached is not None:
            print(f"  Loaded {len(cached)} cached skeleton poses (hash={content_hash})")
            for pose in cached:
                print(f"    {pose.location}_{pose.frame}: Angles={pose.angles}")
                if pose.objects:
                    print(f"      Objects: {list(pose.objects.keys())}")
                if pose.grasping:
                    print(f"      Grasping: {pose.grasping}")
            return cached

    # ── API extraction ──
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not found in environment or .env")

    client = Groq(api_key=api_key)
    prompt = _build_skeleton_prompt(object_names=object_names)
    poses: list[SkeletonPose] = []

    for img in images:
        print(f"  Extracting skeleton: {img.location}_{img.number}...")

        # Images are always JPEG after _resize_and_encode.
        mime = "image/jpeg"

        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime};base64,{img.image_b64}"
                    },
                },
            ],
        }]

        try:
            response = client.chat.completions.create(
                model=VISION_MODEL,
                messages=messages,
                temperature=0.1,
                max_tokens=800,
                response_format={"type": "json_object"},
            )

            reply = response.choices[0].message.content.strip()
            pose = _parse_skeleton_response(reply, img.number, img.location)
            poses.append(pose)
            print(f"    Angles: {pose.angles}")
            if pose.objects:
                print(f"    Objects: {pose.objects}")
            if pose.grasping:
                print(f"    Grasping: {pose.grasping}")

        except Exception as e:
            print(f"    ERROR extracting frame {img.number}: {e}")
            poses.append(SkeletonPose(
                frame=img.number,
                location=img.location,
                joints={},
                angles={},
            ))

    poses.sort(key=lambda p: p.frame)

    if cache_dir is not None:
        _save_skeleton_cache(Path(cache_dir), content_hash, poses)
        print(f"  Cached {len(poses)} skeleton poses (hash={content_hash})")

    print(f"Extracted {len(poses)} skeleton poses")
    return poses


# ── Motion sequence computation ──

def compute_motion_sequence(poses: list[SkeletonPose]) -> MotionSequence:
    """Compute angle changes between sequential frames.

    For each arm segment, calculates the change in angle from one frame
    to the next. These deltas represent the motion that needs to be
    replicated by the robot arm.

    Args:
        poses: List of SkeletonPose objects, ordered by frame number.

    Returns:
        MotionSequence with poses and per-segment angle deltas.
    """
    if not poses:
        return MotionSequence()

    # Sort by frame number
    sorted_poses = sorted(poses, key=lambda p: p.frame)

    # Compute angle deltas between consecutive frames
    angle_deltas: dict[str, list[float]] = {seg: [] for seg in SKELETON_SEGMENTS}

    for i in range(1, len(sorted_poses)):
        prev = sorted_poses[i - 1]
        curr = sorted_poses[i]

        for segment in SKELETON_SEGMENTS:
            prev_angle = prev.angles.get(segment)
            curr_angle = curr.angles.get(segment)

            if prev_angle is not None and curr_angle is not None:
                delta = curr_angle - prev_angle
                # Normalize to [-180, 180]
                while delta > 180:
                    delta -= 360
                while delta < -180:
                    delta += 360
                angle_deltas[segment].append(delta)
            else:
                # Missing data — assume no change
                angle_deltas[segment].append(0.0)

    motion = MotionSequence(
        poses=sorted_poses,
        angle_deltas=angle_deltas,
    )

    # Print summary
    print(f"\nMotion sequence: {len(sorted_poses)} frames, "
          f"{len(sorted_poses) - 1} transitions")
    for seg, deltas in angle_deltas.items():
        total = sum(deltas)
        print(f"  {seg}: total delta = {total:+.1f}°, "
              f"per-frame = [{', '.join(f'{d:+.1f}' for d in deltas)}]")

    return motion


# ── Scene extraction (objects + interactions from images) ──

def _build_scene_prompt(
    task_description: str,
    expected_objects: list[dict] | None = None,
) -> str:
    """Build prompt that extracts arm skeleton + scene objects + interactions."""
    joints_str = ", ".join(SKELETON_JOINTS)
    segments_str = ", ".join(SKELETON_SEGMENTS)

    object_hints = ""
    if expected_objects:
        names = ", ".join(o.get("name", "object") for o in expected_objects)
        object_hints = f"\nObjects likely in scene: {names}"

    return f"""Analyze this image of a person's arm performing a task.
Task context: {task_description}
{object_hints}

Extract TWO things:

1. ARM SKELETON — for the arm performing the task:
   Segments: {segments_str}
   Joints: {joints_str}
   Report joint (x, y) positions as fractions of image dimensions (0.0-1.0, top-left=0,0).
   Report segment angles in degrees relative to horizontal (0°=right, 90°=up, -90°=down).

2. SCENE OBJECTS — every distinct physical object the arm interacts with or that is relevant to the task:
   For each object report:
   - name: descriptive identifier (e.g., "red_bottle", "glass_1", "shaker")
   - shape: best-fit primitive — "cylinder", "box", or "sphere"
   - center: [x, y] position as fraction of image (0.0-1.0)
   - height_frac: approximate height as fraction of image height
   - width_frac: approximate width as fraction of image width
   - interaction: what the arm is doing with it right now — one of:
     "none", "reaching", "grasping", "holding", "lifting", "pouring", "placing"

Return ONLY a JSON object in this exact format:
{{
  "joints": {{"shoulder": [x, y], "elbow": [x, y], "wrist": [x, y], "fingertip": [x, y]}},
  "angles": {{"upper_arm": degrees, "forearm": degrees, "hand": degrees}},
  "objects": [
    {{
      "name": "object_name",
      "shape": "cylinder",
      "center": [x, y],
      "height_frac": 0.15,
      "width_frac": 0.05,
      "interaction": "grasping"
    }}
  ]
}}

Return ONLY the JSON. No explanation."""


def _image_to_world_position(
    center_xy: tuple[float, float],
    height_frac: float,
    width_frac: float,
    workspace_bounds: tuple[float, float, float] = (0.4, 0.4, 0.3),
    table_height: float = 0.05,
) -> tuple[tuple[float, float, float], tuple[float, ...]]:
    """Convert 2D image-normalized position to approximate 3D world coords and size.

    Returns:
        (position, size) — position is (x,y,z), size is shape-dependent tuple.
    """
    # X: map image horizontal to world X (centered on robot base)
    x = (center_xy[0] - 0.5) * workspace_bounds[0] * 2
    # Y: depth — images give limited depth info, use slight offset
    y = (center_xy[1] - 0.5) * workspace_bounds[1] * 0.3
    # Object height in world units
    obj_height = height_frac * workspace_bounds[2] * 3
    obj_radius = width_frac * workspace_bounds[0]
    # Z: objects sit on the table
    z = table_height + obj_height / 2

    position = (round(x, 4), round(y, 4), round(z, 4))
    # Cylinder size = (radius, half_height)
    size = (round(max(obj_radius, 0.01), 4), round(max(obj_height / 2, 0.01), 4))
    return position, size


def _deduplicate_objects(
    all_detections: list[dict],
) -> list[dict]:
    """Merge detections of the same object across frames.

    Groups by name, averages position/size, picks most common shape.
    """
    from collections import defaultdict
    groups: dict[str, list[dict]] = defaultdict(list)
    for det in all_detections:
        groups[det["name"]].append(det)

    merged = []
    for name, dets in groups.items():
        # Average center
        avg_cx = sum(d["center"][0] for d in dets) / len(dets)
        avg_cy = sum(d["center"][1] for d in dets) / len(dets)
        avg_h = sum(d["height_frac"] for d in dets) / len(dets)
        avg_w = sum(d["width_frac"] for d in dets) / len(dets)
        # Most common shape
        shapes = [d["shape"] for d in dets]
        shape = max(set(shapes), key=shapes.count)

        merged.append({
            "name": name,
            "shape": shape,
            "center": [avg_cx, avg_cy],
            "height_frac": avg_h,
            "width_frac": avg_w,
        })

    return merged


def _load_scene_cache(
    cache_dir: Path, cache_key: str
) -> tuple[list[SkeletonPose], dict] | None:
    """Load cached scene extraction if it exists."""
    cache_file = cache_dir / f"scene_{cache_key}.json"
    if not cache_file.exists():
        return None

    try:
        with open(cache_file, "r") as f:
            data = json.load(f)

        poses = []
        for entry in data.get("poses", []):
            joints = {k: tuple(v) for k, v in entry["joints"].items()}
            angles = {k: float(v) for k, v in entry["angles"].items()}
            poses.append(SkeletonPose(
                frame=entry["frame"],
                location=entry["location"],
                joints=joints,
                angles=angles,
            ))

        return poses, data
    except Exception:
        return None


def _save_scene_cache(
    cache_dir: Path, cache_key: str, poses: list[SkeletonPose],
    objects_raw: list[dict], interactions_raw: list[dict],
) -> None:
    """Save scene extraction to cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"scene_{cache_key}.json"

    data = {
        "poses": [{
            "frame": p.frame,
            "location": p.location,
            "joints": {k: list(v) for k, v in p.joints.items()},
            "angles": p.angles,
        } for p in poses],
        "objects": objects_raw,
        "interactions": interactions_raw,
    }

    with open(cache_file, "w") as f:
        json.dump(data, f, indent=2)


def extract_scene(
    images: list[MotionImage],
    task_description: str,
    expected_objects: list[dict] | None = None,
    workspace_bounds: tuple[float, float, float] = (0.4, 0.4, 0.3),
    cache_dir: str | Path | None = _DEFAULT_CACHE_DIR,
) -> tuple[list[SkeletonPose], "SceneExtraction"]:
    """Extract arm skeleton + scene objects + interactions from images.

    Uses the same Groq Vision API as extract_skeleton(), but with an enhanced
    prompt that also detects scene objects and subject-object interactions.
    One API call per image — extracts everything in a single pass.

    Args:
        images: List of MotionImage objects.
        task_description: Short text describing the task (e.g., "bartending").
        expected_objects: Optional hints about what objects to look for.
        workspace_bounds: (x, y, z) max workspace dimensions in meters.
        cache_dir: Cache directory. None disables caching.

    Returns:
        Tuple of (skeleton_poses, SceneExtraction).
    """
    from cadenza_local.task import SceneObject, Interaction, SceneExtraction

    # ── Cache check ──
    content_hash = _compute_directory_hash(images)
    task_hash = hashlib.sha256(task_description.encode()).hexdigest()[:8]
    cache_key = f"{content_hash}_{task_hash}"

    if cache_dir is not None:
        cached = _load_scene_cache(Path(cache_dir), cache_key)
        if cached is not None:
            poses, raw_data = cached
            print(f"  Loaded cached scene extraction (key={cache_key})")
            print(f"    {len(poses)} poses, {len(raw_data.get('objects', []))} objects")

            # Rebuild SceneExtraction from cache
            scene_objects = []
            for obj in raw_data.get("objects", []):
                pos, size = _image_to_world_position(
                    obj["center"], obj["height_frac"], obj.get("width_frac", 0.03),
                    workspace_bounds,
                )
                scene_objects.append(SceneObject(
                    name=obj["name"],
                    shape=obj.get("shape", "cylinder"),
                    size=size,
                    position=pos,
                ))

            interactions = [
                Interaction(
                    frame=ix["frame"],
                    action=ix["action"],
                    object_name=ix["object_name"],
                    confidence=ix.get("confidence", 0.5),
                )
                for ix in raw_data.get("interactions", [])
            ]

            scene = SceneExtraction(
                objects=scene_objects,
                interactions=interactions,
                task_summary=task_description,
            )
            return poses, scene

    # ── API extraction ──
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not found in environment or .env")

    client = Groq(api_key=api_key)
    prompt = _build_scene_prompt(task_description, expected_objects)
    poses: list[SkeletonPose] = []
    all_object_detections: list[dict] = []
    all_interactions: list[dict] = []

    print(f"  Scene extraction: {len(images)} images, task='{task_description}'")

    for img in images:
        print(f"  Extracting frame {img.location}_{img.number}...")

        # Images are always JPEG after _resize_and_encode.
        mime = "image/jpeg"

        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime};base64,{img.image_b64}"
                    },
                },
            ],
        }]

        try:
            response = client.chat.completions.create(
                model=VISION_MODEL,
                messages=messages,
                temperature=0.1,
                max_tokens=800,
                response_format={"type": "json_object"},
            )

            reply = response.choices[0].message.content.strip()
            json_match = re.search(r"\{[\s\S]*\}", reply)
            if not json_match:
                raise ValueError(f"No JSON in response for frame {img.number}")

            data = json.loads(json_match.group())

            # Parse skeleton (reuse existing logic)
            joints = {}
            for jname, coords in data.get("joints", {}).items():
                if isinstance(coords, list) and len(coords) == 2:
                    joints[jname] = (float(coords[0]), float(coords[1]))
            angles = {}
            for sname, angle in data.get("angles", {}).items():
                angles[sname] = float(angle)
            poses.append(SkeletonPose(
                frame=img.number, location=img.location,
                joints=joints, angles=angles,
            ))

            # Parse scene objects
            for obj in data.get("objects", []):
                name = obj.get("name", "")
                if not name:
                    continue
                center = obj.get("center", [0.5, 0.5])
                if isinstance(center, list) and len(center) == 2:
                    center = [float(center[0]), float(center[1])]
                else:
                    center = [0.5, 0.5]

                all_object_detections.append({
                    "name": name,
                    "shape": obj.get("shape", "cylinder"),
                    "center": center,
                    "height_frac": float(obj.get("height_frac", 0.1)),
                    "width_frac": float(obj.get("width_frac", 0.03)),
                })

                interaction = obj.get("interaction", "none")
                if interaction and interaction != "none":
                    all_interactions.append({
                        "frame": img.number,
                        "action": interaction,
                        "object_name": name,
                        "confidence": 0.7,
                    })

            n_objs = len(data.get("objects", []))
            print(f"    Skeleton: {list(angles.keys())}, Objects: {n_objs}")

        except Exception as e:
            print(f"    ERROR frame {img.number}: {e}")
            poses.append(SkeletonPose(
                frame=img.number, location=img.location,
                joints={}, angles={},
            ))

    poses.sort(key=lambda p: p.frame)

    # ── De-duplicate objects across frames ──
    merged_objects = _deduplicate_objects(all_object_detections)

    # ── Convert to 3D scene objects ──
    scene_objects = []
    for obj in merged_objects:
        pos, size = _image_to_world_position(
            obj["center"], obj["height_frac"], obj.get("width_frac", 0.03),
            workspace_bounds,
        )
        scene_objects.append(SceneObject(
            name=obj["name"],
            shape=obj.get("shape", "cylinder"),
            size=size,
            position=pos,
        ))

    interactions = [
        Interaction(
            frame=ix["frame"],
            action=ix["action"],
            object_name=ix["object_name"],
            confidence=ix.get("confidence", 0.5),
        )
        for ix in all_interactions
    ]

    scene = SceneExtraction(
        objects=scene_objects,
        interactions=interactions,
        task_summary=task_description,
    )

    # ── Save cache ──
    if cache_dir is not None:
        _save_scene_cache(
            Path(cache_dir), cache_key, poses,
            merged_objects, all_interactions,
        )
        print(f"  Cached scene extraction (key={cache_key})")

    print(f"\n  Scene extraction complete:")
    print(f"    {len(poses)} poses, {len(scene_objects)} objects, "
          f"{len(interactions)} interactions")
    for obj in scene_objects:
        print(f"    Object: {obj.name} ({obj.shape}) at "
              f"({obj.position[0]:.3f}, {obj.position[1]:.3f}, {obj.position[2]:.3f})")
    for ix in interactions[:5]:
        print(f"    Interaction: frame {ix.frame} — {ix.action} {ix.object_name}")

    return poses, scene
