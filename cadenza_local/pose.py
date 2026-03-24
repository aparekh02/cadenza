"""Pose detection — MediaPipe-based body and arm tracking.

Provides accurate joint positions and limb angles computed from real
pose estimation, replacing LLM-based skeleton guessing. Uses MediaPipe's
PoseLandmarker (heavy model) for 33-point body landmarks.

The pose model is downloaded automatically on first use (~30MB).

Usage:
    from cadenza_local.pose import PoseDetector

    # Single arm (auto-detect dominant)
    detector = PoseDetector()
    result = detector.detect_file("front_1.png")

    # Both arms (for dual-arm robots)
    result = detector.detect_file("front_1.png", both_arms=True)
    # result.joints  → {"left_shoulder": ..., "right_shoulder": ..., "base": ...}
    # result.angles  → {"left_upper_arm": ..., "right_upper_arm": ..., ...}

    # Batch detection
    results = detector.detect_sequence("path/to/images/", both_arms=True)
"""

from __future__ import annotations

import math
import os
import re
import subprocess
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision as mp_vision


# ── Model download ──

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_heavy/float16/latest/"
    "pose_landmarker_heavy.task"
)
_DEFAULT_MODEL_DIR = Path(".cache/cadenza_models")


def _ensure_model(model_dir: Path = _DEFAULT_MODEL_DIR) -> str:
    """Download the pose model if not already cached. Returns model path."""
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "pose_landmarker_heavy.task"

    if model_path.exists():
        return str(model_path)

    print(f"Downloading pose model to {model_path}...")
    try:
        urllib.request.urlretrieve(_MODEL_URL, str(model_path))
    except Exception:
        # Fallback: use curl (handles macOS SSL cert issues)
        try:
            subprocess.run(
                ["curl", "-sL", "-o", str(model_path), _MODEL_URL],
                check=True, timeout=120,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise RuntimeError(
                f"Failed to download pose model. Download manually from "
                f"{_MODEL_URL} to {model_path}"
            ) from e
    print("  Pose model downloaded.")
    return str(model_path)


# ── MediaPipe landmark indices for arm tracking ──

# Left arm landmarks
L_SHOULDER = 11
L_ELBOW = 13
L_WRIST = 15
L_PINKY = 17
L_INDEX = 19
L_THUMB = 21

# Right arm landmarks
R_SHOULDER = 12
R_ELBOW = 14
R_WRIST = 16
R_PINKY = 18
R_INDEX = 20
R_THUMB = 22

# Torso
NOSE = 0
L_HIP = 23
R_HIP = 24

# Named joint mapping for cadenza's skeleton format
# Each entry: (landmark_index_left, landmark_index_right)
ARM_JOINTS = {
    "shoulder": (L_SHOULDER, R_SHOULDER),
    "elbow":    (L_ELBOW, R_ELBOW),
    "wrist":    (L_WRIST, R_WRIST),
    "index":    (L_INDEX, R_INDEX),
    "pinky":    (L_PINKY, R_PINKY),
    "thumb":    (L_THUMB, R_THUMB),
}


@dataclass
class PoseResult:
    """Detected pose from a single image."""
    joints: dict[str, tuple[float, float]]        # name → (x, y) normalized
    joints_3d: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    angles: dict[str, float] = field(default_factory=dict)
    visibility: dict[str, float] = field(default_factory=dict)
    dominant_arm: str = "right"                     # which arm is performing the task
    gripper_openness: float = 45.0                  # degrees: 0=closed, 90=open
    raw_landmarks: list = field(default_factory=list)

    # Source metadata
    frame_number: int = 0
    location: str = ""
    image_path: str = ""


def _angle_from_horizontal(x1: float, y1: float, x2: float, y2: float) -> float:
    """Compute angle of segment (x1,y1)→(x2,y2) from horizontal.

    Returns degrees. 0° = pointing right, 90° = pointing up, -90° = pointing down.
    Note: image y increases downward, so we negate dy.
    """
    dx = x2 - x1
    dy = -(y2 - y1)  # flip y since image coords have y increasing down
    return math.degrees(math.atan2(dy, dx))


def _distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Euclidean distance between two points."""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def _compute_angles_for_arm(
    joints: dict[str, tuple[float, float]],
    prefix: str = "",
) -> dict[str, float]:
    """Compute arm segment angles from detected joint positions.

    Args:
        joints: Joint name → (x, y) dict. Names should already include prefix.
        prefix: Key prefix (e.g., "left_" or "right_" or "" for single arm).
    """
    angles = {}

    shoulder = joints.get(f"{prefix}shoulder")
    elbow = joints.get(f"{prefix}elbow")
    wrist = joints.get(f"{prefix}wrist")
    fingertip = joints.get(f"{prefix}fingertip")

    if shoulder and elbow:
        angles[f"{prefix}upper_arm"] = round(
            _angle_from_horizontal(shoulder[0], shoulder[1], elbow[0], elbow[1]), 1
        )

    if elbow and wrist:
        angles[f"{prefix}forearm"] = round(
            _angle_from_horizontal(elbow[0], elbow[1], wrist[0], wrist[1]), 1
        )

    if wrist and fingertip:
        angles[f"{prefix}hand"] = round(
            _angle_from_horizontal(wrist[0], wrist[1], fingertip[0], fingertip[1]), 1
        )

    # Base rotation: horizontal offset of shoulder from image center
    if shoulder:
        angles[f"{prefix}base_rotation"] = round((shoulder[0] - 0.5) * 90, 1)

    return angles


def _compute_angles(joints: dict[str, tuple[float, float]]) -> dict[str, float]:
    """Compute arm segment angles (single-arm backward compat)."""
    return _compute_angles_for_arm(joints, prefix="")


def _compute_gripper(
    index: tuple[float, float],
    thumb: tuple[float, float],
    pinky: tuple[float, float],
) -> float:
    """Estimate gripper openness from finger positions.

    Returns degrees: 0 = fully closed, 90 = fully open.
    """
    # Distance between thumb and index gives grip aperture
    thumb_index = _distance(thumb[0], thumb[1], index[0], index[1])
    # Normalize against hand span (thumb to pinky)
    span = _distance(thumb[0], thumb[1], pinky[0], pinky[1])

    if span < 0.001:
        return 45.0  # unknown

    ratio = thumb_index / span
    # Map ratio to degrees: ratio ~0.2 = closed, ~0.8 = open
    openness = max(0, min(90, (ratio - 0.2) / 0.6 * 90))
    return round(openness, 1)


def _compute_wrist_roll(
    wrist: tuple[float, float],
    index: tuple[float, float],
    pinky: tuple[float, float],
) -> float:
    """Estimate wrist roll from finger spread orientation.

    Returns degrees: 0 = neutral, positive = clockwise when viewed from arm.
    """
    # Angle of the index-pinky line from horizontal
    angle = _angle_from_horizontal(index[0], index[1], pinky[0], pinky[1])
    return round(angle, 1)


class PoseDetector:
    """MediaPipe-based pose detection for arm tracking.

    Uses PoseLandmarker (heavy model) for accurate 33-point body landmarks.
    Automatically determines which arm is the dominant/working arm based on
    visibility and movement.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        num_poses: int = 1,
        min_detection_confidence: float = 0.5,
    ):
        if model_path is None:
            model_path = _ensure_model()

        options = mp_vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            num_poses=num_poses,
            min_pose_detection_confidence=min_detection_confidence,
        )
        self._detector = mp_vision.PoseLandmarker.create_from_options(options)

    def detect_file(
        self,
        image_path: str,
        dominant_arm: Optional[str] = None,
        both_arms: bool = False,
        frame_number: int = 0,
        location: str = "",
    ) -> PoseResult:
        """Detect pose in a single image file.

        Args:
            image_path: Path to image file.
            dominant_arm: "left" or "right". If None, auto-detected. Ignored if both_arms=True.
            both_arms: Track both arms with prefixed joint names (left_*, right_*).
            frame_number: Frame number for metadata.
            location: Camera location for metadata.

        Returns:
            PoseResult with joints, angles, and gripper state.
        """
        img = mp.Image.create_from_file(str(image_path))
        detection = self._detector.detect(img)

        if not detection.pose_landmarks:
            return PoseResult(
                joints={}, frame_number=frame_number,
                location=location, image_path=str(image_path),
            )

        landmarks = detection.pose_landmarks[0]
        world_landmarks = (
            detection.pose_world_landmarks[0]
            if detection.pose_world_landmarks
            else None
        )

        if both_arms:
            return self._build_both_arms_result(
                landmarks, world_landmarks, frame_number, location, str(image_path),
            )

        # Single arm mode
        arm = dominant_arm or self._detect_dominant_arm(landmarks)
        joints, visibility = self._extract_arm_joints(landmarks, arm)

        joints_3d = {}
        if world_landmarks:
            joints_3d = self._extract_arm_joints_3d(world_landmarks, arm)

        # Build fingertip
        if "index" in joints and "thumb" in joints:
            ix, iy = joints["index"]
            tx, ty = joints["thumb"]
            joints["fingertip"] = ((ix + tx) / 2, (iy + ty) / 2)

        angles = _compute_angles(joints)

        gripper = 45.0
        if all(k in joints for k in ["index", "thumb", "pinky"]):
            gripper = _compute_gripper(joints["index"], joints["thumb"], joints["pinky"])
        angles["gripper"] = gripper

        if all(k in joints for k in ["wrist", "index", "pinky"]):
            angles["wrist_roll"] = _compute_wrist_roll(
                joints["wrist"], joints["index"], joints["pinky"],
            )

        return PoseResult(
            joints=joints, joints_3d=joints_3d, angles=angles,
            visibility=visibility, dominant_arm=arm, gripper_openness=gripper,
            raw_landmarks=list(landmarks),
            frame_number=frame_number, location=location, image_path=str(image_path),
        )

    def _build_both_arms_result(
        self, landmarks, world_landmarks, frame_number, location, image_path,
    ) -> PoseResult:
        """Build PoseResult with both arms tracked (left_* and right_* prefixed joints)."""
        joints = {}
        visibility = {}
        joints_3d = {}
        angles = {}

        for prefix, side_idx in [("left_", 0), ("right_", 1)]:
            # Extract joints for this arm
            for name, (l_idx, r_idx) in ARM_JOINTS.items():
                idx = l_idx if side_idx == 0 else r_idx
                lm = landmarks[idx]
                joints[f"{prefix}{name}"] = (round(lm.x, 4), round(lm.y, 4))
                visibility[f"{prefix}{name}"] = round(lm.visibility, 3)

            # 3D joints
            if world_landmarks:
                for name, (l_idx, r_idx) in ARM_JOINTS.items():
                    idx = l_idx if side_idx == 0 else r_idx
                    wlm = world_landmarks[idx]
                    joints_3d[f"{prefix}{name}"] = (
                        round(wlm.x, 4), round(wlm.y, 4), round(wlm.z, 4),
                    )

            # Build fingertip
            idx_key = f"{prefix}index"
            thb_key = f"{prefix}thumb"
            pnk_key = f"{prefix}pinky"
            wst_key = f"{prefix}wrist"
            if idx_key in joints and thb_key in joints:
                ix, iy = joints[idx_key]
                tx, ty = joints[thb_key]
                joints[f"{prefix}fingertip"] = ((ix + tx) / 2, (iy + ty) / 2)

            # Compute angles for this arm
            arm_angles = _compute_angles_for_arm(joints, prefix=prefix)
            angles.update(arm_angles)

            # Gripper
            gripper = 45.0
            if all(k in joints for k in [idx_key, thb_key, pnk_key]):
                gripper = _compute_gripper(joints[idx_key], joints[thb_key], joints[pnk_key])
            angles[f"{prefix}gripper"] = gripper

            # Wrist roll
            if all(k in joints for k in [wst_key, idx_key, pnk_key]):
                angles[f"{prefix}wrist_roll"] = _compute_wrist_roll(
                    joints[wst_key], joints[idx_key], joints[pnk_key],
                )

        # Shared base (torso midpoint)
        ls = landmarks[L_SHOULDER]
        rs = landmarks[R_SHOULDER]
        joints["base"] = (
            round((ls.x + rs.x) / 2, 4),
            round((ls.y + rs.y) / 2, 4),
        )
        visibility["base"] = round(min(ls.visibility, rs.visibility), 3)

        return PoseResult(
            joints=joints, joints_3d=joints_3d, angles=angles,
            visibility=visibility, dominant_arm="both", gripper_openness=45.0,
            raw_landmarks=list(landmarks),
            frame_number=frame_number, location=location, image_path=image_path,
        )

    def detect_sequence(
        self,
        image_dir: str,
        dominant_arm: Optional[str] = None,
        both_arms: bool = False,
    ) -> list[PoseResult]:
        """Detect poses in a sequence of motion images.

        Expects filenames: {location}_{number}.png

        Args:
            image_dir: Directory with motion images.
            dominant_arm: Force single-arm selection. Ignored if both_arms=True.
            both_arms: Track both arms with prefixed joint names.

        Returns:
            List of PoseResult, sorted by frame number.
        """
        dir_path = Path(image_dir).resolve()
        if not dir_path.is_dir():
            raise FileNotFoundError(f"Image directory not found: {dir_path}")

        pattern = re.compile(r"^(.+)_(\d+)\.(png|jpg|jpeg)$", re.IGNORECASE)
        image_files = []

        for entry in sorted(dir_path.iterdir()):
            if not entry.is_file():
                continue
            match = pattern.match(entry.name)
            if match:
                location = match.group(1).lower()
                number = int(match.group(2))
                image_files.append((number, location, str(entry)))

        image_files.sort(key=lambda x: x[0])

        if not image_files:
            raise ValueError(f"No motion images found in {dir_path}")

        arm = dominant_arm
        if both_arms:
            arm = None
            mode_str = "both arms"
        else:
            if arm is None:
                arm = self._detect_dominant_arm_sequence(image_files)
                print(f"  Auto-detected dominant arm: {arm}")
            mode_str = f"arm={arm}"

        results = []
        print(f"Detecting poses in {len(image_files)} images ({mode_str})...")
        for number, location, path in image_files:
            result = self.detect_file(
                path,
                dominant_arm=arm,
                both_arms=both_arms,
                frame_number=number,
                location=location,
            )
            results.append(result)
            n_joints = len(result.joints)
            print(f"  Frame {number} ({location}): {n_joints} joints")

        return results

    def _detect_dominant_arm_sequence(
        self, image_files: list[tuple[int, str, str]]
    ) -> str:
        """Scan all frames to determine which arm is the primary working arm.

        Uses two signals:
        1. Consistent visibility — the working arm stays visible throughout
        2. Movement in visible frames — real motion, not off-screen jumps
        """
        MIN_VIS = 0.6

        left_frames = []   # (x, y, visibility) per frame
        right_frames = []

        for _, _, path in image_files:
            img = mp.Image.create_from_file(path)
            detection = self._detector.detect(img)
            if not detection.pose_landmarks:
                continue
            lm = detection.pose_landmarks[0]
            left_frames.append((lm[L_WRIST].x, lm[L_WRIST].y, lm[L_WRIST].visibility))
            right_frames.append((lm[R_WRIST].x, lm[R_WRIST].y, lm[R_WRIST].visibility))

        def score_arm(frames):
            n = max(len(frames), 1)
            # Collect only in-frame, visible positions
            good = [
                (x, y) for x, y, v in frames
                if v >= MIN_VIS and 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0
            ]
            if not good:
                return 0.0

            in_frame_ratio = len(good) / n

            # Working arm has LOW variance (stays in workspace)
            xs = [p[0] for p in good]
            ys = [p[1] for p in good]
            x_var = sum((x - sum(xs) / len(xs)) ** 2 for x in xs) / len(xs)
            y_var = sum((y - sum(ys) / len(ys)) ** 2 for y in ys) / len(ys)
            position_stability = 1.0 / (1.0 + (x_var + y_var) * 20)

            # Combined: visibility + stability
            return in_frame_ratio * 0.4 + position_stability * 0.6

        l_score = score_arm(left_frames)
        r_score = score_arm(right_frames)

        return "right" if r_score >= l_score else "left"

    def _detect_dominant_arm(self, landmarks) -> str:
        """Determine which arm is the working arm based on visibility and extension."""
        l_vis = (
            landmarks[L_WRIST].visibility
            + landmarks[L_ELBOW].visibility
            + landmarks[L_INDEX].visibility
        ) / 3
        r_vis = (
            landmarks[R_WRIST].visibility
            + landmarks[R_ELBOW].visibility
            + landmarks[R_INDEX].visibility
        ) / 3

        # Also check extension (distance from shoulder to wrist)
        l_ext = _distance(
            landmarks[L_SHOULDER].x, landmarks[L_SHOULDER].y,
            landmarks[L_WRIST].x, landmarks[L_WRIST].y,
        )
        r_ext = _distance(
            landmarks[R_SHOULDER].x, landmarks[R_SHOULDER].y,
            landmarks[R_WRIST].x, landmarks[R_WRIST].y,
        )

        # Weighted score: visibility + extension
        l_score = l_vis * 0.6 + l_ext * 0.4
        r_score = r_vis * 0.6 + r_ext * 0.4

        return "right" if r_score >= l_score else "left"

    def _extract_arm_joints(
        self, landmarks, arm: str
    ) -> tuple[dict[str, tuple[float, float]], dict[str, float]]:
        """Extract arm joint positions and visibility for the specified arm."""
        side = 0 if arm == "left" else 1
        joints = {}
        visibility = {}

        for name, (l_idx, r_idx) in ARM_JOINTS.items():
            idx = r_idx if side == 1 else l_idx
            lm = landmarks[idx]
            joints[name] = (round(lm.x, 4), round(lm.y, 4))
            visibility[name] = round(lm.visibility, 3)

        # Add base as torso midpoint between shoulders
        ls = landmarks[L_SHOULDER]
        rs = landmarks[R_SHOULDER]
        joints["base"] = (
            round((ls.x + rs.x) / 2, 4),
            round((ls.y + rs.y) / 2, 4),
        )
        visibility["base"] = round(min(ls.visibility, rs.visibility), 3)

        return joints, visibility

    def _extract_arm_joints_3d(
        self, world_landmarks, arm: str
    ) -> dict[str, tuple[float, float, float]]:
        """Extract 3D joint positions from world landmarks."""
        side = 0 if arm == "left" else 1
        joints_3d = {}

        for name, (l_idx, r_idx) in ARM_JOINTS.items():
            idx = r_idx if side == 1 else l_idx
            lm = world_landmarks[idx]
            joints_3d[name] = (round(lm.x, 4), round(lm.y, 4), round(lm.z, 4))

        return joints_3d

    def close(self):
        """Release the detector resources."""
        self._detector.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
