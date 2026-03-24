"""Visual force-space test — depth, segmentation, force flow, and 3D simulation.

Runs the full cadenza force-space pipeline on the bartender motion images,
adds depth estimation (MiDaS v2.1 Large) and composite image
segmentation, then renders:
    1. Per-frame 2x2 analysis panels (skeleton, depth, segmentation, forces)
    2. Smooth force-flow animated GIF interpolating between keyframes
    3. Side-by-side comparison of originals vs force-flow keyframes
    4. Text summary with dynamics, sandbox verdict, and statistics
   10. 3D scene reconstruction from depth + pose data
   11. MuJoCo simulation of SO-101 robot following extracted movements
   12. 3D visualization: simulation GIFs + trajectory plot

Usage:
    python tests/test_forcespace.py
    python tests/test_forcespace.py --skip-depth     # skip MiDaS (flat depth)
    python tests/test_forcespace.py --skip-3d        # skip 3D reconstruction + sim
    python tests/test_forcespace.py --no-video        # skip GIF generation
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Check matplotlib early ──

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import ListedColormap
except ImportError:
    print("ERROR: matplotlib is required for force-space visualization.")
    print("Install with: pip install matplotlib>=3.5.0")
    sys.exit(1)

# ── Cadenza imports ──

from cadenza_local.presets.builder import parse_task_file
from cadenza_local.presets import (
    analyze_motion_images,
    compute_dynamics,
    compute_motor_profile,
    run_force_sandbox,
    ObjectProfile,
)
from cadenza_local.presets.schemas import (
    FrameAnalysis,
    MotionBlueprint,
    DynamicsProfile,
    MotorProfile,
    SandboxResult,
    SegmentKinematics,
    JointTorqueEstimate,
)
from cadenza_local.depth import DepthEstimator
from cadenza_local.pose import PoseResult

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ── Paths ──

MOTION_DIR = str(ROOT / "examples" / "bartender" / "motion_images")
TASK_FILE = str(ROOT / "examples" / "bartender" / "task.txt")
OUTPUT_DIR = ROOT / "tests" / "output_forcespace"

# ── Visualization constants ──

COLOR_GRAVITY = (100, 200, 255)     # light blue — gravity opposing arrows
COLOR_PAYLOAD = (255, 160, 50)      # orange — payload arrows
COLOR_TORQUE_HIGH = (255, 80, 80)   # red — high-load joints
COLOR_COM = (255, 255, 0)           # yellow — center of mass
COLOR_BALANCE_OK = (80, 255, 120)   # green — stable
COLOR_BALANCE_BAD = (255, 60, 60)   # red — unstable

# Segmentation label colors (RGB).
SEG_COLORS = {
    0: (40, 40, 40),          # background
    1: (139, 119, 101),       # table
    2: (100, 180, 255),       # left arm
    3: (255, 100, 80),        # right arm
}

# Force flow video settings.
INTERP_STEPS = 8        # sub-frames between each keyframe pair
GIF_FRAME_MS = 80       # milliseconds per frame (~12.5 fps)
GIF_SIZE = (800, 600)   # w, h for each GIF frame

# ── Depth thresholds (normalized 0-1, 0=near) ──
DEPTH_FG = 0.45          # below = foreground
DEPTH_TABLE = 0.65       # below = table/midground, above = background


# ═══════════════════════════════════════════════════════════════════
# Image Segmentation
# ═══════════════════════════════════════════════════════════════════

class FrameSegmenter:
    """Composite lightweight segmentation: depth + pose + object ROIs."""

    BG = 0
    TABLE = 1
    LEFT_ARM = 2
    RIGHT_ARM = 3
    OBJECT_BASE = 4

    def __init__(self, object_names: list[str]):
        self.object_names = object_names
        self.stats: dict = {"pixel_counts": []}

    def segment(
        self,
        image: np.ndarray,
        depth_map: np.ndarray,
        pose: PoseResult,
    ) -> np.ndarray:
        """Produce (H, W) int label mask."""
        h, w = depth_map.shape
        mask = np.zeros((h, w), dtype=np.int32)

        # Layer 1: depth-based regions.
        mask[depth_map < DEPTH_TABLE] = self.TABLE     # midground
        mask[depth_map < DEPTH_FG] = self.BG            # foreground (will be overwritten)

        # Background stays 0 where depth >= DEPTH_TABLE.
        mask[depth_map >= DEPTH_TABLE] = self.BG

        # Layer 2: pose-based arm masks.
        self._draw_arm_mask(mask, pose, w, h)

        # Layer 3: object isolation via foreground depth + bounding regions.
        self._draw_object_masks(mask, depth_map, pose, w, h)

        # Collect stats.
        counts = {}
        for label in range(self.OBJECT_BASE + len(self.object_names)):
            n = int((mask == label).sum())
            if n > 0:
                counts[label] = n
        self.stats["pixel_counts"].append(counts)

        return mask

    def _draw_arm_mask(
        self,
        mask: np.ndarray,
        pose: PoseResult,
        w: int, h: int,
    ):
        """Draw arm segment capsules using PIL, then paste onto mask."""
        arm_img = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(arm_img)

        # Joint chain for each arm.
        left_chain = ["left_shoulder", "left_elbow", "left_wrist", "left_fingertip"]
        right_chain = ["right_shoulder", "right_elbow", "right_wrist", "right_fingertip"]
        single_chain = ["shoulder", "elbow", "wrist", "fingertip"]

        for chain, label in [
            (left_chain, self.LEFT_ARM),
            (right_chain, self.RIGHT_ARM),
            (single_chain, self.RIGHT_ARM),
        ]:
            pts = []
            for jname in chain:
                if jname in pose.joints:
                    jx, jy = pose.joints[jname]
                    pts.append((int(jx * w), int(jy * h)))
            if len(pts) < 2:
                continue
            # Draw thick polyline (wider at shoulder, thinner at fingertip).
            for i in range(len(pts) - 1):
                width = max(3, int(25 * (1.0 - i / len(pts))))
                draw.line([pts[i], pts[i + 1]], fill=label, width=width)
            # Circles at joints.
            for i, pt in enumerate(pts):
                r = max(3, int(15 * (1.0 - i / len(pts))))
                draw.ellipse([pt[0] - r, pt[1] - r, pt[0] + r, pt[1] + r], fill=label)

        arm_arr = np.array(arm_img)
        mask[arm_arr == self.LEFT_ARM] = self.LEFT_ARM
        mask[arm_arr == self.RIGHT_ARM] = self.RIGHT_ARM

    def _draw_object_masks(
        self,
        mask: np.ndarray,
        depth_map: np.ndarray,
        pose: PoseResult,
        w: int, h: int,
    ):
        """Mark object regions using detected object positions from pose."""
        if not hasattr(pose, "objects") or not pose.objects:
            return

        for idx, (obj_name, (ox, oy)) in enumerate(pose.objects.items()):
            label = self.OBJECT_BASE + idx
            # Create a rectangular ROI around the object position.
            cx, cy = int(ox * w), int(oy * h)
            roi_w, roi_h = max(20, w // 15), max(40, h // 8)
            x1 = max(0, cx - roi_w // 2)
            x2 = min(w, cx + roi_w // 2)
            y1 = max(0, cy - roi_h // 2)
            y2 = min(h, cy + roi_h // 2)

            # Within ROI, mark foreground pixels as this object.
            roi_depth = depth_map[y1:y2, x1:x2]
            fg = roi_depth < DEPTH_FG
            roi_mask = mask[y1:y2, x1:x2]
            # Only label pixels not already claimed by arms.
            unclaimed = (roi_mask != self.LEFT_ARM) & (roi_mask != self.RIGHT_ARM)
            roi_mask[fg & unclaimed] = label


# ═══════════════════════════════════════════════════════════════════
# Force Flow Interpolation
# ═══════════════════════════════════════════════════════════════════

@dataclass
class InterpolatedFrame:
    """A single interpolated sub-frame between two keyframes."""
    t: float                            # 0.0 to 1.0 between source frames
    source_a: int                       # preceding keyframe index
    source_b: int                       # following keyframe index
    joint_angles: dict                  # segment → interpolated angle (deg)
    joint_torques: dict                 # segment → interpolated torque (Nm)
    com_position: tuple                 # (x, y) normalized
    balance_score: float
    phase_label: str
    keypoint_positions: dict            # joint_name → (x, y) normalized


class ForceFlowInterpolator:
    """Linear interpolation of forces between keyframes."""

    def __init__(
        self,
        dynamics: DynamicsProfile,
        frames: list[FrameAnalysis],
        poses: list[PoseResult],
        n_interp: int = INTERP_STEPS,
    ):
        self.dynamics = dynamics
        self.frames = frames
        self.poses = poses
        self.n_interp = n_interp

    def interpolate(self) -> list[InterpolatedFrame]:
        n = len(self.frames)
        if n < 2:
            return []

        dyn = self.dynamics
        kin_map = {sk.segment_name: sk for sk in dyn.kinematics}
        torque_map = {jt.joint_name: jt for jt in dyn.torques}

        results: list[InterpolatedFrame] = []

        for fi in range(n - 1):
            for si in range(self.n_interp + 1):
                t = si / (self.n_interp + 1)
                is_keyframe = (si == 0)

                # Interpolate angles.
                angles = {}
                for seg_name, sk in kin_map.items():
                    a = sk.angles_deg[fi] if fi < len(sk.angles_deg) else 0.0
                    b = sk.angles_deg[fi + 1] if fi + 1 < len(sk.angles_deg) else a
                    angles[seg_name] = a + (b - a) * t

                # Interpolate torques.
                torques = {}
                for seg_name, jt in torque_map.items():
                    a = jt.torques_nm[fi] if fi < len(jt.torques_nm) else 0.0
                    b = jt.torques_nm[fi + 1] if fi + 1 < len(jt.torques_nm) else a
                    torques[seg_name] = a + (b - a) * t

                # Interpolate COM.
                com_a = dyn.com.positions_xy[fi] if dyn.com and fi < len(dyn.com.positions_xy) else (0.5, 0.5)
                com_b = dyn.com.positions_xy[fi + 1] if dyn.com and fi + 1 < len(dyn.com.positions_xy) else com_a
                com = (com_a[0] + (com_b[0] - com_a[0]) * t,
                       com_a[1] + (com_b[1] - com_a[1]) * t)

                # Interpolate balance.
                bal_a = dyn.balance.scores[fi] if dyn.balance and fi < len(dyn.balance.scores) else 0.0
                bal_b = dyn.balance.scores[fi + 1] if dyn.balance and fi + 1 < len(dyn.balance.scores) else bal_a
                bal = bal_a + (bal_b - bal_a) * t

                # Phase: use source frame label until midpoint.
                phase = self.frames[fi].phase_label if t < 0.5 else self.frames[fi + 1].phase_label

                # Interpolate keypoint positions.
                kps = {}
                pose_a = self.poses[fi] if fi < len(self.poses) else None
                pose_b = self.poses[fi + 1] if fi + 1 < len(self.poses) else None
                if pose_a:
                    for jname, (jx, jy) in pose_a.joints.items():
                        if pose_b and jname in pose_b.joints:
                            bx, by = pose_b.joints[jname]
                            kps[jname] = (jx + (bx - jx) * t, jy + (by - jy) * t)
                        else:
                            kps[jname] = (jx, jy)

                results.append(InterpolatedFrame(
                    t=t,
                    source_a=fi,
                    source_b=fi + 1,
                    joint_angles=angles,
                    joint_torques=torques,
                    com_position=com,
                    balance_score=bal,
                    phase_label=phase or "",
                    keypoint_positions=kps,
                ))

        # Add the last keyframe.
        last = len(self.frames) - 1
        last_angles = {s: sk.angles_deg[last] if last < len(sk.angles_deg) else 0.0
                       for s, sk in kin_map.items()}
        last_torques = {s: jt.torques_nm[last] if last < len(jt.torques_nm) else 0.0
                        for s, jt in torque_map.items()}
        last_com = dyn.com.positions_xy[last] if dyn.com and last < len(dyn.com.positions_xy) else (0.5, 0.5)
        last_bal = dyn.balance.scores[last] if dyn.balance and last < len(dyn.balance.scores) else 0.0
        last_kps = {}
        if last < len(self.poses):
            last_kps = dict(self.poses[last].joints)
        results.append(InterpolatedFrame(
            t=1.0,
            source_a=last,
            source_b=last,
            joint_angles=last_angles,
            joint_torques=last_torques,
            com_position=last_com,
            balance_score=last_bal,
            phase_label=self.frames[last].phase_label or "",
            keypoint_positions=last_kps,
        ))

        return results


# ═══════════════════════════════════════════════════════════════════
# Drawing Helpers
# ═══════════════════════════════════════════════════════════════════

def _draw_force_arrow(
    draw: ImageDraw.Draw,
    x: int, y: int,
    magnitude: float,
    direction: str,
    color: tuple,
    scale: float = 120.0,
):
    """Draw a force vector arrow on a PIL ImageDraw."""
    length = int(magnitude * scale)
    length = max(5, min(length, 150))

    if direction == "up":
        end_y = y - length
    else:
        end_y = y + length

    # Shaft.
    draw.line([(x, y), (x, end_y)], fill=color + (200,), width=3)

    # Arrowhead.
    hs = max(4, length // 5)
    if direction == "up":
        draw.polygon([
            (x, end_y), (x - hs, end_y + hs), (x + hs, end_y + hs),
        ], fill=color + (220,))
    else:
        draw.polygon([
            (x, end_y), (x - hs, end_y - hs), (x + hs, end_y - hs),
        ], fill=color + (220,))


def _colorize_depth(depth_map: np.ndarray) -> np.ndarray:
    """Depth (H,W) float [0,1] → (H,W,3) uint8 via plasma colormap."""
    cmap = plt.colormaps["plasma"]
    colored = (cmap(depth_map)[:, :, :3] * 255).astype(np.uint8)
    return colored


def _object_color(idx: int) -> tuple:
    """Generate distinct color for object label via golden-ratio hue spacing."""
    hue = (idx * 0.618033988749895) % 1.0
    # HSV → RGB (s=0.7, v=0.9).
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
    return (int(r * 255), int(g * 255), int(b * 255))


def _colorize_segmentation(seg_mask: np.ndarray, object_names: list[str]) -> np.ndarray:
    """Label mask (H,W) → (H,W,3) uint8 with distinct colors."""
    colors = dict(SEG_COLORS)
    for i, name in enumerate(object_names):
        colors[FrameSegmenter.OBJECT_BASE + i] = _object_color(i)

    h, w = seg_mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for label, color in colors.items():
        rgb[seg_mask == label] = color
    return rgb


def _get_joint_pixel(pose: PoseResult, jname: str, w: int, h: int) -> Optional[tuple]:
    """Get pixel position (x, y) for a joint name, or None."""
    if jname in pose.joints:
        jx, jy = pose.joints[jname]
        return (int(jx * w), int(jy * h))
    return None


def _skeleton_overlay(image: np.ndarray, pose: PoseResult) -> np.ndarray:
    """Draw skeleton lines and joints on image, return modified copy."""
    h, w = image.shape[:2]
    pil = Image.fromarray(image).convert("RGBA")
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Arm chains.
    chains = [
        (["left_shoulder", "left_elbow", "left_wrist", "left_fingertip"], (100, 180, 255)),
        (["right_shoulder", "right_elbow", "right_wrist", "right_fingertip"], (255, 100, 80)),
        (["shoulder", "elbow", "wrist", "fingertip"], (255, 100, 80)),
    ]

    for chain_names, color in chains:
        pts = []
        for jn in chain_names:
            p = _get_joint_pixel(pose, jn, w, h)
            if p:
                pts.append(p)
        if len(pts) >= 2:
            for i in range(len(pts) - 1):
                draw.line([pts[i], pts[i + 1]], fill=color + (180,), width=4)
            for pt in pts:
                draw.ellipse([pt[0] - 5, pt[1] - 5, pt[0] + 5, pt[1] + 5],
                             fill=(0, 255, 100, 200))

    # Angle labels.
    for seg_name, angle in pose.angles.items():
        # Place near the middle joint of the segment.
        mapping = {
            "upper_arm": "elbow", "left_upper_arm": "left_elbow",
            "right_upper_arm": "right_elbow",
            "forearm": "wrist", "left_forearm": "left_wrist",
            "right_forearm": "right_wrist",
            "hand": "fingertip", "left_hand": "left_fingertip",
            "right_hand": "right_fingertip",
        }
        jn = mapping.get(seg_name)
        if jn:
            pt = _get_joint_pixel(pose, jn, w, h)
            if pt:
                draw.text((pt[0] + 8, pt[1] - 10), f"{angle:.0f}\u00b0",
                          fill=(255, 255, 0, 220))

    composite = Image.alpha_composite(pil, overlay)
    return np.array(composite.convert("RGB"))


# ═══════════════════════════════════════════════════════════════════
# Per-Frame Analysis Panel
# ═══════════════════════════════════════════════════════════════════

def render_frame_analysis(
    frame_idx: int,
    image_path: str,
    depth_map: np.ndarray,
    seg_mask: np.ndarray,
    pose: PoseResult,
    frame: FrameAnalysis,
    dynamics: DynamicsProfile,
    object_names: list[str],
    output_path: str,
):
    """Render 2x2 analysis panel for one frame → saved PNG."""
    orig = np.array(Image.open(image_path).convert("RGB"))
    h, w = orig.shape[:2]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f"Frame {frame_idx + 1}  —  Phase: {frame.phase_label or 'unknown'}",
        fontsize=16, fontweight="bold",
    )

    # ── Top-left: Original + Skeleton ──
    skel_img = _skeleton_overlay(orig, pose)
    axes[0, 0].imshow(skel_img)
    axes[0, 0].set_title("Original + Skeleton", fontsize=12)
    # Add interaction text.
    if frame.interactions:
        axes[0, 0].text(
            0.02, 0.02, "\n".join(frame.interactions[:3]),
            transform=axes[0, 0].transAxes, fontsize=8,
            color="white", va="bottom",
            bbox=dict(boxstyle="round", fc="black", alpha=0.6),
        )
    axes[0, 0].axis("off")

    # ── Top-right: Depth Map ──
    depth_rgb = _colorize_depth(depth_map)
    axes[0, 1].imshow(depth_rgb)
    axes[0, 1].set_title("Depth Map (MiDaS)", fontsize=12)
    # Overlay joints.
    for jname, (jx, jy) in pose.joints.items():
        axes[0, 1].plot(jx * w, jy * h, "wo", markersize=4, markeredgecolor="black")
    axes[0, 1].axis("off")

    # ── Bottom-left: Segmentation ──
    seg_rgb = _colorize_segmentation(seg_mask, object_names)
    axes[1, 0].imshow(seg_rgb)
    axes[1, 0].set_title("Segmentation", fontsize=12)
    # Legend.
    legend_entries = [
        mpatches.Patch(color=np.array(SEG_COLORS[0]) / 255, label="Background"),
        mpatches.Patch(color=np.array(SEG_COLORS[1]) / 255, label="Table"),
        mpatches.Patch(color=np.array(SEG_COLORS[2]) / 255, label="Left Arm"),
        mpatches.Patch(color=np.array(SEG_COLORS[3]) / 255, label="Right Arm"),
    ]
    for i, name in enumerate(object_names):
        c = np.array(_object_color(i)) / 255
        legend_entries.append(mpatches.Patch(color=c, label=name))
    axes[1, 0].legend(handles=legend_entries, loc="lower right", fontsize=7,
                       framealpha=0.7)
    axes[1, 0].axis("off")

    # ── Bottom-right: Force Vectors ──
    # Dim original.
    force_bg = (orig.astype(np.float32) * 0.5).astype(np.uint8)
    force_pil = Image.fromarray(force_bg).convert("RGBA")
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    torque_map = {jt.joint_name: jt for jt in dynamics.torques}
    kin_map = {sk.segment_name: sk for sk in dynamics.kinematics}

    # Map segments to their end-joint for placement.
    seg_to_joint = {
        "upper_arm": "elbow", "forearm": "wrist", "hand": "fingertip",
        "base_rotation": "shoulder",
        "left_upper_arm": "left_elbow", "left_forearm": "left_wrist",
        "left_hand": "left_fingertip", "left_base_rotation": "left_shoulder",
        "right_upper_arm": "right_elbow", "right_forearm": "right_wrist",
        "right_hand": "right_fingertip", "right_base_rotation": "right_shoulder",
    }

    max_torque = max((jt.peak_torque_nm for jt in dynamics.torques), default=1.0)
    max_torque = max(max_torque, 0.1)

    for seg_name, jt in torque_map.items():
        torque_val = jt.torques_nm[frame_idx] if frame_idx < len(jt.torques_nm) else 0.0
        jname = seg_to_joint.get(seg_name)
        if not jname:
            continue
        pt = _get_joint_pixel(pose, jname, w, h)
        if not pt:
            continue

        # Color by capacity.
        cap = torque_val / max_torque
        if cap > 0.6:
            color = COLOR_TORQUE_HIGH
        else:
            color = COLOR_GRAVITY
        _draw_force_arrow(draw, pt[0], pt[1], torque_val, "up", color)

        # Label.
        draw.text((pt[0] + 10, pt[1] - 15), f"{torque_val:.2f}Nm",
                  fill=(255, 255, 255, 200))

    # COM dot.
    if dynamics.com and frame_idx < len(dynamics.com.positions_xy):
        cx, cy = dynamics.com.positions_xy[frame_idx]
        px, py = int(cx * w), int(cy * h)
        r = 8
        draw.ellipse([px - r, py - r, px + r, py + r], fill=COLOR_COM + (200,))
        draw.text((px + 10, py - 5), "COM", fill=(255, 255, 0, 220))

    # Balance ring.
    if dynamics.balance and frame_idx < len(dynamics.balance.scores):
        bscore = dynamics.balance.scores[frame_idx]
        if dynamics.com and frame_idx < len(dynamics.com.positions_xy):
            cx, cy = dynamics.com.positions_xy[frame_idx]
            px, py = int(cx * w), int(cy * h)
            ring_r = 20
            ring_color = (
                int(COLOR_BALANCE_OK[0] + (COLOR_BALANCE_BAD[0] - COLOR_BALANCE_OK[0]) * bscore),
                int(COLOR_BALANCE_OK[1] + (COLOR_BALANCE_BAD[1] - COLOR_BALANCE_OK[1]) * bscore),
                int(COLOR_BALANCE_OK[2] + (COLOR_BALANCE_BAD[2] - COLOR_BALANCE_OK[2]) * bscore),
            )
            draw.ellipse(
                [px - ring_r, py - ring_r, px + ring_r, py + ring_r],
                outline=ring_color + (180,), width=3,
            )

    force_img = Image.alpha_composite(force_pil, overlay).convert("RGB")
    axes[1, 1].imshow(np.array(force_img))
    axes[1, 1].set_title("Force Vectors", fontsize=12)

    # Summary text box.
    total_torque = sum(
        jt.torques_nm[frame_idx] if frame_idx < len(jt.torques_nm) else 0.0
        for jt in dynamics.torques
    )
    bal_str = ""
    if dynamics.balance and frame_idx < len(dynamics.balance.scores):
        bal_str = f"Balance: {dynamics.balance.scores[frame_idx]:.2f}"
    info = f"Total torque: {total_torque:.2f} Nm\n{bal_str}"
    axes[1, 1].text(
        0.02, 0.98, info, transform=axes[1, 1].transAxes,
        fontsize=9, color="white", va="top",
        bbox=dict(boxstyle="round", fc="black", alpha=0.7),
    )
    axes[1, 1].axis("off")

    plt.tight_layout()
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# Force Flow GIF
# ═══════════════════════════════════════════════════════════════════

def render_force_flow_gif(
    interpolated: list[InterpolatedFrame],
    image_paths: list[str],
    object_names: list[str],
    output_path: str,
):
    """Render smooth force-flow animated GIF."""
    logger.info(f"Rendering force flow GIF ({len(interpolated)} frames)...")

    # Preload and resize all source images.
    src_images = {}
    for i, p in enumerate(image_paths):
        src_images[i] = Image.open(p).convert("RGB").resize(GIF_SIZE, Image.LANCZOS)

    pil_frames: list[Image.Image] = []
    com_trail: list[tuple] = []
    gw, gh = GIF_SIZE

    for fi, iframe in enumerate(interpolated):
        # Cross-fade source images.
        img_a = src_images.get(iframe.source_a, src_images[0])
        img_b = src_images.get(iframe.source_b, img_a)
        blended = Image.blend(img_a, img_b, alpha=iframe.t)

        # Dim slightly for overlay contrast.
        blended = Image.blend(blended, Image.new("RGB", GIF_SIZE, (0, 0, 0)), alpha=0.25)

        # Overlay.
        overlay = Image.new("RGBA", GIF_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Force arrows at each keypoint.
        max_torque = max(iframe.joint_torques.values(), default=0.1) or 0.1
        for seg_name, torque_val in iframe.joint_torques.items():
            # Find the matching joint.
            seg_to_joint = {
                "upper_arm": "elbow", "forearm": "wrist", "hand": "fingertip",
                "base_rotation": "shoulder",
                "left_upper_arm": "left_elbow", "left_forearm": "left_wrist",
                "left_hand": "left_fingertip", "left_base_rotation": "left_shoulder",
                "right_upper_arm": "right_elbow", "right_forearm": "right_wrist",
                "right_hand": "right_fingertip", "right_base_rotation": "right_shoulder",
            }
            jname = seg_to_joint.get(seg_name)
            if not jname or jname not in iframe.keypoint_positions:
                continue
            jx, jy = iframe.keypoint_positions[jname]
            px, py = int(jx * gw), int(jy * gh)

            cap = torque_val / max_torque
            color = COLOR_TORQUE_HIGH if cap > 0.6 else COLOR_GRAVITY
            _draw_force_arrow(draw, px, py, torque_val, "up", color, scale=100.0)

        # Skeleton lines.
        chains = [
            ["left_shoulder", "left_elbow", "left_wrist", "left_fingertip"],
            ["right_shoulder", "right_elbow", "right_wrist", "right_fingertip"],
            ["shoulder", "elbow", "wrist", "fingertip"],
        ]
        for chain in chains:
            pts = []
            for jn in chain:
                if jn in iframe.keypoint_positions:
                    jx, jy = iframe.keypoint_positions[jn]
                    pts.append((int(jx * gw), int(jy * gh)))
            if len(pts) >= 2:
                for i in range(len(pts) - 1):
                    draw.line([pts[i], pts[i + 1]], fill=(255, 255, 255, 120), width=2)

        # COM trail.
        cx, cy = iframe.com_position
        com_px = (int(cx * gw), int(cy * gh))
        com_trail.append(com_px)
        if len(com_trail) > 15:
            com_trail = com_trail[-15:]

        for ti, tp in enumerate(com_trail):
            alpha = int(40 + (ti / len(com_trail)) * 160)
            r = 2 + int((ti / len(com_trail)) * 4)
            draw.ellipse([tp[0] - r, tp[1] - r, tp[0] + r, tp[1] + r],
                         fill=COLOR_COM + (alpha,))

        # Balance indicator.
        bs = iframe.balance_score
        ring_color = (
            int(COLOR_BALANCE_OK[0] + (COLOR_BALANCE_BAD[0] - COLOR_BALANCE_OK[0]) * bs),
            int(COLOR_BALANCE_OK[1] + (COLOR_BALANCE_BAD[1] - COLOR_BALANCE_OK[1]) * bs),
            int(COLOR_BALANCE_OK[2] + (COLOR_BALANCE_BAD[2] - COLOR_BALANCE_OK[2]) * bs),
        )
        draw.ellipse([com_px[0] - 15, com_px[1] - 15, com_px[0] + 15, com_px[1] + 15],
                     outline=ring_color + (180,), width=2)

        # Phase label + progress bar.
        progress = (fi / max(len(interpolated) - 1, 1))
        draw.rectangle([10, 10, 10 + int(progress * (gw - 20)), 18],
                       fill=(100, 200, 255, 160))
        draw.rectangle([10, 10, gw - 10, 18], outline=(255, 255, 255, 100), width=1)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        except Exception:
            font = ImageFont.load_default()
        label = f"{iframe.phase_label}  |  Frame {iframe.source_a + 1}-{iframe.source_b + 1}"
        draw.text((12, 22), label, fill=(255, 255, 255, 220), font=font)

        # Total torque text.
        total_t = sum(iframe.joint_torques.values())
        draw.text((12, gh - 30), f"Total: {total_t:.2f} Nm  Bal: {bs:.2f}",
                  fill=(255, 255, 255, 200), font=font)

        frame = Image.alpha_composite(blended.convert("RGBA"), overlay).convert("RGB")
        pil_frames.append(frame)

    # Save GIF.
    if pil_frames:
        pil_frames[0].save(
            output_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=GIF_FRAME_MS,
            loop=0,
            optimize=True,
        )
        logger.info(f"Saved force flow GIF: {output_path} ({len(pil_frames)} frames)")


# ═══════════════════════════════════════════════════════════════════
# Comparison View
# ═══════════════════════════════════════════════════════════════════

def render_comparison(
    image_paths: list[str],
    interpolated: list[InterpolatedFrame],
    output_path: str,
):
    """Side-by-side comparison: original images vs force-flow keyframes."""
    n_cols = min(6, len(image_paths))
    indices = np.linspace(0, len(image_paths) - 1, n_cols, dtype=int)

    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8))
    fig.suptitle("Originals (top) vs Force-Flow Keyframes (bottom)", fontsize=14)

    for col, idx in enumerate(indices):
        # Top: original.
        orig = Image.open(image_paths[idx]).convert("RGB")
        ax_top = axes[0, col] if n_cols > 1 else axes[0]
        ax_top.imshow(np.array(orig))
        ax_top.set_title(f"Frame {idx + 1}", fontsize=10)
        ax_top.axis("off")

        # Bottom: find the closest interpolated frame.
        target_frame = idx
        best = None
        best_dist = float("inf")
        for iframe in interpolated:
            dist = abs(iframe.source_a - target_frame) + iframe.t
            if dist < best_dist:
                best_dist = dist
                best = iframe
            if iframe.source_a > target_frame:
                break

        ax_bot = axes[1, col] if n_cols > 1 else axes[1]
        if best:
            # Render a quick force overlay frame.
            src = Image.open(image_paths[best.source_a]).convert("RGB").resize(GIF_SIZE)
            src = Image.blend(src, Image.new("RGB", GIF_SIZE, (0, 0, 0)), alpha=0.2)
            overlay = Image.new("RGBA", GIF_SIZE, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            gw, gh = GIF_SIZE

            max_torque = max(best.joint_torques.values(), default=0.1) or 0.1
            seg_to_joint = {
                "upper_arm": "elbow", "forearm": "wrist", "hand": "fingertip",
                "base_rotation": "shoulder",
                "left_upper_arm": "left_elbow", "left_forearm": "left_wrist",
                "left_hand": "left_fingertip", "left_base_rotation": "left_shoulder",
                "right_upper_arm": "right_elbow", "right_forearm": "right_wrist",
                "right_hand": "right_fingertip", "right_base_rotation": "right_shoulder",
            }
            for seg, tv in best.joint_torques.items():
                jn = seg_to_joint.get(seg)
                if jn and jn in best.keypoint_positions:
                    jx, jy = best.keypoint_positions[jn]
                    px, py = int(jx * gw), int(jy * gh)
                    color = COLOR_TORQUE_HIGH if tv / max_torque > 0.6 else COLOR_GRAVITY
                    _draw_force_arrow(draw, px, py, tv, "up", color, scale=100.0)

            comp = Image.alpha_composite(src.convert("RGBA"), overlay).convert("RGB")
            ax_bot.imshow(np.array(comp))
        else:
            ax_bot.imshow(np.array(orig))
        ax_bot.set_title(f"Force Flow", fontsize=10)
        ax_bot.axis("off")

    plt.tight_layout()
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved comparison: {output_path}")


# ═══════════════════════════════════════════════════════════════════
# Step 10: 3D Scene Reconstruction
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Scene3DFrame:
    """Reconstructed 3D scene for a single frame."""
    frame_idx: int
    joint_positions_3d: dict[str, tuple[float, float, float]]  # joint → (x, y, z) meters
    object_positions_3d: dict[str, tuple[float, float, float]]  # obj → (x, y, z) meters


class Scene3DReconstructor:
    """Combine MediaPipe joints_3d + depth maps for 3D scene reconstruction.

    MediaPipe provides world-space 3D joint positions (hip-centered, meters).
    Depth maps are used to back-project object 2D positions into 3D via
    a pinhole camera model, calibrated against joints_3d shoulder depth.
    """

    def __init__(
        self,
        object_profiles: list,
        image_size: tuple[int, int] = (640, 480),
        focal_length: float = 500.0,
    ):
        self.object_profiles = object_profiles
        self.img_w, self.img_h = image_size
        self.focal = focal_length
        self.stats: dict = {}

    def reconstruct(
        self,
        poses: list[PoseResult],
        depth_maps: list[np.ndarray],
    ) -> list[Scene3DFrame]:
        """Reconstruct 3D scenes from pose and depth data."""
        frames_3d = []

        for i, (pose, depth) in enumerate(zip(poses, depth_maps)):
            # 1. Joint positions from MediaPipe world landmarks (meters)
            joint_3d = {}
            for name, pos in pose.joints_3d.items():
                joint_3d[name] = pos

            # If no joints_3d available, estimate from 2D + depth
            if not joint_3d and pose.joints:
                joint_3d = self._backproject_joints(pose, depth)

            # 2. Object positions via depth back-projection
            obj_3d = self._backproject_objects(pose, depth, joint_3d)

            frames_3d.append(Scene3DFrame(
                frame_idx=i,
                joint_positions_3d=joint_3d,
                object_positions_3d=obj_3d,
            ))

        self.stats = {
            "n_frames": len(frames_3d),
            "avg_joints_per_frame": np.mean([
                len(f.joint_positions_3d) for f in frames_3d
            ]) if frames_3d else 0,
            "avg_objects_per_frame": np.mean([
                len(f.object_positions_3d) for f in frames_3d
            ]) if frames_3d else 0,
        }
        return frames_3d

    def _get_depth_scale(
        self,
        joint_3d: dict[str, tuple[float, float, float]],
        depth_map: np.ndarray,
        pose: PoseResult,
    ) -> float:
        """Calibrate depth map scale using known shoulder 3D depth."""
        # Use shoulder Z from joints_3d as reference
        h, w = depth_map.shape
        for ref_name in ["shoulder", "right_shoulder", "left_shoulder"]:
            if ref_name in joint_3d and ref_name in pose.joints:
                z_real = abs(joint_3d[ref_name][2])
                jx, jy = pose.joints[ref_name]
                px, py = int(jx * w), int(jy * h)
                px = max(0, min(w - 1, px))
                py = max(0, min(h - 1, py))
                d_val = depth_map[py, px]
                if d_val > 0.01 and z_real > 0.01:
                    return z_real / d_val
        return 1.0  # fallback: no calibration

    def _backproject_joints(
        self,
        pose: PoseResult,
        depth_map: np.ndarray,
    ) -> dict[str, tuple[float, float, float]]:
        """Estimate 3D joint positions from 2D joints + depth."""
        h, w = depth_map.shape
        result = {}
        for name, (jx, jy) in pose.joints.items():
            px, py = int(jx * w), int(jy * h)
            px = max(0, min(w - 1, px))
            py = max(0, min(h - 1, py))
            d = float(depth_map[py, px])
            z = d * 2.0  # rough metric conversion
            x = (jx - 0.5) * z * self.img_w / self.focal
            y = (jy - 0.5) * z * self.img_h / self.focal
            result[name] = (x, -y, z)  # flip y for right-hand coords
        return result

    def _backproject_objects(
        self,
        pose: PoseResult,
        depth_map: np.ndarray,
        joint_3d: dict,
    ) -> dict[str, tuple[float, float, float]]:
        """Back-project object 2D positions into 3D using depth + calibration."""
        scale = self._get_depth_scale(joint_3d, depth_map, pose)
        h, w = depth_map.shape
        result = {}

        for op in self.object_profiles:
            # Use estimated position from task.txt as fallback
            pos = op.estimated_position if hasattr(op, 'estimated_position') else (0, 0, 0)
            result[op.name] = (float(pos[0]), float(pos[1]), float(pos[2]))

        return result


# ═══════════════════════════════════════════════════════════════════
# Step 11: MuJoCo Simulation
# ═══════════════════════════════════════════════════════════════════

class SimulationRenderer:
    """Render MuJoCo simulation of SO-101 robot following extracted motion.

    Builds a scene with build_scene_xml(), maps MediaPipe joints_3d to
    SO-101 joint angles via IK, interpolates trajectory, and renders
    offscreen from multiple cameras.
    """

    # SO-101 joint names (after left_ prefix from build_scene_xml)
    JOINT_NAMES = [
        "shoulder_pan", "shoulder_lift", "elbow_flex",
        "wrist_flex", "wrist_roll", "gripper",
    ]

    def __init__(
        self,
        robot_xml: str,
        scene_objects: list,
        render_size: tuple[int, int] = (640, 480),
    ):
        import mujoco
        from cadenza_local.gym.env import build_scene_xml
        from cadenza_local.task import SceneObject

        self._mujoco = mujoco
        self.render_w, self.render_h = render_size

        # Convert object profiles to SceneObjects
        table_center_y = 0.20
        table_surface = 0.30
        mj_objects = []
        for op in scene_objects:
            pos = op.estimated_position if hasattr(op, 'estimated_position') else (0, 0, 0)
            size = op.estimated_size if hasattr(op, 'estimated_size') else (0.04, 0.1)
            mj_objects.append(SceneObject(
                name=op.name,
                shape=op.shape if hasattr(op, 'shape') else "cylinder",
                size=size,
                position=(pos[0], pos[1] + table_center_y, table_surface + pos[2]),
                mass=op.estimated_mass_kg if hasattr(op, 'estimated_mass_kg') else 0.1,
            ))

        # Build scene (single arm for pose-following)
        xml_str, self.n_joints, self.gripper_body = build_scene_xml(
            robot_xml, mj_objects, both_arms=False,
        )
        self.model = mujoco.MjModel.from_xml_string(xml_str)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        # Camera names for rendering
        self.camera_names = ["camera1", "camera2", "camera3"]
        self.camera_labels = ["front", "top", "side"]

        # Create renderer
        self.renderer = mujoco.Renderer(self.model, self.render_h, self.render_w)

    def _map_pose_to_joint_angles(self, pose: PoseResult) -> np.ndarray:
        """Convert MediaPipe joints_3d to SO-101 actuator targets.

        Returns array of 6 joint angles:
            [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
        """
        angles = np.zeros(6)

        j3d = pose.joints_3d
        if not j3d:
            return angles

        # Get key 3D positions (try both prefixed and unprefixed)
        def get_3d(name):
            for prefix in ["", "right_", "left_"]:
                key = f"{prefix}{name}"
                if key in j3d:
                    return np.array(j3d[key])
            return None

        shoulder = get_3d("shoulder")
        elbow = get_3d("elbow")
        wrist = get_3d("wrist")
        index = get_3d("index")

        # shoulder_pan: atan2(upper_arm_y, upper_arm_x) from shoulder→elbow
        if shoulder is not None and elbow is not None:
            upper = elbow - shoulder
            angles[0] = math.atan2(upper[1], upper[0])

            # shoulder_lift: angle from vertical in sagittal plane
            horiz = math.sqrt(upper[0] ** 2 + upper[1] ** 2)
            angles[1] = math.atan2(horiz, -upper[2])  # 0=hanging, pi/2=horizontal

        # elbow_flex: angle between upper_arm and forearm (always negative=flexed)
        if shoulder is not None and elbow is not None and wrist is not None:
            upper = elbow - shoulder
            fore = wrist - elbow
            upper_len = np.linalg.norm(upper)
            fore_len = np.linalg.norm(fore)
            if upper_len > 1e-6 and fore_len > 1e-6:
                cos_angle = np.dot(upper, fore) / (upper_len * fore_len)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                flex = math.acos(cos_angle) - math.pi  # negative = flexed
                angles[2] = max(-2.4, min(0.1, flex))

        # wrist_flex: difference between forearm and hand segment angles
        if elbow is not None and wrist is not None and index is not None:
            fore = wrist - elbow
            hand = index - wrist
            fore_len = np.linalg.norm(fore)
            hand_len = np.linalg.norm(hand)
            if fore_len > 1e-6 and hand_len > 1e-6:
                cos_a = np.dot(fore, hand) / (fore_len * hand_len)
                cos_a = np.clip(cos_a, -1.0, 1.0)
                angles[3] = math.acos(cos_a) - math.pi / 2
                angles[3] = max(-1.57, min(1.57, angles[3]))

        # wrist_roll: from pose angles
        wrist_roll = pose.angles.get("wrist_roll",
                     pose.angles.get("right_wrist_roll",
                     pose.angles.get("left_wrist_roll", 0.0)))
        angles[4] = math.radians(wrist_roll)
        angles[4] = max(-3.14, min(3.14, angles[4]))

        # gripper: from gripper_openness (0-90 deg) → (-0.5 to 0.5)
        openness = pose.gripper_openness
        angles[5] = (openness / 90.0) - 0.5  # 0→-0.5 (closed), 90→0.5 (open)

        return angles

    def render_trajectory(
        self,
        poses: list[PoseResult],
        interpolated: list[InterpolatedFrame],
        n_substeps: int = 5,
    ) -> dict[str, list[np.ndarray]]:
        """Run simulation following the pose trajectory, render from all cameras.

        Args:
            poses: Detected poses per keyframe.
            interpolated: Interpolated sub-frames from ForceFlowInterpolator.
            n_substeps: Physics sub-steps per frame for stability.

        Returns:
            Dict mapping camera label to list of RGB frames (H, W, 3) uint8.
        """
        mujoco = self._mujoco

        # Pre-compute joint angle targets per keyframe
        keyframe_targets = [self._map_pose_to_joint_angles(p) for p in poses]

        # Build per-interpolated-frame targets via LERP
        camera_frames = {label: [] for label in self.camera_labels}

        # Reset simulation
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        for iframe in interpolated:
            a_idx = min(iframe.source_a, len(keyframe_targets) - 1)
            b_idx = min(iframe.source_b, len(keyframe_targets) - 1)
            t = iframe.t

            # LERP between keyframe joint angles
            target_a = keyframe_targets[a_idx]
            target_b = keyframe_targets[b_idx]
            target = target_a + (target_b - target_a) * t

            # Apply to actuators
            n_ctrl = min(len(target), self.model.nu)
            self.data.ctrl[:n_ctrl] = target[:n_ctrl]

            # Step physics
            for _ in range(n_substeps):
                mujoco.mj_step(self.model, self.data)

            # Render from each camera
            for cam_name, cam_label in zip(self.camera_names, self.camera_labels):
                try:
                    self.renderer.update_scene(self.data, camera=cam_name)
                    rgb = self.renderer.render().copy()  # (H, W, 3) uint8
                    camera_frames[cam_label].append(rgb)
                except Exception:
                    # Camera may not exist; use blank frame
                    camera_frames[cam_label].append(
                        np.zeros((self.render_h, self.render_w, 3), dtype=np.uint8)
                    )

        return camera_frames

    def get_joint_trajectories(
        self,
        poses: list[PoseResult],
    ) -> dict[str, list[tuple[float, float, float]]]:
        """Extract 3D trajectory paths for key joints across all frames."""
        trajectories: dict[str, list[tuple[float, float, float]]] = {
            "shoulder": [], "elbow": [], "wrist": [], "fingertip": [],
        }

        for pose in poses:
            j3d = pose.joints_3d
            for joint_name in trajectories:
                # Try prefixed and unprefixed names
                pos = None
                for prefix in ["", "right_", "left_"]:
                    key = f"{prefix}{joint_name}"
                    if key in j3d:
                        pos = j3d[key]
                        break
                if pos is None:
                    # Use last known position or origin
                    pos = trajectories[joint_name][-1] if trajectories[joint_name] else (0, 0, 0)
                trajectories[joint_name].append(pos)

        return trajectories


# ═══════════════════════════════════════════════════════════════════
# Step 12: 3D Visualization
# ═══════════════════════════════════════════════════════════════════

def render_3d_simulation_gif(
    camera_frames: list[np.ndarray],
    interpolated: list[InterpolatedFrame],
    camera_label: str,
    output_path: str,
):
    """Render animated GIF from simulation camera with HUD overlay."""
    logger.info(f"Rendering 3D simulation GIF ({camera_label}, {len(camera_frames)} frames)...")

    pil_frames = []
    for fi, rgb in enumerate(camera_frames):
        pil = Image.fromarray(rgb).convert("RGBA")
        overlay = Image.new("RGBA", pil.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        w, h = pil.size

        # HUD: phase label + progress
        if fi < len(interpolated):
            iframe = interpolated[fi]
            progress = fi / max(len(camera_frames) - 1, 1)

            # Progress bar
            draw.rectangle([10, 10, 10 + int(progress * (w - 20)), 18],
                           fill=(100, 200, 255, 160))
            draw.rectangle([10, 10, w - 10, 18], outline=(255, 255, 255, 100), width=1)

            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 13)
            except Exception:
                font = ImageFont.load_default()

            # Phase + torque info
            total_torque = sum(iframe.joint_torques.values())
            label = f"{iframe.phase_label}  |  Torque: {total_torque:.2f} Nm  |  Bal: {iframe.balance_score:.2f}"
            draw.text((12, 22), label, fill=(255, 255, 255, 220), font=font)

            # Camera label
            draw.text((w - 80, h - 25), f"[{camera_label}]",
                       fill=(200, 200, 200, 180), font=font)

        frame = Image.alpha_composite(pil, overlay).convert("RGB")
        pil_frames.append(frame)

    if pil_frames:
        pil_frames[0].save(
            output_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=GIF_FRAME_MS,
            loop=0,
            optimize=True,
        )
        logger.info(f"Saved 3D simulation GIF: {output_path} ({len(pil_frames)} frames)")


def render_3d_trajectory_plot(
    trajectories: dict[str, list[tuple[float, float, float]]],
    object_profiles: list,
    output_path: str,
):
    """Render 3D trajectory plot of joint paths + object positions."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    colors = {"shoulder": "blue", "elbow": "green", "wrist": "orange", "fingertip": "red"}
    for joint_name, positions in trajectories.items():
        if not positions:
            continue
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        zs = [p[2] for p in positions]
        color = colors.get(joint_name, "gray")
        ax.plot(xs, ys, zs, "-o", color=color, markersize=3, linewidth=1.5,
                label=joint_name, alpha=0.8)
        # Mark start and end
        ax.scatter([xs[0]], [ys[0]], [zs[0]], color=color, s=60, marker="^", zorder=5)
        ax.scatter([xs[-1]], [ys[-1]], [zs[-1]], color=color, s=60, marker="s", zorder=5)

    # Object positions as markers
    for i, op in enumerate(object_profiles):
        pos = op.estimated_position if hasattr(op, 'estimated_position') else (0, 0, 0)
        ax.scatter([pos[0]], [pos[1]], [pos[2]], s=100, marker="D",
                   color=f"C{i + 4}", label=op.name, alpha=0.9, edgecolors="black")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("3D Joint Trajectories + Object Positions", fontsize=14)
    ax.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved 3D trajectory plot: {output_path}")


# ═══════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════

def write_summary(
    dynamics: DynamicsProfile,
    sandbox: SandboxResult,
    motor: MotorProfile,
    depth_stats: dict,
    seg_stats: dict,
    output_path: str,
    scene_3d_stats: Optional[dict] = None,
    sim_info: Optional[dict] = None,
):
    """Write text summary of all force-space analysis."""
    lines = [
        "=" * 60,
        "FORCE-SPACE VISUAL TEST SUMMARY",
        "=" * 60,
        "",
        dynamics.summary(),
        "",
        sandbox.summary_text(),
        "",
        motor.summary(),
        "",
        "--- Depth Estimation Stats ---",
        f"Images processed: {depth_stats.get('n_images', 0)}",
    ]
    for i, (m, s) in enumerate(zip(
        depth_stats.get("mean_depth", []),
        depth_stats.get("std_depth", []),
    )):
        lines.append(f"  Frame {i + 1}: mean={m:.3f}, std={s:.3f}")

    lines.append("")
    lines.append("--- Segmentation Stats ---")
    pixel_counts = seg_stats.get("pixel_counts", [])
    label_names = {0: "BG", 1: "Table", 2: "L_Arm", 3: "R_Arm"}
    for i, counts in enumerate(pixel_counts):
        parts = []
        for label, n in sorted(counts.items()):
            name = label_names.get(label, f"Obj{label - 4}")
            parts.append(f"{name}={n}")
        lines.append(f"  Frame {i + 1}: {', '.join(parts)}")

    if scene_3d_stats:
        lines.append("")
        lines.append("--- 3D Scene Reconstruction ---")
        lines.append(f"Frames reconstructed: {scene_3d_stats.get('n_frames', 0)}")
        lines.append(f"Avg joints/frame: {scene_3d_stats.get('avg_joints_per_frame', 0):.1f}")
        lines.append(f"Avg objects/frame: {scene_3d_stats.get('avg_objects_per_frame', 0):.1f}")

    if sim_info:
        lines.append("")
        lines.append("--- 3D MuJoCo Simulation ---")
        lines.append(f"Simulation frames: {sim_info.get('n_frames', 0)}")
        lines.append(f"Cameras: {', '.join(sim_info.get('cameras', []))}")
        lines.append(f"Robot joints: {sim_info.get('n_joints', 0)}")

    text = "\n".join(lines) + "\n"
    Path(output_path).write_text(text)
    logger.info(f"Saved summary: {output_path}")


# ═══════════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════════

def main(skip_depth: bool = False, no_video: bool = False, skip_3d: bool = False):
    """Run the full force-space visualization pipeline."""

    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Parse task.txt ──
    logger.info("Step 1: Parsing task.txt...")
    sections = parse_task_file(Path(TASK_FILE))

    object_profiles = []
    for obj in sections.get("objects", []):
        size = obj.get("size", (0.05, 0.15))
        pos = obj.get("position", (0.0, 0.0, 0.0))
        object_profiles.append(ObjectProfile(
            name=obj["name"],
            shape=obj.get("shape", "cylinder"),
            estimated_size=size if isinstance(size, tuple) else (0.05, 0.15),
            estimated_position=pos if isinstance(pos, tuple) else (0.0, 0.0, 0.0),
            estimated_mass_kg=float(obj.get("mass", 0.1)),
            interaction_types=obj.get("interactions", []),
            properties=obj.get("properties", {}),
        ))
    object_names = [o.name for o in object_profiles]
    logger.info(f"  Objects: {', '.join(object_names)}")

    # ── 2. Motion analysis ──
    logger.info("Step 2: Running motion analysis (MediaPipe)...")
    both_arms = sections.get("config", {}).get("both_arms", "false").lower() == "true"

    frames, blueprint, poses = analyze_motion_images(
        MOTION_DIR,
        object_names=object_names,
        task_description=sections.get("task", {}).get("description", ""),
        both_arms=both_arms,
        detect_objects=False,  # skip Groq API
    )
    logger.info(f"  Detected {len(frames)} frames, {len(blueprint.task_phases)} phases")

    # ── 2b. Compute dynamics ──
    logger.info("  Computing dynamics...")
    dynamics = compute_dynamics(blueprint, frames, robot=None, objects=object_profiles)
    logger.info(f"  Peak torque: {dynamics.peak_torque_nm:.2f} Nm")

    motor = compute_motor_profile(
        blueprint, frames, dynamics, robot=None, objects=object_profiles,
    )

    sandbox = run_force_sandbox(
        dynamics, motor, robot=None, objects=object_profiles, frames=frames,
    )
    logger.info(f"  Sandbox verdict: {sandbox.verdict}")

    # ── 3. Depth estimation ──
    logger.info("Step 3: Depth estimation...")
    image_paths = sorted(
        Path(MOTION_DIR).glob("front_*.png"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    if not image_paths:
        logger.error(f"No motion images found in {MOTION_DIR}")
        logger.error("Expected: front_1.png through front_12.png")
        sys.exit(1)

    depth_est = DepthEstimator(use_midas=not skip_depth)
    depth_maps = depth_est.estimate_batch([str(p) for p in image_paths])

    # ── 4. Segmentation ──
    logger.info("Step 4: Segmenting frames...")
    segmenter = FrameSegmenter(object_names)
    seg_masks = []
    for i, (img_path, depth_map, pose) in enumerate(zip(image_paths, depth_maps, poses)):
        img_arr = np.array(Image.open(img_path).convert("RGB"))
        mask = segmenter.segment(img_arr, depth_map, pose)
        seg_masks.append(mask)
    logger.info(f"  Segmented {len(seg_masks)} frames")

    # ── 5. Per-frame analysis panels ──
    logger.info("Step 5: Rendering per-frame analysis panels...")
    n_render = min(len(frames), len(image_paths), len(depth_maps), len(seg_masks), len(poses))
    for i in range(n_render):
        out_path = str(output_dir / f"frame_{i + 1:02d}_analysis.png")
        render_frame_analysis(
            frame_idx=i,
            image_path=str(image_paths[i]),
            depth_map=depth_maps[i],
            seg_mask=seg_masks[i],
            pose=poses[i],
            frame=frames[i],
            dynamics=dynamics,
            object_names=object_names,
            output_path=out_path,
        )
        logger.info(f"  Saved frame_{i + 1:02d}_analysis.png")

    # ── 6. Force flow interpolation ──
    logger.info("Step 6: Interpolating force flow...")
    interpolator = ForceFlowInterpolator(
        dynamics=dynamics, frames=frames, poses=poses,
    )
    interpolated = interpolator.interpolate()
    logger.info(f"  Generated {len(interpolated)} interpolated frames")

    # ── 7. Force flow GIF ──
    if not no_video:
        logger.info("Step 7: Rendering force flow GIF...")
        render_force_flow_gif(
            interpolated=interpolated,
            image_paths=[str(p) for p in image_paths],
            object_names=object_names,
            output_path=str(output_dir / "force_flow.gif"),
        )
    else:
        logger.info("Step 7: Skipped (--no-video)")

    # ── 8. Comparison view ──
    logger.info("Step 8: Rendering comparison view...")
    render_comparison(
        image_paths=[str(p) for p in image_paths],
        interpolated=interpolated,
        output_path=str(output_dir / "comparison.png"),
    )

    # ── 9. Summary (preliminary — updated after 3D steps) ──
    scene_3d_stats = None
    sim_info = None

    # ── 10. 3D Scene Reconstruction ──
    if not skip_3d:
        logger.info("Step 10: 3D scene reconstruction...")
        try:
            reconstructor = Scene3DReconstructor(
                object_profiles=object_profiles,
                image_size=(640, 480),
            )
            scenes_3d = reconstructor.reconstruct(poses, depth_maps)
            scene_3d_stats = reconstructor.stats
            logger.info(f"  Reconstructed {len(scenes_3d)} 3D frames")
            logger.info(f"  Avg joints/frame: {scene_3d_stats.get('avg_joints_per_frame', 0):.1f}")
        except Exception as e:
            logger.warning(f"  3D reconstruction failed: {e}")
            scenes_3d = []
    else:
        logger.info("Step 10: Skipped (--skip-3d)")
        scenes_3d = []

    # ── 11. MuJoCo Simulation ──
    if not skip_3d:
        logger.info("Step 11: MuJoCo simulation...")
        try:
            robot_xml = str(ROOT / "examples" / "bartender" / "assets" / "so101.xml")
            sim = SimulationRenderer(
                robot_xml=robot_xml,
                scene_objects=object_profiles,
                render_size=(640, 480),
            )
            camera_frames = sim.render_trajectory(
                poses=poses,
                interpolated=interpolated,
                n_substeps=5,
            )
            trajectories = sim.get_joint_trajectories(poses)
            sim_info = {
                "n_frames": len(interpolated),
                "cameras": sim.camera_labels,
                "n_joints": sim.n_joints,
            }
            logger.info(f"  Simulated {len(interpolated)} frames, {len(sim.camera_labels)} cameras")
        except Exception as e:
            logger.warning(f"  MuJoCo simulation failed: {e}")
            camera_frames = {}
            trajectories = {}
    else:
        logger.info("Step 11: Skipped (--skip-3d)")
        camera_frames = {}
        trajectories = {}

    # ── 12. 3D Visualization ──
    if not skip_3d and camera_frames:
        logger.info("Step 12: Rendering 3D visualization...")
        try:
            # Front camera GIF
            if "front" in camera_frames and camera_frames["front"]:
                render_3d_simulation_gif(
                    camera_frames=camera_frames["front"],
                    interpolated=interpolated,
                    camera_label="front",
                    output_path=str(output_dir / "3d_simulation.gif"),
                )

            # Side camera GIF
            if "side" in camera_frames and camera_frames["side"]:
                render_3d_simulation_gif(
                    camera_frames=camera_frames["side"],
                    interpolated=interpolated,
                    camera_label="side",
                    output_path=str(output_dir / "3d_simulation_side.gif"),
                )

            # 3D trajectory plot
            if trajectories:
                render_3d_trajectory_plot(
                    trajectories=trajectories,
                    object_profiles=object_profiles,
                    output_path=str(output_dir / "3d_trajectory.png"),
                )
        except Exception as e:
            logger.warning(f"  3D visualization failed: {e}")
    elif not skip_3d:
        logger.info("Step 12: Skipped (no simulation frames)")
    else:
        logger.info("Step 12: Skipped (--skip-3d)")

    # ── Final Summary ──
    logger.info("Writing final summary...")
    write_summary(
        dynamics=dynamics,
        sandbox=sandbox,
        motor=motor,
        depth_stats=depth_est.stats,
        seg_stats=segmenter.stats,
        output_path=str(output_dir / "summary.txt"),
        scene_3d_stats=scene_3d_stats,
        sim_info=sim_info,
    )

    logger.info("")
    logger.info("=" * 50)
    logger.info("DONE. Output in: tests/output_forcespace/")
    logger.info(f"  {n_render} analysis panels")
    if not no_video:
        logger.info(f"  force_flow.gif ({len(interpolated)} frames)")
    logger.info("  comparison.png")
    if not skip_3d and camera_frames:
        if "front" in camera_frames:
            logger.info("  3d_simulation.gif")
        if "side" in camera_frames:
            logger.info("  3d_simulation_side.gif")
        if trajectories:
            logger.info("  3d_trajectory.png")
    logger.info("  summary.txt")
    logger.info("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Force-space visual test — bartender example",
    )
    parser.add_argument(
        "--skip-depth", action="store_true",
        help="Skip MiDaS depth estimation (use flat depth maps)",
    )
    parser.add_argument(
        "--skip-3d", action="store_true",
        help="Skip 3D reconstruction, MuJoCo simulation, and 3D visualization",
    )
    parser.add_argument(
        "--no-video", action="store_true",
        help="Skip GIF video generation",
    )
    args = parser.parse_args()
    main(skip_depth=args.skip_depth, no_video=args.no_video, skip_3d=args.skip_3d)
