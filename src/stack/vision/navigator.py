"""VLA-driven trajectory recovery navigator.

When ``TrajectoryMonitor`` flags that the robot is stuck — distance to the
target hasn't decreased over the recent window — the stack hands the
camera frame, the depth estimate, and the relative target bearing to this
navigator. It returns one ``NavigationDecision`` naming a Cadenza action
that should make progress.

Two real models inform the decision:

  1. ``DepthEstimator`` (Depth-Anything-V2-Small) — predicts a per-pixel
     depth map from the RGB camera frame. We sample mean depth in the
     left / centre / right thirds of the frame's lower band (where the
     navigable ground / nearby obstacles live).
  2. ``HuggingFaceTB/SmolVLM-256M-Instruct`` — given the camera frame plus
     a structured prompt (depth in each region + target bearing/distance),
     it picks ONE action by name from the Cadenza vocabulary. We parse the
     action name out of its reply.

If the VLM fails to produce a valid name we fall back to a deterministic
depth-based heuristic: turn toward the target if the bearing is large,
otherwise step toward whichever third has the most free space.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from cadenza.stack.vision.depth import DepthEstimator

_LOG = logging.getLogger("cadenza.stack.vision.navigator")

# Actions the navigator may suggest. Must all exist in the Go1 vocabulary.
_VALID_ACTIONS = (
    "walk_forward",
    "turn_left",
    "turn_right",
    "side_step_left",
    "side_step_right",
    "climb_step",
    "sit",
)


@dataclass
class NavigationDecision:
    action: str
    rationale: str
    depth_left: float
    depth_center: float
    depth_right: float
    target_bearing_deg: float
    target_distance_m: float
    raw_response: str = ""
    used_fallback: bool = False


class VisionNavigator:
    """Vision-language recovery: depth + VLM reason about how to reach the target."""

    DEFAULT_VLM = "HuggingFaceTB/SmolVLM-256M-Instruct"

    def __init__(
        self,
        vlm_id: str | None = None,
        depth: DepthEstimator | None = None,
        device: str | None = None,
    ):
        self.vlm_id = vlm_id or self.DEFAULT_VLM
        self.depth = depth or DepthEstimator(device=device)
        self.device = device
        self._processor: Any = None
        self._vlm: Any = None
        self._loaded = False

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def load(self) -> None:
        if self._loaded:
            return
        self.depth.load()
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "VisionNavigator requires `transformers torch pillow`."
            ) from e
        _LOG.info("VisionNavigator: loading %s ...", self.vlm_id)
        self._processor = AutoProcessor.from_pretrained(self.vlm_id)
        self._vlm = AutoModelForImageTextToText.from_pretrained(
            self.vlm_id, dtype=torch.float32,
        )
        if self.device is None:
            self.device = self.depth.device or "cpu"
        self._vlm.to(self.device).eval()
        self._loaded = True
        _LOG.info("VisionNavigator: ready on %s", self.device)

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ── Inference ────────────────────────────────────────────────────────────

    def decide(
        self,
        rgb: np.ndarray,
        target_xy: tuple[float, float],
        robot_xy: tuple[float, float],
        robot_yaw: float,
    ) -> NavigationDecision:
        """Pick a recovery action by looking at the world and the target."""
        if not self._loaded:
            self.load()

        depth_map = self.depth.predict(rgb)
        d_left, d_center, d_right = self._sample_depth_regions(depth_map)
        bearing_deg, distance_m = self._target_in_robot_frame(
            target_xy, robot_xy, robot_yaw,
        )

        prompt = self._build_prompt(bearing_deg, distance_m, d_left, d_center, d_right)
        raw = ""
        try:
            raw = self._run_vlm(rgb, prompt)
            action = self._parse_action(raw)
        except Exception as e:
            _LOG.warning("VisionNavigator VLM failed (%s); using depth-only heuristic", e)
            action = None

        used_fallback = False
        if action is None:
            action = self._depth_heuristic(d_left, d_center, d_right, bearing_deg)
            used_fallback = True

        return NavigationDecision(
            action=action,
            rationale=(
                f"vision recovery: "
                f"bearing={bearing_deg:+.0f}°, dist={distance_m:.1f}m, "
                f"depth(L,C,R)=({d_left:.2f}, {d_center:.2f}, {d_right:.2f})"
                + (" [depth-fallback]" if used_fallback else " [VLM]")
            ),
            depth_left=d_left, depth_center=d_center, depth_right=d_right,
            target_bearing_deg=bearing_deg,
            target_distance_m=distance_m,
            raw_response=raw.strip()[:200],
            used_fallback=used_fallback,
        )

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _sample_depth_regions(depth: np.ndarray) -> tuple[float, float, float]:
        """Mean depth in the lower-mid band of the frame's left/center/right thirds.

        We focus on the lower 45–85% of the frame: this is roughly the
        navigable ground + nearby obstacles, ignoring sky and background.
        """
        h, w = depth.shape[:2]
        ymin, ymax = int(h * 0.45), int(h * 0.85)
        third = w // 3
        left = float(depth[ymin:ymax, :third].mean())
        center = float(depth[ymin:ymax, third:2 * third].mean())
        right = float(depth[ymin:ymax, 2 * third:].mean())
        return left, center, right

    @staticmethod
    def _target_in_robot_frame(
        target_xy: tuple[float, float],
        robot_xy: tuple[float, float],
        robot_yaw: float,
    ) -> tuple[float, float]:
        """Bearing (deg, +=left, -=right) and distance to the target.

        Cadenza convention: forward = -x in body frame, so the world-frame
        forward heading is ``yaw + π``.
        """
        dx = target_xy[0] - robot_xy[0]
        dy = target_xy[1] - robot_xy[1]
        target_world_heading = math.atan2(dy, dx)
        robot_world_heading = robot_yaw + math.pi
        bearing = target_world_heading - robot_world_heading
        # Wrap to (-π, π]
        bearing = (bearing + math.pi) % (2 * math.pi) - math.pi
        return math.degrees(bearing), math.hypot(dx, dy)

    @staticmethod
    def _build_prompt(
        bearing_deg: float, distance_m: float,
        d_left: float, d_center: float, d_right: float,
    ) -> str:
        if bearing_deg > 25:
            side = "to the left"
        elif bearing_deg < -25:
            side = "to the right"
        else:
            side = "ahead"
        return (
            "The robot is stuck and needs to reach a target "
            f"{distance_m:.1f} meters {side} (bearing {bearing_deg:+.0f} degrees). "
            "Estimated forward depths from the camera (relative units, "
            "higher = farther / more open): "
            f"left={d_left:.2f}, center={d_center:.2f}, right={d_right:.2f}. "
            "Pick ONE action to make progress: "
            "walk_forward, turn_left, turn_right, side_step_left, side_step_right. "
            "Reply with ONLY the action name."
        )

    def _run_vlm(self, rgb: np.ndarray, prompt: str) -> str:
        from PIL import Image
        import torch

        if rgb.dtype != np.uint8:
            arr = rgb * 255.0 if rgb.max() <= 1.0 else rgb
            rgb_u8 = arr.clip(0, 255).astype(np.uint8)
        else:
            rgb_u8 = rgb
        image = Image.fromarray(rgb_u8)

        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ]}]
        prompt_text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self._processor(
            text=prompt_text, images=[image], return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            output_ids = self._vlm.generate(
                **inputs, max_new_tokens=24, do_sample=False,
            )
        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        return self._processor.decode(generated, skip_special_tokens=True)

    @staticmethod
    def _parse_action(text: str) -> str | None:
        cleaned = text.lower()
        # Match longest names first so "side_step_left" wins over "left".
        for name in sorted(_VALID_ACTIONS, key=len, reverse=True):
            if name in cleaned:
                return name
        # Tolerate hyphenated or spaced variants from chatty models.
        if "side step left" in cleaned or "step left" in cleaned:
            return "side_step_left"
        if "side step right" in cleaned or "step right" in cleaned:
            return "side_step_right"
        if "walk" in cleaned and "back" not in cleaned:
            return "walk_forward"
        return None

    @staticmethod
    def _depth_heuristic(
        d_left: float, d_center: float, d_right: float, bearing_deg: float,
    ) -> str:
        # Strong bearing → turn first to face the target.
        if bearing_deg > 30:
            return "turn_left"
        if bearing_deg < -30:
            return "turn_right"
        # Otherwise step toward whichever third is most open.
        widest_label, _widest_value = max(
            (("left", d_left), ("center", d_center), ("right", d_right)),
            key=lambda x: x[1],
        )
        if widest_label == "center":
            return "walk_forward"
        return f"side_step_{widest_label}"
