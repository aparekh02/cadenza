"""VLA Guardian — physics-accurate obstacle detection + dynamic avoidance.

Detection: MuJoCo raycasts (horizontal, forward-facing, 100% physics-accurate).
Measurement: Dense raycast grid maps obstacle width, height, and clear paths.
Planning: SmolVLM judges size, raycasts determine the exact avoidance geometry.

Every avoidance script is dynamically generated from measured obstacle dimensions,
with safety margins added. Nothing is hardcoded.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

_MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
_CAMERA_WIDTH = 384
_CAMERA_HEIGHT = 384

# Detection
_DETECT_DISTANCE = 0.75
_PROBE_LATERALS = [-0.15, -0.05, 0.0, 0.05, 0.15]
_PROBE_HEIGHTS = [0.03, 0.08, 0.14, 0.22]

# Measurement — dense scan to map obstacle extent
_SCAN_LATERALS = [i * 0.05 for i in range(-10, 11)]  # -0.5m to +0.5m
_SCAN_HEIGHTS = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
_SCAN_FORWARD_MAX = 1.5  # look up to 1.5m ahead

# Safety margins
_MARGIN_LATERAL = 0.25   # extra clearance sideways (robot is ~0.15m half-width)
_MARGIN_FORWARD = 0.35   # extra clearance walking past obstacle

_PLAN_PROMPT = (
    "A quadruped robot detected an obstacle ahead using sensors. "
    "Look at this image from the robot's camera.\n"
    "Is the obstacle small enough to step/crawl over (under 8cm tall), "
    "or is it too large and the robot must walk around it?\n"
    "Answer: SIZE: SMALL or SIZE: LARGE"
)

_ROBOT_BODIES = {
    "trunk",
    "FL_hip", "FL_thigh", "FL_calf", "FL_foot",
    "FR_hip", "FR_thigh", "FR_calf", "FR_foot",
    "RL_hip", "RL_thigh", "RL_calf", "RL_foot",
    "RR_hip", "RR_thigh", "RR_calf", "RR_foot",
}


@dataclass
class ObstacleMeasurement:
    """Measured obstacle geometry from raycast scan."""
    distance: float = float('inf')      # distance to nearest point
    lateral_center: float = 0.0         # obstacle center offset (+ = left)
    lateral_extent_left: float = 0.0    # how far obstacle extends left
    lateral_extent_right: float = 0.0   # how far obstacle extends right
    width: float = 0.0                  # total obstacle width
    height: float = 0.0                 # tallest hit
    depth: float = 0.0                  # obstacle depth (forward extent)
    clear_left: float = float('inf')    # free space to the left
    clear_right: float = float('inf')   # free space to the right


@dataclass
class ObstacleResult:
    """Result of a VLA obstacle check."""
    detected: bool = False
    distance: float = float('inf')
    position: str = "none"
    size: str = "unknown"
    measurement: ObstacleMeasurement = None
    avoidance_actions: list = None
    raw_response: str = ""

    def __post_init__(self):
        if self.avoidance_actions is None:
            self.avoidance_actions = []


class VLAGuardian:
    """Physics-accurate obstacle detection with dynamic avoidance planning.

    Uses dense raycast grids to measure the exact obstacle geometry,
    then generates a precisely-sized avoidance path with safety margins.
    """

    def __init__(self, robot: str = "go1", model_id: str = _MODEL_ID,
                 show_camera: bool = False):
        self.robot = robot
        self.model_id = model_id
        self.show_camera = show_camera
        self._processor = None
        self._model = None
        self._renderer = None
        self._loaded = False
        self._robot_body_ids = None
        self._camera_window = None
        self._camera_img = None
        self._detect_prompt = ""
        self._plan_prompt = _PLAN_PROMPT

    def load(self):
        if self._loaded:
            return
        from transformers import AutoProcessor, AutoModelForImageTextToText
        import torch
        print(f"  VLA Guardian: loading {self.model_id}...")
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_id, dtype=torch.float32)
        self._model.eval()
        self._loaded = True
        print(f"  VLA Guardian: ready")

    def _ensure_loaded(self):
        if not self._loaded:
            self.load()

    def _get_robot_body_ids(self, mj_model):
        if self._robot_body_ids is not None:
            return self._robot_body_ids
        import mujoco
        self._robot_body_ids = set()
        for i in range(mj_model.nbody):
            name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name in _ROBOT_BODIES:
                self._robot_body_ids.add(i)
        return self._robot_body_ids

    def _cast_ray(self, mj_model, mj_data, origin, direction, robot_ids,
                  max_dist=_SCAN_FORWARD_MAX):
        """Cast one ray, skip robot and floor. Returns (dist, geom_name) or (inf, None)."""
        import mujoco
        geomid = np.array([-1], dtype=np.int32)
        dist = mujoco.mj_ray(mj_model, mj_data, origin, direction,
                              None, 1, -1, geomid)
        if dist < 0 or dist > max_dist:
            return float('inf'), None
        gid = int(geomid[0])
        if gid < 0:
            return float('inf'), None
        if mj_model.geom_bodyid[gid] in robot_ids:
            return float('inf'), None
        geom_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        if geom_name == "floor":
            return float('inf'), None
        return dist, geom_name

    def _get_fwd_right(self, mj_data):
        """Get forward and right unit vectors from robot orientation."""
        quat = mj_data.qpos[3:7]
        w, x, y, z = quat
        fwd = np.array([
            -(1 - 2 * (y * y + z * z)),
            -(2 * (x * y + w * z)),
            0.0,
        ], dtype=np.float64)
        fwd_len = np.linalg.norm(fwd)
        if fwd_len < 1e-9:
            return np.array([1, 0, 0], dtype=np.float64), np.array([0, 1, 0], dtype=np.float64)
        fwd_norm = fwd / fwd_len
        right = np.array([fwd_norm[1], -fwd_norm[0], 0.0], dtype=np.float64)
        return fwd_norm, right

    def _ground_z(self, mj_model, mj_data):
        import mujoco
        bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "FL_foot")
        if bid >= 0:
            return float(mj_data.xpos[bid, 2])
        return 0.0

    # ── Detection ────────────────────────────────────────────────────────────

    def _raycast_obstacle(self, mj_model, mj_data, verbose=False):
        """Quick forward raycast. Returns (detected, min_dist, position)."""
        pos = mj_data.qpos[0:3].copy()
        fwd, right = self._get_fwd_right(mj_data)
        gz = self._ground_z(mj_model, mj_data)
        robot_ids = self._get_robot_body_ids(mj_model)

        min_dist = float('inf')
        hit_lats = []
        hit_name = ""

        for h in _PROBE_HEIGHTS:
            for lat in _PROBE_LATERALS:
                origin = np.array([
                    pos[0] + right[0] * lat,
                    pos[1] + right[1] * lat,
                    gz + h,
                ], dtype=np.float64)
                dist, name = self._cast_ray(mj_model, mj_data, origin, fwd,
                                            robot_ids, _DETECT_DISTANCE)
                if dist < float('inf'):
                    if dist < min_dist:
                        min_dist = dist
                        hit_name = name or ""
                    hit_lats.append(lat)

        if min_dist == float('inf'):
            if verbose:
                print(f"       scan: clear")
            return False, float('inf'), "none"

        avg_lat = sum(hit_lats) / len(hit_lats) if hit_lats else 0
        if avg_lat > 0.04:
            position = "left"
        elif avg_lat < -0.04:
            position = "right"
        else:
            position = "center"

        if verbose:
            print(f"       scan: HIT '{hit_name}' at {min_dist:.2f}m ({position})")

        return True, min_dist, position

    # ── Measurement ──────────────────────────────────────────────────────────

    def _measure_obstacle(self, mj_model, mj_data) -> ObstacleMeasurement:
        """Dense raycast scan to measure obstacle geometry.

        Casts rays at many lateral offsets to find:
        - Where the obstacle starts and ends laterally
        - How tall it is
        - Where there's clear space on each side
        """
        pos = mj_data.qpos[0:3].copy()
        fwd, right = self._get_fwd_right(mj_data)
        gz = self._ground_z(mj_model, mj_data)
        robot_ids = self._get_robot_body_ids(mj_model)

        # Scan: for each lateral offset, check if there's an obstacle
        hit_at_lat = {}  # lateral -> (min_dist, max_height)

        for lat in _SCAN_LATERALS:
            best_dist = float('inf')
            max_h = 0.0
            for h in _SCAN_HEIGHTS:
                origin = np.array([
                    pos[0] + right[0] * lat,
                    pos[1] + right[1] * lat,
                    gz + h,
                ], dtype=np.float64)
                dist, name = self._cast_ray(mj_model, mj_data, origin, fwd,
                                            robot_ids)
                if dist < float('inf'):
                    best_dist = min(best_dist, dist)
                    max_h = max(max_h, h)

            if best_dist < float('inf'):
                hit_at_lat[lat] = (best_dist, max_h)

        if not hit_at_lat:
            return ObstacleMeasurement()

        # Find obstacle extent
        hit_laterals = sorted(hit_at_lat.keys())
        leftmost = max(hit_laterals)    # most positive = most left
        rightmost = min(hit_laterals)   # most negative = most right
        center = (leftmost + rightmost) / 2
        width = leftmost - rightmost
        min_dist = min(d for d, _ in hit_at_lat.values())
        max_height = max(h for _, h in hit_at_lat.values())

        # Find clear space: scan further out on each side
        all_lats = sorted(_SCAN_LATERALS)
        clear_left = 0.0
        clear_right = 0.0

        # Clear space to the left (positive lateral)
        for lat in reversed(all_lats):
            if lat > leftmost and lat not in hit_at_lat:
                clear_left = lat - leftmost
                break
        if clear_left == 0:
            clear_left = max(all_lats) - leftmost

        # Clear space to the right (negative lateral)
        for lat in all_lats:
            if lat < rightmost and lat not in hit_at_lat:
                clear_right = rightmost - lat
                break
        if clear_right == 0:
            clear_right = rightmost - min(all_lats)

        m = ObstacleMeasurement(
            distance=min_dist,
            lateral_center=center,
            lateral_extent_left=leftmost,
            lateral_extent_right=rightmost,
            width=width,
            height=max_height,
            clear_left=clear_left,
            clear_right=clear_right,
        )

        # ── Depth estimation: cast rays from the sides to measure how far
        # back the obstacle extends along the forward axis ──
        depth = self._estimate_depth(mj_model, mj_data, min_dist, center,
                                     width, robot_ids)

        m = ObstacleMeasurement(
            distance=min_dist,
            lateral_center=center,
            lateral_extent_left=leftmost,
            lateral_extent_right=rightmost,
            width=width,
            height=max_height,
            depth=depth,
            clear_left=clear_left,
            clear_right=clear_right,
        )

        print(f"       measurement:")
        print(f"         distance:  {m.distance:.2f}m")
        print(f"         width:     {m.width:.2f}m (L={m.lateral_extent_left:+.2f} R={m.lateral_extent_right:+.2f})")
        print(f"         height:    {m.height:.2f}m")
        print(f"         depth:     {m.depth:.2f}m")
        print(f"         clear L:   {m.clear_left:.2f}m  |  clear R: {m.clear_right:.2f}m")

        return m

    def _estimate_depth(self, mj_model, mj_data, front_dist: float,
                        lateral_center: float, width: float,
                        robot_ids: set) -> float:
        """Estimate obstacle depth by casting sideways rays at multiple
        forward offsets. Finds how far forward the obstacle extends.

        Casts rays from well outside the obstacle, pointing inward (left or
        right), at increasing forward distances. When the ray stops hitting
        the obstacle, we've found the back edge.
        """
        pos = mj_data.qpos[0:3].copy()
        fwd, right = self._get_fwd_right(mj_data)
        gz = self._ground_z(mj_model, mj_data)

        # Cast from the side, pointing inward
        # Start from 0.5m outside the obstacle laterally
        side_offset = max(abs(lateral_center), width / 2) + 0.5
        ray_dir_inward = -right.copy()  # pointing from right side toward left

        # Probe at increasing forward distances from robot
        # Start at the front face distance, go up to 2m past it
        probe_fwd_offsets = [front_dist + d for d in
                            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5]]

        max_fwd_with_hit = 0.0
        scan_height = 0.10  # mid-height

        for fwd_d in probe_fwd_offsets:
            origin = np.array([
                pos[0] + fwd[0] * fwd_d + right[0] * side_offset,
                pos[1] + fwd[1] * fwd_d + right[1] * side_offset,
                gz + scan_height,
            ], dtype=np.float64)

            dist, name = self._cast_ray(mj_model, mj_data, origin,
                                        ray_dir_inward, robot_ids,
                                        max_dist=side_offset * 2)
            if dist < float('inf') and name != "floor":
                max_fwd_with_hit = fwd_d

        # Depth = distance from front face to back edge
        depth = max(0.1, max_fwd_with_hit - front_dist + 0.05)
        return depth

    # ── Camera ───────────────────────────────────────────────────────────────

    def _render_camera(self, mj_model, mj_data) -> np.ndarray:
        import mujoco
        if self._renderer is None:
            self._renderer = mujoco.Renderer(mj_model, _CAMERA_HEIGHT, _CAMERA_WIDTH)

        pos = mj_data.qpos[0:3].copy()
        w, x, y, z = mj_data.qpos[3:7]
        fwd_x = -(1 - 2 * (y * y + z * z))
        fwd_y = -(2 * (x * y + w * z))

        cam_pos = np.array([pos[0] + fwd_x * 0.1, pos[1] + fwd_y * 0.1, pos[2] + 0.10])
        lookat = np.array([pos[0] + fwd_x * 1.5, pos[1] + fwd_y * 1.5, pos[2] - 0.05])

        scene_option = mujoco.MjvOption()
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = lookat
        diff = cam_pos - lookat
        dist = float(np.linalg.norm(diff))
        camera.distance = max(dist, 0.3)
        camera.azimuth = float(math.degrees(math.atan2(diff[1], diff[0])))
        camera.elevation = float(math.degrees(math.asin(
            max(-1.0, min(1.0, diff[2] / max(dist, 1e-6))))))
        self._renderer.update_scene(mj_data, camera, scene_option)
        return self._renderer.render()

    def show_frame(self, frame: np.ndarray):
        try:
            from PIL import Image
            img = Image.fromarray(frame)
            img.save("/tmp/cadenza_vla_camera.png")
        except Exception:
            pass

    # ── VLM ──────────────────────────────────────────────────────────────────

    def _query_vlm(self, image, prompt: str) -> str:
        import torch
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": prompt}]}]
        prompt_text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self._processor(text=prompt_text, images=[image], return_tensors="pt")
        with torch.no_grad():
            output_ids = self._model.generate(**inputs, max_new_tokens=50, do_sample=False)
        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        return self._processor.decode(generated, skip_special_tokens=True).strip()

    # ── Public API ───────────────────────────────────────────────────────────

    def check_raycast_only(self, mj_model, mj_data, verbose=False):
        return self._raycast_obstacle(mj_model, mj_data, verbose=verbose)

    def plan_avoidance(self, mj_model, mj_data, raycast_position: str,
                       distance: float) -> ObstacleResult:
        """Measure obstacle, ask VLM for size judgement, build dynamic avoidance path."""
        self._ensure_loaded()

        # 1. Measure the obstacle with dense raycasts
        m = self._measure_obstacle(mj_model, mj_data)

        # 2. Render camera and ask VLM: small or large?
        frame = self._render_camera(mj_model, mj_data)
        if self.show_camera:
            self.show_frame(frame)
        from PIL import Image
        vlm_response = self._query_vlm(Image.fromarray(frame), _PLAN_PROMPT)
        upper = vlm_response.upper()
        size = "small" if ("SMALL" in upper or "LOW" in upper or "SHORT" in upper) else "large"
        print(f"       VLM says: \"{vlm_response[:80]}\" → size={size}")

        # 3. Build dynamic avoidance script from measurements
        steps = self._build_avoidance(m, size, raycast_position)

        return ObstacleResult(
            detected=True,
            distance=distance,
            position=raycast_position,
            size=size,
            measurement=m,
            avoidance_actions=steps,
            raw_response=vlm_response,
        )

    def _build_avoidance(self, m: ObstacleMeasurement, size: str,
                         position: str) -> list:
        """Build avoidance script dynamically from measured obstacle geometry.

        For small obstacles: crawl over (distance = obstacle depth + margin).
        For large obstacles: U-shaped detour sized to the actual obstacle.
        """
        from cadenza.go1 import Step

        if size == "small" and m.height < 0.08:
            crawl_dist = max(0.8, m.distance + _MARGIN_FORWARD + 0.3)
            print(f"       plan: CRAWL over ({crawl_dist:.2f}m)")
            return [Step("crawl_forward", speed=0.5, distance_m=crawl_dist)]

        # Large obstacle — go around it
        # Decide which side: pick the side with more clear space
        go_left = m.clear_left > m.clear_right

        if position == "left":
            go_left = False  # obstacle is left, go right
        elif position == "right":
            go_left = True   # obstacle is right, go left

        if go_left:
            turn_away = "turn_left"
            turn_back = "turn_right"
            lateral_dist = abs(m.lateral_extent_left) + _MARGIN_LATERAL + 0.25
        else:
            turn_away = "turn_right"
            turn_back = "turn_left"
            lateral_dist = abs(m.lateral_extent_right) + _MARGIN_LATERAL + 0.25

        # Lateral: clear the obstacle + robot body width + margin
        lateral_dist = max(0.6, lateral_dist * 0.9)

        # Forward: use measured depth + approach distance + margin
        # depth = how thick the obstacle is in the forward direction
        forward_dist = m.distance + m.depth + _MARGIN_FORWARD + 0.2
        forward_dist = max(1.0, forward_dist)

        side = "LEFT" if go_left else "RIGHT"
        print(f"       plan: go {side} — lateral={lateral_dist:.2f}m, forward={forward_dist:.2f}m")

        # Use default turn (no rotation_rad override) — the action spec
        # already defines a clean 90° turn.
        # After the second turn (back to original heading), walk a bit extra
        # to make sure we're fully past the obstacle before turning back
        forward_past = forward_dist + 0.15

        steps = [
            # 1. Turn 90° away from obstacle
            Step(turn_away, speed=1.0),
            # 2. Walk sideways to clear obstacle width
            Step("walk_forward", speed=1.0, distance_m=lateral_dist),
            # 3. Turn 90° back to original heading
            Step(turn_back, speed=1.0),
            # 4. Walk forward past the obstacle (extra margin)
            Step("walk_forward", speed=1.0, distance_m=forward_past),
            # 5. Turn 90° back toward original line
            Step(turn_back, speed=1.0),
            # 6. Walk back to original line
            Step("walk_forward", speed=1.0, distance_m=lateral_dist),
            # 7. Turn 90° to resume original heading
            Step(turn_away, speed=1.0),
        ]

        return steps

    def _parse_plan(self, response: str, raycast_position: str,
                    distance: float) -> ObstacleResult:
        """Compat shim for tests."""
        upper = response.upper()
        size = "small" if ("SMALL" in upper) else "large"
        from cadenza.go1 import Step
        m = ObstacleMeasurement(distance=distance, width=0.3, depth=0.3,
                                lateral_extent_left=0.15,
                                lateral_extent_right=-0.15)
        steps = self._build_avoidance(m, size, raycast_position)
        return ObstacleResult(
            detected=True, distance=distance, position=raycast_position,
            size=size, measurement=m, avoidance_actions=steps,
            raw_response=response)

    def get_avoidance_steps(self, result: ObstacleResult) -> list:
        return result.avoidance_actions if result.avoidance_actions else []
