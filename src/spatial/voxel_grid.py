"""VoxelGrid — sparse 3D occupancy grid with per-voxel confidence.

Unlike ``SpatialMap`` (2.5D: one max-height value per (x, y) column),
this stores true 3D occupancy: ``grid[(x, y, z)]`` exists only for
voxels actually observed as occupied. Built for cases the 2.5D map
can't represent — overhangs, gaps with floor both above and below,
and multi-level structures (e.g. stairwells spanning several floors)
where collapsing to a single height-per-column loses information.

Storage is a dict keyed by integer voxel index, not a dense numpy
array — most of 3D space around a robot is empty air, so a dense
``(N, N, N)`` array would pay memory for cells that never get touched.

Design note — occupancy, not TSDF:
    TSDF (Truncated Signed Distance
    Function) fusion, where each voxel stores signed distance to the
    nearest surface rather than a binary occupied/confidence flag. TSDF
    buys smooth, accurate surface reconstruction by averaging multiple
    overlapping depth readings of the same surface — genuinely valuable
    for grasping or fine terrain following, but it requires camera
    intrinsics (not just FOV), ray-marching every voxel a ray passes
    through (not just where it terminates), and a per-voxel weighted
    running average instead of one confidence float — real added
    compute on every frame.
    For the navigation/staircase use case this module targets — "is
    the path ahead clear, what's the step height" — confident binary
    occupancy answers that as well as TSDF would, at a fraction of the
    cost. Deliberately deferred, not overlooked: revisit if/when
    grasping or fine-grained terrain work actually needs surface
    smoothness this representation can't give.

    grid = VoxelGrid(voxel_m=0.10)
    grid.update_from_depth(depth, robot_pos=(x, y), robot_yaw=yaw, robot_z=z)
    grid.is_occupied(1.2, 0.4, 0.3)   # → True/False
    grid.occupied_in_region(center=(x, y, z), radius_m=1.0)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

VoxelKey = tuple[int, int, int]


@dataclass
class Voxel:
    """One occupied cell. Mutable — confidence decays and gets refreshed
    in place rather than replacing the whole record on every observation."""
    confidence: float = 1.0
    last_seen: float = field(default_factory=time.time)
    hit_count: int = 1


class VoxelGrid:
    """Sparse 3D occupancy grid, indexed by world-space voxel coordinates.

    Args:
        voxel_m: edge length of one voxel cube, in metres. Smaller =
            finer detail, more voxels, more memory/compute.
        decay_per_sec: confidence multiplier applied per second since a
            voxel was last reconfirmed. 1.0 = no decay (permanent,
            matches SpatialMap's current behaviour). Something like
            0.98 means a voxel loses ~2%/sec confidence if not reseen.
        min_confidence: voxels decayed below this are dropped entirely
            on the next prune() call, rather than kept around forever
            at near-zero confidence.
        max_voxels: soft cap. If exceeded, prune() evicts the
            lowest-confidence voxels first. Prevents unbounded growth
            on long-running deployments (the same growth issue the
            existing SpatialMap.points set has today).
    """

    def __init__(
        self,
        voxel_m: float = 0.10,
        decay_per_sec: float = 1.0,
        min_confidence: float = 0.05,
        max_voxels: int = 200_000,
    ):
        self.voxel_m = float(voxel_m)
        self.decay_per_sec = float(decay_per_sec)
        self.min_confidence = float(min_confidence)
        self.max_voxels = int(max_voxels)
        self._voxels: dict[VoxelKey, Voxel] = {}

    # ── Indexing ─────────────────────────────────────────────────────────

    def world_to_voxel(self, x: float, y: float, z: float) -> VoxelKey:
        """World-space metres → integer voxel index.

        Uses floor(v/voxel_m + 0.5) rather than Python's built-in
        round(), which applies banker's rounding (round-half-to-even):
        round(2.5) == 2, not 3. That's invisible most of the time but
        silently misplaces any point landing exactly on a half-voxel
        boundary — and real depth data routinely does, since boundaries
        recur every voxel_m/2 metres. floor(x + 0.5) always rounds
        .5 up, which is the behaviour every caller of this method
        actually expects.
        """
        return (
            int(math.floor(x / self.voxel_m + 0.5)),
            int(math.floor(y / self.voxel_m + 0.5)),
            int(math.floor(z / self.voxel_m + 0.5)),
        )

    def voxel_to_world(self, key: VoxelKey) -> tuple[float, float, float]:
        vx, vy, vz = key
        return (vx * self.voxel_m, vy * self.voxel_m, vz * self.voxel_m)

    # ── Writing ──────────────────────────────────────────────────────────

    def mark_occupied(self, x: float, y: float, z: float, confidence: float = 1.0) -> None:
        """Record (or reconfirm) a single occupied point."""
        key = self.world_to_voxel(x, y, z)
        existing = self._voxels.get(key)
        if existing is None:
            self._voxels[key] = Voxel(confidence=confidence)
        else:
            # Reconfirmed: refresh timestamp, nudge confidence back up
            # (capped at 1.0) rather than overwrite — repeated hits on
            # the same voxel should make it MORE trusted, not reset it.
            existing.last_seen = time.time()
            existing.hit_count += 1
            existing.confidence = min(1.0, existing.confidence + (1.0 - existing.confidence) * 0.5)

    def update_from_depth(
        self,
        depth: np.ndarray,
        *,
        robot_pos: tuple[float, float],
        robot_yaw: float,
        robot_z: float,
        fov_y_deg: float = 70.0,
        camera_forward_offset_m: float = 0.22,
        camera_up_offset_m: float = 0.06,
        max_range_m: float = 6.0,
        stride: int = 4,
        min_height_m: float = 0.03,
    ) -> None:
        """Project one depth frame into the voxel grid.

        Mirrors SpatialMap.observe_from_depth()'s camera-ray math exactly
        (same FOV projection, same Cadenza forward-is-yaw+pi convention)
        so the two can run side by side during the migration and be
        cross-checked against each other. The difference is what happens
        to each projected 3D point: SpatialMap collapses it into a 2D
        column's max-height; this writes the full (x, y, z) into 3D.

        ``min_height_m`` filters out near-floor noise (depth jitter at
        z≈0 would otherwise fill in a layer of spurious "ground" voxels
        at every height near zero).
        """
        if depth is None or depth.size == 0:
            return

        H, W = depth.shape[:2]
        s = max(1, int(stride))

        yaw_w = float(robot_yaw) + math.pi
        fwd = np.array([math.cos(yaw_w), math.sin(yaw_w), 0.0])
        right = np.array([math.cos(yaw_w - math.pi / 2), math.sin(yaw_w - math.pi / 2), 0.0])
        up = np.array([0.0, 0.0, 1.0])

        cam_xy = np.array([float(robot_pos[0]), float(robot_pos[1]), 0.0]) \
                 + fwd * float(camera_forward_offset_m)
        cam_xy[2] = float(robot_z) + float(camera_up_offset_m)

        fov_y = math.radians(fov_y_deg)
        aspect = W / max(H, 1)
        half_y = math.tan(fov_y / 2.0)
        half_x = half_y * aspect

        for v in range(0, H, s):
            for u in range(0, W, s):
                d = float(depth[v, u])
                if not math.isfinite(d) or d <= 0.05 or d > max_range_m:
                    continue
                ndx = (u / W) * 2.0 - 1.0
                ndy = 1.0 - (v / H) * 2.0
                ray = fwd + right * (ndx * half_x) + up * (ndy * half_y)
                norm = float(np.linalg.norm(ray))
                if norm < 1e-6:
                    continue
                ray /= norm
                world = cam_xy + ray * d

                wx, wy, wz = float(world[0]), float(world[1]), float(world[2])
                if wz < min_height_m:
                    continue

                # Confidence by range: distant points are noisier on
                # real depth sensors, so trust them a little less.
                conf = max(0.3, 1.0 - (d / max_range_m) * 0.5)
                self.mark_occupied(wx, wy, wz, confidence=conf)

    # ── Reading ──────────────────────────────────────────────────────────

    def is_occupied(self, x: float, y: float, z: float, min_confidence: float = 0.3) -> bool:
        key = self.world_to_voxel(x, y, z)
        vox = self._voxels.get(key)
        return vox is not None and vox.confidence >= min_confidence

    def occupied_in_region(
        self, center: tuple[float, float, float], radius_m: float
    ) -> list[tuple[float, float, float]]:
        """All occupied-voxel world coordinates within radius_m of center.
        Linear scan — fine up to ~max_voxels; revisit with a spatial
        index (k-d tree) if this becomes a hot path."""
        cx, cy, cz = center
        r2 = radius_m * radius_m
        hits = []
        for key, vox in self._voxels.items():
            wx, wy, wz = self.voxel_to_world(key)
            if (wx - cx) ** 2 + (wy - cy) ** 2 + (wz - cz) ** 2 <= r2:
                hits.append((wx, wy, wz))
        return hits

    def column_height(self, x: float, y: float, search_radius_m: float = 0.5) -> float | None:
        """Highest occupied z at this (x, y), within a small XY tolerance.
        This is the bridge back to SpatialMap's world-view: lets existing
        2.5D-based reasoning (next_subgoal, stair_on_path) query the
        richer 3D grid without being rewritten yet."""
        vx, vy, _ = self.world_to_voxel(x, y, 0.0)
        tol_vox = max(1, int(round(search_radius_m / self.voxel_m)))
        best_z = None
        for (kx, ky, kz), vox in self._voxels.items():
            if abs(kx - vx) <= tol_vox and abs(ky - vy) <= tol_vox:
                z = kz * self.voxel_m
                if best_z is None or z > best_z:
                    best_z = z
        return best_z

    # ── Maintenance ──────────────────────────────────────────────────────

    def decay(self, dt_sec: float | None = None) -> None:
        """Apply time-based confidence decay to every voxel. Call once
        per tick. If decay_per_sec == 1.0 this is a no-op (matches
        SpatialMap's current permanent-memory behaviour)."""
        if self.decay_per_sec >= 1.0:
            return
        now = time.time()
        for vox in self._voxels.values():
            elapsed = dt_sec if dt_sec is not None else max(0.0, now - vox.last_seen)
            vox.confidence *= self.decay_per_sec ** elapsed

    def prune(self) -> int:
        """Drop voxels below min_confidence, then if still over
        max_voxels, evict the lowest-confidence remainder. Returns the
        number of voxels removed."""
        before = len(self._voxels)
        self._voxels = {
            k: v for k, v in self._voxels.items() if v.confidence >= self.min_confidence
        }
        if len(self._voxels) > self.max_voxels:
            ranked = sorted(self._voxels.items(), key=lambda kv: kv[1].confidence)
            n_to_drop = len(self._voxels) - self.max_voxels
            for key, _ in ranked[:n_to_drop]:
                del self._voxels[key]
        return before - len(self._voxels)

    # ── Introspection ────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._voxels)

    def summary(self) -> dict:
        if not self._voxels:
            return {"n_voxels": 0, "mean_confidence": 0.0, "max_height_m": 0.0}
        confidences = [v.confidence for v in self._voxels.values()]
        heights = [k[2] * self.voxel_m for k in self._voxels.keys()]
        return {
            "n_voxels": len(self._voxels),
            "mean_confidence": float(np.mean(confidences)),
            "max_height_m": float(max(heights)),
        }