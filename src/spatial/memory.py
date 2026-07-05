"""SpatialMemory — Modality wrapping SpatialMap + VoxelGrid + TopologicalGraph.

Built on top of the existing modality plug-in interface so it composes with
``DepthAnythingV2Small`` and anything else the client adds:

    sense=[
        SpatialMemory(target=(-5.5, 0.0)),
        DepthAnythingV2Small(),
    ]

Architecture (what runs every tick):
    ┌─────────────────────────────────────────────────────────────┐
    │  depth frame + robot pose                                   │
    │       ↓                  ↓                                  │
    │  SpatialMap         VoxelGrid          ← both update        │
    │  (2.5D, existing)   (true 3D, Phase 1)                      │
    │       ↓                  ↓                                  │
    │  landmarks list    TopologicalGraph    ← graph builds        │
    │       ↓                  ↓                                  │
    │  next_subgoal()    topo.plan()         ← planning           │
    │       ↓ fallback         ↓ primary (once graph has nodes)   │
    │             subgoal + nl_instruction                        │
    │                     ↓                                       │
    │             ModalityResult.keys → downstream / VLA          │
    └─────────────────────────────────────────────────────────────┘

Fallback contract:
    If TopologicalGraph has fewer than ``min_topo_nodes`` nodes (graph
    not yet built — e.g. first few ticks after startup), the modality
    falls back to SpatialMap.next_subgoal() exactly as before. This
    means the existing behaviour is 100% preserved on startup and the
    new system activates progressively as the graph fills in.

New keys in ModalityResult:
    ``nl_instruction``      — NL label for the current subgoal edge,
                              e.g. "climb the step" / "walk forward".
                              Empty string on fallback or at start node.
                              Fed directly to the VLA context injector.
    ``subgoal_chain``       — full ordered list of (x, y, z, nl) tuples
                              for the current plan. Lets the VLA see the
                              whole route ahead, not just the next step.
    ``topo_active``         — bool: True when topology is driving,
                              False when falling back to SpatialMap.
    ``topo_summary``        — TopologicalGraph.summary() dict for
                              debugging and telemetry.
"""

from __future__ import annotations

from cadenza.spatial.map import SpatialMap
from cadenza.spatial.voxel_grid import VoxelGrid
from cadenza.spatial.topology import TopologicalGraph
from cadenza.stack.modalities.base import (
    Modality,
    ModalityResult,
    register_modality,
)


@register_modality
class SpatialMemory(Modality):
    """3D spatial memory built up from multi-modal observations."""

    name = "spatial_memory"
    description = (
        "On-board spatial memory: 2.5D SpatialMap (existing) + true 3D "
        "VoxelGrid + TopologicalGraph with A* subgoal chains and NL edge "
        "labels for VLA context injection."
    )

    DEFAULT_STL_NAME = "cadenza_memory.stl"

    def __init__(
        self,
        target: tuple[float, float],
        *,
        # ── existing SpatialMap params (unchanged) ──────────────────────
        cell_m: float = 0.20,
        extent_m: float = 12.0,
        elevation_threshold: float = 0.15,
        stl_path: "str | None" = "AUTO",
        stl_every_ticks: int = 5,
        # ── VoxelGrid params (Phase 1) ──────────────────────────────────
        voxel_m: float = 0.05,
        voxel_decay_per_sec: float = 1.0,
        voxel_max_cells: int = 200_000,
        # ── TopologicalGraph params (Phase 2) ───────────────────────────
        topo_node_spacing_m: float = 0.40,
        topo_connect_radius_m: float = 0.90,
        topo_update_radius_m: float = 3.0,
        min_topo_nodes: int = 3,
    ):
        """
        Args:
            target: the user-supplied final target (x, y). The modality
                may rewrite ``observation['target_xy']`` to a subgoal
                when geometry demands it — the original is always kept
                as ``observation['final_target_xy']``.
            cell_m: SpatialMap grid resolution (unchanged).
            extent_m: SpatialMap world extent (unchanged).
            elevation_threshold: heights above this count as elevated
                (unchanged).
            stl_path: live 3D map output path. "AUTO" = cadenza_memory.stl
                in cwd. None = disabled (unchanged).
            stl_every_ticks: how often to rewrite the live STL (unchanged).
            voxel_m: VoxelGrid voxel edge length in metres. 0.05 (5cm)
                is the minimum to detect a 15cm stair step (one voxel =
                5cm, step spans 3 voxels). Smaller = finer, more memory.
            voxel_decay_per_sec: confidence multiplier per second for
                voxels not reconfirmed. 1.0 = permanent (default,
                matches existing SpatialMap behaviour).
            voxel_max_cells: hard cap on voxel count. Prune evicts
                lowest-confidence voxels when exceeded.
            topo_node_spacing_m: minimum XY distance between graph nodes.
            topo_connect_radius_m: maximum XY distance for edge creation.
            topo_update_radius_m: radius around robot updated each tick.
            min_topo_nodes: minimum nodes required before topology drives
                planning. Below this, falls back to SpatialMap.next_subgoal().
        """
        from pathlib import Path as _Path

        self.final_target = (float(target[0]), float(target[1]))
        self.elevation_threshold = float(elevation_threshold)
        self.min_topo_nodes = int(min_topo_nodes)

        # ── Existing 2.5D system (unchanged) ───────────────────────────
        self.map = SpatialMap(cell_m=cell_m, extent_m=extent_m)

        # ── Phase 1: true 3D voxel grid ────────────────────────────────
        self.voxel_grid = VoxelGrid(
            voxel_m=voxel_m,
            decay_per_sec=voxel_decay_per_sec,
            max_voxels=voxel_max_cells,
        )

        # ── Phase 2: topological graph + A* planner ─────────────────────
        self.topo = TopologicalGraph(
            self.voxel_grid,
            node_spacing_m=topo_node_spacing_m,
            connect_radius_m=topo_connect_radius_m,
            update_radius_m=topo_update_radius_m,
        )

        # ── Stream-rate-limit state (unchanged from original) ───────────
        self._prev_n_landmarks = 0
        self._prev_subgoal_reason: str | None = None

        # ── Live STL output (unchanged from original) ───────────────────
        if stl_path == "AUTO":
            self.stl_path: "_Path | None" = _Path(self.DEFAULT_STL_NAME).resolve()
        elif stl_path is None:
            self.stl_path = None
        else:
            self.stl_path = _Path(stl_path).expanduser().resolve()
        self.stl_every_ticks = max(1, int(stl_every_ticks))
        self._tick_counter = 0

        if self.stl_path is not None and self.stl_path.exists():
            try:
                self.stl_path.unlink()
            except Exception:
                pass

        # ── Prune voxel grid every N ticks to bound memory ─────────────
        self._prune_every_ticks = 30

    def setup(self) -> None:
        if self.stl_path is not None:
            print(f"  [SpatialMemory] live 3D map → {self.stl_path}")
            print(f"                  (rewritten every {self.stl_every_ticks} ticks)")
        print(f"  [SpatialMemory] VoxelGrid voxel_m={self.voxel_grid.voxel_m}m  "
              f"max_voxels={self.voxel_grid.max_voxels}")
        print(f"  [SpatialMemory] TopologicalGraph active after {self.min_topo_nodes} nodes")

    def teardown(self) -> None:
        return

    def compute(self, observation) -> ModalityResult:
        pos       = observation.pos
        rpy       = observation.rpy
        body_h    = float(observation.body_height)
        robot_xy  = (float(pos[0]), float(pos[1]))
        robot_xyz = (float(pos[0]), float(pos[1]), body_h)

        # ── 1. Always record trajectory (unchanged) ─────────────────────
        self.map.record_pose(float(pos[0]), float(pos[1]), body_h)

        # ── 2. Update both maps from the same depth frame ───────────────
        depth = getattr(observation, "depth", None)
        if depth is not None:
            # Existing 2.5D map — untouched
            self.map.observe_from_depth(
                depth,
                robot_pos=robot_xy,
                robot_yaw=float(rpy[2]),
                robot_z=body_h,
            )
            # New true-3D voxel grid
            self.voxel_grid.update_from_depth(
                depth,
                robot_pos=robot_xy,
                robot_yaw=float(rpy[2]),
                robot_z=body_h,
            )

        # ── 3. Decay + prune voxel grid periodically ────────────────────
        self._tick_counter += 1
        if self._tick_counter % self._prune_every_ticks == 0:
            self.voxel_grid.decay()
            self.voxel_grid.prune()

        # ── 4. Update topology graph ────────────────────────────────────
        self.topo.update(robot_xyz)

        # ── 5. Plan: topology if ready, else fall back to SpatialMap ────
        topo_active = len(self.topo) >= self.min_topo_nodes

        if topo_active:
            subgoal, nl_instruction, subgoal_reason, subgoal_chain = \
                self._plan_from_topology(robot_xy)
        else:
            subgoal, nl_instruction, subgoal_reason, subgoal_chain = \
                self._plan_from_spatial_map(robot_xy, body_h)

        # ── 6. Change-only summary (unchanged logic, extended content) ───
        summary = self._build_summary(subgoal_reason, topo_active)

        # ── 7. Live STL (unchanged) ──────────────────────────────────────
        if self.stl_path is not None and self._tick_counter % self.stl_every_ticks == 0:
            try:
                import os
                tmp = self.stl_path.with_suffix(self.stl_path.suffix + ".tmp")
                self.map.to_stl(save_path=tmp, target=self.final_target)
                os.replace(tmp, self.stl_path)
            except Exception:
                pass

        return ModalityResult(
            keys={
                # ── existing keys (unchanged contracts) ─────────────────
                "target_xy":           subgoal,
                "final_target_xy":     self.final_target,
                "subgoal_reason":      subgoal_reason,
                "spatial_map_summary": self.map.summary(),
                "landmarks":           [(lm.kind, lm.xy, lm.z)
                                        for lm in self.map.landmarks],
                # ── new keys (Phase 1 + 2) ───────────────────────────────
                "nl_instruction":      nl_instruction,
                "subgoal_chain":       subgoal_chain,
                "topo_active":         topo_active,
                "topo_summary":        self.topo.summary(),
            },
            summary=summary,
        )

    # ── Planning helpers ──────────────────────────────────────────────────

    def _plan_from_topology(
        self, robot_xy: tuple[float, float]
    ) -> tuple[
        tuple[float, float],   # subgoal xy
        str,                   # nl_instruction for VLA
        str,                   # subgoal_reason string
        list,                  # subgoal_chain [(x, y, z, nl), ...]
    ]:
        """Drive planning from the topological graph (primary path)."""
        steps = self.topo.plan(
            start_xy=robot_xy,
            goal_xy=self.final_target,
        )

        if not steps:
            # Graph built but no path found — fall back gracefully
            subgoal, reason = self.map.next_subgoal(
                target_xy=self.final_target,
                robot_xy=robot_xy,
                robot_z=0.0,
                elevation_threshold=self.elevation_threshold,
            )
            return subgoal, "", f"topo_no_path:{reason}", []

        # Advance the plan pointer past already-reached nodes
        current = self.topo.current_subgoal(robot_xy)
        if current is None:
            # All subgoals reached — at the goal
            return self.final_target, "", "topo_goal_reached", []

        subgoal_xy = (current.node.x, current.node.y)
        nl_instruction = current.nl_instruction

        # Build the full chain for downstream consumers (e.g. VLA context)
        subgoal_chain = [
            (s.node.x, s.node.y, s.node.z, s.nl_instruction)
            for s in steps
        ]

        reason = f"topo:{current.edge.edge_type.value}" if current.edge else "topo:start"
        return subgoal_xy, nl_instruction, reason, subgoal_chain

    def _plan_from_spatial_map(
        self,
        robot_xy: tuple[float, float],
        body_h: float,
    ) -> tuple[
        tuple[float, float],
        str,
        str,
        list,
    ]:
        """Fallback: use existing SpatialMap.next_subgoal() (unchanged behaviour)."""
        subgoal, reason = self.map.next_subgoal(
            target_xy=self.final_target,
            robot_xy=robot_xy,
            robot_z=body_h,
            elevation_threshold=self.elevation_threshold,
        )
        return subgoal, "", reason, []

    # ── Summary builder ───────────────────────────────────────────────────

    def _build_summary(self, subgoal_reason: str, topo_active: bool) -> str:
        """Emit a log line only when something meaningful changes.
        Keeps the stream quiet on steady-state ticks (unchanged philosophy)."""
        summary_parts = []

        n_landmarks = len(self.map.landmarks)
        if n_landmarks > self._prev_n_landmarks:
            new = self.map.landmarks[self._prev_n_landmarks:]
            kinds = ", ".join(
                f"{lm.kind} at ({lm.xy[0]:.1f}, {lm.xy[1]:.1f})" for lm in new
            )
            summary_parts.append(f"map updated: {kinds}")

        if (subgoal_reason != self._prev_subgoal_reason
                and self._prev_subgoal_reason is not None):
            driver = "topo" if topo_active else "spatial_map"
            summary_parts.append(
                f"[{driver}] subgoal: {self._prev_subgoal_reason} → {subgoal_reason}"
            )

        self._prev_n_landmarks = n_landmarks
        self._prev_subgoal_reason = subgoal_reason

        return "  |  ".join(summary_parts)