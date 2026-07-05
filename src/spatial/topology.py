"""TopologicalGraph — navigable region graph built on top of VoxelGrid.

Sits one abstraction level above the voxel grid: instead of asking
"is voxel (12, 7, 3) occupied?", the planner asks "which regions can
I walk between, and what does it cost?"

Architecture:
    VoxelGrid                  ← raw 3D occupancy (Phase 1)
        ↓  update()
    TopologicalGraph           ← this file
        ↓  plan(start, goal)
    SubgoalChain               ← ordered list of (node, edge_type, nl_label)
        ↓
    VLA context injector       ← feeds nl_label + local_crop per step
    SpatialMemory modality     ← replaces next_subgoal() calls

Design decisions:
    - Nodes are navigable floor clusters, not individual voxels.
      One node ≈ a patch of floor the robot can stand on, represented
      by its centroid (x, y, z).
    - Edges carry: traversal type, cost, and a natural-language label
      that feeds directly to the VLA context injector with no extra
      translation step.
    - A* over cost-weighted edges replaces the two-rule heuristic in
      SpatialMap.next_subgoal(). Cost = distance × terrain_penalty,
      where terrain_penalty encodes energy and stability by edge type.
    - Incremental updates: only voxels near the robot's current
      position are re-clustered on each tick, not the full grid.
      Avoids O(n_voxels) work every frame.

Edge type NL labels (fed verbatim to VLA context injector):
    WALK     → "walk forward"
    CLIMB    → "climb the step"
    DESCEND  → "step down carefully"
    BLOCKED  → triggers replanning, never given to VLA
"""

from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator

import numpy as np

from spatial.voxel_grid import VoxelGrid


# ── Edge types ────────────────────────────────────────────────────────────────

class EdgeType(Enum):
    WALK    = "walk"
    CLIMB   = "climb"
    DESCEND = "descend"
    BLOCKED = "blocked"

    @property
    def nl_label(self) -> str:
        """Natural-language instruction fed directly to the VLA context injector."""
        return {
            EdgeType.WALK:    "walk forward",
            EdgeType.CLIMB:   "climb the step",
            EdgeType.DESCEND: "step down carefully",
            EdgeType.BLOCKED: "",     # never fed to VLA — triggers replan
        }[self]

    @property
    def terrain_penalty(self) -> float:
        """Cost multiplier on top of Euclidean distance.
        BLOCKED = infinity so A* never routes through impassable edges.
        CLIMB > DESCEND > WALK reflects energy expenditure on the Go1/G1."""
        return {
            EdgeType.WALK:    1.0,
            EdgeType.CLIMB:   3.0,
            EdgeType.DESCEND: 1.5,
            EdgeType.BLOCKED: float("inf"),
        }[self]


def classify_edge(dz: float, clearance_ok: bool, *, max_step_up_m: float = 0.22, max_step_down_m: float = 0.22) -> EdgeType:
    """Classify one directed edge from its vertical delta and clearance.

    Args:
        dz: destination_z - source_z in metres. Positive = going up.
        clearance_ok: True if the voxel column between the two nodes
            is free of obstacles at robot-body height. False → BLOCKED.
        max_step_up_m: maximum step height the robot can climb.
        max_step_down_m: maximum step height the robot can descend.
    """
    if not clearance_ok:
        return EdgeType.BLOCKED
    if abs(dz) < 0.03:
        return EdgeType.WALK
    if 0.03 <= dz <= max_step_up_m:
        return EdgeType.CLIMB
    if -max_step_down_m <= dz < -0.03:
        return EdgeType.DESCEND
    return EdgeType.BLOCKED       # too steep in either direction


# ── Core data structures ──────────────────────────────────────────────────────

@dataclass
class Node:
    """One navigable floor region, represented by its centroid."""
    id: int
    x: float
    y: float
    z: float
    last_confirmed: float = field(default_factory=time.monotonic)

    def distance_to(self, other: Node) -> float:
        dx = self.x - other.x
        dy = self.y - other.y
        dz = self.z - other.z
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def xy_distance_to(self, other: Node) -> float:
        dx = self.x - other.x
        dy = self.y - other.y
        return math.sqrt(dx * dx + dy * dy)


@dataclass
class Edge:
    """Directed edge from one node to another."""
    source_id: int
    target_id: int
    edge_type: EdgeType
    cost: float               # Euclidean distance × terrain_penalty
    last_confirmed: float = field(default_factory=time.monotonic)

    @property
    def nl_label(self) -> str:
        return self.edge_type.nl_label

    @property
    def traversable(self) -> bool:
        return self.edge_type != EdgeType.BLOCKED


@dataclass
class SubgoalStep:
    """One step in a planned route."""
    node: Node
    edge: Edge | None         # None for the start node (already here)
    nl_instruction: str       # what to say to the VLA for this step

    def __repr__(self) -> str:
        instr = f'"{self.nl_instruction}"' if self.nl_instruction else "START"
        return f"SubgoalStep(({self.node.x:.1f},{self.node.y:.1f},{self.node.z:.1f}) ← {instr})"


# ── Main class ────────────────────────────────────────────────────────────────

class TopologicalGraph:
    """Navigable region graph built incrementally from a VoxelGrid.

    Args:
        voxel_grid: the Phase-1 VoxelGrid this graph reads from.
        node_spacing_m: minimum XY distance between node centroids.
            Smaller = finer graph, more nodes, slower updates.
        connect_radius_m: maximum XY distance between nodes that may
            share an edge. Should be ≥ node_spacing_m and ≤ ~2×
            so edges only connect truly adjacent regions.
        floor_band_low_m: minimum z to consider as "floor" — filters
            out depth-camera floor noise close to z=0.
        floor_band_high_m: maximum z for a voxel to be classified as
            a potential floor surface (not a wall or ceiling).
        robot_body_height_m: height above a node's z that must be
            clear of obstacles for the node to be navigable.
        update_radius_m: radius around robot position within which
            the graph is re-examined on each update() call.
            Keeps per-tick work bounded regardless of map size.
        max_step_up_m: maximum step height the robot can climb.
        max_step_down_m: maximum step height the robot can descend.
    """

    def __init__(
        self,
        voxel_grid: VoxelGrid,
        *,
        node_spacing_m: float = 0.40,
        connect_radius_m: float = 0.90,
        floor_band_low_m: float = 0.02,
        floor_band_high_m: float = 0.35,
        robot_body_height_m: float = 0.45,
        update_radius_m: float = 3.0,
        max_step_up_m: float = 0.22,
        max_step_down_m: float = 0.22,
    ):
        self.vg = voxel_grid
        self.node_spacing_m = float(node_spacing_m)
        self.connect_radius_m = float(connect_radius_m)
        self.floor_band_low_m = float(floor_band_low_m)
        self.floor_band_high_m = float(floor_band_high_m)
        self.robot_body_height_m = float(robot_body_height_m)
        self.update_radius_m = float(update_radius_m)
        self.max_step_up_m = float(max_step_up_m)
        self.max_step_down_m = float(max_step_down_m)

        self._nodes: dict[int, Node] = {}
        # Edge storage: fast lookup by (src, tgt) key
        self._edges: dict[tuple[int, int], Edge] = {}
        # ADJACENCY LIST: src_id → list of outgoing edges
        # This makes A* O(n_neighbors) per expansion instead of O(n_edges)
        self._adj: dict[int, list[Edge]] = {}
        self._next_node_id: int = 0

        # Active plan — preserved between ticks unless invalidated.
        self._current_plan: list[SubgoalStep] = []
        self._plan_goal: tuple[float, float] | None = None
        # Index pointer into _current_plan instead of mutating the list
        self._plan_index: int = 0

    # ── Public API ────────────────────────────────────────────────────────

    def update(self, robot_pos: tuple[float, float, float]) -> dict:
        """Incrementally refresh nodes and edges near the robot.

        Call once per tick, right after VoxelGrid.update_from_depth().
        Only examines voxels within update_radius_m of the robot —
        O(local voxels), not O(all voxels).

        Returns a summary dict of what changed this tick.
        """
        rx, ry, rz = float(robot_pos[0]), float(robot_pos[1]), float(robot_pos[2])
        n_nodes_before = len(self._nodes)
        n_edges_before = len(self._edges)

        self._discover_nodes(rx, ry, rz)
        self._build_edges(rx, ry)
        self._prune_stale_edges()
        self._invalidate_blocked_plan()

        return {
            "nodes_added": len(self._nodes) - n_nodes_before,
            "edges_added": len(self._edges) - n_edges_before,
            "total_nodes": len(self._nodes),
            "total_edges": len(self._edges),
        }

    def plan(
        self,
        start_xy: tuple[float, float],
        goal_xy: tuple[float, float],
        *,
        replan: bool = False,
    ) -> list[SubgoalStep]:
        """Return an ordered subgoal chain from start to goal.

        Uses A* over cost-weighted edges via adjacency list for O(1)
        neighbor lookup. Returns an empty list if no path exists.
        Re-uses the previous plan if the goal hasn't changed and the
        plan hasn't been invalidated.

        Args:
            start_xy: current robot (x, y) in world metres.
            goal_xy: final target (x, y) in world metres.
            replan: force a fresh A* search even if a valid plan exists.
        """
        same_goal = (
            self._plan_goal is not None
            and math.isclose(self._plan_goal[0], goal_xy[0], abs_tol=0.1)
            and math.isclose(self._plan_goal[1], goal_xy[1], abs_tol=0.1)
        )
        if same_goal and self._current_plan and not replan and self._plan_index < len(self._current_plan):
            return list(self._current_plan)

        start_node = self._nearest_node(start_xy)
        goal_node  = self._nearest_node(goal_xy)

        if start_node is None or goal_node is None:
            self._current_plan = []
            self._plan_index = 0
            self._plan_goal = goal_xy
            return []

        # Only short-circuit if the goal node is genuinely close to the
        # requested goal_xy — if the graph has no node near the goal,
        # _nearest_node() returns the closest node we DO have (which may
        # be the start). Without this distance guard, a disconnected graph
        # would return a one-step "already there" plan instead of [].
        goal_too_far = goal_node.xy_distance_to(
            Node(id=-1, x=goal_xy[0], y=goal_xy[1], z=0)
        ) > self.connect_radius_m * 2

        if goal_too_far:
            self._current_plan = []
            self._plan_index = 0
            self._plan_goal = goal_xy
            return []

        if start_node.id == goal_node.id:
            step = SubgoalStep(node=goal_node, edge=None, nl_instruction="start here")
            self._current_plan = [step]
            self._plan_index = 0
            self._plan_goal = goal_xy
            return list(self._current_plan)

        path_ids = self._astar(start_node.id, goal_node.id)
        if path_ids is None:
            self._current_plan = []
            self._plan_index = 0
            self._plan_goal = goal_xy
            return []

        steps = self._path_to_steps(path_ids)
        self._current_plan = steps
        self._plan_index = 0
        self._plan_goal = goal_xy
        return list(steps)

    def current_subgoal(self, robot_pos: tuple[float, float, float], arrival_radius_m: float = 0.35, z_tolerance_m: float = 0.25) -> SubgoalStep | None:
        """Return the next un-reached step in the active plan.

        Advances the plan pointer when the robot is within
        arrival_radius_m of the current subgoal node's XY and within
        z_tolerance_m of its Z. Returns None if the plan is empty or
        complete.

        Args:
            robot_pos: current robot (x, y, z) in world metres.
            arrival_radius_m: horizontal distance threshold for arrival.
            z_tolerance_m: vertical distance threshold for arrival
                (prevents false arrival when directly above/below).
        """
        rx, ry, rz = float(robot_pos[0]), float(robot_pos[1]), float(robot_pos[2])
        while self._plan_index < len(self._current_plan):
            step = self._current_plan[self._plan_index]
            nx, ny, nz = step.node.x, step.node.y, step.node.z
            h_dist = math.sqrt((rx - nx) ** 2 + (ry - ny) ** 2)
            z_dist = abs(rz - nz)
            if h_dist <= arrival_radius_m and z_dist <= z_tolerance_m:
                self._plan_index += 1   # arrived — advance
            else:
                return step
        return None

    def node_at(self, xy: tuple[float, float], radius_m: float = 0.5) -> Node | None:
        """Return the node nearest to xy if within radius_m."""
        return self._nearest_node(xy, max_dist=radius_m)

    # ── Node discovery ────────────────────────────────────────────────────

    def _discover_nodes(self, rx: float, ry: float, rz: float) -> None:
        """Find new navigable floor regions in the local neighbourhood.

        A voxel is a candidate floor surface if:
          1. Its z is in the floor_band — not ground noise, not a wall.
          2. The column above it (up to robot_body_height_m) is clear.
          3. It isn't too close to an existing node at the same floor level.

        Uses a spatial-hash-like local culling: only checks existing
        nodes within connect_radius_m of the candidate voxel, not all
        nodes in the graph.
        """
        local = self.vg.occupied_in_region(
            center=(rx, ry, rz),
            radius_m=self.update_radius_m,
        )

        for (wx, wy, wz) in local:
            if not (self.floor_band_low_m <= wz <= self.floor_band_high_m):
                continue

            # Clearance check: is the column above this point free?
            body_top = wz + self.robot_body_height_m
            clear = not any(
                self.vg.is_occupied(wx, wy, check_z)
                for check_z in np.arange(wz + 0.05, body_top, 0.05)
            )
            if not clear:
                continue

            # Deduplicate: only check nodes near this candidate (spatial culling).
            # Two nodes at similar XY but different z are valid (floor below
            # vs step surface above) — only skip true spatial duplicates.
            duplicate = False
            for existing in self._nodes.values():
                # Fast reject: skip nodes far in XY
                if abs(existing.x - wx) > self.connect_radius_m or abs(existing.y - wy) > self.connect_radius_m:
                    continue
                dxy = math.sqrt((existing.x - wx) ** 2 + (existing.y - wy) ** 2)
                dz_gap = abs(existing.z - wz)
                if dxy < self.node_spacing_m and dz_gap < 0.05:
                    # Refresh the existing node's timestamp on reconfirmation
                    existing.last_confirmed = time.monotonic()
                    duplicate = True
                    break
            if duplicate:
                continue

            nid = self._next_node_id
            self._next_node_id += 1
            self._nodes[nid] = Node(id=nid, x=wx, y=wy, z=wz)

    # ── Edge construction ─────────────────────────────────────────────────

    def _build_edges(self, rx: float, ry: float) -> None:
        """Connect nearby nodes with classified, costed edges.

        Only processes nodes within update_radius_m of the robot.
        For each candidate pair of nearby nodes, checks clearance
        along the straight-line path between them and classifies
        the edge by height delta.

        Uses adjacency list for O(1) neighbor lookup in A*.
        """
        local_nodes = [
            n for n in self._nodes.values()
            if math.sqrt((n.x - rx) ** 2 + (n.y - ry) ** 2) <= self.update_radius_m
        ]

        for i, src in enumerate(local_nodes):
            for tgt in local_nodes[i + 1:]:
                if src.xy_distance_to(tgt) > self.connect_radius_m:
                    continue

                dz = tgt.z - src.z
                clearance = self._path_clear(src, tgt)
                etype = classify_edge(dz, clearance, max_step_up_m=self.max_step_up_m, max_step_down_m=self.max_step_down_m)
                dist  = src.distance_to(tgt)
                cost  = dist * etype.terrain_penalty

                # Forward edge src → tgt
                fwd_key = (src.id, tgt.id)
                if fwd_key not in self._edges or not self._edges[fwd_key].traversable:
                    edge = Edge(
                        source_id=src.id, target_id=tgt.id,
                        edge_type=etype, cost=cost,
                    )
                    self._edges[fwd_key] = edge
                    self._adj.setdefault(src.id, []).append(edge)

                # Reverse edge tgt → src (may have different type, e.g. DESCEND vs CLIMB)
                rev_dz = -dz
                rev_etype = classify_edge(rev_dz, clearance, max_step_up_m=self.max_step_up_m, max_step_down_m=self.max_step_down_m)
                rev_cost  = dist * rev_etype.terrain_penalty
                rev_key = (tgt.id, src.id)
                if rev_key not in self._edges or not self._edges[rev_key].traversable:
                    edge = Edge(
                        source_id=tgt.id, target_id=src.id,
                        edge_type=rev_etype, cost=rev_cost,
                    )
                    self._edges[rev_key] = edge
                    self._adj.setdefault(tgt.id, []).append(edge)

    def _path_clear(self, src: Node, tgt: Node, n_samples: int = 10) -> bool:
        """Sample n points along the straight line from src to tgt and
        check that no obstacle exists at robot-body height above each.
        Linear interpolation — no raycasting library needed."""
        for t in np.linspace(0.0, 1.0, n_samples):
            mx = src.x + t * (tgt.x - src.x)
            my = src.y + t * (tgt.y - src.y)
            mz = src.z + t * (tgt.z - src.z)
            # Check multiple heights along the robot body at this sample point
            for check_offset in (0.3, 0.45, 0.6):
                check_z = mz + self.robot_body_height_m * check_offset
                if self.vg.is_occupied(mx, my, check_z):
                    return False
        return True

    # ── Edge pruning ──────────────────────────────────────────────────────

    def _prune_stale_edges(self) -> None:
        """Remove edges that are now BLOCKED or whose source/target nodes
        no longer exist. Keeps the adjacency list consistent with _edges."""
        stale_keys = [
            key for key, edge in self._edges.items()
            if not edge.traversable
            or edge.source_id not in self._nodes
            or edge.target_id not in self._nodes
        ]
        for key in stale_keys:
            del self._edges[key]
        # Rebuild adjacency list from remaining edges
        self._adj.clear()
        for edge in self._edges.values():
            self._adj.setdefault(edge.source_id, []).append(edge)

    # ── Plan invalidation ─────────────────────────────────────────────────

    def _invalidate_blocked_plan(self) -> None:
        """Drop the current plan if any of its edges are now BLOCKED.
        Called after every update() so the plan stays consistent with
        the latest obstacle observations."""
        if not self._current_plan:
            return
        for step in self._current_plan:
            if step.edge is None:
                continue
            key = (step.edge.source_id, step.edge.target_id)
            edge = self._edges.get(key)
            if edge is None or not edge.traversable:
                self._current_plan = []
                self._plan_index = 0
                return

    # ── A* ────────────────────────────────────────────────────────────────

    def _astar(self, start_id: int, goal_id: int) -> list[int] | None:
        """A* over the graph using adjacency list, returning node ID path
        or None if unreachable.

        Heuristic: Euclidean distance to goal node (admissible — never
        overestimates, since terrain_penalty >= 1.0 everywhere).

        Complexity: O(E log V) where E = edges, V = nodes. With adjacency
        list, each expansion is O(n_neighbors) instead of O(n_edges).
        """
        goal_node = self._nodes[goal_id]

        def h(nid: int) -> float:
            n = self._nodes[nid]
            return n.distance_to(goal_node)

        open_heap: list[tuple[float, int]] = [(h(start_id), start_id)]
        came_from: dict[int, int] = {}
        g_score: dict[int, float] = {start_id: 0.0}
        visited: set[int] = set()

        while open_heap:
            _, current = heapq.heappop(open_heap)
            if current in visited:
                continue
            visited.add(current)

            if current == goal_id:
                return self._reconstruct_path(came_from, current)

            # ADJACENCY LIST: O(n_neighbors) instead of O(n_edges)
            for edge in self._adj.get(current, []):
                if not edge.traversable:
                    continue
                tgt_id = edge.target_id
                tentative_g = g_score.get(current, float("inf")) + edge.cost
                if tentative_g < g_score.get(tgt_id, float("inf")):
                    g_score[tgt_id] = tentative_g
                    came_from[tgt_id] = current
                    f = tentative_g + h(tgt_id)
                    heapq.heappush(open_heap, (f, tgt_id))

        return None  # no path

    @staticmethod
    def _reconstruct_path(came_from: dict[int, int], current: int) -> list[int]:
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def _path_to_steps(self, path_ids: list[int]) -> list[SubgoalStep]:
        """Convert an ID path into SubgoalStep objects with NL instructions."""
        steps: list[SubgoalStep] = []
        for i, nid in enumerate(path_ids):
            node = self._nodes[nid]
            if i == 0:
                steps.append(SubgoalStep(node=node, edge=None, nl_instruction="start here"))
                continue
            prev_id = path_ids[i - 1]
            edge = self._edges.get((prev_id, nid))
            nl = edge.nl_label if edge else "walk forward"
            steps.append(SubgoalStep(node=node, edge=edge, nl_instruction=nl))
        return steps

    # ── Nearest-node lookup ───────────────────────────────────────────────

    def _nearest_node(
        self, xy: tuple[float, float], max_dist: float = float("inf")
    ) -> Node | None:
        bx, by = float(xy[0]), float(xy[1])
        best: Node | None = None
        best_d = max_dist
        for node in self._nodes.values():
            d = math.sqrt((node.x - bx) ** 2 + (node.y - by) ** 2)
            if d < best_d:
                best_d = d
                best = node
        return best

    # ── Introspection ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._nodes)

    def summary(self) -> dict:
        traversable = sum(1 for e in self._edges.values() if e.traversable)
        blocked     = len(self._edges) - traversable
        type_counts: dict[str, int] = {}
        for e in self._edges.values():
            label = e.edge_type.value
            type_counts[label] = type_counts.get(label, 0) + 1
        return {
            "n_nodes": len(self._nodes),
            "n_edges": len(self._edges),
            "traversable_edges": traversable,
            "blocked_edges": blocked,
            "edge_types": type_counts,
            "plan_length": len(self._current_plan),
            "plan_index": self._plan_index,
        }

    def iter_nodes(self) -> Iterator[Node]:
        return iter(self._nodes.values())

    def iter_edges(self) -> Iterator[Edge]:
        return iter(self._edges.values())
