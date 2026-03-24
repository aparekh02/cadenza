"""MapMem — Terrain map memory.

Clusters of previously observed terrain types, each associated with
recommended locomotion parameters (speed, step height, gait).

Loaded read-only from snapshot.json at runtime.
Built offline by examples/unitree_go1/backend/build_mapmem.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class TerrainCluster:
    """One terrain cluster from the memory snapshot.

    Fields
    ------
    label         : str    human name (e.g. "grass", "stairs", "slope")
    centroid      : (D,)   mean embedding of training observations
    speed_limit   : float  recommended max forward speed (m/s)
    step_height   : float  recommended foot clearance (m)
    gait          : str    preferred gait ("trot", "crawl", "bound", ...)
    extra         : dict   any additional recommendations
    """

    label:       str
    centroid:    np.ndarray
    speed_limit: float = 0.5
    step_height: float = 0.08
    gait:        str   = "trot"
    extra:       dict  = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TerrainCluster":
        return cls(
            label       = str(d["label"]),
            centroid    = np.array(d["centroid"], dtype=np.float32),
            speed_limit = float(d.get("speed_limit", 0.5)),
            step_height = float(d.get("step_height", 0.08)),
            gait        = str(d.get("gait", "trot")),
            extra       = {k: v for k, v in d.items()
                           if k not in ("label", "centroid", "speed_limit",
                                        "step_height", "gait")},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "label":       self.label,
            "centroid":    self.centroid.tolist(),
            "speed_limit": self.speed_limit,
            "step_height": self.step_height,
            "gait":        self.gait,
            **self.extra,
        }


def stm_to_embedding(stm_mean: np.ndarray) -> np.ndarray:
    """Map a raw STM mean vector to a terrain embedding.

    Currently the identity (same vector, L2-normalised).
    Replace with a learned encoder if available.
    """
    v = stm_mean.astype(np.float32)
    n = float(np.linalg.norm(v))
    if n > 1e-8:
        v = v / n
    return v


class MapMem:
    """Nearest-neighbour terrain lookup.

    Args:
        clusters: list of TerrainCluster objects (read-only)
    """

    def __init__(self, clusters: list[TerrainCluster] | None = None):
        self._clusters: list[TerrainCluster] = clusters or []
        self._centroids: np.ndarray | None = None
        if self._clusters:
            self._build_index()

    def _build_index(self) -> None:
        self._centroids = np.stack(
            [c.centroid for c in self._clusters], axis=0
        ).astype(np.float32)

    # ── Query ──────────────────────────────────────────────────────────────

    def recommend(self, terrain_emb: np.ndarray) -> TerrainCluster | None:
        """Return the closest terrain cluster or None if empty.

        Args:
            terrain_emb: (D,) float32 from stm_to_embedding()
        """
        if self._centroids is None or len(self._clusters) == 0:
            return None

        emb = terrain_emb.astype(np.float32)
        dists = np.linalg.norm(self._centroids - emb[None, :], axis=1)
        return self._clusters[int(np.argmin(dists))]

    # ── Persistence ────────────────────────────────────────────────────────

    @classmethod
    def from_list(cls, records: list[dict]) -> "MapMem":
        clusters = [TerrainCluster.from_dict(r) for r in records]
        return cls(clusters)

    def to_list(self) -> list[dict]:
        return [c.to_dict() for c in self._clusters]

    # ── Dunder ─────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._clusters)

    def __repr__(self) -> str:
        return f"MapMem(clusters={len(self._clusters)})"
