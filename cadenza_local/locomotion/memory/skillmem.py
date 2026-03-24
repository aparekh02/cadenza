"""SkillMem — Motion skill library memory.

Each skill is a named library of waypoints (joint positions over time)
paired with a goal embedding. At runtime the controller retrieves the
best-matching skill for the current goal and blends its waypoints into
the action.

Loaded read-only from snapshot.json at runtime.
Built offline by examples/unitree_go1/backend/build_skillmem.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Skill:
    """One motion skill.

    Fields
    ------
    name        : str          human name (e.g. "trot_forward", "stair_climb")
    goal_emb    : (D,) float32 normalised goal embedding this skill was trained for
    waypoints   : (T, N) float32  joint-position trajectory (T steps, N joints)
    tags        : list[str]    optional descriptive tags (terrain, gait, ...)
    extra       : dict         any additional metadata
    """

    name:      str
    goal_emb:  np.ndarray
    waypoints: np.ndarray           # (T, N) float32
    tags:      list[str] = field(default_factory=list)
    extra:     dict      = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Skill":
        return cls(
            name      = str(d["name"]),
            goal_emb  = np.array(d["goal_emb"], dtype=np.float32),
            waypoints = np.array(d["waypoints"], dtype=np.float32),
            tags      = list(d.get("tags", [])),
            extra     = {k: v for k, v in d.items()
                         if k not in ("name", "goal_emb", "waypoints", "tags")},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name":      self.name,
            "goal_emb":  self.goal_emb.tolist(),
            "waypoints": self.waypoints.tolist(),
            "tags":      self.tags,
            **self.extra,
        }


def goal_to_embedding(cmd_vel: np.ndarray, task_text: str = "") -> np.ndarray:
    """Map a (vx, vy, yaw_rate) command + optional text to a 3-dim goal embedding.

    Args:
        cmd_vel:   (3,) float32  vx, vy, yaw_rate
        task_text: free-text task description (ignored by default)
    """
    v = np.asarray(cmd_vel, dtype=np.float32).copy()
    n = float(np.linalg.norm(v))
    if n > 1e-8:
        v = v / n
    return v


# ── Phase-aware 4-dim embedding (used by motor primitives) ────────────────────

_STEP_HEIGHT_BASELINE: float = 0.10   # m — centre of step_height encoding axis
_STEP_HEIGHT_SCALE:    float = 8.0    # scale so ±0.06m = ±0.48 deviation units


def goal_to_embedding_4d(cmd_vel: np.ndarray, step_height: float) -> np.ndarray:
    """4-dim goal embedding encoding velocity direction AND step height.

    Dim 0-2: L2-normalised velocity direction.
    Dim 3:   (step_height - _STEP_HEIGHT_BASELINE) * _STEP_HEIGHT_SCALE
             — signed deviation from baseline; negative = low step, positive = high.

    The final 4-vector is re-normalised to the unit sphere so that
    inner-product == cosine similarity.

    Stored embeddings for seeded motor primitives:
      trot_forward  (step_h=0.08) → [0.987,  0, 0, -0.158]
      trot_hurdle   (step_h=0.15) → [0.928,  0, 0, +0.371]
    Cosine similarity between them ≈ 0.857 — different enough for argmax to
    distinguish, but both above the min_similarity=0.50 threshold.
    """
    v = np.asarray(cmd_vel, dtype=np.float32).copy()
    n = float(np.linalg.norm(v))
    if n > 1e-8:
        v = v / n                               # normalised direction (3-dim)
    step_dev = (float(step_height) - _STEP_HEIGHT_BASELINE) * _STEP_HEIGHT_SCALE
    v4 = np.array([v[0], v[1], v[2], step_dev], dtype=np.float32)
    n4 = float(np.linalg.norm(v4))
    if n4 > 1e-8:
        v4 = v4 / n4
    return v4


class SkillMem:
    """Cosine-similarity skill retrieval.

    Args:
        skills: list of Skill objects (read-only)
    """

    def __init__(self, skills: list[Skill] | None = None):
        self._skills: list[Skill] = skills or []
        self._embs: np.ndarray | None = None
        if self._skills:
            self._build_index()

    def _build_index(self) -> None:
        self._embs = np.stack(
            [s.goal_emb for s in self._skills], axis=0
        ).astype(np.float32)   # (K, D)

    # ── Query ──────────────────────────────────────────────────────────────

    def best_skill(
        self,
        goal_emb:       np.ndarray,
        min_similarity: float = 0.50,
    ) -> "Skill | None":
        """Return the skill whose goal embedding is closest (cosine) to goal_emb.

        Returns None if the best cosine similarity is below min_similarity,
        ensuring an irrelevant skill is never blended into the gait engine.

        Args:
            goal_emb:       (D,) float32 from goal_to_embedding() or goal_to_embedding_4d()
            min_similarity: cosine threshold; default 0.50
        """
        if self._embs is None or len(self._skills) == 0:
            return None

        emb = goal_emb.astype(np.float32)
        # Dimension mismatch guard: stored embeddings may be 3-dim or 4-dim.
        # Pad/truncate query to match stored embedding dimension.
        stored_dim = self._embs.shape[1]
        if len(emb) != stored_dim:
            if len(emb) < stored_dim:
                emb = np.pad(emb, (0, stored_dim - len(emb)))
            else:
                emb = emb[:stored_dim]
            # Re-normalise after dimension adjustment
            n = float(np.linalg.norm(emb))
            if n > 1e-8:
                emb = emb / n

        sims     = self._embs @ emb             # (K,) cosine similarities
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        if best_sim < min_similarity:
            return None                         # no good match — pure gait engine

        return self._skills[best_idx]

    # ── Persistence ────────────────────────────────────────────────────────

    @classmethod
    def from_list(cls, records: list[dict]) -> "SkillMem":
        skills = [Skill.from_dict(r) for r in records]
        return cls(skills)

    def to_list(self) -> list[dict]:
        return [s.to_dict() for s in self._skills]

    # ── Dunder ─────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._skills)

    def __repr__(self) -> str:
        return f"SkillMem(skills={len(self._skills)})"
