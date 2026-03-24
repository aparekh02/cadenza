"""Steering logic — blends VLA raw action with memory-retrieved actions.

Formula:
    final_action = (1 - alpha) * vla_action + alpha * memory_action

where memory_action is a similarity-weighted average of the top-k
nearest demo actions from the MemoryBank.

No model weights are modified — this is purely output-space blending.

Usage:
    final = steer(vla_action, bank, embedding, alpha=0.3, k=5)
"""

from __future__ import annotations

import numpy as np

from cadenza_local.vla_steer.bank import MemoryBank


def steer(
    vla_action: np.ndarray,
    bank: MemoryBank,
    embedding: np.ndarray,
    alpha: float = 0.3,
    k: int = 5,
) -> np.ndarray:
    """Blend VLA output with memory-retrieved actions.

    Args:
        vla_action: raw action from the VLA model, shape (n_act,)
        bank:       populated MemoryBank
        embedding:  VLA encoder embedding for the current obs, shape (dim,)
        alpha:      memory strength in [0, 1].  0 = pure VLA, 1 = pure memory
        k:          number of neighbours to retrieve

    Returns:
        final_action of shape (n_act,), same dtype as vla_action
    """
    vla = np.array(vla_action, dtype=np.float64)

    if len(bank) == 0 or alpha == 0.0:
        return vla.astype(vla_action.dtype)

    neighbors = bank.query(embedding, k=k)
    if not neighbors:
        return vla.astype(vla_action.dtype)

    n_act = len(vla)
    memory_action = _weighted_average(neighbors, n_act)

    blended = (1.0 - alpha) * vla + alpha * memory_action
    return blended.astype(vla_action.dtype)


# ── Internal ──

def _weighted_average(
    neighbors: list[tuple[float, np.ndarray, str]],
    n_act: int,
) -> np.ndarray:
    """Similarity-weighted average of neighbour actions."""
    scores = np.array([s for s, _, _ in neighbors], dtype=np.float64)

    # Shift to [0, 1] in case any cosine similarities are negative
    scores = np.clip(scores, 0.0, None)
    total = scores.sum()
    if total < 1e-8:
        # Uniform fallback
        weights = np.ones(len(neighbors)) / len(neighbors)
    else:
        weights = scores / total

    result = np.zeros(n_act, dtype=np.float64)
    for w, (_, action, _) in zip(weights, neighbors):
        a = np.array(action, dtype=np.float64)
        result += w * a[:n_act]

    return result
