"""MemoryBank — FAISS-backed store of demo embeddings + actions.

Stores (embedding, action, task_label) triples from successful demos.
All embeddings are L2-normalised so inner-product == cosine similarity.

Usage:
    bank = MemoryBank(dim=512)
    bank.add(emb, action, task="pick cup")
    neighbors = bank.query(emb, k=5)
    bank.save("bank.npz"); bank = MemoryBank.load("bank.npz")
"""

from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np


class MemoryBank:
    """FAISS flat index keyed by VLA embeddings.

    Args:
        dim: embedding dimension (must match the VLA encoder output)
    """

    def __init__(self, dim: int):
        self.dim = dim
        self._index = faiss.IndexFlatIP(dim)   # cosine via normalised L2
        self._vectors: list[np.ndarray] = []   # stored for serialisation
        self._actions: list[np.ndarray] = []
        self._tasks: list[str] = []

    # ── Write ──

    def add(self, embedding: np.ndarray, action: np.ndarray, task: str = "") -> None:
        """Store one demo.

        Args:
            embedding: VLA encoder output, shape (dim,) or (1, dim)
            action:    robot action vector, shape (n_act,)
            task:      free-text label (instruction string)
        """
        emb = _norm(embedding)
        self._index.add(emb)
        self._vectors.append(emb[0].copy())
        self._actions.append(np.array(action, dtype=np.float32))
        self._tasks.append(task)

    # ── Read ──

    def query(self, embedding: np.ndarray, k: int = 5) -> list[tuple[float, np.ndarray, str]]:
        """Return top-k similar demos.

        Returns:
            List of (cosine_similarity, action, task_label), highest-first.
            Empty list when bank is empty.
        """
        if len(self) == 0:
            return []

        emb = _norm(embedding)
        k = min(k, len(self))
        scores, indices = self._index.search(emb, k)

        return [
            (float(scores[0, i]), self._actions[indices[0, i]], self._tasks[indices[0, i]])
            for i in range(k)
            if indices[0, i] >= 0
        ]

    # ── Persistence ──

    def save(self, path: str | Path) -> None:
        """Save index + payloads to a .npz file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            dim=np.array(self.dim),
            vectors=np.array(self._vectors, dtype=np.float32),
            actions=np.array(self._actions, dtype=object),
            tasks=np.array(self._tasks),
        )

    @classmethod
    def load(cls, path: str | Path) -> "MemoryBank":
        """Load a saved bank."""
        data = np.load(Path(path), allow_pickle=True)
        dim = int(data["dim"])
        bank = cls(dim)
        vectors = data["vectors"].astype(np.float32)
        actions = data["actions"]
        tasks = data["tasks"]
        for emb, act, task in zip(vectors, actions, tasks):
            emb = emb.reshape(1, -1)
            bank._index.add(emb)
            bank._vectors.append(emb[0].copy())
            bank._actions.append(np.array(act, dtype=np.float32))
            bank._tasks.append(str(task))
        return bank

    def __len__(self) -> int:
        return self._index.ntotal

    def __repr__(self) -> str:
        return f"MemoryBank(dim={self.dim}, n={len(self)})"


# ── Internal ──

def _norm(embedding: np.ndarray) -> np.ndarray:
    """Reshape to (1, dim) and L2-normalise in place."""
    emb = np.array(embedding, dtype=np.float32).reshape(1, -1)
    faiss.normalize_L2(emb)
    return emb
