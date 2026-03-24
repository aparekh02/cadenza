"""build_mapmem.py — Offline terrain cluster builder.

Reads experience JSONL logs and clusters STM mean embeddings (k-means)
to produce a MapMem snapshot ready for the controller.

Usage
-----
    python build_mapmem.py --logs logs/go1/ --out mapmem.json --clusters 6

Output
------
    mapmem.json  — list of TerrainCluster dicts
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load_stm_vectors(log_paths: list[Path], window: int = 50) -> list[np.ndarray]:
    """Extract per-episode STM mean embeddings from JSONL logs."""
    from cadenza_local.locomotion.memory.stm import STM, STMFrame

    vectors: list[np.ndarray] = []
    for lp in log_paths:
        stm = STM(window=window)
        with open(lp) as fh:
            for line in fh:
                rec = json.loads(line)
                if rec["type"] != "step":
                    continue
                f = rec["frame"]
                frame = STMFrame(
                    timestamp    = float(rec["t"]),
                    joint_pos    = np.array(f["joint_pos"],    dtype=np.float32),
                    joint_vel    = np.array(f["joint_vel"],    dtype=np.float32),
                    imu_rpy      = np.array(f["imu_rpy"],      dtype=np.float32),
                    imu_omega    = np.array(f["imu_omega"],    dtype=np.float32),
                    foot_contact = np.array(f["foot_contact"], dtype=np.float32),
                    cmd_vel      = np.array(f["cmd_vel"],      dtype=np.float32),
                )
                stm.push(frame)
        if len(stm) > 0:
            vectors.append(stm.mean_embedding())
    return vectors


def build_mapmem(
    log_dir:  Path,
    out_path: Path,
    n_clusters: int = 6,
    window:     int = 50,
) -> None:
    log_paths = sorted(log_dir.glob("*.jsonl"))
    if not log_paths:
        print(f"No .jsonl files found in {log_dir}")
        return

    print(f"Loading {len(log_paths)} logs …")
    vectors = _load_stm_vectors(log_paths, window=window)

    if len(vectors) < n_clusters:
        print(f"  Only {len(vectors)} episodes — reducing clusters to {len(vectors)}")
        n_clusters = max(1, len(vectors))

    X = np.stack(vectors, axis=0).astype(np.float32)
    print(f"  {X.shape[0]} embeddings (dim={X.shape[1]})  →  {n_clusters} clusters")

    # Simple k-means (no sklearn dependency)
    rng = np.random.default_rng(42)
    centroids = X[rng.choice(len(X), size=n_clusters, replace=False)]
    for _ in range(100):
        dists  = np.linalg.norm(X[:, None] - centroids[None, :], axis=2)  # (N, K)
        labels = dists.argmin(axis=1)
        new_c  = np.stack([
            X[labels == k].mean(axis=0) if (labels == k).any() else centroids[k]
            for k in range(n_clusters)
        ])
        if np.allclose(centroids, new_c, atol=1e-6):
            break
        centroids = new_c

    # Default terrain labels and params (edit as needed)
    default_labels = [
        "flat",   "grass",  "gravel",
        "slope",  "stairs", "rough",
    ]
    gait_map = {
        "flat":   ("trot",  0.50, 0.08),
        "grass":  ("trot",  0.40, 0.10),
        "gravel": ("crawl", 0.30, 0.12),
        "slope":  ("crawl", 0.25, 0.10),
        "stairs": ("crawl", 0.15, 0.14),
        "rough":  ("trot",  0.35, 0.12),
    }

    records = []
    for i, c in enumerate(centroids):
        label = default_labels[i] if i < len(default_labels) else f"terrain_{i}"
        gait, speed, step_h = gait_map.get(label, ("trot", 0.50, 0.08))
        records.append({
            "label":       label,
            "centroid":    c.tolist(),
            "speed_limit": speed,
            "step_height": step_h,
            "gait":        gait,
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(records, fh, indent=2)
    print(f"  Saved {len(records)} clusters → {out_path}")


if __name__ == "__main__":
    import sys
    _REPO = Path(__file__).resolve().parents[3]
    if str(_REPO) not in sys.path:
        sys.path.insert(0, str(_REPO))

    parser = argparse.ArgumentParser(description="Build terrain MapMem from logs")
    parser.add_argument("--logs",     type=Path, required=True,  help="log directory (.jsonl)")
    parser.add_argument("--out",      type=Path, default=Path("mapmem.json"))
    parser.add_argument("--clusters", type=int,  default=6)
    parser.add_argument("--window",   type=int,  default=50)
    args = parser.parse_args()
    build_mapmem(args.logs, args.out, args.clusters, args.window)
