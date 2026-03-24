"""build_skillmem.py — Offline motion skill builder.

Reads experience JSONL logs and groups successful episodes by goal
embedding similarity to produce a SkillMem snapshot.

Each cluster becomes one Skill whose waypoints are the per-step joint
positions from the centroid episode.

Usage
-----
    python build_skillmem.py --logs logs/go1/ --out skillmem.json --skills 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load_episodes(log_paths: list[Path]) -> list[dict]:
    """Parse JSONL logs into episodes (only successful ones)."""
    episodes: list[dict] = []
    for lp in log_paths:
        steps: list[dict] = []
        with open(lp) as fh:
            for line in fh:
                rec = json.loads(line)
                if rec["type"] == "step":
                    steps.append(rec)
                elif rec["type"] == "episode_outcome":
                    if rec.get("success") and steps:
                        episodes.append({"steps": steps, "outcome": rec})
                    steps = []
    return episodes


def build_skillmem(
    log_dir:   Path,
    out_path:  Path,
    n_skills:  int = 8,
) -> None:
    from cadenza_local.locomotion.memory.skillmem import goal_to_embedding

    log_paths = sorted(log_dir.glob("*.jsonl"))
    if not log_paths:
        print(f"No .jsonl files found in {log_dir}")
        return

    print(f"Loading {len(log_paths)} logs …")
    episodes = _load_episodes(log_paths)
    print(f"  {len(episodes)} successful episodes found")

    if not episodes:
        print("  No successful episodes — cannot build SkillMem")
        return

    # Compute goal embedding for each episode (mean of cmd_vel across steps)
    goal_embs: list[np.ndarray] = []
    for ep in episodes:
        vels = [np.array(s["frame"]["cmd_vel"], dtype=np.float32) for s in ep["steps"]]
        mean_vel = np.mean(vels, axis=0)
        goal_embs.append(goal_to_embedding(mean_vel))

    G = np.stack(goal_embs, axis=0)   # (N, D)

    n_skills = min(n_skills, len(episodes))
    print(f"  Clustering {len(episodes)} episodes → {n_skills} skills")

    # Simple k-means
    rng       = np.random.default_rng(0)
    centroids = G[rng.choice(len(G), size=n_skills, replace=False)]
    labels    = np.zeros(len(G), dtype=int)
    for _ in range(100):
        dists  = np.linalg.norm(G[:, None] - centroids[None, :], axis=2)
        labels = dists.argmin(axis=1)
        new_c  = np.stack([
            G[labels == k].mean(axis=0) if (labels == k).any() else centroids[k]
            for k in range(n_skills)
        ])
        if np.allclose(centroids, new_c, atol=1e-6):
            break
        centroids = new_c

    skill_names = [
        "trot_fwd", "trot_side", "trot_turn", "trot_slow",
        "crawl_fwd", "crawl_slope", "bound_fwd", "pivot_turn",
    ]

    records = []
    for k in range(n_skills):
        members = [i for i, l in enumerate(labels) if l == k]
        if not members:
            continue
        # Pick centroid episode (closest to cluster mean)
        dists_k = np.linalg.norm(G[members] - centroids[k][None, :], axis=1)
        rep_idx  = members[int(np.argmin(dists_k))]
        ep       = episodes[rep_idx]
        # Waypoints: joint_pos for each step
        waypoints = [
            np.array(s["frame"]["joint_pos"], dtype=np.float32).tolist()
            for s in ep["steps"]
        ]
        name = skill_names[k] if k < len(skill_names) else f"skill_{k}"
        records.append({
            "name":      name,
            "goal_emb":  centroids[k].tolist(),
            "waypoints": waypoints,
            "tags":      [f"cluster_{k}", f"n_members_{len(members)}"],
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(records, fh, indent=2)
    print(f"  Saved {len(records)} skills → {out_path}")


if __name__ == "__main__":
    import sys
    _REPO = Path(__file__).resolve().parents[3]
    if str(_REPO) not in sys.path:
        sys.path.insert(0, str(_REPO))

    parser = argparse.ArgumentParser(description="Build motion SkillMem from logs")
    parser.add_argument("--logs",   type=Path, required=True, help="log directory (.jsonl)")
    parser.add_argument("--out",    type=Path, default=Path("skillmem.json"))
    parser.add_argument("--skills", type=int,  default=8)
    args = parser.parse_args()
    build_skillmem(args.logs, args.out, args.skills)
