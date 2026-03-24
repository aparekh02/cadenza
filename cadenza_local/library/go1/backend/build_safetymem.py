"""build_safetymem.py — Offline safety rule builder.

Reads experience JSONL logs, finds failed episodes, and extracts sensor
conditions that preceded each failure.  Also accepts hand-authored rules
from a YAML file (merged with auto-discovered rules).

Usage
-----
    # Auto-discover from logs + merge hand-authored rules
    python build_safetymem.py --logs logs/go1/ --rules rules.yaml --out safetymem.json

    # Hand-authored only (no log analysis)
    python build_safetymem.py --rules rules.yaml --out safetymem.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


_DEFAULT_RULES = [
    # Pitch safety: stop if robot pitches beyond ±35° (0.61 rad)
    {
        "name":     "pitch_forward_limit",
        "field":    "imu_rpy",
        "axis":     1,           # pitch
        "max_val":  0.61,
        "override": {"cmd_vx": 0.0, "cmd_vy": 0.0},
        "priority": 10,
    },
    {
        "name":     "pitch_backward_limit",
        "field":    "imu_rpy",
        "axis":     1,
        "min_val":  -0.61,
        "override": {"cmd_vx": 0.0, "cmd_vy": 0.0},
        "priority": 10,
    },
    # Roll safety: stop if robot rolls beyond ±30° (0.52 rad)
    {
        "name":     "roll_limit",
        "field":    "imu_rpy",
        "axis":     0,           # roll
        "min_val":  -0.52,
        "max_val":   0.52,
        "override": {"cmd_vx": 0.0, "cmd_vy": 0.0},
        "priority": 10,
    },
]


def _auto_rules_from_logs(log_paths: list[Path]) -> list[dict]:
    """Heuristically extract safety rules from failure episodes."""
    # Collect sensor snapshots from the last N steps before each failure
    pre_fail: list[dict] = []
    lookback = 5

    for lp in log_paths:
        steps: list[dict] = []
        with open(lp) as fh:
            for line in fh:
                rec = json.loads(line)
                if rec["type"] == "step":
                    steps.append(rec)
                elif rec["type"] == "episode_outcome":
                    if not rec.get("success") and steps:
                        # Take last `lookback` frames before failure
                        for s in steps[-lookback:]:
                            pre_fail.append(s["frame"])
                    steps = []

    if not pre_fail:
        return []

    # Aggregate stats and flag any field that consistently exceeds typical bounds
    rules = []
    for field_name in ("imu_rpy", "imu_omega"):
        vals = np.array([f[field_name] for f in pre_fail if field_name in f], dtype=np.float32)
        if vals.size == 0:
            continue
        for axis in range(vals.shape[1]):
            v = vals[:, axis]
            q_lo = float(np.percentile(v, 5))
            q_hi = float(np.percentile(v, 95))
            # Only generate a rule if the range is suspiciously wide
            if abs(q_hi - q_lo) > 0.3:
                rules.append({
                    "name":     f"auto_{field_name}_{axis}_lo",
                    "field":    field_name,
                    "axis":     axis,
                    "min_val":  round(q_lo * 0.9, 4),
                    "override": {"cmd_vx": 0.0, "cmd_vy": 0.0},
                    "priority": 5,
                })
                rules.append({
                    "name":     f"auto_{field_name}_{axis}_hi",
                    "field":    field_name,
                    "axis":     axis,
                    "max_val":  round(q_hi * 0.9, 4),
                    "override": {"cmd_vx": 0.0, "cmd_vy": 0.0},
                    "priority": 5,
                })
    return rules


def build_safetymem(
    log_dir:    Path | None,
    rules_yaml: Path | None,
    out_path:   Path,
) -> None:
    rules = list(_DEFAULT_RULES)

    # Hand-authored YAML rules
    if rules_yaml and rules_yaml.exists():
        try:
            import yaml
            with open(rules_yaml) as fh:
                extra = yaml.safe_load(fh) or []
            rules.extend(extra)
            print(f"  {len(extra)} hand-authored rules from {rules_yaml}")
        except ImportError:
            print("  PyYAML not installed — skipping hand-authored rules")

    # Auto-discovered rules from logs
    if log_dir and log_dir.is_dir():
        log_paths = sorted(log_dir.glob("*.jsonl"))
        if log_paths:
            auto = _auto_rules_from_logs(log_paths)
            rules.extend(auto)
            print(f"  {len(auto)} auto-discovered rules from {len(log_paths)} logs")

    # Deduplicate by name
    seen: set[str] = set()
    unique_rules: list[dict] = []
    for r in rules:
        if r["name"] not in seen:
            seen.add(r["name"])
            unique_rules.append(r)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(unique_rules, fh, indent=2)
    print(f"  Saved {len(unique_rules)} safety rules → {out_path}")


if __name__ == "__main__":
    import sys
    _REPO = Path(__file__).resolve().parents[3]
    if str(_REPO) not in sys.path:
        sys.path.insert(0, str(_REPO))

    parser = argparse.ArgumentParser(description="Build SafetyMem from logs + YAML rules")
    parser.add_argument("--logs",  type=Path, default=None, help="log directory (.jsonl)")
    parser.add_argument("--rules", type=Path, default=None, help="YAML hand-authored rules")
    parser.add_argument("--out",   type=Path, default=Path("safetymem.json"))
    args = parser.parse_args()
    build_safetymem(args.logs, args.rules, args.out)
