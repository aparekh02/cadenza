"""assemble_snapshot.py — Merge memory JSONs into a single snapshot.json.

The controller loads snapshot.json at startup (read-only at runtime).

Usage
-----
    python assemble_snapshot.py \
        --mapmem   mapmem.json   \
        --skillmem skillmem.json \
        --safetymem safetymem.json \
        --usermem  usermem.json  \
        --out      snapshot.json

Any section that is not provided is stored as an empty list in the snapshot.

snapshot.json schema
--------------------
{
  "version": 1,
  "mapmem":    [...],   # list of TerrainCluster dicts
  "skillmem":  [...],   # list of Skill dicts
  "safetymem": [...],   # list of SafetyRule dicts
  "usermem":   [...]    # list of UserPreference dicts
}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path | None) -> list:
    if path and path.exists():
        with open(path) as fh:
            data = json.load(fh)
        return data if isinstance(data, list) else []
    return []


def assemble(
    mapmem_path:    Path | None,
    skillmem_path:  Path | None,
    safetymem_path: Path | None,
    usermem_path:   Path | None,
    out_path:       Path,
) -> None:
    snapshot = {
        "version":   1,
        "mapmem":    load_json(mapmem_path),
        "skillmem":  load_json(skillmem_path),
        "safetymem": load_json(safetymem_path),
        "usermem":   load_json(usermem_path),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(snapshot, fh, indent=2)

    print(f"snapshot.json assembled:")
    print(f"  mapmem    : {len(snapshot['mapmem'])} terrain clusters")
    print(f"  skillmem  : {len(snapshot['skillmem'])} skills")
    print(f"  safetymem : {len(snapshot['safetymem'])} safety rules")
    print(f"  usermem   : {len(snapshot['usermem'])} user preferences")
    print(f"  → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assemble memory snapshot.json")
    parser.add_argument("--mapmem",    type=Path, default=None)
    parser.add_argument("--skillmem",  type=Path, default=None)
    parser.add_argument("--safetymem", type=Path, default=None)
    parser.add_argument("--usermem",   type=Path, default=None)
    parser.add_argument("--out",       type=Path, default=Path("snapshot.json"))
    args = parser.parse_args()
    assemble(args.mapmem, args.skillmem, args.safetymem, args.usermem, args.out)
