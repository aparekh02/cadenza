"""ExperienceLogger — append-only JSONL experience recorder.

Writes one JSON line per controller step plus episode_outcome sentinel lines.
These logs can be replayed offline to build/update memory snapshots.

Format
------
Step line:
    {"type": "step", "t": <float>, "frame": {...}, "cmd": {...}}

Episode outcome:
    {"type": "episode_outcome", "t": <float>, "success": <bool>, "note": <str>}
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cadenza_local.locomotion.memory.stm      import STMFrame
    from cadenza_local.locomotion.runtime.controller import LocoCommand


class ExperienceLogger:
    """Append-only JSONL log writer.

    Args:
        log_dir: directory for log files (created if absent)
        run_id:  string identifier for this run (used as filename prefix)
    """

    def __init__(self, log_dir: str | Path = "logs", run_id: str = "run"):
        self._dir = Path(log_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        ts  = int(time.time())
        self._path = self._dir / f"{run_id}_{ts}.jsonl"
        self._fh   = open(self._path, "a")
        self._n    = 0

    # ── Write ──────────────────────────────────────────────────────────────

    def log(self, frame: "STMFrame", cmd: "LocoCommand") -> None:
        """Record one controller step."""
        record = {
            "type": "step",
            "t":    frame.timestamp,
            "frame": {
                "joint_pos":    frame.joint_pos.tolist(),
                "joint_vel":    frame.joint_vel.tolist(),
                "imu_rpy":      frame.imu_rpy.tolist(),
                "imu_omega":    frame.imu_omega.tolist(),
                "foot_contact": frame.foot_contact.tolist(),
                "cmd_vel":      frame.cmd_vel.tolist(),
            },
            "cmd": cmd.to_dict(),
        }
        self._write(record)

    def episode_outcome(self, success: bool, note: str = "") -> None:
        """Append an episode-outcome sentinel."""
        record = {
            "type":    "episode_outcome",
            "t":       time.monotonic(),
            "success": success,
            "note":    note,
        }
        self._write(record)

    def flush(self) -> None:
        self._fh.flush()

    def close(self) -> None:
        self._fh.flush()
        self._fh.close()

    # ── Properties ─────────────────────────────────────────────────────────

    @property
    def path(self) -> Path:
        return self._path

    @property
    def n_steps(self) -> int:
        return self._n

    # ── Internal ───────────────────────────────────────────────────────────

    def _write(self, record: dict) -> None:
        self._fh.write(json.dumps(record, default=_json_default) + "\n")
        self._n += 1

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __repr__(self) -> str:
        return f"ExperienceLogger(path={self._path}, steps={self._n})"


# ── JSON serialisation helper ───────────────────────────────────────────────

def _json_default(obj):
    import numpy as np
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")
