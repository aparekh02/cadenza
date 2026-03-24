"""gym3d/env.py — Go1Env: shared MuJoCo 3D environment.

Single source of truth for the Go1 physics simulation used by both
examples (visual_gym.py, run_controller.py) and tests (test_standup_3d.py).

Design
------
  - Headless by default (no viewer required for tests).
  - Optional passive viewer via render=True (mjpython required).
  - Exposes a clean step/reset API so callers don't touch MuJoCo internals.
  - All sensor reads are bundled into Go1Obs (a dataclass), not raw arrays.
  - Physics constants (timestep, ctrl_hz, max_torque) mirror go1.xml exactly.

Joint ordering
--------------
  Index  Name         Axis convention
  ─────  ───────────  ─────────────────────────────
  0      FL_hip       +X  (abduct = positive)
  1      FL_thigh     -Y
  2      FL_calf      -Y
  3      FR_hip       -X  (abduct = positive in joint space)
  4      FR_thigh     -Y
  5      FR_calf      -Y
  6      RL_hip       +X
  7      RL_thigh     -Y
  8      RL_calf      -Y
  9      RR_hip       -X
  10     RR_thigh     -Y
  11     RR_calf      -Y

Foot contact detection
----------------------
  A foot is considered "in contact" when its body z-position is within
  2.5 × foot_radius of the ground plane (z=0).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import mujoco

# ── Paths ─────────────────────────────────────────────────────────────────────
_ASSETS = Path(__file__).parent / "assets"
_GO1_XML = _ASSETS / "go1.xml"

# ── Physics constants (must match go1.xml) ────────────────────────────────────
CTRL_HZ      = 50           # cadenza control rate (Hz)
TIMESTEP     = 0.002        # MuJoCo timestep (s) — matches go1.xml option
MAX_TORQUE   = 33.5         # Nm — matches go1.xml ctrlrange
FOOT_RADIUS  = 0.0165       # m  — matches go1.xml sphere size

_FOOT_BODIES = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
_JOINT_NAMES = [
    "FL_hip", "FL_thigh", "FL_calf",
    "FR_hip", "FR_thigh", "FR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
]


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Go1Obs:
    """All sensor observations for one control step.

    All arrays are float32. Angles in radians, velocities in rad/s,
    positions in metres, angular rates in rad/s.
    """
    # Free-body state
    pos:    np.ndarray    # (3,) body CoM position [x, y, z]
    quat:   np.ndarray    # (4,) body quaternion   [w, x, y, z]
    linvel: np.ndarray    # (3,) body linear velocity
    angvel: np.ndarray    # (3,) body angular velocity (≈ IMU gyro)

    # Derived orientation
    roll:   float         # rad
    pitch:  float         # rad
    yaw:    float         # rad

    # Joint state
    q:      np.ndarray    # (12,) joint positions  [rad]
    dq:     np.ndarray    # (12,) joint velocities [rad/s]

    # Contact
    contacts: np.ndarray  # (4,) float32 {0,1} — [FL, FR, RL, RR]

    # Convenience
    trunk_z: float        # = pos[2]
    sim_time: float       # MuJoCo simulation time (s)

    @property
    def n_contacts(self) -> int:
        return int(self.contacts.sum())


# ─────────────────────────────────────────────────────────────────────────────
class Go1Env:
    """MuJoCo 3D environment for the Unitree Go1 quadruped.

    Parameters
    ----------
    xml_path    : Path to MuJoCo XML. Defaults to gym3d/assets/go1.xml.
    render      : If True, opens a MuJoCo passive viewer (requires mjpython).
    ctrl_hz     : Control frequency. Physics substeps are computed automatically.
    realtime    : If True and render=True, pace wall-clock to ctrl_hz.
    drop_steps  : Physics steps (damping only) on reset before returning obs.
                  Allows robot to settle to ground from initial prone pose.
    drop_damping: Joint damping coefficient used during drop phase.

    Example
    -------
    >>> env = Go1Env()
    >>> obs = env.reset()
    >>> for _ in range(500):
    ...     torques = np.zeros(12)          # replace with your controller
    ...     obs, done, info = env.step(torques)
    ...     if done:
    ...         break
    >>> env.close()
    """

    def __init__(
        self,
        xml_path:     Optional[Path] = None,
        render:       bool           = False,
        ctrl_hz:      float          = CTRL_HZ,
        realtime:     bool           = True,
        drop_steps:   int            = 400,
        drop_damping: float          = 1.5,
    ):
        self._xml_path    = Path(xml_path) if xml_path else _GO1_XML
        self._render      = render
        self._ctrl_hz     = ctrl_hz
        self._realtime    = realtime
        self._drop_steps  = drop_steps
        self._drop_damping = drop_damping

        # Physics substeps per control step
        self._phys_per_cmd = max(1, int(round(1.0 / (TIMESTEP * ctrl_hz))))

        # Build model
        self._model = mujoco.MjModel.from_xml_path(str(self._xml_path))
        self._data  = mujoco.MjData(self._model)

        # Cache foot body IDs
        self._foot_ids: list[int] = []
        for name in _FOOT_BODIES:
            bid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, name)
            self._foot_ids.append(bid)

        # Viewer (optional)
        self._viewer = None
        if render:
            import mujoco.viewer as _mv
            self._viewer = _mv.launch_passive(self._model, self._data)
            self._viewer.cam.distance  = 2.5
            self._viewer.cam.elevation = -18
            self._viewer.cam.azimuth   = 30
            self._viewer.cam.lookat[:] = [0.0, 0.0, 0.20]

        # Real-time pacing state
        self._step_count   = 0
        self._t_ep_start:  Optional[float] = None

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def model(self) -> mujoco.MjModel:
        """Raw MuJoCo model (read-only access for callers that need it)."""
        return self._model

    @property
    def data(self) -> mujoco.MjData:
        """Raw MuJoCo data (read-only access for callers that need it)."""
        return self._data

    @property
    def ctrl_hz(self) -> float:
        return self._ctrl_hz

    @property
    def is_open(self) -> bool:
        if self._viewer is None:
            return True
        return self._viewer.is_running()

    def reset(
        self,
        init_qpos:   Optional[np.ndarray] = None,
        init_height: float = 0.12,
        quat:        Optional[np.ndarray] = None,
    ) -> Go1Obs:
        """Reset simulation and drop robot to ground.

        Parameters
        ----------
        init_qpos   : (12,) initial joint positions. Defaults to prone pose
                      from the MuJoCo XML (thigh=1.5, calf=-2.8, hip=0).
        init_height : Initial z of the trunk before the drop phase.
        quat        : (4,) initial quaternion [w,x,y,z]. Defaults to upright.

        Returns
        -------
        obs after drop/settle phase.
        """
        d = self._data
        m = self._model

        d.qpos[:] = 0.0
        d.qvel[:] = 0.0
        d.ctrl[:] = 0.0

        # Trunk position
        d.qpos[2] = float(init_height)
        # Orientation: upright by default
        if quat is not None:
            d.qpos[3:7] = np.asarray(quat, dtype=np.float64)
        else:
            d.qpos[3]   = 1.0   # w=1 → identity quaternion

        # Joint positions: use provided or default prone pose
        if init_qpos is not None:
            d.qpos[7:19] = np.asarray(init_qpos, dtype=np.float64)
        else:
            # Prone pose — same as spec.poses.prone: (hip=0, thigh=1.5, calf=-2.8)×4
            prone = np.tile([0.0, 1.5, -2.8], 4)
            d.qpos[7:19] = prone

        mujoco.mj_forward(m, d)

        # Drop phase: gravity + pure velocity damping (no position control)
        for _ in range(self._drop_steps):
            dq = d.qvel[6:18].astype(np.float64)
            d.ctrl[:] = np.clip(-self._drop_damping * dq, -MAX_TORQUE, MAX_TORQUE)
            mujoco.mj_step(m, d)

        self._step_count  = 0
        self._t_ep_start  = time.monotonic()

        if self._viewer is not None:
            self._viewer.sync()

        return self._read_obs()

    def step(
        self,
        torques: np.ndarray,
        camera_follow: bool = True,
    ) -> tuple[Go1Obs, bool, dict]:
        """Apply torques, advance physics, return (obs, done, info).

        Parameters
        ----------
        torques      : (12,) joint torques in Nm. Clipped to ±MAX_TORQUE.
        camera_follow: If viewer is open, pan camera to follow robot height.

        Returns
        -------
        obs   : Go1Obs with all sensor readings after the step.
        done  : True if viewer was closed (user pressed Escape).
        info  : dict with step metadata.
        """
        d = self._data
        m = self._model

        # Apply torques
        d.ctrl[:] = np.clip(
            np.asarray(torques, dtype=np.float64), -MAX_TORQUE, MAX_TORQUE
        )

        # Physics substeps
        for _ in range(self._phys_per_cmd):
            mujoco.mj_step(m, d)

        self._step_count += 1
        obs = self._read_obs()

        # Viewer sync + optional camera follow
        done = False
        if self._viewer is not None:
            if camera_follow:
                self._viewer.cam.lookat[2] = max(obs.trunk_z * 0.8, 0.10)
            self._viewer.sync()
            done = not self._viewer.is_running()

        # Real-time pacing
        if self._render and self._realtime and self._t_ep_start is not None:
            deadline  = self._t_ep_start + self._step_count / self._ctrl_hz
            sleep_for = deadline - time.monotonic()
            if sleep_for > 0.0:
                time.sleep(sleep_for)

        info = {
            "step":      self._step_count,
            "sim_time":  obs.sim_time,
            "n_contacts": obs.n_contacts,
        }
        return obs, done, info

    def set_camera(
        self,
        distance:  float = 2.5,
        elevation: float = -18.0,
        azimuth:   float = 30.0,
        lookat:    Optional[list] = None,
    ) -> None:
        """Adjust viewer camera (no-op if headless)."""
        if self._viewer is None:
            return
        self._viewer.cam.distance  = distance
        self._viewer.cam.elevation = elevation
        self._viewer.cam.azimuth   = azimuth
        if lookat is not None:
            self._viewer.cam.lookat[:] = lookat

    def close(self) -> None:
        """Close viewer if open."""
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:
                pass
            self._viewer = None

    # ── Sensor helpers ────────────────────────────────────────────────────────

    def _read_obs(self) -> Go1Obs:
        d = self._data
        pos    = d.qpos[0:3].astype(np.float32).copy()
        quat   = d.qpos[3:7].astype(np.float32).copy()   # [w,x,y,z]
        linvel = d.qvel[0:3].astype(np.float32).copy()
        angvel = d.qvel[3:6].astype(np.float32).copy()
        q      = d.qpos[7:19].astype(np.float32).copy()
        dq     = d.qvel[6:18].astype(np.float32).copy()

        roll, pitch, yaw = _quat_to_rpy(quat)
        contacts         = self._read_contacts()

        return Go1Obs(
            pos=pos, quat=quat, linvel=linvel, angvel=angvel,
            roll=roll, pitch=pitch, yaw=yaw,
            q=q, dq=dq, contacts=contacts,
            trunk_z=float(pos[2]),
            sim_time=float(d.time),
        )

    def _read_contacts(self) -> np.ndarray:
        fc = np.zeros(4, dtype=np.float32)
        for i, bid in enumerate(self._foot_ids):
            if bid >= 0 and self._data.xpos[bid, 2] < FOOT_RADIUS * 2.5:
                fc[i] = 1.0
        return fc

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "Go1Env":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def __repr__(self) -> str:
        return (f"Go1Env(ctrl_hz={self._ctrl_hz}, "
                f"render={self._render}, "
                f"phys_per_cmd={self._phys_per_cmd})")


# ─────────────────────────────────────────────────────────────────────────────
#  Utility
# ─────────────────────────────────────────────────────────────────────────────

def _quat_to_rpy(q: np.ndarray) -> tuple[float, float, float]:
    """Convert quaternion [w,x,y,z] → (roll, pitch, yaw) in radians."""
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    roll  = math.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    pitch = math.asin(max(-1.0, min(1.0, 2*(w*y - z*x))))
    yaw   = math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return roll, pitch, yaw
