"""visualizer.py — Real-time 3D stick-figure viewer for Unitree Go1/Go2.

Renders the robot body + 4 legs using FK from current joint angles.
Updates live during a run. Activated with --visualize.

Requires matplotlib (already in venv).
"""

from __future__ import annotations

import math
import threading
from typing import TYPE_CHECKING

import numpy as np

try:
    import matplotlib
    matplotlib.use("TkAgg")          # works headless-safe on macOS too
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    _MPL = True
except Exception:
    _MPL = False

if TYPE_CHECKING:
    from cadenza_local.locomotion.robot_spec import RobotSpec
    from cadenza_local.locomotion.runtime.controller import LocoCommand


# Leg colour: FL=blue, FR=red, RL=cyan, RR=orange
_LEG_COLORS = {"FL": "#4C9BE8", "FR": "#E85454", "RL": "#4CE8C4", "RR": "#E8A24C"}
_FOOT_ORDER  = ["FL", "FR", "RL", "RR"]
_LEG_INDICES = {"FL": [0,1,2], "FR": [3,4,5], "RL": [6,7,8], "RR": [9,10,11]}
_HIP_LON     = {"FL": +1, "FR": +1, "RL": -1, "RR": -1}
_HIP_LAT     = {"FL": +1, "FR": -1, "RL": +1, "RR": -1}


def _fk_leg(leg: str, q3, kin) -> list[np.ndarray]:
    """Return [hip_pos, knee_pos, foot_pos] in body frame."""
    q_hip, q_thigh, q_calf = float(q3[0]), float(q3[1]), float(q3[2])
    hs  = _HIP_LAT[leg]
    lon = _HIP_LON[leg]

    hip_pos = np.array([lon * kin.hip_offset_lon, hs * kin.hip_offset_lat, 0.0])

    def Rx(a):
        c, s = math.cos(a), math.sin(a)
        return np.array([[1,0,0],[0,c,-s],[0,s,c]])
    def Ry(a):
        c, s = math.cos(a), math.sin(a)
        return np.array([[c,0,s],[0,1,0],[-s,0,c]])

    R = Rx(hs * q_hip)
    thigh_dir = R @ Ry(-q_thigh) @ np.array([0.0, 0.0, -1.0])
    knee_pos  = hip_pos + R @ np.array([0.0, hs * kin.hip_offset_lat, 0.0]) + thigh_dir * kin.thigh_length
    calf_dir  = R @ Ry(-(q_thigh + q_calf)) @ np.array([0.0, 0.0, -1.0])
    foot_pos  = knee_pos + calf_dir * kin.calf_length
    return [hip_pos, knee_pos, foot_pos]


class RobotVisualizer:
    """Live 3D viewer. Call update() from the control loop (thread-safe).

    Args:
        spec:        RobotSpec
        refresh_hz:  how often to redraw (default 10 — doesn't need to match control rate)
    """

    def __init__(self, spec: "RobotSpec", refresh_hz: float = 10.0):
        self._spec     = spec
        self._refresh  = 1.0 / refresh_hz
        self._lock     = threading.Lock()
        self._running  = False

        # Shared state updated by control loop
        self._q12:    np.ndarray = np.array(spec.poses.stand, dtype=np.float32)
        self._rpy:    np.ndarray = np.zeros(3, dtype=np.float32)
        self._fc:     np.ndarray = np.ones(4,  dtype=np.float32)
        self._gait:   str        = "trot"
        self._terrain: str       = "unknown"
        self._cmd_vx: float      = 0.0
        self._step:   int        = 0

        if not _MPL:
            print("[Visualizer] matplotlib not available — visual disabled")

    def start(self) -> None:
        """Set up the figure on the main thread (call once before the control loop)."""
        if not _MPL:
            return
        plt.ion()
        self._fig = plt.figure(figsize=(8, 6), facecolor="#1a1a2e")
        self._ax  = self._fig.add_subplot(111, projection="3d", facecolor="#1a1a2e")
        self._fig.canvas.manager.set_window_title(
            f"Cadenza — {self._spec.model.upper()} Live View"
        )
        plt.tight_layout()
        self._running = True

    def tick(self) -> None:
        """Redraw one frame. Must be called from the main thread."""
        if not _MPL or not self._running or not hasattr(self, "_fig"):
            return
        try:
            with self._lock:
                q12     = self._q12.copy()
                rpy     = self._rpy.copy()
                fc      = self._fc.copy()
                gait    = self._gait
                terrain = self._terrain
                cmd_vx  = self._cmd_vx
                step    = self._step
            self._ax.cla()
            self._ax.set_facecolor("#1a1a2e")
            self._draw(self._ax, q12, rpy, fc, gait, terrain, cmd_vx, step)
            self._fig.canvas.draw()
            self._fig.canvas.flush_events()
        except Exception:
            pass

    def update(self, q12: np.ndarray, rpy: np.ndarray, foot_contact: np.ndarray,
               gait: str, terrain: str, cmd_vx: float, step: int) -> None:
        """Feed latest state (safe to call at 50 Hz)."""
        if not _MPL:
            return
        with self._lock:
            self._q12     = q12.copy()
            self._rpy     = rpy.copy()
            self._fc      = foot_contact.copy()
            self._gait    = gait
            self._terrain = terrain
            self._cmd_vx  = cmd_vx
            self._step    = step

    def stop(self) -> None:
        self._running = False
        if _MPL and hasattr(self, "_fig"):
            try:
                plt.close(self._fig)
            except Exception:
                pass

    def _draw(self, ax, q12, rpy, fc, gait, terrain, cmd_vx, step):
        kin  = self._spec.kin
        roll, pitch, yaw = float(rpy[0]), float(rpy[1]), float(rpy[2])

        # Body corners (simple box)
        hl, hw, hh = kin.body_length * 0.5, kin.hip_offset_lat, 0.04
        corners = np.array([
            [ hl,  hw, hh], [ hl, -hw, hh], [-hl, -hw, hh], [-hl,  hw, hh],
            [ hl,  hw,-hh], [ hl, -hw,-hh], [-hl, -hw,-hh], [-hl,  hw,-hh],
        ])
        # Apply body rotation (approximate — body frame only)
        Ry_p = np.array([[math.cos(pitch),0,math.sin(pitch)],[0,1,0],[-math.sin(pitch),0,math.cos(pitch)]])
        corners = (Ry_p @ corners.T).T

        # Draw body edges
        body_edges = [
            (0,1),(1,2),(2,3),(3,0),  # top face
            (4,5),(5,6),(6,7),(7,4),  # bottom
            (0,4),(1,5),(2,6),(3,7),  # sides
        ]
        for i, j in body_edges:
            ax.plot([corners[i,0], corners[j,0]],
                    [corners[i,1], corners[j,1]],
                    [corners[i,2], corners[j,2]],
                    color="#7ec8e3", lw=0.8, alpha=0.7)

        # Forward arrow
        fwd = Ry_p @ np.array([hl * 1.3, 0, 0])
        ax.quiver(0, 0, 0, fwd[0], fwd[1], fwd[2],
                  color="#ffffff", arrow_length_ratio=0.3, lw=1.5)

        # Draw legs
        for li, leg in enumerate(_FOOT_ORDER):
            idx = _LEG_INDICES[leg]
            pts = _fk_leg(leg, q12[idx[0]:idx[2]+1], kin)
            # Apply body pitch to all points
            pts = [(Ry_p @ p) for p in pts]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            zs = [p[2] for p in pts]
            col   = _LEG_COLORS[leg]
            alpha = 1.0 if fc[li] > 0.5 else 0.45
            ax.plot(xs, ys, zs, color=col, lw=2.5, alpha=alpha)
            # Joint dots
            for x, y, z in zip(xs, ys, zs):
                ax.scatter(x, y, z, color=col, s=18, alpha=alpha, depthshade=False)
            # Foot contact indicator
            if fc[li] > 0.5:
                ax.scatter(xs[-1], ys[-1], zs[-1], color=col, s=80,
                           marker="*", depthshade=False)

        # Ground plane
        gs = 0.6
        xx, yy = np.meshgrid([-gs, gs], [-gs, gs])
        ax.plot_surface(xx, yy, np.full_like(xx, -kin.com_height_stand * 0.8),
                        alpha=0.12, color="#4a4a6a")

        # Styling
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.4, 0.4)
        ax.set_zlim(-0.55, 0.2)
        ax.set_xlabel("x (fwd)", color="#aaaacc", fontsize=7)
        ax.set_ylabel("y (left)", color="#aaaacc", fontsize=7)
        ax.set_zlabel("z (up)",  color="#aaaacc", fontsize=7)
        ax.tick_params(colors="#555577", labelsize=6)
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.fill = False
            pane.set_edgecolor("#333355")
        ax.set_title(
            f"step {step:>5}  |  {gait:<10}  |  terrain: {terrain:<13}"
            f"  |  vx: {cmd_vx:+.2f} m/s\n"
            f"roll: {math.degrees(roll):+.1f}°   pitch: {math.degrees(pitch):+.1f}°",
            color="#e0e0ff", fontsize=9, pad=6,
        )
        ax.view_init(elev=20, azim=-60)
