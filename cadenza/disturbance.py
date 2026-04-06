"""cadenza.disturbance — DisturbanceEngine for stress-testing robot control.

Applies random environmental perturbations during MuJoCo simulation using
built-in MuJoCo APIs (xfrc_applied, geom_friction, gravity). Designed to
stress-test VLA models against real-world conditions.

Usage::

    from cadenza.disturbance import DisturbanceEngine

    engine = DisturbanceEngine(model, data, temperature=0.5)
    engine.enable()

    # Call before each mj_step:
    engine.pre_step()

    mujoco.mj_step(model, data)

    # Call after each mj_step:
    engine.post_step()

Temperature controls intensity (0.0 = off, 1.0 = maximum chaos):
    0.0       — No disturbance (identical to engine disabled)
    0.0–0.2   — Mild: light breezes, minor friction variation
    0.2–0.5   — Moderate: gusts, uneven terrain friction, slight slopes
    0.5–0.8   — Harsh: strong pushes, icy/muddy patches, steep slopes
    0.8–1.0   — Extreme: violent impacts, near-zero friction, heavy noise
"""

from __future__ import annotations

import numpy as np
import mujoco


class DisturbanceEngine:
    """Injects random environmental disturbances into a MuJoCo simulation.

    All perturbations scale with ``temperature`` (0.0–1.0). The engine can be
    toggled on/off at any time via :meth:`enable` / :meth:`disable`.

    Disturbance channels (all active simultaneously):

    1. **External force pushes** — Random impulses on the trunk body via
       ``data.xfrc_applied``. Simulates bumps, wind gusts, collisions.
    2. **Friction variation** — Modifies floor ``geom_friction`` per-episode.
       Simulates ice, mud, gravel surfaces.
    3. **Gravity perturbation** — Tilts the gravity vector laterally.
       Simulates slopes and uneven terrain.
    4. **Control noise** — Adds Gaussian noise to ``data.ctrl``.
       Simulates actuator imprecision and signal degradation.
    5. **Persistent wind** — Constant lateral force on the trunk.
       Simulates sustained wind or current.
    """

    # --- Scale constants (at temperature=1.0) ---
    _PUSH_FORCE_MAX = 30.0        # N, peak impulse force on trunk
    _PUSH_PROB_MAX = 0.005         # probability of push per physics step
    _FRICTION_RANGE = (0.05, 1.2)  # sliding friction bounds (nominal ~0.6)
    _GRAVITY_TILT_MAX = 3.0        # m/s², max lateral gravity component
    _CTRL_NOISE_MAX = 0.08         # rad, max std-dev of control noise
    _WIND_FORCE_MAX = 8.0          # N, persistent lateral force

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        temperature: float = 0.5,
        seed: int | None = None,
    ):
        if not 0.0 <= temperature <= 1.0:
            raise ValueError(f"temperature must be in [0, 1], got {temperature}")

        self.model = model
        self.data = data
        self._temperature = temperature
        self._enabled = False
        self._rng = np.random.default_rng(seed)

        # Cache body/geom IDs
        self._trunk_id = self._find_trunk_body()
        self._floor_geom_id = self._find_floor_geom()

        # Store original values for clean restore
        self._orig_gravity = self.model.opt.gravity.copy()
        self._orig_friction = (
            self.model.geom_friction[self._floor_geom_id].copy()
            if self._floor_geom_id is not None else None
        )

        # Per-episode randomized state (set on enable/reset)
        self._wind_force = np.zeros(3)
        self._friction_scale = 1.0
        self._gravity_offset = np.zeros(3)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def temperature(self) -> float:
        return self._temperature

    @temperature.setter
    def temperature(self, value: float):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"temperature must be in [0, 1], got {value}")
        self._temperature = value
        if self._enabled:
            self._randomize_episode()

    @property
    def enabled(self) -> bool:
        return self._enabled

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def enable(self):
        """Turn on disturbances. Randomizes per-episode parameters."""
        self._enabled = True
        self._randomize_episode()

    def disable(self):
        """Turn off disturbances and restore original model parameters."""
        self._enabled = False
        self._restore_originals()

    def reset(self):
        """Re-roll per-episode random parameters (call on env reset)."""
        if self._enabled:
            self._restore_originals()
            self._randomize_episode()

    # ------------------------------------------------------------------
    # Step hooks — call around mj_step
    # ------------------------------------------------------------------

    def pre_step(self):
        """Apply disturbances before ``mujoco.mj_step()``.

        Call this once per physics step, before stepping the simulation.
        """
        if not self._enabled or self._temperature == 0.0:
            return

        t = self._temperature

        # 1. Random impulse push
        if self._trunk_id is not None:
            if self._rng.random() < self._PUSH_PROB_MAX * t:
                force = self._rng.standard_normal(3) * self._PUSH_FORCE_MAX * t
                self.data.xfrc_applied[self._trunk_id, :3] += force

        # 2. Persistent wind on trunk
        if self._trunk_id is not None:
            self.data.xfrc_applied[self._trunk_id, :3] += self._wind_force

        # 3. Control noise
        if t > 0:
            noise_std = self._CTRL_NOISE_MAX * t
            noise = self._rng.standard_normal(self.data.ctrl.shape) * noise_std
            self.data.ctrl[:] += noise.astype(self.data.ctrl.dtype)

    def post_step(self):
        """Clean up after ``mujoco.mj_step()``.

        Call this once per physics step, after stepping the simulation.
        Clears transient forces so they don't accumulate.
        """
        if not self._enabled:
            return

        # Clear xfrc_applied so forces don't persist into next step
        if self._trunk_id is not None:
            self.data.xfrc_applied[self._trunk_id] = 0.0

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _find_trunk_body(self) -> int | None:
        """Find the main body (trunk/torso/pelvis)."""
        for name in ("trunk", "torso", "pelvis", "base_link"):
            try:
                bid = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, name
                )
                if bid >= 0:
                    return bid
            except Exception:
                continue
        # Fallback: body 1 (first non-world body)
        return 1 if self.model.nbody > 1 else None

    def _find_floor_geom(self) -> int | None:
        """Find the floor/ground geom."""
        for name in ("floor", "ground", "terrain", "plane"):
            try:
                gid = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_GEOM, name
                )
                if gid >= 0:
                    return gid
            except Exception:
                continue
        return None

    def _randomize_episode(self):
        """Roll per-episode disturbance parameters scaled by temperature."""
        t = self._temperature

        # Wind: random direction, magnitude scaled by temperature
        angle = self._rng.uniform(0, 2 * np.pi)
        magnitude = self._rng.uniform(0.3, 1.0) * self._WIND_FORCE_MAX * t
        self._wind_force = np.array([
            magnitude * np.cos(angle),
            magnitude * np.sin(angle),
            0.0,
        ])

        # Friction: interpolate between extreme and nominal
        if self._floor_geom_id is not None and self._orig_friction is not None:
            nominal = self._orig_friction[0]
            lo, hi = self._FRICTION_RANGE
            # At t=0 stay at nominal; at t=1 sample full range
            sampled = self._rng.uniform(lo, hi)
            friction = nominal + t * (sampled - nominal)
            self.model.geom_friction[self._floor_geom_id, 0] = friction
            self._friction_scale = friction

        # Gravity tilt: add lateral component to simulate slopes
        lateral = self._rng.standard_normal(2) * self._GRAVITY_TILT_MAX * t
        self._gravity_offset = np.array([lateral[0], lateral[1], 0.0])
        self.model.opt.gravity[:] = self._orig_gravity + self._gravity_offset

    def _restore_originals(self):
        """Restore model to pre-disturbance state."""
        self.model.opt.gravity[:] = self._orig_gravity
        if self._floor_geom_id is not None and self._orig_friction is not None:
            self.model.geom_friction[self._floor_geom_id] = self._orig_friction
        if self._trunk_id is not None:
            self.data.xfrc_applied[self._trunk_id] = 0.0
        self._wind_force = np.zeros(3)
        self._gravity_offset = np.zeros(3)

    def __repr__(self):
        state = "ON" if self._enabled else "OFF"
        return f"DisturbanceEngine(temperature={self._temperature}, {state})"
