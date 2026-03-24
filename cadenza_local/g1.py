"""cadenza.g1 — Developer API for the Unitree G1 humanoid.

Same structure as Go1: define actions, run them.
The robot moves ONLY through motor commands. No teleporting.

Usage::

    import cadenza

    g1 = cadenza.g1()
    g1.run([
        g1.stand(),
        g1.crouch(),
        g1.walk_forward(distance_m=1.0),
        g1.stand(),
        g1.jump(),
        g1.stand(),
    ])
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import mujoco, mujoco.viewer

from cadenza_local.go1 import Step  # reuse the same Step descriptor


class G1:
    """Unitree G1 humanoid controller.

    Example::

        import cadenza
        g1 = cadenza.g1()
        g1.run([
            g1.stand(),
            g1.crouch(),
            g1.walk_forward(distance_m=1.0),
            g1.stand(),
        ])
    """

    def __init__(self, cam_distance: float = 4.0, cam_elevation: float = -15,
                 cam_azimuth: float = 120):
        self._cam_distance = cam_distance
        self._cam_elevation = cam_elevation
        self._cam_azimuth = cam_azimuth

    # ── Action methods (return descriptors, no execution) ────────────────

    def stand(self, duration=2.0):
        return Step("stand", speed=duration)

    def crouch(self, duration=2.0):
        return Step("crouch", speed=duration)

    def walk_forward(self, distance_m=1.0, **kw):
        return Step("walk_forward", distance_m=distance_m)

    def jump(self, **kw):
        return Step("jump")

    def hold(self, duration=1.0):
        return Step("hold", speed=duration)

    # ── Run ──────────────────────────────────────────────────────────────

    def run(self, sequence: list):
        """Execute actions in MuJoCo. Continuous physics, no teleporting."""
        from cadenza_local.g1_gait import (
            setup_model, _exec_stand, _exec_crouch,
            _exec_walk, _exec_jump, _hold,
        )

        model, data = setup_model()
        steps = [s if isinstance(s, Step) else Step(s) for s in sequence]

        print(f"\n  Cadenza G1  |  {len(steps)} steps\n")

        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.distance = self._cam_distance
            viewer.cam.elevation = self._cam_elevation
            viewer.cam.azimuth = self._cam_azimuth

            for i, step in enumerate(steps):
                if not viewer.is_running():
                    break

                print(f"  [{i+1}/{len(steps)}] {step.name}", end="")
                if step.distance_m > 0:
                    print(f"  {step.distance_m}m", end="")
                print()

                if step.name == "stand":
                    _exec_stand(model, data, step.speed, viewer)
                elif step.name == "crouch":
                    _exec_crouch(model, data, step.speed, viewer)
                elif step.name == "walk_forward":
                    _exec_walk(model, data, step.distance_m or 1.0, viewer)
                elif step.name == "jump":
                    _exec_jump(model, data, viewer)
                elif step.name == "hold":
                    _hold(model, data, step.speed, viewer)

            print("\n  Done. Close viewer to exit.")
            while viewer.is_running():
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(0.02)

    # ── Deploy ───────────────────────────────────────────────────────────

    def deploy(self, sequence: list, **kw):
        """Deploy actions to a physical G1 over DDS."""
        from cadenza_local.deploy.g1_driver import G1Driver
        steps = [s if isinstance(s, Step) else Step(s) for s in sequence]
        driver = G1Driver(**kw)
        driver.connect()
        try:
            driver.deploy(steps)
        finally:
            driver.disconnect()

    def deploy_ssh(self, script: str, host: str = "192.168.123.164",
                   user: str = "unitree", key: str | None = None, **kw):
        """Deploy and run a script on the physical G1 over SSH."""
        from cadenza_local.deploy.ssh import SSHDeploy
        conn = SSHDeploy(host=host, user=user, key=key, **kw)
        conn.deploy_and_run(script)
