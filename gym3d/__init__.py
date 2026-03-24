"""gym3d — Shared 3D MuJoCo gym for Cadenza examples and tests.

Provides robot environments backed by MuJoCo physics.  Any example or test
that needs a 3D simulation imports from here instead of instantiating MuJoCo
directly.

Environments
------------
    Go1Env   — Unitree Go1 quadruped (12 DOF, full physics, optional viewer)

Usage
-----
    from gym3d import Go1Env, Go1Obs

    env = Go1Env()                        # headless
    env = Go1Env(render=True)             # with MuJoCo passive viewer

    obs = env.reset()                     # drop robot to ground, return obs
    obs, done, info = env.step(torques)   # apply (12,) Nm torques, step physics

    env.close()

Assets
------
    gym3d/assets/go1.xml  — MuJoCo XML model (Go1 with corrected joint axes)
"""

from gym3d.env import Go1Env, Go1Obs

__all__ = ["Go1Env", "Go1Obs"]
