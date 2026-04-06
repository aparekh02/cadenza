"""Cadenza G1 — action sequence demo.

Run:  mjpython test_g1_trained_stability.py
"""
import sys, os

import cadenza as cadenza

g1 = cadenza.g1()

g1.run([
    g1.stand(duration=2.0),
    g1.crouch(duration=3.0),
    g1.walk_forward(distance_m=1.0),
    g1.stand(duration=3.0),
    g1.jump(),
    g1.stand(duration=2.0),
])
