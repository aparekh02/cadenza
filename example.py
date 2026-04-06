"""Cadenza Go1 — Developer API example.

Usage:
    mjpython example.py
"""
import cadenza as cadenza

go1 = cadenza.go1()

go1.run([
    go1.stand(),
    go1.walk_forward(speed=1.5, distance_m=2.0),
    [go1.turn_left(), go1.walk_forward()],   # concurrent: walking arc
    go1.jump(speed=2.0, extension=1.2),
    go1.sit(),
])
