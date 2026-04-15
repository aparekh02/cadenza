"""Cadenza Go1 demo.

Usage:
    mjpython tests/test_go1_actions.py
    mjpython tests/test_go1_actions.py "walk forward then turn left then jump"
"""
import cadenza as cadenza

go1 = cadenza.go1()

go1.run([
    go1.stand(),
    go1.walk_forward(distance_m=1.0),
    go1.turn_left(),
    go1.walk_forward(distance_m=1.0),
    go1.turn_right(),
    go1.jump(),
    go1.sit(),
    go1.stand_up(),
])
