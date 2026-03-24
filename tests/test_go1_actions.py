"""Cadenza Go1 demo.

Usage:
    mjpython tests/test_go1_actions.py
    mjpython tests/test_go1_actions.py "walk forward then turn left then jump"
"""
import sys, os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import cadenza_local as cadenza

commands = sys.argv[1] if len(sys.argv) > 1 else [
    "stand",
    "walk forward 1 meter",
    "turn left",
    "walk forward 1 meter",
    "turn right",
    "jump",
    "sit down",
    "stand up",
]

cadenza.run(commands)
