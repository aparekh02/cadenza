"""Run cadenza from the command line.

Usage:
    mjpython -m cadenza "walk forward 2 meters then turn left then jump"
    mjpython -m cadenza "shake hand then rear kick"
"""
import sys
from cadenza_local.sim import run

if len(sys.argv) < 2:
    print("Usage: mjpython -m cadenza \"<commands>\"")
    print()
    print("Examples:")
    print('  mjpython -m cadenza "walk forward 2 meters then turn left"')
    print('  mjpython -m cadenza "jump then shake hand then rear kick"')
    print('  mjpython -m cadenza "stand on hind legs"')
    sys.exit(1)

run(" then ".join(sys.argv[1:]))
