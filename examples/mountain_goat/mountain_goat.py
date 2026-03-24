"""Mountain Goat — Go1 navigates complex terrain.
   mjpython examples/mountain_goat/mountain_goat.py
"""
import math
from pathlib import Path
import cadenza_local as cadenza
from cadenza_local import ActionCall

TERRAIN = Path(__file__).parent / "terrain.xml"
GOAL_X = -12.0

ZONES = [
    dict(name="flat",        x=(0.5, -1.0),   w=4.0, cmd="walk forward",
         sens=dict(slope_deg=0,  floor_roughness=0.1)),
    dict(name="rocky_slope", x=(-1.0, -4.0),  w=2.0, cmd="walk forward carefully",
         sens=dict(slope_deg=4,  floor_roughness=0.5)),
    dict(name="stairs",      x=(-4.0, -5.0),  w=1.6, cmd="climb step",
         sens=dict(slope_deg=12, floor_roughness=0.3)),
    dict(name="ridge",       x=(-5.0, -7.0),  w=0.6, cmd="crawl forward carefully",
         sens=dict(slope_deg=0,  floor_roughness=0.2, passage_width=0.6)),
    dict(name="downslope",   x=(-7.0, -9.5),  w=2.0, cmd="walk forward",
         sens=dict(slope_deg=5,  floor_roughness=0.4)),
    dict(name="ice",         x=(-9.5, -11.0), w=2.0, cmd="crawl forward carefully",
         sens=dict(slope_deg=0,  floor_roughness=0.02, slip_detected=True, slip_velocity=0.15)),
    dict(name="finish",      x=(-11.0, -13.0),w=2.0, cmd="walk forward",
         sens=dict(slope_deg=0,  floor_roughness=0.1)),
]


def _zone(x):
    for z in ZONES:
        if z["x"][0] >= x >= z["x"][1]: return z

mem = {"steps": {}, "corr": {}, "stair_det": 0, "climbed": 0, "climbing": False,
       "base_z": 0.0, "slowed": False}


def memory_fn(state):
    x, bz = float(state["pos"][0]), float(state["pos"][2])
    z = _zone(x)
    if not z: return {"command": "walk forward", "sensors": {}, "zone": "?"}

    cmd, sens, zn = z["cmd"], dict(z["sens"]), z["name"]
    step_up = state.get("terrain_ahead", {}).get("max_step_up", 0.0)
    gz = state.get("terrain_ahead", {}).get("ground_z_here")
    cm = step_up * 100

    if cm >= 5:
        mem["stair_det"] += 1
        if mem["stair_det"] >= 2 and not mem["climbing"]:
            mem["climbing"], mem["base_z"] = True, gz or bz - 0.27
    elif cm >= 3:
        mem["stair_det"] = max(1, mem["stair_det"])
    elif mem["climbing"]:
        if (gz or 0) - mem["base_z"] > 0.05: mem["climbed"] += 1
        mem["stair_det"] -= 1
        if mem["stair_det"] <= 0:
            mem["climbing"], mem["stair_det"], mem["slowed"] = False, 0, False
    else:
        mem["stair_det"] = max(0, mem["stair_det"] - 1)

    if mem["climbing"]:
        cmd, sens, zn = "climb step", dict(slope_deg=12, floor_roughness=0.3), f"{zn}->climb"
    elif mem["stair_det"] > 0 and cm >= 3:
        cmd, mem["slowed"] = "crawl forward carefully", True
        sens["slope_deg"] = max(sens.get("slope_deg", 0), 8)

    cr = mem["corr"].get(zn, 0) / max(mem["steps"].get(zn, 0), 1)
    if cr > 0.3 and "carefully" not in cmd:
        cmd = cmd.replace("walk forward", "walk forward carefully")

    mem["steps"][zn] = mem["steps"].get(zn, 0) + 1
    return {"command": cmd, "sensors": sens, "zone": zn,
            "swing_height": min(step_up + 0.03, 0.22) if step_up > 0.01 else None}


def vla_fn(state):
    x, y, yaw = float(state["pos"][0]), float(state["pos"][1]), float(state["yaw"])
    z = _zone(x)
    zn, zw = (z["name"], z["w"]) if z else ("?", 4.0)

    yaw_err = yaw - math.atan2(-y, 2.0)
    while yaw_err > math.pi:  yaw_err -= 2 * math.pi
    while yaw_err < -math.pi: yaw_err += 2 * math.pi

    step_up = state.get("terrain_ahead", {}).get("max_step_up", 0.0)
    step_s = (f" step:{step_up*100:.0f}cm" if step_up > 0.01 else "") + \
             (f" CLIMB#{mem['climbed']}" if mem["climbing"] else "")

    thresh = math.radians(12 if mem["climbing"] else 4 if zw < 1 else 7)
    dlim = zw * 0.25 if mem["climbing"] else 0.08 if zw < 1 else zw * 0.15
    status = f"x={x:+.2f} y={y:+.2f} yaw={math.degrees(yaw):+.0f} | {zn:<10s}{step_s}"

    if abs(yaw_err) <= thresh and abs(y) <= dlim:
        return {"ok": True, "turn": None, "log": f"{status} | ok"}

    mag = min(max(abs(yaw_err) * 1.3, math.radians(5)), math.radians(35))
    name = "turn_right" if yaw_err > 0 else "turn_left"
    mem["corr"][zn] = mem["corr"].get(zn, 0) + 1
    return {"ok": False, "turn": ActionCall(action_name=name, repeat=1, rotation_rad=mag),
            "narrow": zw < 1, "log": f"{status} | TURN {math.degrees(mag):.0f}"}


go1 = cadenza.go1(xml_path=TERRAIN, cam_distance=3.0, cam_azimuth=250)
go1.run_reactive(memory_fn, vla_fn, lambda s: float(s["pos"][0]) <= GOAL_X)
