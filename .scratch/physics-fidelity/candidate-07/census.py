"""Ticket 07: per-robot geom census (collision vs visual, by class and contype bit) (run after the prune: 193/299; before it, 209/315).

    uv run python .scratch/physics-fidelity/candidate-07/census.py > census.txt
"""
import collections, mujoco, numpy as np
m = mujoco.MjModel.from_xml_path("models/robot_rollers.xml")
def name(g): return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or f"#{g}"
def cls(n):
    for k in ("roller", "chassis", "bumper", "motor", "dribbler", "kicker", "chipper", "wheel", "hub", "Direct"):
        if k.lower() in n.lower(): return k
    return "other"
col = [g for g in range(m.ngeom) if m.geom_contype[g] | m.geom_conaffinity[g]]
vis = [g for g in range(m.ngeom) if g not in col]
print(f"ngeom={m.ngeom} collision={len(col)} visual={len(vis)} nbody={m.nbody} (world excluded: {m.nbody-1})")
c = collections.Counter((cls(name(g)), int(m.geom_contype[g]), int(m.geom_conaffinity[g]), mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]).split("_roller")[0]) for g in col)
for k, v in sorted(c.items()): print("collision", k, v)
