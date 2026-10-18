"""Ticket 06: which geoms touch the floor on tier 0 in 1 s idle (run before and after the 4->5 floor change).

    uv run python .scratch/physics-fidelity/candidate-06/tier0.py
"""
import mujoco
from mjsoccer import scene
m, rs, b = scene.build(1, 0, tier=0); d = mujoco.MjData(m); mujoco.mj_resetData(m, d); rs[0].place(d, 0, 0, 0)
fl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor"); hits = set()
for i in range(500):
    d.ctrl[:] = 0; mujoco.mj_step(m, d)
    hits |= {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2 if c.geom1 == fl else c.geom1) for c in d.contact[:d.ncon] if fl in (c.geom1, c.geom2)}
print("tier0 floor contacts:", sorted(hits))
