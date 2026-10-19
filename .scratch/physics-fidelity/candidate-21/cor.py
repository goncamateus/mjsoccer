"""Ticket 21: ticket 17's COR rows, read off the model as shipped (nothing patched on the MjModel).

    uv run python .scratch/physics-fidelity/candidate-21/cor.py > .scratch/physics-fidelity/candidate-21/cor.txt

Same harness as candidate-17/drop.py (imported), on models/scene.xml, at 1/960 s and 0.5 ms.
Also the ball's contacts at the deepest point of a 3 m/s wall throw: one per touched hull,
carrying the pair's solref, i.e. the explicit pair replaced the dynamic contact, not added to it.
"""

import sys

import mujoco
import numpy as np

sys.path.insert(0, ".scratch/physics-fidelity/candidate-17")
import drop  # noqa: E402

print(drop.hdr)
for dt in (drop.DT_REF, 5e-4):
    for surface in ("floor", "wall"):
        m = mujoco.MjModel.from_xml_path("models/scene.xml")
        m.opt.timestep = dt
        if surface == "wall":
            m.opt.gravity[:] = 0
        vals = drop.row(f"{surface} {'1/960 s' if dt == drop.DT_REF else '0.5 ms'}", m, surface)
        pen, ms, ok = (lambda rs: (max(r[1] for r in rs) * 1e3, max(r[2] for r in rs) * 1e3, all(r[3] for r in rs)))(
            [drop.bounce(m, surface, 6.5, g) for g in drop.GAPS])
        print(f"{'':<28}pen {pen:.1f} mm, contact {ms:.1f} ms, field side {ok} @6.5; all in 0.47-0.56: "
              f"{all(0.47 <= c <= 0.56 for c in vals)}")

m = mujoco.MjModel.from_xml_path("models/scene.xml")
m.opt.gravity[:] = 0
d = mujoco.MjData(m)
d.qpos[drop.BALL_Q : drop.BALL_Q + 3] = [0.865, drop.WALL_Y - drop.R - 0.001, drop.FLOOR_Z + drop.R + 0.05]
d.qvel[drop.BALL_V + 1] = 3.0
deepest = []
for _ in range(100):
    mujoco.mj_step(m, d)
    cs = [(m.geom(c.geom2 if c.geom1 == drop.BALL_GEOM else c.geom1).name, tuple(np.round(c.solref, 4)), c.dist)
          for c in d.contact[: d.ncon] if drop.BALL_GEOM in (c.geom1, c.geom2)]
    if cs and min(x[2] for x in cs) < min((x[2] for x in deepest), default=0):
        deepest = cs
print("wall contacts at the deepest step:", [(n, s, f"{dist * 1e3:.1f} mm") for n, s, dist in deepest])
