"""Ticket 04: Per-wheel floor-contact fraction and normal force, at rest and in forward drive.

    MJSOCCER_MODELS=<candidate models dir> uv run python .scratch/physics-fidelity/candidate-04/contact.py > .scratch/physics-fidelity/candidate-04/contact_<label>.txt
"""
import mujoco, numpy as np
from mjsoccer import scene
model, robots, ball = scene.build(1, 0, 1)
r = robots[0]; d = mujoco.MjData(model)
hi = r.drive_range[0, 1]
fl = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
def wheel_of(g):
    n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
    for k in "1234":
        if f"wheel{k}" in n: return int(k) - 1
    return None
f6 = np.zeros(6)
for label, cmd in (("rest", np.zeros(4)), ("fwd", hi * np.array([-1, -.8165, .8165, 1]))):
    mujoco.mj_resetData(model, d); r.place(d, -3, 0, 0); ball.place(d, 4, 3); mujoco.mj_forward(model, d)
    n = round(2.0 / model.opt.timestep); touch = np.zeros(4); fn = np.zeros(4); vs = []
    for i in range(n):
        d.ctrl[:] = 0; d.ctrl[list(r.drive)] = cmd; mujoco.mj_step(model, d)
        hit = np.zeros(4)
        for ci in range(d.ncon):
            c = d.contact[ci]
            if fl in (c.geom1, c.geom2):
                w = wheel_of(c.geom2 if c.geom1 == fl else c.geom1)
                if w is not None:
                    hit[w] = 1; mujoco.mj_contactForce(model, d, ci, f6); fn[w] += f6[0]
        touch += hit
        vs.append(np.hypot(*d.qvel[r.qvel_adr:r.qvel_adr + 2]))
    print(f"{label}: contact frac {np.round(touch / n, 2)} mean Fn {np.round(fn / n, 2)} N (mg={2.69207*9.81:.1f}) |v| end {np.mean(vs[-50:]):.2f}")
