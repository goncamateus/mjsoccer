"""Ticket 04: Wheel tracking: uniform yaw at full command, and straight-line step response.

    MJSOCCER_MODELS=<candidate models dir> uv run python .scratch/physics-fidelity/candidate-04/track.py > .scratch/physics-fidelity/candidate-04/track_<label>.txt
"""

import mujoco, numpy as np
from mjsoccer import scene
model, robots, ball = scene.build(1, 0, 1)
r = robots[0]; d = mujoco.MjData(model)
hi = r.drive_range[:, 1]
wq = [model.jnt_dofadr[model.actuator_trnid[a, 0]] for a in r.drive]
def run(cmd, T=3.0):
    mujoco.mj_resetData(model, d); r.place(d, 0, 0, 0); ball.place(d, 4, 3); mujoco.mj_forward(model, d)
    out = []
    for i in range(round(T / model.opt.timestep)):
        d.ctrl[:] = 0; d.ctrl[list(r.drive)] = cmd; mujoco.mj_step(model, d)
        R = d.xmat[r.body].reshape(3, 3)
        v = R.T @ d.qvel[r.qvel_adr:r.qvel_adr+3]
        out.append((d.time, *d.qvel[wq], *d.actuator_force[list(r.drive)], v[0], v[1], (R @ d.qvel[r.qvel_adr+3:r.qvel_adr+6])[2]))
    return np.array(out)
a = run(hi)
print("yaw full: wheel qvel last", a[-1, 1:5].round(1), "force", a[-1, 5:9].round(4), "yaw", a[-1, 11].round(2))
