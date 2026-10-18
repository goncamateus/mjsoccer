"""Ticket 18: terminal forward speed. Full forward command along the field diagonal, 3.5 s.

    MJSOCCER_MODELS=<variant dir> uv run python .scratch/physics-fidelity/candidate-18/terminal.py <label>
"""
import sys
from types import SimpleNamespace

import mujoco
import numpy as np

from mjsoccer import scene
from mjsoccer.tasks import wheel_command

m, robots, ball = scene.build(1, 0, 1)
r = robots[0]
d = mujoco.MjData(m)
dt = m.opt.timestep
mujoco.mj_resetData(m, d)
r.place(d, -5.5, -3.8, 0.6435)  # ~12 m of diagonal before a wall, as candidate-14/accel.py
ball.place(d, 5.5, 4.0)
mujoco.mj_forward(m, d)
for _ in range(round(0.3 / dt)):
    mujoco.mj_step(m, d)
cmd = wheel_command(SimpleNamespace(robots=robots, model=m, data=d), 0, 10.0, 0.0, 0.0) * r.drive_range[:, 1]
v = []
for _ in range(round(3.5 / dt)):
    d.ctrl[list(r.drive)] = cmd
    mujoco.mj_step(m, d)
    R = d.xmat[r.body].reshape(3, 3)
    v.append((R.T @ d.qvel[r.qvel_adr : r.qvel_adr + 3])[0])
print(f"{sys.argv[1]}: v(t) at 1/2/2.5/3/3.5 s", [round(float(v[round(t / dt) - 1]), 2) for t in (1, 2, 2.5, 3, 3.5)],
      f"of 3.30; final pos {d.xpos[r.body][:2].round(2)}")
