"""Tier-0 full-command step response from rest, one axis at a time (ticket 12).

    uv run python .scratch/physics-fidelity/candidate-12/step.py > .scratch/physics-fidelity/candidate-12/step_<before|after>.txt

Peak acceleration over the first 50 ms, steady speed after 2 s, for world x, world y and yaw (the robot starts at yaw 0, so these are its body axes too).
"""

import mujoco
import numpy as np

from mjsoccer import scene as S

model, robots, ball = S.build(1, 0, tier=0)
r = robots[0]
print("ctrlrange", r.drive_range[:, 1], "forcerange", model.actuator_forcerange[list(r.drive), 1])
for axis, name in enumerate(("x", "y", "yaw")):
    data = mujoco.MjData(model)
    r.place(data, -5.0, -4.0, 0.0)  # room for 7 m before a wall
    ball.place(data, 4.0, 3.0)
    mujoco.mj_forward(model, data)
    cmd = np.zeros(3)
    cmd[axis] = r.drive_range[axis, 1]
    dt = model.opt.timestep
    vel = []
    for _ in range(round(2.0 / dt)):
        data.ctrl[list(r.drive)] = cmd
        mujoco.mj_step(model, data)
        vel.append(data.qvel[r.qvel_adr + axis])
    vel = np.array(vel)
    acc = np.diff(vel[: round(0.05 / dt)]) / dt
    print(f"{name:3s} accel {acc.max():8.3f}  steady {vel[-1]:7.3f}")
