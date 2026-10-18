"""Ticket 20: full spin in place on tier 1, 5 s at the physics rate. Yaw rate against the wheel speeds.

    MJSOCCER_MODELS=<models dir> uv run python .scratch/physics-fidelity/candidate-20/spin.py <label>

Prints, at the end of each second, the chassis yaw rate, the mean |wheel speed| and their ratio
(the effective r/R: 0.378 = 30.089 / 79.6 mm kinematic), and the peak yaw rate.
"""
import sys
import mujoco
import numpy as np
from mjsoccer import scene

m, robots, ball = scene.build(1, 0, 1)
r = robots[0]
d = mujoco.MjData(m)
r.place(d, 0.0, 0.0, 0.0)
ball.place(d, 4.0, 3.0)
mujoco.mj_forward(m, d)
dofs = [m.jnt_dofadr[m.actuator_trnid[a, 0]] for a in r.drive]
dt, peak, out = m.opt.timestep, 0.0, []
for i in range(round(5.0 / dt)):
    d.ctrl[list(r.drive)] = r.drive_range[:, 1]
    mujoco.mj_step(m, d)
    R = d.xmat[r.body].reshape(3, 3)
    w = abs((R @ d.qvel[r.qvel_adr + 3 : r.qvel_adr + 6])[2])
    peak = max(peak, w)
    if (i + 1) % round(1.0 / dt) == 0:
        q = np.abs(d.qvel[dofs]).mean()
        out.append(f"{(i + 1) * dt:.0f}s yaw {w:.2f} wheel {q:.1f} r/R {w / q:.4f}")
print(f"{sys.argv[1]}: peak yaw {peak:.2f} rad/s; " + "; ".join(out))
