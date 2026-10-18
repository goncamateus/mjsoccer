"""Ticket 19: per-wheel yaw moment under a lateral command (body y, fraction k of the saturated command).

    MJSOCCER_MODELS=<variant dir> uv run python .scratch/physics-fidelity/candidate-19/yaw.py <label> <k>

Every 50 ms for the first 0.6 s: yaw rate, body-frame vy, and each wheel's floor force (along roll,
along axle; N) and its yaw moment about the robot's centre of mass (mN*m), 50 ms means. Wheels in
actuator order. Printed first: the body-frame wheel positions, the CoM offset from the body origin,
the static normal force per wheel and the wheel speed command.
"""
import sys
from types import SimpleNamespace

import mujoco
import numpy as np

from mjsoccer import scene
from mjsoccer.tasks import wheel_command

label, k = sys.argv[1], float(sys.argv[2])
m, robots, ball = scene.build(1, 0, 1)
r = robots[0]
d = mujoco.MjData(m)
env = SimpleNamespace(robots=robots, model=m, data=d)
dt = m.opt.timestep
fl = m.geom("floor").id
spin = [m.actuator_trnid[a, 0] for a in r.drive]
wheel_of = {g: w for w in range(4) for g in range(m.ngeom) if f"col_wheel{w+1}_" in (m.geom(g).name or "")}
mujoco.mj_resetData(m, d)
r.place(d, -5.5, -3.8, 0.6435 - np.pi / 2)
ball.place(d, 5.5, 4.0)
mujoco.mj_forward(m, d)
for _ in range(round(0.3 / dt)):
    mujoco.mj_step(m, d)
f6 = np.zeros(6)
N = np.zeros(4)
for _ in range(round(0.2 / dt)):  # static normal force per wheel, 0.2 s mean at rest
    mujoco.mj_step(m, d)
    for ci, c in enumerate(d.contact[: d.ncon]):
        g = c.geom2 if c.geom1 == fl else c.geom1
        if fl in (c.geom1, c.geom2) and g in wheel_of:
            mujoco.mj_contactForce(m, d, ci, f6)
            N[wheel_of[g]] += f6[0]
N /= round(0.2 / dt)
R = d.xmat[r.body].reshape(3, 3)
print(f"{label} k={k}: wheel xy in body frame (mm)", [(1e3 * (R.T @ (d.xpos[m.jnt_bodyid[j]] - d.xpos[r.body]))[:2]).round(1).tolist() for j in spin],
      "| com - body origin (mm)", (1e3 * (R.T @ (d.subtree_com[r.body] - d.xpos[r.body]))[:2]).round(2).tolist())
print(f"  static normal force per wheel {N.round(2).tolist()} N (sum {N.sum():.2f})")
print(f"  wheel speed command (rad/s) {(k * wheel_command(env, 0, 0, 10, 0) * r.drive_range[:, 1]).round(1).tolist()}")
win = round(0.05 / dt)
acc = np.zeros((4, 3))
for i in range(round(0.6 / dt)):
    d.ctrl[list(r.drive)] = k * wheel_command(env, 0, 0, 10, 0) * r.drive_range[:, 1]
    mujoco.mj_step(m, d)
    com = d.subtree_com[r.body]
    for ci, c in enumerate(d.contact[: d.ncon]):
        if fl not in (c.geom1, c.geom2):
            continue
        g = c.geom2 if c.geom1 == fl else c.geom1
        if g not in wheel_of:
            continue
        w = wheel_of[g]
        b = m.jnt_bodyid[spin[w]]
        ax = d.xmat[b].reshape(3, 3) @ m.jnt_axis[spin[w]]
        roll = np.cross(ax, (0, 0, 1)); roll /= np.linalg.norm(roll); axh = np.cross((0, 0, 1), roll)
        mujoco.mj_contactForce(m, d, ci, f6)
        F = c.frame.reshape(3, 3).T @ f6[:3] * (1 if c.geom1 == fl else -1)  # floor on wheel
        acc[w] += (roll @ F, axh @ F, np.cross(c.pos - com, F)[2])
    if (i + 1) % win == 0:
        R = d.xmat[r.body].reshape(3, 3)
        vy = (R.T @ d.qvel[r.qvel_adr : r.qvel_adr + 3])[1]
        a = acc / win
        print(f"  t {(i + 1) * dt:.2f} yaw {d.qvel[r.qvel_adr + 5]:6.2f} vy {vy:5.2f} | roll {a[:, 0].round(2).tolist()} axle {a[:, 1].round(2).tolist()}"
              f" | Mz {(1e3 * a[:, 2]).round(0).tolist()} sum {1e3 * a[:, 2].sum():.0f}")
        acc[:] = 0
