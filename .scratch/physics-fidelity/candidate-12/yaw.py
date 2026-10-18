"""Tier 1 yaw envelope after ticket 20 (ticket 12, second pass).

    uv run python .scratch/physics-fidelity/candidate-12/yaw.py > .scratch/physics-fidelity/candidate-12/yaw.txt

Wheel geometry against the robot CoM, the yaw ceilings it gives, the static wheel loads and the
friction-bound accelerations, and a full spin from rest on tier 1 (alpha over 50 ms, peak yaw).
"""

import mujoco
import numpy as np

from mjsoccer import scene as S
from mjsoccer.tasks import WHEEL_R

model, robots, ball = S.build(1, 0, tier=1)
r = robots[0]
data = mujoco.MjData(model)
r.place(data, 0.0, 0.0, 0.0)
ball.place(data, 4.0, 3.0)
mujoco.mj_forward(model, data)
for _ in range(round(0.5 / model.opt.timestep)):  # settle on the wheels
    mujoco.mj_step(model, data)

R = data.xmat[r.body].reshape(3, 3)
com = R.T @ (data.subtree_com[r.body] - data.xpos[r.body])
mass = model.body_subtreemass[r.body]
g = -model.opt.gravity[2]
mu = 0.25
v_rim = r.drive_range[0, 1] * WHEEL_R
print(f"mass {mass:.5f} kg, CoM {com[0] * 1e3:+.2f} {com[1] * 1e3:+.2f} mm from the chassis origin")

# Hub of each wheel and its rolling direction, body frame; moment arm about a point p is
# roll . (z x (hub - p)), the distance the wheel sees when the chassis spins about p.
hubs, rolls = [], []
for act in r.drive:
    j = model.actuator_trnid[act, 0]
    b = model.jnt_bodyid[j]
    hubs.append((R.T @ (data.xpos[b] - data.xpos[r.body]))[:2])
    axis = R.T @ (data.xmat[b].reshape(3, 3) @ model.jnt_axis[j])
    rolls.append(np.cross(axis, (0.0, 0.0, 1.0))[:2])
hubs, rolls = np.array(hubs), np.array(rolls)


def arms(p):
    rel = hubs - p
    return np.abs(rolls[:, 0] * -rel[:, 1] + rolls[:, 1] * rel[:, 0])


d0, dc = arms(np.zeros(2)), arms(com[:2])
print("arm about centre (mm)", np.round(d0 * 1e3, 2), " about CoM", np.round(dc * 1e3, 2))
print(f"w ceiling about centre   v_rim / R         = {v_rim / d0.mean():.2f} rad/s")
print(f"w balance about CoM      v_rim sum d/sum d2 = {v_rim * dc.sum() / (dc**2).sum():.2f} rad/s")
print(f"w ceiling about CoM      v_rim / min d      = {v_rim / dc.min():.2f} rad/s"
      "  (past it every wheel brakes the spin)")

# Static wheel loads: the floor contacts of each wheel's rollers, summed.
loads = np.zeros(len(r.drive))
wheel_body = [model.jnt_bodyid[model.actuator_trnid[a, 0]] for a in r.drive]
for k in range(data.ncon):
    c = data.contact[k]
    for gid in (c.geom1, c.geom2):
        b = model.geom_bodyid[gid]
        while b and b not in wheel_body:
            b = model.body_parentid[b]
        if b in wheel_body:
            f = np.zeros(6)
            mujoco.mj_contactForce(model, data, k, f)
            loads[wheel_body.index(b)] += f[0]
            break
print("static wheel loads (N)", np.round(loads, 3), f"sum {loads.sum():.3f}, mg {mass * g:.3f}")
cos_x, cos_y = np.abs(rolls[:, 0]), np.abs(rolls[:, 1])
front = hubs[:, 0] > 0
ax = mu * (cos_x * loads).sum() / mass
# Superseded by lp.py: equal forces leave a moment about the CoM, and the exact bound is 1.305.
ay = mu * cos_y.sum() * loads[~front].mean() / mass
print(f"a_x friction bound  mu sum(cos N) / m          = {ax:.3f} m/s^2")
print(f"a_y friction bound  mu sum(cos) N_rear / m      = {ay:.3f} m/s^2 (F equal at every wheel, no moment)")
torque = mu * (loads * dc).sum()
print(f"yaw friction torque mu sum(N d) about CoM      = {torque:.4f} N*m")

# Full spin from rest.
data = mujoco.MjData(model)
r.place(data, 0.0, 0.0, 0.0)
ball.place(data, 4.0, 3.0)
mujoco.mj_forward(model, data)
for _ in range(round(0.5 / model.opt.timestep)):
    mujoco.mj_step(model, data)
dt, w = model.opt.timestep, []
for _ in range(round(5.0 / dt)):
    data.ctrl[list(r.drive)] = r.drive_range[:, 1]
    mujoco.mj_step(model, data)
    Rm = data.xmat[r.body].reshape(3, 3)
    w.append(abs((Rm @ data.qvel[r.qvel_adr + 3 : r.qvel_adr + 6])[2]))
w = np.array(w)
k = round(0.05 / dt)
print(f"tier 1 full spin: alpha over the first 50 ms {w[k - 1] / (k * dt):.1f} rad/s^2, "
      f"best 50 ms window {(w[k:] - w[:-k]).max() / (k * dt):.1f}, peak {w.max():.2f} rad/s, "
      f"mean of the last second {w[-round(1 / dt):].mean():.2f}")
