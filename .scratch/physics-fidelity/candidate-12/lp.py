"""Tier 1 friction-bound acceleration as a linear programme (ticket 12, review of the reopen).

    uv run python .scratch/physics-fidelity/candidate-12/lp.py > .scratch/physics-fidelity/candidate-12/lp.txt

yaw.py's bounds assume every wheel at its own grip (x) or every wheel at the rear mean (y).
The exact bound: wheel forces f_i along each rolling direction, |f_i| <= mu N_i at the settled
static loads, the off-axis force and the moment about the CoM both zero; maximise the force on
the axis. Four unknowns and two equalities, so the optimum is a vertex with two wheels at
their grip: enumerated, no solver. Also the CoM-basis alpha_max.
"""

import itertools

import mujoco
import numpy as np

from mjsoccer import scene as S

model, robots, ball = S.build(1, 0, tier=1)
r = robots[0]
data = mujoco.MjData(model)
r.place(data, 0.0, 0.0, 0.0)
ball.place(data, 4.0, 3.0)
mujoco.mj_forward(model, data)
for _ in range(round(0.5 / model.opt.timestep)):
    mujoco.mj_step(model, data)

R = data.xmat[r.body].reshape(3, 3)
com = (R.T @ (data.subtree_com[r.body] - data.xpos[r.body]))[:2]
mass = model.body_subtreemass[r.body]
mu = 0.25
wheel_body = [model.jnt_bodyid[model.actuator_trnid[a, 0]] for a in r.drive]
hubs, rolls = [], []
for act, b in zip(r.drive, wheel_body):
    j = model.actuator_trnid[act, 0]
    hubs.append((R.T @ (data.xpos[b] - data.xpos[r.body]))[:2] - com)
    axis = R.T @ (data.xmat[b].reshape(3, 3) @ model.jnt_axis[j])
    rolls.append(np.cross(axis, (0.0, 0.0, 1.0))[:2])
hubs, rolls = np.array(hubs), np.array(rolls)
loads = np.zeros(4)
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
moment = hubs[:, 0] * rolls[:, 1] - hubs[:, 1] * rolls[:, 0]  # z of hub x roll, about the CoM
cap = mu * loads
print("loads (N)", np.round(loads, 3))


def bound(axis):
    off = rolls[:, 1 - axis]
    best, arg = -np.inf, None
    for i, j in itertools.combinations(range(4), 2):
        for si, sj in itertools.product((-1, 1), repeat=2):
            A = np.vstack([off, moment, np.eye(4)[i], np.eye(4)[j]])
            b = np.array([0.0, 0.0, si * cap[i], sj * cap[j]])
            f = np.linalg.solve(A, b)
            if np.all(np.abs(f) <= cap + 1e-12):
                val = f @ rolls[:, axis]
                if val > best:
                    best, arg = val, f
    return best / mass, arg


for axis, name in ((0, "x"), (1, "y")):
    a, f = bound(axis)
    print(f"a_{name} LP bound {a:.3f} m/s^2, wheel forces {np.round(f, 3)} N, grip used {np.round(np.abs(f) / cap, 3)}")
izz = 0.009545
torque = mu * (loads * np.abs(moment)).sum()
print(f"yaw: mu sum(N d) about CoM {torque:.4f} N*m, / Izz {izz} = {torque / izz:.2f} rad/s^2")

# The settled split is diagonal (wheels 1 and 3 heavier than 2 and 4): four wheels on a rigid
# chassis are statically indeterminate, so it is the solver's, not the robot's. The same LP
# with each axle's load shared left/right:
front = hubs[:, 0] + com[0] > 0
loads = np.where(front, loads[front].mean(), loads[~front].mean())
cap = mu * loads
print("symmetric loads (N)", np.round(loads, 3))
for axis, name in ((0, "x"), (1, "y")):
    a, f = bound(axis)
    print(f"a_{name} LP bound {a:.3f} m/s^2, grip used {np.round(np.abs(f) / cap, 3)}")
