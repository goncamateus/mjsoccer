"""Ticket 18: what drives the contact toggling, and where the speed goes.

    MJSOCCER_MODELS=<variant dir> uv run python .scratch/physics-fidelity/candidate-18/diag.py <label> [k]

1. Ripple: min roller<->floor distance over one roller pitch (18 deg) of wheel 1, chassis held.
2. One forward run (x, command k, 2 s after settling): per wheel the dominant frequency of its
   contact on/off signal against the roller-passing frequency 20*|w|/2pi; mean total normal force
   over mg; the wheel's speed against its command; slip = 1 - v_contact_roll / (w R); v(t).
"""
import sys
from types import SimpleNamespace

import mujoco
import numpy as np

from mjsoccer import scene
from mjsoccer.tasks import WHEEL_R, wheel_command

label, k = sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
m, robots, ball = scene.build(1, 0, 1)
r = robots[0]
d = mujoco.MjData(m)
fl = m.geom("floor").id
dt = m.opt.timestep
wheel = np.full(m.ngeom, -1)
for g in range(m.ngeom):
    for w in "1234":
        if f"col_wheel{w}_roller" in (m.geom(g).name or ""):
            wheel[g] = int(w) - 1
spin = [m.actuator_trnid[a, 0] for a in r.drive]
qv = [m.jnt_dofadr[j] for j in spin]

mujoco.mj_resetData(m, d)
r.place(d, -5.5, 0.0, 0.0)
mujoco.mj_forward(m, d)
rollers = np.flatnonzero(wheel == 0)
qa = m.jnt_qposadr[spin[0]]
prof = []
for t in np.linspace(0, 2 * np.pi / 20, 73):
    d.qpos[qa] = t
    mujoco.mj_kinematics(m, d)
    prof.append(min(mujoco.mj_geomDistance(m, d, g, fl, 0.05, None) for g in rollers))
print(f"# {label}: dt {dt:.3g}; ripple of wheel 1 over one roller pitch {1e3 * np.ptp(prof):.3f} mm")

mujoco.mj_resetData(m, d)
r.place(d, -5.5, 0.0, 0.0)
ball.place(d, 5.5, 4.0)
mujoco.mj_forward(m, d)
for _ in range(round(0.3 / dt)):
    mujoco.mj_step(m, d)
env = SimpleNamespace(robots=robots, model=m, data=d)
cmd = k * wheel_command(env, 0, 10.0, 0.0, 0.0) * r.drive_range[:, 1]
n = round(2.0 / dt)
on, fn, om, slip, v = np.zeros((n, 4)), np.zeros(n), np.zeros((n, 4)), [], []
f6 = np.zeros(6)
for i in range(n):
    d.ctrl[list(r.drive)] = cmd
    mujoco.mj_step(m, d)
    for ci, c in enumerate(d.contact[: d.ncon]):
        if fl in (c.geom1, c.geom2):
            g = c.geom2 if c.geom1 == fl else c.geom1
            if wheel[g] >= 0:
                on[i, wheel[g]] = 1
                mujoco.mj_contactForce(m, d, ci, f6)
                fn[i] += f6[0]
    om[i] = d.qvel[qv]
    R = d.xmat[r.body].reshape(3, 3)
    v.append((R.T @ d.qvel[r.qvel_adr : r.qvel_adr + 3])[0])
    if i >= n // 2:  # slip in the second second: hub velocity along the roll direction vs w R
        s = []
        for w, j in enumerate(spin):
            b = m.jnt_bodyid[j]
            vel = np.zeros(6)
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, b, vel, 0)
            roll = np.cross(d.xmat[b].reshape(3, 3) @ m.jnt_axis[j], (0, 0, 1))
            s.append(1 - (roll @ vel[3:]) / (om[i, w] * WHEEL_R))
        slip.append(s)
mg = 9.81 * m.body_subtreemass[r.body]
f = np.fft.rfftfreq(n // 2, dt)
peaks = []
for w in range(4):
    x = on[n // 2 :, w] - on[n // 2 :, w].mean()
    peaks.append(f[1:][np.argmax(np.abs(np.fft.rfft(x))[1:])])
wbar = np.abs(om[n // 2 :]).mean(0)
print(f"k={k}: contact frac {on[n // 2:].mean(0).round(2)}; on/off peak Hz {np.round(peaks, 0)}; "
      f"roller-pass Hz {np.round(20 * wbar / 2 / np.pi, 0)}")
print(f"  mean sum Fn / mg {fn[n // 2:].mean() / mg:.3f}; wheel |w| {wbar.round(1)} of cmd {np.abs(cmd).round(1)}; "
      f"slip {np.mean(slip, 0).round(3)}")
print("  v(t) at 0.25/0.5/1/1.5/2 s:", [round(float(v[round(t / dt) - 1]), 2) for t in (0.25, 0.5, 1.0, 1.5, 2.0)])
