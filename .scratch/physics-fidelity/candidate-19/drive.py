"""Ticket 19: full command along body x or y from rest, 4.5 s at the physics rate.

    MJSOCCER_MODELS=<variant dir> uv run python .scratch/physics-fidelity/candidate-19/drive.py <label> <x|y> [ramp m/s^2]

Along the field diagonal, on candidate-19 variants (no wall collision), command recomputed every step
by tasks.wheel_command (saturating, body frame, no heading feedback). Prints the kinematic speed on that
axis, the speed along it at 1/2/3/4/4.5 s, a_max (best 0.1 s window, as candidate-14/accel.py), the
peak |yaw rate|, the first time |yaw rate| > 2 rad/s (the collapse) and the net heading change
(mod 2 pi). With a ramp, the commanded speed rises at that rate instead of stepping to the saturated
command (still saturated at the end).
"""
import sys
from types import SimpleNamespace

import mujoco
import numpy as np

from mjsoccer import scene
from mjsoccer.tasks import wheel_command

label, axis = sys.argv[1], sys.argv[2]
ramp = float(sys.argv[3]) if len(sys.argv) > 3 else np.inf
e = np.array((1.0, 0.0) if axis == "x" else (0.0, 1.0))
m, robots, ball = scene.build(1, 0, 1)
r = robots[0]
d = mujoco.MjData(m)
env = SimpleNamespace(robots=robots, model=m, data=d)
dt = m.opt.timestep
mujoco.mj_resetData(m, d)
r.place(d, -5.5, -3.8, 0.6435 - (0 if axis == "x" else np.pi / 2))  # the commanded axis on the diagonal
ball.place(d, 5.5, 4.0)
mujoco.mj_forward(m, d)
for _ in range(round(0.3 / dt)):
    mujoco.mj_step(m, d)
raw = wheel_command(env, 0, *(0.1 * e), 0.0)  # unsaturated at 0.1 m/s
v_kin = 0.1 / np.abs(raw).max()
v, yaw = [], []
h0 = np.arctan2(d.xmat[r.body][3], d.xmat[r.body][0])
for i in range(round(4.5 / dt)):
    d.ctrl[list(r.drive)] = wheel_command(env, 0, *(min(10.0, ramp * (i + 1) * dt) * e), 0.0) * r.drive_range[:, 1]
    mujoco.mj_step(m, d)
    R = d.xmat[r.body].reshape(3, 3)
    v.append((R.T @ d.qvel[r.qvel_adr : r.qvel_adr + 3])[:2] @ e)
    yaw.append(d.qvel[r.qvel_adr + 5])
v = np.array(v)
k = round(0.1 / dt)
i2 = int(np.argmax(np.abs(yaw) > 2))
heading = (np.arctan2(d.xmat[r.body][3], d.xmat[r.body][0]) - h0 + np.pi) % (2 * np.pi) - np.pi
acc = (v[k:] - v[:-k]).max() / (k * dt)
print(f"{label} {axis} ramp {ramp:g}: kinematic {v_kin:.2f} m/s; v at 1/2/3/4/4.5 s {[round(float(v[round(t / dt) - 1]), 2) for t in (1, 2, 3, 4, 4.5)]}"
      f" ({v[-1] / v_kin:.0%}); a_max {acc:.2f} m/s^2; peak |yaw| {np.abs(yaw).max():.1f} rad/s"
      f" {f"(> 2 at {(i2 + 1) * dt:.2f} s)" if np.abs(yaw).max() > 2 else ""}; heading change {heading:+.2f} rad")
