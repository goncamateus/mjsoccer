"""Ticket 18: per-wheel floor contact while driving, on both body axes.

    MJSOCCER_MODELS=<variant dir> uv run python .scratch/physics-fidelity/candidate-18/contact.py <label> [k ...]

Full saturating command (tasks.wheel_command of a 10 m/s ask) scaled by k, from a settled rest,
2 s of raw mj_step. Per run: fraction of steps each wheel touches the floor, histogram of how many
wheels touch at once (0 = the robot is airborne, 3 = one wheel "left over"), the fraction of steps
anything else of the robot touches the floor, chassis z std, final body-frame speed on the
commanded axis (mean of the last 0.1 s) against the kinematic one, and peak yaw rate.
"""
import sys
from types import SimpleNamespace

import mujoco
import numpy as np

from mjsoccer import scene
from mjsoccer.tasks import wheel_command

label = sys.argv[1]
KS = [float(k) for k in sys.argv[2:]] or [1.0]
m, robots, ball = scene.build(1, 0, 1)
r = robots[0]
d = mujoco.MjData(m)
fl = m.geom("floor").id
wheel = np.full(m.ngeom, -1)
robot_geom = np.zeros(m.ngeom, bool)
for g in range(m.ngeom):
    n = m.geom(g).name or ""
    robot_geom[g] = n.startswith(r.prefix)
    for k in "1234":
        if f"col_wheel{k}_roller" in n:
            wheel[g] = int(k) - 1
dt = m.opt.timestep
print(f"# {label}: dt {dt:.3g} s, mjsoccer models {scene.MODELS_DIR}")
for axis, ask, start in ((0, (10.0, 0.0, 0.0), (-5.5, 0.0)), (1, (0.0, 10.0, 0.0), (0.0, -4.0))):
    for k in KS:
        mujoco.mj_resetData(m, d)
        r.place(d, *start, 0.0)
        ball.place(d, 5.5, 4.0)
        mujoco.mj_forward(m, d)
        for _ in range(round(0.3 / dt)):  # settle onto the carpet
            mujoco.mj_step(m, d)
        env = SimpleNamespace(robots=robots, model=m, data=d)
        a = wheel_command(env, 0, *ask)
        v_kin = 0.01 / abs(wheel_command(env, 0, *(np.array(ask) / 1000))).max()  # unsaturated
        cmd = k * a * r.drive_range[:, 1]
        n = round(2.0 / dt)
        touch, hist, other, z, v, yaw = np.zeros(4), np.zeros(5), 0, [], [], 0.0
        for _ in range(n):
            d.ctrl[list(r.drive)] = cmd
            mujoco.mj_step(m, d)
            hit, o = np.zeros(4, bool), False
            for c in d.contact[: d.ncon]:
                if fl in (c.geom1, c.geom2):
                    g = c.geom2 if c.geom1 == fl else c.geom1
                    if wheel[g] >= 0:
                        hit[wheel[g]] = True
                    elif robot_geom[g]:
                        o = True
            touch += hit
            hist[hit.sum()] += 1
            other += o
            R = d.xmat[r.body].reshape(3, 3)
            z.append(d.xpos[r.body][2])
            v.append((R.T @ d.qvel[r.qvel_adr : r.qvel_adr + 3])[axis])
            yaw = max(yaw, abs(d.qvel[r.qvel_adr + 5]))
        vf = np.mean(v[-round(0.1 / dt):])
        print(f"{'xy'[axis]} k={k:.2f}: frac {np.round(touch / n, 2)} n_wheels(0..4) {np.round(hist / n, 2)} "
              f"other {other / n:.2f} z_std {1e3 * np.std(z[n // 4:]):.3f} mm  v {vf:.2f} of {k * v_kin:.2f} m/s  peak|yaw| {yaw:.2f}", flush=True)
