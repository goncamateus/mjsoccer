"""Ticket 21: vertical COR of an oblique floor impact, which ticket 17's normal-impact solref does not set.

    uv run python .scratch/physics-fidelity/candidate-21/oblique.py > .scratch/physics-fidelity/candidate-21/oblique_after.txt
    MJSOCCER_MODELS=<copy of models/ at HEAD> uv run python ... > .../oblique_before.txt

A chip lands at ~5 m/s horizontal with spin, and bounced on the critically damped carpet too.
Drop from 0.3 m (2.43 m/s down) with horizontal speed vx, either with no spin or already
rolling (w = vx / R, no slip at contact). COR = vz out / vz in, v_in on the last step before
contact and v_out on the first step with no contact left, mean over four contact-onset
heights (the step phase at onset moves it ~2 %).
"""

import mujoco
import numpy as np

from mjsoccer import scene

R = 0.0215
model, robots, ball = scene.build(1, 0, tier=0)


def cor(vx, rolling):
    out = []
    for dz in (0.0, 0.0003, 0.0006, 0.0009):
        d = mujoco.MjData(model)
        robots[0].place(d, -5.0, -4.0, 0.0)
        ball.place(d, -4.0, 0.0)
        q, v = ball.qpos_adr, ball.qvel_adr
        d.qpos[q + 2] = scene.FLOOR_Z + R + 0.001 + dz
        d.qvel[v : v + 3] = (vx, 0.0, -np.sqrt(2 * 9.81 * 0.3))
        if rolling:
            d.qvel[v + 4] = vx / R
        hit, vin = False, 0.0
        for _ in range(round(0.2 / model.opt.timestep)):
            prev = d.qvel[v + 2]
            mujoco.mj_step(model, d)
            touch = any(ball.geom in (c.geom1, c.geom2) for c in d.contact[: d.ncon])
            if touch and not hit:
                hit, vin = True, -prev
            elif hit and not touch:
                break
        out.append(d.qvel[v + 2] / vin if not touch else 0.0)
    return np.mean(out)


print("vx_m/s  cor_no_spin  cor_rolling")
for vx in (0.0, 1.0, 2.0, 4.0, 6.0):
    print(f"{vx:5.1f}  {cor(vx, False):.3f}  {cor(vx, True):.3f}", flush=True)
