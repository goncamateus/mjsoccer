"""Ticket 04 (copy of 03's script, run against a candidate via MJSOCCER_MODELS) step 1/2: uniform-wheel spin sweep on one tier-1 robot.

    uv run python .scratch/physics-fidelity/candidate-04/sweep.py [label]

Set MJSOCCER_MODELS to a patched copy of models/ for the zeroed-ipos run.
"""
import sys

import mujoco
import numpy as np

from mjsoccer import scene

label = sys.argv[1] if len(sys.argv) > 1 else "cad"
model, robots, ball = scene.build(1, 0, 1)
r = robots[0]
data = mujoco.MjData(model)

mujoco.mj_resetData(model, data)
r.place(data, 0.0, 0.0, 0.0)
ball.place(data, 4.0, 3.0)  # far from the robot: nothing else in contact
mujoco.mj_forward(model, data)
off = data.subtree_com[r.body] - data.xpos[r.body]
print(f"# {label}: subtree_com={data.subtree_com[r.body].round(7)} "
      f"xpos={data.xpos[r.body].round(7)} |xy offset|={np.hypot(*off[:2]) * 1e3:.3f} mm "
      f"ipos={model.body_ipos[r.body]}")
print("label,k,w_cmd,peak_yaw_world,peak_wz_body,steady_yaw_last1s,peak_tilt_deg,t_peak_tilt_s,final_tilt_deg,tipped60")

hi = r.drive_range[:, 1]
for k in (0.2, 0.4, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0):
    mujoco.mj_resetData(model, data)
    r.place(data, 0.0, 0.0, 0.0)
    ball.place(data, 4.0, 3.0)
    mujoco.mj_forward(model, data)
    peak_w = peak_b = peak_tilt = t_peak = 0.0
    n = round(5.0 / model.opt.timestep)
    last = []
    for step in range(n):
        data.ctrl[:] = 0.0
        data.ctrl[list(r.drive)] = k * hi
        mujoco.mj_step(model, data)
        R = data.xmat[r.body].reshape(3, 3)
        wz = (R @ data.qvel[r.qvel_adr + 3 : r.qvel_adr + 6])[2]  # world-frame yaw rate
        peak_w = max(peak_w, abs(wz))
        peak_b = max(peak_b, abs(data.qvel[r.qvel_adr + 5]))
        if step >= n - round(1.0 / model.opt.timestep):
            last.append(wz)
        t = np.degrees(np.arccos(np.clip(R[2, 2], -1, 1)))
        if t > peak_tilt:
            peak_tilt, t_peak = t, data.time
    tilt = np.degrees(np.arccos(np.clip(R[2, 2], -1, 1)))
    print(f"{label},{k},{k * hi[0]:.0f},{peak_w:.2f},{peak_b:.2f},{np.mean(last):.2f},{peak_tilt:.2f},{t_peak:.3f},{tilt:.2f},{peak_tilt > 60}")
