"""Ticket 19: do rollers of the same wheel touch each other while driving? (They share contype 4 and no <exclude>.)

    MJSOCCER_MODELS=<variant dir> uv run python .scratch/physics-fidelity/candidate-19/probe.py <k>
"""
import sys
from types import SimpleNamespace
import mujoco, numpy as np
from mjsoccer import scene
from mjsoccer.tasks import wheel_command
k = float(sys.argv[1])
m, robots, ball = scene.build(1, 0, 1); r = robots[0]; d = mujoco.MjData(m); dt = m.opt.timestep
isroll = np.array(["col_wheel" in (m.geom(g).name or "") for g in range(m.ngeom)])
mujoco.mj_resetData(m, d); r.place(d, -5.5, 0, 0); ball.place(d, 5.5, 4); mujoco.mj_forward(m, d)
print("at rest: roller-roller contacts", sum(isroll[c.geom1] and isroll[c.geom2] for c in d.contact[:d.ncon]))
for _ in range(round(0.3/dt)): mujoco.mj_step(m, d)
cmd = k * wheel_command(SimpleNamespace(robots=robots, model=m, data=d), 0, 10, 0, 0) * r.drive_range[:, 1]
n = round(2/dt); cnt = 0; F = 0.0; f6 = np.zeros(6); pairs = set()
for i in range(n):
    d.ctrl[list(r.drive)] = cmd; mujoco.mj_step(m, d)
    for ci, c in enumerate(d.contact[:d.ncon]):
        if isroll[c.geom1] and isroll[c.geom2]:
            cnt += 1; mujoco.mj_contactForce(m, d, ci, f6); F += np.linalg.norm(f6[:3])
            pairs.add((m.geom(c.geom1).name, m.geom(c.geom2).name))
print(f"k={k}: roller-roller contacts {cnt} over {n} steps, mean |F| {F/max(cnt,1):.3f} N; pairs e.g. {sorted(pairs)[:4]}")
