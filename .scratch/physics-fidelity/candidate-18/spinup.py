"""Ticket 18: step by step, the wheel-1 roller(s) touching the floor: spin vs the spin the hub's axial
speed needs, normal and axial force, and the net constraint torque on the roller joint. Steps 1440-1480 at 1/960 s.

    MJSOCCER_MODELS=<variant dir> uv run python .scratch/physics-fidelity/candidate-18/spinup.py <k>
"""
import sys
from types import SimpleNamespace
import mujoco, numpy as np
from mjsoccer import scene
from mjsoccer.tasks import wheel_command
k = float(sys.argv[1])
m, robots, ball = scene.build(1, 0, 1); r = robots[0]; d = mujoco.MjData(m); dt = m.opt.timestep
fl = m.geom("floor").id
j1 = m.joint("b0_wheel1_spin").id; b1 = m.jnt_bodyid[j1]
mujoco.mj_resetData(m, d); r.place(d, -5.5, 0, 0); ball.place(d, 5.5, 4); mujoco.mj_forward(m, d)
for _ in range(round(0.3/dt)): mujoco.mj_step(m, d)
cmd = k * wheel_command(SimpleNamespace(robots=robots, model=m, data=d), 0, 10, 0, 0) * r.drive_range[:, 1]
f6 = np.zeros(6)
for i in range(round(1.5/dt) + 40):
    d.ctrl[list(r.drive)] = cmd; mujoco.mj_step(m, d)
    if i < round(1.5/dt): continue
    ax = d.xmat[b1].reshape(3,3) @ m.jnt_axis[j1]; roll = np.cross(ax, (0,0,1)); roll /= np.linalg.norm(roll); axh = np.cross((0,0,1), roll)
    hub = np.zeros(6); mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, b1, hub, 0)
    out = []
    for ci, c in enumerate(d.contact[:d.ncon]):
        if fl not in (c.geom1, c.geom2): continue
        g = c.geom2 if c.geom1 == fl else c.geom1
        if "col_wheel1" not in m.geom(g).name: continue
        rb = m.geom_bodyid[g]; dof = m.jnt_dofadr[m.body_jntadr[rb]]
        mujoco.mj_contactForce(m, d, ci, f6)
        F = c.frame.reshape(3,3).T @ f6[:3] * (1 if c.geom1 == fl else -1)
        out.append(f"{m.body(rb).name.split('_')[-1]:>3} w_r {d.qvel[dof]:7.1f} Fn {f6[0]:6.2f} Fax {F @ axh:6.2f} dist {1e3*c.dist:6.3f}mm qfrc_c {d.qfrc_constraint[dof]:.2e} qacc {d.qacc[dof]:.1e}")
    print(f"{i} hub_ax {hub[3:] @ axh:.3f} need {hub[3:] @ axh / 0.0063:6.1f} | " + (" || ".join(out) or "-- no contact"))
