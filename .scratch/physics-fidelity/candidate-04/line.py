"""Ticket 04: Straight-line step: kinematic forward command, full ctrlrange, body vx(t).

    MJSOCCER_MODELS=<candidate models dir> uv run python .scratch/physics-fidelity/candidate-04/line.py > .scratch/physics-fidelity/candidate-04/line_<label>.txt
"""
import mujoco, numpy as np
from mjsoccer import scene
model, robots, ball = scene.build(1, 0, 1)
r = robots[0]; d = mujoco.MjData(model)
hi = r.drive_range[0, 1]
mujoco.mj_resetData(model, d); r.place(d, 0, 0, 0); ball.place(d, 4, 3); mujoco.mj_forward(model, d)
dirs = []
for a in r.drive:
    j = model.actuator_trnid[a, 0]; b = model.jnt_bodyid[j]
    ax = d.xmat[b].reshape(3, 3) @ model.jnt_axis[j]
    dirs.append(np.cross(ax, [0, 0, -1])[:2])
dirs = np.array(dirs)
for axis, name in ((1, "y"), (0, "x")):
    u = dirs[:, axis] / np.abs(dirs[:, axis]).max()
    cmd = hi * u
    mujoco.mj_resetData(model, d); r.place(d, -3 if axis == 0 else 0, 0 if axis == 0 else -2, 0); ball.place(d, 4, 3); mujoco.mj_forward(model, d)
    ts, vs, yaw = [], [], 0.0
    for i in range(round(2.0 / model.opt.timestep)):
        d.ctrl[:] = 0; d.ctrl[list(r.drive)] = cmd; mujoco.mj_step(model, d)
        R = d.xmat[r.body].reshape(3, 3)
        v = (R.T @ d.qvel[r.qvel_adr:r.qvel_adr + 3])[axis]
        ts.append(d.time); vs.append(v); yaw = max(yaw, abs((R @ d.qvel[r.qvel_adr + 3:r.qvel_adr + 6])[2]))
    ts, vs = np.array(ts), np.array(vs); vs *= np.sign(vs[-50:].mean()); vf = vs[-50:].mean()
    a0 = np.polyfit(ts[(vs > .1 * vf) & (vs < .5 * vf)], vs[(vs > .1 * vf) & (vs < .5 * vf)], 1)[0]
    t63 = ts[np.argmax(vs > 0.632 * vf)]; t90 = ts[np.argmax(vs > 0.9 * vf)]
    print(f"{name}: cmd={cmd.round(1)} v_kin={hi*0.030089/np.abs(dirs[:,axis]).max():.2f} v_final={vf:.2f} a(10-50%)={a0:.2f} m/s2 t63={t63:.3f}s t90={t90:.3f}s peak|yaw|={yaw:.2f}")
wq = [model.jnt_dofadr[model.actuator_trnid[a, 0]] for a in r.drive]
print("last x/y run: wheel qvel", d.qvel[wq].round(1), "act force", d.actuator_force[list(r.drive)].round(4))
R = d.xmat[r.body].reshape(3, 3); print("tilt deg", np.degrees(np.arccos(R[2, 2])).round(2))
# contacts: who touches the floor
fl = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
names = {}
for c in d.contact[:d.ncon]:
    if fl in (c.geom1, c.geom2):
        g = c.geom2 if c.geom1 == fl else c.geom1
        n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or str(g)
        names[n.rsplit("_c", 1)[0][:30]] = names.get(n.rsplit("_c", 1)[0][:30], 0) + 1
print("floor contacts", names)
