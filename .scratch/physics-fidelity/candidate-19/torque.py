"""Ticket 19: where along the roller the axial drag comes from, and what spins the roller.

    MJSOCCER_MODELS=<variant dir> uv run python .scratch/physics-fidelity/candidate-19/torque.py <label> <k> [tors]

Forward command k, second 1-2, all four wheels. For every roller<->floor contact: s = contact position
along the roller axle (mm from the roller centre), rho = distance from the contact to the axle line,
slip = axial slip speed of the roller material at the contact, and the torque about the axle from the
tangential force, the normal force and the torsional friction. The axial floor force on the wheel is
binned by |s| and by slip, signed positive when it opposes the hub's axial motion (drag).
"tors" zeroes the torsional friction of rollers and floor on the compiled model, as candidate-18/axial.py.
"""
import sys
from types import SimpleNamespace
import mujoco, numpy as np
from mjsoccer import scene
from mjsoccer.tasks import wheel_command
label, k = sys.argv[1], float(sys.argv[2])
m, robots, ball = scene.build(1, 0, 1); r = robots[0]; d = mujoco.MjData(m); dt = m.opt.timestep
fl = m.geom("floor").id
if "tors" in sys.argv[3:]:
    m.geom_friction[[g for g in range(m.ngeom) if "col_wheel" in (m.geom(g).name or "")], 1] = 0; m.geom_friction[fl, 1] = 0
spin = [m.actuator_trnid[a, 0] for a in r.drive]
wheel_of = {}  # roller geom -> wheel index
for w, j in enumerate(spin):
    for g in range(m.ngeom):
        if f"col_wheel{w+1}_" in (m.geom(g).name or ""): wheel_of[g] = w
mujoco.mj_resetData(m, d); r.place(d, -5.5, 0, 0); ball.place(d, 5.5, 4); mujoco.mj_forward(m, d)
for _ in range(round(0.3/dt)): mujoco.mj_step(m, d)
cmd = k * wheel_command(SimpleNamespace(robots=robots, model=m, data=d), 0, 10, 0, 0) * r.drive_range[:, 1]
n = round(2/dt); f6 = np.zeros(6); jac = np.zeros((3, m.nv))
SB = [0, 1.5, 3, 4.5, 6, 99]  # |s| bins, mm
VB = [0, 0.02, 0.1, 99]  # |slip| bins, m/s
Fs, Fv, rows = np.zeros(len(SB)-1), np.zeros(len(VB)-1), []
for i in range(n):
    d.ctrl[list(r.drive)] = cmd; mujoco.mj_step(m, d)
    if i < n // 2: continue
    for ci, c in enumerate(d.contact[:d.ncon]):
        if fl not in (c.geom1, c.geom2): continue
        g = c.geom2 if c.geom1 == fl else c.geom1
        if g not in wheel_of: continue
        w = wheel_of[g]; wb = m.jnt_bodyid[spin[w]]
        ax = d.xmat[wb].reshape(3,3) @ m.jnt_axis[spin[w]]; roll = np.cross(ax, (0,0,1)); roll /= np.linalg.norm(roll); axh = np.cross((0,0,1), roll)
        rb = m.geom_bodyid[g]; rj = m.body_jntadr[rb]
        a = d.xmat[rb].reshape(3,3) @ m.jnt_axis[rj]; cen = d.xanchor[rj]
        dv = c.pos - cen; s = dv @ a; rho = np.linalg.norm(dv - s * a)
        mujoco.mj_contactForce(m, d, ci, f6)
        sg = 1 if c.geom1 == fl else -1  # force on the roller
        fr = c.frame.reshape(3,3)
        Fn, Ft = sg * f6[0] * fr[0], sg * (f6[1] * fr[1] + f6[2] * fr[2])
        tn, tt = np.cross(dv, Fn) @ a, np.cross(dv, Ft) @ a
        ttor = sg * f6[3] * (fr[0] @ a) if m.geom_condim[g] >= 4 or c.dim >= 4 else 0.0
        mujoco.mj_jac(m, d, jac, None, c.pos, rb); slip = (jac @ d.qvel) @ axh
        hub = np.zeros(6); mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, wb, hub, 0)
        Fa = -(Ft @ axh) * np.sign(hub[3:] @ axh)  # drag on the wheel, + against the hub's axial motion
        Fs[np.searchsorted(SB, abs(s) * 1e3) - 1] += Fa * dt; Fv[np.searchsorted(VB, abs(slip)) - 1] += Fa * dt
        rows.append((abs(s) * 1e3, rho * 1e3, slip, f6[0], Fa, tt, tn, ttor, d.qvel[m.jnt_dofadr[rj]]))
T = (n - n // 2) * dt
R = np.array(rows)
print(f"{label} k={k} {' '.join(sys.argv[3:])}: {len(R)} contacts. Axial drag on the wheels (sum of 4, N, time mean; per wheel = /4):")
print(f"  total {Fs.sum() / T:.3f} N")
print("  by |s| (mm)", {f"{SB[b]}-{SB[b+1]}": round(Fs[b] / T, 3) for b in range(len(SB)-1)})
print("  by |slip| (m/s)", {f"{VB[b]}-{VB[b+1]}": round(Fv[b] / T, 3) for b in range(len(VB)-1)})
print(f"  |s| mean {R[:,0].mean():.2f} mm, rho mean {R[:,1].mean():.2f} mm (min {R[:,1].min():.2f}), |slip| mean {np.abs(R[:,2]).mean():.3f} m/s")
print(f"  torque about the axle, mean |.| N*m: tangential {np.abs(R[:,5]).mean():.2e}, normal {np.abs(R[:,6]).mean():.2e}, torsional {np.abs(R[:,7]).mean():.2e}")
print(f"  mean normal-force lever |tn|/Fn {np.abs(R[:,6]).sum() / R[:,3].sum() * 1e3:.3f} mm, tangential lever |tt|/|Fa| {np.abs(R[:,5]).sum() / np.abs(R[:,4]).sum() * 1e3:.3f} mm")
