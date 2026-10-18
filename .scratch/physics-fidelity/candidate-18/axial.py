"""Ticket 18: per-wheel floor force split along roll and along the axle, forward command k, second 1-2.

    MJSOCCER_MODELS=<variant dir> uv run python .scratch/physics-fidelity/candidate-18/axial.py <k> [arm|tors|damp0|imp|pyr joined by +]

Mutations act on the compiled model: roller armature 1e-9, no torsional friction, roller damping 0,
impratio 10, pyramidal cone.
"""
import sys
from types import SimpleNamespace
import mujoco, numpy as np
from mjsoccer import scene
from mjsoccer.tasks import wheel_command
k = float(sys.argv[1])
m, robots, ball = scene.build(1, 0, 1); r = robots[0]; d = mujoco.MjData(m); dt = m.opt.timestep
fl = m.geom("floor").id
mut = sys.argv[2] if len(sys.argv) > 2 else ""
rdofs = [m.jnt_dofadr[m.body_jntadr[b]] for b in range(m.nbody) if "roller" in (m.body(b).name or "")]
rgeoms = [g for g in range(m.ngeom) if "col_wheel" in (m.geom(g).name or "")]
if "arm" in mut: m.dof_armature[rdofs] = 1e-9
if "damp0" in mut: m.dof_damping[rdofs] = 0
if "tors" in mut: m.geom_friction[rgeoms, 1] = 0; m.geom_friction[fl, 1] = 0
if "imp" in mut: m.opt.impratio = 10
if "pyr" in mut: m.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
print("mutation:", mut or "none")
spin = [m.actuator_trnid[a, 0] for a in r.drive]
mujoco.mj_resetData(m, d); r.place(d, -5.5, 0, 0); ball.place(d, 5.5, 4); mujoco.mj_forward(m, d)
for _ in range(round(0.3/dt)): mujoco.mj_step(m, d)
print("wheel axis z-comp (camber):", [round(float((d.xmat[m.jnt_bodyid[j]].reshape(3,3) @ m.jnt_axis[j])[2]), 4) for j in spin])
cmd = k * wheel_command(SimpleNamespace(robots=robots, model=m, data=d), 0, 10, 0, 0) * r.drive_range[:, 1]
n = round(2/dt); f6 = np.zeros(6)
Fr, Fa, va, vr, off = np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(4), np.zeros((4, 3)); cnt = np.zeros(4)
for i in range(n):
    d.ctrl[list(r.drive)] = cmd; mujoco.mj_step(m, d)
    if i < n // 2: continue
    for w, j in enumerate(spin):
        b = m.jnt_bodyid[j]; ax = d.xmat[b].reshape(3,3) @ m.jnt_axis[j]
        roll = np.cross(ax, (0,0,1)); roll /= np.linalg.norm(roll); axh = np.cross((0,0,1), roll)
        vel = np.zeros(6); mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, b, vel, 0)
        va[w] += axh @ vel[3:]; vr[w] += roll @ vel[3:]
        for ci, c in enumerate(d.contact[:d.ncon]):
            if fl not in (c.geom1, c.geom2): continue
            g = c.geom2 if c.geom1 == fl else c.geom1
            if f"col_wheel{w+1}" not in (m.geom(g).name or ""): continue
            mujoco.mj_contactForce(m, d, ci, f6)
            F = c.frame.reshape(3,3).T @ f6[:3] * (1 if c.geom1 == fl else -1)
            Fr[w] += roll @ F; Fa[w] += axh @ F; off[w] += (c.pos - d.xpos[b]); cnt[w] += 1
N = n - n // 2
print(f"k={k}: per wheel mean floor force along roll {(Fr/N).round(3)} N, along axle {(Fa/N).round(3)} N")
print(f"  hub v along roll {(vr/N).round(3)}, along axle {(va/N).round(3)} m/s")
print("  mean contact point - hub (world, mm):", (1e3*off/cnt[:,None]).round(2).tolist())
