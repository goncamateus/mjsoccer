"""Ticket 10 Q3: is 60 deg the recoverable/irrecoverable boundary?

Tier 1, current models. Tilt the robot by theta about a horizontal axis at azimuth phi,
set it down with its lowest collision vertex 0.5 mm above the carpet, zero velocity and
zero ctrl, step 2 s. Recovered = final tilt < 10 deg. Prints, per phi, the smallest
theta that does NOT recover (the static recovery boundary).

    uv run python .scratch/physics-fidelity/candidate-10/recover.py
"""
import mujoco
import numpy as np

from mjsoccer import scene

m, robots, _ = scene.build(1, 0, 1)
r, d = robots[0], mujoco.MjData(m)
carpet = m.geom_pos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")][2]
geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] != 0 and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
         and (m.geom_contype[g] | m.geom_conaffinity[g])]


def lowest():
    z = np.inf
    for g in geoms:
        mid = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        z = min(z, (v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])[:, 2].min())
    return z


def final_tilt(phi, theta):
    mujoco.mj_resetData(m, d)
    r.place(d, 0, 0, 0)
    a = r.qpos_adr
    q = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q, np.array([np.cos(phi), np.sin(phi), 0.0]), theta)
    d.qpos[a + 3: a + 7] = q
    mujoco.mj_forward(m, d)
    d.qpos[a + 2] += carpet + 0.0005 - lowest()
    for _ in range(round(2.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    return np.degrees(np.arccos(np.clip(d.xmat[r.body][8], -1, 1)))


worst = 90.0
for phi in range(0, 360, 15):
    for th in range(5, 90, 1):
        if final_tilt(np.radians(phi), np.radians(th)) > 10:
            print(f"phi={phi:3d} first unrecovered at {th:2d} deg")
            worst = min(worst, th)
            break
    else:
        print(f"phi={phi:3d} recovers up to 89 deg")
print("min unrecovered tilt deg", worst)
