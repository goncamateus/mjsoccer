"""Tilt angle at which the first non-roller collision vertex reaches the floor plane, pivoting on the rollers."""
import mujoco, numpy as np
from mjsoccer import scene
m, robots, ball = scene.build(1, 0, 1); r = robots[0]; d = mujoco.MjData(m)
mujoco.mj_resetData(m, d); r.place(d, 0, 0, 0); mujoco.mj_forward(m, d)
o = d.xpos[r.body].copy(); roll, chas = [], []
for g in range(m.ngeom):
    if m.geom_bodyid[g] == 0 or not (m.geom_contype[g] | m.geom_conaffinity[g]) or m.geom_type[g] != 7: continue
    mid = m.geom_dataid[g]; a, k = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
    p = m.mesh_vert[a:a+k] @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g] - o
    n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)
    (roll if "roller" in n else chas if m.geom_contype[g] == 1 else []).append(p)
roll, chas = np.vstack(roll), np.vstack(chas)
worst = 90
for phi in np.radians(np.arange(0, 360, 5)):
    ax = np.array([np.cos(phi), np.sin(phi), 0])
    for th in np.radians(np.arange(0, 90, 0.25)):
        R = np.zeros(9); q = np.zeros(4); mujoco.mju_axisAngle2Quat(q, ax, th); mujoco.mju_quat2Mat(R, q); R = R.reshape(3, 3)
        if (chas @ R.T)[:, 2].min() < (roll @ R.T)[:, 2].min():
            worst = min(worst, np.degrees(th)); print(f"phi={np.degrees(phi):5.0f} touch at {np.degrees(th):5.2f} deg"); break
print("min touch tilt deg", worst)
