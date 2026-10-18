"""Ticket 06: lowest point of each robot collision geom above the carpet at rest, and which body owns the bit-1 hulls.

    uv run python .scratch/physics-fidelity/candidate-06/clearance.py > .scratch/physics-fidelity/candidate-06/clearance.txt
"""
import mujoco, numpy as np, gymnasium as gym, mjsoccer
env = gym.make("GoToBall-v0", tier=1); env.reset(seed=0)
u = env.unwrapped; m, d = u.model, u.data
for _ in range(200): mujoco.mj_step(m, d)
rows=[]
for g in range(m.ngeom):
    if m.geom_bodyid[g] == 0 or (m.geom_contype[g]|m.geom_conaffinity[g]) == 0: continue
    n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
    if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
        mid = m.geom_dataid[g]; a, k = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        z = (m.mesh_vert[a:a+k] @ d.geom_xmat[g].reshape(3,3).T + d.geom_xpos[g])[:,2].min()
    else: z = d.geom_xpos[g][2] - m.geom_rbound[g]
    rows.append((z-0.02, n, m.geom_contype[g], m.geom_conaffinity[g], m.geom_type[g]))
rows.sort()
print("total", len(rows))
for r in rows[:12]+[None]+[r for r in rows if r[2]==1][:8]: print(r if r is None else f"{r[0]*1000:6.1f}mm {r[1]} ct={r[2]} ca={r[3]} type={r[4]}")
print("bodies", m.nbody, "ngeom", m.ngeom)
import collections
c = collections.Counter(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) for g in range(m.ngeom) if m.geom_contype[g] == 1 and m.geom_bodyid[g] != 0 and m.geom_bodyid[g] in {r.body for r in u.robots[:1]} | {b for b in range(m.nbody) if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b).startswith("b0_wheel")})
print("bit-1 geoms by body, robot b0:", dict(c))
