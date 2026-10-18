"""Ticket 07 (run against the pre-07 models, 209 hulls): exact buried-hull test for the hulls rigid on the `robot` body (bit 1: chassis, bumpers,
motor housings, dribbler structure). Sample each hull's surface (vertices + a barycentric grid on
every face); a hull is buried when every sample lies inside the union of the *other* rigid hulls,
within `tol`. Moving parts (hubs, rollers, dribbler arm, plungers) are not used as occluders.

    uv run python .scratch/physics-fidelity/candidate-07/buried.py > buried.txt
"""
import mujoco, numpy as np
m = mujoco.MjModel.from_xml_path("models/robot_rollers.xml"); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
rb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robot")
name = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)
hulls = [g for g in range(m.ngeom) if m.geom_bodyid[g] == rb and m.geom_contype[g] & 1 and m.geom_type[g] == 7]
H = {}
for g in hulls:
    mid = m.geom_dataid[g]; va, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]; fa, fn = m.mesh_faceadr[mid], m.mesh_facenum[mid]
    R = d.geom_xmat[g].reshape(3, 3); V = m.mesh_vert[va:va+vn] @ R.T + d.geom_xpos[g]; F = m.mesh_face[fa:fa+fn]
    tri = V[F]; n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]); n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-15
    cen = V.mean(0); s = np.sign(((cen - tri[:, 0]) * n).sum(1)); n *= -np.where(s == 0, 1, s)[:, None]  # outward
    off = (n * tri[:, 0]).sum(1)
    k = 6; w = np.array([(i, j, k - i - j) for i in range(k + 1) for j in range(k + 1 - i)]) / k
    S = np.vstack([V, (w[None, :, :, None] * tri[:, None, :, :]).sum(2).reshape(-1, 3)])
    H[g] = (n, off, S)
print(f"rigid bit-1 hulls on robot body: {len(hulls)}")
for tol in (0.0, 0.5e-3):
    buried = []
    for g in hulls:
        S = H[g][2]; inside = np.zeros(len(S), bool)
        for h in hulls:
            if h == g: continue
            n, off, _ = H[h]; inside |= ((S @ n.T - off) <= tol).all(1)
        if inside.all(): buried.append(g)
    print(f"tol={tol*1000:.1f}mm buried={len(buried)}:", " ".join(name(g) for g in buried))
