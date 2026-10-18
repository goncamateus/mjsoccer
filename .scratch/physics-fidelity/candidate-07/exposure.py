"""Ticket 07 (run against the pre-07 models, 209 hulls): which collision hulls form the robot's outer surface. Rays from a 0.3 m sphere around
the robot toward uniform points in its collision bbox; a hull that is never the first hit is buried
inside the others (CAD noise). Also reports the share of first hits per class.

    uv run python .scratch/physics-fidelity/candidate-07/exposure.py [N] [--extended] > exposure*.txt
"""
import collections, mujoco, numpy as np
import sys
N = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
m = mujoco.MjModel.from_xml_path("models/robot_rollers.xml"); d = mujoco.MjData(m)
if "--extended" in sys.argv:  # plungers out 30 mm, dribbler arm at its 0.14 rad stop
    for j, q in (("kicker", 0.03), ("chipper", 0.03), ("dribbler_arm", 0.14)):
        jid = m.body_jntadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, j)]; d.qpos[m.jnt_qposadr[jid]] = q
mujoco.mj_forward(m, d)
name = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)
col = [g for g in range(m.ngeom) if m.geom_contype[g] | m.geom_conaffinity[g]]
pts = []
for g in col:
    if m.geom_type[g] != 7: continue
    mid = m.geom_dataid[g]; a, k = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
    pts.append(m.mesh_vert[a:a+k] @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])
pts = np.vstack(pts); lo, hi = pts.min(0), pts.max(0); c = (lo + hi) / 2
rng = np.random.default_rng(N)
u = rng.normal(size=(N, 3)); u /= np.linalg.norm(u, axis=1, keepdims=True)
src = c + 0.3 * u; dst = rng.uniform(lo, hi, size=(N, 3))
grp = np.array([0, 0, 0, 1, 0, 0], np.uint8); gid = np.zeros(1, np.int32)
hits = collections.Counter(); miss = 0
for s, t in zip(src, dst):
    v = t - s; v /= np.linalg.norm(v)
    if mujoco.mj_ray(m, d, s, v, grp, 1, -1, gid) < 0: miss += 1
    else: hits[int(gid[0])] += 1
def cls(n):
    if "roller" in n: return "roller"
    if "Direct" in n: return "hub"
    if m.geom_contype[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)] == 2: return "mech(bit2)"
    return "body(bit1)"
byc = collections.Counter(); tot = collections.Counter(); buried = []
for g in col:
    k = cls(name(g)); tot[k] += 1; byc[k] += hits[g]
    if hits[g] == 0: buried.append(name(g))
print(f"rays={N} miss={miss} hit={N-miss}")
for k in tot: print(f"class={k} geoms={tot[k]} first_hit_share={byc[k]/(N-miss):.3f} buried={sum(cls(b)==k for b in buried)}")
print("buried:", " ".join(buried))
print("rare (<10 hits):", " ".join(f"{name(g)}={hits[g]}" for g in col if 0 < hits[g] < 10))
