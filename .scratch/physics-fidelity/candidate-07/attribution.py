"""Ticket 07: which contacts deliver tipping impulses. Hooks every physics step and attributes the
horizontal (roll/pitch) angular impulse each contact applies about each robot's CoM to its pair class.
Supports (roller|floor, ball|floor) are counted but not attributed.

    MJSOCCER_MODELS=<variant dir> uv run python .scratch/physics-fidelity/candidate-07/attribution.py <scenario> > attr_<scenario>.txt
    scenario: random | robot | wall | ball
"""
import collections, sys, mujoco, numpy as np, gymnasium as gym, mjsoccer
from mjsoccer import scene
SC = sys.argv[1]
from pathlib import Path
BURIED = set(open(Path(__file__).parent / "buried.txt").read().split("tol=0.0mm")[1].split(":", 1)[1].split("\n")[0].split())
CLS = ("floor", "wall", "goal", "ball", "roller", "hub", "body", "mech")
def classify(m):
    c = np.zeros(m.ngeom, int)
    for g in range(m.ngeom):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        c[g] = (0 if n == "floor" else 1 if "walls" in n else 2 if "goalpost" in n else 3 if n == "ball"
                else 4 if "roller" in n else 5 if "Direct" in n else 7 if m.geom_contype[g] == 2 else 6)
    return c
class Rec:
    def __init__(s, m, robots):
        s.m, s.robots, s.c = m, robots, classify(m); s.dt = m.opt.timestep
        s.root = np.array([m.body_rootid[b] for b in range(m.nbody)]); s.rid = {r.body: i for i, r in enumerate(robots)}
        s.L = collections.defaultdict(list); s.J = collections.Counter(); s.n = collections.Counter(); s.tilt = np.zeros(len(robots))
        s.f = np.zeros(6); s.bur = 0
        s.isbur = np.array([(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or '').split('_', 1)[-1] in BURIED for g in range(m.ngeom)])
    def __call__(s, d):
        m = s.m; nc = d.ncon
        for i, r in enumerate(s.robots):
            s.tilt[i] = max(s.tilt[i], np.degrees(np.arccos(np.clip(d.xmat[r.body][8], -1, 1))))
        if not nc: return
        g = d.contact.geom[:nc]; k = np.sort(s.c[g], 1)
        sup = ((k[:, 0] == 0) & ((k[:, 1] == 4) | (k[:, 1] == 3)))
        for a, b in k[sup]: s.n[f"{CLS[a]}|{CLS[b]}"] += 1
        acc = collections.defaultdict(lambda: np.zeros(2))
        for j in np.flatnonzero(~sup):
            con = d.contact[j]; mujoco.mj_contactForce(m, d, j, s.f)
            s.bur += s.isbur[con.geom].any()
            F = con.frame.reshape(3, 3).T @ s.f[:3]; g1, g2 = con.geom
            c1, c2 = s.c[g1], s.c[g2]
            pair = "|".join(sorted((CLS[c1], CLS[c2])))
            ro1, ro2 = s.root[m.geom_bodyid[g1]], s.root[m.geom_bodyid[g2]]
            if c1 in (4, 5, 6, 7) and c2 in (4, 5, 6, 7) and ro1 != ro2: pair = "robot~" + pair
            s.J[pair] += np.linalg.norm(F) * s.dt; s.n[pair] += 1
            for ro, sgn in ((ro1, -1), (ro2, 1)):
                if ro in s.rid:
                    r = con.pos - d.subtree_com[ro]; acc[(ro, pair)] += np.cross(r, sgn * F)[:2]
        for (ro, pair), t in acc.items(): s.L[pair].append(np.linalg.norm(t) * s.dt)
    def report(s, tag):
        for pair in sorted(set(s.n) | set(s.L), key=lambda p: -sum(s.L.get(p, [0]))):
            L = 1000 * np.array(s.L.get(pair, [0.0]))
            print(f"{tag},{pair},contacts={s.n[pair]},J={s.J[pair]:.3f}Ns,steps={len(s.L.get(pair, []))},"
                  f"Ltip_sum={L.sum():.2f},Ltip_p99={np.percentile(L, 99):.4f},Ltip_max={L.max():.4f}mNms")
        print(f"{tag},buried_hull_contacts={s.bur}")
        print(f"{tag},peak_tilt_deg," + ",".join(f"{t:.2f}" for t in s.tilt))
orig = mujoco.mj_step; rec = None
def hooked(m, d, *a):
    orig(m, d, *a); rec(d)
mujoco.mj_step = hooked
def drive_cmd(model, d, r, ang):
    """Wheel command for full chassis drive along body-frame direction `ang`."""
    dirs = []
    for a in r.drive:
        j = model.actuator_trnid[a, 0]; R = d.xmat[r.body].reshape(3, 3)
        ax = R.T @ (d.xmat[model.jnt_bodyid[j]].reshape(3, 3) @ model.jnt_axis[j])
        dirs.append(np.cross([0, 0, -1], ax)[:2])  # empirically: +ctrl along this rolls the chassis forward
    v = np.array(dirs) @ [np.cos(ang), np.sin(ang)]
    return r.drive_range[0, 1] * v / np.abs(v).max()
def run(model, d, r, cmd, n):
    for _ in range(n):
        d.ctrl[:] = 0; d.ctrl[list(r.drive)] = cmd; mujoco.mj_step(model, d)
if SC == "random":
    for env_id in mjsoccer.ENV_IDS:
        env = gym.make(env_id, tier=1); u = env.unwrapped; rec = Rec(u.model, u.robots); tips = 0
        for seed in range(5):
            env.reset(seed=seed); env.action_space.seed(seed)
            while True:
                *_, te, tr, info = env.step(env.action_space.sample())
                if te or tr: tips += bool(info.get("tipped")); break
        rec.report(f"random/{env_id}/tips={tips}")
else:
    model, robots, ball = scene.build(2, 0, 1); d = mujoco.MjData(model); A, B = robots
    def fresh():
        mujoco.mj_resetData(model, d); A.place(d, -5, -3, 0); B.place(d, 5, -3, 0); ball.place(d, 5, 3); mujoco.mj_forward(model, d)
    if SC == "robot":  # A full speed into B (free, idle) at lateral offsets and B yaws
        for off in (0.0, 0.06, 0.12):
            for yaw in (0, 45, 90, 135, 180):
                fresh(); A.place(d, -1.2, 0, 0); B.place(d, 0, off, np.radians(yaw)); mujoco.mj_forward(model, d)
                rec = Rec(model, robots); run(model, d, A, drive_cmd(model, d, A, 0), 1000); rec.report(f"robot/off={off}/yaw={yaw}")
    elif SC == "wall":  # full speed into the +y wall at 15/45/90 deg of incidence, 3 s
        for inc in (15, 45, 90):
            fresh(); A.place(d, -2, 4.2, np.radians(inc)); mujoco.mj_forward(model, d)
            rec = Rec(model, robots); run(model, d, A, drive_cmd(model, d, A, 0), 1500); rec.report(f"wall/inc={inc}")
    elif SC == "ball":  # ball rolled at an idle robot from behind/side, offsets across the wheel line
        for v in (1.0, 2.0, 4.0):
            for side in (90, 180):
                for off in (-0.06, 0.0, 0.06):
                    fresh(); A.place(d, 0, 0, 0); a = np.radians(side); p = np.array([np.cos(a), np.sin(a)]); t = np.array([-p[1], p[0]])
                    x, y = 0.5 * p + off * t; ball.place(d, x, y); mujoco.mj_forward(model, d)
                    d.qvel[ball.qvel_adr:ball.qvel_adr + 2] = -v * p; d.qvel[ball.qvel_adr + 3:ball.qvel_adr + 5] = (v / ball.radius) * np.array([p[1], -p[0]])
                    rec = Rec(model, robots); run(model, d, A, np.zeros(4), 500)
                    bv = d.qvel[ball.qvel_adr:ball.qvel_adr + 3]; rec.report(f"ball/v={v}/side={side}/off={off}")
                    print(f"ball/v={v}/side={side}/off={off},ball_out_v={np.linalg.norm(bv[:2]):.2f},ball_vz={bv[2]:.2f},ball_xy={d.qpos[ball.qpos_adr]:.2f},{d.qpos[ball.qpos_adr+1]:.2f}")
