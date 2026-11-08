"""Ticket 05: raw physics benchmark of MJX-Warp, MJX-JAX and the CPU. One stage per process.

    cpu      <nr> <nbatch> [nthread]          mujoco.rollout on the base model
    contacts                                  CPU max(ncon)/max(nefc): tasks + scripted, forced clusters
    warp     <nr> <nworld> [k=v ...]          base model; k: graph=WARP|WARP_STAGED|NONE nacon=<per world>
                                              njmax=<per world> mode=random|cluster stats=1 cold=1
    jax      <nr> <nworld> [k=v ...]          variant; k: hull=<n-gon> drib=cyl|cap tol=1e-8 x64=1 mode stats
    sdf                                       ball x dribbler-bar contact: MJX-JAX/Warp SDF and capsule vs CPU

Physics only: no task logic. Each decision = 24 physics steps (40 Hz at 1/960 s), ctrl resampled
per decision. Prints one `RESULT {json}` line per run. Not part of src/.
"""

import json
import os
import resource
import subprocess
import sys
import time

import mujoco
import numpy as np

from mjsoccer import scene as S

SHAPES = {2: (1, 1), 7: (1, 6)}
SUB = 24  # physics steps per control step
DECISIONS = int(os.environ.get("DECISIONS", 10))  # control steps per timed call


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def gpu_used_mb():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return int(out.strip() or 0)


def result(**kw):
    print("RESULT " + json.dumps(kw), flush=True)


# ------------------------------------------------------------------ models


def chassis_prism(ngon):
    """Body-frame vertices of an ngon prism wrapping the robot's colliding chassis meshes.

    Support points of the real meshes in ngon directions (so it sits inside the true hull),
    extruded between the meshes' z range. Script-only stand-in for the variant's chassis hull.
    """
    child = mujoco.MjSpec.from_file(str(S.MODELS_DIR / S.ROBOTS[0]))
    m = child.compile()
    body = m.body("robot").id
    pts = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != body or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or not m.geom_contype[g]:
            continue
        mid = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, m.geom_quat[g])
        pts.append(m.geom_pos[g] + v @ R.reshape(3, 3).T)
    P = np.vstack(pts)
    ang = np.linspace(0, 2 * np.pi, ngon, endpoint=False)
    dirs = np.stack([np.cos(ang), np.sin(ang)], 1)
    ring = np.unique(P[np.argmax(P[:, :2] @ dirs.T, axis=0), :2], axis=0)
    z0, z1 = P[:, 2].min(), P[:, 2].max()
    return np.vstack([np.c_[ring, np.full(len(ring), z0)], np.c_[ring, np.full(len(ring), z1)]])


def build_spec(nb, ny, variant=None):
    """scene.build without compile, with an optional script-only variant transform per robot.

    variant = dict(hull=<ngon>, drib='cyl'|'cap'). The variant switches every robot mesh off
    collision, adds one prism hull (robot<->robot, robot<->field, ball<->chassis), and gives the
    ball explicit pairs with the kicker/chipper plates. Masses/inertias stay (asserted).
    """
    spec = mujoco.MjSpec.from_file(str(S.MODELS_DIR / S.SCENES[0]))
    robot_geoms = {g.name for g in spec.body("robot").find_all(mujoco.mjtObj.mjOBJ_GEOM)}
    ball_pairs = [(p.geomname2, p.condim, np.array(p.friction), np.array(p.solref), np.array(p.solimp))
                  for p in spec.pairs if p.geomname1 == "ball" and p.geomname2 in robot_geoms]
    ball = spec.geom("ball")
    spec.delete(spec.body("robot"))
    teams = [f"b{i}_" for i in range(nb)] + [f"y{i}_" for i in range(ny)]
    hull = chassis_prism(variant["hull"]) if variant else None
    plate_pairs = []
    for i, prefix in enumerate(teams):
        child = mujoco.MjSpec.from_file(str(S.MODELS_DIR / S.ROBOTS[0]))
        if variant:
            ref = None
            for bname in ("kicker", "chipper", "robot"):  # plates first: find_all recurses
                for g in child.body(bname).find_all(mujoco.mjtObj.mjOBJ_GEOM):
                    if g.type != mujoco.mjtGeom.mjGEOM_MESH or not (g.contype or g.conaffinity):
                        continue
                    if bname == "robot":
                        ref = ref or g
                    else:  # the plates only ever touch the ball: explicit pairs, mixed like MuJoCo
                        plate_pairs.append((g.name, max(g.condim, ball.condim),
                                            np.maximum(g.friction, ball.friction)[[0, 0, 1, 2, 2]],
                                            (np.array(g.solref) + ball.solref) / 2,
                                            (np.array(g.solimp) + ball.solimp) / 2))
                    g.contype = g.conaffinity = 0
            child.add_mesh(name="chassis_hull", uservert=hull.ravel().tolist())
            h = child.body("robot").add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname="chassis_hull",
                                             name="col_hull", contype=1, conaffinity=1, condim=ref.condim,
                                             friction=ref.friction, solref=ref.solref, solimp=ref.solimp,
                                             group=3, density=0)
            h.mass = 0
            if variant["drib"] == "cap":
                for g in child.body("dribbler").find_all(mujoco.mjtObj.mjOBJ_GEOM):
                    if g.type == mujoco.mjtGeom.mjGEOM_CYLINDER:
                        r, hl = g.size[0], g.size[1]
                        g.type = mujoco.mjtGeom.mjGEOM_CAPSULE
                        g.size = [r, hl - r, 0]
        frame = spec.worldbody.add_frame()
        frame.pos = [0.0, (i - (len(teams) - 1) / 2) * 0.5, 0.0]
        spec.attach(child, prefix=prefix, frame=frame)
    if variant and kwshare():
        # MJX-JAX compiles one collision function per distinct (mesh, mesh) pair and attach copies
        # every mesh per robot: point every robot's colliding meshes at robot 0's copy.
        for g in spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM):
            if g.meshname and g.meshname[:3] in [t for t in teams[1:]] and (g.contype or g.name.split("_", 1)[1] in {p[0] for p in plate_pairs} or g.name.endswith("col_hull")):
                g.meshname = teams[0] + g.meshname[len(teams[0]) if g.meshname.startswith(teams[0]) else 3:]
    for prefix in teams:
        pairs = ball_pairs + (plate_pairs[: len(plate_pairs) // len(teams)] if variant else [])
        for geom, condim, friction, solref, solimp in pairs:
            pair = spec.add_pair()
            pair.geomname1, pair.geomname2 = "ball", prefix + geom
            pair.condim, pair.friction, pair.solref, pair.solimp = condim, friction, solref, solimp
    return spec


def build(nr, variant=None):
    nb, ny = SHAPES[nr]
    base, robots, ball = S.build(nb, ny, 0)
    if not variant:
        return base, robots, ball
    m = build_spec(nb, ny, variant).compile()
    assert m.nq == base.nq and m.nu == base.nu
    np.testing.assert_allclose(m.body_mass, base.body_mass, rtol=1e-6)
    np.testing.assert_allclose(m.body_inertia, base.body_inertia, rtol=1e-6)
    return m, robots, ball


def kwshare():
    return os.environ.get("SHARE", "1") == "1"


def parse_variant(kw):
    return dict(hull=int(kw.get("hull", 16)), drib=kw.get("drib", "cyl"))


# ------------------------------------------------------------------ layouts / scripted ctrl (numpy + jax)


def layout(xp, u, nr, mode):
    """Robot (x, y, yaw) and ball (x, y) from uniforms u[nr*3+2]. xp is numpy or jax.numpy."""
    i = xp.arange(nr)
    if mode == "cluster":  # robots on a 0.30-0.40 m ring round the ball, facing it
        bx, by = -3 + 6 * u[0], -2 + 4 * u[1]
        a = 2 * np.pi * i / nr + 0.4 * (u[2:2 + nr] - 0.5)
        rad = 0.30 + 0.10 * u[2 + nr:2 + 2 * nr]
        x, y = bx + rad * xp.cos(a), by + rad * xp.sin(a)
        yaw = a + np.pi
    else:  # one 1.5 m-wide x slot per robot and one for the ball: nothing starts overlapping
        x = -5 + 1.5 * i + (u[2:2 + nr] - 0.5)
        y = -4 + 8 * u[2 + nr:2 + 2 * nr]
        yaw = 2 * np.pi * u[2 + 2 * nr:2 + 3 * nr]
        bx, by = -5 + 1.5 * nr + (u[0] - 0.5), -4 + 8 * u[1]
    return x, y, yaw, bx, by


def qpos_init(xp, m, robots, ball, u, mode):
    x, y, yaw, bx, by = layout(xp, u, len(robots), mode)
    q = xp.array(m.qpos0)  # a copy: slides measure from qpos0, mutating it zeroes the layout
    for k, r in enumerate(robots):
        a = r.qpos_adr
        vals = (x[k] - r.xy0[0], y[k] - r.xy0[1], yaw[k])
        if xp is np:
            q[a:a + 3] = vals
        else:
            q = q.at[a:a + 3].set(xp.stack(vals))
    if xp is np:
        q[ball.qpos_adr:ball.qpos_adr + 2] = (bx, by)
    else:
        q = q.at[ball.qpos_adr:ball.qpos_adr + 2].set(xp.stack([bx, by]))
    return q


def ctrl_of(xp, m, robots, ball, qpos, u, mode):
    """Random drive commands (mode random) or a P-chase of the ball by every robot (cluster)."""
    c = xp.zeros(m.nu)
    lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
    ball_xy = qpos[ball.qpos_adr:ball.qpos_adr + 2]
    for k, r in enumerate(robots):
        a = r.qpos_adr
        if mode == "cluster":
            pos = xp.stack([qpos[a] + r.xy0[0], qpos[a + 1] + r.xy0[1]])
            e = ball_xy - pos
            v = 3.0 * e / xp.maximum(1.0, xp.linalg.norm(3.0 * e) / 1.5)
            v = v + 0.3 * (u[3 * k:3 * k + 2] - 0.5)
            dyaw = xp.arctan2(e[1], e[0]) - qpos[a + 2]
            w = 4.0 * xp.arctan2(xp.sin(dyaw), xp.cos(dyaw))
            vals = xp.stack([v[0], v[1], w])
        else:
            vals = xp.stack([lo[r.drive[j]] + (hi[r.drive[j]] - lo[r.drive[j]]) * u[3 * k + j] for j in range(3)])
        vals = xp.clip(vals, lo[list(r.drive)], hi[list(r.drive)])
        if xp is np:
            c[list(r.drive)] = vals
            c[r.dribble] = S.DRIBBLER_OMEGA
        else:
            c = c.at[xp.array(r.drive)].set(vals).at[r.dribble].set(S.DRIBBLER_OMEGA)
    return c


# ------------------------------------------------------------------ CPU


def stage_cpu(nr, nbatch, nthread=12):
    from mujoco import rollout

    m, robots, ball = build(nr)
    rng = np.random.default_rng(0)
    d = mujoco.MjData(m)
    spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
    nstate = mujoco.mj_stateSize(m, spec)
    x0 = np.empty((nbatch, nstate))
    ctrl = np.empty((nbatch, DECISIONS * SUB, m.nu))
    for b in range(nbatch):
        mujoco.mj_resetData(m, d)
        d.qpos[:] = qpos_init(np, m, robots, ball, rng.uniform(size=3 * nr + 2), "random")
        mujoco.mj_getState(m, d, x0[b], spec)
        for t in range(DECISIONS):
            ctrl[b, t * SUB:(t + 1) * SUB] = ctrl_of(np, m, robots, ball, d.qpos, rng.uniform(size=3 * nr), "random")
    datas = [mujoco.MjData(m) for _ in range(nthread)]
    rollout.rollout(m, datas, x0, ctrl, persistent_pool=True)  # warm the pool
    t0 = time.perf_counter()
    reps = 0
    while time.perf_counter() - t0 < 5 or reps < 2:
        st, _ = rollout.rollout(m, datas, x0, ctrl, persistent_pool=True)
        reps += 1
    dt = time.perf_counter() - t0
    sps = reps * nbatch * DECISIONS * SUB / dt
    finite = bool(np.isfinite(st).all())
    result(engine="cpu", nr=nr, nworld=nbatch, nthread=nthread, sps=sps, dps=sps / SUB, finite=finite)


def stage_contacts():
    """Worst-case ncon/nefc on the CPU base model: the six tasks under the scripted chase, and
    forced clusters (every robot chasing the ball from a 0.3 m ring)."""
    import gymnasium as gym

    import mjsoccer  # noqa: F401
    from mjsoccer.tasks import goto_ball_policy

    for env_id in ("GoToBall-v0", "GoToBallShoot-v0", "Dribbling-v0", "StaticDefenders-v0",
                   "ContestedPossession-v0", "PassEndurance-v0"):
        env = gym.make(env_id, tier=0)
        u = env.unwrapped
        obs, _ = env.reset(seed=0)
        ncon, nefc = [], []
        for t in range(3000):
            a = np.zeros(env.action_space.shape, np.float32)
            a[:3] = goto_ball_policy(obs)
            a[-1] = 1.0  # dribbler on
            obs, _, term, trunc, _ = env.step(a)
            ncon.append(u.data.ncon)
            nefc.append(u.data.nefc)
            if term or trunc:
                obs, _ = env.reset()
        result(stage="contacts", src=env_id, nv=u.model.nv, ncon_mean=float(np.mean(ncon)), ncon_max=int(max(ncon)),
               nefc_mean=float(np.mean(nefc)), nefc_max=int(max(nefc)))
        env.close()
    for nr in (2, 7):
        m, robots, ball = build(nr)
        d = mujoco.MjData(m)
        rng = np.random.default_rng(0)
        ncon, nefc = [], []
        for ep in range(50):
            mujoco.mj_resetData(m, d)
            d.qpos[:] = qpos_init(np, m, robots, ball, rng.uniform(size=3 * nr + 2), "cluster")
            for t in range(40):  # 1 s per episode
                d.ctrl[:] = ctrl_of(np, m, robots, ball, d.qpos, rng.uniform(size=3 * nr), "cluster")
                for _ in range(SUB):
                    mujoco.mj_step(m, d)
                    ncon.append(d.ncon)
                    nefc.append(d.nefc)
        result(stage="contacts", src=f"cluster{nr}", nv=m.nv, ncon_mean=float(np.mean(ncon)), ncon_max=int(max(ncon)),
               nefc_mean=float(np.mean(nefc)), nefc_max=int(max(nefc)))


# ------------------------------------------------------------------ MJX


def stage_mjx(impl, nr, nworld, kw):
    if kw.get("x64") == "1":
        os.environ["JAX_ENABLE_X64"] = "1"
    if kw.get("cold") == "1":
        import warp as wp
        wp.config.kernel_cache_dir = os.path.expanduser(f"~/.cache/mjsoccer-bench/warp-cold-{os.getpid()}")
    import jax
    import jax.numpy as jp
    from mujoco import mjx

    mode = kw.get("mode", "random")
    stats = kw.get("stats") == "1"
    variant = parse_variant(kw) if impl == "jax" else None
    m, robots, ball = build(nr, variant)
    if "tol" in kw:
        m.opt.tolerance = float(kw["tol"])
    if kw.get("solver") == "CG":  # numerics, not physical description (map constraint 7)
        m.opt.solver = mujoco.mjtSolver.mjSOL_CG
    cfg = dict(engine=impl, nr=nr, nworld=nworld, mode=mode, tol=float(m.opt.tolerance),
               solver=mujoco.mjtSolver(m.opt.solver).name[6:],
               x64=jax.config.read("jax_enable_x64"), decisions=DECISIONS)
    put = {}
    make = {}
    if impl == "warp":
        from mujoco.mjx.warp import types as wt
        cfg["graph"] = kw.get("graph", "WARP")
        put["graph_mode"] = getattr(wt.GraphMode, cfg["graph"])
        cfg["nacon_pw"], cfg["njmax"] = int(kw.get("nacon", 128)), int(kw.get("njmax", 64))
        make = dict(naconmax=nworld * cfg["nacon_pw"], njmax=cfg["njmax"])
    else:
        cfg.update(variant)
    cfg["ngeom_pairs"] = mjx_npairs(m)
    gpu0 = gpu_used_mb()
    t0 = time.perf_counter()
    mx = mjx.put_model(m, impl=impl, **put)
    cfg["put_s"] = time.perf_counter() - t0

    nu3 = 3 * nr

    def init(key):
        u = jax.random.uniform(key, (3 * nr + 2,))
        d = mjx.make_data(m, impl=impl, **make)
        return d.replace(qpos=qpos_init(jp, m, robots, ball, u, mode))

    keys = jax.random.split(jax.random.PRNGKey(0), nworld)
    dx = jax.jit(jax.vmap(init))(keys)
    print(f"put {cfg['put_s']:.1f}s, data ready; compiling", flush=True)

    def wstats(d):
        out = {"niter": d._impl.solver_niter}
        if impl == "warp":
            for f in ("nacon", "ncollision", "nefc", "overflow"):
                out[f] = getattr(d._impl, f)
        return out

    def decide(carry, key):
        d, st = carry
        u = jax.random.uniform(key, (nworld, nu3))
        ctrl = jax.vmap(lambda q, uu: ctrl_of(jp, m, robots, ball, q, uu, mode))(d.qpos, u)
        d = d.replace(ctrl=ctrl)

        def phys(_, c):
            d, st = c
            d = jax.vmap(mjx.step, in_axes=(None, 0))(mx, d)
            if stats:
                new = wstats(d)
                st = {k: jp.maximum(st[k], jp.max(new[k])) for k in new} | {"niter_sum": st["niter_sum"] + jp.sum(new["niter"])}
            return d, st

        d, st = jax.lax.fori_loop(0, SUB, phys, (d, st))
        return (d, st), None

    def roll(d, key):
        st = {}
        if stats:
            st = {k: jp.max(v) * 0 for k, v in wstats(d).items()} | {"niter_sum": jp.zeros((), jp.float32)}
        (d, st), _ = jax.lax.scan(decide, (d, st), jax.random.split(key, DECISIONS))
        return d, st

    roll_j = jax.jit(roll, compiler_options={"xla_gpu_graph_min_graph_size": 1})
    key = jax.random.PRNGKey(1)
    try:
        t0 = time.perf_counter()
        comp = roll_j.lower(dx, key).compile()
        cfg["compile_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        dx, st = comp(dx, key)
        jax.block_until_ready(dx)
        cfg["first_call_s"] = time.perf_counter() - t0
        t0, reps = time.perf_counter(), 0
        while reps < 1 or (time.perf_counter() - t0 < 5 and reps < 20):
            key, k = jax.random.split(key)
            dx, st = comp(dx, k)
            jax.block_until_ready(dx)
            reps += 1
        dt = time.perf_counter() - t0
    except Exception as e:  # GPU OOM and friends: record, don't hide
        msg = str(e).splitlines()[0][:300]
        result(**cfg, error=msg, rss_mb=rss_mb())
        return
    sps = reps * nworld * DECISIONS * SUB / dt
    cfg.update(sps=sps, dps=sps / SUB, reps=reps, finite=bool(np.isfinite(np.asarray(dx.qpos)).all()),
               rss_mb=rss_mb(), gpu_mb=gpu_used_mb() - gpu0)
    try:
        cfg["jax_peak_mb"] = jax.devices()[0].memory_stats()["peak_bytes_in_use"] / 2**20
    except Exception:
        pass
    if stats:
        st = {k: float(v) for k, v in st.items()}
        st["niter_mean"] = st.pop("niter_sum") / (nworld * DECISIONS * SUB)  # last call only
        cfg["stats"] = st
    result(**cfg)


def mjx_npairs(m):
    from mujoco.mjx._src import collision_driver as cd

    return sum(1 for _ in cd.geom_pairs(m))


# ------------------------------------------------------------------ SDF precision


def stage_sdf(n=512):
    """Ball against the dribbler bar: MJX-JAX (sphere_cylinder SDF), MJX-Warp, and the capsule
    stand-in, each against the CPU's cylinder contact on the base model."""
    import jax
    import jax.numpy as jp
    from mujoco import mjx

    base, robots, ball = build(2)
    r = robots[0]
    bar = base.geom(r.prefix + "r_col_barDribbler").id
    d = mujoco.MjData(base)
    mujoco.mj_forward(base, d)
    c, R = d.geom_xpos[bar].copy(), d.geom_xmat[bar].reshape(3, 3).copy()
    rad, hl = base.geom_size[bar][:2]
    rb = ball.radius
    rng = np.random.default_rng(0)
    s = rng.uniform(-1.15 * hl, 1.15 * hl, n)  # past the ends too: caps differ cylinder vs capsule
    phi = rng.uniform(-np.pi / 2, np.pi / 2, n)  # the half facing out of the robot
    delta = rng.uniform(-2e-3, 0.5e-3, n)  # penetration .. separation
    # bar local axis is z; local +x points... find the outward side: away from the robot centre
    out = c - d.xpos[r.body]
    out_l = R.T @ out
    base_ang = np.arctan2(out_l[1], out_l[0])
    a = base_ang + phi
    rho = rad + rb + delta
    local = np.stack([rho * np.cos(a), rho * np.sin(a), s], 1)
    centres = c + local @ R.T

    def cpu_contacts(m, gname):
        g = m.geom(r.prefix + gname).id
        dd = mujoco.MjData(m)
        out = np.full((n, 7), np.nan)
        for k in range(n):
            mujoco.mj_resetData(m, dd)
            dd.qpos[ball.qpos_adr:ball.qpos_adr + 3] = centres[k]
            mujoco.mj_forward(m, dd)
            for ci in range(dd.ncon):
                ct = dd.contact[ci]
                if {ct.geom1, ct.geom2} == {ball.geom, g}:
                    sign = 1 if ct.geom1 == ball.geom else -1
                    out[k] = np.r_[ct.dist, ct.pos, sign * ct.frame[:3]]
        return out

    def mjx_contacts(m, impl, gname):
        g = m.geom(r.prefix + gname).id
        mx = mjx.put_model(m, impl=impl)
        make = dict(naconmax=n * 48, njmax=64) if impl == "warp" else {}

        def one(p):
            dd = mjx.make_data(m, impl=impl, **make)
            dd = dd.replace(qpos=jp.asarray(m.qpos0).at[ball.qpos_adr:ball.qpos_adr + 3].set(p))
            return dd

        dx = jax.jit(jax.vmap(one))(jp.asarray(centres))
        dx = jax.jit(jax.vmap(mjx.forward, in_axes=(None, 0)))(mx, dx)
        out = np.full((n, 7), np.nan)
        if impl == "jax":
            con = dx.contact
            geom, dist, pos, frame = (np.asarray(x) for x in (con.geom, con.dist, con.pos, con.frame))
            for i in np.flatnonzero([set(gg) == {ball.geom, g} for gg in geom[0]]):
                hit = dist[:, i] < m.geom_margin[g] + m.geom_margin[ball.geom]  # MJX-JAX keeps every pair slot
                sign = 1 if geom[0, i, 0] == ball.geom else -1
                out[hit] = np.c_[dist[hit, i], pos[hit, i], sign * frame[hit, i, 0]]
        else:  # Warp: contacts are one flat buffer over all worlds, tagged by worldid
            I = dx._impl
            na = int(np.asarray(I.nacon).ravel()[0])
            geom = np.asarray(I.contact__geom).reshape(-1, 2)[:na]
            wid = np.asarray(I.contact__worldid).ravel()[:na]
            dist = np.asarray(I.contact__dist).ravel()[:na]
            pos = np.asarray(I.contact__pos).reshape(-1, 3)[:na]
            frame = np.asarray(I.contact__frame).reshape(-1, 3, 3)[:na]
            for i in np.flatnonzero([set(gg) == {ball.geom, g} for gg in geom]):
                sign = 1 if geom[i, 0] == ball.geom else -1
                out[wid[i]] = np.r_[dist[i], pos[i], sign * frame[i, 0]]
        return out

    ref = cpu_contacts(base, "r_col_barDribbler")
    capm = build_spec(*SHAPES[2], dict(hull=16, drib="cap")).compile()
    cylm = build_spec(*SHAPES[2], dict(hull=16, drib="cyl")).compile()
    rows = {"cpu_capsule": cpu_contacts(capm, "r_col_barDribbler"),
            "mjxjax_cyl_sdf": mjx_contacts(cylm, "jax", "r_col_barDribbler"),
            "mjxjax_capsule": mjx_contacts(capm, "jax", "r_col_barDribbler"),
            "mjxwarp_cyl_base": mjx_contacts(base, "warp", "r_col_barDribbler")}
    end = np.abs(s) > hl
    for name, got in rows.items():
        if got is None:
            continue
        for part, sel in (("side", ~end), ("ends", end)):
            both = sel & np.isfinite(ref[:, 0]) & np.isfinite(got[:, 0])
            miss = int(np.sum(sel & (np.isfinite(ref[:, 0]) != np.isfinite(got[:, 0]))))
            if not both.any():
                result(stage="sdf", cmp=name, part=part, n=0, detect_mismatch=miss)
                continue
            dd = np.abs(got[both, 0] - ref[both, 0])
            dp = np.linalg.norm(got[both, 1:4] - ref[both, 1:4], axis=1)
            cosn = np.clip(np.sum(got[both, 4:7] * ref[both, 4:7], 1), -1, 1)
            dn = np.degrees(np.arccos(cosn))
            result(stage="sdf", cmp=name, part=part, n=int(both.sum()), detect_mismatch=miss,
                   dist_err_mm=dict(mean=1e3 * dd.mean(), p99=1e3 * np.quantile(dd, .99), max=1e3 * dd.max()),
                   pos_err_mm=dict(mean=1e3 * dp.mean(), p99=1e3 * np.quantile(dp, .99), max=1e3 * dp.max()),
                   normal_err_deg=dict(mean=dn.mean(), p99=np.quantile(dn, .99), max=dn.max()))


if __name__ == "__main__":
    import faulthandler
    faulthandler.dump_traceback_later(int(os.environ.get("TB_EVERY", 300)), repeat=True)
    print("mujoco", mujoco.__version__, flush=True)
    st, args = sys.argv[1], sys.argv[2:]
    kw = dict(a.split("=", 1) for a in args if "=" in a)
    pos = [a for a in args if "=" not in a]
    if st == "cpu":
        stage_cpu(int(pos[0]), int(pos[1]), int(pos[2]) if len(pos) > 2 else 12)
    elif st == "contacts":
        stage_contacts()
    elif st == "sdf":
        stage_sdf()
    else:
        stage_mjx(st, int(pos[0]), int(pos[1]), kw)
