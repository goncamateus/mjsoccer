"""Ticket 15: kick exit speed on MJX-Warp vs the CPU, and the cost of the firing mechanisms.

    cpu                      CPU (float64) exit speed per (plunger, force) for each schedule K
    warp  [K ...]            same on MJX-Warp, per-world opt.timestep schedule (in_axes)
    cost  <nworld> <K> <L>   decisions/s, 2-robot bench scene, every world on schedule (K, L)

Schedule K for the firing control step: K physics steps at dt/4, then 24 - K/4 at dt (K = 0 is no
substep, K = 96 is today's whole step at dt/4). Non-firing worlds run 24 at dt and are frozen
(Data.where) for the rest of the L = K + 24 - K/4 iterations. Start: Dribbling-v0 tier 0 after
reset(seed=0) + 30 zero-action steps (calibrate.py's settle), robot 0 fires, ctrl otherwise 0.
Exit speed = peak 3-D ball speed at control-step boundaries over 30 steps. Not part of src/.
"""
import json, os, sys, time
import numpy as np, mujoco
import gymnasium as gym
import mjsoccer  # noqa: F401
from mjsoccer import scene as S

SUB, AFTER, SETTLE = 24, 30, 30
FORCES = np.arange(0.0, 501.0, 25.0)


def result(**kw):
    print("RESULT " + json.dumps(kw), flush=True)


def snapshot():
    env = gym.make("Dribbling-v0", tier=0).unwrapped
    env.reset(seed=0)
    a = np.zeros(env.action_space.shape, np.float32)
    for _ in range(SETTLE):
        env.step(a)
    r = env.robots[0]
    return env.model, env.data, (r.kick, r.chip), env.ball


ZERO = bool(int(os.environ.get("ZERO", 0)))  # freeze idle worlds with timestep 0, no Data.where
SHORT = bool(int(os.environ.get("SHORT", 0)))  # firing step = K steps at dt/4 only (lasts K*dt/4)


def rest(K):
    return 0 if SHORT else SUB - K // 4


def sched(K):
    return max(K + rest(K), SUB)


def cpu(Ks):
    m, d0, acts, ball = snapshot()
    dt = m.opt.timestep
    for K in Ks:
        rows = {}
        for j, act in enumerate(acts):
            out = []
            for F in FORCES:
                d = mujoco.MjData(m)
                d.qpos[:], d.qvel[:], d.act[:], d.time = d0.qpos, d0.qvel, d0.act, d0.time
                d.ctrl[:] = 0
                d.ctrl[act] = F
                m.opt.timestep = dt / 4
                for _ in range(K):
                    mujoco.mj_step(m, d)
                m.opt.timestep = dt
                for _ in range(rest(K)):
                    mujoco.mj_step(m, d)
                best = np.linalg.norm(ball.vel(d))
                d.ctrl[act] = 0
                for _ in range(AFTER):
                    for _ in range(SUB):
                        mujoco.mj_step(m, d)
                    best = max(best, np.linalg.norm(ball.vel(d)))
                out.append(round(float(best), 6))
            rows["kick chip".split()[j]] = out
        result(stage="cpu", K=K, short=SHORT, **rows)


def warp(Ks):
    import jax, jax.numpy as jp
    from mujoco import mjx
    m, d0, acts, ball = snapshot()
    dt = float(m.opt.timestep)
    nF = len(FORCES)
    nworld = 2 * nF + 2  # every (plunger, force) fires; 2 extra worlds never fire (masking check)
    mx = mjx.put_model(m, impl="warp")
    fire = np.r_[np.ones(2 * nF, bool), False, False]
    ctrl = np.zeros((nworld, m.nu), np.float32)
    for j, act in enumerate(acts):
        ctrl[j * nF:(j + 1) * nF, act] = FORCES
    ctrl, fire = jp.asarray(ctrl), jp.asarray(fire)
    vadr = m.jnt_dofadr[m.body_jntadr[ball.body]]

    def one(_):
        d = mjx.make_data(m, impl="warp", naconmax=nworld * 128, njmax=256)
        return d.replace(qpos=jp.asarray(d0.qpos), qvel=jp.asarray(d0.qvel), act=jp.asarray(d0.act))

    in_ax = jax.tree.map(lambda _: None, mx)
    in_ax = in_ax.replace(opt=in_ax.opt.replace(timestep=0))
    step_v = jax.vmap(mjx.step, in_axes=(in_ax, 0))
    speed = lambda d: jp.linalg.norm(d.qvel[:, vadr:vadr + 3], axis=1)

    for K in Ks:
        L = sched(K)

        def run(d):
            d = d.replace(ctrl=ctrl)
            def body(i, d):
                ts = jp.where(fire & (i < K), dt / 4, dt)
                active = jp.where(fire, i < K + rest(K), i < SUB)
                if ZERO:
                    ts = jp.where(active, ts, 0.0)
                mb = mx.replace(opt=mx.opt.replace(timestep=ts.astype(jp.float32)))
                nd = step_v(mb, d)
                if ZERO:
                    return nd
                return jax.vmap(lambda a, n, o: o.where(a, n))(active, nd, d)  # where(m, x): x where m
            d = jax.lax.fori_loop(0, L, body, d)
            best = speed(d)
            d = d.replace(ctrl=jp.zeros_like(ctrl))
            def after(_, c):
                d, best = c
                d = jax.lax.fori_loop(0, SUB, lambda _, d: jax.vmap(mjx.step, in_axes=(None, 0))(mx, d), d)
                return d, jp.maximum(best, speed(d))
            d, best = jax.lax.fori_loop(0, AFTER, after, (d, best))
            return best, d.time, d._impl.overflow
        dx = jax.jit(jax.vmap(one))(jp.arange(nworld))
        t0 = time.perf_counter()
        best, tm, ovf = jax.jit(run)(dx)
        best = np.asarray(best)
        result(stage="warp", K=K, L=L, short=SHORT, zero=ZERO, s=round(time.perf_counter() - t0, 1),
               kick=np.round(best[:nF], 6).tolist(), chip=np.round(best[nF:2 * nF], 6).tolist(),
               idle_speed=np.round(best[2 * nF:], 4).tolist(),
               time_spread=float(np.ptp(np.asarray(tm))), overflow=int(np.max(np.asarray(ovf))))


def cost(nworld, K, L, batched):
    """Decisions/s on the 2-robot bench scene: every decision runs L iterations on schedule K for
    the worlds in `fire` (every 20th world), masked; batched=0 uses one shared timestep (no in_axes)."""
    import jax, jax.numpy as jp
    from mujoco import mjx
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "05-bench"))
    import bench
    m, robots, ball = bench.build(2, None)
    dt = float(m.opt.timestep)
    mx = mjx.put_model(m, impl="warp")
    make = dict(naconmax=nworld * 128, njmax=64)
    keys = jax.random.split(jax.random.PRNGKey(0), nworld)

    def init(key):
        u = jax.random.uniform(key, (3 * 2 + 2,))
        return mjx.make_data(m, impl="warp", **make).replace(qpos=bench.qpos_init(jp, m, robots, ball, u, "random"))
    dx = jax.jit(jax.vmap(init))(keys)
    fire = jp.arange(nworld) % 20 == 0
    in_ax = jax.tree.map(lambda _: None, mx)
    in_ax = in_ax.replace(opt=in_ax.opt.replace(timestep=0))
    step_b = jax.vmap(mjx.step, in_axes=(in_ax, 0))
    step_s = jax.vmap(mjx.step, in_axes=(None, 0))

    def decide(d, key):
        u = jax.random.uniform(key, (nworld, 6))
        d = d.replace(ctrl=jax.vmap(lambda q, uu: bench.ctrl_of(jp, m, robots, ball, q, uu, "random"))(d.qpos, u))
        def body(i, d):
            if not batched:
                return step_s(mx, d)
            if batched == 2:  # per-world timestep, no mask
                ts = jp.where(fire & (i < K), dt / 4, dt).astype(jp.float32)
                return step_b(mx.replace(opt=mx.opt.replace(timestep=ts)), d)
            if batched == 3:  # shared timestep, mask only
                nd = step_s(mx, d)
                return jax.vmap(lambda a, n, o: o.where(a, n))(jp.where(fire, i < K + rest(K), i < SUB), nd, d)
            if batched == 5:  # per-world timestep, idle worlds frozen by timestep 0
                active = jp.where(fire, i < K + rest(K), i < SUB)
                ts = jp.where(active, jp.where(fire & (i < K), dt / 4, dt), 0.0).astype(jp.float32)
                return step_b(mx.replace(opt=mx.opt.replace(timestep=ts)), d)
            if batched == 4:  # shared dt/4 for the whole batch (substep everything)
                return step_s(mx.replace(opt=mx.opt.replace(timestep=jp.float32(dt / 4))), d)
            ts = jp.where(fire & (i < K), dt / 4, dt).astype(jp.float32)
            nd = step_b(mx.replace(opt=mx.opt.replace(timestep=ts)), d)
            active = jp.where(fire, i < K + rest(K), i < SUB)
            return jax.vmap(lambda a, n, o: o.where(a, n))(active, nd, d)
        return jax.lax.fori_loop(0, L, body, d), None

    D = 10
    roll = jax.jit(lambda d, k: jax.lax.scan(decide, d, jax.random.split(k, D))[0])
    k = jax.random.PRNGKey(1)
    jax.block_until_ready(roll(dx, k))
    t0, reps = time.perf_counter(), 0
    while time.perf_counter() - t0 < 5 or reps < 2:
        k, kk = jax.random.split(k)
        dx = roll(dx, kk)
        jax.block_until_ready(dx)
        reps += 1
    dps = nworld * D * reps / (time.perf_counter() - t0)
    result(stage="cost", nworld=nworld, K=K, L=L, batched=batched, short=SHORT, dps=round(dps), sps=round(dps * SUB),
           overflow=int(np.max(np.asarray(dx._impl.overflow))))


if __name__ == "__main__":
    st, args = sys.argv[1], [int(a) for a in sys.argv[2:]]
    if st == "cpu":
        cpu(args or [0, 4, 8, 16, 32, 96])
    elif st == "warp":
        warp(args or [0, 8, 96])
    elif st == "cost":
        cost(*args)
