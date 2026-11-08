"""Ticket 04 smoke test: MJX 4.0 on this machine. Run one stage per process, under MemoryMax.

    stage cpu            -- MuJoCo CPU facts per task (nv, nmesh, pairs, ncon/nefc over rollouts)
    stage warp  <nr>     -- impl="warp" put_model + a few batched steps, base model, nr in {2, 7}
    stage jax   <nr> [cut|nomesh]  -- impl="jax"; cut = drop chassis<->chassis, nomesh = also robot meshes

Not part of src/. Reads the project's scene.build through PYTHONPATH=src.
"""

import collections
import resource
import sys
import time

import mujoco
import numpy as np

from mjsoccer import scene as S

SHAPES = {2: (1, 1), 7: (1, 6)}


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def mjx_pairs(m):
    from mujoco.mjx._src import collision_driver as cd

    by_type = collections.Counter()
    for g1, g2, _ in cd.geom_pairs(m):
        by_type[(mujoco.mjtGeom(m.geom_type[g1]).name[7:], mujoco.mjtGeom(m.geom_type[g2]).name[7:])] += 1
    return by_type


def stage_cpu():
    import gymnasium as gym

    import mjsoccer  # noqa: F401  registers the ids

    for nr, (nb, ny) in SHAPES.items():
        m, robots, _ = S.build(nb, ny, 0)
        coll = (m.geom_contype | m.geom_conaffinity) != 0
        mesh_coll = int(np.sum(coll & (m.geom_type == mujoco.mjtGeom.mjGEOM_MESH)))
        pairs = mjx_pairs(m)
        print(f"[{nr} robots] nq={m.nq} nv={m.nv} nu={m.nu} ngeom={m.ngeom} nmesh={m.nmesh} "
              f"nmeshvert={m.nmeshvert} colliding_mesh_geoms={mesh_coll} npair={m.npair} "
              f"nbody={m.nbody}")
        print(f"[{nr} robots] mjx geom_pairs total={sum(pairs.values())} {dict(pairs)}")
        # Which robot asset meshes repeat per robot: nmesh / robots.
        print(f"[{nr} robots] nmesh/robot ~ {m.nmesh / nr:.1f}")

    ids = [k for k in gym.registry if k.startswith("mjsoccer") or k.split("/")[-1].endswith("-v0")]
    ids = [k for k in ids if any(t in k for t in ("GoToBall", "Dribbling", "StaticDefenders",
                                                   "ContestedPossession", "PassEndurance"))]
    for env_id in ids:
        env = gym.make(env_id, tier=0)
        u = env.unwrapped
        rng = np.random.default_rng(0)
        ncon, nefc = [], []
        env.reset(seed=0)
        for t in range(3000):
            _, _, term, trunc, _ = env.step(rng.uniform(-1, 1, env.action_space.shape).astype(np.float32))
            ncon.append(u.data.ncon)
            nefc.append(u.data.nefc)
            if term or trunc:
                env.reset()
        print(f"{env_id}: nv={u.model.nv} ncon mean={np.mean(ncon):.2f} max={max(ncon)} "
              f"nefc mean={np.mean(nefc):.1f} max={max(nefc)}")
        env.close()


def cut_chassis(nb, ny, nomesh=False):
    """Throwaway variant: exclude every robot-body<->robot-body pair (script-only, via MjSpec).

    nomesh: also stop every robot mesh from colliding by contype/conaffinity; the explicit
    ball<->chassis <pair>s survive (a <pair> ignores contype), so only robot<->field and
    robot<->robot go. Smoke only: the real variant needs a hull for robot<->wall.
    """
    spec = _scene_spec(nb, ny)
    prefixes = tuple(f"b{i}_" for i in range(nb)) + tuple(f"y{i}_" for i in range(ny))
    roots = [b.name for b in spec.worldbody.bodies if b.name.startswith(prefixes)]
    if nomesh:
        for r in roots:
            for g in spec.body(r).find_all(mujoco.mjtObj.mjOBJ_GEOM):
                if g.type == mujoco.mjtGeom.mjGEOM_MESH:
                    g.contype = g.conaffinity = 0
    for i, a in enumerate(roots):
        for b in roots[i + 1:]:
            ex = spec.add_exclude()
            ex.bodyname1, ex.bodyname2 = a, b
    return spec.compile(), roots


def _scene_spec(nb, ny):
    """scene.build without compile; mirrors src/mjsoccer/scene.py build() for the script."""
    spec = mujoco.MjSpec.from_file(str(S.MODELS_DIR / S.SCENES[0]))
    robot_geoms = {g.name for g in spec.body("robot").find_all(mujoco.mjtObj.mjOBJ_GEOM)}
    ball_pairs = [(p.geomname2, p.condim, np.array(p.friction), np.array(p.solref), np.array(p.solimp))
                  for p in spec.pairs if p.geomname1 == "ball" and p.geomname2 in robot_geoms]
    spec.delete(spec.body("robot"))
    teams = [f"b{i}_" for i in range(nb)] + [f"y{i}_" for i in range(ny)]
    for i, prefix in enumerate(teams):
        child = mujoco.MjSpec.from_file(str(S.MODELS_DIR / S.ROBOTS[0]))
        frame = spec.worldbody.add_frame()
        frame.pos = [0.0, (i - (len(teams) - 1) / 2) * 0.5, 0.0]
        spec.attach(child, prefix=prefix, frame=frame)
    for prefix in teams:
        for geom, condim, friction, solref, solimp in ball_pairs:
            pair = spec.add_pair()
            pair.geomname1, pair.geomname2 = "ball", prefix + geom
            pair.condim, pair.friction, pair.solref, pair.solimp = condim, friction, solref, solimp
    return spec


def run_mjx(impl, nr, cut, nworld=int(__import__("os").environ.get("NWORLD", 64)), nsteps=24):
    import jax
    import jax.numpy as jp
    from mujoco import mjx

    nb, ny = SHAPES[nr]
    if cut:
        m, roots = cut_chassis(nb, ny, nomesh=cut == "nomesh")
        print(f"{cut} variant: excluded {len(roots)} robot roots {roots[:3]}...")
    else:
        m, _, _ = S.build(nb, ny, 0)
    pairs = mjx_pairs(m)
    print(f"impl={impl} robots={nr} cut={cut} geom_pairs={sum(pairs.values())} "
          f"mesh-mesh={pairs.get(("MESH", "MESH"), 0)} rss0={rss_mb():.0f}MB")

    t0 = time.perf_counter()
    mx = mjx.put_model(m, impl=impl)
    print(f"put_model {time.perf_counter() - t0:.1f}s rss={rss_mb():.0f}MB")

    kw = dict(naconmax=nworld * int(__import__("os").environ.get("NACON_PER_WORLD", 64)), njmax=256) if impl == "warp" else {}
    keys = jax.random.split(jax.random.PRNGKey(0), nworld)

    def init(key):
        d = mjx.make_data(m, impl=impl, **kw)
        return d.replace(ctrl=jax.random.uniform(key, (m.nu,), minval=-0.5, maxval=0.5)
                         * jp.asarray(m.actuator_ctrlrange[:, 1]))

    dx = jax.jit(jax.vmap(init))(keys)

    def roll(mx, d):
        return jax.lax.fori_loop(0, nsteps, lambda _, d: mjx.step(mx, d), d)

    step = jax.jit(jax.vmap(roll, in_axes=(None, 0)))
    t0 = time.perf_counter()
    dx = step(mx, dx)
    jax.block_until_ready(dx)
    tc = time.perf_counter() - t0
    print(f"first call (compile+{nsteps} steps, nworld={nworld}) {tc:.1f}s rss={rss_mb():.0f}MB")
    t0 = time.perf_counter()
    for _ in range(5):
        dx = step(mx, dx)
    jax.block_until_ready(dx)
    dt = time.perf_counter() - t0
    print(f"steady: {5 * nsteps * nworld / dt:,.0f} physics steps/s (nworld={nworld}; smoke, not a benchmark)")
    qpos = np.asarray(dx.qpos)
    print(f"finite={np.isfinite(qpos).all()} ncon/world={np.asarray(dx.ncon).mean() if hasattr(dx, 'ncon') else '?'}")
    impl_d = getattr(dx, "_impl", None)
    for name in ("nacon", "ncollision", "overflow"):
        if impl_d is not None and hasattr(impl_d, name):
            print(f"_impl.{name} = {np.asarray(getattr(impl_d, name))[:4]}")
    try:
        import subprocess
        print(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"],
                             capture_output=True, text=True).stdout.strip(), "GPU used (whole device)")
    except Exception:
        pass
    print(f"peak rss={rss_mb():.0f}MB")


if __name__ == "__main__":
    print("mujoco", mujoco.__version__)
    st = sys.argv[1]
    if st == "cpu":
        stage_cpu()
    else:
        run_mjx(st, int(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else None)
