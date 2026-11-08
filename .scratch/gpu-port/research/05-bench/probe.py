"""Run time of one vmapped mjx.<fn> (impl=jax) on the variant at several nworld. Script-only."""
import os, sys, time, mujoco, numpy as np, jax
from mujoco import mjx
import bench
nr, fn = int(sys.argv[1]), sys.argv[2]
ns = [int(x) for x in sys.argv[3].split(",")]
opts = {"xla_gpu_graph_min_graph_size": 1} if os.environ.get("GRAPH") == "1" else {}
m, robots, ball = bench.build(nr, dict(hull=16, drib="cyl"))
if "TOL" in os.environ: m.opt.tolerance = float(os.environ["TOL"])
if "ITERS" in os.environ: m.opt.iterations = int(os.environ["ITERS"])
if "LS" in os.environ: m.opt.ls_iterations = int(os.environ["LS"])
if os.environ.get("SOLVER") == "CG": m.opt.solver = mujoco.mjtSolver.mjSOL_CG
mx = mjx.put_model(m, impl="jax")
f = jax.jit(jax.vmap(getattr(mjx, fn), in_axes=(None, 0)), compiler_options=opts)
for n in ns:
    u = jax.random.uniform(jax.random.PRNGKey(0), (n, 3 * nr + 2))
    d = jax.vmap(lambda uu: mjx.make_data(m, impl="jax").replace(qpos=bench.qpos_init(jax.numpy, m, robots, ball, uu, "random")))(u)
    t = time.perf_counter(); c = f.lower(mx, d).compile(); tc = time.perf_counter() - t
    r = c(mx, d); jax.block_until_ready(r)
    t = time.perf_counter(); r = c(mx, d); jax.block_until_ready(r); tr = time.perf_counter() - t
    print(f"{fn} tol={m.opt.tolerance} it={m.opt.iterations} ls={m.opt.ls_iterations} niter={np.asarray(r._impl.solver_niter).max() if fn=='step' else '-'} nworld={n} graph={bool(opts)} compile={tc:.1f}s run={1e3*tr:.1f}ms per-world={1e6*tr/n:.1f}us", flush=True)
