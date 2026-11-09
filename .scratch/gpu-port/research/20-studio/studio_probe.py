"""Ticket 20: cost of feeding one world of an MJX-Warp batch to Studio / Filament.

    PYTHONPATH=../../../../src ~/.venvs/mjsoccer-gpu/bin/python studio_probe.py [nworld] [viewer]

viewer: none | web | native. Prints RESULT {...} lines.
"""
import json, os, sys, time
import numpy as np
import mujoco
import jax, jax.numpy as jp
from mujoco import mjx, mjrf
from mujoco.mjx.warp import types as wt
from mujoco.rendering.filament import model_scene, renderer
from mujoco.studio import window  # noqa: F401  registers Filament's material providers
from mjsoccer import scene as S

NW = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
VIEWER = sys.argv[2] if len(sys.argv) > 2 else "none"
m, robots, ball = S.build(1, 1, 0)
mx = mjx.put_model(m, impl="warp", graph_mode=wt.GraphMode.WARP_STAGED)
def init(k):
    d = mjx.make_data(m, impl="warp", naconmax=NW * 128, njmax=256)
    return d.replace(qvel=d.qvel.at[:].set(jax.random.uniform(k, (m.nv,), minval=-1, maxval=1)))
dx = jax.jit(jax.vmap(init))(jax.random.split(jax.random.PRNGKey(0), NW))
step = jax.jit(lambda d: jax.lax.fori_loop(0, 24, lambda _, d: jax.vmap(mjx.step, (None, 0))(mx, d), d))
dx = step(dx); jax.block_until_ready(dx)

def med(f, n=30):
    ts = []
    for _ in range(n):
        a = time.perf_counter(); f(); ts.append(time.perf_counter() - a)
    return float(np.median(ts[3:])) * 1e3

d = mujoco.MjData(m)
i = 3
def full():  # whole-world copy: slice every leaf, then mjx.get_data
    global d
    d = mjx.get_data(m, jax.tree.map(lambda x: x[i] if hasattr(x, "shape") and x.ndim and x.shape[0] == NW else x, dx))
def slim():  # only what the viewer needs: qpos/qvel/time, then forward kinematics on CPU
    d.qpos[:] = np.asarray(dx.qpos[i]); d.qvel[:] = np.asarray(dx.qvel[i]); d.time = float(dx.time[i])
    mujoco.mj_forward(m, d)
res = dict(nworld=NW, step24_ms=med(lambda: jax.block_until_ready(step(dx)), 10))
try:
    res["get_data_ms"] = med(full)
except Exception as e:  # noqa: BLE001
    res["get_data_err"] = f"{type(e).__name__}: {e}"[:200]
res["slim_ms"] = med(slim)

# Filament offscreen frame (Vulkan, no window, no display needed) of that world.
ctx = mjrf.Context(mjrf.ContextConfig(graphics_api=mjrf.GraphicsApi.GRAPHICS_API_VULKAN))
r = renderer.Renderer(ctx); ms = model_scene.ModelScene(ctx, m, r.scene("scn"))
buf = np.empty((360, 640, 3), np.uint8); r.target("out", buf)
cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
cam.fixedcamid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "overview")
def frame():
    slim(); ms.update(d)
    r.render([renderer.View(scene="scn", target="out", camera=mujoco.mjv_camera2GLCamera(m, d, cam))])
res["slim_plus_filament_640x360_ms"] = med(frame)

if VIEWER != "none":
    from mujoco.studio import app as studio_app, viewer_protocol
    if VIEWER == "web":
        from mujoco.studio import launch_web as L
        cfg = viewer_protocol.ViewerConfig(gfx="web", http_port=8099)
    else:
        from mujoco.studio import launch_native as L
        cfg = viewer_protocol.ViewerConfig(title="probe", width=640, height=360)
    with L.launch(cfg, plugins=[studio_app.App()]) as h:
        mm, dd = m, d
        ts = []; t_end = time.time() + 6
        while h.is_running() and time.time() < t_end:
            dx = step(dx)
            slim()
            a = time.perf_counter(); mm, dd = h.sync(mm, d); ts.append(time.perf_counter() - a)
            time.sleep(0.02)
        res[f"{VIEWER}_sync_ms"] = float(np.median(ts[5:])) * 1e3; res[f"{VIEWER}_iters"] = len(ts)
print("RESULT " + json.dumps(res), flush=True)
