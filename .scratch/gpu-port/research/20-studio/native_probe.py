"""Ticket 20: native Studio viewer on its own, fed by a CPU loop. argv: gfx ('' | opengl | opengl_headless | vulkan)."""
import json, sys, time
import numpy as np, mujoco
from mujoco.studio import app as studio_app, launch_native, viewer_protocol
from mjsoccer import scene as S
gfx = sys.argv[1] if len(sys.argv) > 1 else ""
m, *_ = S.build(1, 1, 0); d = mujoco.MjData(m)
cfg = viewer_protocol.ViewerConfig(title="probe", width=640, height=360, gfx=gfx)
ts = []
with launch_native.launch(cfg, plugins=[studio_app.App()]) as h:
    t_end = time.time() + 5
    while h.is_running() and time.time() < t_end:
        a = time.perf_counter(); m, d = h.sync(m, d); ts.append(time.perf_counter() - a)
        mujoco.mj_step(m, d); time.sleep(0.01)
print("RESULT " + json.dumps(dict(gfx=gfx, native_sync_ms=float(np.median(ts[5:])) * 1e3, iters=len(ts))), flush=True)
