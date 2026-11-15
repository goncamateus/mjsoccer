"""EPA (CCD) slots Warp takes per mjwarp.step, all worlds, eager (no graph) (ticket 05).

    uv run --group mjlab python .scratch/mjlab-port/bench/epa_slots.py <nworld> <decisions> [jam]

Every decision, blue's target is yellow's position: the robots drive into each other in
every world. With `jam`, blue starts teleported 0.12-0.20 m from yellow (overlapping; it
overflows the broadphase too). Prints the peak slots per decision and per world; slots are
taken only when GJK finds a convex pair penetrating (`collision_convex.py`)."""
import math
import sys

import numpy as np
import torch
import warp as wp

import mujoco_warp._src.collision_convex as cc
import mjsoccer.mjlab
from mjsoccer import scene
n, decisions = int(sys.argv[1]), int(sys.argv[2])
env = mjsoccer.mjlab.make("ContestedPossession-v0", nworld=n, seed=0)
env.refill_pool(); env.reset()
env.sim.use_cuda_graph = False; env.sim.create_graph()
d, (blue, yellow) = env.sim.data, env.logic.robots
g = torch.Generator(device=env.device).manual_seed(2)
r = 0.12 + 0.08 * torch.rand(n, device=env.device, generator=g)
th = 2 * math.pi * torch.rand(n, device=env.device, generator=g)
q = d.qpos[:].clone()
jam = d.xpos[:, yellow.body, :2] + torch.stack([r * th.cos(), r * th.sin()], -1)
q[:, blue.qpos_adr] = jam[:, 0] - blue.xy0[0]; q[:, blue.qpos_adr + 1] = jam[:, 1] - blue.xy0[1]
if "jam" in sys.argv: d.qpos[:] = q; env.sim.forward()
grab = []; orig = cc.wp.zeros
class W:
    def __getattr__(s, k): return getattr(wp, k)
    def zeros(s, *a, **k):
        x = orig(*a, **k); grab.append(x); return x
cc.wp = W()
peak = 0; bits = 0
for i in range(decisions):
    grab.clear()
    # drive blue at yellow: target = yellow's position
    yp = d.xpos[:, yellow.body, :2]
    a = torch.stack([yp[:, 0] / scene.FIELD_HALF_X, yp[:, 1] / scene.FIELD_HALF_Y,
                     torch.zeros(n, device=env.device), torch.ones(n, device=env.device),
                     *[-torch.ones(n, device=env.device)] * 3], -1)
    env.step(a)
    m = max(int(x.numpy().max()) for x in grab)
    peak = max(peak, m)
    bits |= int(np.bitwise_or.reduce(env.metrics["overflow"].cpu().numpy().astype(int)))
    if m or i % 20 == 0: print(i, "epa slots", m, "per world", round(m / n, 3), "overflowed worlds", int(env.metrics["overflow"].sum()), flush=True)
print("RESULT nworld", n, "peak", peak, "peak/world", round(peak / n, 3))
