"""mjlab decisions/s, end to end (tickets 04, 08): the twin of ../../gpu-port-impl/10-bench/dps_mjx.py.

    systemd-run --user --scope -p MemoryMax=10G \
        uv run --group mjlab python .scratch/mjlab-port/bench/dps_mjlab.py <env_id> <nworld> [...]

The whole env, `mjsoccer.mjlab.make(env_id)` -- mjlab's `step()` with its managers, the
task's controller, kicker, possession, the 48-iteration firing schedule, auto-reset from
the first-state pool -- 40 Hz, repeat_action 1, uniform random actions drawn on the device
(kick/chip fire whenever re-armed). Best of 3 timed runs of SECONDS after a warm-up, like
the MJX bench. Also: the fraction of decisions where a world fired, the pool's wall time,
overflow bits OR-ed over the run, and memory (torch peak, Warp mempool high-water, device
used). One `RESULT {json}` line per nworld; a failure prints `FAIL {json}`.

Ticket 08 adds, read at the end of every decision before the auto-reset clears them
(a hook on `SoccerEnv.end_decision`): the overflow bits per type as world-decision rates
(`rate_<TYPE>`, the ITERATIONS bit of mjlab #1263 among them), the batch's peak broadphase
`ncollision` per world (what fills `nconmax`) and peak `nefc` (against `njmax`), and the
solver's iteration count against `opt.iterations` (sampled on the decision's last physics
iteration only; the sticky ITERATIONS bit covers all of them). CHASE=1 swaps the random target
pose for `tasks.goto_ball_policy` (kick, chip and dribbler stay random), as the MJX bench:
robot 0 drives through the parked robots, the contact stress run.
"""

import json
import os
import sys
import time

import numpy as np
import torch
import warp as wp

import mjsoccer.mjlab
from mjsoccer import scene
from mujoco_warp import OverflowType

SECONDS = 10.0
CHASE = os.environ.get("CHASE") == "1"


def chase(obs, a):
    """`tasks.goto_ball_policy` on the batch: target the ball, face it."""
    ball, pos = obs[:, 0:2], obs[:, 6:8]
    h = torch.atan2((ball - pos)[:, 1] * scene.FIELD_HALF_Y, (ball - pos)[:, 0] * scene.FIELD_HALF_X)
    a[:, :4] = torch.stack([ball[:, 0], ball[:, 1], torch.sin(h), torch.cos(h)], 1).clamp(-1, 1)
    return a


def bench(env_id: str, nworld: int) -> dict:
    t0 = time.perf_counter()
    env = mjsoccer.mjlab.make(env_id, nworld=nworld, seed=0)
    torch.cuda.synchronize()
    make_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    env.refill_pool()
    torch.cuda.synchronize()
    pool_s = time.perf_counter() - t0
    obs = env.reset()[0]["actor"]
    d = env.sim.data
    types = list(OverflowType)
    w = dict(bits=torch.zeros(len(types), dtype=torch.int64, device=env.device),
             ncol=torch.zeros((), dtype=torch.int64, device=env.device),
             nefc=torch.zeros((), dtype=torch.int64, device=env.device),
             niter=torch.zeros((), dtype=torch.int64, device=env.device))
    tv = torch.tensor([t.value for t in types], device=env.device)
    end = env.end_decision

    def watched():  # the decision's last bits, before the auto-reset clears them
        out = end()
        ov = d.overflow[:]
        w["bits"] += ((ov[:, None] & tv) != 0).sum(0)
        w["ncol"] = torch.maximum(w["ncol"], d.ncollision[:].max().long())
        w["nefc"] = torch.maximum(w["nefc"], d.nefc[:].max().long())
        w["niter"] = torch.maximum(w["niter"], d.solver_niter[:].max().long())
        return out

    env.end_decision = watched
    g = torch.Generator(device=env.device).manual_seed(1)
    n_act = env.logic.action_size
    bits = torch.zeros(nworld, dtype=torch.int32, device=env.device)
    stats = {"ends": 0, "fired": 0, "decisions": 0}

    def step():
        nonlocal obs
        a = torch.rand(nworld, n_act, device=env.device, generator=g) * 2 - 1
        if CHASE:
            a = chase(obs, a)
        o, _, te, to, _ = env.step(a)
        obs = o["actor"]
        stats["ends"] += (te | to).sum()
        stats["fired"] += env.world["fired"].any(-1).sum()
        stats["decisions"] += nworld
        bits.bitwise_or_(env.sim.data.overflow[:])

    for _ in range(20):  # warm-up
        step()
    torch.cuda.synchronize()
    best = 0.0
    for _ in range(3):
        n, t0 = 0, time.perf_counter()
        while time.perf_counter() - t0 < SECONDS:
            step()
            n += 1
            if n % 20 == 0:
                torch.cuda.synchronize()
        torch.cuda.synchronize()
        best = max(best, n * nworld / (time.perf_counter() - t0))
    free, total = torch.cuda.mem_get_info()
    rates = {f"rate_{t.name}": float(n) / stats["decisions"]
             for t, n in zip(types, w["bits"].tolist())
             if n or t in (OverflowType.ITERATIONS, OverflowType.LS_ITERATIONS)}
    return dict(
        env=env_id, backend="mjlab", chase=CHASE, nworld=nworld, dps=round(best),
        decisions_per_world=stats["decisions"] // nworld, episode_ends=int(stats["ends"]),
        fired=round(int(stats["fired"]) / stats["decisions"], 3),
        make_s=round(make_s, 1), pool_s=round(pool_s, 2),
        overflow_worlds=int((bits != 0).sum()), overflow_or=int(np.bitwise_or.reduce(bits.cpu().numpy())),
        nconmax=env.cfg.sim.nconmax, njmax=env.cfg.sim.njmax, **rates,
        ncollision_peak_per_world=round(int(w["ncol"]) / nworld, 1), nefc_peak=int(w["nefc"]),
        solver_niter_peak=int(w["niter"]), iterations=int(env.sim.mj_model.opt.iterations),
        torch_peak_mb=round(torch.cuda.max_memory_allocated() / 2**20),
        warp_mempool_peak_mb=round(wp.get_mempool_used_mem_high("cuda:0") / 2**20),
        device_used_mb=round((total - free) / 2**20),
    )


if __name__ == "__main__":
    env_id = sys.argv[1]
    for nworld in map(int, sys.argv[2:]):
        try:
            print("RESULT " + json.dumps(bench(env_id, nworld)), flush=True)
        except Exception as e:  # OOM and friends
            print("FAIL " + json.dumps(dict(nworld=nworld, error=f"{type(e).__name__}: {e}"[:300])),
                  flush=True)
