"""mjlab decisions/s, end to end (ticket 04): the twin of ../../gpu-port-impl/10-bench/dps_mjx.py.

    systemd-run --user --scope -p MemoryMax=10G \
        uv run --group mjlab python .scratch/mjlab-port/bench/dps_mjlab.py <env_id> <nworld> [...]

The whole env, `mjsoccer.mjlab.make(env_id)` -- mjlab's `step()` with its managers, the
task's controller, kicker, possession, the 48-iteration firing schedule, auto-reset from
the first-state pool -- 40 Hz, repeat_action 1, uniform random actions drawn on the device
(kick/chip fire whenever re-armed). Best of 3 timed runs of SECONDS after a warm-up, like
the MJX bench. Also: the fraction of decisions where a world fired, the pool's wall time,
overflow bits OR-ed over the run, and memory (torch peak, Warp mempool high-water, device
used). One `RESULT {json}` line per nworld; a failure prints `FAIL {json}`.
"""

import json
import sys
import time

import numpy as np
import torch
import warp as wp

import mjsoccer.mjlab

SECONDS = 10.0


def bench(env_id: str, nworld: int) -> dict:
    t0 = time.perf_counter()
    env = mjsoccer.mjlab.make(env_id, nworld=nworld, seed=0)
    torch.cuda.synchronize()
    make_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    env.refill_pool()
    torch.cuda.synchronize()
    pool_s = time.perf_counter() - t0
    env.reset()
    g = torch.Generator(device=env.device).manual_seed(1)
    n_act = env.logic.action_size
    bits = torch.zeros(nworld, dtype=torch.int32, device=env.device)
    stats = {"ends": 0, "fired": 0, "decisions": 0}

    def step():
        a = torch.rand(nworld, n_act, device=env.device, generator=g) * 2 - 1
        _, _, te, to, _ = env.step(a)
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
    return dict(
        env=env_id, backend="mjlab", nworld=nworld, dps=round(best),
        decisions_per_world=stats["decisions"] // nworld, episode_ends=int(stats["ends"]),
        fired=round(int(stats["fired"]) / stats["decisions"], 3),
        make_s=round(make_s, 1), pool_s=round(pool_s, 2),
        overflow_worlds=int((bits != 0).sum()), overflow_or=int(np.bitwise_or.reduce(bits.cpu().numpy())),
        nconmax=env.cfg.sim.nconmax, njmax=env.cfg.sim.njmax,
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
