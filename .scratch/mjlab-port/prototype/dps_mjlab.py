"""PROTOTYPE: mjlab decisions/s, end to end -- the twin of ../../gpu-port-impl/10-bench/dps_mjx.py.

    uv run python .scratch/mjlab-port/prototype/dps_mjlab.py <nworld> [...]

ContestedPossession-v0 on `mjsoccer.mjlab_prototype` (mjlab's `step()`, 48 iterations per
decision, the kick firing whenever re-armed), uniform random actions drawn on the device.
Best of 3 timed runs of SECONDS after a warm-up, like the MJX bench. Also:
- `physics_dps`: decisions/s of the 48 bare graph launches alone (mjlab's `measure_physics_sps`
  idea), so `overhead` = 1 - dps / physics_dps is the managers' + our terms' share;
- `fired`: fraction of decisions where the world fired (ran the fine schedule);
- memory: torch's peak allocated, Warp's mempool high-water, and the device's used memory
  (`mem_get_info`, which includes the CUDA context).
One `RESULT {json}` line per nworld; a failure prints `FAIL {json}`.
"""

import json
import sys
import time

import torch
import warp as wp

from mjsoccer import mjlab_prototype as P

SECONDS = 10.0


def bench(nworld: int) -> dict:
    t0 = time.perf_counter()
    env = P.make(nworld)
    torch.cuda.synchronize()
    init_s = time.perf_counter() - t0
    task, sim = env.task, env.sim
    pool_s = task.seat_pool()
    env.reset()

    def act():
        return torch.rand(nworld, 7, device=env.device) * 2 - 1

    stats = dict(ends=0, fired=0, decisions=0)

    def step():
        _, _, te, to, _ = env.step(act())
        stats["ends"] += (te | to).sum()
        stats["fired"] += task.fire.sum()
        stats["decisions"] += nworld

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

    # Bare physics: the 48 captured-graph launches per decision, nothing else.
    phys, n, t0 = 0.0, 0, time.perf_counter()
    while time.perf_counter() - t0 < SECONDS:
        for _ in range(env.cfg.decimation):
            sim.step()
        n += 1
        if n % 20 == 0:
            torch.cuda.synchronize()
    torch.cuda.synchronize()
    phys = n * nworld / (time.perf_counter() - t0)

    free, total = torch.cuda.mem_get_info()
    bits = task.overflow_seen
    return dict(
        env="ContestedPossession-v0", backend="mjlab", nworld=nworld, dps=round(best),
        physics_dps=round(phys), overhead=round(1 - best / phys, 3),
        decisions_per_world=stats["decisions"] // nworld, episode_ends=int(stats["ends"]),
        fired=round(int(stats["fired"]) / stats["decisions"], 3),
        init_s=round(init_s, 1), pool_s=round(pool_s, 2),
        overflow_worlds=int((bits != 0).sum()), overflow_or=sum(1 << b for b in range(16) if ((bits >> b) & 1).any()),
        nconmax=env.cfg.sim.nconmax, njmax=env.cfg.sim.njmax,
        torch_peak_mb=round(torch.cuda.max_memory_allocated() / 2**20),
        warp_mempool_peak_mb=round(wp.get_mempool_used_mem_high("cuda:0") / 2**20),
        device_used_mb=round((total - free) / 2**20),
    )


if __name__ == "__main__":
    with P.warp_stream():
        for nworld in map(int, sys.argv[1:]):
            try:
                print("RESULT " + json.dumps(bench(nworld)), flush=True)
            except Exception as e:  # OOM and friends: the max-nworld probe wants the message
                print("FAIL " + json.dumps(dict(nworld=nworld, error=f"{type(e).__name__}: {e}"[:300])),
                      flush=True)
