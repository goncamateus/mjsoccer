"""Does torch off Warp's stream corrupt the env? (ticket 05)

    systemd-run --user --scope -p MemoryMax=10G \
        uv run --group mjlab python .scratch/mjlab-port/bench/streams_mjlab.py <nworld> <decisions>

Three ways of stepping `mjsoccer.mjlab.make("ContestedPossession-v0")`:

- `warp`: as shipped -- `SoccerEnv` runs torch on Warp's stream inside step/reset/refill_pool.
- `default`: that wrapper removed, torch on its default stream (the legacy null stream).
- `side`: wrapper removed and the caller on a torch side stream (`torch.cuda.Stream()`,
  non-blocking), nothing ordering it against Warp's.

Warp's physics is not bit-reproducible run to run (contact/solver atomics: two `warp` runs'
pools already differ by ~1 mm), so runs are not compared with each other. Each run checks an
invariant instead: torch writes every iteration's per-world `opt.timestep` in place
(`SoccerEnv.substep`) and the step graph adds it to `data.time`, so a world's `time` must be
*exactly* the float32 running sum of the plan torch wrote (`SoccerEnv._plan`). A race
between torch's writes and Warp's graph shows up as a world whose time is off. Worlds that
ever ended (reset, time restarts) are dropped from the check. Also reports decisions/s.
"""

import contextlib
import json
import sys
import time

import torch
import warp as wp

import mjsoccer.mjlab

ENV = "ContestedPossession-v0"


def run(mode: str, nworld: int, decisions: int) -> dict:
    env = mjsoccer.mjlab.make(ENV, nworld=nworld, seed=0)
    if mode != "warp":
        env._on_warp_stream = contextlib.nullcontext
    caller = torch.cuda.Stream() if mode == "side" else torch.cuda.current_stream()
    plans, ended = [], torch.zeros(nworld, dtype=torch.bool)
    with torch.cuda.stream(caller):
        env.refill_pool()
        env.reset()
        torch.cuda.synchronize()
        t = env.sim.data.time[:].cpu()  # the reset's time, 0
        g = torch.Generator(device=env.device).manual_seed(1)
        t0 = time.perf_counter()
        for _ in range(decisions):
            a = torch.rand(nworld, env.logic.action_size, device=env.device, generator=g) * 2 - 1
            _, _, term, trunc, _ = env.step(a)
            plans.append(env._plan.clone())  # in caller's stream, after this step's writes
            ended |= (term | trunc).cpu()
        torch.cuda.synchronize()
        dps = decisions * nworld / (time.perf_counter() - t0)
    for plan in plans:
        for dt in plan.cpu().unbind(1):
            t = t + dt
    got = env.sim.data.time[:].cpu()
    off = (got != t) & ~ended
    return dict(mode=mode, nworld=nworld, decisions=decisions, dps=round(dps),
                checked=int((~ended).sum()), time_off=int(off.sum()),
                max_err=float((got - t)[~ended].abs().max()))


if __name__ == "__main__":
    print(json.dumps({"torch_stream": torch.cuda.current_stream().cuda_stream,
                      "warp_stream": wp.get_stream("cuda:0").cuda_stream}), flush=True)
    for mode in sys.argv[3:] or ("warp", "default", "side"):
        print("RESULT " + json.dumps(run(mode, int(sys.argv[1]), int(sys.argv[2]))), flush=True)
