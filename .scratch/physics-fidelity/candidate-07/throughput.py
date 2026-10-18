"""Ticket 07: env steps/s, best of 3 x 2000 random-action steps, tier 1.

    MJSOCCER_MODELS=<variant dir> uv run python .scratch/physics-fidelity/candidate-07/throughput.py <label>
"""
import sys, time, gymnasium as gym, mjsoccer
for env_id in ("GoToBall-v0", "ContestedPossession-v0", "StaticDefenders-v0"):
    best = 0
    for _ in range(3):
        env = gym.make(env_id, tier=1); env.reset(seed=0); env.action_space.seed(0)
        t0 = time.perf_counter()
        for _ in range(2000):
            *_, te, tr, _ = env.step(env.action_space.sample())
            if te or tr: env.reset()
        best = max(best, 2000 / (time.perf_counter() - t0)); env.close()
    print(f"{sys.argv[1]},{env_id},{best:.1f}", flush=True)
