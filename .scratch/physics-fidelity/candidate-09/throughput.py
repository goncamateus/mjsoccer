"""Ticket 09: cost of FIRING_SUBSTEPS. Env steps/s, best of 3 x 2000 random-action steps, tier 1.

    uv run python .scratch/physics-fidelity/candidate-09/throughput.py

Random actions fire each plunger at every re-arm, the worst case; a policy that rarely
kicks pays ~nothing.
"""
import time

import gymnasium as gym

import mjsoccer  # noqa: F401
from mjsoccer import scene

for substeps in (1, 8):
    scene.FIRING_SUBSTEPS = substeps
    for env_id in ("GoToBall-v0", "ContestedPossession-v0", "StaticDefenders-v0"):
        best = 0
        for _ in range(3):
            env = gym.make(env_id, tier=1); env.reset(seed=0); env.action_space.seed(0)
            t0 = time.perf_counter()
            for _ in range(2000):
                *_, te, tr, _ = env.step(env.action_space.sample())
                if te or tr: env.reset()
            best = max(best, 2000 / (time.perf_counter() - t0)); env.close()
        print(f"substeps {substeps},{env_id},{best:.1f}", flush=True)
