"""Ticket 03 step 4: wall-clock env.step throughput, single process, random actions.

    uv run python .scratch/physics-fidelity/baseline/throughput.py

2000 timed steps per (env, tier) after one untimed reset; resets on episode end count
toward the wall time, as they would in training.
"""
import time

import gymnasium as gym

import mjsoccer

print("env,tier,steps,wall_s,steps_per_s")
for env_id in mjsoccer.ENV_IDS:
    for tier in (0, 1):
        env = gym.make(env_id, tier=tier)
        env.reset(seed=0)
        env.action_space.seed(0)
        t0 = time.perf_counter()
        for _ in range(2000):
            _, _, term, trunc, _ = env.step(env.action_space.sample())
            if term or trunc:
                env.reset()
        wall = time.perf_counter() - t0
        env.close()
        print(f"{env_id},{tier},2000,{wall:.3f},{2000 / wall:.1f}", flush=True)
