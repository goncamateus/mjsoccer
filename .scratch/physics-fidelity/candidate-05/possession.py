"""Idle possession: holds() steps out of 120 with the ball seated, 8 seeds (docs/index.md:199-212).

    uv run python .scratch/physics-fidelity/candidate-05/possession.py
"""

import gymnasium as gym
import numpy as np

import mjsoccer  # noqa: F401

for env_id, index in [("Dribbling-v0", 0), ("ContestedPossession-v0", 1)]:
    for tier in (0, 1):
        env = gym.make(env_id, tier=tier).unwrapped
        held = []
        for seed in range(8):
            env.reset(seed=seed)
            zero = np.zeros(env.action_space.shape, dtype=np.float32)
            n = 0
            for _ in range(120):
                env.step(zero)
                n += int(env.holds(index))
            held.append(n)
        env.close()
        print(f"{env_id} tier {tier}: min {min(held)} max {max(held)} median {np.median(held)} {held}")

# Raw ball/dribbler contact vs holds() on Dribbling-v0 (the docs/index.md "Possession" table).
for tier in (0, 1):
    env = gym.make("Dribbling-v0", tier=tier).unwrapped
    raw = held = 0
    for seed in range(8):
        env.reset(seed=seed)
        zero = np.zeros(env.action_space.shape, dtype=np.float32)
        for _ in range(120):
            env.step(zero)
            raw += int(env._grace[0] == env._grace_steps)
            held += int(env.holds(0))
    env.close()
    print(f"Dribbling-v0 tier {tier}: raw contact {raw / 960:.0%}, holds {held / 960:.0%}")
