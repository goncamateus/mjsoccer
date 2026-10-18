"""holds() steps out of 120 idle ones, seeds 0-7, ContestedPossession robot 1, tier 0 (ticket 12).

    uv run python .scratch/physics-fidelity/candidate-12/held.py
"""

import gymnasium as gym
import numpy as np

import mjsoccer  # noqa: F401

env = gym.make("ContestedPossession-v0", tier=0).unwrapped
out = []
for seed in range(8):
    env.reset(seed=seed)
    hold = np.zeros(env.action_space.shape, dtype=np.float32)
    hold[-1] = 0.5
    out.append(sum(int(env.holds(1)) for _ in range(120) if env.step(hold) or True))
print(out)
