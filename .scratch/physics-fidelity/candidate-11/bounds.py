"""Ticket 11: do the declared observation bounds hold? Random action, seeds 0-9, full episodes.

    uv run python .scratch/physics-fidelity/candidate-11/bounds.py <tier>

One row per env: steps, steps outside the Box, and the smallest gap between a physically
bounded channel (robot x/y, ball z, waypoint) and its bound, in metres, with the channel
index. sin/cos/has_ball are skipped: they sit on their bounds by construction.
"""
import sys

import gymnasium as gym
import numpy as np

import mjsoccer
from mjsoccer import ENV_IDS

tier = int(sys.argv[1])
print("env,steps,outside,min_gap,channel")
for env_id in ENV_IDS:
    env = gym.make(env_id, tier=tier)
    sp = env.observation_space
    lo, hi = sp.low.astype(np.float64), sp.high.astype(np.float64)
    steps = outside = 0
    gap = np.full(lo.shape, np.inf)
    by_construction = (hi == 1) & ((lo == -1) | (lo == 0))
    nr = len(env.unwrapped.robots)
    # back to metres: ball z by GOAL_HEIGHT, robot x/y and waypoint by FIELD_HALF_X/Y
    scale = np.ones(lo.shape)
    scale[2] = 0.155
    scale[6:6 + 8 * nr:8] = scale[6 + 8 * nr::2] = 6.0
    scale[7:6 + 8 * nr:8] = scale[7 + 8 * nr::2] = 4.5
    for seed in range(10):
        o, _ = env.reset(seed=seed)
        env.action_space.seed(seed)
        while True:
            steps += 1
            outside += not sp.contains(o)
            gap = np.minimum(gap, np.minimum(o - lo, hi - o))
            o, _, term, trunc, _ = env.step(env.action_space.sample())
            if term or trunc:
                break
    env.close()
    gap = np.where(by_construction, np.inf, gap) * scale
    k = int(np.argmin(gap))
    print(f"{env_id},{steps},{outside},{gap[k]:.4f},{k}", flush=True)
