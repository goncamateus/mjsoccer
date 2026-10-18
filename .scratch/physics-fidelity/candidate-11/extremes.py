"""Ticket 11: per-channel observation extremes on main, random action, seeds 0-19.

    uv run python .scratch/physics-fidelity/candidate-11/extremes.py <tier>

Prints one row per env: max |obs| of each channel group, plus the ball-height channel
for a ball resting on the carpet.
"""
import sys

import gymnasium as gym
import numpy as np

import mjsoccer  # noqa: F401
from mjsoccer import ENV_IDS

tier = int(sys.argv[1])
print("env,ball_xy,ball_z,ball_z_rest_min,ball_v,robot_xy,robot_v,robot_w,extra,abs_max_all")
for env_id in ENV_IDS:
    env = gym.make(env_id, tier=tier)
    base = env.unwrapped
    nr = len(base.robots)
    obs = []
    for seed in range(20):
        o, _ = env.reset(seed=seed)
        env.action_space.seed(seed)
        obs.append(o)
        while True:
            o, _, term, trunc, _ = env.step(env.action_space.sample())
            obs.append(o)
            if term or trunc:
                break
    env.close()
    a = np.abs(np.asarray(obs))
    rob = a[:, 6 : 6 + 8 * nr].reshape(len(a), nr, 8)
    extra = a[:, 6 + 8 * nr :]
    print(f"{env_id},{a[:, :2].max():.3f},{a[:, 2].max():.3f},{np.asarray(obs)[:, 2].min():.3f},"
          f"{a[:, 3:6].max():.3f},{rob[..., :2].max():.3f},{rob[..., 4:6].max():.3f},"
          f"{rob[..., 6].max():.3f},{extra.max() if extra.size else 0:.3f},{a.max():.3f}")
