"""Ticket 04 (copy of 03's script, run against a candidate via MJSOCCER_MODELS) step 3 follow-up: peak |world yaw rate| per random GoToBall-v0 episode,
same seeding as baseline.py.

    uv run python .scratch/physics-fidelity/candidate-04/random_yaw.py > .scratch/physics-fidelity/candidate-04/random_yaw_GoToBall.txt
"""
import gymnasium as gym
import numpy as np

import mjsoccer  # noqa: F401

env = gym.make("GoToBall-v0", tier=1)
b = env.unwrapped
r = b.robots[0]
peaks = []
for s in range(50):
    env.reset(seed=s)
    env.action_space.seed(s)
    p = 0.0
    while True:
        _, _, te, tr, _ = env.step(env.action_space.sample())
        R = b.data.xmat[r.body].reshape(3, 3)
        # free-joint angular velocity is body-frame; rotate to get world yaw
        p = max(p, abs((R @ b.data.qvel[r.qvel_adr + 3 : r.qvel_adr + 6])[2]))
        if te or tr:
            break
    peaks.append(p)
print("GoToBall-v0 random: max |world yaw rate| per episode p50/p95/max =",
      np.percentile(peaks, [50, 95, 100]).round(2))
