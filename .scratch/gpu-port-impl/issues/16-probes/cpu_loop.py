"""CPU sensitivity: hold_turn from seed 0's state with qpos/qvel perturbed by eps * N(0,1)."""
import sys

import gymnasium as gym
import numpy as np

import mjsoccer  # noqa: F401
from mjsoccer.base import MAX_W

eps = float(sys.argv[1]) if len(sys.argv) > 1 else 1e-6
n = int(sys.argv[2]) if len(sys.argv) > 2 else 16
env = gym.make("Dribbling-v0").unwrapped
rng = np.random.default_rng(int(sys.argv[3]) if len(sys.argv) > 3 else 0)
out = []
for k in range(n):
    env.reset(seed=0)
    if eps:
        env.data.qvel[:] += eps * rng.standard_normal(env.data.qvel.shape)
    held = 0
    for _ in range(120):
        env.step_raw(np.array([0.0, 0.0, 2.0 / MAX_W, 0.0, 0.0, 1.0]))
        held += env.holds(0)
    out.append(held)
print("eps", eps, "held", sorted(out), "median", np.median(out))
