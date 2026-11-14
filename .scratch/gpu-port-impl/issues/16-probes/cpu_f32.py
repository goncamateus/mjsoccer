"""CPU hold_turn from seed 0, qpos/qvel rounded to float32 after every physics step."""
import gymnasium as gym
import mujoco
import numpy as np

import mjsoccer  # noqa: F401
from mjsoccer.base import MAX_W

env = gym.make("Dribbling-v0").unwrapped
_step = mujoco.mj_step


def rounded(m, d, *a):
    _step(m, d, *a)
    d.qpos[:] = d.qpos.astype(np.float32)
    d.qvel[:] = d.qvel.astype(np.float32)


mujoco.mj_step = rounded
env.reset(seed=0)
held = 0
for _ in range(120):
    env.step_raw(np.array([0.0, 0.0, 2.0 / MAX_W, 0.0, 0.0, 1.0]))
    held += env.holds(0)
print("f32-rounded state held", held)
