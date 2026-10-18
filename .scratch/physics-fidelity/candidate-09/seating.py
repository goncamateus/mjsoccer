"""How much does exit speed at a = 1 depend on how the ball sits? (ticket 09)

    uv run python .scratch/physics-fidelity/candidate-09/seating.py

Dribbling-v0 seed 0, robot standing. Wait N steps with the dribbler at D, then fire.
Calibration is the (0 steps, 0.0) cell.
"""

import gymnasium as gym
import numpy as np

import mjsoccer  # noqa: F401

for tier in (0, 1):
    env = gym.make("Dribbling-v0", tier=tier).unwrapped
    for j, name in ((0, "kick"), (1, "chip")):
        row = []
        for wait in (0, 15, 60):
            for dribble in (0.0, 0.5, 1.0):
                env.reset(seed=0)
                a = np.zeros(env.action_space.shape, dtype=np.float32)
                a[-1] = dribble
                for _ in range(wait):
                    env.step(a)
                a[env.drive_dim + j] = 1.0
                best = 0.0
                for _ in range(31):
                    env.step(a)
                    a[env.drive_dim + j] = 0.0
                    best = max(best, float(np.linalg.norm(env.ball.vel(env.data))))
                row.append(f"{wait}/{dribble}:{best:.2f}")
        print(f"tier {tier} {name}  " + "  ".join(row))
    env.close()
