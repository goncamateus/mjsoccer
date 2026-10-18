"""Exit speed vs plunger force, seated ball, dribbler off (ticket 09).

Pre-table history: patches scene.KICK_FORCE, which ticket 09 then removed, so it only
reproduces its .txt at the parent commit (061b887). calibrate.py is the live one.

    uv run python .scratch/physics-fidelity/candidate-09/sweep.py

Peak 3-D ball speed over 0.5 s after a one-step pulse; KICK_FORCE patched per row.
"""

import gymnasium as gym
import numpy as np

import mjsoccer  # noqa: F401
from mjsoccer import scene

for tier in (0, 1):
    env = gym.make("Dribbling-v0", tier=tier).unwrapped
    for j, name in ((0, "kick"), (1, "chip")):
        row = []
        for force in (25, 50, 75, 100, 125, 150, 175, 200, 250):
            scene.KICK_FORCE = float(force)
            env.reset(seed=0)
            a = np.zeros(env.action_space.shape, dtype=np.float32)
            a[env.drive_dim + j] = 1.0
            best = 0.0
            for _ in range(31):
                env.step(a)
                a[env.drive_dim + j] = 0.0
                best = max(best, float(np.linalg.norm(env.ball.vel(env.data))))
            row.append(f"{force}:{best:.2f}")
        print(f"tier {tier} {name}  " + "  ".join(row))
    env.close()
