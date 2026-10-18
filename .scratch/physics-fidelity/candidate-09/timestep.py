"""Is the kick sawtooth timestep phase? Fine sweep at dt 2 ms (default) vs 1 ms vs 0.5 ms (ticket 09).

Pre-table history: patches scene.KICK_FORCE, which ticket 09 then removed, so it only
reproduces its .txt at the parent commit (061b887). calibrate.py is the live one.

    uv run python .scratch/physics-fidelity/candidate-09/timestep.py

The plunger pulse stays 16 ms (one control step) at every dt: decimation scales with it.
"""

import gymnasium as gym
import numpy as np

import mjsoccer  # noqa: F401
from mjsoccer import scene

for tier in (0, 1):
    for div in (1, 2, 4):
        scene.build.cache_clear()  # the MjModel is shared and we mutate it
        env = gym.make("Dribbling-v0", tier=tier).unwrapped
        env.model.opt.timestep /= div
        env.decimation *= div
        for j, name in ((0, "kick"), (1, "chip")):
            row = []
            for force in range(10, 201, 10):
                scene.KICK_FORCE = float(force)
                env.reset(seed=0)
                a = np.zeros(env.action_space.shape, dtype=np.float32)
                a[env.drive_dim + j] = 1.0
                best = 0.0
                for _ in range(31):
                    env.step(a)
                    a[env.drive_dim + j] = 0.0
                    best = max(best, float(np.linalg.norm(env.ball.vel(env.data))))
                row.append(f"{best:.2f}")
            print(f"tier {tier} dt/{div} {name}  " + "  ".join(row))
        env.close()
