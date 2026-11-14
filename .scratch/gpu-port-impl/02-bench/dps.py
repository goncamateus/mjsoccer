"""CPU decisions/s baseline (ticket 02): AsyncVectorEnv, 40 Hz, random actions, autoreset.

    uv run python .scratch/gpu-port-impl/02-bench/dps.py

Same set-up as the 1 535 / 299 DPS of gpu-port issue 06 (mujoco 3.11): ContestedPossession
with 24 envs (2 robots), GoToBall with 12 envs (7 robots), and 1 sync env of each. Best of 3
timed runs of SECONDS each after a warm-up.
"""

import time

import gymnasium as gym
import mujoco
import numpy as np

import mjsoccer  # noqa: F401

SECONDS = 20.0


def bench(env_id, n, sync):
    fns = [lambda: gym.make(env_id, control_hz=40) for _ in range(n)]
    venv = gym.vector.SyncVectorEnv(fns) if sync else gym.vector.AsyncVectorEnv(fns)
    venv.reset(seed=0)
    venv.action_space.seed(0)
    for _ in range(50):
        venv.step(venv.action_space.sample())
    best = 0.0
    for _ in range(3):
        steps, t0 = 0, time.perf_counter()
        while time.perf_counter() - t0 < SECONDS:
            venv.step(venv.action_space.sample())
            steps += 1
        best = max(best, steps * n / (time.perf_counter() - t0))
    venv.close()
    return best


if __name__ == "__main__":
    print(f"mujoco {mujoco.__version__}")
    for env_id, n in (("ContestedPossession-v0", 24), ("GoToBall-v0", 12)):
        for nn, sync in ((n, False), (1, True)):
            print(f"{env_id},{nn},{'sync' if sync else 'async'},{bench(env_id, nn, sync):.0f} DPS", flush=True)
