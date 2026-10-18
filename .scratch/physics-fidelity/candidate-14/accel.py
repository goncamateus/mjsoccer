"""Ticket 14, criterion 5: tier-1 achievable acceleration and speed, full command from rest, 2 s.

    uv run python .scratch/physics-fidelity/candidate-14/accel.py > accel.txt
Acceleration is dv/dt of the body-frame speed, best 0.1 s window, through env.step (60 Hz).
"""
import gymnasium as gym
import numpy as np

import mjsoccer  # noqa: F401
from mjsoccer.tasks import wheel_command

for name, cmd in (("forward", (10.0, 0.0, 0.0)), ("lateral", (0.0, 10.0, 0.0))):
    env = gym.make("GoToBall-v0", tier=1).unwrapped
    env.reset(seed=0)
    r = env.robots[0]
    r.place(env.data, -5.0, -3.5, 0.6435)  # 7 m of diagonal before a wall
    env.ball.place(env.data, 5.5, 4.0)
    a = np.zeros(env.action_space.shape, dtype=np.float32)
    v = []
    for _ in range(120):
        a[:4] = wheel_command(env, 0, *cmd)  # saturates: the wheels' full envelope on that axis
        env.step(a)
        R = env.data.xmat[r.body].reshape(3, 3)
        vx, vy, _ = r.velocity(env.data)
        v.append(np.linalg.norm((R.T @ (vx, vy, 0.0))[:2]))
    v = np.array(v)
    k = round(0.1 / env.dt)
    acc = (v[k:] - v[:-k]).max() / (k * env.dt)
    print(f"{name}: a_max {acc:.2f} m/s^2 (0.1 s window), v at 2 s {v[-1]:.2f} m/s, peak {v.max():.2f}")
    env.close()
