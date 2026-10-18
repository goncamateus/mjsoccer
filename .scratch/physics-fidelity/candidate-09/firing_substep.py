"""Substep only the firing control step at dt/8; does it match a global dt/8? (ticket 09)

Pre-table history: patches scene.KICK_FORCE, which ticket 09 then removed, so it only
reproduces its .txt at the parent commit (061b887). calibrate.py is the live one.

    uv run python .scratch/physics-fidelity/candidate-09/firing_substep.py

ctrlrange opened to 500 N so the force that reaches 6.5 m/s can be read off.
"""

import gymnasium as gym
import mujoco
import numpy as np

import mjsoccer  # noqa: F401
from mjsoccer import scene

FORCES = (100, 200, 250, 300, 350, 400, 450, 500)


def run(env, j, force, mode):
    scene.KICK_FORCE = float(force)
    env.reset(seed=0)
    m, d = env.model, env.data
    a = np.zeros(env.action_space.shape, dtype=np.float32)
    a[env.drive_dim + j] = 1.0
    best = 0.0
    for k in range(31):
        if mode == "firing" and k == 0:
            d.ctrl[:] = 0
            env._apply(a.reshape(env.n_controlled, -1))
            dt = m.opt.timestep
            m.opt.timestep = dt / 8
            for _ in range(env.decimation * 8):
                mujoco.mj_step(m, d)
            m.opt.timestep = dt
            env.steps += 1
            np.maximum(env._plunger_lock - 1, 0, out=env._plunger_lock)
        else:
            env.step(a)
        a[env.drive_dim + j] = 0.0
        best = max(best, float(np.linalg.norm(env.ball.vel(d))))
    return best


for tier in (0, 1):
    for mode, div in (("global", 8), ("firing", 1), ("stock", 1)):
        scene.build.cache_clear()  # the MjModel is shared and we mutate it
        env = gym.make("Dribbling-v0", tier=tier).unwrapped
        env.model.opt.timestep /= div
        env.decimation *= div
        for r in env.robots:
            env.model.actuator_ctrlrange[[r.kick, r.chip], 1] = 500.0
        for j, name in ((0, "kick"), (1, "chip")):
            row = "  ".join(f"{f}:{run(env, j, f, mode):.2f}" for f in FORCES)
            print(f"tier {tier} {mode:6s} {name}  {row}")
        env.close()
