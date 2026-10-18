"""Kick/chip exit speed from a seated ball, post-05 contact (ticket 09).

    uv run python .scratch/physics-fidelity/candidate-09/exit_speed.py

Dribbling-v0 seats the ball. One step of kick (or chip), then coast 0.5 s. Reports the peak
horizontal ball speed, peak height, and the chassis recoil (body-frame forward speed change
over the kick step). Rulebook cap is 6.5 m/s (ticket 02, §8.4.2).
"""

import gymnasium as gym
import numpy as np

import mjsoccer  # noqa: F401

for tier in (0, 1):
    env = gym.make("Dribbling-v0", tier=tier).unwrapped
    m, d = env.model, env.data
    for j, name in ((0, "kick"), (1, "chip")):
        for dribble in (0.0, 0.5):
            vxy, apex, recoil = [], [], []
            for seed in range(8):
                env.reset(seed=seed)
                r = env.robots[0]
                z0 = env.ball.pos(d)[2]
                a = np.zeros(env.action_space.shape, dtype=np.float32)
                a[-1] = dribble
                a[env.drive_dim + j] = 1.0
                f = r.forward(d)[:2]
                v0 = np.array(r.velocity(d)[:2]) @ f
                env.step(a)
                v1 = np.array(r.velocity(d)[:2]) @ f
                a[env.drive_dim + j] = 0.0
                best_v = best_z = 0.0
                for _ in range(30):
                    best_v = max(best_v, float(np.hypot(*env.ball.vel(d)[:2])))
                    best_z = max(best_z, float(env.ball.pos(d)[2] - z0))
                    env.step(a)
                vxy.append(best_v)
                apex.append(best_z)
                recoil.append(v1 - v0)
            print(
                f"tier {tier} {name} dribble {dribble}: v_xy {np.mean(vxy):.2f} "
                f"[{min(vxy):.2f}, {max(vxy):.2f}] m/s, apex {np.mean(apex) * 1000:.0f} mm, "
                f"recoil {np.mean(recoil):+.3f} m/s"
            )
    env.close()
