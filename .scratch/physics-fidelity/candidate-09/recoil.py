"""Research doc §13.2 scenario 5 as amended by ticket 09: kick recoil with the chassis at 0 / ~1.5 m/s.

    uv run python .scratch/physics-fidelity/candidate-09/recoil.py

Dribbling-v0, ball seated, dribbler 0.5. Drive straight ahead for 1 s, then a = 1 on the
kicker (or chipper). Whole-robot CoM forward speed one step before vs one and five steps after
the firing step; peak ball speed, absolute and relative to the pre-kick chassis.
"""

import gymnasium as gym
import mujoco
import numpy as np

import mjsoccer  # noqa: F401

for tier in (0, 1):
    env = gym.make("Dribbling-v0", tier=tier).unwrapped
    m, d = env.model, env.data
    r = env.robots[0]
    env.reset(seed=0)
    if tier == 0:
        fwd = np.array([1.0, 0.0, 0.0])
    else:  # wheel speeds that drive body +x, as in candidate-04/line.py
        dirs = []
        for act in r.drive:
            j = m.actuator_trnid[act, 0]
            ax = d.xmat[r.body].reshape(3, 3).T @ d.xmat[m.jnt_bodyid[j]].reshape(3, 3) @ m.jnt_axis[j]
            dirs.append(np.cross(ax, [0, 0, -1])[0])
        fwd = -np.array(dirs) / np.abs(dirs).max()  # sign measured: + drove backwards
    for j, name in ((0, "kick"), (1, "chip")):
        for drive in (0.0, 0.375, 0.6):
            env.reset(seed=0)
            a = np.zeros(env.action_space.shape, dtype=np.float32)
            a[: env.drive_dim] = drive * fwd
            a[-1] = 0.5

            def v_fwd():
                # Whole-robot (chassis + plungers + wheels) CoM velocity, not the body
                # origin's: a pitching chassis moves its origin, and the plunger carries
                # momentum the chassis alone does not show.
                mujoco.mj_subtreeVel(m, d)
                return float(d.subtree_linvel[r.body][:2] @ r.forward(d)[:2])

            for _ in range(60):
                env.step(a)
            v0 = v_fwd()
            a[env.drive_dim + j] = 1.0
            env.step(a)
            a[env.drive_dim + j] = 0.0
            v1 = v_fwd()
            best, v5 = 0.0, None
            for k in range(30):
                env.step(a)
                if k == 3:
                    v5 = v_fwd()
                best = max(best, float(np.linalg.norm(env.ball.vel(d))))
            print(
                f"tier {tier} {name} chassis {v0:.2f} m/s: ball {best:.2f} abs, {best - v0:.2f} rel; "
                f"dv robot {v1 - v0:+.3f} (1 step) {v5 - v0:+.3f} (5 steps), "
                f"ball p {0.046 * (best - v0):.3f} N s vs robot p {m.body_subtreemass[r.body] * (v5 - v0):+.3f}"
            )
    env.close()
