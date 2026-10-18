"""Why the kick curve saws: plunger/ball contacts by body, per substep around the 120->130 N drop (ticket 09).

Pre-table history: patches scene.KICK_FORCE, which ticket 09 then removed, so it only
reproduces its .txt at the parent commit (061b887). calibrate.py is the live one.

    uv run python .scratch/physics-fidelity/candidate-09/rehit.py
"""

import gymnasium as gym
import mujoco
import numpy as np

import mjsoccer  # noqa: F401
from mjsoccer import scene

env = gym.make("Dribbling-v0", tier=1).unwrapped
m, d = env.model, env.data
kicker = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "b0_kicker")
jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "b0_kicker_slide")
for force in (110, 120, 130, 140):
    scene.KICK_FORCE = float(force)
    env.reset(seed=0)
    a = np.zeros(env.action_space.shape, dtype=np.float32)
    a[env.drive_dim] = 1.0
    env.data.ctrl[:] = 0
    env._apply(a.reshape(1, -1))
    trace = []
    for sub in range(40):
        if sub == env.decimation:
            d.ctrl[:] = 0
        mujoco.mj_step(m, d)
        other = sorted(
            mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[c.geom1] + m.geom_bodyid[c.geom2] - env.ball.body))
            for c in d.contact[: d.ncon]
            if env.ball.body in (m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2])
        )
        if other:
            trace.append(f"{sub}:{'+'.join(other)}")
    q = d.qpos[m.jnt_qposadr[jid]]
    v = np.linalg.norm(env.ball.vel(d))
    print(f"{force} N  {' '.join(trace)}  ball {v:.2f} m/s")
