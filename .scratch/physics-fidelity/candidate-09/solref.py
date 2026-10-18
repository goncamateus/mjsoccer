"""Can a softer plate contact make dt = 2 ms match the converged kick? (ticket 09)

Pre-table history: patches scene.KICK_FORCE, which ticket 09 then removed, so it only
reproduces its .txt at the parent commit (061b887). calibrate.py is the live one.

    uv run python .scratch/physics-fidelity/candidate-09/solref.py

Plate solref timeconst patched at runtime on every kicker/chipper geom; dt/8 (0.25 ms) with
the stock 0.004 is the reference.
"""

import gymnasium as gym
import mujoco
import numpy as np

import mjsoccer  # noqa: F401
from mjsoccer import scene

FORCES = range(20, 201, 20)
for tier in (0, 1):
    for div, tc in ((8, 0.004), (1, 0.004), (1, 0.008), (1, 0.012), (1, 0.02), (4, 0.012)):
        scene.build.cache_clear()  # the MjModel is shared and we mutate it
        env = gym.make("Dribbling-v0", tier=tier).unwrapped
        m = env.model
        m.opt.timestep /= div
        env.decimation *= div
        for g in range(m.ngeom):
            body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g]))
            if body.endswith(("_kicker", "_chipper")):
                m.geom_solref[g, 0] = tc
        for j, name in ((0, "kick"), (1, "chip")):
            row = []
            for force in FORCES:
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
            print(f"tier {tier} dt/{div} tc {tc} {name}  " + "  ".join(row))
        env.close()
