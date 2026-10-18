"""holds() steps out of 120 idle ones, seeds 0-7, both tiers, on the 1/960 s step (ticket 13).
Also the tier-1 full-spin peak yaw rate the way test_full_spin_stays_under_max_w measures it.

    uv run python .scratch/physics-fidelity/candidate-13/held.py > .scratch/physics-fidelity/candidate-13/held.txt
"""

import gymnasium as gym
import mujoco
import numpy as np

import mjsoccer  # noqa: F401
from mjsoccer import scene as S

for env_id, idx in (("ContestedPossession-v0", 1), ("Dribbling-v0", 0)):
    for tier in (0, 1):
        env = gym.make(env_id, tier=tier).unwrapped
        out = []
        for seed in range(8):
            env.reset(seed=seed)
            hold = np.zeros(env.action_space.shape, dtype=np.float32)
            hold[-1] = 0.5
            out.append(sum(int(env.holds(idx)) for _ in range(120) if env.step(hold) or True))
        print(env_id, "tier", tier, out)

model, robots, ball = S.build(1, 0, tier=1)
r = robots[0]
data = mujoco.MjData(model)
r.place(data, 0.0, 0.0, 0.0)
ball.place(data, 4.0, 3.0)
mujoco.mj_forward(model, data)
peak_w = peak_tilt = 0.0
for _ in range(round(5.0 / model.opt.timestep)):
    data.ctrl[list(r.drive)] = r.drive_range[:, 1]
    mujoco.mj_step(model, data)
    R = data.xmat[r.body].reshape(3, 3)
    peak_w = max(peak_w, abs((R @ data.qvel[r.qvel_adr + 3 : r.qvel_adr + 6])[2]))
    peak_tilt = max(peak_tilt, np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))))
print(f"full spin tier 1: peak {peak_w:.2f} rad/s, tilt {peak_tilt:.2f} deg")
