"""Measure scene.KICK_TABLE_MS: exit speed at each force on KICK_TABLE_N (ticket 09).

    uv run python .scratch/physics-fidelity/candidate-09/calibrate.py

Seated and settled ball (Dribbling-v0), robot and dribbler stopped, one firing step through the real
env.step (so with FIRING_SUBSTEPS), peak 3-D ball speed over 0.5 s. Each force is applied
by pinning that plunger's table to the straight line (0, 0)-(KICK_SPEED_MAX, F), so a = 1
means exactly F. Prints the literal (each row a prefix of KICK_TABLE_N) to paste into scene.py.
"""

import gymnasium as gym
import numpy as np

import mjsoccer  # noqa: F401
from mjsoccer import scene

FORCES = np.arange(0.0, 501.0, 25.0)
# Steps standing before the shot. seat_ball hands the ball over still settling, and a tier 1
# chip fired straight away left ~9 % slower than one fired once the ball sits (seating.py).
SETTLE = 30
saved_n, saved_ms = scene.KICK_TABLE_N, dict(scene.KICK_TABLE_MS)
table = {}
for tier in (0, 1):
    env = gym.make("Dribbling-v0", tier=tier).unwrapped
    for j in (0, 1):
        speeds = []
        for force in FORCES:
            scene.KICK_TABLE_N = (0.0, force)  # base slices it to the row length, 2
            scene.KICK_TABLE_MS[tier, j] = (0.0, scene.KICK_SPEED_MAX)
            env.reset(seed=0)
            a = np.zeros(env.action_space.shape, dtype=np.float32)
            for _ in range(SETTLE):
                env.step(a)
            a[env.drive_dim + j] = 1.0
            best = 0.0
            for _ in range(31):
                env.step(a)
                a[env.drive_dim + j] = 0.0
                best = max(best, float(np.linalg.norm(env.ball.vel(env.data))))
            speeds.append(0.0 if force == 0 else round(best, 2))
        # Keep the prefix up to the first point past the cap: above it the curve may saw,
        # and np.interp needs it strictly increasing.
        cut = next(i for i, v in enumerate(speeds) if v >= scene.KICK_SPEED_MAX) + 1
        table[tier, j] = tuple(speeds[:cut])
        assert all(np.diff(table[tier, j]) > 0), (tier, j, table[tier, j])
        print(f"# tier {tier} {'kick chip'.split()[j]}: {scene.KICK_SPEED_MAX} m/s at ~{np.interp(scene.KICK_SPEED_MAX, speeds[:cut], FORCES[:cut]):.0f} N; full row {speeds}")
    env.close()
scene.KICK_TABLE_N, scene.KICK_TABLE_MS = saved_n, saved_ms
print(f"KICK_TABLE_N = tuple(np.arange(0.0, 501.0, 25.0))")
print("KICK_TABLE_MS = {")
for k, v in table.items():
    print(f"    {k}: {v},")
print("}")
