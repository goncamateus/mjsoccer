"""Ticket 03 step 3: random-action tip-over baseline, tier 1, seeds 0-49.

    uv run python .scratch/physics-fidelity/baseline/baseline.py <env-id>

Writes episodes_<env>.csv (one row per episode) and tilt_<env>.npy (per-step chassis
tilt in degrees of every controlled robot, all episodes concatenated).
"""
import csv
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

import mjsoccer  # noqa: F401

env_id = sys.argv[1]
out = Path(__file__).parent
env = gym.make(env_id, tier=1)
base = env.unwrapped
rows, tilts = [], []
t0 = time.perf_counter()
for seed in range(50):
    env.reset(seed=seed)
    env.action_space.seed(seed)
    steps, peak = 0, 0.0
    while True:
        _, _, term, trunc, info = env.step(env.action_space.sample())
        steps += 1
        for r in base.robots[: base.n_controlled]:
            c = base.data.xmat[r.body].reshape(3, 3)[2, 2]
            t = float(np.degrees(np.arccos(np.clip(c, -1, 1))))
            tilts.append(t)
            peak = max(peak, t)
        if term or trunc:
            break
    reason = ("tipped" if info.get("tipped") else "ball_out" if info.get("ball_out")
              else "goal" if info.get("goal") else "terminated" if term else "truncated")
    rows.append(dict(env=env_id, seed=seed, steps=steps, sim_s=round(steps * base.dt, 4),
                     reason=reason, tipped=bool(info.get("tipped")),
                     t_tip_s=round(steps * base.dt, 4) if info.get("tipped") else "",
                     peak_tilt_deg=round(peak, 3)))
wall = time.perf_counter() - t0
env.close()

with open(out / f"episodes_{env_id}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
np.save(out / f"tilt_{env_id}.npy", np.asarray(tilts, dtype=np.float32))
n_steps = sum(r["steps"] for r in rows)
minutes = sum(r["sim_s"] for r in rows) / 60
tips = sum(r["tipped"] for r in rows)
p50, p95, p99 = np.percentile(tilts, [50, 95, 99])
print(f"{env_id},{len(rows)},{n_steps},{minutes:.3f},{tips},{tips / minutes:.4f},"
      f"{p50:.3f},{p95:.3f},{p99:.3f},{max(tilts):.3f},{n_steps / wall:.1f}")
