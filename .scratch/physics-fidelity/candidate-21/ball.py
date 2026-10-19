"""Ticket 21: what the ball's floor contact touches, before and after ticket 17's solref.

    uv run python .scratch/physics-fidelity/candidate-21/ball.py > .scratch/physics-fidelity/candidate-21/ball_after.txt
    MJSOCCER_MODELS=<copy of models/ at HEAD> uv run python ... > .../ball_before.txt

Envs at their defaults (1/960 s, 60 Hz). Per tier:
  rest_mm   ball penetration into the carpet after 1 s at rest
  roll2s    distance in 2 s of a ball rolling at 3 m/s (ticket 13's metric)
  cor       first bounce of a drop from 0.3 m (ticket 13's metric)
  kick50/kick100/chip100  peak ball speed after one kick / chip step from a seated ball
  chip_h    chip apex height; chip_x1 distance of the first landing; hops = bounces over
            1 cm after it in 3 s; hop2_h the second apex
  heldCP/heldDr  holds() steps out of 120 idle ones, mean (min) over seeds 0-7 (ticket 13)
"""

import gymnasium as gym
import mujoco
import numpy as np

import mjsoccer  # noqa: F401
from mjsoccer import scene

R = 0.0215


def ball_metrics(env):
    m, d, b = env.model, env.data, env.ball
    env.reset(seed=0)
    for r in env.robots:
        r.place(d, -5.0, -4.0, 0.0)
    b.place(d, 0.0, 0.0)
    mujoco.mj_forward(m, d)
    for _ in range(round(1.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    rest = (R - (b.pos(d)[2] - scene.FLOOR_Z)) * 1e3
    b.place(d, -3.0, 0.0)
    d.qvel[b.qvel_adr : b.qvel_adr + 3] = [3.0, 0.0, 0.0]
    d.qvel[b.qvel_adr + 3 : b.qvel_adr + 6] = [0.0, 3.0 / R, 0.0]  # rolling
    mujoco.mj_forward(m, d)
    x0 = b.pos(d)[0]
    for _ in range(round(2.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    roll = b.pos(d)[0] - x0
    b.place(d, 0.0, 0.0)
    d.qpos[b.qpos_adr + 2] = scene.FLOOR_Z + R + 0.3
    mujoco.mj_forward(m, d)
    vin = vout = 0.0
    up = False
    for _ in range(round(1.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
        vz = d.qvel[b.qvel_adr + 2]
        if not up:
            vin = max(vin, -vz)
            up = vz > 0
        elif vz > vout:
            vout = vz
        else:
            break
    return rest, roll, vout / vin


def kick(env, j):
    out = []
    for frac in (0.5, 1.0):
        env.reset(seed=0)
        a = np.zeros(env.action_space.shape, dtype=np.float32)
        a[env.drive_dim + j] = frac
        best = 0.0
        for _ in range(31):
            env.step(a)
            a[env.drive_dim + j] = 0.0
            best = max(best, float(np.linalg.norm(env.ball.vel(env.data))))
        out.append(best)
    return out


def chip_flight(env):
    """Full chip from a seated ball, then 3 s of flight, sampled every physics step."""
    m, d, b = env.model, env.data, env.ball
    env.reset(seed=0)
    a = np.zeros(env.action_space.shape, dtype=np.float32)
    a[env.drive_dim + 1] = 1.0
    env.step(a)
    x0 = b.pos(d)[:2]
    for r in env.robots:  # out of the way of the landing
        r.place(d, -5.0, -4.0, 0.0)
    h, xy = [], []
    for _ in range(round(3.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
        h.append(b.pos(d)[2] - scene.FLOOR_Z - R)
        xy.append(np.linalg.norm(b.pos(d)[:2] - x0))
    h = np.array(h)
    land = np.flatnonzero((h[1:] <= 1e-3) & (h[:-1] > 1e-3))
    apexes = [h[i:j].max() for i, j in zip(land[:-1], land[1:])]
    hops = sum(x > 0.01 for x in apexes)
    return h.max(), xy[land[0]] if len(land) else float("nan"), hops, apexes[0] if apexes else 0.0


def held(env_id, idx, tier, seeds=8):
    env = gym.make(env_id, tier=tier).unwrapped
    out = []
    for s in range(seeds):
        env.reset(seed=s)
        a = np.zeros(env.action_space.shape, dtype=np.float32)
        a[-1] = 0.5
        n = 0
        for _ in range(120):
            env.step(a)
            n += int(env.holds(idx))
        out.append(n)
    return np.mean(out), min(out)


print("tier rest_mm roll2s cor kick50 kick100 chip100 chip_h chip_x1 hops hop2_h heldCP heldDr")
for tier in (0, 1):
    env = gym.make("GoToBall-v0", tier=tier).unwrapped
    rest, roll, cor = ball_metrics(env)
    dr = gym.make("Dribbling-v0", tier=tier).unwrapped
    k50, k100 = kick(dr, 0)
    _, c100 = kick(dr, 1)
    ch, cx, hops, h2 = chip_flight(dr)
    hcp = held("ContestedPossession-v0", 1, tier)
    hdr = held("Dribbling-v0", 0, tier)
    print(f"{tier} {rest:.2f} {roll:.3f} {cor:.3f} {k50:.2f} {k100:.2f} {c100:.2f} {ch:.3f} {cx:.2f} "
          f"{hops} {h2:.3f} {hcp[0]:.1f}({hcp[1]}) {hdr[0]:.1f}({hdr[1]})", flush=True)
