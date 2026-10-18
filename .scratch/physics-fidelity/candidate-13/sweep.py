"""Ticket 13: resolution sweep. Every behaviour metric at dt = 0.5, 1, 2, 4 ms (and 0.25 ms on tier 1).

    uv run python .scratch/physics-fidelity/candidate-13/sweep.py <tier> [dt_ms,...] > .scratch/physics-fidelity/candidate-13/sweep_t<tier>.txt

Exactly, as run (with FIRING_SUBSTEPS = 8, before ticket 13 set it to 4):

    uv run python .scratch/physics-fidelity/candidate-13/sweep.py 0 > .scratch/physics-fidelity/candidate-13/sweep_t0.txt
    uv run python .scratch/physics-fidelity/candidate-13/sweep.py 1 > .scratch/physics-fidelity/candidate-13/sweep_t1.txt
    uv run python .scratch/physics-fidelity/candidate-13/sweep.py 1 0.25 > .scratch/physics-fidelity/candidate-13/sweep_t1_025.txt

The other outputs here come from earlier tickets' scripts, rerun on the 1/960 s step:

    uv run python .scratch/physics-fidelity/candidate-09/calibrate.py > .scratch/physics-fidelity/candidate-13/calibrate.txt
    uv run python .scratch/physics-fidelity/candidate-11/extremes.py <tier> > .scratch/physics-fidelity/candidate-13/extremes_t<tier>.csv
    uv run python .scratch/physics-fidelity/candidate-11/bounds.py <tier> > .scratch/physics-fidelity/candidate-13/bounds_t<tier>.csv

The control period stays 16 ms at every dt (decimation = 16 ms / dt), so only the physics
resolution changes. The firing step keeps running at dt / FIRING_SUBSTEPS (as on main);
`kick1` is the same kick with FIRING_SUBSTEPS = 1, i.e. the firing step at the global dt.
"""
import sys
import time

import gymnasium as gym
import mujoco
import numpy as np

import mjsoccer  # noqa: F401
from mjsoccer import scene

tier = int(sys.argv[1])
PERIOD = 0.016


DT = 0.002


def make(env_id):
    env = gym.make(env_id, tier=tier).unwrapped
    env.model.opt.timestep = DT  # the cached MjModel is shared, so this sticks for the next make too
    env.decimation = round(PERIOD / DT)
    return env


def drive(env, cmd, seconds, place=(-3.0, 0.0)):
    """Full drive command from rest, robot 0 alone; body vx(t), peak |yaw rate|."""
    env.reset(seed=0)
    r = env.robots[0]
    r.place(env.data, *place, 0.0)
    env.ball.place(env.data, 4.0, 3.0)
    mujoco.mj_forward(env.model, env.data)
    a = np.zeros(env.action_space.shape, dtype=np.float32)
    a[: env.drive_dim] = cmd
    vs, w = [], 0.0
    for _ in range(round(seconds / env.dt)):
        env.step(a)
        R = env.data.xmat[r.body].reshape(3, 3)
        vs.append((R.T @ env.data.qvel[r.qvel_adr : r.qvel_adr + 3])[0])
        w = max(w, abs((R @ env.data.qvel[r.qvel_adr + 3 : r.qvel_adr + 6])[2]) if tier else abs(env.data.qvel[r.qvel_adr + 2]))
    vs = np.abs(vs)
    return vs, w


def forward_cmd(env):
    if tier == 0:
        return np.array([1.0, 0.0, 0.0])
    m, d, r = env.model, env.data, env.robots[0]
    env.reset(seed=0)  # body frames are zero until the first forward
    dirs = []
    for act in r.drive:
        j = m.actuator_trnid[act, 0]
        ax = d.xmat[m.jnt_bodyid[j]].reshape(3, 3) @ m.jnt_axis[j]
        dirs.append(np.cross(ax, [0, 0, -1])[0])
    dirs = np.array(dirs)
    return dirs / np.abs(dirs).max()


def ball_metrics(env):
    m, d, b = env.model, env.data, env.ball
    env.reset(seed=0)
    for r in env.robots:
        r.place(d, -5.0, -4.0, 0.0)
    b.place(d, -3.0, 0.0)
    d.qvel[b.qvel_adr : b.qvel_adr + 3] = [3.0, 0.0, 0.0]
    d.qvel[b.qvel_adr + 3 : b.qvel_adr + 6] = [0.0, 3.0 / 0.0215, 0.0]  # rolling
    mujoco.mj_forward(m, d)
    x0 = b.pos(d)[0]
    for _ in range(round(2.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    roll = b.pos(d)[0] - x0
    # drop from 0.3 m above carpet, COR of first bounce
    env.reset(seed=0)
    for r in env.robots:
        r.place(d, -5.0, -4.0, 0.0)
    b.place(d, 0.0, 0.0)
    d.qpos[b.qpos_adr + 2] = scene.FLOOR_Z + 0.0215 + 0.3
    d.qvel[b.qvel_adr : b.qvel_adr + 6] = 0
    mujoco.mj_forward(m, d)
    vin = vout = 0.0
    up = False
    for _ in range(round(1.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
        vz = d.qvel[b.qvel_adr + 2]
        if not up:
            vin = max(vin, -vz)  # peak fall speed, i.e. at contact onset
            up = vz > 0
        elif vz > vout:
            vout = vz
        else:
            break  # past the top of the rebound's velocity: contact is over
    return roll, vout / vin


def kick(env, j, substeps):
    old = scene.FIRING_SUBSTEPS
    scene.FIRING_SUBSTEPS = substeps
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
    scene.FIRING_SUBSTEPS = old
    return out


def held(env_id, idx, seeds=8):
    env = make(env_id)
    out = []
    for s in range(seeds):
        env.reset(seed=s)
        a = np.zeros(env.action_space.shape, dtype=np.float32)
        a[-1] = 0.5
        out.append(sum(int(env.holds(idx)) for _ in range(120) if env.step(a) or True))
    return np.mean(out), min(out)


def random_stats(env_id, seeds=10, steps=600):
    env = make(env_id)
    tilts, speeds = [], []
    t0 = time.perf_counter()
    n = 0
    for s in range(seeds):
        env.reset(seed=s)
        env.action_space.seed(s)
        for _ in range(steps):
            *_, te, tr, _ = env.step(env.action_space.sample())
            n += 1
            r = env.robots[0]
            c = env.data.xmat[r.body].reshape(3, 3)[2, 2]
            tilts.append(np.degrees(np.arccos(np.clip(c, -1, 1))))
            speeds.append(np.linalg.norm(env.data.qvel[r.qvel_adr : r.qvel_adr + 2]))
            if te or tr:
                break
    sps = n / (time.perf_counter() - t0)
    return np.percentile(tilts, 99), np.mean(speeds), sps


print("dt_ms vfwd a_fwd vspin roll2s cor kick50 kick100 chip100 kick1_100 heldCP heldCPmin heldDr tilt99 meanv sps_GTB sps_SD")
for dt in [float(x) * 1e-3 for x in (sys.argv[2] if len(sys.argv) > 2 else "0.5,1,2,4").split(",")]:
    DT = dt
    env = make("GoToBall-v0")
    vs, _ = drive(env, forward_cmd(env), 2.0)
    vf = vs[-10:].mean()
    t = np.arange(1, len(vs) + 1) * env.dt
    sel = (vs > 0.1 * vf) & (vs < 0.5 * vf)
    a_fwd = np.polyfit(t[sel], vs[sel], 1)[0] if sel.sum() > 1 else float("nan")
    spin = np.ones(env.drive_dim) if tier else np.array([0.0, 0.0, 1.0])
    _, w = drive(env, spin, 3.0, place=(0.0, 0.0))
    roll, cor = ball_metrics(env)
    dr = make("Dribbling-v0")
    k50, k100 = kick(dr, 0, scene.FIRING_SUBSTEPS)
    _, c100 = kick(dr, 1, scene.FIRING_SUBSTEPS)
    _, k1 = kick(dr, 0, 1)
    hcp, hcpmin = held("ContestedPossession-v0", 1)
    hdr, _ = held("Dribbling-v0", 0)
    tilt99, meanv, sps = random_stats("GoToBall-v0")
    *_, sps_sd = random_stats("StaticDefenders-v0", seeds=2, steps=300)
    print(f"{dt*1e3:g} {vf:.3f} {a_fwd:.3f} {w:.2f} {roll:.3f} {cor:.3f} {k50:.2f} {k100:.2f} {c100:.2f} {k1:.2f} "
          f"{hcp:.1f} {hcpmin} {hdr:.1f} {tilt99:.2f} {meanv:.3f} {sps:.0f} {sps_sd:.0f}", flush=True)
