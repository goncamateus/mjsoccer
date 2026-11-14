"""Ticket 14's validation harness: the §13.2 scenarios, the §13.1 metrics, written to CSV,
on either backend (gpu-port ticket 12).

Every scenario is a scripted sequence against the unwrapped env, no wrappers and no
trained policy. `tests/test_validation.py` asserts the acceptance criteria and the MJX x CPU
parity on the same scenarios; this script produces the distributions behind them.

    uv run python scripts/validation_sweep.py --out runs/validation --seeds 0-4   # all, CPU
    uv run python scripts/validation_sweep.py --scenario spin --out runs/v
    # the MJX-Warp backend (needs `uv sync --group gpu` and a CUDA GPU; exit 77 without):
    systemd-run --user --scope -p MemoryMax=10G \\
        uv run --group gpu python scripts/validation_sweep.py --backend mjx --seeds 0-4 \\
        --out runs/validation-mjx

    scenario      env              parameter
    spin          GoToBall         raw yaw command k in 0.2..1.0, 7.5 s
    kick_recoil   Dribbling        full kick from 0/2/4 m/s of chassis speed, raw drive
    envelope      GoToBall         full raw command on vx / vy / omega from rest, 3 s
    kick          Dribbling        kick or chip at a = 0.5 / 1 off the seated ball, standing
    bounce        GoToBall         ball thrown at the floor / wall at 1 / 6.5 m/s, no gravity
    hold_turn     Dribbling        seated ball, raw 2 rad/s turn, bar at full speed, 3 s
    navigate      GoToBall         target pose 0.5/2/6 m ahead, turned 0/90/180 deg, 10 s
    random        all six ids      action_space.sample()
    competent     all six ids      goto_ball_policy

All but `navigate`, `random` and `competent` go through the env's raw velocity path
(`step_raw`, body-frame (vx, vy, omega) scaled by ctrlrange), not the navigation controller.
Only `random` and `competent` depend on the seed. The scripted scenarios write the whole
initial state over the reset, so they run once per parameter rather than once per seed.
The kick-recoil speed is written into qvel, so the robot starts moving rather than
accelerating up to it.

Backends. `cpu` runs one episode per process (`--jobs`). `mjx` runs every (scenario, param,
seed) of an env as one world of a batch on MJX-Warp. A scripted scenario is set up on the CPU
env and its qpos/qvel (and xfrc_applied) written into its world, so both backends start from
the same state; a seeded one resets on MJX from `PRNGKey(seed)` (no seed-by-seed pairing:
JAX's RNG is not NumPy's) and the policy acts on the world's own observations. The MJX import
is lazy, in that branch only. MJX rows have no impulse hook, no `success` (the jitted step
returns no `info`), and `reason` is only terminated/truncated.

Writes into --out (every row carries `backend`):
    episodes.csv   one row per episode
    summary.csv    one row per (backend, scenario, env, param): yaw percentiles, possession,
                   return -- medians and IQRs for the skewed ones
    navigation.csv one row per navigate episode: final error, settle time against the
                   analytic trapezoid, chatter RMS, the controller's peak omega
    impulse.csv    per-step contact impulse on the controlled robot, by contact pair (CPU)
    throughput.csv steps/s per env, one process, no impulse hook (CPU, with --scenario random)
"""

from __future__ import annotations

import argparse
import collections
import csv
import multiprocessing
import os
import time
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

import mjsoccer
from mjsoccer import scene
from mjsoccer.base import MAX_V, MAX_W
from mjsoccer.tasks import goto_ball_policy

BACKENDS = ("cpu", "mjx")
NO_MJX = 77  # main()'s exit code when the MJX backend cannot run here (pytest's "skipped")


# ------------------------------------------------------------------ scenario set-up
# Each takes a freshly reset unwrapped CPU env and the parameter, writes the initial state,
# and returns the policy: (obs, step) -> action. A policy may carry `metrics(trace, dt)`,
# the scenario's own numbers off the recorded trace (see `_row`).


def _action(env) -> np.ndarray:
    return np.zeros(env.action_space.shape, dtype=np.float32)


def _raw(env) -> np.ndarray:
    """(vx, vy, omega, kick, chip, dribble) for `step_raw`."""
    return np.zeros(6)


def _far_ball(env) -> None:
    env.ball.place(env.data, 4.0, -3.0)


def _line_up_yellows(env, y: float) -> None:
    """The yellows on a touchline, 2 m apart, out of the way."""
    for k, r in enumerate(env.robots[env.n_blue :]):
        r.place(env.data, -5.0 + 2.0 * k, y, 0.0)


def launch(env, i: int, speed: float) -> np.ndarray:
    """Robot `i` already moving forward at `speed`. Returns the drive action that holds it;
    past MAX_V, step() clips it to full command and the chassis slows to MAX_V."""
    r, d = env.robots[i], env.data
    mujoco.mj_forward(env.model, d)
    d.qvel[r.qvel_adr : r.qvel_adr + 2] = speed * r.forward(d)[:2]  # world-axis slides
    return np.array([speed / MAX_V, 0.0, 0.0], dtype=np.float32)


def spin(env, k):
    env.robots[0].place(env.data, 0.0, 0.0, 0.0)
    _far_ball(env)
    a = _raw(env)
    a[2] = k
    return lambda obs, t: a


def kick_recoil(env, v):
    r = env.robots[0]
    r.place(env.data, -4.0, 0.0, 0.0)  # 4 m/s for a second needs the room
    mujoco.mj_forward(env.model, env.data)
    env.seat_ball(r)
    a = _raw(env)
    a[:3] = launch(env, 0, v)
    env.data.qvel[env.ball.qvel_adr] = v  # carried along

    def policy(obs, t):
        b = a.copy()
        b[3] = 1.0 if t == 0 else 0.0  # kick
        return b

    return policy


# The chassis envelope (tests/test_envs.py::test_chassis_has_the_hardware_envelope, ADR
# 0001): per axis, the acceleration and top speed a full raw command reaches.
# axis name -> (qvel index, a_max, v_max)
ENVELOPE = {"vx": (0, 1.30, 3.30), "vy": (1, 1.30, 3.30), "omega": (2, 55.0, 37.1)}
ENVELOPE_REL = 0.03  # the CPU's own test, at 1/960 s; holds at 40 Hz too (see the ticket)


def envelope(env, axis):
    """From rest at (-5, -4) facing +x, the full raw command on one axis for 3 s
    (3.30 m/s at 1.30 m/s^2 takes 2.54 s; 7 m of room before a wall)."""
    i = ENVELOPE[axis][0]
    env.robots[0].place(env.data, -5.0, -4.0, 0.0)
    _line_up_yellows(env, 4.2)
    env.ball.place(env.data, 4.0, 3.0)
    a = _raw(env)
    a[i] = 1.0

    def policy(obs, t):
        return a

    def metrics(tr, dt):
        v = tr["vel0"][:, i]
        return dict(env_accel=round(float(np.diff(v, prepend=0.0).max() / dt), 4),
                    env_v_end=round(float(v[-1]), 4))

    policy.metrics = metrics
    return policy


# tests/test_envs.py::test_kick_exit_speed: a in (0, 1] asks for KICK_SPEED_MAX * a.
KICK_GRID = [(p, a) for p in ("kick", "chip") for a in (0.5, 1.0)]
KICK_ABS = 0.1  # m/s, the table's own tolerance


def kick(env, param):
    """Seated, settled ball (30 still raw steps, as calibrate.py), robot standing, dribbler
    off; one shot at `a` on the kicker or the chipper, peak 3-D ball speed over 31 steps."""
    plunger, a = param
    j = 3 + ("kick", "chip").index(plunger)
    for _ in range(30):
        env.step_raw(_raw(env))

    def policy(obs, t):
        b = _raw(env)
        b[j] = a if t == 0 else 0.0
        return b

    def metrics(tr, dt):
        return dict(kick_speed=round(float(np.linalg.norm(tr["ball_v"], axis=1).max()), 4),
                    kick_target=round(scene.KICK_SPEED_MAX * a, 4))

    policy.metrics = metrics
    return policy


# tests/test_envs.py::test_ball_bounces_at_its_measured_cor: COR 0.47-0.56 off the carpet
# and the wall, at 1 m/s and the top kick speed.
BOUNCE_GRID = [(s, v) for s in ("floor", "wall") for v in (1.0, 6.5)]
COR_RANGE = (0.47, 0.56)
WALL_Y = 4.7865  # inner face of col_f_walls_1_c05 at x = 0.865, by contact onset (ticket 17)
BOUNCE_GAP = 0.0013  # m from the surface at t = 0


def bounce(env, param):
    """The ball `BOUNCE_GAP` off the surface, moving into it at v, gravity cancelled by an
    applied force so it only ever touches that surface and leaves it at a constant speed:
    COR = v_out / v_in, read 0.2 s later. Robot and yellows parked away from it."""
    surface, v = param
    env.robots[0].place(env.data, -3.0, 0.0, 0.0)
    _line_up_yellows(env, -4.2)
    b, d = env.ball, env.data
    b.place(d, 0.865, 0.0)
    q, w = b.qpos_adr, b.qvel_adr
    axis, sign = (2, -1.0) if surface == "floor" else (1, 1.0)
    if surface == "floor":
        d.qpos[q + 2] = scene.FLOOR_Z + b.radius + BOUNCE_GAP
    else:
        d.qpos[q + 1 : q + 3] = (WALL_Y - b.radius - BOUNCE_GAP, scene.FLOOR_Z + b.radius + 0.05)
    d.qvel[w + axis] = sign * v
    d.xfrc_applied[b.body, 2] = -env.model.body_mass[b.body] * env.model.opt.gravity[2]
    a = _raw(env)

    def policy(obs, t):
        return a

    def metrics(tr, dt):
        return dict(cor=round(float(-sign * tr["ball_v"][-1, axis] / v), 5),
                    ball_beta=tr.get("ball_beta", ""))

    policy.metrics = metrics
    return policy


# tests/test_envs.py::test_seated_ball_survives_a_2_rad_s_turn: 78/120 on every seed at
# 40 Hz; the bar there sits 5 % under.
HOLD_TURN_MIN = 74 / 120


def hold_turn(env, w):
    a = _raw(env)
    a[2], a[5] = w / MAX_W, 1.0
    return lambda obs, t: a


# Ticket 05's navigation criteria, pinned to the CPU measurement plus margin
# (.scratch/gpu-port-impl/issues/05-navigation-acceptance-criteria.md). Worst cell measured
# on the grid in the comment; the ticket's starting point in brackets.
NAV_POS_ERR = 0.01  # m, final; 0.0019 [0.05]
NAV_YAW_ERR_DEG = 2.5  # final; 2.37 -- the controller's 2.5 deg deadzone, cannot go lower
NAV_TIME_RATIO = 1.05  # settle-within-NAV_POS_ERR time / trapezoid_time; 0.950 [1.25]
NAV_CHATTER_RMS = 0.05  # m/s, chassis speed over the last second; 0.0444 [0.05]
NAV_GRID = [(d, t) for d in (0.5, 2.0, 6.0) for t in (0, 90, 180)]
NAV_START = (-3.0, 2.0)


def trapezoid_time(d: float, v_max: float = MAX_V, a_up: float = 1.30, a_down: float = 1.0) -> float:
    """Rest-to-rest time over d metres: up at the chassis's 1.30 m/s^2, cruise at v_max,
    down at the controller's planned 1.0 m/s^2. A triangle when d is too short to cruise."""
    ramps = v_max**2 / 2 * (1 / a_up + 1 / a_down)
    if d <= ramps:
        v = (2 * d / (1 / a_up + 1 / a_down)) ** 0.5
        return v / a_up + v / a_down
    return v_max / a_up + v_max / a_down + (d - ramps) / v_max


def navigate(env, param):
    """From rest at NAV_START facing +x, to a target pose d m along +x turned by t degrees.
    The six yellows and the ball are lined up on the far touchline, out of the way."""
    d, t = param
    x0, y0 = NAV_START
    env.robots[0].place(env.data, x0, y0, 0.0)
    _line_up_yellows(env, -4.2)
    env.ball.place(env.data, 0.0, -3.5)
    th = np.deg2rad(t)
    a = _action(env)
    a[:4] = ((x0 + d) / scene.FIELD_HALF_X, y0 / scene.FIELD_HALF_Y, np.sin(th), np.cos(th))
    xy0 = np.asarray(env.robots[0].xy0)

    def policy(obs, step):
        return a

    def metrics(tr, dt):
        q, v = tr["qpos0"], tr["vel0"]
        return nav_metrics(dt, (x0 + d, y0, th, d), q[:, :2] + xy0, q[:, 2],
                           np.hypot(v[:, 0], v[:, 1]), tr["cmd_w"])

    policy.metrics = metrics
    return policy


def nav_metrics(dt: float, target, xy: np.ndarray, yaw: np.ndarray, speed: np.ndarray,
                cmd_w: np.ndarray) -> dict:
    """Ticket 05's four numbers off one navigate episode, one sample per control step."""
    tx, ty, th, d = target
    err = np.hypot(xy[:, 0] - tx, xy[:, 1] - ty)
    yaw_err = np.abs((yaw - th + np.pi) % (2 * np.pi) - np.pi)
    out = np.flatnonzero(err >= NAV_POS_ERR)  # settled = inside the tolerance from then on
    settle = (out[-1] + 1 if len(out) else 0) * dt if len(out) < len(err) else np.inf
    out_y = np.flatnonzero(np.rad2deg(yaw_err) >= NAV_YAW_ERR_DEG)
    settle_y = (out_y[-1] + 1 if len(out_y) else 0) * dt if len(out_y) < len(err) else np.inf
    last = max(1, round(1.0 / dt))
    return dict(
        nav_pos_err=round(float(err[-1]), 5),
        nav_yaw_err_deg=round(float(np.rad2deg(yaw_err[-1])), 4),
        nav_settle_s=round(float(settle), 4), nav_yaw_settle_s=round(float(settle_y), 4),
        nav_trap_s=round(trapezoid_time(d), 4),
        nav_time_ratio=round(float(settle / trapezoid_time(d)), 4),
        nav_chatter_rms=round(float(np.sqrt(np.mean(speed[-last:] ** 2))), 5),
        nav_cmd_w_max=round(float(np.abs(cmd_w).max()), 4),
    )


def random(env, _):
    return lambda obs, t: env.action_space.sample()


def competent(env, _):
    def policy(obs, t):
        a = _action(env)
        a[:4] = goto_ball_policy(obs)
        return a

    return policy


ALL = list(mjsoccer.ENV_IDS)
# name -> (set-up, [(env, param)], control steps or None for the whole episode, seeded,
#          raw velocity path)
SCENARIOS = {
    "spin": (spin, [("GoToBall-v0", k) for k in (0.2, 0.4, 0.6, 0.8, 1.0)], 300, False, True),
    "kick_recoil": (kick_recoil, [("Dribbling-v0", v) for v in (0, 2, 4)], 60, False, True),
    "envelope": (envelope, [("GoToBall-v0", k) for k in ENVELOPE], 120, False, True),
    "kick": (kick, [("Dribbling-v0", p) for p in KICK_GRID], 31, False, True),
    "bounce": (bounce, [("GoToBall-v0", p) for p in BOUNCE_GRID], 8, False, True),
    "hold_turn": (hold_turn, [("Dribbling-v0", 2.0)], 120, False, True),
    "navigate": (navigate, [("GoToBall-v0", p) for p in NAV_GRID], 400, False, False),
    "random": (random, [(e, None) for e in ALL], None, True, False),
    "competent": (competent, [(e, None) for e in ALL], None, True, False),
}


def units(names=None, seeds=range(5)) -> list[tuple]:
    """(scenario, env, param, seed) for every episode of `names` (default: all): the
    scripted ones once (seed 0), the seeded ones once per seed."""
    names = names or list(SCENARIOS)
    return [(n, e, p, s) for n in names for e, p in SCENARIOS[n][1]
            for s in (seeds if SCENARIOS[n][3] else [0])]


# ------------------------------------------------------------- contact impulse hook

ROBOT = {"body", "mech"}
_mj_step = mujoco.mj_step
_ctx = None  # (geom classes, body roots, controlled roots) of the env being measured
_impulse = collections.Counter()  # pair -> N*s this control step


def _classify(m) -> np.ndarray:
    out = []
    for g in range(m.ngeom):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        out.append("floor" if n == "floor" else "wall" if "walls" in n else "goal" if "goalpost" in n
                   else "ball" if n == "ball" else "mech" if m.geom_contype[g] == 2 else "body")
    return np.array(out)


def _hooked(m, d, *args):
    """mj_step, then |F| * dt of every contact touching a controlled robot, by pair."""
    _mj_step(m, d, *args)
    if _ctx is None or not d.ncon:
        return
    cls, root, mine = _ctx
    geoms = d.contact.geom[: d.ncon]
    roots = root[m.geom_bodyid[geoms]]
    f = np.zeros(6)
    for j in np.flatnonzero(np.isin(roots, mine).any(axis=1)):
        mujoco.mj_contactForce(m, d, int(j), f)
        a, b = sorted(cls[geoms[j]])
        pair = f"{a}|{b}"
        if a in ROBOT and b in ROBOT and roots[j, 0] != roots[j, 1]:
            pair = "robot~" + pair
        _impulse[pair] += float(np.linalg.norm(f[:3])) * m.opt.timestep


# -------------------------------------------------------------------- one episode


def _ball_beta(model, ball) -> float:
    """The ball's I / (m r^2) as the backend's model holds it: 0.5, the measured golf ball."""
    b = ball.body
    return round(float(np.asarray(model.body_inertia)[b][0]
                       / (np.asarray(model.body_mass)[b] * ball.radius**2)), 5)


def _runs(held) -> list[int]:
    """Lengths of the possession streaks, in control steps."""
    return [len(g) for g in "".join("1" if h else "0" for h in held).split("0") if g]


def _row(scenario, env_id, param, seed, backend, dt, policy, tr) -> dict:
    """The episode's CSV row off its trace, the same for both backends. `tr`: steps, ret,
    reason, success, wall_s, and per control step held, vel0/qpos0 (robot 0's slides and
    hinge), ball_v, cmd_w (the controller's omega for robot 0, navigate only)."""
    y = np.abs(tr["vel0"][:, 2])
    held = tr["held"]
    runs = _runs(held)
    row = dict(
        backend=backend, scenario=scenario, env=env_id, param=param, seed=seed,
        steps=tr["steps"], sim_s=round(tr["steps"] * dt, 4), reason=tr["reason"],
        success=tr["success"],
        yaw_p99=round(np.percentile(y, 99), 3), yaw_max=round(y.max(), 3),
        yaw_frac_over_max_w=round(float((y > MAX_W).mean()), 5), held_frac=round(float(np.mean(held)), 4),
        poss_mean_steps=round(float(np.mean(runs)), 2) if runs else 0,
        poss_p95_steps=round(float(np.percentile(runs, 95)), 2) if runs else 0,
        ret=round(tr["ret"], 4), wall_s=round(tr["wall_s"], 3),
    )
    if hasattr(policy, "metrics"):
        row.update(policy.metrics(tr, dt))
    return row


def episode(scenario: str, env_id: str, param, seed: int, backend: str = "cpu") -> dict:
    """Run one episode and return its row plus the per-step arrays the summary pools.
    `backend="mjx"` runs it as a batch of one world (`run_mjx`)."""
    if backend == "mjx":
        return run_mjx([(scenario, env_id, param, seed)])[0]
    global _ctx
    setup, _, max_steps, _, raw = SCENARIOS[scenario]
    env = gym.make(env_id).unwrapped
    m = env.model
    root = m.body_rootid
    mine = [root[r.body] for r in env.robots[: env.n_controlled]]
    obs, _ = env.reset(seed=seed)
    env.action_space.seed(seed)
    policy = setup(env, param)
    step = env.step_raw if raw else env.step
    _ctx = (_classify(m), root, mine)
    navigate_ = env._navigate
    cmd_w = []

    def recorded(i, pose):  # the controller's own omega, before the servo's ctrlrange clip
        cmd = navigate_(i, pose)
        if i == 0:
            cmd_w.append(cmd[2])
        return cmd

    env._navigate = recorded
    r, b = env.robots[0], env.ball
    rec = collections.defaultdict(list)
    impulses, total, info = collections.defaultdict(list), 0.0, {}
    steps, terminated, truncated, t0 = 0, False, False, time.perf_counter()
    # ponytail: patches the mujoco module and keeps the hook state in globals -- one episode
    # per process at a time, which Pool gives. Threads would need a per-env hook.
    mujoco.mj_step = _hooked  # env.step calls mujoco.mj_step through the module
    try:
        while not (terminated or truncated) and (max_steps is None or steps < max_steps):
            _impulse.clear()
            obs, reward, terminated, truncated, info = step(policy(obs, steps))
            steps += 1
            total += reward
            rec["held"].append(env.holds(0))
            rec["qpos0"].append(env.data.qpos[r.qpos_adr : r.qpos_adr + 3].copy())
            rec["vel0"].append(env.data.qvel[r.qvel_adr : r.qvel_adr + 3].copy())
            rec["ball_v"].append(b.vel(env.data))
            for pair, j in _impulse.items():
                impulses[pair].append(j)
    finally:
        mujoco.mj_step, _ctx = _mj_step, None
    wall_s = time.perf_counter() - t0
    reason = ("ball_out" if info.get("ball_out") else "goal" if info.get("goal")
              else "success" if info.get("is_success") else "truncated" if truncated
              else "terminated" if terminated else "scripted_end")
    tr = {k: np.array(v) for k, v in rec.items()}
    tr.update(steps=steps, ret=total, reason=reason, success=bool(info.get("is_success")),
              wall_s=wall_s, cmd_w=np.array(cmd_w), ball_beta=_ball_beta(m, b))
    row = _row(scenario, env_id, param, seed, "cpu", env.dt, policy, tr)
    env.close()
    return dict(row=row, yaws=np.abs(tr["vel0"][:, 2]), runs=_runs(tr["held"]),
                impulses=dict(impulses))


def _unit(args):
    return episode(*args)


# ------------------------------------------------------------------------ MJX


def mjx_unavailable() -> str | None:
    """Why the MJX backend cannot run here (no `gpu` group, no CUDA device), or None."""
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")  # leave the GPU shared
    try:
        import jax
        import mujoco.mjx  # noqa: F401

        import mjsoccer.mjx  # noqa: F401
    except ImportError as e:
        return f"the gpu group is not installed ({e})"
    try:
        jax.devices("gpu")
    except RuntimeError as e:
        return f"no CUDA device ({e})"
    return None


def run_mjx(us: list[tuple]) -> list[dict]:
    """Every unit (scenario, env, param, seed) as one world of an MJX-Warp batch, one batch
    per (env, raw path). Same return as `episode`, one per unit, in order."""
    out = {}
    groups = collections.defaultdict(list)
    for u in us:
        groups[(u[1], SCENARIOS[u[0]][4])].append(u)
    for (env_id, raw), group in groups.items():
        for u, res in zip(group, _mjx_batch(env_id, raw, group)):
            out[u] = res
    return [out[u] for u in us]


def _mjx_batch(env_id: str, raw: bool, group: list[tuple]) -> list[dict]:
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax
    import jax.numpy as jnp
    from mujoco import mjx

    import mjsoccer.mjx

    n = len(group)
    env = mjsoccer.mjx.make(env_id, nworld=n)
    cpus, policies, caps = [], [], []
    given, qpos, qvel, xfrc, keys = [], [], [], [], []
    for scenario, _, param, seed in group:
        setup, _, max_steps, seeded, _ = SCENARIOS[scenario]
        cpu = gym.make(env_id).unwrapped
        cpu.reset(seed=seed)
        cpu.action_space.seed(seed)
        policies.append(setup(cpu, param))
        cpus.append(cpu)
        caps.append(max_steps or np.inf)
        given.append(not seeded)
        qpos.append(cpu.data.qpos.copy())
        qvel.append(cpu.data.qvel.copy())
        xfrc.append(cpu.data.xfrc_applied.copy())
        keys.append(jax.random.PRNGKey(seed))
    dt = cpus[0].dt
    r, b = env.robots[0], env.ball

    def init(s, qpos, qvel, xfrc, given):
        """For a scripted world, the CPU's set-up state over the reset one. (Its own jit:
        a second mjx.forward in the reset's Warp graph fails to allocate in the capture.)"""
        d = s.pipeline_state
        d = d.replace(qpos=jnp.where(given, qpos.astype(d.qpos.dtype), d.qpos),
                      qvel=jnp.where(given, qvel.astype(d.qvel.dtype), d.qvel),
                      xfrc_applied=jnp.where(given, xfrc.astype(d.xfrc_applied.dtype),
                                             d.xfrc_applied))
        d = mjx.forward(env.sys, d)
        return env.init_state(d, s.info["world"]["rng"], s.metrics["reset_overlap"])

    def one(s, a):
        cmd_w = jnp.zeros(()) if raw else env._navigate(s.pipeline_state, 0, a[:4])[2]
        s = env.step(s, a, raw)
        d = s.pipeline_state
        return s, dict(
            reward=s.reward, done=s.done, trunc=s.info["truncation"],
            held=s.info["world"]["grace"][0] > 0,
            qpos0=d.qpos[r.qpos_adr : r.qpos_adr + 3], vel0=d.qvel[r.qvel_adr : r.qvel_adr + 3],
            ball_v=d.qvel[b.qvel_adr : b.qvel_adr + 3], cmd_w=cmd_w,
            diverged=s.metrics["diverged"], overflow=s.metrics["overflow"],
        )

    t0 = time.perf_counter()
    state = jax.jit(jax.vmap(env.reset))(jnp.stack(keys))
    state = jax.jit(jax.vmap(init))(state, jnp.asarray(np.array(qpos)),
                                    jnp.asarray(np.array(qvel)), jnp.asarray(np.array(xfrc)),
                                    jnp.asarray(given))
    step = jax.jit(jax.vmap(one))
    dim = 6 if raw else env.act_dim
    rec = [collections.defaultdict(list) for _ in range(n)]
    total, steps, ends = np.zeros(n), np.zeros(n, int), [""] * n
    active = np.ones(n, bool)
    obs = np.asarray(state.obs)
    t = 0
    while active.any():
        a = np.zeros((n, env.n_controlled * dim), np.float32)
        for i in np.flatnonzero(active):
            a[i] = policies[i](obs[i], t)
        state, out = step(state, jnp.asarray(a))
        out = jax.device_get(out)
        obs = np.asarray(state.obs)
        bad = active & ((out["diverged"] > 0) | (out["overflow"] > 0))
        if bad.any():  # an ended world may wander off; a live one must not
            mjsoccer.mjx.check_failures(state)
        for i in np.flatnonzero(active):
            for k in ("held", "qpos0", "vel0", "ball_v", "cmd_w"):
                rec[i][k].append(out[k][i])
            total[i] += out["reward"][i]
            steps[i] += 1
            if out["done"][i] > 0:
                ends[i] = "truncated" if out["trunc"][i] > 0 else "terminated"
                active[i] = False
            elif steps[i] >= caps[i]:
                ends[i] = "scripted_end"
                active[i] = False
        t += 1
    wall_s = time.perf_counter() - t0
    beta = _ball_beta(env.sys, b)
    results = []
    for i, (scenario, _, param, seed) in enumerate(group):
        tr = {k: np.array(v) for k, v in rec[i].items()}
        tr.update(steps=int(steps[i]), ret=float(total[i]), reason=ends[i], success="",
                  wall_s=wall_s / n, ball_beta=beta)
        row = _row(scenario, env_id, param, seed, "mjx", dt, policies[i], tr)
        results.append(dict(row=row, yaws=np.abs(tr["vel0"][:, 2]), runs=_runs(tr["held"]),
                            impulses={}))
        cpus[i].close()
    return results


def run(us: list[tuple], backend: str = "cpu", jobs: int | None = None,
        progress: bool = False) -> list[dict]:
    """`episode` over every unit, in order: a process pool on the CPU, batches on MJX."""
    if backend == "mjx":
        return run_mjx(us)
    jobs = jobs or max(1, multiprocessing.cpu_count() - 1)
    # Longest first, so the 7-robot episodes do not start last.
    order = sorted(range(len(us)), key=lambda i: us[i][1] == "StaticDefenders-v0", reverse=True)
    out = [None] * len(us)
    with multiprocessing.Pool(jobs) as pool:
        for k, (i, r) in enumerate(zip(order, pool.imap(_unit, [us[i] for i in order])), 1):
            out[i] = r
            if progress:
                print(f"\r{k}/{len(us)}", end="", flush=True)
    if progress:
        print()
    return out


# --------------------------------------------------------------------- summarising


def _q(x, qs):
    qs = np.atleast_1d(qs)
    return [round(float(v), 4) for v in np.percentile(x, qs)] if len(x) else [""] * len(qs)


def summarise(results: list[dict], out: Path) -> None:
    rows = [r["row"] for r in results]
    with open(out / "episodes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)), restval="")
        w.writeheader()
        w.writerows(rows)
    nav = [r for r in rows if "nav_pos_err" in r]
    if nav:
        with open(out / "navigation.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(nav[0]))
            w.writeheader()
            w.writerows(nav)

    cells = collections.defaultdict(list)
    for r in results:
        k = r["row"]
        cells[(k["backend"], k["scenario"], k["env"], str(k["param"]))].append(r)

    summary, impulse = [], []
    for key, rs in cells.items():
        rr = [r["row"] for r in rs]
        y = np.concatenate([r["yaws"] for r in rs])
        runs = sum((r["runs"] for r in rs), [])
        minutes = sum(r["sim_s"] for r in rr) / 60
        ret = [r["ret"] for r in rr]
        held = [r["held_frac"] for r in rr]
        succ = [r["success"] for r in rr if r["success"] != ""]
        cell = dict(backend=key[0], scenario=key[1], env=key[2], param=key[3])
        summary.append(dict(
            **cell, episodes=len(rr),
            minutes=round(minutes, 3),
            yaw_p50=_q(y, 50)[0], yaw_p99=_q(y, 99)[0], yaw_max=round(y.max(), 3),
            yaw_frac_over_max_w=round(float((y > MAX_W).mean()), 5),
            held_frac_p50=_q(held, 50)[0], poss_p50_steps=_q(runs, 50)[0],
            poss_mean_steps=round(float(np.mean(runs)), 2) if runs else "", poss_p95_steps=_q(runs, 95)[0],
            ret_p25_p50_p75=" ".join(map(str, _q(ret, (25, 50, 75)))),
            success_rate=round(float(np.mean(succ)), 3) if succ else "",
            steps_per_s_pooled=round(sum(r["steps"] for r in rr) / sum(r["wall_s"] for r in rr), 1),
        ))
        pairs = collections.defaultdict(list)
        for r in rs:
            for p, js in r["impulses"].items():
                pairs[p] += js
        for p, js in sorted(pairs.items(), key=lambda kv: -max(kv[1])):
            impulse.append(dict(**cell, pair=p, steps_active=len(js),
                                **dict(zip(("p50", "p95", "p99"), _q(js, (50, 95, 99)))),
                                max=round(max(js), 5)))

    for name, table in (("summary", summary), ("impulse", impulse)):
        if not table:  # navigate alone touches nothing; MJX has no impulse hook
            continue
        with open(out / f"{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(table[0]))
            w.writeheader()
            w.writerows(table)


def throughput(out: Path, steps: int = 600) -> None:
    """steps/s of random actions, one process, no hook -- comparable to docs/index.md."""
    with open(out / "throughput.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("env", "steps_per_s"))
        for env_id in ALL:
            env = gym.make(env_id)
            env.reset(seed=0)
            env.action_space.seed(0)
            t0 = time.perf_counter()
            for _ in range(steps):
                *_, te, tr, _ = env.step(env.action_space.sample())
                if te or tr:
                    env.reset()
            w.writerow((env_id, round(steps / (time.perf_counter() - t0), 1)))
            env.close()


def _seeds(s: str) -> list[int]:
    lo, _, hi = s.partition("-")
    return list(range(int(lo), int(hi or lo) + 1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", action="append", choices=list(SCENARIOS), help="default: all")
    ap.add_argument("--seeds", default="0-4", help="for the seeded scenarios, e.g. 0-4 or 3")
    ap.add_argument("--backend", choices=BACKENDS, default="cpu")
    ap.add_argument("--jobs", type=int, default=max(1, multiprocessing.cpu_count() - 1),
                    help="CPU worker processes")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.backend == "mjx" and (why := mjx_unavailable()):
        print(f"MJX backend unavailable: {why}")
        return NO_MJX
    args.out.mkdir(parents=True, exist_ok=True)

    names = args.scenario or list(SCENARIOS)
    results = run(units(names, _seeds(args.seeds)), args.backend, args.jobs, progress=True)
    summarise(results, args.out)
    if "random" in names and args.backend == "cpu":
        throughput(args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
