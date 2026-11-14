"""Ticket 14's validation harness: the §13.2 scenarios, the §13.1 metrics, written to CSV.

Every scenario is a scripted sequence against the unwrapped env, no wrappers and no
trained policy. `tests/test_validation.py` asserts the acceptance criteria on the same
scenarios; this script produces the distributions behind them.

    uv run python scripts/validation_sweep.py --out runs/validation --seeds 0-4   # all
    uv run python scripts/validation_sweep.py --scenario spin --out runs/v

    scenario      env              parameter
    spin          GoToBall         raw yaw command k in 0.2..1.0, 7.5 s
    kick_recoil   Dribbling        full kick from 0/2/4 m/s of chassis speed, raw drive
    navigate      GoToBall         target pose 0.5/2/6 m ahead, turned 0/90/180 deg, 10 s
    random        all six ids      action_space.sample()
    competent     all six ids      goto_ball_policy

`spin` and `kick_recoil` are envelope tests: they go through the env's raw velocity path
(`step_raw`, body-frame (vx, vy, omega) scaled by ctrlrange), not the navigation controller.
Only `random` and `competent` depend on the seed. The scripted scenarios write the whole
initial state over the reset, so they run once per parameter rather than once per seed.
The kick-recoil speed is written into qvel, so the robot starts moving rather than
accelerating up to it.

Writes into --out:
    episodes.csv   one row per episode
    summary.csv    one row per (scenario, env, param): yaw percentiles, possession, return
                   -- medians and IQRs for the skewed ones
    navigation.csv one row per navigate episode: final error, settle time against the
                   analytic trapezoid, chatter RMS, the controller's peak omega
    impulse.csv    per-step contact impulse on the controlled robot, by contact pair
    throughput.csv steps/s per env, one process, no impulse hook (with --scenario random)
"""

from __future__ import annotations

import argparse
import collections
import csv
import multiprocessing
import time
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

import mjsoccer
from mjsoccer import scene
from mjsoccer.base import MAX_V, MAX_W
from mjsoccer.tasks import goto_ball_policy


# ------------------------------------------------------------------ scenario set-up
# Each takes a freshly reset unwrapped env and the parameter, writes the initial state, and
# returns the policy: (obs, step) -> action.


def _action(env) -> np.ndarray:
    return np.zeros(env.action_space.shape, dtype=np.float32)


def _raw(env) -> np.ndarray:
    """(vx, vy, omega, kick, chip, dribble) for `step_raw`."""
    return np.zeros(6)


def _far_ball(env) -> None:
    env.ball.place(env.data, 4.0, -3.0)


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
    for k, r in enumerate(env.robots[env.n_blue :]):
        r.place(env.data, -5.0 + 2.0 * k, -4.2, 0.0)
    env.ball.place(env.data, 0.0, -3.5)
    th = np.deg2rad(t)
    a = _action(env)
    a[:4] = ((x0 + d) / scene.FIELD_HALF_X, y0 / scene.FIELD_HALF_Y, np.sin(th), np.cos(th))

    def policy(obs, step):
        return a

    policy.target = (x0 + d, y0, th, d)
    return policy


def nav_metrics(env, target, xy: np.ndarray, yaw: np.ndarray, speed: np.ndarray,
                cmd_w: np.ndarray) -> dict:
    """Ticket 05's four numbers off one navigate episode, one sample per control step."""
    tx, ty, th, d = target
    err = np.hypot(xy[:, 0] - tx, xy[:, 1] - ty)
    yaw_err = np.abs((yaw - th + np.pi) % (2 * np.pi) - np.pi)
    out = np.flatnonzero(err >= NAV_POS_ERR)  # settled = inside the tolerance from then on
    settle = (out[-1] + 1 if len(out) else 0) * env.dt if len(out) < len(err) else np.inf
    out_y = np.flatnonzero(np.rad2deg(yaw_err) >= NAV_YAW_ERR_DEG)
    settle_y = (out_y[-1] + 1 if len(out_y) else 0) * env.dt if len(out_y) < len(err) else np.inf
    last = max(1, round(1.0 / env.dt))
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
    "navigate": (navigate, [("GoToBall-v0", p) for p in NAV_GRID], 400, False, False),
    "random": (random, [(e, None) for e in ALL], None, True, False),
    "competent": (competent, [(e, None) for e in ALL], None, True, False),
}


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


def episode(scenario: str, env_id: str, param, seed: int) -> dict:
    """Run one episode and return its row plus the per-step arrays the summary pools."""
    global _ctx
    xy, yaw, speed, cmd_w = [], [], [], []
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

    def recorded(i, pose):  # the controller's own omega, before the servo's ctrlrange clip
        cmd = navigate_(i, pose)
        if i == 0:
            cmd_w.append(cmd[2])
        return cmd

    env._navigate = recorded
    r = env.robots[0]
    yaws, held, impulses, total, info = [], [], collections.defaultdict(list), 0.0, {}
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
            yaws.append(abs(r.velocity(env.data)[2]))
            held.append(env.holds(0))
            q = env.data.qpos[r.qpos_adr : r.qpos_adr + 3]
            xy.append(q[:2] + np.asarray(r.xy0))
            yaw.append(q[2])
            speed.append(np.hypot(*r.velocity(env.data)[:2]))
            for pair, j in _impulse.items():
                impulses[pair].append(j)
    finally:
        mujoco.mj_step, _ctx = _mj_step, None
    wall_s = time.perf_counter() - t0
    env.close()

    runs = [len(g) for g in "".join("1" if h else "0" for h in held).split("0") if g]
    reason = ("ball_out" if info.get("ball_out") else "goal" if info.get("goal")
              else "success" if info.get("is_success") else "truncated" if truncated
              else "terminated" if terminated else "scripted_end")
    y = np.array(yaws)
    row = dict(
        scenario=scenario, env=env_id, param=param, seed=seed, steps=steps,
        sim_s=round(steps * env.dt, 4), reason=reason, success=bool(info.get("is_success")),
        yaw_p99=round(np.percentile(y, 99), 3), yaw_max=round(y.max(), 3),
        yaw_frac_over_max_w=round(float((y > MAX_W).mean()), 5), held_frac=round(float(np.mean(held)), 4),
        poss_mean_steps=round(float(np.mean(runs)), 2) if runs else 0,
        poss_p95_steps=round(float(np.percentile(runs, 95)), 2) if runs else 0,
        ret=round(total, 4), wall_s=round(wall_s, 3),
    )
    if hasattr(policy, "target"):
        row.update(nav_metrics(env, policy.target, np.array(xy), np.array(yaw),
                               np.array(speed), np.array(cmd_w)))
    return dict(row=row, yaws=y, runs=runs, impulses=dict(impulses))


def _unit(args):
    return episode(*args)


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
        cells[(k["scenario"], k["env"], str(k["param"]))].append(r)

    summary, impulse = [], []
    for key, rs in cells.items():
        rr = [r["row"] for r in rs]
        y = np.concatenate([r["yaws"] for r in rs])
        runs = sum((r["runs"] for r in rs), [])
        minutes = sum(r["sim_s"] for r in rr) / 60
        ret = [r["ret"] for r in rr]
        held = [r["held_frac"] for r in rr]
        summary.append(dict(
            scenario=key[0], env=key[1], param=key[2], episodes=len(rr),
            minutes=round(minutes, 3),
            yaw_p50=_q(y, 50)[0], yaw_p99=_q(y, 99)[0], yaw_max=round(y.max(), 3),
            yaw_frac_over_max_w=round(float((y > MAX_W).mean()), 5),
            held_frac_p50=_q(held, 50)[0], poss_p50_steps=_q(runs, 50)[0],
            poss_mean_steps=round(float(np.mean(runs)), 2) if runs else "", poss_p95_steps=_q(runs, 95)[0],
            ret_p25_p50_p75=" ".join(map(str, _q(ret, (25, 50, 75)))),
            success_rate=round(float(np.mean([r["success"] for r in rr])), 3),
            steps_per_s_pooled=round(sum(r["steps"] for r in rr) / sum(r["wall_s"] for r in rr), 1),
        ))
        pairs = collections.defaultdict(list)
        for r in rs:
            for p, js in r["impulses"].items():
                pairs[p] += js
        for p, js in sorted(pairs.items(), key=lambda kv: -max(kv[1])):
            impulse.append(dict(scenario=key[0], env=key[1], param=key[2], pair=p,
                                steps_active=len(js), **dict(zip(("p50", "p95", "p99"), _q(js, (50, 95, 99)))),
                                max=round(max(js), 5)))

    for name, table in (("summary", summary), ("impulse", impulse)):
        if not table:  # navigate alone touches nothing
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
    ap.add_argument("--jobs", type=int, default=max(1, multiprocessing.cpu_count() - 1))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    names = args.scenario or list(SCENARIOS)
    units = [(n, e, p, s) for n in names for e, p in SCENARIOS[n][1]
             for s in (_seeds(args.seeds) if SCENARIOS[n][3] else [0])]
    # Longest first, so the 7-robot episodes do not start last.
    units.sort(key=lambda u: u[1] == "StaticDefenders-v0", reverse=True)
    with multiprocessing.Pool(args.jobs) as pool:
        results = []
        for i, r in enumerate(pool.imap_unordered(_unit, units), 1):
            results.append(r)
            print(f"\r{i}/{len(units)}", end="", flush=True)
    print()
    summarise(results, args.out)
    if "random" in names:
        throughput(args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
