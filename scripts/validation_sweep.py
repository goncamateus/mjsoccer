"""Ticket 14's validation harness: the §13.2 scenarios, the §13.1 metrics, written to CSV.

Every scenario is a scripted sequence against the unwrapped env, no wrappers and no
trained policy. `tests/test_validation.py` asserts the acceptance criteria on the same
scenarios; this script produces the distributions behind them.

    uv run python scripts/validation_sweep.py --out runs/validation              # all, seeds 0-49
    uv run python scripts/validation_sweep.py --scenario wall --scenario spin --out /tmp/v

    scenario      env, tier             parameter
    spin          GoToBall, 1           wheel action k in 0.2..1.0, 5 s
    head_on       ContestedPossession,1 closing speed 1/2/3/4 m/s, both robots front to front
    wall          GoToBall, 1           (speed 2/4/6 m/s, incidence 5/15/30 deg) into the +y wall
    dribble_turn  Dribbling, 1          commanded yaw 0/10/20/30/50 rad/s, bar at 157 rad/s
    kick_recoil   Dribbling, 1          full kick from 0/2/4 m/s of chassis speed
    random        all six ids, 0 and 1  action_space.sample()
    competent     all six ids, 0 and 1  goto_ball_policy (wheel_command on tier 1)

Only `random` and `competent` depend on the seed. The scripted scenarios write the whole
initial state over the reset, so they run once per parameter rather than 50 identical times.
Speeds above what tier 1 can drive to (ticket 18: ~2 m/s) are written into qvel, chassis and
wheels together, so the robot starts rolling rather than skidding.

Writes into --out:
    episodes.csv   one row per episode
    summary.csv    one row per (scenario, env, tier, param): tip rate, tilt/yaw percentiles,
                   possession, return -- medians and IQRs for the skewed ones
    impulse.csv    per-step contact impulse on the controlled robot, by contact pair
    pretip.csv     the 0.5 s of body-frame velocity before every tip
    throughput.csv steps/s per (env, tier), one process, no impulse hook (with --scenario random)
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
from mjsoccer import scene as S
from mjsoccer.base import MAX_V, MAX_W
from mjsoccer.tasks import goto_ball_policy, wheel_command

HISTORY_S = 0.5  # of velocity kept before a tip
GAP = 0.05  # m between bumpers when a scripted collision starts
ROBOT_R = 0.09  # m, chassis radius (SSL rulebook 180 mm diameter)


# ------------------------------------------------------------------ scenario set-up
# Each takes a freshly reset unwrapped env and the parameter, writes the initial state, and
# returns the policy: (obs, step) -> action.


def _action(env) -> np.ndarray:
    return np.zeros(env.action_space.shape, dtype=np.float32)


def _far_ball(env) -> None:
    env.ball.place(env.data, 4.0, -3.0)


def launch(env, i: int, speed: float) -> np.ndarray:
    """Robot `i` already rolling forward at `speed`: chassis and wheel joints together.

    Returns the wheel action that holds it there (saturated to the ctrlrange)."""
    r, m, d = env.robots[i], env.model, env.data
    mujoco.mj_forward(m, d)
    d.qvel[r.qvel_adr : r.qvel_adr + 2] = speed * r.forward(d)[:2]
    # wheel_command is linear until it saturates, so a 1 cm/s ask scales to the true speeds.
    q = wheel_command(env, i, 0.01, 0.0, 0.0) * r.drive_range[:, 1] * speed / 0.01
    for act, qk in zip(r.drive, q):
        d.qvel[m.jnt_dofadr[m.actuator_trnid[act, 0]]] = qk
    return wheel_command(env, i, speed, 0.0, 0.0)


def spin(env, k):
    env.robots[0].place(env.data, 0.0, 0.0, 0.0)
    _far_ball(env)
    a = _action(env)
    a[:4] = k
    return lambda obs, t: a


def head_on(env, v):
    env.robots[0].place(env.data, -ROBOT_R - GAP / 2, 0.0, 0.0)
    env.robots[1].place(env.data, ROBOT_R + GAP / 2, 0.0, np.pi)
    _far_ball(env)
    a = _action(env)
    a[:4] = launch(env, 0, v / 2)
    launch(env, 1, v / 2)  # scripted: its drive is idle, so it brakes from here
    return lambda obs, t: a


def wall(env, p):
    v, deg = p
    env.robots[0].place(env.data, -3.0, S.WALL_HALF_Y - ROBOT_R - GAP, np.radians(deg))
    _far_ball(env)
    a = _action(env)
    a[:4] = launch(env, 0, v)
    return lambda obs, t: a


def dribble_turn(env, w):
    a = _action(env)  # Dribbling-v0 seats the ball at reset
    a[:4] = wheel_command(env, 0, 0.0, 0.0, w)
    a[-1] = 0.5  # 157 rad/s, what the possession figures are measured at (ticket 08)
    return lambda obs, t: a


def kick_recoil(env, v):
    r = env.robots[0]
    r.place(env.data, -4.0, 0.0, 0.0)  # 4 m/s for a second needs the room
    mujoco.mj_forward(env.model, env.data)
    env.seat_ball(r)
    a = _action(env)
    a[:4] = launch(env, 0, v)
    env.data.qvel[env.ball.qvel_adr] = v  # carried along

    def policy(obs, t):
        b = a.copy()
        b[env.drive_dim] = 1.0 if t == 0 else 0.0
        return b

    return policy


def random(env, _):
    return lambda obs, t: env.action_space.sample()


def competent(env, _):
    def policy(obs, t):
        a = _action(env)
        c = goto_ball_policy(obs)
        if env.tier == 0:
            a[:3] = c
        else:
            a[:4] = wheel_command(env, 0, c[0] * MAX_V, c[1] * MAX_V, c[2] * MAX_W)
        return a

    return policy


ALL = [(e, t) for e in mjsoccer.ENV_IDS for t in (0, 1)]
# name -> (set-up, [(env, tier, param)], control steps or None for the whole episode, seeded)
SCENARIOS = {
    "spin": (spin, [("GoToBall-v0", 1, k) for k in (0.2, 0.4, 0.6, 0.8, 1.0)], 300, False),
    "head_on": (head_on, [("ContestedPossession-v0", 1, v) for v in (1, 2, 3, 4)], 60, False),
    "wall": (wall, [("GoToBall-v0", 1, (v, d)) for v in (2, 4, 6) for d in (5, 15, 30)], 60, False),
    "dribble_turn": (dribble_turn, [("Dribbling-v0", 1, w) for w in (0, 10, 20, 30, 50)], 120, False),
    "kick_recoil": (kick_recoil, [("Dribbling-v0", 1, v) for v in (0, 2, 4)], 60, False),
    "random": (random, [(e, t, None) for e, t in ALL], None, True),
    "competent": (competent, [(e, t, None) for e, t in ALL], None, True),
}


# ------------------------------------------------------------- contact impulse hook

CLASSES = ("floor", "wall", "goal", "ball", "roller", "hub", "body", "mech")
ROBOT = {"roller", "hub", "body", "mech"}
_mj_step = mujoco.mj_step
_ctx = None  # (geom classes, body roots, controlled roots) of the env being measured
_impulse = collections.Counter()  # pair -> N*s this control step


def _classify(m) -> np.ndarray:
    out = []
    for g in range(m.ngeom):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        out.append("floor" if n == "floor" else "wall" if "walls" in n else "goal" if "goalpost" in n
                   else "ball" if n == "ball" else "roller" if "roller" in n else "hub" if "Direct" in n
                   else "mech" if m.geom_contype[g] == 2 else "body")
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


def episode(scenario: str, env_id: str, tier: int, param, seed: int) -> dict:
    """Run one episode and return its row plus the per-step arrays the summary pools."""
    global _ctx
    setup, _, max_steps, _ = SCENARIOS[scenario]
    env = gym.make(env_id, tier=tier).unwrapped
    m = env.model
    root = m.body_rootid
    mine = [root[r.body] for r in env.robots[: env.n_controlled]]
    obs, _ = env.reset(seed=seed)
    env.action_space.seed(seed)
    policy = setup(env, param)
    _ctx = (_classify(m), root, mine)
    r = env.robots[0]
    hist = collections.deque(maxlen=round(HISTORY_S / env.dt))
    tilts, yaws, held, impulses, total, info = [], [], [], collections.defaultdict(list), 0.0, {}
    steps, terminated, truncated, t0 = 0, False, False, time.perf_counter()
    # ponytail: patches the mujoco module and keeps the hook state in globals -- one episode
    # per process at a time, which Pool gives. Threads would need a per-env hook.
    mujoco.mj_step = _hooked  # env.step calls mujoco.mj_step through the module
    try:
        while not (terminated or truncated) and (max_steps is None or steps < max_steps):
            _impulse.clear()
            obs, reward, terminated, truncated, info = env.step(policy(obs, steps))
            steps += 1
            total += reward
            R = env.data.xmat[r.body].reshape(3, 3)
            tilts.append(float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1)))))
            vx, vy, w = r.velocity(env.data)
            yaws.append(abs(w))
            held.append(env.holds(0))
            body = R.T @ (vx, vy, 0.0)
            hist.append((steps * env.dt, body[0], body[1], w))
            for pair, j in _impulse.items():
                impulses[pair].append(j)
    finally:
        mujoco.mj_step, _ctx = _mj_step, None
    wall_s = time.perf_counter() - t0
    env.close()

    runs = [len(g) for g in "".join("1" if h else "0" for h in held).split("0") if g]
    tipped = bool(info.get("tipped"))
    reason = ("tipped" if tipped else "ball_out" if info.get("ball_out") else "goal" if info.get("goal")
              else "success" if info.get("is_success") else "truncated" if truncated
              else "terminated" if terminated else "scripted_end")
    t, y = np.array(tilts), np.array(yaws)
    row = dict(
        scenario=scenario, env=env_id, tier=tier, param=param, seed=seed, steps=steps,
        sim_s=round(steps * env.dt, 4), reason=reason, tipped=tipped,
        t_tip_s=round(steps * env.dt, 4) if tipped else "", success=bool(info.get("is_success")),
        tilt_mean=round(t.mean(), 3), tilt_p99=round(np.percentile(t, 99), 3), tilt_max=round(t.max(), 3),
        yaw_p99=round(np.percentile(y, 99), 3), yaw_max=round(y.max(), 3),
        yaw_frac_over_max_w=round(float((y > MAX_W).mean()), 5), held_frac=round(float(np.mean(held)), 4),
        poss_mean_steps=round(float(np.mean(runs)), 2) if runs else 0,
        poss_p95_steps=round(float(np.percentile(runs, 95)), 2) if runs else 0,
        ret=round(total, 4), wall_s=round(wall_s, 3),
    )
    pretip = [(scenario, env_id, tier, param, seed, round(ts - steps * env.dt, 4), *map(float, v))
              for ts, *v in hist] if tipped else []
    return dict(row=row, tilts=t, yaws=y, runs=runs, impulses=dict(impulses), pretip=pretip)


def _unit(args):
    return episode(*args)


# --------------------------------------------------------------------- summarising


def _q(x, qs):
    qs = np.atleast_1d(qs)
    return [round(float(v), 4) for v in np.percentile(x, qs)] if len(x) else [""] * len(qs)


def summarise(results: list[dict], out: Path) -> None:
    rows = [r["row"] for r in results]
    with open(out / "episodes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    cells = collections.defaultdict(list)
    for r in results:
        k = r["row"]
        cells[(k["scenario"], k["env"], k["tier"], str(k["param"]))].append(r)

    summary, impulse = [], []
    for key, rs in cells.items():
        rr = [r["row"] for r in rs]
        t = np.concatenate([r["tilts"] for r in rs])
        y = np.concatenate([r["yaws"] for r in rs])
        runs = sum((r["runs"] for r in rs), [])
        minutes = sum(r["sim_s"] for r in rr) / 60
        tips = sum(r["tipped"] for r in rr)
        ttip = [r["t_tip_s"] for r in rr if r["tipped"]]
        ret = [r["ret"] for r in rr]
        held = [r["held_frac"] for r in rr]
        summary.append(dict(
            scenario=key[0], env=key[1], tier=key[2], param=key[3], episodes=len(rr),
            minutes=round(minutes, 3), tips=tips, tips_per_min=round(tips / minutes, 4),
            t_tip_p25_p50_p75=" ".join(map(str, _q(ttip, (25, 50, 75)))),
            tilt_p50=_q(t, 50)[0], tilt_p95=_q(t, 95)[0], tilt_p99=_q(t, 99)[0], tilt_max=round(t.max(), 3),
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
            impulse.append(dict(scenario=key[0], env=key[1], tier=key[2], param=key[3], pair=p,
                                steps_active=len(js), **dict(zip(("p50", "p95", "p99"), _q(js, (50, 95, 99)))),
                                max=round(max(js), 5)))

    for name, table in (("summary", summary), ("impulse", impulse)):
        with open(out / f"{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(table[0]))
            w.writeheader()
            w.writerows(table)
    with open(out / "pretip.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("scenario", "env", "tier", "param", "seed", "t_rel_s", "vx", "vy", "w"))
        for r in results:
            w.writerows(r["pretip"])


def throughput(out: Path, steps: int = 600) -> None:
    """steps/s of random actions, one process, no hook -- comparable to docs/index.md."""
    with open(out / "throughput.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("env", "tier", "steps_per_s"))
        for env_id, tier in ALL:
            env = gym.make(env_id, tier=tier)
            env.reset(seed=0)
            env.action_space.seed(0)
            t0 = time.perf_counter()
            for _ in range(steps):
                *_, te, tr, _ = env.step(env.action_space.sample())
                if te or tr:
                    env.reset()
            w.writerow((env_id, tier, round(steps / (time.perf_counter() - t0), 1)))
            env.close()


def _seeds(s: str) -> list[int]:
    lo, _, hi = s.partition("-")
    return list(range(int(lo), int(hi or lo) + 1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", action="append", choices=list(SCENARIOS), help="default: all")
    ap.add_argument("--seeds", default="0-49", help="for the seeded scenarios, e.g. 0-49 or 3")
    ap.add_argument("--jobs", type=int, default=max(1, multiprocessing.cpu_count() - 1))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    names = args.scenario or list(SCENARIOS)
    units = [(n, e, t, p, s) for n in names for e, t, p in SCENARIOS[n][1]
             for s in (_seeds(args.seeds) if SCENARIOS[n][3] else [0])]
    # Longest first, so the 7-robot tier-1 episodes do not start last.
    units.sort(key=lambda u: (u[1] == "StaticDefenders-v0") * 4 + u[2], reverse=True)
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
