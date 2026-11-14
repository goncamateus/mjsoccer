"""Ticket 14's validation harness: the §13.2 scenarios, the §13.1 metrics, written to CSV.

Every scenario is a scripted sequence against the unwrapped env, no wrappers and no
trained policy. `tests/test_validation.py` asserts the acceptance criteria on the same
scenarios; this script produces the distributions behind them.

    uv run python scripts/validation_sweep.py --out runs/validation --seeds 0-4   # all
    uv run python scripts/validation_sweep.py --scenario spin --out runs/v

    scenario      env              parameter
    spin          GoToBall         raw yaw command k in 0.2..1.0, 7.5 s
    kick_recoil   Dribbling        full kick from 0/2/4 m/s of chassis speed, raw drive
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
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

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
