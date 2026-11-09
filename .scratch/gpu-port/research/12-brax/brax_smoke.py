"""Ticket 12: brax ppo.train end to end on MJX-Warp, toy 2-robot env. One run per process.

    brax_smoke.py <nworld> [k=v ...]
      graph=WARP|WARP_STAGED|NONE   Warp graph mode (default WARP)
      dr=0|1                        per-slot ball-mass DR via randomization_fn + in_axes
      pmap_reset=1|0                brax use_pmap_on_reset
      steps=<int>                   num_timesteps (env steps = decisions)
      nacon=<per world> njmax=<per world>

Toy env: robot 0 chases the ball (reward = -distance, done when within 12 cm); robot 1 idle.
One env step = one decision = 24 physics steps. Autoreset uses mjx.Data.where (Playground's
BraxAutoResetWrapper with where_done swapped). Prints one `RESULT {json}` line. Not part of src/.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "05-bench"))

import warp as wp

# Count Warp graph captures (WARP mode keys them on buffer addresses; a growing count = recapture).
CAPTURES = [0]
_Scoped = wp.ScopedCapture


class CountingCapture(_Scoped):
    def __enter__(self):
        if not getattr(self, "external", False):
            CAPTURES[0] += 1
        return super().__enter__()


wp.ScopedCapture = CountingCapture

if os.environ.get("CACHEMAX"):  # cap Warp's per-callable graph cache (default 32); mjx passes none
    import functools
    wp.jax_callable = functools.partial(wp.jax_callable, graph_cache_max=int(os.environ["CACHEMAX"]))

import jax
import jax.numpy as jp
import numpy as np
from brax.envs.base import Env, State, Wrapper
from brax.envs.wrappers import training as bt
from brax.training.agents.ppo import train as ppo
from mujoco import mjx
from mujoco.mjx.warp import types as wt

import bench as B

SUB = 24


class ToyChase(Env):
    def __init__(self, nworld, graph="WARP", nacon_pw=64, njmax=64):
        self.m, self.robots, self.ball = B.build(2)
        self.sys = mjx.put_model(self.m, impl="warp", graph_mode=getattr(wt.GraphMode, graph))
        self._make = dict(naconmax=nworld * nacon_pw, njmax=njmax)
        r = self.robots[0]
        self._lo = jp.array(self.m.actuator_ctrlrange[list(r.drive), 0])
        self._hi = jp.array(self.m.actuator_ctrlrange[list(r.drive), 1])
        self._dof = int(self.m.jnt_dofadr[np.flatnonzero(self.m.jnt_qposadr == r.qpos_adr)[0]])

    def _obs(self, d):
        r, b = self.robots[0], self.ball.qpos_adr
        a = r.qpos_adr
        rel = d.qpos[b:b + 2] - (d.qpos[a:a + 2] + jp.array(r.xy0))
        yaw = d.qpos[a + 2]
        return jp.concatenate([rel, jp.stack([jp.sin(yaw), jp.cos(yaw)]), d.qvel[self._dof:self._dof + 3]])

    def _dist(self, d):
        return jp.linalg.norm(self._obs(d)[:2])

    def reset(self, rng):
        u = jax.random.uniform(rng, (3 * 2 + 2,))
        d = mjx.make_data(self.m, impl="warp", **self._make)
        d = d.replace(qpos=B.qpos_init(jp, self.m, self.robots, self.ball, u, "random"))
        d = mjx.forward(self.sys, d)
        z = jp.zeros(())
        return State(d, self._obs(d), z, z, {"dist": self._dist(d)}, {})

    def step(self, state, action):
        r = self.robots[0]
        d = state.pipeline_state
        v = self._lo + (self._hi - self._lo) * (jp.clip(action, -1, 1) + 1) / 2
        d = d.replace(ctrl=d.ctrl.at[jp.array(r.drive)].set(v))
        d = jax.lax.fori_loop(0, SUB, lambda _, x: mjx.step(self.sys, x), d)
        dist = self._dist(d)
        done = (dist < 0.12).astype(jp.float32)
        return state.replace(pipeline_state=d, obs=self._obs(d), reward=-dist, done=done,
                             metrics={**state.metrics, "dist": dist})

    @property
    def observation_size(self):
        return 7

    @property
    def action_size(self):
        return 3

    @property
    def backend(self):
        return "mjx-warp"


class DataWhereAutoReset(Wrapper):
    """Playground BraxAutoResetWrapper (cached first state) with where_done -> mjx.Data.where."""

    def reset(self, rng):
        s = self.env.reset(rng)
        s.info["first_data"], s.info["first_obs"] = s.pipeline_state, s.obs
        return s

    def step(self, state, action):
        if "steps" in state.info:
            state.info.update(steps=jp.where(state.done, 0, state.info["steps"]))
        state = self.env.step(state.replace(done=jp.zeros_like(state.done)), action)
        data = jax.vmap(lambda d, f, dn: d.where(dn.astype(bool), f))(
            state.pipeline_state, state.info["first_data"], state.done)
        obs = jp.where(state.done[:, None] > 0, state.info["first_obs"], state.obs)
        return state.replace(pipeline_state=data, obs=obs)


def wrap(env, episode_length=1000, action_repeat=1, randomization_fn=None):
    env = bt.VmapWrapper(env) if randomization_fn is None else bt.DomainRandomizationVmapWrapper(env, randomization_fn)
    return DataWhereAutoReset(bt.EpisodeWrapper(env, episode_length, action_repeat))


def ball_mass_dr(sys, rng, ball_body):
    """Per-slot ball mass x U(0.8, 1.2). Toy: no mj_setConst (derivatives stale; ticket 11 does it on host)."""
    n = rng.shape[0]
    scale = jax.vmap(lambda k: jax.random.uniform(k, (), minval=0.8, maxval=1.2))(rng)
    mass = jp.tile(sys.body_mass, (n, 1)).at[:, ball_body].multiply(scale)
    sys_v = sys.replace(body_mass=mass)
    in_axes = jax.tree.map(lambda _: None, sys).replace(body_mass=0)
    return sys_v, in_axes


def gpu_used_mb():
    return B.gpu_used_mb()


def main():
    nworld = int(sys.argv[1])
    kw = dict(a.split("=", 1) for a in sys.argv[2:])
    graph = kw.get("graph", "WARP")
    dr = kw.get("dr", "0") == "1"
    cfg = dict(nworld=nworld, graph=graph, dr=dr, pmap_reset=kw.get("pmap_reset", "1") == "1",
               brax=__import__("brax").__version__, jax=jax.__version__, devices=jax.local_device_count())
    nacon, njmax = int(kw.get("nacon", 64)), int(kw.get("njmax", 64))
    env = ToyChase(nworld, graph, nacon, njmax)
    neval = 128
    eval_env = ToyChase(neval, graph, nacon, njmax)
    rfn = (lambda s, rng: ball_mass_dr(s, rng, env.ball.body)) if dr else None
    batch = nworld // 8  # batch_size * num_minibatches == nworld * 4 (Playground-ish)
    hp = dict(num_envs=nworld, episode_length=200, unroll_length=20, batch_size=batch, num_minibatches=32,
              num_updates_per_batch=4, num_evals=int(kw.get("evals", 4)), num_eval_envs=neval, learning_rate=3e-4,
              entropy_cost=1e-2, discounting=0.97, normalize_observations=True, reward_scaling=1.0)
    steps = int(kw.get("steps", (int(kw.get("evals", 4)) - 1) * int(kw.get("ts", 4)) * 80 * nworld))  # 3 epochs x ts training steps
    cfg.update(hp=hp, num_timesteps=steps)
    gpu0 = gpu_used_mb()
    log = []
    t_start = time.perf_counter()

    def progress(step, metrics):
        log.append(dict(step=int(step), t=time.perf_counter() - t_start, captures=CAPTURES[0],
                        reward=float(metrics.get("eval/episode_reward", np.nan)),
                        dist=float(metrics.get("eval/episode_dist", np.nan)),
                        sps=float(metrics.get("training/sps", np.nan)), gpu_mb=gpu_used_mb() - gpu0))
        print("PROGRESS", json.dumps(log[-1]), flush=True)

    try:
        ppo.train(env, num_timesteps=steps, wrap_env_fn=wrap, randomization_fn=rfn, eval_env=eval_env,
                  use_pmap_on_reset=cfg["pmap_reset"], progress_fn=progress, seed=0, **hp)
    except Exception as e:  # record, don't hide
        import traceback
        traceback.print_exc()
        cfg.update(error=f"{type(e).__name__}: {str(e).splitlines()[0][:400]}", log=log, captures=CAPTURES[0])
        print("RESULT " + json.dumps(cfg), flush=True)
        return
    # steady-state decisions/s from the last two progress points (excludes compile + first epoch)
    if len(log) >= 3:
        a, b = log[-2], log[-1]
        cfg["dps"] = (b["step"] - a["step"]) / (b["t"] - a["t"])
        cfg["sps_phys"] = cfg["dps"] * SUB
    cfg.update(log=log, captures=CAPTURES[0], wall_s=time.perf_counter() - t_start,
               rss_mb=B.rss_mb(), gpu_mb=gpu_used_mb() - gpu0)
    try:
        cfg["jax_peak_mb"] = jax.devices()[0].memory_stats()["peak_bytes_in_use"] / 2**20
    except Exception:
        pass
    print("RESULT " + json.dumps(cfg), flush=True)


if __name__ == "__main__":
    main()
