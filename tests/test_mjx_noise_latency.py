"""Observation noise and action latency on MJX-Warp (ticket 15), through the public seam:
`mjsoccer.mjx.make(..., action_latency=N)` and `mjsoccer.mjx.wrappers.wrap(..., obs_noise=)`.
Both are off by default, and off they change nothing.

MJX-Warp is not bitwise reproducible (two identical rollouts differ by ~1e-6 at reset and
~1e-4 after a few decisions, atomics in the solver), so "the same" below is within ATOL;
the noise it is told apart from has std 0.05.

    uv sync --group gpu && uv run pytest tests/test_mjx_noise_latency.py

Skipped without the `gpu` group or a CUDA device; each run is a fresh subprocess (as in
tests/test_mjx_warp_smoke.py), so the GPU is let go when it ends.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys

import pytest

# Present, but not imported: importing jax here would put it on the GPU before
# tests/test_mjx_task_logic.py (collected after this file) sets JAX_PLATFORMS=cpu.
if any(importlib.util.find_spec(m) is None for m in ("jax", "brax", "warp")):
    pytest.skip("needs the gpu group (uv sync --group gpu)", allow_module_level=True)

ENV_ID = "ContestedPossession-v0"
NWORLD = 4
SKIP = 77
ATOL = 2e-3


def _run(*args) -> dict:
    env = {k: v for k, v in os.environ.items() if k != "JAX_PLATFORMS"}
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    p = subprocess.run([sys.executable, __file__, *map(str, args)],
                       env=env, capture_output=True, text=True, timeout=1800)
    if p.returncode == SKIP:
        pytest.skip("no CUDA device")
    assert p.returncode == 0, p.stderr[-4000:]
    return json.loads(p.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("repeat_action", [1, 2])
def test_off_is_the_env_as_before(repeat_action):
    """`obs_noise=0` and `action_latency=0`, passed explicitly, give the same wrapper stack,
    the same `info` keys and the same states (within Warp's own jitter) as an env and
    wrapper made without them."""
    out = _run("off", repeat_action)
    assert out["same_stack"], out
    assert out["same_keys"], out
    assert out["same"], out


@pytest.mark.parametrize("repeat_action", [1, 2])
def test_latency_applies_the_action_n_control_steps_later(repeat_action):
    """A kick asked for on the first decision only fires at control step N (0-based): the
    plunger lock, set to the cooldown on the firing control step and counted down at the
    end of each one, reads cooldown - 1 - (c - N) after control step c >= N and 0 before.
    Without latency it fires on control step 0. Until the first action arrives the robot
    holds the pose it was placed at."""
    n = 3
    out = _run("latency", repeat_action, n)
    cooldown, ra = out["cooldown"], repeat_action
    for c0, lag in ((out["lock_now"], 0), (out["lock_late"], n)):
        expected = []
        for k in range(len(c0)):
            c = (k + 1) * ra - 1  # the last control step of decision k
            expected.append(max(cooldown - 1 - (c - lag), 0) if c >= lag else 0)
        assert c0 == expected, (lag, out)
    assert out["held_drift"] < 0.02, out  # metres, over the first N control steps


def test_latency_buffer_is_reset_per_world_by_the_auto_reset():
    """Under `wrap`, a world that times out comes back with its action buffer as at reset
    (with the rest of `info["world"]`), not with the actions of the episode that ended."""
    out = _run("latency_reset", 1, 3)
    assert out["has_buffer"], out
    assert out["buffer_restored"] == [True] * NWORLD, out


def test_obs_noise_is_reproducible_per_seed():
    """With `obs_noise=sigma` the observations are the clean ones plus noise of about that
    std, the same for the same seed and different for another; the physics is untouched."""
    out = _run("noise", 1)
    assert out["same_seed_equal"], out
    assert not out["other_seed_equal"], out
    assert out["physics_equal"], out  # qpos with and without noise, same seed
    assert 0.03 < out["noise_std"] < 0.07, out  # sigma = 0.05
    assert out["reset_noisy"] and out["fresh_each_step"], out


# ------------------------------------------------------------------ the children


def _hold(env, nworld=NWORLD):
    import numpy as np

    a = np.zeros((nworld, env.action_size), np.float32)
    a[:, 3], a[:, 4:7] = 1.0, -1.0  # robot 0 holds the origin facing +x, no kick, no dribble
    return a


def _tree_close(a, b, atol=ATOL) -> bool:
    import jax
    import numpy as np

    la, lb = jax.tree.leaves(a), jax.tree.leaves(b)
    return len(la) == len(lb) and all(np.allclose(np.asarray(x), np.asarray(y), atol=atol)
                                      for x, y in zip(la, lb))


def _stack(w) -> list[str]:
    names = []
    while hasattr(w, "env"):
        names.append(type(w).__name__)
        w = w.env
    return names + [type(w).__name__]


def off(repeat_action: int) -> dict:
    import jax

    import mjsoccer.mjx
    from mjsoccer.mjx import wrappers

    keys = jax.random.split(jax.random.PRNGKey(0), NWORLD)
    runs, stacks = [], []
    for kw, wkw in (({}, {}), ({"action_latency": 0}, {"obs_noise": 0.0})):
        env = mjsoccer.mjx.make(ENV_ID, nworld=NWORLD, repeat_action=repeat_action, **kw)
        wenv = wrappers.wrap(env, episode_length=10**6, **wkw)
        s = jax.jit(wenv.reset)(keys)
        step = jax.jit(wenv.step)
        for _ in range(4):
            s = step(s, _hold(env))
        runs.append(s)
        stacks.append(_stack(wenv))
    a, b = runs

    def keys_of(s):
        return sorted(s.info), sorted(s.info["world"])

    # Warp's jitter grows over control steps (repeat 2 passed 2e-3 now and then), so 1e-2;
    # the exact part of "off" is the stack and the keys. Not the raw qvel: the dribbler
    # bar's spin is large enough that the jitter on it passes 1e-2; the obs carries the
    # robots' and ball's velocities, normalised.
    pick = lambda s: (s.obs, s.reward, s.done, s.metrics, s.info["world"],  # noqa: E731
                      s.pipeline_state.qpos)
    return dict(same_stack=stacks[0] == stacks[1], same_keys=keys_of(a) == keys_of(b),
                same=_tree_close(pick(a), pick(b), atol=1e-2))


def latency(repeat_action: int, n: int) -> dict:
    import jax
    import numpy as np

    import mjsoccer.mjx

    keys = jax.random.split(jax.random.PRNGKey(0), NWORLD)
    out = {}
    for name, lag in (("lock_now", 0), ("lock_late", n)):
        env = mjsoccer.mjx.make(ENV_ID, nworld=NWORLD, repeat_action=repeat_action,
                                action_latency=lag)
        s = jax.jit(jax.vmap(env.reset))(keys)
        r = env.robots[0]
        xy0 = np.asarray(s.pipeline_state.xpos[:, r.body, :2])
        step = jax.jit(jax.vmap(env.step))
        rest = _hold(env)
        kick = rest.copy()
        kick[:, 4] = 1.0
        locks, drift = [], 0.0
        for k in range(6 // repeat_action + 2):
            s = step(s, kick if k == 0 else rest)
            mjsoccer.mjx.check_failures(s)
            lk = np.asarray(s.info["world"]["plunger_lock"])[:, 0, 0]
            assert (lk == lk[0]).all(), lk
            locks.append(int(lk[0]))
            if (k + 1) * repeat_action <= n:
                xy = np.asarray(s.pipeline_state.xpos[:, r.body, :2])
                drift = max(drift, float(np.linalg.norm(xy - xy0, axis=-1).max()))
        out[name] = locks
        if lag:
            out["held_drift"] = drift
        out["cooldown"] = int(env.kick_cooldown)
    return out


def latency_reset(repeat_action: int, n: int) -> dict:
    import math

    import jax
    import numpy as np

    from mjsoccer import ENV_IDS
    from mjsoccer.mjx import tasks, wrappers

    cls = getattr(tasks, ENV_IDS[ENV_ID][0])
    env = cls(horizon_s=0.1, nworld=NWORLD, repeat_action=repeat_action, action_latency=n)
    length = math.ceil(env.max_steps / repeat_action)
    wenv = wrappers.wrap(env, episode_length=length)
    keys = jax.random.split(jax.random.PRNGKey(0), NWORLD)
    first = s = jax.jit(wenv.reset)(keys)
    step = jax.jit(wenv.step)
    rng = np.random.default_rng(0)
    for _ in range(length):
        s = step(s, rng.uniform(-1, 1, (NWORLD, env.action_size)).astype(np.float32))
    buf = [k for k in first.info["world"] if "action" in k]
    now, then = (np.asarray(x.info["world"][buf[0]]) for x in (s, first)) if buf else (0, 0)
    return dict(has_buffer=bool(buf), buffer_restored=[
        bool(np.array_equal(now[w], then[w])) for w in range(NWORLD)] if buf else [])


def noise(repeat_action: int) -> dict:
    import jax
    import numpy as np

    import mjsoccer.mjx
    from mjsoccer.mjx import wrappers

    env = mjsoccer.mjx.make(ENV_ID, nworld=NWORLD, repeat_action=repeat_action)

    def rollout(seed, sigma):
        wenv = wrappers.wrap(env, episode_length=10**6, obs_noise=sigma)
        s = jax.jit(wenv.reset)(jax.random.split(jax.random.PRNGKey(seed), NWORLD))
        step = jax.jit(wenv.step)
        obs, qpos = [np.asarray(s.obs)], []
        for _ in range(5):
            s = step(s, _hold(env))
            obs.append(np.asarray(s.obs))
            qpos.append(np.asarray(s.pipeline_state.qpos))
        return np.stack(obs), np.stack(qpos)

    clean, q_clean = rollout(0, 0.0)
    noisy, q_noisy = rollout(0, 0.05)
    again, _ = rollout(0, 0.05)
    other, _ = rollout(1, 0.05)
    clean1, _ = rollout(1, 0.0)
    eps = noisy - clean
    return dict(
        same_seed_equal=bool(np.allclose(noisy, again, atol=ATOL)),
        other_seed_equal=bool(np.allclose(eps, other - clean1, atol=ATOL)),
        physics_equal=bool(np.allclose(q_clean, q_noisy, atol=ATOL)),
        noise_std=float(eps.std()),
        reset_noisy=bool(np.abs(eps[0]).max() > 0),
        fresh_each_step=bool(not np.allclose(eps[1], eps[2], atol=ATOL)),
    )


def main() -> None:
    import jax

    try:
        jax.devices("gpu")
    except RuntimeError:
        sys.exit(SKIP)
    mode, *args = sys.argv[1:]
    run = dict(off=off, latency=latency, latency_reset=latency_reset, noise=noise)[mode]
    print(json.dumps(run(*map(int, args))))


if __name__ == "__main__":
    main()
