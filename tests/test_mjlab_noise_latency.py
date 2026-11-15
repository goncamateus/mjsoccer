"""Observation noise and action latency on mjlab (ticket 10), through the public seam:
`mjsoccer.mjlab.make(..., obs_noise=sigma, action_latency=N)`, the twins of MJX's
`wrappers.wrap(obs_noise=)` and `mjsoccer.mjx.make(action_latency=)`
(tests/test_mjx_noise_latency.py). Both are off by default, and off they change nothing.

Warp is not bitwise reproducible (atomics in the solver), so "the same" below is within
ATOL; the noise it is told apart from has std 0.05.

    uv sync --group mjlab && uv run pytest tests/test_mjlab_noise_latency.py

Each run is a fresh GPU subprocess (`_gpu.py`), skipped without CUDA or the `mjlab` group.
"""

from __future__ import annotations

import json
import sys

import pytest
from _gpu import SKIP, run_json

pytest.importorskip("torch")
pytest.importorskip("mjlab")

ENV_ID = "ContestedPossession-v0"
NWORLD = 4
ATOL = 2e-3


def _run(*args) -> dict:
    return run_json(__file__, *args)


@pytest.mark.parametrize("repeat_action", [1, 2])
def test_off_is_the_env_as_before(repeat_action):
    """`obs_noise=0` and `action_latency=0`, passed explicitly, give the same per-world
    state keys and the same states (within Warp's own jitter) as an env made without."""
    out = _run("off", repeat_action)
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
    """A world that ends comes back with its action buffer as at reset (its pooled first
    state's), not with the actions of the episode that ended."""
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
    assert out["auto_reset_noisy"], out  # a world's fresh first obs after it ended


# ------------------------------------------------------------------ the children


def _make(repeat_action=1, seed=0, **kw):
    import mjsoccer.mjlab

    return mjsoccer.mjlab.make(ENV_ID, nworld=NWORLD, repeat_action=repeat_action, seed=seed,
                               **kw)


def _hold(env):
    import torch

    a = torch.zeros(NWORLD, env.logic.action_size, device=env.device)
    a[:, 3], a[:, 4:7] = 1.0, -1.0  # robot 0 holds the origin facing +x, no kick, no dribble
    return a


def off(repeat_action: int) -> dict:
    import torch

    runs = []
    for kw in ({}, {"action_latency": 0, "obs_noise": 0.0}):
        env = _make(repeat_action, **kw)
        obs = env.reset()[0]["actor"]
        for _ in range(4):
            obs, reward, *_ = env.step(_hold(env))
        runs.append((obs["actor"].clone(), reward.clone(), env.sim.data.qpos[:].clone(),
                     {k: v.clone() for k, v in env.world.items()}))
        env.close()
    (oa, ra, qa, wa), (ob, rb, qb, wb) = runs
    close = lambda x, y: bool(torch.allclose(x.float(), y.float(), atol=1e-2))
    return {"same_keys": sorted(wa) == sorted(wb),
            "same": close(oa, ob) and close(ra, rb) and close(qa, qb)
                    and all(close(wa[k], wb[k]) for k in wa)}


def latency(repeat_action: int, n: int) -> dict:
    out = {}
    for name, lag in (("lock_now", 0), ("lock_late", n)):
        env = _make(repeat_action, action_latency=lag)
        env.reset()
        r = env.logic.robots[0]
        xy0 = env.sim.data.xpos[:, r.body, :2].clone()
        rest = _hold(env)
        kick = rest.clone()
        kick[:, 4] = 1.0
        locks, drift = [], 0.0
        for k in range(6 // repeat_action + 2):
            env.step(kick if k == 0 else rest)
            lk = env.world["plunger_lock"][:, 0, 0]
            assert (lk == lk[0]).all(), lk
            locks.append(int(lk[0]))
            if (k + 1) * repeat_action <= n:
                xy = env.sim.data.xpos[:, r.body, :2]
                drift = max(drift, float((xy - xy0).norm(dim=-1).max()))
        out[name] = locks
        if lag:
            out["held_drift"] = drift
        out["cooldown"] = int(env.logic.kick_cooldown)
        env.close()
    return out


def latency_reset(repeat_action: int, n: int) -> dict:
    import torch

    env = _make(repeat_action, action_latency=n)
    env.reset()
    buf = [k for k in env.world if "action" in k]
    if not buf:
        return {"has_buffer": False, "buffer_restored": []}
    g = torch.Generator(device=env.device).manual_seed(0)
    restored = [False] * NWORLD
    for _ in range(4 * env.logic.max_steps):
        a = torch.rand(NWORLD, env.logic.action_size, device=env.device, generator=g) * 2 - 1
        _, _, term, trunc, _ = env.step(a)
        for w in (term | trunc).nonzero().flatten().tolist():
            restored[w] = bool(torch.equal(env.world[buf[0]][w], env.pool["world"][buf[0]][w]))
        if all(restored):
            break
    return {"has_buffer": True, "buffer_restored": restored}


def noise(repeat_action: int) -> dict:
    import torch

    def rollout(seed, sigma):
        env = _make(repeat_action, seed=seed, obs_noise=sigma)
        o = env.reset()[0]["actor"]
        obs, qpos, clean = [o.clone()], [], [env.obs.clone()]
        for _ in range(5):
            o = env.step(_hold(env))[0]["actor"]
            obs.append(o.clone())
            clean.append(env.obs.clone())
            qpos.append(env.sim.data.qpos[:].clone())
        # A world that ends: its fresh first obs comes back noised too.
        env.world["steps"][0] = env.logic.max_steps - 1
        o, _, term, trunc, _ = env.step(_hold(env))
        assert bool((term | trunc)[0])
        fresh = (o["actor"][0] - env.pool["obs"][0]).abs().max()
        env.close()
        return torch.stack(obs), torch.stack(qpos), float(fresh)

    clean, q_clean, _ = rollout(0, 0.0)
    noisy, q_noisy, fresh = rollout(0, 0.05)
    again, _, _ = rollout(0, 0.05)
    other, _, _ = rollout(1, 0.05)
    clean1, _, _ = rollout(1, 0.0)
    eps = noisy - clean
    return {
        "same_seed_equal": bool(torch.allclose(noisy, again, atol=ATOL)),
        "other_seed_equal": bool(torch.allclose(eps, other - clean1, atol=ATOL)),
        "physics_equal": bool(torch.allclose(q_clean, q_noisy, atol=ATOL)),
        "noise_std": float(eps.std()),
        "reset_noisy": bool(eps[0].abs().max() > 0),
        "fresh_each_step": bool(not torch.allclose(eps[1], eps[2], atol=ATOL)),
        "auto_reset_noisy": fresh > 0,
    }


def main(mode: str, *args) -> None:
    try:
        import mjlab  # noqa: F401
        import torch
    except ImportError:
        sys.exit(SKIP)
    if not torch.cuda.is_available():
        sys.exit(SKIP)
    run = {"off": off, "latency": latency, "latency_reset": latency_reset, "noise": noise}[mode]
    print(json.dumps(run(*map(int, args))))


if __name__ == "__main__":
    main(*sys.argv[1:])
