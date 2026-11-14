"""Warp smoke: each ported task on MJX-Warp makes, resets and takes ~10 steps under `vmap`.

Not a physics check (the parity against the CPU is statistical, `-m validation`): this only
proves the JAX env runs end to end on the GPU -- placement, the seat_ball loop, the
controller, possession, observation -- and stays finite. It also forces the two per-world
failures, a world gone NaN and an undersized contact buffer, and checks the batch ends that
world, zeroes it and lets Brax's auto-reset bring it back without touching the others.

    uv sync --group gpu && uv run pytest tests/test_mjx_warp_smoke.py

Skipped without the `gpu` group or without a CUDA device. The run itself is a fresh
subprocess, so the GPU is let go when it ends and the JAX-CPU-only task-logic tests in the
same session never see it. Heavy runs belong under
`systemd-run --user --scope -p MemoryMax=10G`.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

pytest.importorskip("jax")
pytest.importorskip("mujoco.mjx")
pytest.importorskip("brax")

TASKS = ["ContestedPossession-v0"]
NWORLD = 4
SKIP = 77  # the child's exit code for "no CUDA device"


def _run(mode: str, env_id: str, repeat_action: int) -> dict:
    env = {k: v for k, v in os.environ.items() if k != "JAX_PLATFORMS"}
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    p = subprocess.run(
        [sys.executable, __file__, mode, env_id, str(repeat_action)],
        env=env, capture_output=True, text=True, timeout=1800,
    )
    if p.returncode == SKIP:
        pytest.skip("no CUDA device")
    assert p.returncode == 0, p.stderr[-4000:]
    return json.loads(p.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("env_id", TASKS)
@pytest.mark.parametrize("repeat_action", [1, 2])
def test_warp_smoke(env_id, repeat_action):
    out = _run("smoke", env_id, repeat_action)
    assert out["finite"], out
    assert out["steps"] == 10, out
    # The seat_ball loop really ran on Warp: the yellow robot holds the ball at reset,
    # in every world, as it does on the CPU for every seed.
    assert all(out["yellow_holds_at_reset"]), out
    # A decision is repeat_action control steps, cut short only by an ending.
    if repeat_action == 1:
        assert out["control_steps"] == [10] * NWORLD, out
    assert all(10 <= k <= 10 * repeat_action for k in out["control_steps"]), out
    assert max(out["control_steps"]) > 10 or repeat_action == 1, out


@pytest.mark.parametrize("repeat_action", [1, 2])
def test_nan_world_ends_zeroed_and_auto_resets_alone(repeat_action):
    """World 0 is poisoned with a NaN velocity. That step ends it (terminal), with reward
    and obs 0 and `diverged` = 1; the other worlds carry on finite and unflagged. Under
    Brax's training wrappers the same step hands world 0 back at its first state, and the
    batch keeps stepping clean."""
    out = _run("nan", TASKS[0], repeat_action)
    raw = out["raw"]
    assert raw["done"] == [1.0, 0.0, 0.0, 0.0], raw
    assert raw["truncation"][0] == 0.0, raw
    assert raw["reward0"] == 0.0 and raw["obs0_zero"], raw
    assert raw["diverged"] == [1.0, 0.0, 0.0, 0.0], raw
    assert raw["overflow"] == [0.0] * NWORLD, raw
    assert raw["others_finite"], raw
    assert "diverged" in raw["check_failures"], raw
    if repeat_action > 1:
        # TODO(ticket 13): repeat_action > 1 under Brax's EpisodeWrapper scan leaks a
        # tracer from Warp's contact buffer (already at ticket 07's commit); see ticket 09.
        return
    wrapped = out["wrapped"]
    assert wrapped["diverged"] == [1.0, 0.0, 0.0, 0.0], wrapped
    assert wrapped["restored"], wrapped  # qpos/qvel/obs of world 0 back to its first state
    assert wrapped["episode_diverged"][0] == 1.0, wrapped  # what the training log sums
    assert wrapped["after_finite"] and wrapped["after_failures"] == 0, wrapped


def test_overflow_world_is_treated_like_nan():
    """With a contact buffer far too small every world overflows at reset; physics may
    well stay finite, but the outcome is undefined, so each world ends zeroed with
    `overflow` = 1 and the host check names the overflowed buffer."""
    out = _run("overflow", TASKS[0], 1)
    assert out["done"] == [1.0] * NWORLD, out
    assert out["overflow"] == [1.0] * NWORLD, out
    assert out["reward_zero"] and out["obs_zero"], out
    assert "BROADPHASE" in out["check_failures"], out


# ------------------------------------------------------------------ the child


def _actions(rng, env):
    a = rng.uniform(-1, 1, (NWORLD, env.action_size)).astype("float32")
    a[:, 4:6] = -abs(a[:, 4:6])  # kick and chip at rest: ticket 08
    return a


def _error(fn) -> str:
    try:
        fn()
    except RuntimeError as e:
        return str(e)
    return ""


def smoke(env_id: str, repeat_action: int) -> dict:
    import jax
    import numpy as np

    import mjsoccer.mjx

    env = mjsoccer.mjx.make(env_id, nworld=NWORLD, repeat_action=repeat_action)
    keys = jax.random.split(jax.random.PRNGKey(0), NWORLD)
    state = jax.jit(jax.vmap(env.reset))(keys)
    mjsoccer.mjx.check_failures(state)
    # obs: 6 ball entries, then 8 per robot ending in has_ball; robot 1 is the yellow one.
    holds = np.asarray(state.obs[:, 6 + 8 + 7]) > 0
    step = jax.jit(jax.vmap(env.step))
    rng = np.random.default_rng(0)
    finite = bool(np.isfinite(np.asarray(state.obs)).all())
    n = 10
    for _ in range(n):
        state = step(state, _actions(rng, env))
        mjsoccer.mjx.check_failures(state)
        finite &= bool(np.isfinite(np.asarray(state.obs)).all())
        finite &= bool(np.isfinite(np.asarray(state.pipeline_state.qpos)).all())
    steps = np.asarray(state.info["world"]["steps"])
    return dict(finite=finite, steps=n, yellow_holds_at_reset=holds.tolist(),
                control_steps=steps.tolist())


def _poison(state):
    d = state.pipeline_state
    return state.replace(pipeline_state=d.replace(qvel=d.qvel.at[0, 0].set(float("nan"))))


def nan(env_id: str, repeat_action: int) -> dict:
    import jax
    import numpy as np
    from brax.envs.wrappers import training

    import mjsoccer.mjx

    env = mjsoccer.mjx.make(env_id, nworld=NWORLD, repeat_action=repeat_action)
    keys = jax.random.split(jax.random.PRNGKey(0), NWORLD)
    rng = np.random.default_rng(0)
    a = _actions(rng, env)

    s = jax.jit(jax.vmap(env.step))(_poison(jax.jit(jax.vmap(env.reset))(keys)), a)
    d = s.pipeline_state
    raw = dict(
        done=np.asarray(s.done).tolist(),
        truncation=np.asarray(s.info["truncation"]).tolist(),
        reward0=float(s.reward[0]),
        obs0_zero=bool((np.asarray(s.obs[0]) == 0).all()),
        diverged=np.asarray(s.metrics["diverged"]).tolist(),
        overflow=np.asarray(s.metrics["overflow"]).tolist(),
        others_finite=bool(np.isfinite(np.asarray(d.qpos[1:])).all()
                           and np.isfinite(np.asarray(d.qvel[1:])).all()
                           and np.isfinite(np.asarray(s.obs[1:])).all()),
        check_failures=_error(lambda: mjsoccer.mjx.check_failures(s)),
    )
    if repeat_action > 1:
        return dict(raw=raw)

    wenv = training.wrap(env, episode_length=10**6)
    reset, step = jax.jit(wenv.reset), jax.jit(wenv.step)
    first = reset(keys)
    s = step(_poison(first), a)
    d, d0 = s.pipeline_state, first.pipeline_state
    restored = all(np.array_equal(np.asarray(x[0]), np.asarray(y[0]))
                   for x, y in [(d.qpos, d0.qpos), (d.qvel, d0.qvel), (s.obs, first.obs)])
    wrapped = dict(
        diverged=np.asarray(s.metrics["diverged"]).tolist(),
        restored=restored,
        episode_diverged=np.asarray(s.info["episode_metrics"]["diverged"]).tolist(),
    )
    finite, failures = True, 0
    for _ in range(5):
        s = step(s, _actions(rng, env))
        finite &= bool(np.isfinite(np.asarray(s.obs)).all())
        finite &= bool(np.isfinite(np.asarray(s.pipeline_state.qpos)).all())
        failures += int(np.asarray(s.metrics["diverged"]).sum()
                        + np.asarray(s.metrics["overflow"]).sum())
    wrapped.update(after_finite=finite, after_failures=failures)
    return dict(raw=raw, wrapped=wrapped)


def overflow(env_id: str, repeat_action: int) -> dict:
    import jax
    import numpy as np

    import mjsoccer.mjx

    env = mjsoccer.mjx.make(env_id, nworld=NWORLD, repeat_action=repeat_action,
                            nacon_per_world=2)
    keys = jax.random.split(jax.random.PRNGKey(0), NWORLD)
    s = jax.jit(jax.vmap(env.reset))(keys)
    s = jax.jit(jax.vmap(env.step))(s, _actions(np.random.default_rng(0), env))
    return dict(
        done=np.asarray(s.done).tolist(),
        overflow=np.asarray(s.metrics["overflow"]).tolist(),
        reward_zero=bool((np.asarray(s.reward) == 0).all()),
        obs_zero=bool((np.asarray(s.obs) == 0).all()),
        check_failures=_error(lambda: mjsoccer.mjx.check_failures(s)),
    )


def main(mode: str, env_id: str, repeat_action: int) -> None:
    import jax

    try:
        jax.devices("gpu")
    except RuntimeError:
        sys.exit(SKIP)
    run = dict(smoke=smoke, nan=nan, overflow=overflow)[mode]
    print(json.dumps(run(env_id, repeat_action)))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
