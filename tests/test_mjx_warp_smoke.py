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

TASKS = ["ContestedPossession-v0", "GoToBall-v0"]
NWORLD = 4
SKIP = 77  # the child's exit code for "no CUDA device"


def _run(*args) -> dict:
    env = {k: v for k, v in os.environ.items() if k != "JAX_PLATFORMS"}
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    p = subprocess.run(
        [sys.executable, __file__, *map(str, args)],
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
    assert out["overflow_bits"] == [0] * NWORLD, out  # check_failures also aborts on these
    assert out["reset_overlap"] == [0.0] * NWORLD, out
    if env_id == "ContestedPossession-v0":
        # The seat_ball loop really ran on Warp: the yellow robot holds the ball at reset,
        # in every world, as it does on the CPU for every seed.
        assert all(out["yellow_holds_at_reset"]), out
    if env_id == "GoToBall-v0":
        # The scatter: seven robots and the ball, pairwise at least 0.2 m apart.
        assert min(out["min_spacing_at_reset"]) >= 0.2 - 1e-5, out
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


@pytest.mark.parametrize("env_id", ["ContestedPossession-v0"])
def test_warp_kick_chip_and_frozen_idle_worlds(env_id):
    """With a ball seated on the blue robot in every world, world 0 kicks, world 1 chips,
    2 and 3 do neither. The kick sends the ball off fast and low, the chip lifts it, the
    idle balls stay put. Inside that firing control step, the idle worlds' qpos/qvel do not
    change by a bit over the iterations they spend at timestep 0."""
    out = _run("kick", env_id)
    assert out["fired"] == [True, True, False, False], out
    assert out["lock"] == [[9, 0], [0, 9], [0, 0], [0, 0]], out
    kick, chip, *idle = range(NWORLD)
    assert out["speed"][kick] > 5.0 and out["lift"][kick] < 0.05, out  # asked for 6.5 m/s
    assert out["speed"][chip] > 4.0 and out["lift"][chip] > 0.2, out
    assert all(out["speed"][w] < 1.0 and out["lift"][w] < 0.02 for w in idle), out
    assert out["frozen_iters"] == 24, out
    assert out["idle_bitwise"] == [False, False, True, True], out


def kick(env_id: str) -> None:
    """The child for test_warp_kick_chip_and_frozen_idle_worlds."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    try:
        jax.devices("gpu")
    except RuntimeError:
        sys.exit(SKIP)

    import mjsoccer.mjx
    from mjsoccer import scene

    env = mjsoccer.mjx.make(env_id, nworld=NWORLD)
    keys = jax.random.split(jax.random.PRNGKey(0), NWORLD)
    state = jax.jit(jax.vmap(env.reset))(keys)
    blue = env.robots[0]
    d = jax.jit(jax.vmap(lambda d: env.seat_ball(d, blue)))(state.pipeline_state)
    state = jax.jit(jax.vmap(env.init_state))(d, keys)
    # Hold the pose the robot is placed at (origin, facing +x), dribbler off.
    a = np.zeros((NWORLD, env.action_size), np.float32)
    a[:, 3], a[:, 4:7] = 1.0, -1.0
    a[0, 4] = a[1, 5] = 1.0  # world 0 kicks, world 1 chips, at the 6.5 m/s cap
    rest = a.copy()
    rest[:, 4:6] = -1.0

    # Inside the firing step, iteration by iteration.
    c = jax.jit(jax.vmap(env.control))(state, a)
    fired = np.asarray(c.info["world"]["fired"])
    ts = jax.vmap(env.timesteps)(jnp.asarray(fired.any(-1)))
    substep = jax.jit(jax.vmap(env.substep))
    d, n = c.pipeline_state, env.decimation
    for i in range(int(ts.shape[1])):
        if i == n:
            ref = np.asarray(d.qpos), np.asarray(d.qvel)
        d = substep(d, ts[:, i])
    idle = [bool((np.asarray(d.qpos)[w] == ref[0][w]).all() and
                 (np.asarray(d.qvel)[w] == ref[1][w]).all()) for w in range(NWORLD)]

    # And through env.step, the ball afterwards.
    step = jax.jit(jax.vmap(env.step))
    state = step(state, a)
    lock = np.asarray(state.info["world"]["plunger_lock"])[:, 0].tolist()
    z0 = float(scene.FLOOR_Z + env.mj_model.geom_size[env.ball.geom][0])
    speed = np.zeros(NWORLD)
    lift = np.zeros(NWORLD)
    for _ in range(15):
        bd = state.pipeline_state
        speed = np.maximum(speed, np.linalg.norm(
            np.asarray(bd.qvel[:, env.ball.qvel_adr : env.ball.qvel_adr + 3]), axis=1))
        lift = np.maximum(lift, np.asarray(bd.xpos[:, env.ball.body, 2]) - z0)
        state = step(state, rest)
    print(json.dumps(dict(
        fired=fired[:, 0].tolist(), lock=lock, speed=speed.tolist(), lift=lift.tolist(),
        frozen_iters=int((np.asarray(ts) == 0).sum(1).max()), idle_bitwise=idle,
    )))


# ------------------------------------------------------------------ the children


def _actions(rng, env):
    return rng.uniform(-1, 1, (NWORLD, env.action_size)).astype("float32")


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
    from mjsoccer.mjx.base import overflow_bits

    env = mjsoccer.mjx.make(env_id, nworld=NWORLD, repeat_action=repeat_action)
    keys = jax.random.split(jax.random.PRNGKey(0), NWORLD)
    state = jax.jit(jax.vmap(env.reset))(keys)
    mjsoccer.mjx.check_failures(state)
    # obs: 6 ball entries, then 8 per robot ending in has_ball; robot 1 is the yellow one.
    holds = np.asarray(state.obs[:, 6 + 8 + 7]) > 0
    xy = np.concatenate([np.asarray(state.pipeline_state.xpos[:, env._body, :2]),
                         np.asarray(state.pipeline_state.xpos[:, env.ball.body, None, :2])], 1)
    gaps = np.linalg.norm(xy[:, :, None] - xy[:, None], axis=-1)
    gaps[:, np.arange(xy.shape[1]), np.arange(xy.shape[1])] = np.inf
    overlap = np.asarray(state.metrics["reset_overlap"]).tolist()
    bits = np.zeros(NWORLD, np.int64)
    step = jax.jit(jax.vmap(env.step))
    rng = np.random.default_rng(0)
    finite = bool(np.isfinite(np.asarray(state.obs)).all())
    n = 10
    for _ in range(n):
        state = step(state, _actions(rng, env))
        mjsoccer.mjx.check_failures(state)
        bits |= np.asarray(overflow_bits(state.pipeline_state)).astype(np.int64)
        finite &= bool(np.isfinite(np.asarray(state.obs)).all())
        finite &= bool(np.isfinite(np.asarray(state.pipeline_state.qpos)).all())
    steps = np.asarray(state.info["world"]["steps"])
    return dict(finite=finite, steps=n, yellow_holds_at_reset=holds.tolist(),
                control_steps=steps.tolist(), reset_overlap=overlap,
                min_spacing_at_reset=gaps.min((1, 2)).tolist(), overflow_bits=bits.tolist())


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
    if sys.argv[1] == "kick":
        kick(sys.argv[2])
    else:
        main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
