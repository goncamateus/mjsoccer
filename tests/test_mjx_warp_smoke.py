"""Warp smoke: each ported task on MJX-Warp makes, resets and takes ~10 steps under `vmap`.

Not a physics check (the parity against the CPU is statistical, `-m validation`): this only
proves the JAX env runs end to end on the GPU -- placement, the seat_ball loop, the
controller, possession, observation -- and stays finite.

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
    # The seat_ball loop really ran on Warp: the yellow robot holds the ball at reset,
    # in every world, as it does on the CPU for every seed.
    assert all(out["yellow_holds_at_reset"]), out
    # A decision is repeat_action control steps, cut short only by an ending.
    if repeat_action == 1:
        assert out["control_steps"] == [10] * NWORLD, out
    assert all(10 <= k <= 10 * repeat_action for k in out["control_steps"]), out
    assert max(out["control_steps"]) > 10 or repeat_action == 1, out


@pytest.mark.parametrize("env_id", TASKS)
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


def smoke(env_id: str, repeat_action: int) -> None:
    import jax
    import numpy as np

    try:
        jax.devices("gpu")
    except RuntimeError:
        sys.exit(SKIP)

    import mjsoccer.mjx

    env = mjsoccer.mjx.make(env_id, nworld=NWORLD, repeat_action=repeat_action)
    keys = jax.random.split(jax.random.PRNGKey(0), NWORLD)
    state = jax.jit(jax.vmap(env.reset))(keys)
    # obs: 6 ball entries, then 8 per robot ending in has_ball; robot 1 is the yellow one.
    holds = np.asarray(state.obs[:, 6 + 8 + 7]) > 0
    step = jax.jit(jax.vmap(env.step))
    rng = np.random.default_rng(0)
    finite = bool(np.isfinite(np.asarray(state.obs)).all())
    n = 10
    for _ in range(n):
        a = rng.uniform(-1, 1, (NWORLD, env.action_size)).astype(np.float32)
        state = step(state, a)
        finite &= bool(np.isfinite(np.asarray(state.obs)).all())
        finite &= bool(np.isfinite(np.asarray(state.pipeline_state.qpos)).all())
    steps = np.asarray(state.info["world"]["steps"])
    print(json.dumps(dict(
        finite=finite, steps=n, yellow_holds_at_reset=holds.tolist(),
        control_steps=steps.tolist(),
    )))


if __name__ == "__main__":
    if sys.argv[1] == "kick":
        kick(sys.argv[2])
    else:
        smoke(sys.argv[2], int(sys.argv[3]))
