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


def _run(env_id: str, repeat_action: int) -> dict:
    env = {k: v for k, v in os.environ.items() if k != "JAX_PLATFORMS"}
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    p = subprocess.run(
        [sys.executable, __file__, env_id, str(repeat_action)],
        env=env, capture_output=True, text=True, timeout=1800,
    )
    if p.returncode == SKIP:
        pytest.skip("no CUDA device")
    assert p.returncode == 0, p.stderr[-4000:]
    return json.loads(p.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("env_id", TASKS)
@pytest.mark.parametrize("repeat_action", [1, 2])
def test_warp_smoke(env_id, repeat_action):
    out = _run(env_id, repeat_action)
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


def main(env_id: str, repeat_action: int) -> None:
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
        a[:, 4:6] = -np.abs(a[:, 4:6])  # kick and chip at rest: ticket 08
        state = step(state, a)
        finite &= bool(np.isfinite(np.asarray(state.obs)).all())
        finite &= bool(np.isfinite(np.asarray(state.pipeline_state.qpos)).all())
    steps = np.asarray(state.info["world"]["steps"])
    print(json.dumps(dict(
        finite=finite, steps=n, yellow_holds_at_reset=holds.tolist(),
        control_steps=steps.tolist(),
    )))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
