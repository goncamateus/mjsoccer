"""Domain randomization on MJX: `mjsoccer.mjx.make(id, randomize=sampler)`.

What is checked, through the env's public API (`env.randomization_fn`, what Brax's
`ppo.train` calls, and `wrappers.wrap`): a sampler that varies a mass gives each slot its
own mass *and* the `body_invweight0` MuJoCo's own `mj_setConst` gives that mass (checked
against a fresh copy of the model, not the env's code path); the same keys give the same
variants; the cached model is nominal again afterwards; a sampler that touches
`opt.timestep` or a structural field is refused; and an env made without a sampler stays
nominal under `wrap` even when Brax hands it a randomization_fn. The logic runs on
JAX-CPU against a JAX `mjx.Model`; the last test steps a randomized batch on MJX-Warp in a
subprocess (skipped without CUDA), the way the smoke does.

    uv sync --group gpu && uv run pytest tests/test_mjx_randomize.py
"""

from __future__ import annotations

import copy
import json
import sys

import mujoco
import numpy as np
import pytest
from _gpu import SKIP, run_json

jax = pytest.importorskip("jax")
mjx = pytest.importorskip("mujoco.mjx")
pytest.importorskip("brax")

import mjsoccer.mjx
from mjsoccer.mjx import wrappers

ENV = "ContestedPossession-v0"
N = 4


def heavier_robot(m, rng):
    """Robot 0's chassis 0.8-1.2x its nominal mass (and inertia with it)."""
    b = ROBOT0
    k = rng.uniform(0.8, 1.2)
    m.body_mass[b] *= k
    m.body_inertia[b] *= k


ROBOT0 = mjsoccer.mjx.make(ENV).robots[0].body


@pytest.fixture(scope="module")
def env():
    return mjsoccer.mjx.make(ENV, randomize=heavier_robot, nworld=N)


def _sys(env):
    # impl="jax": a CPU-only test harness for the DR logic, never a backend (ADR 0002).
    return mjx.put_model(env.mj_model, impl="jax")


def _snapshot(m):
    return {a: getattr(m, a).copy() for a in dir(m)
            if not a.startswith("_") and isinstance(getattr(m, a), np.ndarray)}


def test_mass_sampler_gives_each_slot_its_mass_and_invweight(env):
    m = env.mj_model
    nominal = _snapshot(m)
    sys_v, in_axes = env.randomization_fn(_sys(env), jax.random.split(jax.random.PRNGKey(0), N))
    b = env.robots[0].body
    mass = np.asarray(sys_v.body_mass)
    assert mass.shape == (N, m.nbody)
    assert len(set(np.round(mass[:, b], 6))) == N  # every slot its own
    assert np.all((mass[:, b] >= 0.8 * nominal["body_mass"][b] - 1e-5)
                  & (mass[:, b] <= 1.2 * nominal["body_mass"][b] + 1e-5))
    others = np.delete(mass, b, axis=1)
    np.testing.assert_allclose(others, np.broadcast_to(np.delete(nominal["body_mass"], b), others.shape),
                               rtol=1e-6)  # float32 on the mjx.Model
    # The derived field, against MuJoCo itself on a fresh copy with the slot's mass.
    for i in range(N):
        ref = copy.copy(m)
        k = mass[i, b] / nominal["body_mass"][b]
        ref.body_mass[b] *= k
        ref.body_inertia[b] *= k
        mujoco.mj_setConst(ref, mujoco.MjData(ref))
        np.testing.assert_allclose(np.asarray(sys_v.body_invweight0)[i], ref.body_invweight0,
                                   rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(np.asarray(sys_v.dof_invweight0)[i], ref.dof_invweight0,
                                   rtol=1e-5, atol=1e-7)
    assert in_axes.body_mass == 0 and in_axes.body_invweight0 == 0
    assert in_axes.geom_friction is None and in_axes.pair_friction is None
    assert in_axes.opt.timestep is None
    # The cached model (shared with the CPU backend) is nominal again, bit for bit.
    for a, v in nominal.items():
        np.testing.assert_array_equal(getattr(m, a), v, err_msg=a)


def test_same_seed_same_variants(env):
    keys = jax.random.split(jax.random.PRNGKey(7), N)
    a, _ = env.randomization_fn(_sys(env), keys)
    b, _ = env.randomization_fn(_sys(env), keys)
    np.testing.assert_array_equal(np.asarray(a.body_mass), np.asarray(b.body_mass))
    np.testing.assert_array_equal(np.asarray(a.body_invweight0), np.asarray(b.body_invweight0))
    # Per slot: slot 1 of a batch is slot 0 of the batch that starts at its key.
    c, _ = env.randomization_fn(_sys(env), keys[1:3])
    np.testing.assert_array_equal(np.asarray(c.body_mass), np.asarray(a.body_mass)[1:3])
    d, _ = env.randomization_fn(_sys(env), jax.random.split(jax.random.PRNGKey(8), N))
    assert not np.array_equal(np.asarray(d.body_mass), np.asarray(a.body_mass))


@pytest.mark.parametrize("field", ["timestep", "contype"])
def test_sampler_that_changes_timestep_or_a_structural_field_is_refused(field):
    def bad(m, rng):
        if field == "timestep":
            m.opt.timestep *= 0.5
        else:
            m.geom_contype[0] += 1

    e = mjsoccer.mjx.make(ENV, randomize=bad, nworld=N)
    nominal = _snapshot(e.mj_model)
    with pytest.raises(ValueError, match=field):
        e.randomization_fn(_sys(e), jax.random.split(jax.random.PRNGKey(0), N))
    for a, v in nominal.items():
        np.testing.assert_array_equal(getattr(e.mj_model, a), v, err_msg=a)
    assert e.mj_model.opt.timestep == mjsoccer.mjx.make(ENV).mj_model.opt.timestep


def test_without_a_sampler_the_env_stays_nominal_under_wrap():
    """`ppo.train` hands the same randomization_fn to the train and the eval env; an env
    made with `randomize=None` (the default eval) never calls it."""
    e = mjsoccer.mjx.make(ENV, nworld=N)
    assert e.randomization_fn is None

    def boom(sys):
        raise AssertionError("the nominal env called the randomization_fn")

    wrappers.wrap(e, episode_length=10, randomization_fn=boom)


def test_randomized_batch_steps_on_warp():
    out = run_json(__file__, "warp")
    assert out["finite"] and out["failures"] == 0, out
    assert len(set(out["mass"])) == N, out


def _warp_main():
    """Child: Brax's own randomization wrapper over the env's randomization_fn, 5 steps."""
    import functools

    import jax.numpy as jnp

    if not any(d.platform == "gpu" for d in jax.devices()):
        sys.exit(SKIP)
    e = mjsoccer.mjx.make(ENV, randomize=heavier_robot, nworld=N)
    keys = jax.random.split(jax.random.PRNGKey(0), N)
    w = wrappers.wrap(e, episode_length=wrappers.episode_length(e),
                      randomization_fn=functools.partial(e.randomization_fn, rng=keys))
    s = jax.jit(w.reset)(jax.random.split(jax.random.PRNGKey(1), N))
    step = jax.jit(w.step)
    fail = 0
    for _ in range(5):
        s = step(s, jnp.zeros((N, e.action_size)))
        fail += int(np.asarray(s.metrics["diverged"]).sum() + np.asarray(s.metrics["overflow"]).sum())
    finite = bool(np.isfinite(np.asarray(s.obs)).all()
                  and np.isfinite(np.asarray(s.pipeline_state.qpos)).all())
    mass = np.asarray(w._sys_v.body_mass)[:, e.robots[0].body].round(6).tolist()
    print(json.dumps({"finite": finite, "failures": fail, "mass": mass}))


if __name__ == "__main__" and sys.argv[1:] == ["warp"]:
    _warp_main()
