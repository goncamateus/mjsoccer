"""Domain randomization on mjlab: `mjsoccer.mjlab.make(id, randomize=sampler)`, the same
`sampler(mj_model, rng)` as `mjsoccer.mjx.make(randomize=)` (tests/test_mjx_randomize.py).

What is checked, through the env's public API: the sampler runs on the CPU model (the ids
MJX's samplers use), and a sampler that varies a mass gives each world its own mass *and*
the `body_invweight0` MuJoCo's own `mj_setConst` gives that mass (checked against a fresh
copy of the model), on the body of that name in mjlab's model (whose anchor shifts every
body id); a world's variant depends on the seed and its index alone; the cached CPU model
is nominal again afterwards; the ball's `<pair>` contacts randomize too; a sampler that
touches `opt.timestep` or a structural field is refused; an env made without a sampler
stays nominal; and a randomized batch steps finite. The fields a sampler may vary are the
ones Warp batches, a superset of MJX's (checked in-process when the `gpu` group is there).

    uv sync --group mjlab && uv run pytest tests/test_mjlab_randomize.py

The GPU checks run in a fresh subprocess (`_gpu.py`), skipped without CUDA or the group.
"""

from __future__ import annotations

import copy
import json
import sys

import mujoco
import numpy as np
import pytest
from _gpu import SKIP, run_json

pytest.importorskip("torch")
pytest.importorskip("mjlab")

ENV = "ContestedPossession-v0"
N = 4


def _robot0_body():
    import mjsoccer
    from mjsoccer import tasks

    cpu = getattr(tasks, mjsoccer.ENV_IDS[ENV][0])()
    b = cpu.robots[0].body
    cpu.close()
    return b


def heavier_robot(m, rng):
    """Robot 0's chassis 0.8-1.2x its nominal mass (and inertia with it), by its CPU id."""
    b = _robot0_body()
    k = rng.uniform(0.8, 1.2)
    m.body_mass[b] *= k
    m.body_inertia[b] *= k


def softer_ball(m, rng):
    """The ball's contact pairs' solref timeconst 1-2x nominal."""
    g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "ball")
    p = np.flatnonzero((m.pair_geom1 == g) | (m.pair_geom2 == g))
    m.pair_solref[p, 0] *= rng.uniform(1.0, 2.0)


def slower_clock(m, rng):
    m.opt.timestep *= 0.5


def more_contypes(m, rng):
    m.geom_contype[0] += 1


def test_every_field_mjx_randomizes_mjlab_randomizes():
    pytest.importorskip("jax")
    mjx = pytest.importorskip("mujoco.mjx")
    import mjsoccer.mjx
    from mjsoccer.mjlab import env as mjlab_env
    from mjsoccer.mjx import randomize

    m = mjsoccer.mjx.make(ENV).mj_model
    top, opt = randomize.variable_fields(mjx.put_model(m, impl="jax"))
    mine_top, mine_opt = mjlab_env.variable_fields(m)
    # Every sampler MJX takes, mjlab takes; mjlab also batches a few fields the JAX-side
    # mjx.Model lacks (site_quat, actuator_armature, opt.tolerance, ...).
    assert {a for a in top if hasattr(m, a)} <= set(mine_top)
    assert set(opt) <= set(mine_opt)


def test_mass_sampler_gives_each_world_its_mass_and_invweight():
    out = run_json(__file__, "mass")
    nom_mass, nom_inertia = out["nominal_mass"], np.array(out["nominal_inertia"])
    mass = np.array(out["mass"])
    assert len(set(np.round(mass, 6))) == N, out  # every world its own
    assert np.all((mass >= 0.8 * nom_mass - 1e-5) & (mass <= 1.2 * nom_mass + 1e-5)), out
    assert out["other_bodies_nominal"], out
    # The derived field, against MuJoCo itself on a fresh CPU model with the world's mass.
    from mjsoccer import scene

    b = _robot0_body()
    for i in range(N):
        ref = copy.copy(scene.build(1, 1)[0])  # the cached model, never touched
        k = mass[i] / nom_mass
        ref.body_mass[b] *= k
        ref.body_inertia[b] *= k
        mujoco.mj_setConst(ref, mujoco.MjData(ref))
        np.testing.assert_allclose(out["invweight"][i], ref.body_invweight0[b], rtol=1e-5,
                                   atol=1e-7)
        np.testing.assert_allclose(out["inertia"][i], nom_inertia * k, rtol=1e-5)
    assert out["cpu_model_nominal"], out
    assert out["finite"] and out["failures"] == 0, out


def test_a_world_s_variant_is_its_seed_and_index_alone():
    out = run_json(__file__, "seeds")
    assert out["same_seed_equal"], out
    assert out["prefix_equal"], out  # world i of 2 worlds is world i of 4, same seed
    assert not out["other_seed_equal"], out


def test_ball_pair_solref_randomizes_per_world():
    out = run_json(__file__, "pair")
    k = np.array(out["timeconst"]) / out["nominal"]
    assert len(set(np.round(k, 6))) == N and np.all((k >= 1 - 1e-6) & (k <= 2 + 1e-6)), out
    assert out["finite"] and out["failures"] == 0, out


def test_refused_samplers_and_the_nominal_env():
    out = run_json(__file__, "refused")
    assert "timestep" in out["slower_clock"], out
    assert "contype" in out["more_contypes"], out
    assert out["cpu_model_nominal"], out
    assert out["nominal_not_expanded"], out


# ------------------------------------------------------------------ the children


def _make(sampler, nworld=N, seed=0):
    import mjsoccer.mjlab

    return mjsoccer.mjlab.make(ENV, nworld=nworld, randomize=sampler, seed=seed)


def _step(env, n=5):
    import torch

    fail = 0
    for _ in range(n):
        env.step(torch.zeros(env.num_envs, env.logic.action_size, device=env.device))
        fail += int(env.metrics["diverged"].sum() + env.metrics["overflow"].sum())
    d = env.sim.data
    finite = bool(torch.isfinite(env.obs).all() and torch.isfinite(d.qpos[:]).all())
    return finite, fail


def _snapshot(m):
    return {a: getattr(m, a).copy() for a in dir(m)
            if not a.startswith("_") and isinstance(getattr(m, a), np.ndarray)}


def _same(m, snap):
    return all(np.array_equal(getattr(m, a), v) for a, v in snap.items())


def mass() -> dict:
    env = _make(heavier_robot)
    cpu = env.logic.mj_model
    snap = _snapshot(cpu)
    env.reset()
    b = env.logic.robots[0].body  # mjlab's id of robot 0
    m = env.sim.model
    mass = m.body_mass[:, b].cpu().numpy()
    nominal = env.sim.mj_model.body_mass
    others = np.delete(m.body_mass.cpu().numpy(), b, axis=1)
    finite, fail = _step(env)
    return {"mass": mass.tolist(), "nominal_mass": float(nominal[b]),
            "nominal_inertia": env.sim.mj_model.body_inertia[b].tolist(),
            "inertia": m.body_inertia[:, b].cpu().numpy().tolist(),
            "invweight": m.body_invweight0[:, b].cpu().numpy().tolist(),
            "other_bodies_nominal": bool(np.allclose(others, np.delete(nominal, b), rtol=1e-6)),
            "cpu_model_nominal": _same(cpu, snap), "finite": finite, "failures": fail}


def seeds() -> dict:
    def masses(nworld, seed):
        env = _make(heavier_robot, nworld, seed)
        out = env.sim.model.body_mass[:, env.logic.robots[0].body].cpu().numpy()
        env.close()
        return out

    a, b, c, d = masses(N, 7), masses(N, 7), masses(2, 7), masses(N, 8)
    return {"same_seed_equal": bool(np.array_equal(a, b)),
            "prefix_equal": bool(np.array_equal(c, a[:2])),
            "other_seed_equal": bool(np.array_equal(a, d))}


def pair() -> dict:
    env = _make(softer_ball)
    mj = env.sim.mj_model
    g = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_GEOM, "ball")
    p = int(np.flatnonzero((mj.pair_geom1 == g) | (mj.pair_geom2 == g))[0])
    tc = env.sim.model.pair_solref[:, p, 0].cpu().numpy()
    env.reset()
    finite, fail = _step(env)
    return {"timeconst": tc.tolist(), "nominal": float(mj.pair_solref[p, 0]),
            "finite": finite, "failures": fail}


def refused() -> dict:
    from mjsoccer import scene

    cpu = scene.build(1, 1)[0]  # the cached CPU model the samplers run on
    snap = _snapshot(cpu)
    out = {}
    for s in (slower_clock, more_contypes):
        try:
            _make(s)
            out[s.__name__] = ""
        except ValueError as e:
            out[s.__name__] = str(e)
    env = _make(None)
    out["cpu_model_nominal"] = env.logic.mj_model is cpu and _same(cpu, snap)
    out["nominal_not_expanded"] = env.sim.wp_model.body_mass.shape[0] == 1
    return out


def main(mode: str) -> None:
    try:
        import mjlab  # noqa: F401
        import torch
    except ImportError:
        sys.exit(SKIP)
    if not torch.cuda.is_available():
        sys.exit(SKIP)
    print(json.dumps({"mass": mass, "seeds": seeds, "pair": pair, "refused": refused}[mode]()))


if __name__ == "__main__":
    main(sys.argv[1])
