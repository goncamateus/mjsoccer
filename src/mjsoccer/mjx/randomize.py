"""Domain randomization (DR) per world slot: one model variant per slot, for the whole run.

    def sampler(m: mujoco.MjModel, rng: np.random.Generator) -> None:
        m.body_mass[b] *= rng.uniform(0.9, 1.1)        # mutate in place, return nothing

    env = mjsoccer.mjx.make(env_id, randomize=sampler, nworld=n)
    ppo.train(env, wrap_env_fn=wrappers.wrap, randomization_fn=env.randomization_fn, ...)

`variants` (`mjsoccer.randomize`, shared with the mjlab backend) runs the sampler once
per slot on the env's cached `MjModel` (shared with the CPU backend, never copied: 170 ms
with 7 robots), calls `mj_setConst` so the derived fields (`body_invweight0`,
`dof_invweight0`, `actuator_acc0`, `body_subtreemass`, ...) match the new values, copies
what differs from the nominal into a stack, and puts the nominal back. The fields that
differ in any slot get `in_axes` 0, everything else None: Brax's `randomization_fn(sys, rng) -> (sys, in_axes)`, with one numpy RNG per slot seeded
from that slot's key, so a slot's variant depends on its key alone.

Only real-valued fields that the MJX model batches on both engines can vary (`jax.Array`
on the `mjx.Model` and batched in MJX-Warp's `_BATCH_DIM`), plus the float `opt` fields.
`opt.timestep` is refused (the kick's firing schedule and `refsafe` are not DR), and so is
any other field: structural ones (meshes, `contype`, ids, `*_limited`) need a recompile.

The ball's contacts are explicit `<pair>`s: their friction and `solref`/`solimp` are
`pair_*` (`pair_friction`, `pair_solref`, ...), and the `geom_*` of the ball, the carpet
or the dribbler are not read for them.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from ..randomize import variants


def _batch_dim() -> dict:
    from mujoco.mjx.warp import types as wt

    return wt._BATCH_DIM["Model"]


def variable_fields(sys) -> tuple[list[str], list[str]]:
    """(model fields, `opt` fields) a sampler may change: the ones `in_axes` can batch."""
    bd = _batch_dim()
    top = [f.name for f in dataclasses.fields(sys) if not f.name.startswith("_")
           and isinstance(getattr(sys, f.name), jax.Array) and bd.get(f.name, False)]
    opt = [f.name for f in dataclasses.fields(sys.opt) if not f.name.startswith("_")
           and isinstance(getattr(sys.opt, f.name), jax.Array)
           and bd.get(f"opt__{f.name}", False) and f.name != "timestep"]
    return top, opt


def slot_rng(key) -> np.random.Generator:
    """The numpy RNG of one slot, from its JAX key (raw uint32 or typed)."""
    k = jax.random.key_data(key) if jnp.issubdtype(key.dtype, jax.dtypes.prng_key) else key
    return np.random.default_rng(np.asarray(k, np.uint32).ravel())


def randomization_fn(m: mujoco.MjModel, sampler, sys, rng):
    """Brax's `randomization_fn(sys, rng) -> (sys_v, in_axes)` for `sampler` on `m`."""
    top, opt = variable_fields(sys)
    stacks = variants(m, sampler, [slot_rng(k) for k in rng], top, opt)
    in_axes = jax.tree.map(lambda _: None, sys)
    sys_v, axes, opt_v, opt_axes = {}, {}, {}, {}
    for a, v in stacks.items():
        if a.startswith("opt."):
            f = a[4:]
            opt_v[f] = jnp.asarray(v, getattr(sys.opt, f).dtype)
            opt_axes[f] = 0
        else:
            sys_v[a] = jnp.asarray(v, getattr(sys, a).dtype)
            axes[a] = 0
    if opt_v:
        sys_v["opt"], axes["opt"] = sys.opt.replace(**opt_v), in_axes.opt.replace(**opt_axes)
    return sys.replace(**sys_v), in_axes.replace(**axes)
