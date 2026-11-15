"""Domain randomization (DR) variants, shared by the GPU backends (`mjsoccer.mjx.randomize`,
`mjsoccer.mjlab`): numpy and mujoco only, so neither backend's stack is imported.

A sampler `sampler(m: mujoco.MjModel, rng: np.random.Generator) -> None` mutates the CPU
model in place. `variants` runs it once per world on the env's cached `MjModel`
(shared with the CPU backend, never copied: 170 ms with 7 robots), calls `mj_setConst` so
the derived fields (`body_invweight0`, `dof_invweight0`, `actuator_acc0`,
`body_subtreemass`, ...) match the new values, copies what differs from the nominal into a
stack, and puts the nominal back. Only the fields Warp batches per world may change (the
backend passes them); `opt.timestep` is refused (the kick's firing schedule and `refsafe`
are not DR), and so is any other field: structural ones (meshes, `contype`, ids,
`*_limited`) need a recompile. The CPU backend itself is never randomized.
"""

from __future__ import annotations

import zlib

import mujoco
import numpy as np

_SMALL = 1 << 20
# Derived by mj_setConst, but unbatched in Warp and read by neither engine: put back
# after every slot, never stacked.
UNREAD = ("dof_M0",)


def _arrays(m: mujoco.MjModel) -> list[str]:
    return [a for a in dir(m) if not a.startswith("_") and isinstance(getattr(m, a), np.ndarray)]


def _digest(m, names) -> dict:
    return {a: zlib.crc32(np.ascontiguousarray(getattr(m, a))) for a in names}


def variants(m: mujoco.MjModel, sampler, rngs, top, opt) -> dict[str, np.ndarray]:
    """{"field" or "opt.field": (n, ...) stack} of what the sampler changed, derived fields
    included; one slot per numpy Generator in `rngs`. `top`/`opt`: the model and `opt`
    fields that may vary. `m` is nominal again on return, raise or not."""
    names = [a for a in top if hasattr(m, a)]
    nominal = {a: getattr(m, a).copy() for a in names + list(UNREAD)}
    nominal_opt = {a: np.array(getattr(m.opt, a), copy=True) for a in opt}
    timestep = m.opt.timestep
    rest = [a for a in _arrays(m) if a not in nominal]
    before = _digest(m, rest)
    # The fields no sampler may touch, kept to put back if one does; the big ones (meshes,
    # textures: ~300 MB with 7 robots) only by checksum.
    small = {a: getattr(m, a).copy() for a in rest if getattr(m, a).nbytes <= _SMALL}
    d = mujoco.MjData(m)
    changed: dict[str, list] = {}

    def restore(everything=False):
        for a, v in (nominal | small if everything else nominal).items():
            getattr(m, a)[...] = v
        for a, v in nominal_opt.items():
            setattr(m.opt, a, v)
        m.opt.timestep = timestep

    try:
        for i, rng in enumerate(rngs):
            sampler(m, rng)
            if m.opt.timestep != timestep:
                raise ValueError(f"the sampler changed opt.timestep ({timestep} -> "
                                 f"{m.opt.timestep}); timestep is not randomized")
            mujoco.mj_setConst(m, d)
            now = {a: getattr(m, a) for a in names}
            now.update({f"opt.{a}": np.asarray(getattr(m.opt, a)) for a in opt})
            ref = {**nominal, **{f"opt.{a}": v for a, v in nominal_opt.items()}}
            for a, v in now.items():
                if a not in changed and not np.array_equal(v, ref[a]):
                    changed[a] = [ref[a]] * i  # slots before this one were nominal here
                if a in changed:
                    changed[a].append(np.array(v, copy=True))
            restore()
        after = _digest(m, rest)
        bad = [a for a in rest if after[a] != before[a]]
        if bad:
            restore(everything=True)
            raise ValueError(f"the sampler changed {', '.join(bad)}: not a field domain "
                             "randomization can vary per world (structural, or not batched "
                             "by the GPU backends); see mjsoccer.randomize")
    finally:
        restore()
    return {a: np.stack(v) for a, v in changed.items()}
