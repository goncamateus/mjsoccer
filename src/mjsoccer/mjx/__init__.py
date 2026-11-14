"""The MJX-Warp backend: the six tasks again, in functional JAX, as `brax.envs.Env`.

    env = mjsoccer.mjx.make("ContestedPossession-v0", nworld=1024)
    state = jax.jit(jax.vmap(env.reset))(jax.random.split(key, 1024))
    state = jax.jit(jax.vmap(env.step))(state, actions)

For training, wrap with `mjsoccer.mjx.wrappers.wrap`, not Brax's `training.wrap` (which
neither resets the per-world state nor keeps the env's truncation); examples/ppo_mjx.py.

A second implementation of the CPU tasks (`mjsoccer.tasks`) over MJX-Warp, on the exact base
model; the CPU backend stays the physics oracle (ADR 0002). Every constant and task config is
read off the CPU side, and tests/test_mjx_task_logic.py holds the two together state by
state. Needs the optional group (`uv sync --group gpu`) and, to step, a CUDA GPU.
`import mjsoccer` never imports this package.
"""

from __future__ import annotations

from .. import ENV_IDS
from . import tasks
from .base import MjxEnv, check_failures

__all__ = ["ENV_IDS", "MjxEnv", "check_failures", "make"]


def make(env_id: str, *, repeat_action: int = 1, graph_mode: str = "WARP", randomize=None,
         **kw) -> MjxEnv:
    """The MJX twin of `gymnasium.make(env_id)`: same ids, same horizons (`ENV_IDS`).

    `nworld` (in `kw`) is how many worlds the env will be vmapped over; it sizes Warp's
    contact buffer, which is shared by the whole batch. Undersizing it overflows;
    `nacon_per_world` (in `kw`) overrides the per-world share of it.
    """
    cls_name, horizon_s = ENV_IDS[env_id]
    cls = getattr(tasks, cls_name, None)
    if cls is None:
        raise NotImplementedError(f"{env_id} is not ported to MJX yet")
    return cls(horizon_s=horizon_s, repeat_action=repeat_action, graph_mode=graph_mode,
               randomize=randomize, **kw)
