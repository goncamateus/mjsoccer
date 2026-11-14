"""Training-side wrappers for the MJX envs: `wrap`, Brax's training wrapper stack fitted to
an env that owns its episode (later: observation noise and action latency, ticket 15).

    env = mjsoccer.mjx.make(env_id, nworld=n)
    train_env = wrap(env, episode_length=episode_length(env))   # what ppo.train wants

Brax's `training.wrap` gets two things wrong for these envs:

- its `AutoResetWrapper` puts back only `pipeline_state` and `obs`, so the per-world state
  in `info["world"]` (`steps`, `grace`, `plunger_lock`, the task's own) runs on from the
  episode that ended -- a reset world inherits its predecessor's step count and truncates
  early;
- its `EpisodeWrapper` writes `info["truncation"]` from its own counter only, and as
  `1 - done`, so the env's own timeout (done, flagged truncation by the env) comes out as
  truncation 0 -- a terminal, nothing to bootstrap from.

`EpisodeWrapper` here keeps the env's truncation and adds its own on top; `AutoResetWrapper`
restores `info["world"]` with the rest, and merges the data with `mjx.Data.where`, which
knows Warp's batch-shared buffers. The reset is Brax's cached first state per world slot
(a pool of reset states is out of scope; spec, Out of Scope).
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from brax.envs.base import Env, State, Wrapper
from brax.envs.wrappers import training


def episode_length(env) -> int:
    """The env's horizon in decisions, Brax's `episode_length`: its own truncation and the
    wrapper's then fall on the same step."""
    return math.ceil(env.max_steps / env.repeat_action)


def wrap(env: Env, episode_length: int, action_repeat: int = 1,
         randomization_fn=None) -> Wrapper:
    """`brax.envs.wrappers.training.wrap` with this module's episode and auto-reset
    wrappers; the signature `ppo.train(wrap_env_fn=...)` calls. `action_repeat` must stay
    1: the repeat is the env's `repeat_action`, which runs the physics as the CPU does."""
    if action_repeat != 1:
        raise ValueError("action_repeat must be 1; pass repeat_action to mjsoccer.mjx.make")
    if randomization_fn is None:
        env = training.VmapWrapper(env)
    else:
        env = training.DomainRandomizationVmapWrapper(env, randomization_fn)
    return AutoResetWrapper(EpisodeWrapper(env, episode_length, 1))


class EpisodeWrapper(training.EpisodeWrapper):
    """Brax's episode bookkeeping (`info["steps"]`, `episode_metrics`, `episode_done`) for
    one env step per decision, keeping the env's own `info["truncation"]`."""

    def step(self, state: State, action: jax.Array) -> State:
        state = self.env.step(state, action)
        steps = state.info["steps"] + 1
        timeout = steps >= self.episode_length
        # The env flags truncation only when it did not terminate; `done` without it is
        # a terminal, which a timeout on the same step does not turn into a truncation.
        terminal = (state.done > 0) & (state.info["truncation"] == 0)
        truncation = jnp.where(terminal, 0.0, jnp.maximum(state.info["truncation"], timeout))
        done = jnp.maximum(state.done, timeout.astype(state.done.dtype))
        prev_done = state.info["episode_done"]
        em = dict(state.info["episode_metrics"])
        em["sum_reward"] = em["sum_reward"] * (1 - prev_done) + state.reward
        em["length"] = em["length"] * (1 - prev_done) + 1
        for k, v in state.metrics.items():
            if k != "reward":
                em[k] = em[k] * (1 - prev_done) + v
        info = {**state.info, "steps": steps, "truncation": truncation.astype(done.dtype),
                "episode_done": done, "episode_metrics": em}
        return state.replace(done=done, info=info)


class AutoResetWrapper(Wrapper):
    """Brax's `AutoResetWrapper` (cached first state), restoring `info["world"]` too."""

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        state.info["first_pipeline_state"] = state.pipeline_state
        state.info["first_obs"] = state.obs
        state.info["first_world"] = state.info["world"]
        return state

    def step(self, state: State, action: jax.Array) -> State:
        if "steps" in state.info:
            state.info.update(steps=jnp.where(state.done > 0, 0, state.info["steps"]))
        state = self.env.step(state.replace(done=jnp.zeros_like(state.done)), action)
        done = state.done > 0

        def where(first, now):
            return jnp.where(done.reshape(done.shape + (1,) * (now.ndim - done.ndim)), first, now)

        data = jax.vmap(lambda f, d, dn: d.where(dn, f))(
            state.info["first_pipeline_state"], state.pipeline_state, done)
        obs = where(state.info["first_obs"], state.obs)
        world = jax.tree.map(where, state.info["first_world"], state.info["world"])
        return state.replace(pipeline_state=data, obs=obs,
                             info={**state.info, "world": world})
