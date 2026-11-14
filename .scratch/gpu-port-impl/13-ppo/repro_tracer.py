"""Ticket 13: where does repeat_action > 1 leak a tracer under Brax's wrappers?

    XLA_PYTHON_CLIENT_PREALLOCATE=false python repro_tracer.py vmap|scan_vmap|wrap
"""
import sys

import jax
import jax.numpy as jnp
import numpy as np
from brax.envs.wrappers import training

import mjsoccer.mjx

mode = sys.argv[1]
N = 4
env = mjsoccer.mjx.make("ContestedPossession-v0", nworld=N, repeat_action=2)
keys = jax.random.split(jax.random.PRNGKey(0), N)
a = jnp.zeros((N, env.action_size))
if mode == "vmap":
    s = jax.jit(jax.vmap(env.reset))(keys)
    s = jax.jit(jax.vmap(env.step))(s, a)
elif mode == "scan_vmap":
    s = jax.jit(jax.vmap(env.reset))(keys)
    f = jax.jit(lambda s: jax.lax.scan(lambda s, _: (jax.vmap(env.step)(s, a), None), s, (), 1)[0])
    s = f(s)
elif mode == "vmap_info":  # a second Data riding along in info, as AutoResetWrapper's
    s = jax.jit(jax.vmap(env.reset))(keys)
    s.info["first"] = s.pipeline_state
    s = jax.jit(jax.vmap(env.step))(s, a)
elif mode == "vmap_scan":
    s = jax.jit(jax.vmap(env.reset))(keys)
    f = jax.jit(jax.vmap(lambda s, a: jax.lax.scan(lambda s, _: (env.step(s, a), None), s, (), 1)[0]))
    s = f(s, a)
elif mode == "wrap":
    w = training.wrap(env, episode_length=1000)
    s = jax.jit(w.reset)(keys)
    s = jax.jit(w.step)(s, a)
print(mode, "OK", np.asarray(s.done))
