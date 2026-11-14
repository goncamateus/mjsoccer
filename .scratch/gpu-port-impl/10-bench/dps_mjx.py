"""MJX-Warp decisions/s, end to end (ticket 10): the twin of ../02-bench/dps.py on the GPU.

    XLA_PYTHON_CLIENT_PREALLOCATE=false systemd-run --user --scope -p MemoryMax=10G \
        uv run python .scratch/gpu-port-impl/10-bench/dps_mjx.py <env_id> <nworld> [...]

The whole env -- task, navigation controller, kicker, possession, the 48-iteration physics
schedule -- under Brax's training wrappers (`EpisodeWrapper` + `AutoResetWrapper`, the
cached first state per world, as the spec's auto-reset), 40 Hz, repeat_action 1, uniform
random actions drawn on the device (so kick/chip fire whenever re-armed, as on the CPU
bench). One call = CHUNK decisions in a `fori_loop`; best of 3 timed runs of SECONDS after a
warm-up. Also the batch's worst Warp numbers over the run: overflow bits OR-ed, peak
`ncollision` (broadphase candidates of the whole batch, what fills the `naconmax` pool;
`nacon`, the contacts kept, is far lower) and peak `nefc` per world (against `njmax`). Prints one `RESULT {json}` line per nworld.

CHASE=1 swaps the random target pose for `tasks.goto_ball_policy` in jnp (kick, chip and
dribbler stay random): robot 0 drives at the ball, into the yellow robot in
ContestedPossession and through the parked ones in GoToBall, so episodes end and contacts
pile up -- the stress run for the contact buffer, not the throughput number.
"""

import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from brax.envs.wrappers import training

import mjsoccer.mjx
import mjsoccer.scene
from mjsoccer.mjx.base import overflow_bits

SECONDS = 10.0
CHUNK = 20
CHASE = os.environ.get("CHASE") == "1"


def chase(obs, a):
    """`tasks.goto_ball_policy` on the batch: target the ball, face it."""
    ball, pos = obs[:, 0:2], obs[:, 6:8]
    hx, hy = (ball - pos)[:, 0] * mjsoccer.scene.FIELD_HALF_X, (ball - pos)[:, 1] * mjsoccer.scene.FIELD_HALF_Y
    h = jnp.arctan2(hy, hx)
    pose = jnp.stack([ball[:, 0], ball[:, 1], jnp.sin(h), jnp.cos(h)], 1).clip(-1, 1)
    return a.at[:, :4].set(pose)


def bench(env_id: str, nworld: int) -> dict:
    env = mjsoccer.mjx.make(env_id, nworld=nworld)
    wenv = training.wrap(env, episode_length=env.max_steps)

    def stats(s):
        i = s.pipeline_state._impl
        return (overflow_bits(s.pipeline_state).astype(jnp.int32), jnp.max(i.ncollision),
                jnp.max(i.nefc), s.done.sum(), s.metrics["reset_overlap"].sum())

    @jax.jit
    def run(s, key):
        def body(_, c):
            s, key, bits, nacon, nefc, ends = c
            key, k = jax.random.split(key)
            a = jax.random.uniform(k, (nworld, env.action_size), minval=-1.0, maxval=1.0)
            if CHASE:
                a = chase(s.obs, a)
            s = wenv.step(s, a)
            b, na, ne, done, _ = stats(s)
            return (s, key, bits | b, jnp.maximum(nacon, na), jnp.maximum(nefc, ne),
                    ends + done)
        z = jnp.zeros((), jnp.int32)
        init = (s, key, jnp.zeros(nworld, jnp.int32), z, z, jnp.zeros(()))
        return jax.lax.fori_loop(0, CHUNK, body, init)

    t0 = time.perf_counter()
    s = jax.jit(wenv.reset)(jax.random.split(jax.random.PRNGKey(0), nworld))
    jax.block_until_ready(s.obs)
    reset_s = time.perf_counter() - t0
    overlap = float(s.metrics["reset_overlap"].sum())
    key = jax.random.PRNGKey(1)
    t0 = time.perf_counter()
    s, key, bits, nacon, nefc, ends = run(s, key)
    jax.block_until_ready(s.obs)
    compile_s = time.perf_counter() - t0
    bits, nacon, nefc, ends = np.array(bits), int(nacon), int(nefc), float(ends)
    best, decisions = 0.0, CHUNK
    for _ in range(3):
        n, t0 = 0, time.perf_counter()
        while time.perf_counter() - t0 < SECONDS:
            s, key, b, na, ne, e = run(s, key)
            jax.block_until_ready(s.obs)
            n += 1
            bits |= np.asarray(b)
            nacon, nefc, ends = max(nacon, int(na)), max(nefc, int(ne)), ends + float(e)
        decisions += n * CHUNK
        best = max(best, n * CHUNK * nworld / (time.perf_counter() - t0))
    return dict(env=env_id, chase=CHASE, nworld=nworld, dps=round(best), decisions_per_world=decisions,
                episode_ends=ends, reset_overlap=overlap, reset_s=round(reset_s, 1),
                compile_s=round(compile_s, 1), overflow_worlds=int((bits != 0).sum()),
                overflow_or=int(np.bitwise_or.reduce(bits)), ncollision_peak=nacon, ncollision_peak_per_world=round(nacon / nworld, 1),
                naconmax=nworld * env.nacon_per_world, nefc_peak=nefc, njmax=env.njmax)


if __name__ == "__main__":
    env_id = sys.argv[1]
    for nworld in map(int, sys.argv[2:]):
        print("RESULT " + json.dumps(bench(env_id, nworld)), flush=True)
