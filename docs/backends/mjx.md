# MJX-Warp backend

The six tasks in functional JAX, as `brax.envs.Env`, stepped by MuJoCo Warp through the
`mujoco-mjx` API on the exact base model (`mjsoccer.mjx`). The CPU backend stays the
oracle. [Backends](index.md) covers what all backends share: parity, per-world failures,
DR, noise and latency.

## Requirements and install

```bash
uv sync --group gpu
```

The group pins `mujoco-mjx[warp]`, `jax[cuda12]` and Brax from git (`pyproject.toml`
explains why). Stepping needs a CUDA GPU. `import mjsoccer` never imports this package.

## API

```python
import jax
import mjsoccer.mjx

n = 1024
env = mjsoccer.mjx.make("ContestedPossession-v0", nworld=n)
state = jax.jit(jax.vmap(env.reset))(jax.random.split(jax.random.key(0), n))
state = jax.jit(jax.vmap(env.step))(state, actions)          # actions (n, 7)
state.obs, state.reward, state.done, state.info["truncation"]
```

`mjsoccer.mjx.make(env_id, *, repeat_action=1, graph_mode="WARP", randomize=None, **kw)`
takes the same ids as `gymnasium.make` (`ENV_IDS`). `kw` carries `nworld`,
`nacon_per_world` and `action_latency`.

- `nworld` is the batch you will vmap over. It sizes Warp's contact buffer, which the whole
  batch shares. Pass the real batch size, because an undersized buffer overflows.
  `nacon_per_world` overrides each world's share.
- One `step` is one decision of `repeat_action` control steps. The env owns the episode:
  `done` includes the time-out, and the time-out is flagged in `info["truncation"]`.
- Per-world state lives in `state.info["world"]`: step count, kick cooldown, the task's own
  entries, the action queue and the world's RNG key.
- `graph_mode` picks Warp's CUDA-graph mode. `WARP` is the fastest. `WARP_STAGED` is the
  conservative fallback, at about half the speed.

**For training, wrap with `mjsoccer.mjx.wrappers.wrap`, not Brax's `training.wrap`.** Brax's
auto-reset restores only `pipeline_state` and `obs`. A reset world would then keep its
predecessor's step count and truncate early. Brax's episode wrapper also turns the env's
own time-out into a terminal. `wrap` fixes both, and it is the `wrap_env_fn` signature
`ppo.train` calls:

```python
from mjsoccer.mjx import wrappers

train_env = wrappers.wrap(env, episode_length=wrappers.episode_length(env), obs_noise=0.0)
```

Keep Brax's `action_repeat` at 1, because the repeat belongs to the env (`repeat_action`).

## Training

`examples/ppo_mjx.py` runs vanilla Brax `ppo.train` on any task, on one GPU:

```bash
systemd-run --user --scope -p MemoryMax=10G \
    uv run --group gpu examples/ppo_mjx.py --env GoToBall-v0
```

- **Outputs** go in `runs/ppo_mjx_<env>/` (`--save`): one Orbax checkpoint per epoch in
  Brax's layout, plus the flags in `args.json`.
- **Budget**: `--epochs` (30) with an eval after each. Without `--timesteps`, each stretch
  between two resets gets the PPO iterations one episode takes, so resets do not cut
  episodes short.
- **`--num-envs`** defaults by robot count (`NUM_ENVS` in the script): the largest batch
  that trained 30 epochs with >= 2 GB of the GPU free.
- **PPO flags** keep Brax's names (`--learning-rate`, `--entropy-cost`, `--unroll-length`,
  `--num-minibatches`, ...). Their defaults are a baseline, not a tuning. The Brax variants
  (`--learning-rate-schedule ADAPTIVE_KL`, `--distributional-critic`) are off by default.
- **Log**, per epoch: decisions so far, decisions/s, eval return, the eval episodes'
  failure rates, and GPU memory in use.
- **`--randomize module:sampler`**, `--obs-noise` and `--action-latency` are described in
  [Backends](index.md#domain-randomization-observation-noise-action-latency).

Run long trainings under a RAM cap (`systemd-run ... MemoryMax=10G`), one process per run.

## Eval and play

```bash
uv run --group gpu examples/ppo_mjx.py --eval runs/ppo_mjx_GoToBall-v0   # on MJX, once per world
uv run --group gpu examples/ppo_mjx.py --play runs/ppo_mjx_GoToBall-v0   # on the CPU backend
```

`--eval` runs the latest checkpoint on MJX for one episode per world. It aborts if any
world diverged, overflowed or reset overlapping. `--play` loads the checkpoint with JAX on
the CPU and drives `gymnasium.make` in the Studio viewer (`--render none`, `--episodes N`
for headless). Both run nominal: no DR, no noise, no latency.

## Domain randomization

```python
env = mjsoccer.mjx.make(env_id, randomize=sampler, nworld=n)
ppo.train(env, wrap_env_fn=wrappers.wrap, randomization_fn=env.randomization_fn, ...)
```

`env.randomization_fn` is Brax's `randomization_fn(sys, rng) -> (sys, in_axes)`. Only the
fields that differ in some slot get `in_axes` 0, so DR costs no throughput. Observation
noise is the `wrap(obs_noise=)` wrapper, and action latency is
`make(action_latency=)`.

## Limits and pitfalls

- **No asymmetric critic.** Brax's critic reads a privileged observation key, and these
  envs return one flat observation vector.
- **Captured graphs pile up** in `WARP` mode. The training script drops Warp's graph cache
  at the end of each epoch. Do the same in your own loop, or memory grows.
- **The contact buffer is batch-wide.** Size `nworld` to the real batch. An overflowing
  world ends, zeroed, and the counters report it, but too many of them is a sign the buffer
  is too small.
- **Not bitwise reproducible**: atomics in the Warp solver give ~1e-6 at reset and ~1e-4
  after a few decisions between identical runs.
- MJX's pure-JAX engine (MJX-JAX) is not supported ([ADR 0002](../adr/0002-mjx-warp-only.md)).

## Tests

| test | needs | checks |
|---|---|---|
| `tests/test_mjx_task_logic.py` | JAX-CPU | task logic = CPU, state by state |
| `tests/test_mjx_warp_smoke.py` | GPU | every task makes, resets, steps; per-world failures end only their world |
| `tests/test_mjx_randomize.py` | GPU | DR variants, `mj_setConst`, refused fields |
| `tests/test_mjx_noise_latency.py` | GPU | noise and latency; off changes nothing |
| `tests/test_ppo_mjx.py` | JAX-CPU | script flags; `--play` from a checkpoint |
| `tests/test_validation.py -m validation` | GPU, ~19 min | acceptance criteria and MJX × CPU parity |

```bash
systemd-run --user --scope -p MemoryMax=10G uv run --group gpu python -m pytest tests -q
systemd-run --user --scope -p MemoryMax=10G uv run --group gpu python -m pytest tests -m validation -q
uv run --group gpu python scripts/validation_sweep.py --backend mjx --seeds 0-49 --out runs/validation-mjx
```

GPU tests are skipped without CUDA.
