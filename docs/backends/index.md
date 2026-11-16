# Backends

The six tasks run on three backends. All three share the env ids (`mjsoccer.ENV_IDS`), the
horizons, the observation and action layouts and the reward. They differ in what steps the
physics and what trains on top of it.

| | CPU | [MJX-Warp](mjx.md) | [mjlab](mjlab.md) |
|---|---|---|---|
| package | `mjsoccer` (Gymnasium) | `mjsoccer.mjx` | `mjsoccer.mjlab` |
| install | `uv sync` | `uv sync --group gpu` | `uv sync --group mjlab` |
| array library | numpy | JAX | torch |
| physics | MuJoCo C | MuJoCo Warp, through `mujoco-mjx` | MuJoCo Warp, through mjlab |
| env type | `gymnasium.Env`, one world | `brax.envs.Env`, vmapped over `nworld` | mjlab `ManagerBasedRlEnv`, `nworld` batched |
| example trainer | SB3 / CleanRL (`examples/ppo_agent.py`, `sac_*.py`) | Brax PPO (`examples/ppo_mjx.py`) | RSL-RL PPO (`examples/ppo_mjlab.py`) |
| checkpoint | SB3 zip | Orbax (Brax's layout) | RSL-RL `.pt` + ONNX policy |
| needs | nothing | a CUDA GPU | a CUDA GPU, driver >= 580 (torch cu130) |

`import mjsoccer` imports neither JAX nor torch. Each GPU backend sits behind its own uv
group, and both groups install together in one environment.

## Choosing a GPU backend

Choose by ecosystem. MJX-Warp gives you JAX and Brax. mjlab gives you torch, RSL-RL and
ONNX export. Both run the same exact base model on the same engine, and their throughput
matches (see [Measured](#measured)). The one RSL-RL × Brax comparison so far used one task
and one seed, which is not enough to rank them.

## The CPU backend is the oracle

The GPU backends are reimplementations of the CPU task logic (`mjsoccer.tasks`), ported
line by line. They are not a separate design. Every constant and task config is read off
the CPU side. Two layers hold them to the CPU:

- **Task-logic parity**, state by state. CPU rollouts are snapshotted, and each GPU backend
  runs its controller, observation, reward, ending and per-world state on the CPU's own
  physics. All of it must match to float32 tolerance. These tests run without a GPU, on
  JAX-CPU (`tests/test_mjx_task_logic.py`) and torch-CPU (`tests/test_mjlab_task_logic.py`).
- **Physics acceptance and parity** (`-m validation`, `tests/test_validation.py`). Every
  acceptance criterion in [`docs/index.md`](../index.md) runs on the three backends with
  the same threshold. On top of that, each GPU backend runs the GPU × CPU parity: envelope,
  kick and COR within 2 %, arrival time within 5 %, and the CPU median return and
  possession rate inside the GPU min-max over seeds 0-49.

The GPU backends run in float32, and Warp's solver uses atomics, so two identical rollouts
are not bitwise equal. Expect agreement to a tolerance, not to the bit.

When the CPU and a GPU backend disagree, the CPU is right. The policies trained on the GPU
are played back on the CPU (`--play`) for exactly this reason.

## Batching

One `step` is one decision of `repeat_action` control steps for every world at once. The
kick's firing step needs finer physics than the others. A world that fires runs the CPU's
firing schedule, and every other world sits frozen at `timestep = 0` for those iterations,
so the whole batch runs one loop of the same length. An idle world stays bit-identical
through that loop.

A world that ends restarts from a cached first state for its slot. On MJX this is Brax's
per-slot cache, re-drawn `--num-resets-per-eval` times an epoch. On mjlab it is a pool,
re-drawn every epoch by `refill_pool()`.

## Per-world failures

Each world fails alone. Nothing aborts inside the batch. When a world's `qpos`/`qvel`
stops being finite (**diverged**), or the world sets a bit of Warp's overflow mask because
the contact or constraint buffer is too small (**overflow**), it ends on that step as a
terminal with reward and obs 0, and the auto-reset brings it back. A reset whose placement
found no free spot is counted as **reset_overlap**. The three counters are env metrics:

- MJX: `State.metrics`; `mjsoccer.mjx.check_failures` turns them into a host-side abort.
- mjlab: episode sums in `extras["log"]`.

Both training scripts log the failure rate every epoch, and `--eval` aborts when any world
fails.

## Domain randomization, observation noise, action latency

All three are off by default, and when off they change nothing. They mean the same thing on
both GPU backends:

- **DR** is `randomize=sampler`, where `sampler(m: mujoco.MjModel, rng: np.random.Generator)`
  mutates the CPU model in place and returns nothing. It runs once per world slot for the
  whole run. `mj_setConst` recomputes the derived fields, and only what differs from the
  nominal model is batched (`mjsoccer.randomize.variants`, shared by both backends). A
  slot's variant depends on the seed and its index alone. Real-valued model fields and the
  float `opt` fields can vary. `opt.timestep` and structural fields (meshes, `contype`,
  ids) are refused. The ball's contacts are explicit `<pair>`s, so to vary them you write
  `pair_friction`/`pair_solref`, not the geoms' `geom_*`.
- **Observation noise** `obs_noise = σ` adds N(0, σ²) to every returned observation entry,
  reset observations included, on the already normalised obs. The physics never sees it.
- **Action latency** `action_latency = N` delays each action by N control steps, through a
  per-world queue that resets with the world.

```python
def sampler(m, rng):
    b = m.body("b0_robot").id                     # blue robot 0's chassis (prefix b<i>_ / y<i>_)
    k = rng.uniform(0.9, 1.1)
    m.body_mass[b] *= k
    m.body_inertia[b] *= k
```

The training scripts take the sampler as `--randomize module:sampler`, plus `--obs-noise`
and `--action-latency`. The per-epoch eval runs on the nominal model unless you pass
`--eval-randomize`. `--eval` and `--play` always run nominal, with no noise and no latency.

## Measured

These numbers are a report, not a guarantee. They were measured in November 2026 on an
RTX 3060 12 GB with mujoco 4.0.1, one process per run under `MemoryMax=10G`. The defaults
the scripts actually use live in their `NUM_ENVS`.

| | MJX-Warp | mjlab |
|---|---:|---:|
| ContestedPossession, 4 096 worlds, decisions/s ([prototype](../../.scratch/mjlab-port/research/02-prototype-results.md)) | 16 550 | 16 588 |
| ContestedPossession, 8 192 worlds, PPO training decisions/s ([RSL-RL × Brax](../../.scratch/mjlab-port/research/07-rsl-rl-vs-brax.md)) | 16 795 | 16 226 |
| largest 2-robot batch in the step bench (~40 s, not a training run) | 4 096 (OOM at 8 192, which still trains) | 16 384 (10.9 GB) |
| GoToBall (7 robots), PPO training decisions/s ([num-envs](../../.scratch/mjlab-port/research/06-num-envs.md)) | | ~2 050, flat from 4 096 to 32 768 worlds |

The research notes these numbers come from:
[MJX benchmark](../../.scratch/gpu-port/research/05-mjx-benchmark.md),
[the kick on Warp](../../.scratch/gpu-port/research/15-kick-on-warp.md),
[streams and memory on mjlab](../../.scratch/mjlab-port/research/04-streams-and-memory.md).
Why the backends are what they are: ADRs [0002](../adr/0002-mjx-warp-only.md) and
[0003](../adr/0003-mjx-warp-and-mjlab.md).
