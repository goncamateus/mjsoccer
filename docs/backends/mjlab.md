# mjlab backend

The six tasks in torch on mjlab's `ManagerBasedRlEnv`, stepped by MuJoCo Warp on the exact
base model and trained with RSL-RL (`mjsoccer.mjlab`, [ADR 0003](../adr/0003-mjx-warp-and-mjlab.md)).
It is a line-by-line port of the [MJX-Warp backend](mjx.md), which it sits beside and does
not replace. The CPU backend stays the oracle. [Backends](index.md) covers what all
backends share: parity, per-world failures, DR, noise and latency.

## Requirements and install

```bash
uv sync --group mjlab
```

The group pins mjlab from git and pulls torch from the cu130 index, which needs NVIDIA
driver >= 580. Making an env needs a CUDA GPU. The `mjlab` and `gpu` groups install
together. `import mjsoccer` never imports this package, and importing `mjsoccer.mjlab`
imports neither torch nor mjlab until `make` is called.

## API

```python
import torch
import mjsoccer.mjlab

env = mjsoccer.mjlab.make("ContestedPossession-v0", nworld=1024, seed=0)
obs, extras = env.reset()                                    # obs["actor"]: (1024, obs_dim)
obs, reward, terminated, truncated, extras = env.step(actions)   # actions (1024, 7), CUDA
```

`mjsoccer.mjlab.make(env_id, *, nworld, repeat_action=1, randomize=None, device="cuda:0",
seed=None, nconmax=None, njmax=None, naccdmax=None, action_latency=0, obs_noise=0.0)`
takes the same ids, horizons, spaces and obs/action layout as `gymnasium.make`
(`ENV_IDS`). It returns a `SoccerEnv`, which is mjlab's `ManagerBasedRlEnv`. Call `reset()`
before stepping.

- One `step` is one decision of `repeat_action` control steps. The horizon, the reward
  (summed, unscaled) and the time-out come from the task, not from mjlab's `step_dt`.
- `nconmax`/`njmax` override Warp's per-world contact and constraint buffers. The default
  is the MJX sizing. `naccdmax` overrides the batch-wide EPA scratch. An undersized buffer
  overflows, and the overflowing world ends, zeroed.
- First states come from a **pool**, placed and seated in one batched loop at startup. A
  world that ends copies its slot back. Call `env.refill_pool()` and then `env.reset()` to
  draw the pool anew; the training script does this every epoch.
- `obs_noise` and `action_latency` are `make` arguments here. On MJX, the noise is a
  wrapper instead.
- The failure counters (`diverged`, `overflow`, `reset_overlap`) are episode sums in
  `extras["log"]`.

### Streams

`step`, `reset` and `refill_pool` run torch on Warp's stream and synchronise with the
caller's stream at entry and exit. Calling them from torch's default stream or from a side
stream is safe. Do not touch `env.sim.data` from a side stream outside those calls, because
it races Warp's graph.

## Training

`examples/ppo_mjlab.py` runs RSL-RL PPO through mjlab's `MjlabOnPolicyRunner`, on one GPU.
It is the sibling of `ppo_mjx.py`, with the same flags and the same PPO defaults:

```bash
systemd-run --user --scope -p MemoryMax=10G \
    uv run --group mjlab examples/ppo_mjlab.py --env ContestedPossession-v0
```

- **Outputs** go in `runs/ppo_mjlab_<env>/` (`--save`): RSL-RL's `model_<iteration>.pt` at
  the end of every epoch, the flags in `args.json`, and the policy in ONNX
  (`<run dir name>.onnx`, deterministic: normalizer + MLP to the action mean) at every
  checkpoint. A failed export aborts the run.
- **Epoch**: `refill_pool()`, then the PPO iterations, then an eval on `--num-eval-envs`
  worlds. Without `--timesteps`, an epoch gets the iterations one episode takes.
- **`--num-envs`** defaults by robot count (`NUM_ENVS` in the script), measured in long
  training runs. A robot count without a measured default takes the next larger measured
  count's.
- **PPO flags** keep `ppo_mjx`'s names, mapped onto RSL-RL (`--unroll-length` is
  `num_steps_per_env`, and so on). Not carried over: `--batch-size` (an RSL-RL iteration is
  one unroll of every world), `--reward-scaling`, the tanh-squashed policy (RSL-RL clips its
  Gaussian to the action box) and the Brax-only variants. Same flags do not mean same
  optimisation: RSL-RL takes 4× more gradient steps per decision, on minibatches 4× smaller.
- **Log**, per epoch: decisions so far, decisions/s, eval return, the eval failure rates,
  the training world-decisions with each failure, and the device memory in use (the whole
  GPU's).
- **`--randomize module:sampler`**, `--obs-noise` and `--action-latency` are described in
  [Backends](index.md#domain-randomization-observation-noise-action-latency).

## Eval and play

```bash
uv run --group mjlab examples/ppo_mjlab.py --eval runs/ppo_mjlab_ContestedPossession-v0   # on mjlab
uv run --group mjlab examples/ppo_mjlab.py --play runs/ppo_mjlab_ContestedPossession-v0   # CPU backend
```

Both run the ONNX policy. `--eval` runs it on mjlab for one episode per world. It aborts if
any world diverged, overflowed or reset overlapping, and names Warp's overflow bits by
type. `--play` runs it with onnxruntime on the CPU backend in the Studio viewer
(`--render none`, `--episodes N` for headless). Both run nominal. Because the policy is
ONNX, it also runs outside this repo with no torch.

## Domain randomization

```python
env = mjsoccer.mjlab.make(env_id, nworld=n, seed=0, randomize=sampler)
```

The sampler is the same function MJX takes. A startup event runs it on the CPU model once
per world (seeded by `seed` and the world's index) and writes what changed into Warp's
model, `<pair>` fields included. mjlab's own `set_const` DR events are not used, so a
sampler means the same thing on both GPU backends.

## Limits and pitfalls

- **Symmetric critic.** The critic reads the actor's observation (`obs_groups`). mjlab and
  RSL-RL support an asymmetric critic, but the envs expose no privileged observation group.
- **mjlab's `step_dt` is wrong for these envs and unused.** Read time from the task
  (`max_steps`, `repeat_action`), not from mjlab.
- **Rare policy-driven overflows** with 2 robots, about 4e-7 per world-decision. They end
  only their world and show in the log.
- **Not bitwise reproducible**, as on MJX (atomics in the Warp solver).
- The robots are not mjlab `Entity`s and get no mocap wrapper. The scene is the CPU's base
  model attached whole, with the CPU's names, plus one empty entity the action term needs.
  mjlab features built on a robot `Entity` do not apply. For example, mjlab's own action
  delay is the actuator's, in physics steps. This backend uses its own queue, in control
  steps.

## Tests

| test | needs | checks |
|---|---|---|
| `tests/test_mjlab_task_logic.py` | torch-CPU | task logic = CPU, state by state |
| `tests/test_mjlab_smoke.py` | GPU | every task makes, resets, steps; idle worlds frozen; per-world failures |
| `tests/test_mjlab_randomize.py` | GPU | DR variants match MJX's sampler semantics |
| `tests/test_mjlab_noise_latency.py` | GPU | noise and latency; off changes nothing |
| `tests/test_ppo_mjlab.py` | GPU | flags, RSL-RL config, train → checkpoint + ONNX → `--eval` → `--play` |
| `tests/test_validation.py -m validation` | GPU | acceptance criteria and mjlab × CPU parity |

```bash
systemd-run --user --scope -p MemoryMax=12G \
    uv run --group gpu --group mjlab python -m pytest tests -m validation -q   # CPU + MJX + mjlab
uv run --group mjlab python scripts/validation_sweep.py --backend mjlab --seeds 0-49 --out runs/validation-mjlab
```

GPU tests are skipped without CUDA or without the group.
