# Run the GPU port on MJX-Warp and mjlab; the CPU backend stays the oracle

Supersedes [0002](0002-mjx-warp-only.md) ("MJX-Warp only"). Decisions 1-16 in
`.scratch/mjlab-port/decisions.md`; spec in `.scratch/mjlab-port/spec.md`.

ADR 0002 made MJX-Warp, driven through the `mujoco-mjx` JAX API and trained with Brax PPO,
the only GPU backend. That left out the torch ecosystem mjlab ships ready-made: per-event
domain randomization on the GPU, native observation noise and delay, the asymmetric critic,
ONNX policy export and the RSL-RL runner. mjlab runs on MuJoCo Warp too, so it can step the
same exact base model. The prototype (branch `prototype/mjlab`, `45b25d5`, not merged;
`.scratch/mjlab-port/research/02-prototype-results.md`) ran ContestedPossession with the
kick on mjlab on an RTX 3060, both backends on mujoco 4.0.1. The results:

- **Throughput matches.** 16 588 decisions/s against MJX-Warp's 16 550 at 4 096 worlds.
- **Batches are larger.** 16 384 worlds fit (10.9 GB on the device). The MJX bench goes out
  of memory at 8 192.
- **Manager overhead is small.** 6-7 % at 4-8 k worlds, because the 48 physics iterations
  of the firing step dominate.
- **The kick schedule works.** An idle world stays bit-identical through the iterations at
  `timestep = 0`.
- **Failed worlds can end.** `reset_data` clears `Data.overflow` for the reset worlds only,
  so a termination that reads overflow works.
- **Both stacks install together.** torch (cu130) and JAX resolve in one `uv lock`, with no
  `conflicts`.

So mjlab becomes the **third backend**: the CPU backend, MJX-Warp (with Brax) and mjlab
(with RSL-RL) live side by side, with one catalogue (`ENV_IDS`) and one env contract.

## Consequences

- MJX-Warp is not deleted or frozen. "MJX-Warp only" no longer holds. ADR 0002's case
  against MJX-JAX and against a model variant still holds for mjlab: mjlab runs the exact
  base model, so the robots get no `Entity` and no mocap wrapper.
- **A third implementation of the task logic.** The six tasks are rewritten in torch as
  pure functions over tensors of the physical state and per-world state
  (`mjsoccer.mjlab`). They are a line-by-line port of `mjsoccer.mjx`, with every constant
  and config read off the CPU side. A torch-CPU task-logic parity test holds them to the
  CPU state by state, as `tests/test_mjx_task_logic.py` does for JAX. `import mjsoccer`
  imports neither torch nor mjlab.
- **Same contract, same parity.** mjlab has to pass every acceptance criterion with the
  CPU's thresholds, and the same deterministic and seeded parity as MJX-Warp
  (`validation_sweep.py --backend mjlab`). The CPU backend stays the physics oracle.
- The firing step keeps MJX-Warp's schedule: `decimation = 48 × repeat_action`, with a
  per-world `opt.timestep` written every substep. Horizon, reward and time-out are the
  task's, not mjlab's.
- The first state of each world comes from a **pool of first states** instead of
  `seat_ball` at reset (mjlab steps every world at once). The pool is rebuilt each epoch.
- The `main` branch moves to mujoco 4.0.1 (MJX-Warp included), and CPU↔MJX parity is rerun
  there. The optional `mjlab` group (mjlab pinned on git, torch ≥ 2.14 cu130, onnxruntime)
  shares the lock with `gpu`. The universal lock also moves the `rl` group's torch.
- Open risks: mjlab #1263 (ITERATIONS overflow) on the 7-robot scenes; mjlab has no release
  on mujoco 4.x (git pin); observation freshness (mjlab calls `forward` before observing).
