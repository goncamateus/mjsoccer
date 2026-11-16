# mjsoccer

Gymnasium environments for a RoboCup SSL omni-directional soccer robot in MuJoCo.
Standalone: it depends on `gymnasium`, `mujoco` and `numpy`, and carries its own copy of
the MJCF and meshes in `models/`. Nothing imports the CAD pipeline that generated them.

## Install

```bash
uv sync
```

## Quickstart

```python
import gymnasium as gym
import mjsoccer                                  # registers the ids

env = gym.make("GoToBall-v0")                    # action (7,)
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

## Environments

| id | robots | obs | action | horizon | ends on |
|---|---|---|---|---|---|
| [`GoToBall-v0`](docs/environments/go_to_ball.md) | 1 blue + 6 yellow | 62 | 7 | 30 s | within 0.2 m of the ball (no facing check) |
| [`GoToBallShoot-v0`](docs/environments/go_to_ball_shoot.md) | 1 blue | 14 | 7 | 60 s | ball over the +x goal line; robot or ball leaves the attacking half; robot enters the penalty area |
| [`Dribbling-v0`](docs/environments/dribbling.md) | 1 blue + 4 yellow | 47 | 7 | 120 s | 7 checkpoints; last gap crossed backwards; robot leaves the course; a yellow moves > 0.05 m/s |
| [`StaticDefenders-v0`](docs/environments/static_defenders.md) | 1 blue + 6 yellow | 62 | 7 | 25 s | as GoToBallShoot |
| [`ContestedPossession-v0`](docs/environments/contested_possession.md) | 1 blue + 1 yellow | 22 | 7 | 30 s | as GoToBallShoot, or the yellow shoved > 0.1 m/s |
| [`PassEndurance-v0`](docs/environments/pass_endurance.md) | 2 blue (1 controlled) | 22 | 7 | 30 s | team-mate receives the pass; ball leaves the passer-receiver box or stalls |

Every environment exposes the same per-robot action — a target pose `(x, y, sin θ, cos θ)`
plus kick, chip, dribble — whether or not its reward uses it, so one action head transfers
across all six. The navigation controller (Planning-the-path-with-rl's `go_to_point_new`)
turns the pose into a chassis velocity every 25 ms control step.

## GPU backends

The same six tasks also run batched on the GPU, with the same ids, observations, actions
and rewards. The CPU backend stays the physics oracle, and parity tests hold both GPU
backends to it.

| backend | install | trains with | checkpoint | docs |
|---|---|---|---|---|
| MJX-Warp (`mjsoccer.mjx`, JAX) | `uv sync --group gpu` | Brax PPO | Orbax | [`docs/backends/mjx.md`](docs/backends/mjx.md) |
| mjlab (`mjsoccer.mjlab`, torch) | `uv sync --group mjlab` | RSL-RL PPO | `.pt` + ONNX | [`docs/backends/mjlab.md`](docs/backends/mjlab.md) |

```bash
uv run --group gpu   examples/ppo_mjx.py   --env GoToBall-v0
uv run --group mjlab examples/ppo_mjlab.py --env ContestedPossession-v0
```

Both need a CUDA GPU. [`docs/backends/`](docs/backends/index.md) compares them and covers
parity, per-world failures and domain randomization.

## Physics

The robot is an ideal kinematic chassis (ADR 0001) with the envelope derived from the
real robot's motor (Maxon EC 45 flat 50 W, direct drive: ±95 rad/s and ±0.078 N·m per
wheel) and its wheel friction: `v_max` 3.30 m/s, `a_max` 1.30 m/s², `ω_max` 37.1 rad/s,
`α_max` 55.0 rad/s², isotropic. Deliberately not modelled: slip, traction budget, weight
transfer, coupling between axes. The chassis slides on the carpet and cannot tip. The control step is a fixed 25 ms
(40 Hz): 24 physics steps of 1/960 s (a firing step is simulated more finely). Every contact and body number has a source or is
recorded as a deliberate approximation in [`docs/physics.md`](docs/physics.md).

## Run it

```bash
uv run python examples/random.py --env GoToBall-v0
uv run python examples/random.py --env GoToBall-v0 --view      # MuJoCo Studio viewer
uv run python -m pytest tests -q
uv run python -m pytest tests -m validation -q                  # physics acceptance criteria, ~1 min
uv run python scripts/validation_sweep.py --out runs/validation --seeds 0-4  # their distributions, as CSV
```

Train on the CPU (stable-baselines3 / CleanRL):

```bash
uv sync --group rl
uv run --group rl python examples/ppo_agent.py   --env GoToBall-v0          # -> runs/ppo_GoToBall-v0.zip
uv run --group rl python examples/sac_agent.py   --env GoToBall-v0          # -> runs/sac_GoToBall-v0.zip
uv run --group rl python examples/sac_cleanrl.py --env GoToBall-v0          # -> runs/sac_cleanrl_GoToBall-v0.pt
uv run --group rl python examples/sac_agent.py   --env GoToBall-v0 --play runs/sac_GoToBall-v0.zip
examples/sweep.sh sac all 4                                                 # every env, 4 at a time -> runs/sweep/sac_YYYYmmdd-HHMMSS/
uv run --group rl python examples/visualize_sweep.py runs/sweep/sac_20261018-231040  # a sweep folder: one GIF per policy
```

It runs from a checkout: the MJCF and meshes live in `models/`, beside the package, not
inside it. `mjsoccer.scene` resolves them from the source tree; set `MJSOCCER_MODELS` to
point at another build.

## Documentation

- [`docs/index.md`](docs/index.md): user reference. Install, action and observation
  contracts, timing, field constants, rendering.
- [`docs/physics.md`](docs/physics.md): the chassis and its envelope, contact parameters
  and their sources, how possession is measured, the multi-robot `MjSpec` attach.
- [`docs/validation.md`](docs/validation.md): the acceptance criteria and GPU × CPU parity.
- [`docs/backends/`](docs/backends/index.md): the GPU backends, MJX-Warp and mjlab.
- [`docs/environments/`](docs/environments/): one page per task.
