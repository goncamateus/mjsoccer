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

env = gym.make("GoToBall-v0")                    # default tier 1: roller physics, action (7,), obs (14,)
env = gym.make("GoToBallShoot-v0", tier=0)       # tier 0: fast kinematic chassis, action (6,)
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

## Environments

| id | robots | obs | action (tier 1, default / tier 0) | horizon | ends on |
|---|---|---|---|---|---|
| [`GoToBall-v0`](docs/environments/go_to_ball.md) | 1 blue | 14 | 7 / 6 | 20 s | at the ball, pointing at it |
| [`GoToBallShoot-v0`](docs/environments/go_to_ball_shoot.md) | 1 blue | 14 | 7 / 6 | 30 s | ball in the +x goal |
| [`Dribbling-v0`](docs/environments/dribbling.md) | 1 blue | 16 | 7 / 6 | 40 s | 4 waypoints carried, or ball lost for 2 s |
| [`StaticDefenders-v0`](docs/environments/static_defenders.md) | 1 blue + 6 yellow | 62 | 7 / 6 | 30 s | ball in the +x goal |
| [`ContestedPossession-v0`](docs/environments/contested_possession.md) | 1 blue + 1 yellow | 22 | 7 / 6 | 20 s | blue holds the ball 0.5 s |
| [`PassEndurance-v0`](docs/environments/pass_endurance.md) | 2 blue (1 controlled) | 22 | 7 / 6 | 20 s | team-mate receives the pass |

Every environment exposes the same per-robot action — drive, kick, chip, dribble — whether
or not its reward uses it, so one action head transfers across all six.

## Physics

Tier 1 drives four wheels through 80 rollers that really slip, at the envelope of the
robot's motor (Maxon EC 45 flat 50 W, direct drive): ±95 rad/s and ±0.078 N·m per wheel,
which derives to 3.30 m/s, 2.30-2.45 m/s² and 35.9 rad/s at the chassis. Measured, it
reaches that acceleration but not yet that speed (~2.1 m/s forward, ~0.4 lateral, ~30 rad/s
of spin): the rollers lose floor contact while driving, an open issue. Tier 0 is the same
envelope on an ideal kinematic chassis — the same four per-axis limits, `v_max` 3.30 m/s,
`a_max` 2.30 m/s², `ω_max` 35.9 rad/s, `α_max` 55.1 rad/s², isotropic — and deliberately
not the same mechanism: no slip, no traction budget, no weight transfer, no coupling between
axes. A robot that tips past 60° ends the episode at `-100`; under random action on tier 1
that has not happened once in 101 minutes of episode. Physics steps at 1/960 s, so the
default 60 Hz control rate is exact. Every contact and body number has a source or is
recorded as a deliberate approximation in [`docs/index.md`](docs/index.md).

## Run it

```bash
uv run python examples/random.py --env GoToBall-v0
uv run python examples/random.py --env GoToBall-v0 --view      # MuJoCo viewer
uv run python -m pytest tests -q
uv run python -m pytest tests -m validation -q                  # physics acceptance criteria, ~2 min
uv run python scripts/validation_sweep.py --out runs/validation  # their distributions, as CSV
```

## Documentation

[`docs/index.md`](docs/index.md) is the reference: the action and observation contracts,
the two physics tiers and their envelope, the wheel matrix, turnover, the contact
parameters and their sources, how possession is measured, the multi-robot `MjSpec` attach, the
field constants and rendering. Each environment has its own page under
[`docs/environments/`](docs/environments/).
