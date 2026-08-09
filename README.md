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

env = gym.make("GoToBall-v0")                    # tier 0: action (6,), obs (14,)
env = gym.make("GoToBallShoot-v0", tier=1)       # tier 1: roller physics, action (7,)
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

## Environments

| id | robots | obs | action (tier 0 / 1) | horizon | ends on |
|---|---|---|---|---|---|
| [`GoToBall-v0`](docs/environments/go_to_ball.md) | 1 blue | 14 | 6 / 7 | 20 s | at the ball, pointing at it |
| [`GoToBallShoot-v0`](docs/environments/go_to_ball_shoot.md) | 1 blue | 14 | 6 / 7 | 30 s | ball in the +x goal |
| [`Dribbling-v0`](docs/environments/dribbling.md) | 1 blue | 16 | 6 / 7 | 40 s | 4 waypoints carried, or ball lost for 2 s |
| [`StaticDefenders-v0`](docs/environments/static_defenders.md) | 1 blue + 6 yellow | 62 | 6 / 7 | 30 s | ball in the +x goal |
| [`ContestedPossession-v0`](docs/environments/contested_possession.md) | 1 blue + 1 yellow | 22 | 6 / 7 | 20 s | blue holds the ball 0.5 s |
| [`PassEndurance-v0`](docs/environments/pass_endurance.md) | 2 blue (1 controlled) | 22 | 6 / 7 | 20 s | team-mate receives the pass |

Every environment exposes the same per-robot action — drive, kick, chip, dribble — whether
or not its reward uses it, so one action head transfers across all six.

## Run it

```bash
uv run python examples/random.py --env GoToBall-v0
uv run python examples/random.py --env GoToBall-v0 --view      # MuJoCo viewer
uv run python -m pytest tests -q
```

## Documentation

[`docs/index.md`](docs/index.md) is the reference: the action and observation contracts,
the two physics tiers, how possession is measured, the multi-robot `MjSpec` attach, the
field constants and rendering. Each environment has its own page under
[`docs/environments/`](docs/environments/).
