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
| [`GoToBall-v0`](docs/environments/go_to_ball.md) | 1 blue | 14 | 7 | 20 s | at the ball, pointing at it |
| [`GoToBallShoot-v0`](docs/environments/go_to_ball_shoot.md) | 1 blue | 14 | 7 | 30 s | ball in the +x goal |
| [`Dribbling-v0`](docs/environments/dribbling.md) | 1 blue | 16 | 7 | 40 s | 4 waypoints carried, or ball lost for 2 s |
| [`StaticDefenders-v0`](docs/environments/static_defenders.md) | 1 blue + 6 yellow | 62 | 7 | 30 s | ball in the +x goal |
| [`ContestedPossession-v0`](docs/environments/contested_possession.md) | 1 blue + 1 yellow | 22 | 7 | 20 s | blue holds the ball 0.5 s |
| [`PassEndurance-v0`](docs/environments/pass_endurance.md) | 2 blue (1 controlled) | 22 | 7 | 20 s | team-mate receives the pass |

Every environment exposes the same per-robot action — a target pose `(x, y, sin θ, cos θ)`
plus kick, chip, dribble — whether or not its reward uses it, so one action head transfers
across all six. The navigation controller (Planning-the-path-with-rl's `go_to_point_new`)
turns the pose into a chassis velocity every 25 ms control step.

## Physics

The robot is an ideal kinematic chassis (ADR 0001) with the envelope derived from the
real robot's motor (Maxon EC 45 flat 50 W, direct drive: ±95 rad/s and ±0.078 N·m per
wheel) and its wheel friction: `v_max` 3.30 m/s, `a_max` 1.30 m/s², `ω_max` 37.1 rad/s,
`α_max` 55.0 rad/s², isotropic. Deliberately not modelled: slip, traction budget, weight
transfer, coupling between axes. The chassis slides on the carpet and cannot tip. Physics steps at 1/960 s, so the
default 60 Hz control rate is exact. Every contact and body number has a source or is
recorded as a deliberate approximation in [`docs/index.md`](docs/index.md).

## Run it

```bash
uv run python examples/random.py --env GoToBall-v0
uv run python examples/random.py --env GoToBall-v0 --view      # MuJoCo viewer
uv run python -m pytest tests -q
uv run python -m pytest tests -m validation -q                  # physics acceptance criteria, ~2 min
uv run python scripts/validation_sweep.py --out runs/validation --seeds 0-4  # their distributions, as CSV
```

## Documentation

[`docs/index.md`](docs/index.md) is the reference: the action and observation contracts,
the chassis and its envelope, the contact
parameters and their sources, how possession is measured, the multi-robot `MjSpec` attach, the
field constants and rendering. Each environment has its own page under
[`docs/environments/`](docs/environments/).
