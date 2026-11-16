# mjsoccer

Gymnasium environments for a RoboCup SSL Division A omni-directional soccer robot,
simulated in MuJoCo. Six tasks share one robot model, one action contract and one
observation contract, so a policy trained on any of them can be evaluated on the rest
without reshaping its heads.

The package is standalone: it depends on `gymnasium`, `mujoco` and `numpy`, and carries its
own MJCF and meshes in `models/`.

## Documentation map

| page | covers |
|---|---|
| this page | install, quickstart, environments, action and observation spaces, timing, field, rendering |
| [`environments/`](environments/go_to_ball.md) | one page per task: reward, endings, reset layout |
| [`physics.md`](physics.md) | chassis, drive envelope, kicker, dribbler, possession, contact parameters, multi-robot build |
| [`validation.md`](validation.md) | acceptance criteria, GPU × CPU parity, the tests that pin them |
| [`backends/`](backends/index.md) | the MJX-Warp and mjlab GPU backends: APIs, training, parity, domain randomization |

## Install

```bash
uv sync                  # the CPU backend
uv sync --group rl       # optional: SB3 / CleanRL for the CPU examples
uv sync --group gpu      # optional: the MJX-Warp backend, mjsoccer.mjx (CUDA GPU)
uv sync --group mjlab    # optional: the mjlab backend, mjsoccer.mjlab (CUDA GPU, driver >= 580)
```

Details per backend: [`backends/`](backends/index.md).

## Quickstart

```python
import gymnasium as gym
import mjsoccer                                  # registers the ids

env = gym.make("GoToBall-v0")
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
env.close()
```

```bash
uv run python examples/random.py --env GoToBall-v0
uv run python examples/random.py --list
uv run python -m pytest tests -q                 # default suite
uv run python -m pytest tests -m validation -q   # physics acceptance criteria, CPU, ~1 min
```

The acceptance criteria and how to run them on every backend: [`validation.md`](validation.md).
GPU training and tests: [`backends/`](backends/index.md).

## Environment index

| id | robots | obs dim | action dim | horizon (s) | terminates on |
|---|---|---|---|---|---|
| [`GoToBall-v0`](environments/go_to_ball.md) | 1 blue + 6 yellow | 62 | 7 | 30 | within 0.2 m of the ball (+1) |
| [`GoToBallShoot-v0`](environments/go_to_ball_shoot.md) | 1 blue | 14 | 7 | 60 | goal (+5); robot or ball leaves the attacking half, robot in the penalty area |
| [`Dribbling-v0`](environments/dribbling.md) | 1 blue + 4 yellow | 47 | 7 | 120 | 7 zigzag checkpoints (+1 each); robot off the course, a yellow robot moved |
| [`StaticDefenders-v0`](environments/static_defenders.md) | 1 blue + 6 yellow | 62 | 7 | 25 | as GoToBallShoot |
| [`ContestedPossession-v0`](environments/contested_possession.md) | 1 blue + 1 yellow | 22 | 7 | 30 | as GoToBallShoot, or the yellow robot shoved over 0.1 m/s |
| [`PassEndurance-v0`](environments/pass_endurance.md) | 2 blue (1 controlled) | 22 | 7 | 30 | team-mate receives the pass (+1); wrong ball (-1) |

Rewards, endings and reset layouts are robocin/rSoccer's, ported task by task on our Div A
field (each env page says what could not be literal). Every environment truncates at its
horizon. All six control exactly one blue robot; the remaining robots are scripted or idle.

## Action space

One flat `Box(-1, 1, (n_controlled * act_dim,), float32)`. `BaseEnv.step` reshapes it to
`(n_controlled, act_dim)` and writes each row to that robot's actuators. Scripted robots
never appear in the action; they are driven inside the task's `_script`.

The per-robot slice is identical in every environment:

```
x, y, sin θ, cos θ    target pose                    (4)
kick, chip, dribble                                  (3)
```

so `act_dim` is 7 for all six environments, with `kick`, `chip` and `dribble` at indices 4,
5 and 6. Every robot always exposes every control it physically has. A task that does not
care about one simply does not reward it, which keeps a single action head transferable
across the suite.

| control | commanded |
|---|---|
| `x`, `y` | target position `(6.0 a_x, 4.5 a_y)` m, target velocity 0 |
| `sin θ`, `cos θ` | target heading `atan2(a_sin, a_cos)` |
| `kick` | `kicker_force` (0 .. 500 N): pulse for `6.5 * a` m/s if `a > 0` |
| `chip` | `chipper_force` (0 .. 500 N): pulse for `6.5 * a` m/s if `a > 0` |
| `dribble` | `dribbler_vel` (-1500 .. 1500 rad/s): `314 * max(a, 0)` |

A zero action is a target pose too: the centre spot, facing +x.

**Navigation controller.** Every control step, `navigation.go_to_point` -- the
`go_to_point_new` of Planning-the-path-with-rl (commit `2287531`), ported as a pure
function -- turns the target pose into a body-frame `vx, vy, omega` from the robot's own
joint state (`qpos`/`qvel` of `robot_x/y/yaw`, not the observation). Its quirks are kept
verbatim: it adds `MAX_DV = 0.1` m/s per step to the *measured* velocity (the servo, at
1.30 m/s², is what really limits acceleration), plans braking at 1 m/s², has no position
deadband (±0.1 m/s chatter at the target), a 2.5° heading deadzone and `omega = 2 Δθ` with
no limit of its own. `max_velocity` is the 3.3 m/s envelope, per axis in the frame rotated
onto the robot→target line. Only the controlled robots go through it; scripted ones are
driven in `_script`. In PassEndurance the controller runs whole and then `vx, vy` go to 0:
the passer only turns, and `x, y` of the action mean nothing.

The controller's output goes to the chassis velocity servos (3.3 m/s, 37.1 rad/s), rotated
from body to world frame: [Chassis](physics.md#chassis).

**Kick and chip.** Half range: `a <= 0` never fires, and `a in (0, 1]` asks for an exit
speed of `KICK_SPEED_MAX * a` = `6.5 * a` m/s, the SSL §8.4.2 cap at `a = 1`. That is what
SSL firmware takes: a speed, not a trigger. Speeds are relative to a standing robot; a moving
robot adds its own. After a shot that plunger is locked for 250 ms (10 control steps); the
kicker and chipper locks are independent. Mechanism: [Kicker and chipper](physics.md#kicker-and-chipper).

**Dribble.** `a <= 0` is off and `(0, 1]` maps to `(0, 314]` rad/s, so a zero action does
not dribble and `action_space.sample()` dribbles half the time, at a mean of 157 rad/s when
on. It is a speed rather than a switch because the robot's firmware takes a speed. It is
never driven backwards. Why 314 and not the `ctrlrange`: [Dribbler](physics.md#dribbler).

## Observation space

Flat `Box((6 + 8 * n_robots + extra_obs,), float32)`, bounded per channel below, where
`n_robots` is every robot in the model, controlled or not.

```
[0:6]                       ball    x, y, z, vx, vy, vz
[6 + 8*i : 14 + 8*i]        robot i x, y, sin yaw, cos yaw, vx, vy, omega, has_ball
[6 + 8*n_robots : ]         task extras
```

Robots are ordered controlled blue first, then the remaining blue, then yellow — the same
order `scene.build` attaches them in, so the observation index of `b0_robot` is the same in
every environment.

Robot linear velocity is world-frame (it is read straight off `qvel`), unlike the drive
command, which is body-frame. `has_ball` is `1.0` while the robot is inside the possession
grace window and `0.0` otherwise: [Possession](physics.md#possession).

Every entry is normalised by a constant from `base.py` or `scene.py`:

| quantity | divisor | value |
|---|---|---|
| ball x, robot x | `scene.FIELD_HALF_X` | 6.0 m |
| ball y, robot y | `scene.FIELD_HALF_Y` | 4.5 m |
| ball z above the carpet (`z - scene.FLOOR_Z`) | `scene.GOAL_HEIGHT` | 0.155 m |
| ball vx, vy, vz | `base.MAX_BALL_V` | 8.0 m/s (kicks are capped at 6.5 off a standing robot) |
| robot vx, vy | `base.MAX_V` | 3.3 m/s (the chassis servo ctrlrange) |
| robot omega | `base.MAX_W` | 37.1 rad/s (likewise) |

`MAX_V` and `MAX_W` are pinned to the servo `ctrlrange` by
`test_velocity_normalisers_are_the_chassis_servo_range`.

Nothing clips these. The declared `Box` is finite only where the simulator guarantees a
bound, so `observation_space.contains` is true of every observation:

| channel | bound | why |
|---|---|---|
| robot x, y | ±1.05, ±1.067 | the inner wall faces, `scene.WALL_HALF_X = 6.3`, `WALL_HALF_Y = 4.8` m |
| sin, cos yaw | ±1 | by construction |
| `has_ball` | 0, 1 | by construction |
| ball z | 0 to `inf` | the ball sits on the carpet; a chip has no ceiling |
| ball x, y | `inf` | a chip clears the 0.2 m wall |
| every velocity | `inf` | collisions have no ceiling |
| Dribbling checkpoints | -1 to 4/3 | rSoccer's `(count / 6) * 2 - 1`, count 0-7 |

Measured under random actions (seeds 0-4, six envs, up to 400 control steps each): robot
`|v|` peaks at 0.33 and `|omega|` at 0.14 of their divisors, ball speed at 0.56, and a
chipped ball reads up to ~2.3 in height. A robot behind the goal line or by the side wall
can read past 1 in x or y.

## Timing

The physics step is 1/960 s (~1.04 ms) and the control step is a fixed 25 ms (40 Hz,
`decimation = 24`, `dt = 0.025` s), rSoccer's step and the rate the navigation controller
was written for. Every step count quoted in these docs is in control steps:
`max_steps = round(horizon_s / dt)`. A policy decides every `repeat_action` control steps
(kwarg, default 1); one `step` call is one decision, its rewards summed over the control
steps and cut short when the episode ends.

The control step is the controller's and the firmware's tick; the decision
(`repeat_action` control steps) is the policy's rate. The 24 physics steps inside a control
step are the integration: the chassis velocity servos act on every one of them. So
`repeat_action` is a policy setting, and the timestep is a resolution setting. Why 1/960 s:
[Contact and body parameters](physics.md#contact-and-body-parameters).

Seconds are the contract. Horizons, the 250 ms plunger lock (10 control steps) and the
50 ms possession grace (2 control steps) are declared in seconds and converted with
`round(seconds / dt)`. Per-step rewards are rSoccer's, per control step, and match it at
40 Hz. Horizons are rSoccer's (robocin/rSoccer). `test_horizon_is_seconds_at_exactly_40_hz`
pins each env's step count, so a change of rate is never silent.

Throughput: `scripts/validation_sweep.py --scenario random` writes steps/s per env to
`throughput.csv` on the machine it runs on.

## Field constants

RoboCup SSL Division A, taken from the rulebook rather than measured off the mesh — the mesh
gives the *painted line* extents, and which edge of a 20 mm line is the boundary is a rule,
not a geometric fact.

| constant | value | meaning |
|---|---|---|
| `FIELD_HALF_X` | 6.0 m | half the field length; the goal line is at x = +/-6.0 |
| `FIELD_HALF_Y` | 4.5 m | half the field width |
| `GOAL_HALF_Y` | 0.9 m | half the 1.8 m goal mouth |
| `GOAL_DEPTH` | 0.18 m | how far the goal extends behind the line |
| `GOAL_HEIGHT` | 0.155 m | the crossbar height above the carpet, and the ball-z normaliser |
| `FLOOR_Z` | 0.02 m | the carpet plane's world z; every "height" subtracts it |
| `WALL_HALF_X`, `WALL_HALF_Y` | 6.3, 4.8 m | the inner wall faces; they bound the robot position observations |

`scene._check_field` checks them against the CAD at build time, from world-space AABBs of
the mesh geoms:

| constant | checked against | tolerance |
|---|---|---|
| `FIELD_HALF_X`, `FIELD_HALF_Y` | the `f_lines` meshes | 25 mm |
| `FIELD_HALF_X` (goal line) | the near face of the `f_goalpost_1` collision hulls | 10 mm |
| `GOAL_DEPTH` | the depth of those hulls | 30 mm |
| `GOAL_HALF_Y` | under the posts' outer edge, which must lie within 60 mm of it | — |
| `WALL_HALF_X`, `WALL_HALF_Y` | the outer faces of the `col_f_walls` hulls, within 25 mm outside (the wall is 20 mm thick) | 25 mm |
| `FLOOR_Z` | the `floor` geom's z | exact |

`GOAL_HEIGHT` is not checked. The check is cheap, and the error it catches poisons every
reward at once: an env that believes the goal is at x = 6.0 on a field whose goal is
elsewhere trains a policy to shoot at nothing, and every check downstream still passes.

`scene.in_field(x, y)` and `scene.in_goal(pos, sign=1.0)` are derived from these. A ball is
in the goal when its centre is past the goal line, inside the mouth and under the crossbar,
its centre height measured from the carpet.

## Rendering

`render_mode="human"` opens a MuJoCo Studio viewer (`mujoco.studio`, `src/mjsoccer/viewer.py`)
and syncs it once per `step`. The viewer is a spectator: it gets a copy of the state, and
whatever it hands back from `sync` (edits made in its UI) is ignored. Closing the window
does not end the episode; the next `render()` opens a new one.

The `viewer_gfx` kwarg picks how it draws:

- `opengl_headless` (default): a native window. Filament renders to a texture and SDL
  blits it, the one native mode that runs under Wayland here.
- `web`: no window and no GPU on the host. The URL is printed on launch, and the browser
  draws the scene (e.g. over `ssh -L`). **It listens on all interfaces with no
  authentication**, so use it only on a trusted network or behind a tunnel. In mujoco 4.0.x
  only one web viewer works per process; a second one raises.

The view starts on the whole field — south side, both goals in frame, `+x` on the right —
rather than on Studio's default, which is placed off the model's extent statistic and lands
close in on a patch of grass. A `ViewerPlugin` sets it once, when the model first reaches the
viewer. It stays a *free* camera, so the mouse still orbits, pans and zooms from there; only
the opening view is set.

`render_mode="rgb_array"` returns a 720x1280 RGB frame from an off-screen
`mujoco.Renderer`, using the model's `overview` camera.

Headless machines need an off-screen GL backend:

```bash
MUJOCO_GL=egl uv run python examples/random.py --env GoToBall-v0
```

The GIFs on the environment pages are regenerated by:

```bash
MUJOCO_GL=egl uv run --group dev python scripts/make_gifs.py
```

That script builds its own smaller renderer (640x360 by default, `--width`/`--height`)
rather than the 720x1280 one `BaseEnv.render` hardcodes, and drives the clips with the
repo's scripted `goto_ball_policy`, so the documentation shows a robot doing the task rather
than a robot twitching at random. `pillow` is a dev-only dependency and nothing in
`mjsoccer` imports it.
