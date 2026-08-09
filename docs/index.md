# mjsoccer

Gymnasium environments for a RoboCup SSL Division A omni-directional soccer robot,
simulated in MuJoCo. Six tasks share one robot model, one action contract and one
observation contract, so a policy trained on any of them can be evaluated on the rest
without reshaping its heads.

The package is standalone: it depends on `gymnasium`, `mujoco` and `numpy`, and carries its
own MJCF and meshes in `models/`.

## Install

```bash
uv sync
```

## Quickstart

```python
import gymnasium as gym
import mjsoccer                                  # registers the ids

env = gym.make("GoToBall-v0")                    # tier 0
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
env.close()
```

```bash
uv run python examples/random.py --env GoToBall-v0 --tier 1
uv run python examples/random.py --list
uv run python -m pytest tests -q
```

## Environment index

| id | robots | obs dim | action dim (tier 0 / tier 1) | horizon (s) | terminates on |
|---|---|---|---|---|---|
| [`GoToBall-v0`](environments/go_to_ball.md) | 1 blue | 14 | 6 / 7 | 20 | within 0.13 m of the ball and facing it (cos > 0.9) |
| [`GoToBallShoot-v0`](environments/go_to_ball_shoot.md) | 1 blue | 14 | 6 / 7 | 30 | ball in the +x goal |
| [`Dribbling-v0`](environments/dribbling.md) | 1 blue | 16 | 6 / 7 | 40 | 4 waypoints carried, or ball lost for 2 s |
| [`StaticDefenders-v0`](environments/static_defenders.md) | 1 blue + 6 yellow | 62 | 6 / 7 | 30 | ball in the +x goal |
| [`ContestedPossession-v0`](environments/contested_possession.md) | 1 blue + 1 yellow | 22 | 6 / 7 | 20 | blue holds the ball for 0.5 s |
| [`PassEndurance-v0`](environments/pass_endurance.md) | 2 blue (1 controlled) | 22 | 6 / 7 | 20 | team-mate receives the pass |

Every environment also terminates when the ball leaves the field, and truncates at its
horizon. All six controls exactly one blue robot; the remaining robots are scripted.

## Action space

One flat `Box(-1, 1, (n_controlled * act_dim,), float32)`. `BaseEnv.step` reshapes it to
`(n_controlled, act_dim)` and writes each row to that robot's actuators. Scripted robots
never appear in the action; they are driven inside the task's `_script`.

The per-robot slice is identical in every environment:

```
tier 0:  vx, vy, omega          body-frame velocity vector V   (3)
tier 1:  w1, w2, w3, w4         raw wheel speeds               (4)
then always:  kick, chip, dribble                              (3)
```

so `act_dim` is 6 on tier 0 and 7 on tier 1, for all six environments. Every robot always
exposes every control it physically has. A task that does not care about one simply does
not reward it, which is what keeps a single action head transferable across the suite.

| control | tier 0 actuator | tier 1 actuator | ctrlrange | commanded |
|---|---|---|---|---|
| `vx` | `vel_x` | — | -4 .. 4 m/s | `a * 4` |
| `vy` | `vel_y` | — | -4 .. 4 m/s | `a * 4` |
| `omega` | `vel_yaw` | — | -30 .. 30 rad/s | `a * 30` |
| `w1..w4` | — | `wheel{1..4}_vel` | -200 .. 200 rad/s | `a * 200` |
| `kick` | `kicker_force` | `kicker_force` | 0 .. 200 N | 200 N pulse if `a > 0` |
| `chip` | `chipper_force` | `chipper_force` | 0 .. 200 N | 200 N pulse if `a > 0` |
| `dribble` | `dribbler_vel` | `dribbler_vel` | -1500 .. 1500 rad/s | `314 * (a + 1) / 2` |

Everything is in `[-1, 1]` and scaled by the actuator's own `ctrlrange`, so the bounds come
from the model rather than from a constant in the Python. The drive scale is
`max(hi, -lo)` over the actuator's range, which is symmetric, so a zero action is a zero
command rather than the midpoint of the range.

**Body frame.** The tier-0 chassis joints are *world*-axis slides, but the action means a
body-frame velocity — that is how SSL firmware is commanded, and it is what tier 1's wheel
speeds inherently produce. `base._drive_robot` rotates the command through the chassis
orientation before writing it, which is what makes the two tiers mean the same thing. A
diagonal command can therefore ask for `4 * sqrt(2)` on one world axis; MuJoCo clamps it to
`ctrlrange`, exactly as a real servo would.

**Kicker and chipper.** `kick > 0` and `chip > 0` each fire a solenoid pulse: full
`KICK_FORCE = 200 N` on the plunger for one control step, then a 250 ms lockout on that
plunger. Held continuously the plunger pins against its 30 mm stop and never releases the
ball, so the lockout is what makes the mechanism usable at all. The two locks are
independent — a kick must not be able to block a chip. At the default control rate the
lockout is 16 control steps.

**Dribbler.** `[-1, 1]` maps to `[0, DRIBBLER_OMEGA]` = `[0, 314] rad/s`. It is never driven
backwards: a reversed bar throws the ball out of the mouth. Note that 314 rad/s is the
bar's design operating point, not the actuator's `ctrlrange`, which is deliberately about
5x larger to leave room for the speed feedforward a real controller adds on top. Scaling
the action by `ctrlrange` here would command a bar speed the mechanism was never
characterised at.

## Observation space

Flat `Box(-inf, inf, (6 + 8 * n_robots + extra_obs,), float32)`, where `n_robots` is every
robot in the model, controlled or not.

```
[0:6]                       ball    x, y, z, vx, vy, vz
[6 + 8*i : 14 + 8*i]        robot i x, y, sin yaw, cos yaw, vx, vy, omega, has_ball
[6 + 8*n_robots : ]         task extras
```

Robots are ordered controlled blue first, then the remaining blue, then yellow — the same
order `scene.build` attaches them in, so the observation index of `b0_robot` is the same in
every environment.

Robot linear velocity is world-frame (it is read straight off `qvel`), unlike the tier-0
drive command, which is body-frame. `has_ball` is `1.0` while the robot is inside the
possession grace window (see below) and `0.0` otherwise.

Every entry is normalised by a constant from `base.py` or `scene.py`:

| quantity | divisor | value |
|---|---|---|
| ball x, robot x | `scene.FIELD_HALF_X` | 6.0 m |
| ball y, robot y | `scene.FIELD_HALF_Y` | 4.5 m |
| ball z | `scene.GOAL_HEIGHT` | 0.155 m |
| ball vx, vy, vz | `base.MAX_BALL_V` | 8.0 m/s (a flat kick leaves at 6-7) |
| robot vx, vy | `base.MAX_V` | 4.0 m/s (the tier-0 chassis servo ctrlrange) |
| robot omega | `base.MAX_W` | 30.0 rad/s (likewise) |

The declared `Box` is unbounded because nothing clips these: a ball can be above
`GOAL_HEIGHT`, and a tier-1 chassis pushed by another robot can exceed `MAX_V`. In normal
play the values sit in roughly `[-1, 1]`.

## Tiers

`tier=0` is the kinematic chassis (slide-x / slide-y / hinge-z, 3 velocity servos);
`tier=1` gives every wheel its 20 rollers and real slip. Measured here, 100 random steps:

| env | tier 0 | tier 1 |
|---|---|---|
| `GoToBall-v0` | 3 300 steps/s | 650 steps/s |
| `StaticDefenders-v0` (7 robots) | 820 steps/s | 88 steps/s |

Those figures were measured with the older, narrower action space (drive only on some
tasks), so treat them as indicative of the ratio between the tiers rather than as a current
benchmark. The physics cost, which dominates, is unchanged.

Train on tier 0; use tier 1 to check what the policy does when the wheels can slip.

## Possession

Possession is ball/dribbler **contact**, not the arm-deflection sensor the hardware uses.
On this model a ball parked square in the mouth deflects the arm the *wrong way* — measured
-0.003 to -0.015 rad with the ball at (94, 0, 21) mm — so any positive threshold on
`dribbler_arm_pos` reports "no ball" forever. Contact is unambiguous, and it is what the
mjssl retention benchmarks score against.

`BaseEnv._update_possession` makes one pass over `data.contact` per step and marks every
dribbler body currently touching the ball, which is cheaper than asking per robot.

A spinning bar does not touch the ball every step either; it sticks, slips and re-grips, and
a held ball reads as contact on roughly half of them. So contact is held over a
`POSSESSION_GRACE = 0.05 s` window — 3 control steps at the default rate — the same
criterion the retention sweeps use before calling a ball lost. Without it, "carry the ball
for N seconds" never terminates successfully. `BaseEnv.holds(i)` reads that window.

`Dribbling-v0` and `ContestedPossession-v0` start with the ball already in a mouth, seated
the way the mechanism actually takes one: `BaseEnv.seat_ball` spins the bar up, rolls a
ball in at 2 m/s from 200 mm, and lets it be caught over 0.35 s of physics. Placing the ball
at the sphere/bar tangency instead is a launcher, not a seat — the contact needs a few
tenths of a millimetre of overlap to exist at all, and on a 46 g ball that overlap is a
compressed spring worth several newtons. Below ~0.1 mm MuJoCo stops generating the contact
at all, so there is no overlap that works.

## Multi-robot

`scene.py` attaches the one-robot MJCF N times with `mujoco.MjSpec.attach(prefix=)`, which
namespaces every body, joint, actuator, sensor and geom (`b0_robot`, `b0_vel_x`,
`b0_r_col_barDribbler`, ...). The parent is `scene.xml` itself with its one robot
**deleted**, so the `<option>` block, the `<visual>` block, the lights, the cameras, the
ball and the ball/floor contact pair are exactly what the model ships with — there is no
second copy of those numbers to drift out of sync. `MjSpec.delete` on the robot body
cascades to its actuators, its sensors and its two dribbler contact pairs.

Those dribbler pairs are then re-emitted once per robot, copied off the originals **before**
the delete rather than retyped. Skipping the step compiles fine and silently discards the
ball/bar contact tuning: MuJoCo would fall back to the elementwise max of the two geoms' own
friction, which is not what the model was tuned against.

The compiled `MjModel` is cached per `(n_blue, n_yellow, tier)` and shared by every env of
the same shape; each env owns its own `MjData`. That is safe only while nothing mutates the
model, so the first caller that wants domain randomisation has to drop the cache and pay the
~0.2 s compile.

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
| `GOAL_HEIGHT` | 0.155 m | the crossbar height, and the ball-z normaliser |

`scene._check_field` asserts every one of them against the CAD at build time, by taking the
world-space AABB of the line meshes and the goalpost collision geoms. It is cheap, and it is
the one thing that poisons every reward at once: an env that believes the goal is at
x = 6.0 on a field whose goal is elsewhere trains a policy to shoot at nothing, and every
check downstream still passes.

`scene.in_field(x, y)` and `scene.in_goal(pos, sign=1.0)` are derived from these. A ball is
in the goal when its centre is past the goal line, inside the mouth and under the crossbar.

## Rendering

`render_mode="human"` opens `mujoco.viewer.launch_passive` and syncs it once per `step`.

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

That script builds its own smaller renderer (480x270 by default) rather than the 720x1280
one `BaseEnv.render` hardcodes, and drives the clips with the repo's scripted
`goto_ball_policy` so the documentation shows a robot doing the task rather than a robot
twitching at random. `pillow` is a dev-only dependency and nothing in `mjsoccer` imports it.

## Timing

`control_hz` is a request, not a guarantee: `decimation = round(1 / (control_hz * timestep))`
is an integer number of physics steps. The model's timestep is 2 ms, so the default
`control_hz=60` gives `decimation = 8` and `dt = 0.016 s`, i.e. an actual control rate of
62.5 Hz. Every step count quoted in these docs is at that rate:
`max_steps = round(horizon_s / dt)`.
