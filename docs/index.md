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

env = gym.make("GoToBall-v0")                    # default tier 1
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
env.close()
```

```bash
uv run python examples/random.py --env GoToBall-v0 --tier 0
uv run python examples/random.py --list
uv run python -m pytest tests -q
```

## Environment index

| id | robots | obs dim | action dim (tier 1, default / tier 0) | horizon (s) | terminates on |
|---|---|---|---|---|---|
| [`GoToBall-v0`](environments/go_to_ball.md) | 1 blue | 14 | 7 / 6 | 20 | within 0.13 m of the ball and facing it (cos > 0.9) |
| [`GoToBallShoot-v0`](environments/go_to_ball_shoot.md) | 1 blue | 14 | 7 / 6 | 30 | ball in the +x goal |
| [`Dribbling-v0`](environments/dribbling.md) | 1 blue | 16 | 7 / 6 | 40 | 4 waypoints carried, or ball lost for 2 s |
| [`StaticDefenders-v0`](environments/static_defenders.md) | 1 blue + 6 yellow | 62 | 7 / 6 | 30 | ball in the +x goal |
| [`ContestedPossession-v0`](environments/contested_possession.md) | 1 blue + 1 yellow | 22 | 7 / 6 | 20 | blue holds the ball for 0.5 s |
| [`PassEndurance-v0`](environments/pass_endurance.md) | 2 blue (1 controlled) | 22 | 7 / 6 | 20 | team-mate receives the pass |

Every environment also terminates when the ball leaves the field, and truncates at its
horizon. All six controls exactly one blue robot; the remaining robots are scripted.

## Action space

One flat `Box(-1, 1, (n_controlled * act_dim,), float32)`. `BaseEnv.step` reshapes it to
`(n_controlled, act_dim)` and writes each row to that robot's actuators. Scripted robots
never appear in the action; they are driven inside the task's `_script`.

The per-robot slice is identical in every environment:

```
tier 1 (default):  w1, w2, w3, w4    raw wheel speeds               (4)
tier 0:            vx, vy, omega     body-frame velocity vector V   (3)
then always:       kick, chip, dribble                              (3)
```

so `act_dim` is 7 on the default tier 1 and 6 on tier 0, for all six environments. Only the
drive slice changes; `kick`, `chip` and `dribble` are always the last three, which puts them
at indices 4, 5 and 6 on tier 1 and at 3, 4 and 5 on tier 0. Every robot always exposes
every control it physically has. A task that does not care about one simply does not reward
it, which is what keeps a single action head transferable across the suite.

| control | tier 1 actuator (default) | tier 0 actuator | ctrlrange | commanded |
|---|---|---|---|---|
| `w1..w4` | `wheel{1..4}_vel` | — | -95 .. 95 rad/s, ±0.078 N·m | `a * 95` |
| `vx` | — | `vel_x` | -3.3 .. 3.3 m/s, ±6.19 N | `a * 3.3` |
| `vy` | — | `vel_y` | -3.3 .. 3.3 m/s, ±6.19 N | `a * 3.3` |
| `omega` | — | `vel_yaw` | -30 .. 30 rad/s, ±0.527 N·m | `a * 30` |
| `kick` | `kicker_force` | `kicker_force` | 0 .. 500 N | pulse for `6.5 * a` m/s if `a > 0` |
| `chip` | `chipper_force` | `chipper_force` | 0 .. 500 N | pulse for `6.5 * a` m/s if `a > 0` |
| `dribble` | `dribbler_vel` | `dribbler_vel` | -1500 .. 1500 rad/s | `314 * max(a, 0)` |

Everything is in `[-1, 1]` and scaled by the actuator's own `ctrlrange`, so the bounds come
from the model rather than from a constant in the Python. The drive scale is
`max(hi, -lo)` over the actuator's range, which is symmetric, so a zero action is a zero
command rather than the midpoint of the range.

**Body frame.** Tier 1's wheel speeds inherently produce a *body-frame* motion, which is
also how SSL firmware is commanded. Tier 0 has to be made to agree: its chassis joints are
*world*-axis slides, so `base._drive_robot` rotates the `vx, vy, omega` command through the
chassis orientation before writing it, which is what makes the two tiers mean the same
thing. A diagonal tier-0 command can therefore ask for `3.3 * sqrt(2)` on one world axis;
MuJoCo clamps it to `ctrlrange`, exactly as a real servo would.

**Kicker and chipper.** Half range, like the dribbler: `a <= 0` never fires, and
`a in (0, 1]` asks for an exit speed of `KICK_SPEED_MAX * a` = `6.5 * a` m/s, the §8.4.2 cap
at `a = 1`. That is what SSL firmware takes, a speed, not a trigger. The plunger is still a
force actuator against its return spring: a per-tier, per-plunger table
(`scene.KICK_TABLE_MS`) turns the asked-for speed into a one-step force, the way the
firmware's calibration table turns it into a pulse width. The cap takes ~290-345 N. Speeds
are relative to a standing robot with the ball settled in the mouth (within ±4 % across
seatings); a moving robot adds its own speed, as a real one does. The control step a
plunger fires on runs at an eighth of the timestep (`scene.FIRING_SUBSTEPS`): at 2 ms the
plate closes ~9 mm per step and the exit speed swung ±35 % with force. That costs ~1.6-1.7x in
throughput under random actions, which fire at every re-arm, and nothing for a policy that
kicks rarely. Each shot is followed by a 250 ms lockout on that plunger, a deliberate
approximation (no capacitor recharge time is known). Held continuously the plunger pins against its 30 mm stop and never releases the
ball, so the lockout is what makes the mechanism usable at all. The two locks are
independent — a kick must not be able to block a chip. At the default control rate the
lockout is 16 control steps.

**Dribbler.** `a <= 0` is off and `(0, 1]` maps to `(0, DRIBBLER_OMEGA]` = `(0, 314] rad/s`,
so a zero action does not dribble and `action_space.sample()` dribbles half the time, at a
mean of 157 rad/s when on, instead of always. It stays a
speed rather than a switch because the robot's firmware takes a speed. It is never driven
backwards: a reversed bar throws the ball out of the mouth. 314 rad/s is an assumption
(about 3000 rpm), not a team firmware setpoint. It stands in for the bar's
design operating point, not the actuator's `ctrlrange`, which is deliberately about
5x larger to leave room for the speed feedforward a real controller adds on top. Scaling
the action by `ctrlrange` here would command a bar speed the mechanism was never
characterised at.

## Observation space

Flat `Box((6 + 8 * n_robots + extra_obs,), float32)`, bounded per channel below, where `n_robots` is every
robot in the model, controlled or not.

```
[0:6]                       ball    x, y, z, vx, vy, vz
[6 + 8*i : 14 + 8*i]        robot i x, y, sin yaw, cos yaw, vx, vy, omega, has_ball
[6 + 8*n_robots : ]         task extras
```

Robots are ordered controlled blue first, then the remaining blue, then yellow — the same
order `scene.build` attaches them in, so the observation index of `b0_robot` is the same in
every environment.

Robot linear velocity is world-frame (it is read straight off `qvel`), unlike the drive
command, which is per-wheel on tier 1 and body-frame on tier 0. `has_ball` is `1.0` while
the robot is inside the possession grace window (see below) and `0.0` otherwise.

Every entry is normalised by a constant from `base.py` or `scene.py`:

| quantity | divisor | value |
|---|---|---|
| ball x, robot x | `scene.FIELD_HALF_X` | 6.0 m |
| ball y, robot y | `scene.FIELD_HALF_Y` | 4.5 m |
| ball z above the carpet (`z - scene.FLOOR_Z`) | `scene.GOAL_HEIGHT` | 0.155 m |
| ball vx, vy, vz | `base.MAX_BALL_V` | 8.0 m/s (kicks are capped at 6.5 off a standing robot) |
| robot vx, vy | `base.MAX_V` | 3.3 m/s (the tier-0 chassis servo ctrlrange) |
| robot omega | `base.MAX_W` | 30.0 rad/s (likewise) |

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
| Dribbling waypoint | ±2.57, ±3.427 | an in-field waypoint seen from a robot inside the walls |

Measured under random action, seeds 0-19, six envs: tier 1 robot `|v|` peaks at 0.46 and
`|omega|` at 0.14 of their divisors; tier 0 at 0.42 and 0.46. Under its own full command
the tier-1 chassis spins at 29.2 rad/s (< `MAX_W`, pinned by `test_full_spin_stays_under_max_w`)
and drives at ~2 m/s. `MAX_V` and `MAX_W` are the tier-0 servo `ctrlrange`, pinned by
`test_velocity_normalisers_are_the_tier0_servo_range`, and the same on both tiers so one
observation means one speed. The channels that do leave `[-1, 1]` are physical: a chipped
ball reads up to ~3 in height, a kick off a moving robot reaches ~8.5 m/s (1.06 of
`MAX_BALL_V`), and a robot behind the goal line or by the side wall reads up to ~1.05.

## Tiers

`tier=1` is the **default**: every wheel gets its 20 rollers and real slip, and the drive
slice is the four wheel speeds `w1..w4`. `tier=0` is the kinematic chassis (slide-x /
slide-y / hinge-z, 3 velocity servos) driven by a body-frame `vx, vy, omega`.

Tier 1 is the default because it is the physics the hardware actually has. An omni-wheel
robot does not track a commanded velocity vector; it applies four wheel torques through
contacts that slip, and everything that follows from that — the recoil off a ball it is
carrying, the yaw it picks up under lateral acceleration, the fact that a saturated wheel
does not mean a saturated chassis — only exists on tier 1. A policy trained against the
kinematic approximation has never met any of it.

Tier 0 is the fast approximation. It is several times cheaper to step, and it is what to
drop to when throughput is the binding constraint — early sweeps, reward debugging,
anything where the number of samples matters more than their fidelity. Measured here, 100
random steps:

| env | tier 0 | tier 1 |
|---|---|---|
| `GoToBall-v0` | 3 300 steps/s | 650 steps/s |
| `StaticDefenders-v0` (7 robots) | 820 steps/s | 88 steps/s |

Those figures were measured with the older, narrower action space (drive only on some
tasks), so treat them as indicative of the ratio between the tiers rather than as a current
benchmark. The physics cost, which dominates, is unchanged.

The two tiers are not interchangeable as evaluation targets — see
[Possession](#possession) for a measured behavioural difference — so a number produced on
one tier should always be reported with the tier it came from.

## Possession

Possession is ball/dribbler **contact**, not the arm-deflection sensor the hardware uses.
On this model a ball parked square in the mouth deflects the arm the *wrong way* — measured
-0.003 to -0.015 rad with the ball at (94, 0, 21) mm — so any positive threshold on
`dribbler_arm_pos` reports "no ball" forever. Contact is unambiguous, and it is what the
mjssl retention benchmarks score against.

`BaseEnv._update_possession` makes one pass over `data.contact` per step and marks every
dribbler body currently touching the ball, which is cheaper than asking per robot.

A spinning bar does not touch the ball every step either; it sticks, slips and re-grips. So
contact is held over a `POSSESSION_GRACE = 0.05 s` window — 3 control steps at the default
rate — the same criterion the retention sweeps use before calling a ball lost. Without it,
"carry the ball for N seconds" never terminates successfully. `BaseEnv.holds(i)` reads that
window, not the raw contact.

That distinction is worth a number, because the two differ a lot and by tier. A ball seated
in the mouth for 120 idle steps (no drive, `dribble = 0.5`, i.e. 157 rad/s), measured on `Dribbling-v0` over 8 seeds:

| | raw ball/dribbler contact | `holds()` |
|---|---|---|
| tier 0 | 74% of steps | 99% |
| tier 1 (default) | **65%** of steps | 95% |

On the default tier the bar is genuinely off the ball on a third of the steps, and the
grace window is the whole reason a carry task is solvable at all. Read a raw contact count
as "has the ball" and tier 1 looks like a robot that keeps dropping it.

**The tiers do not retain a carried ball alike.** Measured over 8 seeds, counting `holds()`
steps out of 120 idle ones — the robot seated on the ball, no drive command given, bar at `dribble = 0.5`:

| env | tier 0 | tier 1 (default) |
|---|---|---|
| `ContestedPossession-v0` | 69-95 / 120 | 89-96 / 120 |
| `Dribbling-v0` | 119 / 120 | 114 / 120 |

Tier 0 holds a ball it is not even trying to hold; on `Dribbling-v0` it drops it for a single
step. Tier 1 loses about 4% there, because real wheel slip lets the chassis recoil off the
ball it is carrying — a rigid kinematic chassis has nothing to recoil with. On
`ContestedPossession-v0`, where a defender is pressing, the order flips. The grace window
absorbs most of the difference, so `holds` stays usable on both, but any retention number,
ball-lost rate or possession-gated reward is tier-dependent and comparing one across tiers
is comparing two different mechanisms.

These numbers are from after ticket 05's contact changes (roller μ 0.25, ball β 0.5, the
motor-envelope wheel actuators of ticket 04) and ticket 12's tier-0 envelope, which
moved the tier-0 `ContestedPossession-v0` floor from 82 to 69 by contact noise alone. Before them `Dribbling-v0` held 120 / 111-112
and `ContestedPossession-v0` 96-103 / 87-95; the 5% retention budget ticket 05 was allowed
is enforced by `test_seated_ball_is_possessed`.

`Dribbling-v0` and `ContestedPossession-v0` start with the ball already in a mouth, seated
the way the mechanism actually takes one: `BaseEnv.seat_ball` spins the bar up, rolls a
ball in at 1 m/s from 200 mm, and lets it be caught over 0.35 s of physics. Not faster: with
the measured ball inertia (β = 0.5) a ball arriving at 2 m/s from 200 mm rebounds off the
bar and rolls out of reach on both tiers — at 1.0-1.5 m/s it is caught. Placing the ball
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
| `GOAL_HEIGHT` | 0.155 m | the crossbar height above the carpet, and the ball-z normaliser |
| `FLOOR_Z` | 0.02 m | the carpet plane's world z; every "height" subtracts it |
| `WALL_HALF_X`, `WALL_HALF_Y` | 6.3, 4.8 m | the inner wall faces; they bound the robot position observations |

`scene._check_field` asserts every one of them against the CAD at build time, by taking the
world-space AABB of the line meshes and the goalpost collision geoms. It is cheap, and it is
the one thing that poisons every reward at once: an env that believes the goal is at
x = 6.0 on a field whose goal is elsewhere trains a policy to shoot at nothing, and every
check downstream still passes.

`scene.in_field(x, y)` and `scene.in_goal(pos, sign=1.0)` are derived from these. A ball is
in the goal when its centre is past the goal line, inside the mouth and under the crossbar, its centre height measured from the carpet.

## Rendering

`render_mode="human"` opens `mujoco.viewer.launch_passive` and syncs it once per `step`.
The view starts on the whole field — south side, both goals in frame, `+x` on the right —
rather than on the viewer's own default, which is placed off the model's extent statistic
and lands close in on a patch of grass. It stays a *free* camera, so the mouse still
orbits, pans and zooms from there; only the opening view is set.

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

**The committed clips are rendered on tier 0**, not on the default tier 1, because
`goto_ball_policy` is a body-frame velocity controller: `vx, vy, omega` is a command four
wheel servos cannot be given, so on tier 1 the same policy has nothing meaningful to write
and the clips degenerate. What the GIFs show of the field, the ball, the goal geometry and
the dribbler mechanism is the same either way — only the chassis underneath the robot
differs.

## Timing

`control_hz` is a request, not a guarantee: `decimation = round(1 / (control_hz * timestep))`
is an integer number of physics steps. The model's timestep is 2 ms, so the default
`control_hz=60` gives `decimation = 8` and `dt = 0.016 s`, i.e. an actual control rate of
62.5 Hz. Every step count quoted in these docs is at that rate:
`max_steps = round(horizon_s / dt)`.
