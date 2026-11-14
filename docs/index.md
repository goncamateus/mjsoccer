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
uv sync --group gpu      # optional: the MJX-Warp backend, mjsoccer.mjx (needs a CUDA GPU to step)
```

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
uv run python -m pytest tests -q
uv run python -m pytest tests -m validation -q                  # acceptance criteria, ~2 min
# with the gpu group the default run adds the MJX task-logic parity (JAX-CPU) and the Warp
# smoke (skipped without CUDA); run it under a RAM cap:
systemd-run --user --scope -p MemoryMax=10G uv run python -m pytest tests -q
uv run python scripts/validation_sweep.py --out runs/validation --seeds 0-4  # the distributions behind them, CSV
```

PPO on the GPU (Brax `ppo.train` over `mjsoccer.mjx`, one process per run; checkpoints in
`runs/ppo_mjx_<env>/`, see the script's docstring for the flags):

```bash
systemd-run --user --scope -p MemoryMax=10G uv run --group gpu examples/ppo_mjx.py --env GoToBall-v0
uv run --group gpu examples/ppo_mjx.py --eval runs/ppo_mjx_GoToBall-v0   # MJX, aborts on a failed world
uv run --group gpu examples/ppo_mjx.py --play runs/ppo_mjx_GoToBall-v0   # CPU backend, Studio viewer
```

The `validation` suite is excluded from the default run (`pyproject.toml`). It is the
regression check for the physics acceptance criteria: yaw p99 under `MAX_W` in every
scripted scenario, in one random-action episode per env and in five `goto_ball_policy`
episodes; and the navigation controller's, from rest to target poses 0.5 / 2 / 6 m away
turned 0 / 90 / 180 deg: final error under 1 cm and 2.5 deg, settled within 1.05x the
analytic trapezoid (3.3 m/s, up at 1.30 m/s², down at 1.0), chatter at the target under
0.05 m/s RMS over the last second, commanded omega never over `MAX_W`. `validation_sweep.py` runs the same scenarios over
seeds 0-4 and writes per-episode, summary, navigation, contact-impulse and throughput CSVs;
its docstring lists the scenarios.

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
horizon. All six control exactly
one blue robot; the remaining robots are scripted or idle.

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
5 and 6. Every robot always exposes
every control it physically has. A task that does not care about one simply does not reward
it, which is what keeps a single action head transferable across the suite.

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

The chassis servos it drives:

| servo | actuator | ctrlrange |
|---|---|---|
| `vx` | `vel_x` | -3.3 .. 3.3 m/s, ±3.51 N |
| `vy` | `vel_y` | -3.3 .. 3.3 m/s, ±3.51 N |
| `omega` | `vel_yaw` | -37.1 .. 37.1 rad/s, ±0.526 N·m |

`BaseEnv.step_raw` is an internal path outside the action space, for the envelope tests:
its slice is `(vx, vy, omega, kick, chip, dribble)`, the drive scaled by `ctrlrange` and
written straight to the servos.

**Hardware wheel geometry.** The chassis has no wheels to command, but its envelope is
derived from the real robot's four omni-wheels, mounted at `a_i = 60, 135, 225, 300` deg from
+x, `R = 0.0796` m from the centre, rolling on an effective radius `r_eff = 0.030089` m.
The geometry also fixes how much of the four wheels' force a translation can use. A
forward push projects each wheel by `|sin a|` and a sideways one by `|cos a|`, so the sums
are **`Σ|sin a| = 3.14626` forward** and **`Σ|cos a| = 2.41421` lateral**, not 4 (the
force-maximising sign pattern has `Σ f_i = 0`, so it costs no yaw). Every acceleration
quoted for this robot, or for rSim's, is the per-wheel force times these factors.

**Drive envelope.** The chassis limits in the action table come from a motor, not a guess (tickets
01, 04, 05, 16; development in `.scratch/physics-fidelity/research/01-drivetrain-envelope.md`).
The robot is a Maxon EC 45 flat 50 W (winding 251601, 24 V, `k_M` 33.5 mN·m/A) in **direct
drive** on a ~60 mm wheel — the CAD's hubs are `Direct_Transmission_*`, and rSim's
`0.070 N·m`/`1557 rpm` describe the team's earlier geared robot.

| | forward (x) | lateral (y) | yaw | binding limit |
|---|---|---|---|---|
| wheel torque | ±0.078 N·m | | | `I_nom · k_M`, the motor's continuous rating (VENDOR) |
| wheel speed | ±95 rad/s | | | a design choice: 25 % under the spin tip-over rate `ω_tip = 47.7` rad/s, far under the motor's ~702 rad/s no-load (DELIBERATE APPROXIMATION) |
| acceleration | **1.95** m/s² (torque alone: 2.98) | **1.30** m/s² (torque alone: 2.30) | **55.0** rad/s² | wheel friction `μ = 0.25`: each wheel pushes along its own tangent (`μ Σ N cos / m`); lateral, no moment about the CoM puts the lighter rear wheels at their grip first; yaw `μ Σ N d / I_zz` about the CoM |
| speed | **3.30** m/s | **4.04** m/s | **37.1** rad/s | wheel speed through the matrix; yaw about the CoM, 3.45 mm behind the wheel centre (35.9 about the centre) |

**Body frame.** SSL firmware is commanded in the *body* frame, but the chassis joints are
*world*-axis slides, so `base._drive_robot` rotates the controller's `vx, vy, omega` command through the
chassis orientation before writing it. A diagonal command can therefore ask for
`3.3 * sqrt(2)` on one world axis; MuJoCo clamps it to `ctrlrange`, exactly as a real servo
would.

**Kicker and chipper.** Half range, like the dribbler: `a <= 0` never fires, and
`a in (0, 1]` asks for an exit speed of `KICK_SPEED_MAX * a` = `6.5 * a` m/s, the §8.4.2 cap
at `a = 1`. That is what SSL firmware takes, a speed, not a trigger. The plunger is still a
force actuator against its return spring: a per-plunger table
(`scene.KICK_TABLE_MS`) turns the asked-for speed into a one-step force, the way the
firmware's calibration table turns it into a pulse width, measured at 40 Hz on mujoco 4.0. The cap takes ~295-315 N. Speeds
are relative to a standing robot with the ball settled in the mouth (within ±4 % across
seatings); a moving robot adds its own speed, as a real one does. The control step a
plunger fires on opens with 32 physics steps at a quarter of the timestep
(`scene.FIRING_FINE_STEPS` at `scene.FIRING_SUBSTEPS`, ~0.26 ms each, 8.3 ms that cover the
stroke and the hit) and runs the rest at the timestep: at 2 ms the plate closed ~9 mm per step
and the exit speed swung ±35 % with force, 1 ms still saws, and 32 is the smallest count whose
curve is monotone for both plungers. At 40 Hz that is 48 iterations instead of 24, which costs
throughput under random actions, which fire at every re-arm, and nothing for a policy that
kicks rarely. Each shot is followed by a 250 ms lockout on that plunger, a deliberate
approximation (no capacitor recharge time is known). Held continuously the plunger pins against its 30 mm stop and never releases the
ball, so the lockout is what makes the mechanism usable at all. The two locks are
independent — a kick must not be able to block a chip. At the default control rate the
lockout is 15 control steps.

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
command, which is body-frame. `has_ball` is `1.0` while
the robot is inside the possession grace window (see below) and `0.0` otherwise.

Every entry is normalised by a constant from `base.py` or `scene.py`:

| quantity | divisor | value |
|---|---|---|
| ball x, robot x | `scene.FIELD_HALF_X` | 6.0 m |
| ball y, robot y | `scene.FIELD_HALF_Y` | 4.5 m |
| ball z above the carpet (`z - scene.FLOOR_Z`) | `scene.GOAL_HEIGHT` | 0.155 m |
| ball vx, vy, vz | `base.MAX_BALL_V` | 8.0 m/s (kicks are capped at 6.5 off a standing robot) |
| robot vx, vy | `base.MAX_V` | 3.3 m/s (the chassis servo ctrlrange) |
| robot omega | `base.MAX_W` | 37.1 rad/s (likewise) |

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

Measured under random action, seeds 0-19, six envs: robot `|v|` peaks at 0.30 and
`|omega|` at 0.43 of their divisors. `MAX_V` and `MAX_W` are the servo `ctrlrange`, pinned by
`test_velocity_normalisers_are_the_chassis_servo_range`. The channels that do leave `[-1, 1]` are physical: a chipped
ball reads up to ~3 in height, a kick off a moving robot reaches ~8.5 m/s (1.06 of
`MAX_BALL_V`), and a robot behind the goal line or by the side wall reads up to ~1.05.

## Chassis

The robot is a kinematic chassis (slide-x / slide-y / hinge-z, 3 velocity servos) driven by
a body-frame `vx, vy, omega` (ADR 0001). Measured by
`scripts/validation_sweep.py --scenario random`, one process, random actions, at the
1/960 s default (`.scratch/physics-fidelity/candidate-14/throughput.csv`):

| env | steps/s |
|---|---|
| `GoToBall-v0` | ~1 500 |
| `Dribbling-v0` | ~1 200 |
| `StaticDefenders-v0` (7 robots) | ~360 |

The 1/960 s step costs about half the throughput of the old 2 ms one: the price of
resolving the contacts (see Timing). The validation criterion is "within 10 % of that file on the same machine".

**The hardware envelope, not the hardware mechanism.** The servos carry the four per-axis
limits of the derived envelope above, each isotropic at the smaller of the forward and
lateral value:

| limit | chassis | from |
|---|---|---|
| `v_max` | 3.30 m/s (`ctrlrange`) | forward speed |
| `a_max` | 1.30 m/s² (`forcerange` ±3.5129 N = m·a) | lateral, friction-bound acceleration |
| `ω_max` | 37.1 rad/s (`ctrlrange`) | the ceiling of a spin about the CoM |
| `α_max` | 55.0 rad/s² (`forcerange` ±0.5263 N·m = `I_zz`·α) | friction-limited yaw |

Isotropic because the slides are world axes: a body-frame limit would need a clip in
the Python, and the limits come from the model. The chassis joint damping is 0, so nothing
else eats the thrust. `test_chassis_has_the_hardware_envelope` pins the step response to 3 %.

**What is deliberately not modelled**: slip, traction, weight transfer, and the coupling
between axes. The envelope is a box — it can reach all three limits at once, and along a
world diagonal up to `√2 ×` its `v` and `a`. The real robot's is a polytope: at full
forward speed no wheel is left over for yaw. The chassis slides on the carpet and cannot tip.

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

That distinction is worth a number. A ball seated
in the mouth for 120 idle steps (no drive, `dribble = 0.5`, i.e. 157 rad/s), measured on `Dribbling-v0` over 8 seeds:

| raw ball/dribbler contact | `holds()` |
|---|---|
| 97% of steps | 98% |

Before ticket 22 these read 74 % / 99 %: the old ball↔bar contact was
underdamped and ~10× stiffer than the bar's PU sleeve, so the bar hammered the ball (normal
force p10/p90 0.44 / 11 N) and was genuinely off it on a third of the steps. With the
PU 65A contact it stays on, and the grace window only covers the odd re-grip.

Counting `holds()` steps out of 120 idle ones over 8 seeds — the robot seated on the ball,
no drive command given, bar at `dribble = 0.5` — `ContestedPossession-v0` holds 120 / 120 and
`Dribbling-v0` 118 / 120, every seed alike.

These numbers are from after ticket 05's contact changes (ball β 0.5), ticket 12's
envelope, ticket 13's 1/960 s step, ticket 21's ball bounce and ticket 22's PU 65A
ball↔bar contact (before it: 120 and 117). On the old 2 ms step
`ContestedPossession-v0` scattered 69-95 and looked chaotic; it was the step under-resolving the ball/dribbler contact (120 at 0.5 ms,
87 at 2 ms, 0 at 4 ms). Before ticket 05 `Dribbling-v0` held 120
and `ContestedPossession-v0` 96-103; the 5% retention budget ticket 05 was allowed
is enforced by `test_seated_ball_is_possessed`.

`Dribbling-v0` and `ContestedPossession-v0` start with the ball already in a mouth, seated
the way the mechanism actually takes one: `BaseEnv.seat_ball` spins the bar up, rolls a
ball in at 1 m/s from 200 mm, and lets it be caught over 0.35 s of physics. Not faster: with
the measured ball inertia (β = 0.5) a ball arriving at 2 m/s from 200 mm rebounds off the
bar and rolls out of reach — at 1.0-1.5 m/s it is caught. Placing the ball
at the sphere/bar tangency instead is a launcher, not a seat — the contact needs a few
tenths of a millimetre of overlap to exist at all, and on a 46 g ball that overlap is a
compressed spring worth several newtons. Below ~0.1 mm MuJoCo stops generating the contact
at all, so there is no overlap that works.

**Turning with the ball loses it past ~2 rad/s.** Bar at full speed (`dribble = 1`), robot
spinning in place, `holds()` steps out of 120, worst of seeds 0-2 (ticket 22):

| yaw | 0.5 rad/s | 1 rad/s | 2 rad/s | 3 rad/s |
|---|---|---|---|---|
| `holds()` | 107 | 100 | 81 | 11 |

Before ticket 22 the hammering bar dropped it from ~1 rad/s (13-14 of 120 at 2 rad/s);
`test_seated_ball_survives_a_2_rad_s_turn` pins the new numbers. More bar speed does not
help: the faster the bar, the more of the friction cone goes along the spin. 3 rad/s is an
open fidelity gap, not a target — published SSL dribblers hold 3-7 rad/s, by contact area
(soft or split bars) rather than friction (`.scratch/physics-fidelity/research/03-dribbler-material.md`).
A task that needs a turn in possession has to keep it under ~2 rad/s.

## Contact and body parameters

Every number below either has a source or is recorded as a deliberate approximation. The
classes rank RULEBOOK > MEASURED > VENDOR > DELIBERATE APPROXIMATION (development in
`.scratch/physics-fidelity/research/02-physical-constants.md` and ticket 05).

| parameter | value | class | source / reason |
|---|---|---|---|
| ball mass, radius | 0.046 kg, 0.0215 m | RULEBOOK | SSL rules §2.2 |
| kick and chip speed cap | 6.5 m/s | RULEBOOK | SSL rules §8.4.2 |
| ball inertia | `diaginertia` for β = 0.5 (`I = β m r²`), not MuJoCo's solid-sphere 0.4 | MEASURED | `models/ball.xml` |
| ball ↔ floor sliding μ | 0.35 | MEASURED | ER-Force's empirical μ_k |
| ball rolling μ | 0.0009 → 0.293 m/s² rolling deceleration | MEASURED | measured range 0.26-0.31 m/s² |
| ball ↔ floor, ball ↔ wall restitution | COR 0.48-0.55 floor, 0.49-0.53 wall (normal impacts, 0.5-6.5 m/s): `<pair>` `solref="0.012 0.225"` on the carpet, `"0.012 0.25"` on the 12 wall hulls, both in `models/ball.xml` | MEASURED | target 0.50, measured 0.47-0.56 on carpet. MuJoCo has no restitution coefficient, so the `solref` was found by measurement (tickets 17, 21); 0.012 s is the shortest time constant whose COR holds within 5 % between 1/960 s and 0.5 ms from 1 m/s up (the floor at 0.5 m/s is 6 % off). MuJoCo's COR rises with speed where the real ball's falls, but stays in the band up to the 6.5 m/s kick. Oblique impacts are not calibrated: a backspinning chip's first landing reads 0.59-0.60. The goalposts stay critically damped |
| chassis ↔ floor | collided (floor `contype/conaffinity` 5), no proxy | — | ticket 06 |
| collision hulls | per robot, 16 buried hulls pruned | — | ticket 07 |
| ball ↔ kicker/chipper plates | μ 0.4 (geom-mixed with the ball's 0.35) | DELIBERATE APPROXIMATION | no source |
| ball ↔ dribbler | μ 2.5, `solref="0.008 1"` | μ: DELIBERATE APPROXIMATION; solref: VENDOR (derived) | μ above 1 so the bar can hold the ball, no source measures it against a golf ball. solref from the bar's PU 65A sleeve (RobôCIn ETDP 2026): Shore→modulus and Hertz give ~0.2 mm of sink at 3 N, i.e. timeconst 7-9 ms critically damped (ticket 22, research 03 §5.5) |
| walls, bumpers | `solref="0.004 1"` (the ball has its own pairs with the walls) | DELIBERATE APPROXIMATION | neither this nor MuJoCo's `0.02 1` has a source |
| robot ↔ robot, robot ↔ wall μ | 0.5, 0.6 | DELIBERATE APPROXIMATION | no source |
| `impratio` | 1 | DELIBERATE APPROXIMATION | no ODE/rSim counterpart; revisit if slip turns abrupt |
| dribbler bar speed | 314 rad/s at `a = 1` | DELIBERATE APPROXIMATION | assumed, not a firmware setpoint (see Dribbler) |
| plunger lockout | 250 ms | DELIBERATE APPROXIMATION | no capacitor recharge time is known |

## Multi-robot

`scene.py` attaches the one-robot MJCF N times with `mujoco.MjSpec.attach(prefix=)`, which
namespaces every body, joint, actuator, sensor and geom (`b0_robot`, `b0_vel_x`,
`b0_r_col_barDribbler`, ...). The parent is `scene.xml` itself with its one robot
**deleted**, so the `<option>` block, the `<visual>` block, the lights, the cameras, the
ball and its floor and wall contact pairs are exactly what the model ships with — there is no
second copy of those numbers to drift out of sync. `MjSpec.delete` on the robot body
cascades to its actuators, its sensors and its two dribbler contact pairs.

Those dribbler pairs — the ones the delete removes; the floor and wall pairs stay — are then
re-emitted once per robot, copied off the originals **before** the delete rather than retyped. Skipping the step compiles fine and silently discards the
ball/bar contact tuning: MuJoCo would fall back to the elementwise max of the two geoms' own
friction, which is not what the model was tuned against.

The compiled `MjModel` is cached per `(n_blue, n_yellow)` and shared by every env of
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

`render_mode="human"` opens a MuJoCo Studio viewer (`mujoco.studio`, `src/mjsoccer/viewer.py`)
and syncs it once per `step`. The viewer is a spectator: it gets a copy of the state, and
whatever it hands back from `sync` (edits made in its UI) is ignored. Closing the window
does not end the episode; the next `render()` opens a new one.

The `viewer_gfx` kwarg picks how it draws:

- `opengl_headless` (default): a native window. Filament renders to a texture and SDL
  blits it, the one native mode that runs under Wayland here.
- `web`: no window and no GPU on the host. The URL is printed on launch, and the browser
  draws the scene (e.g. over `ssh -L`). **It listens on all interfaces with no
  authentication**, so use it only on a trusted network or behind a tunnel. In mujoco 4.0.0
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

That script builds its own smaller renderer (480x270 by default) rather than the 720x1280
one `BaseEnv.render` hardcodes, and drives the clips with the repo's scripted
`goto_ball_policy` so the documentation shows a robot doing the task rather than a robot
twitching at random. `pillow` is a dev-only dependency and nothing in `mjsoccer` imports it.

## Timing

The physics step is 1/960 s (~1.04 ms) and the control step is a fixed 25 ms (40 Hz,
`decimation = 24`, `dt = 0.025` s), rSoccer's step and the rate the navigation controller
was written for. Every step count quoted in these docs is in control steps:
`max_steps = round(horizon_s / dt)`. A policy decides every `repeat_action` control steps
(kwarg, default 1); one `step` call is one decision, its rewards summed over the control
steps and cut short when the episode ends.

**What the rates mean.** The control step is the controller's and the firmware's tick; the
decision (`repeat_action` control steps) is the policy's rate. The 24 physics steps inside a
control step stand for the integration: the chassis velocity servos act on every one of
them. So `repeat_action` is a policy setting, and the timestep is a resolution setting.

**Seconds are the contract.** Horizons, the 250 ms plunger lock and the 50 ms possession
grace are declared in seconds and converted. Per-step rewards are rSoccer's, per control
step, and match it at 40 Hz. Horizons are rSoccer's (robocin/rSoccer).
`test_horizon_is_seconds_at_exactly_40_hz` pins each env's step count, so a change of rate
is never silent.

**Why 1/960 s.** The contact time constants are 4-6 ms (`solref`), and 2 ms resolved them in
two steps. Measured against a 0.25 ms reference, 2 ms carried 11-24 % errors in possession
retention and chip speed, and possession scattered 69-95 of 120. At ~1 ms they are within
5 %, for half the throughput. Numbers in
`.scratch/physics-fidelity/candidate-13/`.
