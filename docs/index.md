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
uv run python -m pytest tests -m validation -q                  # acceptance criteria, ~2 min
uv run python scripts/validation_sweep.py --out runs/validation  # the distributions behind them, CSV
```

The `validation` suite is excluded from the default run (`pyproject.toml`). It is the
regression check for the physics acceptance criteria: no turnover, yaw p99 under `MAX_W` and
tilt p99 under 15° in every scripted scenario, in one random-action episode per env and tier,
and in five `goto_ball_policy` episodes per tier; tier-1 acceleration within 1.5× of the
derived 2.30 m/s². The pooled criteria — under 1 turnover per episode-minute, tilt p95/p99
under 4°/5° under random action — need `validation_sweep.py`, which runs the same scenarios
over seeds 0-49 and writes per-episode, summary, contact-impulse and throughput CSVs; its
docstring lists the scenarios. The
reference outputs are in `.scratch/physics-fidelity/candidate-14/`.

## Environment index

| id | robots | obs dim | action dim (tier 1, default / tier 0) | horizon (s) | terminates on |
|---|---|---|---|---|---|
| [`GoToBall-v0`](environments/go_to_ball.md) | 1 blue | 14 | 7 / 6 | 20 | within 0.13 m of the ball and facing it (cos > 0.9) |
| [`GoToBallShoot-v0`](environments/go_to_ball_shoot.md) | 1 blue | 14 | 7 / 6 | 30 | ball in the +x goal |
| [`Dribbling-v0`](environments/dribbling.md) | 1 blue | 16 | 7 / 6 | 40 | 4 waypoints carried, or ball lost for 2 s |
| [`StaticDefenders-v0`](environments/static_defenders.md) | 1 blue + 6 yellow | 62 | 7 / 6 | 30 | ball in the +x goal |
| [`ContestedPossession-v0`](environments/contested_possession.md) | 1 blue + 1 yellow | 22 | 7 / 6 | 20 | blue holds the ball for 0.5 s |
| [`PassEndurance-v0`](environments/pass_endurance.md) | 2 blue (1 controlled) | 22 | 7 / 6 | 20 | team-mate receives the pass |

Every environment also terminates when the ball leaves the field or when a controlled robot
tips over (more than 60° off vertical, `-100.0`, see [Turnover](#turnover)), and truncates at
its horizon. All six controls exactly one blue robot; the remaining robots are scripted.

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
| `omega` | — | `vel_yaw` | -35.9 .. 35.9 rad/s, ±0.527 N·m | `a * 35.9` |
| `kick` | `kicker_force` | `kicker_force` | 0 .. 500 N | pulse for `6.5 * a` m/s if `a > 0` |
| `chip` | `chipper_force` | `chipper_force` | 0 .. 500 N | pulse for `6.5 * a` m/s if `a > 0` |
| `dribble` | `dribbler_vel` | `dribbler_vel` | -1500 .. 1500 rad/s | `314 * max(a, 0)` |

Everything is in `[-1, 1]` and scaled by the actuator's own `ctrlrange`, so the bounds come
from the model rather than from a constant in the Python. The drive scale is
`max(hi, -lo)` over the actuator's range, which is symmetric, so a zero action is a zero
command rather than the midpoint of the range.

**Tier-1 wheel matrix.** What a wheel vector does to the chassis, in the body frame (x
forward, y left, ω counter-clockwise), at the measured geometry:

```
w_i = (1 / r_eff) * ( vx * sin(a_i) - vy * cos(a_i) - R * omega )

a_i   = 60, 135, 225, 300 deg       wheel1..wheel4, mount angle from +x
r_eff = 0.030089 m                  measured: axle height minus the lowest roller vertex
R     = 0.0796 m                    measured: hub mount radius
```

`w_i` is positive about each hub's own joint axis. `tasks.wheel_command(env, i, vx, vy, w)`
reads the same rolling constraint off the compiled model and returns the four actions. All
four wheels at `-1` (`-95` rad/s) asks for a pure spin of `+35.9` rad/s (kinematic; the
chassis measures ~30-32 rad/s, the rest is roller slip). **The sign convention is
the mirror of rSim's** (`sslrobot.cpp:383-386` is `-vx sin a + vy cos a + R w`, with
`r = 0.02475`, `R = 0.090`), so a wheel vector ported from rSoccer drives backwards and yaws
the wrong way (`W2` in [the audit](research/rsoccer-vs-mjsoccer.md)).

The geometry also fixes how much of the four wheels' force a translation can use. A
forward push projects each wheel by `|sin a|` and a sideways one by `|cos a|`, so the sums
are **`Σ|sin a| = 3.14626` forward** and **`Σ|cos a| = 2.41421` lateral**, not 4 (the
force-maximising sign pattern has `Σ f_i = 0`, so it costs no yaw). Every acceleration
quoted for this robot, or for rSim's, is the per-wheel force times these factors.

**Drive envelope.** The wheel bounds in the action table are a motor, not a guess (tickets
01, 04, 05, 16; development in `.scratch/physics-fidelity/research/01-drivetrain-envelope.md`).
The robot is a Maxon EC 45 flat 50 W (winding 251601, 24 V, `k_M` 33.5 mN·m/A) in **direct
drive** on a ~60 mm wheel — the CAD's hubs are `Direct_Transmission_*`, and rSim's
`0.070 N·m`/`1557 rpm` describe the team's earlier geared robot.

| | forward (x) | lateral (y) | yaw | binding limit |
|---|---|---|---|---|
| wheel torque | ±0.078 N·m | | | `I_nom · k_M`, the motor's continuous rating (VENDOR) |
| wheel speed | ±95 rad/s | | | a design choice: 25 % under the spin tip-over rate `ω_tip = 47.7` rad/s, far under the motor's ~702 rad/s no-load (DELIBERATE APPROXIMATION) |
| acceleration | **2.45** m/s² (torque alone: 2.98) | **2.30** m/s² | **55.1** rad/s² | forward and yaw by roller friction `μ = 0.25` (`μg`, `μmgR / I_zz`), lateral by torque |
| speed | **3.30** m/s | **4.04** m/s | **35.9** rad/s | wheel speed through the matrix |

The forward acceleration clears the static forward tip threshold (`g · 0.04325 / 0.05399 =
7.86` m/s²) by a factor of three. The `velocity` actuators' `kv = 0.103696` is kept: it
saturates above 0.75 rad/s of error, which approximates a firmware speed loop with a current
clamp (DELIBERATE APPROXIMATION). A constant `forcerange` cannot express an ESC's transient
overcurrent, so launch acceleration may be low.

**Tier 1 does not yet reach its derived speed.** Measured by the validation sweep, full
command from rest: 2.47 m/s² forward and 1.87 lateral (inside 1.5× of 2.30, which the
`validation` suite asserts), but 2 s later only 2.14 m/s forward and 0.43 m/s lateral
against 3.30 and 4.04. The rollers lose floor contact while driving; that is open as
ticket 18, and no acceptance criterion uses the kinematic `v_max` until it closes.

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
firmware's calibration table turns it into a pulse width. The cap takes ~285-365 N. Speeds
are relative to a standing robot with the ball settled in the mouth (within ±4 % across
seatings); a moving robot adds its own speed, as a real one does. The control step a
plunger fires on runs at a quarter of the timestep (`scene.FIRING_SUBSTEPS`, ~0.26 ms): at
2 ms the plate closed ~9 mm per step and the exit speed swung ±35 % with force, and 1 ms still
saws. That costs extra throughput under random actions, which fire at every re-arm, and
nothing for a policy that kicks rarely. Each shot is followed by a 250 ms lockout on that plunger, a deliberate
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
command, which is per-wheel on tier 1 and body-frame on tier 0. `has_ball` is `1.0` while
the robot is inside the possession grace window (see below) and `0.0` otherwise.

Chassis tilt is not observed: a robot past 60° ends the episode instead (see
[Turnover](#turnover)), and below that the tier-1 chassis stays within a few degrees.

Every entry is normalised by a constant from `base.py` or `scene.py`:

| quantity | divisor | value |
|---|---|---|
| ball x, robot x | `scene.FIELD_HALF_X` | 6.0 m |
| ball y, robot y | `scene.FIELD_HALF_Y` | 4.5 m |
| ball z above the carpet (`z - scene.FLOOR_Z`) | `scene.GOAL_HEIGHT` | 0.155 m |
| ball vx, vy, vz | `base.MAX_BALL_V` | 8.0 m/s (kicks are capped at 6.5 off a standing robot) |
| robot vx, vy | `base.MAX_V` | 3.3 m/s (the tier-0 chassis servo ctrlrange) |
| robot omega | `base.MAX_W` | 35.9 rad/s (likewise; the wheels' kinematic yaw ceiling) |

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

Measured under random action, seeds 0-19, six envs: tier 1 robot `|v|` peaks at 0.45 and
`|omega|` at 0.14 of their divisors; tier 0 at 0.42 and 0.45. Under its own full command
the tier-1 chassis spins at ~30 rad/s (< `MAX_W`, pinned by `test_full_spin_stays_under_max_w`;
the figure creeps up as the timestep shrinks, 31.8 at 0.25 ms, which is why `MAX_W` is the
derived 35.9 and not a measured value) and drives at ~2 m/s. `MAX_V` and `MAX_W` are the tier-0 servo `ctrlrange`, pinned by
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
anything where the number of samples matters more than their fidelity. Measured by
`scripts/validation_sweep.py --scenario random`, one process, random actions, at the
1/960 s default (`.scratch/physics-fidelity/candidate-14/throughput.csv`):

| env | tier 0 | tier 1 |
|---|---|---|
| `GoToBall-v0` | ~1 500 steps/s | ~320 steps/s |
| `Dribbling-v0` | ~1 200 steps/s | ~280 steps/s |
| `StaticDefenders-v0` (7 robots) | ~360 steps/s | ~54 steps/s |

The 1/960 s step costs about half the throughput of the old 2 ms one: the price of
resolving the contacts (see Timing). The validation criterion is "within 10 % of that file on the same machine".

**Tier 0 has tier 1's envelope, not its mechanism.** One action head should mean one thing
on both tiers, so tier 0's servos carry the four per-axis limits of the derived tier-1
envelope above, each isotropic at the smaller of tier 1's forward and lateral value:

| limit | tier 0 | from |
|---|---|---|
| `v_max` | 3.30 m/s (`ctrlrange`) | tier-1 forward speed |
| `a_max` | 2.30 m/s² (`forcerange` ±6.1918 N = m·a) | tier-1 lateral acceleration |
| `ω_max` | 35.9 rad/s (`ctrlrange`) | the wheels' kinematic yaw ceiling |
| `α_max` | 55.1 rad/s² (`forcerange` ±0.5273 N·m = `I_zz`·α) | tier-1 friction-limited yaw |

Isotropic because tier 0's slides are world axes: a body-frame limit would need a clip in
the Python, and the limits come from the model. The chassis joint damping is 0, so nothing
else eats the thrust. `test_tier0_has_the_tier1_envelope` pins the step response to 3 %.

**What is deliberately not matched**, and is what tier 1 is for: slip, traction, weight
transfer, and the coupling between axes. Tier 0's envelope is a box — it can reach all three
limits at once, and along a world diagonal up to `√2 ×` its `v` and `a`. Tier 1's is a
polytope: at full forward speed no wheel is left over for yaw.

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
| `ContestedPossession-v0` | 114-118 / 120 | 112-118 / 120 |
| `Dribbling-v0` | 119 / 120 | 115-117 / 120 |

Tier 0 holds a ball it is not even trying to hold; on `Dribbling-v0` it drops it for a single
step. Tier 1 loses about 3% there, because real wheel slip lets the chassis recoil off the
ball it is carrying — a rigid kinematic chassis has nothing to recoil with. The grace window
absorbs most of the difference, so `holds` stays usable on both, but any retention number,
ball-lost rate or possession-gated reward is tier-dependent and comparing one across tiers
is comparing two different mechanisms.

These numbers are from after ticket 05's contact changes (roller μ 0.25, ball β 0.5, the
motor-envelope wheel actuators of ticket 04), ticket 12's tier-0 envelope and ticket 13's
1/960 s step. On the old 2 ms step tier-0 `ContestedPossession-v0` scattered 69-95 and
looked chaotic; it was the step under-resolving the ball/dribbler contact (120 at 0.5 ms,
87 at 2 ms, 0 at 4 ms). Before ticket 05 `Dribbling-v0` held 120 / 111-112
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

**Turning with the ball loses it.** With the bar at 157 rad/s (`dribble = 0.5`) a robot
spinning in place at 10 rad/s or more drops a seated ball within 5 control steps; standing
still it holds 117 of 120 (the validation sweep's `dribble_turn` scenario). A task that
needs a turn in possession has to do it slowly, or with more bar speed.

## Turnover

A controlled robot whose up-axis is more than 60° off vertical (`base.TIPPED_COS = 0.5`)
ends the episode. That step pays `base.TIPOVER_PENALTY = 100.0` off the task reward and sets
`info["tipped"]`, plus `info["tilt_deg"]` (the worst controlled robot's tilt) and
`info["tip_step"]` (control steps since reset, so time-to-tip is `tip_step * dt`). Scripted
robots never trigger it, and tier 0's chassis cannot tilt at all.

60° is not arbitrary: it sits about 1° from the measured static recovery boundary. A robot
rested on the carpet at tilt θ about a horizontal axis, controls at zero, stands back up
below 59-67° depending on the axis azimuth, median 61°
(`.scratch/physics-fidelity/candidate-10/recover.txt`); a robot falling through 60° carries
momentum towards the floor, so the dynamic boundary is lower still. Past it the robot only
lies there until truncation, so ending the episode loses nothing.

It is a safety net, not a mechanism a policy meets. Under random action, tier 1, seeds 0-49,
all six envs: **0 turnovers in 101 minutes of episode**, tilt p99 3.6-3.9° and max 7-8°.
Before the drivetrain envelope above, the same sweep tipped 35.9 times a minute — from the
old ±3 N·m wheel torque, not from spin. The chassis collides with the carpet (its hulls
first touch it from 9.5° of tilt), so a robot on its side lies on the floor instead of
sinking through it. There is no righting moment and none is planned; that reopens only if a
trained policy shows a non-zero `tipped` rate.

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
| ball ↔ floor, ball ↔ wall restitution | **none yet** (`solref` damping ratio 1); target COR 0.50 | MEASURED target | measured 0.47-0.56 on carpet; the `solref` that reaches it is open as ticket 17. **This is the model's largest open contact gap**: a landing chip or a missed shot dies instead of bouncing |
| roller ↔ floor μ | 0.25, set on both the roller and the floor (MuJoCo takes the elementwise max) | MEASURED | Williams et al., IEEE T-RA 18(3), 2002, tilt table — the only published omni-roller-on-carpet value |
| chassis ↔ floor | collided (floor `contype/conaffinity` 5), no proxy | — | ticket 06 |
| collision hulls | 193 per robot (16 buried hulls pruned) | — | ticket 07; the wheel hub, not a roller, is what deflects a ball rolling under the robot |
| ball ↔ kicker/chipper plates | μ 0.4 (geom-mixed with the ball's 0.35) | DELIBERATE APPROXIMATION | no source |
| ball ↔ dribbler | μ 2.5, `solref="0.004 0.5"` | DELIBERATE APPROXIMATION | above 1 so the bar can hold the ball; no physical retention model |
| walls, bumpers | `solref="0.004 1"` | DELIBERATE APPROXIMATION | neither this nor MuJoCo's `0.02 1` has a source |
| robot ↔ robot, robot ↔ wall μ | 0.5, 0.6 | DELIBERATE APPROXIMATION | no source |
| `impratio` | 1 | DELIBERATE APPROXIMATION | no ODE/rSim counterpart; revisit if slip turns abrupt |
| dribbler bar speed | 314 rad/s at `a = 1` | DELIBERATE APPROXIMATION | assumed, not a firmware setpoint (see Dribbler) |
| plunger lockout | 250 ms | DELIBERATE APPROXIMATION | no capacitor recharge time is known |

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

The physics step is 1/960 s (~1.04 ms) and the default `control_hz=60` is exact:
`decimation = 16`, `dt = 1/60` s. A `control_hz` that is not a whole number of physics steps
(anything that does not divide 960 Hz: 50, 62.5, ...) raises `ValueError` instead of being
rounded to a neighbour. Every step count quoted in these docs is at 60 Hz:
`max_steps = round(horizon_s / dt)`.

**What the two rates mean.** The control step is the *policy's* decision rate, the ~60 Hz at
which SSL vision feeds the AI. The 16 physics steps inside it stand for the firmware and the
integration: the wheel and chassis velocity servos act on every one of them. So `control_hz`
is a fidelity setting (how often the policy decides), and the timestep is a resolution
setting.

**Seconds are the contract.** Horizons, the 250 ms plunger lock and the 50 ms possession
grace are declared in seconds and converted, so episodes last the same physical time at any
rate. Per-step rewards are per 1/60 s step. `test_horizon_is_seconds_at_exactly_60_hz` pins
each env's step count, so a change of rate is never silent.

**Why 1/960 s.** The contact time constants are 4-6 ms (`solref`), and 2 ms resolved them in
two steps. Measured against a 0.25 ms reference, 2 ms carried 12-24 % errors on tier 1 —
possession retention -24 %, chip speed -11 %, forward acceleration -12 % — and tier 0's
possession scattered 69-95 of 120. At ~1 ms all three are within 5 %, for half the
throughput. What stays dt-sensitive at ~1 ms is the tier-1 full spin (-5.6 %) and top
forward speed (-12 %), both through the roller contact loss tracked by ticket 18. Numbers in
`.scratch/physics-fidelity/candidate-13/`.
