# Physics

How the robot, the ball and the scene are modelled, and where each number comes from. The
action and observation contracts are in [`index.md`](index.md); the checks that hold these
numbers are in [`validation.md`](validation.md).

## Chassis

The robot is a kinematic chassis: slide-x, slide-y and hinge-z joints, driven by 3 velocity
servos from a body-frame `vx, vy, omega` command (ADR 0001).

| servo | actuator | ctrlrange |
|---|---|---|
| `vx` | `vel_x` | -3.3 .. 3.3 m/s, ±3.51 N |
| `vy` | `vel_y` | -3.3 .. 3.3 m/s, ±3.51 N |
| `omega` | `vel_yaw` | -37.1 .. 37.1 rad/s, ±0.526 N·m |

**Body frame.** SSL firmware is commanded in the body frame, but the chassis joints are
world-axis slides. `BaseEnv._drive_robot` rotates the `vx, vy, omega` command through the
chassis orientation before writing it. A diagonal command can therefore ask for
`3.3 * sqrt(2)` on one world axis; MuJoCo clamps it to `ctrlrange`, as a real servo would.

**Envelope, not mechanism.** The servos carry the four per-axis limits of the
[drive envelope](#drive-envelope). Each is isotropic, at the smaller of the forward and
lateral value:

| limit | chassis | from |
|---|---|---|
| `v_max` | 3.30 m/s (`ctrlrange`) | forward speed |
| `a_max` | 1.30 m/s² (`forcerange` ±3.5129 N = m·a) | lateral, friction-bound acceleration |
| `ω_max` | 37.1 rad/s (`ctrlrange`) | the ceiling of a spin about the CoM |
| `α_max` | 55.0 rad/s² (`forcerange` ±0.5263 N·m = `I_zz`·α) | friction-limited yaw |

The limits are isotropic because the slides are world axes: a body-frame limit would need a
clip in Python, and the limits come from the model. Chassis joint damping is 0, so nothing
else eats the thrust. `test_chassis_has_the_hardware_envelope` (tests/test_envs.py) pins the
step response to 3 %.

**Not modelled:** slip, traction, weight transfer, and coupling between axes. The envelope is
a box: it can reach all three limits at once, and along a world diagonal up to `√2 ×` its
`v` and `a`. The real robot's envelope is a polytope: at full forward speed no wheel is left
over for yaw. The chassis slides on the carpet and cannot tip.

`BaseEnv.step_raw` is an internal path outside the action space, used by the envelope tests:
its slice is `(vx, vy, omega, kick, chip, dribble)`, the drive scaled by `ctrlrange` and
written straight to the servos.

## Drive envelope

The chassis limits come from a motor, not a guess (development in
`.scratch/physics-fidelity/research/01-drivetrain-envelope.md`). The robot uses a Maxon EC 45
flat 50 W (winding 251601, 24 V, `k_M` 33.5 mN·m/A) in **direct drive** on a ~60 mm wheel.
The CAD's hubs are `Direct_Transmission_*`; rSim's `0.070 N·m` / `1557 rpm` describe the
team's earlier geared robot.

**Wheel geometry.** The chassis has no wheels to command, but its envelope is derived from
the real robot's four omni-wheels, mounted at `a_i = 60, 135, 225, 300` deg from +x,
`R = 0.0796` m from the centre, rolling on an effective radius `r_eff = 0.030089` m. A
forward push projects each wheel by `|sin a|` and a sideways one by `|cos a|`, so the sums
are **`Σ|sin a| = 3.14626` forward** and **`Σ|cos a| = 2.41421` lateral**, not 4. The
force-maximising sign pattern has `Σ f_i = 0`, so it costs no yaw. Every acceleration quoted
for this robot, or for rSim's, is the per-wheel force times these factors.

| | forward (x) | lateral (y) | yaw | binding limit |
|---|---|---|---|---|
| wheel torque | ±0.078 N·m | | | `I_nom · k_M`, the motor's continuous rating (VENDOR) |
| wheel speed | ±95 rad/s | | | a design choice: 25 % under the spin tip-over rate `ω_tip = 47.7` rad/s, far under the motor's ~702 rad/s no-load (DELIBERATE APPROXIMATION) |
| acceleration | **1.95** m/s² (torque alone: 2.98) | **1.30** m/s² (torque alone: 2.30) | **55.0** rad/s² | wheel friction `μ = 0.25`: each wheel pushes along its own tangent (`μ Σ N cos / m`); lateral, no moment about the CoM puts the lighter rear wheels at their grip first; yaw `μ Σ N d / I_zz` about the CoM |
| speed | **3.30** m/s | **4.04** m/s | **37.1** rad/s | wheel speed through the matrix; yaw about the CoM, 3.45 mm behind the wheel centre (35.9 about the centre) |

## Kicker and chipper

The action asks for an exit speed (see [Action space](index.md#action-space)). The plunger
is a force actuator against its return spring. A per-plunger table, `scene.KICK_TABLE_MS`,
turns the asked-for speed into a one-step force, the way the firmware's calibration table
turns it into a pulse width. It was measured at 40 Hz on mujoco 4.0.x, float64. The 6.5 m/s
cap takes ~295-315 N. Speeds are relative to a standing robot with the ball settled in the
mouth (within ±4 % across seatings); a moving robot adds its own speed, as a real one does.
`test_kick_exit_speed` holds the table to 0.1 m/s ([validation](validation.md)).

**Firing step.** The control step a plunger fires on opens with 32 physics steps at a
quarter of the timestep (`scene.FIRING_FINE_STEPS` at `scene.FIRING_SUBSTEPS`, ~0.26 ms
each, 8.3 ms that cover the stroke and the hit) and runs the rest at the timestep. 32 is
the smallest count whose speed-vs-force curve is monotone for both plungers. A firing
control step runs 48 iterations instead of 24. That costs throughput under random actions,
which fire at every re-arm, and nothing for a policy that kicks rarely.

**Lockout.** Each shot is followed by a 250 ms lockout on that plunger, 10 control steps
(`round(0.25 / dt)`, `BaseEnv._kick_cooldown`). This is a deliberate approximation: no
capacitor recharge time is known. Held continuously, the plunger pins against its 30 mm stop
and never releases the ball, so the lockout is what makes the mechanism usable. The kicker
and chipper locks are independent: a kick cannot block a chip.

## Dribbler

`dribble` in `(0, 1]` maps to `(0, DRIBBLER_OMEGA]` = `(0, 314]` rad/s; `a <= 0` is off.
The bar is never driven backwards: a reversed bar throws the ball out of the mouth. 314 rad/s
(about 3000 rpm) is an assumption, not a team firmware setpoint. It stands for the bar's
design operating point, not the actuator's `ctrlrange` (-1500 .. 1500 rad/s), which is
about 5x larger to leave room for the speed feedforward a real controller adds. Scaling the
action by `ctrlrange` would command a bar speed the mechanism was never characterised at.

## Possession

Possession is ball/dribbler **contact**, not the arm-deflection sensor the hardware uses.
On this model a ball parked square in the mouth deflects the arm the wrong way (measured
-0.003 to -0.015 rad with the ball at (94, 0, 21) mm), so any positive threshold on
`dribbler_arm_pos` reports "no ball" forever. Contact is unambiguous, and it is what the
mjssl retention benchmarks score against.

**Contact sensors.** `scene.build` adds one MuJoCo `contact` sensor per robot and dribbler
geom (`scene.DRIBBLER_GEOMS`), ball × geom, `found` only. `RobotIds.touches_ball` reads
them from `sensordata`, and `BaseEnv._update_possession` calls it for every robot each
observation. The MJX and mjlab backends read the same sensors, so possession has one
definition on all three. `test_possession_reads_no_contact_list` (tests/test_envs.py)
empties `data.contact` under a seated ball and checks `holds()` does not change.

**Grace window.** A spinning bar does not touch the ball every step: it sticks, slips and
re-grips. So contact is held over `POSSESSION_GRACE = 0.05` s, 2 control steps
(`round(0.05 / dt)`), the criterion the mjssl retention sweeps use before calling a ball
lost. Without it, "carry the ball for N seconds" never terminates successfully.
`BaseEnv.holds(i)` and the `has_ball` observation read that window, not the raw contact.

**Idle hold.** Ball seated, robot holding its reset pose through the navigation
controller, bar at `dribble = 0.5` (157 rad/s), 120 control steps, `Dribbling-v0`, seeds 0-4
(every seed alike):

| raw ball/dribbler contact | `holds()` |
|---|---|
| 118 / 120 (98 %) | 119 / 120 (99 %) |

`test_seated_ball_is_possessed` (tests/test_envs.py) asserts at least 111/120 `holds()` on
every seed; the margin is for seed noise. ContestedPossession is excluded: its ball is seated in the
yellow robot's mouth, and that robot gets no commands, so its bar does not spin.

**Seating.** `Dribbling-v0` and `PassEndurance-v0` start with the ball in the controlled
robot's mouth; `ContestedPossession-v0` seats it in the yellow opponent's
(`seat_ball(self.robots[self.n_blue])`). `BaseEnv.seat_ball` seats it the way the mechanism
takes one: it spins the bar up, rolls a ball in at 1 m/s from 200 mm, and lets it be caught
over 0.35 s of physics. Not faster: with the measured ball inertia (β = 0.5) a ball arriving
at 2 m/s from 200 mm rebounds off the bar and rolls out of reach; at 1.0-1.5 m/s it is
caught. Placing the ball at the sphere/bar tangency instead is a launcher, not a seat: the
contact needs a few tenths of a millimetre of overlap to exist, and on a 46 g ball that
overlap is a compressed spring worth several newtons. Below ~0.1 mm MuJoCo stops generating
the contact, so no overlap works.

**Turning with the ball loses it past ~2 rad/s.** Bar at full speed (`dribble = 1`), robot
spinning in place through `step_raw`, `holds()` steps out of 120 (3 s), seeds 0-2 (every seed
alike):

| yaw | 0.5 rad/s | 1 rad/s | 2 rad/s | 3 rad/s |
|---|---|---|---|---|
| `holds()` | 102 | 96 | 78 | 7 |

`test_seated_ball_survives_a_2_rad_s_turn` asserts at least 74/120 at 2 rad/s, on the CPU
(tests/test_envs.py, and the `hold_turn` scenario in tests/test_validation.py). The turn is
chaotic; [validation](validation.md#possession-through-a-turn) covers the perturbed-copy
check every backend runs. More bar speed does not help: the faster the bar, the more of the
friction cone goes along the spin. 3 rad/s is an open fidelity gap, not a target. Published
SSL dribblers hold 3-7 rad/s, by contact area (soft or split bars) rather than friction
(`.scratch/physics-fidelity/research/03-dribbler-material.md`). A task that needs a turn in
possession has to keep it under ~2 rad/s.

## Contact and body parameters

Every number below has a source or is recorded as a deliberate approximation. The classes
rank RULEBOOK > MEASURED > VENDOR > DELIBERATE APPROXIMATION (development in
`.scratch/physics-fidelity/research/02-physical-constants.md`).

| parameter | value | class | source / reason |
|---|---|---|---|
| ball mass, radius | 0.046 kg, 0.0215 m | RULEBOOK | SSL rules §2.2 |
| kick and chip speed cap | 6.5 m/s | RULEBOOK | SSL rules §8.4.2 |
| ball inertia | `diaginertia` for β = 0.5 (`I = β m r²`), not MuJoCo's solid-sphere 0.4 | MEASURED | `models/ball.xml` |
| ball ↔ floor sliding μ | 0.35 | MEASURED | ER-Force's empirical μ_k |
| ball rolling μ | 0.0009 → 0.293 m/s² rolling deceleration | MEASURED | measured range 0.26-0.31 m/s² |
| ball ↔ floor, ball ↔ wall restitution | COR 0.48-0.55 floor, 0.49-0.53 wall (normal impacts, 0.5-6.5 m/s): `<pair>` `solref="0.012 0.225"` on the carpet, `"0.012 0.25"` on the 12 wall hulls, both in `models/ball.xml` | MEASURED | target 0.50, measured 0.47-0.56 on carpet. MuJoCo has no restitution coefficient, so the `solref` was found by measurement. 0.012 s is the shortest time constant whose COR holds within 5 % between 1/960 s and 0.5 ms from 1 m/s up (the floor at 0.5 m/s is 6 % off). MuJoCo's COR rises with speed where the real ball's falls, but stays in the band up to the 6.5 m/s kick. Oblique impacts are not calibrated: a backspinning chip's first landing reads 0.59-0.60. The goalposts stay critically damped |
| ball geom (unpaired contacts) | `solref="0.006 1"` | DELIBERATE APPROXIMATION | `models/ball.xml` |
| chassis ↔ floor | collided (floor `contype/conaffinity` 5), no proxy | — | ticket 06 |
| collision hulls | per robot, 16 buried hulls pruned | — | ticket 07 |
| ball ↔ kicker/chipper plates | μ 0.4 (geom-mixed with the ball's 0.35) | DELIBERATE APPROXIMATION | no source |
| ball ↔ dribbler | μ 2.5, `solref="0.008 1"` | μ: DELIBERATE APPROXIMATION; solref: VENDOR (derived) | μ above 1 so the bar can hold the ball; no source measures it against a golf ball. solref from the bar's PU 65A sleeve (RobôCIn ETDP 2026): Shore→modulus and Hertz give ~0.2 mm of sink at 3 N, i.e. timeconst 7-9 ms critically damped (research 03 §5.5) |
| walls, goalposts, robot hulls | `solref="0.004 1"` (the ball has its own pairs with the walls) | DELIBERATE APPROXIMATION | neither this nor MuJoCo's `0.02 1` has a source |
| robot ↔ robot, robot ↔ wall μ | 0.5, 0.6 | DELIBERATE APPROXIMATION | no source |
| `impratio` | 1 | DELIBERATE APPROXIMATION | no ODE/rSim counterpart; revisit if slip turns abrupt |
| dribbler bar speed | 314 rad/s at `a = 1` | DELIBERATE APPROXIMATION | assumed, not a firmware setpoint (see [Dribbler](#dribbler)) |
| plunger lockout | 250 ms | DELIBERATE APPROXIMATION | no capacitor recharge time is known |

**Timestep.** The contact time constants run 4-12 ms (`solref` above). The physics step is
1/960 s (~1.04 ms), so the stiffest of them spans ~4 steps. Measured against a 0.25 ms
reference, a 2 ms step carried 11-24 % errors in possession retention and chip speed; at
~1 ms they are within 5 %, for half the throughput of 2 ms. Numbers in
`.scratch/physics-fidelity/candidate-13/`. The rates are in [Timing](index.md#timing).

## Multi-robot build

`scene.py` attaches the one-robot MJCF N times with `mujoco.MjSpec.attach(prefix=)`, which
namespaces every body, joint, actuator, sensor and geom (`b0_robot`, `b0_vel_x`,
`b0_r_col_barDribbler`, ...). The parent is `scene.xml` itself with its one robot
**deleted**, so the `<option>` block, the `<visual>` block, the lights, the cameras, the
ball and its floor and wall contact pairs are exactly what the model ships with. There is
no second copy of those numbers to drift out of sync. `MjSpec.delete` on the robot body
cascades to its actuators, its sensors and its two dribbler contact pairs.

The dribbler pairs are then re-emitted once per robot, copied off the originals **before**
the delete rather than retyped; the floor and wall pairs stay. Skipping this step compiles
fine and silently discards the ball/bar contact tuning: MuJoCo would fall back to the
elementwise max of the two geoms' own friction. The per-robot possession sensors are added
in the same loop.

**Model cache.** `scene.build` is `lru_cache`d on `(n_blue, n_yellow, possession_sensors)`:
every env of the same shape shares one compiled `MjModel`, and each env owns its own
`MjData`. That is safe only while nothing mutates the model. The CPU backend never
randomizes it. GPU-backend domain randomization (`src/mjsoccer/randomize.py`) mutates the
cached model in place once per world, copies out what changed, and restores the nominal
before it returns, raise or not.
