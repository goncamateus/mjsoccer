# 01 — Drivetrain envelope: what can this robot's motors actually do?

Ticket: `.scratch/physics-fidelity/issues/01-drivetrain-envelope.md`. Resolved 2026-09-08.
Vendor data from maxongroup.com/maxongroup.us product pages (fetched 2026-09-08); rules from the
RoboCup SSL rule book source on `master` (fetched 2026-09-08); everything else cited to
`path:LINE` in this repo or in `/tmp/rsim-upstream` (`b413932`).

Measured quantities taken as given from `docs/research/rsoccer-vs-mjsoccer.md` §4/§6: total mass
`m = 2.69207` kg, `r_eff = 30.089` mm, `R = 0.0796` m, wheels at 60/135/225/300°, CoM 53.99 mm
above the carpet with a 3.459 mm lateral offset, forward static tip threshold 7.86 m/s²,
`ω_tip = 47.7` rad/s, friction budget `μ·m·g = 26.4` N at `μ = 1.0`. All were re-checked
arithmetically here and all reproduce (7.8585 m/s², 47.665 rad/s, 26.409 N).

## Recommended values

| declaration | today (`models/robot_rollers.xml:827-830`) | recommended | basis |
|---|---|---|---|
| `forcerange` (N·m per wheel) | `-3 3` | **`-0.078 0.078`** | EC 45 flat 50 W design operating point, direct drive (§2, §3) |
| `ctrlrange` (rad/s per wheel) | `-200 200` | **`-95 95`** | `ω_tip` with a stated 25 % design margin (§3) |
| `kv` | `0.103696` | unchanged | saturates above 0.75 rad/s of error, i.e. torque-limited in every transient (§3) |

| resulting quantity | today | recommended | threshold | verdict |
|---|---|---|---|---|
| `a_max` forward (+x) | 9.81 (friction-clipped) | **2.98 m/s²** | 7.86 m/s² static tip | clears, 62 % below |
| `a_max` lateral (+y) | 9.81 (friction-clipped) | 2.30 m/s² | — | — |
| ground force demanded, forward | 1255 N vs a 26.4 N budget | 8.16 N = 31 % of budget | 26.4 N | torque-limited, not friction-limited |
| wheel surface speed | 6.02 m/s | **2.86 m/s** | — | — |
| `v_max` forward / lateral | 6.95 / 8.51 m/s | **3.30 / 4.04 m/s** | — | in the real SSL band (§5) |
| `ω_max` (uniform wheel command) | 75.6 rad/s (59.4 measured) | **35.9 rad/s** | 47.7 rad/s dynamic tip | clears, 25 % below |

The headline change is qualitative: `forcerange="-3 3"` makes the wheel a friction-limited
infinite-torque servo, `forcerange="-0.078 0.078"` makes it a motor.

## 1. The Maxon EC 45 flat 50 W datasheet figures

maxon publishes **five** catalogue numbers under the name "EC 45 flat … 50 W", in three distinct
windings across three frame diameters. All values are maxon's own product pages.

| part | frame | U_nom | no-load speed | nominal speed | nominal torque (max. continuous) | nominal current (max. continuous) | stall torque | stall / starting current | torque constant | source |
|---|---|---|---|---|---|---|---|---|---|---|
| 251601 | Ø42.8 mm | 24 V | 6710 rpm | 5240 rpm | 83.4 mN·m | 2.33 A | 780 mN·m | 23.3 A | 33.5 mN·m/A | [maxon 251601](https://www.maxongroup.com/maxon/view/product/251601) |
| 651607 | Ø43.5 mm | 24 V | 6250 rpm | 5170 rpm | 91.1 mN·m | 2.52 A | 918 mN·m | 25.5 A | 36 mN·m/A | [maxon 651607](https://www.maxongroup.com/maxon/view/product/651607) |
| 339286 | Ø42.8 mm | 24 V | 4730 rpm | 3480 rpm | 69.6 mN·m | 1.41 A | 402 mN·m | 8.47 A | 47.5 mN·m/A | [maxon 339286](https://www.maxongroup.com/maxon/view/product/339286) |
| 400107 | Ø45 mm, with cable | 24 V | 4730 rpm | 3480 rpm | 69.6 mN·m | 1.41 A | 402 mN·m | 8.47 A | 47.5 mN·m/A | [maxon 400107](https://www.maxongroup.com/maxon/view/product/400107) |
| 339287 | Ø42.8 mm | 36 V | 3360 rpm | 2360 rpm | 90.5 mN·m | 0.828 A | 484 mN·m | 4.81 A | 101 mN·m/A | [maxon 339287](https://www.maxongroup.us/maxon/view/product/motor/ecmotor/ecflat/ecflat45/339287) |

Supporting characteristics for 251601, the variant used below: no-load current 185 mA, terminal
resistance 1.03 Ω, speed constant 285 rpm/V, speed/torque gradient 8.77 rpm/mN·m, max. efficiency
83 %, rotor inertia 135 g·cm²
([maxon 251601](https://www.maxongroup.com/maxon/view/product/251601)).

**The current limit "50 W" implies.** "50 W" is the nominal *output* power class, not a separate
limit; the limit is the nominal current. For 251601 the continuous electrical limit is 2.33 A,
which through the torque constant is

```
τ_cont = I_nom · k_M = 2.33 A · 33.5 mN·m/A = 78.1 mN·m
```

and the mechanical output at the nominal point is
`0.0834 N·m · 5240 rpm · 2π/60 = 0.0834 · 548.7 = 45.8 W`; for 651607,
`0.0911 · 541.4 = 49.3 W`. So the class label checks out, and the datasheet's own
"nominal torque 83.4 mN·m" is about 7 % above `I_nom · k_M`. **78.1 mN·m is the figure to build
on**, because it is the one the driver's current limit actually enforces.

Two facts worth carrying forward: `I_nom · k_M = 78.1` mN·m is also **exactly 10.0 % of the
780 mN·m stall torque**, and 10 % of stall is where the team's own hardware paper says these
motors are operated (§2). Three independent routes land on ~78 mN·m.

**Assumption (labelled).** Which of the five variants this robot carries is not recorded anywhere
in the repo. 251601 is assumed. The torque band across the five is 40.2–91.8 mN·m at 10 % of
stall and 69.6–91.1 mN·m at nominal, which moves `a_max` between 1.54 and 3.51 m/s² (§3) — a
factor of 2.3. The winding choice is the largest single unresolved input to this ticket.

## 2. Through 18:60 to the wheel, and what rSim's 0.070 N·m is

18 teeth on the motor pinion driving 60 on the wheel is a reduction of `60/18 = 3.3333:1`, so an
ideal (100 %-efficient, backlash-free) stage gives

```
τ_wheel = τ_motor · 3.3333          ω_wheel = ω_motor / 3.3333
```

Both directions applied to 251601:

| quantity at the motor | value | at the wheel through 18:60 |
|---|---|---|
| `I_nom · k_M` (continuous) | 78.1 mN·m | 260.3 mN·m |
| nominal torque | 83.4 mN·m | 278.0 mN·m |
| stall torque | 780 mN·m | 2600 mN·m |
| nominal speed | 5240 rpm | 1572 rpm = 164.6 rad/s |
| no-load speed | 6710 rpm | 2013 rpm = 210.8 rad/s |

### 2.1 rSim's `wheelMotorMaxRPM = 1557` is the motor's nominal speed

`1557 rpm · 3.3333 = 5190 rpm` at the motor. That sits inside the nominal-speed band of the two
24 V high-speed 50 W windings — 5170 rpm (651607) and 5240 rpm (251601) — within 1 %, and is
77 % of 251601's 6710 rpm no-load speed. So `wheelMotorMaxRPM`
(`/tmp/rsim-upstream/src/robosim/sslconfig.h:160`) is the motor's **nominal (rated-load) speed**
geared down, not the no-load speed. The residual 1 % is consistent with a different catalogue
edition; it is not resolved here and does not matter.

### 2.2 rSim's `wheelMotorMaxTorque = 0.070` is a deliberate ~3.7× derate

Working backwards through the same stage:

```
τ_motor = 0.070 N·m / 3.3333 = 0.0210 N·m = 21.0 mN·m
        = 26.9 % of I_nom·k_M (78.1)   = 25.2 % of nominal torque (83.4)   = 2.7 % of stall (780)
implied current = 21.0 / 33.5 = 0.627 A, i.e. 27 % of the 2.33 A continuous limit
```

It is neither nominal nor stall nor a thermal figure. It is a deliberate cut to roughly a quarter
of the continuous rating, which is exactly what the comment "reduced to 3m/s2 accel"
(`sslconfig.h:159`) claims. What it was reduced *from* is 260 mN·m at the wheel — a 3.72× derate.

### 2.3 Does the "3 m/s²" comment check out? Yes — as the weak-axis figure

The four wheels sit at 60/135/225/300°, so their ground drive directions are tangential and no
direction gets all four at unit projection. With `α_i` the mount angles, the reachable ground
force from a per-wheel force `F` is `Σ|sin α_i| = 3.14626` along +x and `Σ|cos α_i| = 2.41421`
along +y (both with zero net yaw moment: the sign pattern that maximises force also cancels
`Σ f_i`). On rSim's own numbers (`m = 2.42` kg, `wheelRadius = 0.02475`,
`sslconfig.h:153,155-157`, `sslrobot.cpp:50-56,63-68`):

```
F_wheel = 0.070 / 0.02475            = 2.828 N
forward:  3.14626 · 2.828 / 2.42     = 3.68 m/s²   (3.63 including wheel spin-up inertia)
lateral:  2.41421 · 2.828 / 2.42     = 2.82 m/s²   (2.79 including wheel spin-up inertia)
naive "all four wheels":  4 · 2.828 / 2.42 = 4.67 m/s²
```

So: the comment's "3 m/s²" is right to 6 % **in the weak (lateral) direction** and understates the
forward figure by 19 %. The 4.67 m/s² quoted in `W3` of `docs/research/rsoccer-vs-mjsoccer.md`
is **not achievable in any direction** — it assumes a projection factor of 4 where the geometry
caps it at 3.146 — and overstates rSim's real forward ceiling by 27 %. `W3`'s conclusion survives
(rSim is torque-limited: 8.90 N of forward force against its own
`0.8 · 2.42 · 9.81 = 19.0` N tangential friction budget), but its number should be corrected to
3.68 m/s² forward / 2.82 m/s² lateral.

### 2.4 The 18:60 assumption does not hold for this CAD

Two pieces of evidence say this robot has **no wheel reduction stage**.

1. **The model's own mesh names.** Each wheel hub's visual and collision meshes are
   `Direct_Transmission_new_Simplify_2` (`models/robot_rollers.xml:300-301`, and the same mesh on
   `wheel2`/`wheel3`/`wheel4`). The only gear meshes in the file — `gear_z18_21101800_di3` (an
   18-tooth gear) and `Middle_Gear_Simplify_1` — are attached to the **chassis** at
   `x = 0.0634, y = -0.04325` next to the dribbler and are named
   `r_vis_gearMotorDribbler` / `r_vis_gearMiddleDribbler` (`models/robot_rollers.xml:164-165`,
   `250-257`). The 18-tooth pinion in this CAD drives the dribbler, not a wheel.
2. **The team's own hardware paper.** RobôCIn's 2024 ETDP states that the v2022 wheels were
   45 mm in diameter with an "embedded gearing system (3.3:1)", that the v2024 robots moved to
   **direct drive with no gears**, and that for this the wheel grew to "a total diameter of
   62 mm" ([2024 ETDP RoboCIn](https://ssl.robocup.org/wp-content/uploads/2024/04/2024_ETDP_RoboCIn.pdf),
   §1.2 *Direct Drive*). This model's measured `r_eff = 30.089` mm is a 60.18 mm wheel — the
   direct-drive generation, not the 45 mm geared one. The same section states the design
   operating point: "The 50W Maxon motors through direct drive work at a maximum operating point
   around 10 percent of the stall torque, near maximum efficiency." The CAD's dribbler mesh is
   named `Dribbler_Robocup2025v3` (`models/robot_rollers.xml:159`), i.e. a 2025-generation robot.

3. **Arithmetic agrees.** If 18:60 *were* present on this wheel, the continuous wheel torque
   would be 260 mN·m = 8.65 N per wheel = **27.2 N of forward ground force against a 26.4 N
   friction budget**. The robot would still be friction-limited at 9.81 m/s², still above its own
   7.86 m/s² tip threshold, and the torque cap would still be behaviourally inert. A 3.33:1 stage
   and a 30 mm wheel are not a combination anyone would build.

This supersedes standing constraint 2 of `.scratch/physics-fidelity/map.md` ("Motor envelope
default: Maxon EC45 flat 50 W, 18:60 gearing. No real drivetrain spec was available"): a
drivetrain spec **is** available, it is the team's own ETDP, and it says direct drive. rSim's
0.070 N·m and 1557 rpm describe the *previous, geared* robot generation, which is why they do not
transfer as numbers.

## 3. `a_max`, `v_max`, `ω_max` and the `forcerange` / `ctrlrange` pair

Note first that `forcerange` on a MuJoCo `velocity` actuator attached to the hub joint is a
**wheel** torque. The gearing question of §2 therefore only decides *which* wheel torque the
motor can hold; it never appears in the declaration itself.

### 3.1 Torque → ground force → acceleration

Direct drive, so `τ_wheel = τ_motor = 78 mN·m` (§1: `I_nom · k_M = 78.1` mN·m = 10.0 % of stall,
the operating point the ETDP names). Rounding down to 0.078 N·m:

```
F_wheel = τ / r_eff        = 0.078 / 0.030089            = 2.5923 N
F_x,total = 3.14626 · F    = 8.1561 N     (Σ|sin α_i|, zero net yaw)
F_y,total = 2.41421 · F    = 6.2584 N     (Σ|cos α_i|, zero net yaw)
a_x = F_x / m              = 8.1561 / 2.69207            = 3.029 m/s²
a_y = F_y / m              = 6.2584 / 2.69207            = 2.325 m/s²
```

Including the wheels' own spin-up (hub `Iyy = 5.93133e-6` plus `armature="1e-5"`,
`models/robot_rollers.xml:298-299`, so `I = 1.5931e-5` kg·m²; the added effective mass is
`I·Σsin²α_i/r_eff² = 1.5931e-5 · 2.5 / 0.030089² = 0.0440` kg, +1.63 %):

```
a_x = 8.1561 / 2.73606 = 2.981 m/s²        a_y = 6.2584 / 2.71847 = 2.302 m/s²
```

Sanity checks on the recommendation: 8.16 N of forward ground force is **31 % of the 26.4 N
friction budget**, so the wheel is torque-limited and slip is a rare event rather than the
operating regime; and 0.078 N·m is 100 % of the continuous current limit and 10 % of stall, so
nothing thermal is being violated. It is a hard ceiling, not a peak-plus-headroom figure — real
brushless drives can pull several times `I_nom` briefly, which this model does not represent
(noted as a deliberate approximation).

### 3.2 Speed → surface speed → body speed and yaw

For a uniform wheel command `w` on all four wheels the measured wheel matrix
(`docs/research/rsoccer-vs-mjsoccer.md` §6) reduces to `ω = w · r_eff / R` (sign inverted, `W2`),
and for pure translation the binding wheel is the one whose drive direction is closest to the
motion: `w = v_x · sin 60° / r_eff` forward, `w = v_y · cos 45° / r_eff` lateral. So

```
surface speed = w · r_eff
v_x,max       = w · r_eff / 0.866025
v_y,max       = w · r_eff / 0.707107
ω_max         = w · r_eff / R = w · 0.030089 / 0.0796 = 0.37800 · w
```

Direct drive means the motor imposes essentially no speed ceiling on this wheel: 251601's nominal
5240 rpm **is** the wheel speed, i.e. 548.7 rad/s = 16.5 m/s of surface speed. The speed cap is
therefore a design choice, and the only physical anchor available is `ω_tip = 47.7` rad/s, which
`ω = 0.378 w` reaches at `w = 126.2` rad/s:

| `ctrlrange` (rad/s) | `ω_max` (rad/s) | margin under 47.7 | surface (m/s) | `v_x` (m/s) | `v_y` (m/s) | motor rpm |
|---|---|---|---|---|---|---|
| 86 | 32.5 | 32 % | 2.59 | 2.99 | 3.66 | 821 |
| **95 (recommended)** | **35.9** | **25 %** | **2.86** | **3.30** | **4.04** | **907** |
| 110 | 41.6 | 13 % | 3.31 | 3.82 | 4.68 | 1050 |
| 126.2 | 47.7 | 0 % | 3.80 | 4.39 | 5.37 | 1205 |
| 164.6 (18:60 nominal, hypothetical) | 62.2 | −30 % | 4.95 | 5.72 | 7.00 | — |
| 200 (today) | 75.6 | −59 % | 6.02 | 6.95 | 8.51 | 1910 |

**A 25 % margin under `ω_tip` is a stated design choice, not a hardware number.** It is the one
place in this ticket where a number is chosen rather than derived. The justification is that the
tip prediction is a static-moment balance and the measured onset brackets it rather than matching
it (42.1 rad/s of yaw already tilts the chassis 6.0° at a 120 rad/s command,
`docs/research/rsoccer-vs-mjsoccer.md` §4), so the envelope should not be set at the threshold.

`±95` rad/s uses only 17 % of the motor's rated speed. The honest reading is that a direct-drive
30 mm wheel is speed-rich and torque-poor: acceleration is hardware-limited, top speed is not,
and yaw is limited by the chassis rather than by the drivetrain. If the full straight-line
capability is ever wanted, the alternative is to leave `ctrlrange` at the motor limit and cap
**yaw** in the action layer instead — which is what rSoccer's environments do (`max_w = 10`
rad/s, `docs/research/rsoccer-vs-mjsoccer.md` §7). That is an action-space decision, not an
actuator one, and is left to the tickets that own the action space.

### 3.3 The pair

```xml
<velocity name="wheelN_vel" joint="wheelN_spin" kv="0.103696"
          ctrlrange="-95 95" forcerange="-0.078 0.078" />
```

with `kv` unchanged: the servo saturates above `0.078 / 0.103696 = 0.75` rad/s of tracking error,
so it behaves as a torque-limited motor throughout every transient and only enters its
proportional region within 0.75 rad/s of the target. (Today it saturates above 28.9 rad/s, which
is why it reads as an ideal servo.) Whether `kv` should be re-derived from the motor's
speed/torque gradient is a separate question and is not answered here.

Time to reach `v_x,max` at `a_x`: `3.30 / 2.98 = 1.11` s. Full-envelope numbers:

| | forward (+x) | lateral (+y) | yaw |
|---|---|---|---|
| `a_max` | 2.98 m/s² | 2.30 m/s² | — |
| `v_max` | 3.30 m/s | 4.04 m/s | 35.9 rad/s |

Per map constraint 4, tier 0 should be given the same envelope: `2.69207 · 2.98 = 8.02` N on the
x/y slide actuators (against today's `±19.1261`, `models/robot_kinematic.xml:389-390`) and a yaw
`ctrlrange` of 35.9 rad/s (against today's 30). Deriving tier 0's yaw *force* range needs the
robot's `Izz` and is out of this ticket's scope.

## 4. Does it clear the two thresholds?

| threshold | value | recommended envelope | verdict |
|---|---|---|---|
| forward static tip | 7.86 m/s² | 2.98 m/s² | **clears — 4.88 m/s² of margin, 62 % below** |
| dynamic yaw tip `ω_tip` | 47.7 rad/s | 35.9 rad/s | **clears — 11.8 rad/s of margin, 25 % below** |
| friction budget `μ·m·g` | 26.4 N | 8.16 N demanded | clears — 31 % of budget used |

Both cleared, and the ratio to the tip threshold (0.379 forward) is *more* conservative than
rSim's own (3.68 / 6.28 = 0.586, or 2.82 / 6.28 = 0.449 on the weak axis). The robot can no
longer tip itself by accelerating in a straight line, and can no longer command a yaw rate that
tips it. It also cannot reach 47.7 rad/s of yaw *transiently* from a saturated command, because
the yaw rate is now capped kinematically rather than by the torque available.

Two caveats on the yaw side. First, `ω_max` above is the kinematic prediction; the measured value
runs slightly below it (a 120 rad/s command predicts 45.4 rad/s and measures 42.1), so the true
margin is a little larger. Second, `ω_tip = 47.7` rad/s depends on the 3.459 mm CoM offset, which
is a CAD artifact of unknown fidelity; if that offset is corrected or found to be an asymmetry
the real robot does not have, `ω_tip` moves and the 95 rad/s recommendation should be revisited.
The torque recommendation does not depend on it.

## 5. Do SSL rules or Division A practice cap speed, acceleration or motor power?

**The rule book caps none of the three.** Chapter 3 *Robots* → *Hardware And Software
Constraints* covers safety, shape, colour, dribbling device, vision pattern, radio and autonomy,
and contains **no** limit on velocity, acceleration, wheel speed, motor power, battery capacity or
drivetrain of any kind
([chapters/robots.adoc](https://github.com/RoboCup-SSL/ssl-rules/blob/master/chapters/robots.adoc)).
The only dimensional constraint that touches the drivetrain indirectly is *Shape*: "A robot must
fit inside a 0.18 meters wide and 0.15 meters high cylinder at any point in time" — which bounds
wheel diameter and hence the torque/speed trade, and nothing else.

What the rule book does limit:

| rule | limit | where |
|---|---|---|
| Ball Speed (non-stopping foul) | "A robot must not accelerate the ball faster than 6.5 meters per second in 3D space" | [chapters/offenses.adoc](https://github.com/RoboCup-SSL/ssl-rules/blob/master/chapters/offenses.adoc) |
| Robot Stop Speed | "A robot must not move faster than 1.5 meters per second during stop", 2 s grace, not applicable during ball placement | same |
| Crashing | at contact between opposing robots, the closing-speed component along the line joining them must be under 1.5 m/s; both are at fault if the two robots' speeds differ by under 0.3 m/s | same |
| Stop command | "all robots have to slow down to less than 1.5 m/s" plus 0.5 m from the ball | [chapters/refereecommands.adoc](https://github.com/RoboCup-SSL/ssl-rules/blob/master/chapters/refereecommands.adoc) |
| Safety | qualitative only: "high power or high velocity uncontrolled robot operation" is listed as an unsafe failure a referee may remove a robot for | [chapters/offenses.adoc](https://github.com/RoboCup-SSL/ssl-rules/blob/master/chapters/offenses.adoc) |

So the 1.5 m/s figures are game-state and collision limits, not a top-speed limit; free play is
uncapped, and there is no rule the recommended 3.30 m/s / 2.98 m/s² envelope could violate. The
6.5 m/s ball cap belongs to the kicker ticket, not this one.

**Division A practice** is not codified anywhere and no single figure can be cited as "the"
practice. What is citable, from teams' own published papers: RobôCIn state that "most of the time,
an SSL robot is accelerating, rarely moving at a constant speed", and that they size the wheel to
put the motor's operating point near maximum efficiency at ~10 % of stall torque
([2024 ETDP RoboCIn](https://ssl.robocup.org/wp-content/uploads/2024/04/2024_ETDP_RoboCIn.pdf)
§1.2) — i.e. the design target is acceleration, not top speed, which is consistent with a
torque-limited envelope like the recommended one. TIGERs Mannheim's 2024 ETDP names the
"Maxon EC-45 flat 70 W" as a drive option and refers to "constraints, like maximum velocity and
acceleration" being handed to their path planner, but publishes them as planner inputs rather than
as hardware figures
([2024 ETDP TIGERs Mannheim](https://ssl.robocup.org/wp-content/uploads/2024/04/2024_ETDP_TIGERsMannheim.pdf)).
The recommended 3.30 m/s is in the right band relative to the only hard number the league itself
uses for robot speed — the 1.5 m/s stop limit, which exists precisely because robots normally
travel much faster.

## Unresolved, and stated assumptions

1. **Which EC 45 flat 50 W winding.** Assumed 251601 (24 V, `k_M = 33.5` mN·m/A, 780 mN·m stall).
   Not recorded in the repo. Across the five catalogue variants the design torque spans
   40.2–91.8 mN·m at 10 % of stall, i.e. `a_x` between 1.54 and 3.51 m/s². If the drive is the
   low-speed 24 V winding (339286/400107) the recommendation drops to `forcerange="-0.040 0.040"`
   and 1.54 m/s²; if it is 651607, `-0.092 0.092` and 3.51 m/s². **This is the one input that
   would most change the answer, and one line of team documentation would settle it.**
2. **The driver's current limit.** Assumed equal to the motor's nominal current (2.33 A), i.e. the
   ESC allows no transient overcurrent. Real drives allow several times `I_nom` for tens of
   milliseconds; representing that would need a torque envelope, not a constant `forcerange`.
   Deliberate approximation, cited as such.
3. **Drivetrain efficiency.** The 18:60 arithmetic in §2 assumes an ideal stage. Irrelevant to the
   recommendation because §2.4 concludes there is no stage.
4. **The 25 % yaw margin** in §3.2 is a design choice, not a derived quantity.
5. **`ω_tip` depends on the 3.459 mm CoM offset**, whose fidelity to the physical robot is
   unverified (`M1`). It gates `ctrlrange`, not `forcerange`.
6. **Whether this CAD is the RobôCIn 2025 robot** is inferred from mesh names
   (`Dribbler_Robocup2025v3`, `Direct_Transmission_new_Simplify_2`) and from the 60.18 mm wheel
   matching the ETDP's 62 mm direct-drive wheel. It is not stated in the repo. The direct-drive
   conclusion of §2.4 does not rest on the identification alone — §2.4 item 3 rules out 18:60 on
   arithmetic regardless of whose robot this is.
7. **`kv`** was not re-derived from the motor's speed/torque gradient (8.77 rpm/mN·m for 251601).
   Left unchanged; flagged for whichever ticket owns the servo model.

## Sources

Vendor (maxon group, owner of the data), all fetched 2026-09-08:
- EC 45 flat Ø42.8 mm, 50 W, 24 V, part 251601 — https://www.maxongroup.com/maxon/view/product/251601 and https://www.maxongroup.us/maxon/view/product/motor/ecmotor/ecflat/ecflat45/251601
- EC 45 flat Ø43.5 mm, 50 W, 24 V, part 651607 — https://www.maxongroup.com/maxon/view/product/651607
- EC 45 flat Ø42.8 mm, 50 W, 24 V, part 339286 — https://www.maxongroup.com/maxon/view/product/339286
- EC 45 flat Ø45 mm, 50 W, 24 V, part 400107 — https://www.maxongroup.com/maxon/view/product/400107
- EC 45 flat Ø42.8 mm, 50 W, 36 V, part 339287 — https://www.maxongroup.us/maxon/view/product/motor/ecmotor/ecflat/ecflat45/339287

League:
- Rules of the RoboCup Small Size League — https://robocup-ssl.github.io/ssl-rules/sslrules.html; chapter sources on `master`: https://github.com/RoboCup-SSL/ssl-rules/blob/master/chapters/robots.adoc, .../offenses.adoc, .../refereecommands.adoc
- RobôCIn SSL Extended Team Description Paper for RoboCup 2024, §1.2 Direct Drive — https://ssl.robocup.org/wp-content/uploads/2024/04/2024_ETDP_RoboCIn.pdf
- TIGERs Mannheim Extended Team Description Paper 2024 — https://ssl.robocup.org/wp-content/uploads/2024/04/2024_ETDP_TIGERsMannheim.pdf

Source (this repo, local revision):
- `models/robot_rollers.xml:153` (chassis mass/inertia), `:159` `:164-165` `:250-257` (dribbler gear meshes), `:297-301` (wheel1 body, joint, inertia, Direct_Transmission meshes), `:827-830` (the four wheel actuators)
- `models/robot_kinematic.xml:389-391` (tier 0 actuators)
- `docs/research/rsoccer-vs-mjsoccer.md` §4 (`M1`–`M6`), §6 (`W1`–`W7`, measured wheel matrix), §7 (limits table)
- `docs/analysis/ssl-simulator-rsoccer-comparison.md` (drive-limits row)

Upstream (`/tmp/rsim-upstream`, `b413932`):
- `src/robosim/sslconfig.h:146-160` (wheel angles, radius, masses, `wheelMotorMaxTorque = 0.070`, `wheelMotorMaxRPM = 1557.0`)
- `src/robosim/sslrobot.cpp:23-68` (per-wheel hinge + AMotor, `dParamFMax` = the torque, `dParamVel` = the target, RPM→rad/s clamp)
