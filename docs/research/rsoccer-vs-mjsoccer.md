# rSoccer/rSim versus mjsoccer: a source-level difference audit

This page enumerates every behaviourally meaningful difference between the SSL simulator in
this repository (`mjsoccer`, MuJoCo) and `robocin/rSoccer` (the Gymnasium layer) together with
its engine `robocin/rSim` (C++/ODE, published to Python as `robosim`). It is derived entirely
from the two sources plus the official engine references; nothing here is recalled from
memory or from secondary write-ups. Every claim carries a `path:LINE` citation or a URL, and
every number is quoted from the place it is defined.

To re-derive it: read `src/mjsoccer/{base,scene,tasks,__init__}.py` and the `<option>`,
`<default>`, body/joint/inertial/actuator and `<contact>` blocks of `models/*.xml` on this
side; read `src/robosim/{sslworld,sslrobot,sslconfig,utils,robosim_py}.*` and
`src/robosim/physics/*` on the engine side, and `rsoccer_gym/Simulators/rsim.py`,
`rsoccer_gym/ssl/ssl_gym_base.py`, `rsoccer_gym/Entities/*` and
`rsoccer_gym/ssl/ssl_hw_challenge/*` on the task side. Compiled-model numbers (subtree mass,
centre of mass, effective wheel radius, the wheel-to-chassis matrix) were read out of the
compiled `MjModel`/`MjData` with `mujoco.mj_forward`; those are labelled **measured** and the
exact procedure is given where they appear. Nothing in `src/`, `models/`, `examples/`,
`scripts/` or `tests/` was modified to produce this document.

## Provenance

| item | revision |
|---|---|
| this repository (`mjsoccer`) | `147d2af9a3f389058d4abf005efcf1105d1fc2de` (`main`) |
| `robocin/rSoccer` | `3adf7c3e89fe6d1b47431ee4099dc5ee89420415` |
| `robocin/rSim` | `b413932b7b4bbc0fc399f6a979e4cfcc4f56264f` |
| MuJoCo | `mujoco>=3.11.0` (`pyproject.toml:12`) |
| ODE | version not pinned by the build — `find_package(ODE REQUIRED)` (`/tmp/rsim-upstream/CMakeLists.txt:34`), so ODE defaults are taken from the manual, and single- vs double-precision CFM cannot be resolved from source (see `P1`) |
| date | 2026-09-08 |

Engine-knob semantics are cited from the MuJoCo XML reference
(<https://mujoco.readthedocs.io/en/stable/XMLreference.html>) and the ODE manual
(<https://ode.org/wiki/index.php/Manual>).

Both clones are complete (`git log` and `git cat-file` succeed on every file cited); no blob
was missing and no network fetch was needed beyond the two documentation URLs above.

## Status after the physics-fidelity map (2026-10-18)

**The mjsoccer column of this audit describes `147d2af`.** The physics-fidelity map
(`.scratch/physics-fidelity/`, tickets 01-18) has since changed most of what it describes on
this side. The sections below are kept as the audit of that commit, with inline corrections
only where the audit itself was wrong *at* that commit (`W3`, `B9`, `P4`, `C2`, `B6`). Where
this table and a section disagree about mjsoccer's *current* state, this table wins, and
[`docs/index.md`](../index.md) is the reference for today's numbers.

| ids | at `147d2af` | now | ticket |
|---|---|---|---|
| `L1`, `W3`, `W4`, `W7`, §7 limits | wheel ±200 rad/s, ±3 N·m; friction-limited at ~1 g | ±95 rad/s, ±0.078 N·m from the motor (Maxon EC 45 flat 50 W, **direct drive**, not 18:60); derived 1.95 / 1.30 m/s² forward/lateral (roller friction), 3.30 / 4.04 m/s, 37.1 rad/s; `kv` kept | 01, 04, 12, 16, 19 |
| `L2`, `M1`, `TO1`, `TO2` | full spin 59.4 rad/s; 35.9 turnovers/min under random action | full spin 36.2 rad/s measured (37.1 ceiling); **0 turnovers in 101 min**. The turnovers came from wheel torque authority, not from the 3.459 mm CoM offset (unchanged, harmless). Terminate at 60° with -100 kept; `info` gains `tilt_deg`, `tip_step` | 03, 04, 10, 12, 20 |
| `B5` | chassis does not collide with the floor | collided (floor `contype/conaffinity` 5), touches from 9.5° of tilt | 06 |
| `B1`, `C7` | 209 collision geoms per robot | 193 (16 buried hulls pruned); no hull contact gives a trip impulse in normal play | 07 |
| `C1`, §5 wheel↔ground | roller μ 1.0, floor 0.8 | 0.25 on both (measured, Williams et al. 2002) | 05 |
| `C2`, §5 ball↔ground/wall | no bounce | bounces at COR 0.48-0.55 (floor) / 0.49-0.53 (wall), normal impacts to 6.5 m/s: `<pair>` `solref` `0.012 0.225` / `0.012 0.25`; oblique impacts uncalibrated | 17, 21 |
| §3 ball | MuJoCo default inertia, β = 0.4 | explicit `diaginertia`, β = 0.5 (measured) | 05 |
| `L5`, §7 dribbler | `314 * (a + 1) / 2`: a zero action dribbles at 157 rad/s | `314 * max(a, 0)`: `a ≤ 0` off; 314 rad/s is an assumption | 08 |
| `L4`, `L7`, `L8`, `B4`, §7 kick | 200 N for one control step, exit speed emergent (6-7 m/s) | the action asks for `6.5 * a` m/s, half range; a per-tier force table, `ctrlrange` 0..500 N; the firing step runs at `dt / 4`; 250 ms lockout = 15 steps | 09, 13 |
| `L9`, `U2`, `U4` | `Box(-inf, inf)`; ball z and the goal test on world z | `Box` finite where physics bounds it, nothing clipped; `MAX_V` 3.3, `MAX_W` 37.1; ball height and `in_goal` measured above the carpet (`FLOOR_Z = 0.02`), so the goal is 0.155 m tall over the carpet, not 0.135 | 11, 12, 13 |
| `W5`, `W6`, `P3` (tier 0) | `forcerange` ±19.1261 N / ±3.427 N·m, `ctrlrange` 4.0 m/s, damping 0.1/0.005 | tier 1's derived per-axis envelope: ±3.5129 N (1.30 m/s²), ±0.5263 N·m (55.0 rad/s²), ±3.3 m/s, ±37.1 rad/s; damping 0 | 12, 13 |
| `P2`, `T1`, `T2`, `T4`, §9 | 2 ms step, decimation 8, 62.5 Hz, 1250/1875/2500 steps | 1/960 s step, decimation 16, **exactly 60 Hz**, 1200/1800/2400 steps; an unreachable `control_hz` raises `ValueError` | 13 |
| `W1` | no kinematic matrix anywhere in the Python | the action is still raw wheel speeds, but `tasks.wheel_command` converts a body velocity to them | 14 |
| `W2` | matrix undocumented | stated in `docs/index.md`, with the sign inversion | 15 |

`M2`'s CoM comparison (54.0 mm above the carpet here, 72.3 mm in rSim, so mjsoccer's is
18.3 mm *lower*) is correct. The independent comparison referenced by the map,
`docs/analysis/ssl-simulator-rsoccer-comparison.md`, got this one wrong by omitting the
floor's 20 mm step (`models/field.xml:42`); that file is not in this repository.

## Severity counts

| severity | count |
|---|---|
| **High** — visible every episode, changes policy behaviour | 39 |
| **Medium** — visible in specific situations or shifts magnitudes | 26 |
| **Low/cosmetic** — a trap for a reader or a porter, not for a policy | 15 |
| total | 80 |

Each difference below carries a stable id (`P`hysics, `U`nits, `B`odies, `M`ass, `C`ontacts,
`W`heels, `L`imits, `O`rdering, `T`iming, `R`andomness, reset/`X`, `TO` turnover).

## High-impact summary

The eleven differences that dominate everything else, in the order a policy meets them.

| id | what differs | mjsoccer | rSoccer/rSim | observable effect |
|---|---|---|---|---|
| `U1` | field the task runs on | Division A, 12.0 × 9.0 m (`scene.py:57-58`) | hardware-challenge type 2, 6.0 × 4.0 m (`sslconfig.h:99-109`, `static_defenders.py:45`) | every distance, every shaping gradient and every horizon means a different thing; a policy's spatial priors do not transfer at all |
| `U2` | goal mouth | 1.8 m wide, 0.155 m high, 3-D goal test (`scene.py:59-61`, `scene.py:286-292`) | 0.70 m wide, `ball.x > half_len` 2-D test (`sslconfig.h:108`, `static_defenders.py:190-194`) | scoring probability from the same shot differs by ~2.5× in mouth width, and a chipped ball scores in rSoccer at any height |
| `W1` | drive interface | 4 raw wheel speeds, no kinematic matrix (`base.py:143-156`, `__init__` doc `base.py:13`) | body-frame `(vx, vy, vθ)` through `setDesiredSpeedLocal` (`static_defenders.py:119-124`, `sslrobot.cpp:376-391`) | the action spaces are not the same object; an rSoccer policy cannot be loaded into tier 1 without a conversion |
| `W2` | wheel sign and radius | measured matrix `w = (1/0.030089)(vx sinα − vy cosα − 0.0796 ω)` | `w = (1/0.02475)(−vx sinα + vy cosα + 0.090 vw)` (`sslrobot.cpp:383-386`) | **every sign is inverted.** A ported wheel vector drives backwards and spins the wrong way |
| `L1` | wheel torque ceiling | ±3 N·m per wheel (`models/robot_rollers.xml:827-830`) | 0.070 N·m per wheel (`sslconfig.h:159`) | 43×. rSim is torque-limited to ≈3.7 m/s² forward, 2.8 lateral; mjsoccer is only friction-limited (≈1 g), so it can command accelerations rSim physically cannot produce |
| `L2` | yaw rate reachable | saturated uniform wheel command *requests* 75.6 rad/s, **measured** peak 59.4 rad/s | env-capped at `max_w = 10` rad/s (`static_defenders.py:77`) | ~6×. This is the direct cause of tip-over (see `TO1`) and it puts the `omega` observation channel 2× outside its own normaliser |
| `M1` | centre of mass offset from the spin axis | 3.46 mm, **measured** (CAD asymmetry) | exactly 0 by construction (`sslrobot.cpp:242`, a centred cylinder) | at high yaw the offset CoM is centrifugally levered outward. **Measured** tip threshold ≈50 rad/s, and the reachable yaw is 59 rad/s |
| `TO1` | what happens on turnover | terminate the episode, `reward -= 100.0` (`base.py:41-42`, `base.py:129-132`) | teleport upright, restore the spawn yaw, zero velocities, continue (`sslworld.cpp:388-392`, `sslrobot.cpp:289-303`) | rSoccer episodes never end from tipping and never show the state; mjsoccer's do, and the -100 dominates the return |
| `L3` | possession mechanism | friction contact with a 314 rad/s spinning bar plus a 50 ms grace window (`base.py:166`, `base.py:188-211`) | an ODE hinge joint welding the ball to the kicker plate (`sslrobot.cpp:202-222`) | in rSoccer a held ball cannot be taken by contact at all; `ContestedPossession` is therefore a fundamentally different task on the two sides |
| `L4` | kick | 200 N one-control-step solenoid pulse against a 2000 N/m return spring; exit speed emergent (`base.py:160-163`, `models/robot_rollers.xml:271`) | `dBodySetLinearVel(ball, …)` at exactly `kick_v_x = 5.0` m/s (`sslrobot.cpp:167-200`, `static_defenders.py:78`) | rSoccer's kick is deterministic and instantaneous; mjsoccer's varies with the ball's seating, the chassis velocity and the plunger stroke, and imparts recoil |
| `R1` | reset seeding | `self.np_random`, Gymnasium-seeded (`base.py:96-97`, `base.py:288-294`) | the **global `random` module** (`static_defenders.py:222-228`, `contested_possession.py:217-218`, `pass_endurance.py:162-166`) | `reset(seed=…)` does not control the rSoccer SSL layout. rSoccer SSL episodes are not reproducible from the Gymnasium seed |

---

## 1. Physics configuration

| aspect | mjsoccer | rSoccer/rSim |
|---|---|---|
| engine | MuJoCo (`base.py:27`) | ODE via `robosim` (`rsim.py:2`, `CMakeLists.txt:34`) |
| integrator | `implicitfast` (`models/scene_rollers.xml:4`) | ODE's own semi-implicit step; no choice exposed |
| solver | `Newton`, `iterations="100"`, `ls_iterations="50"`, `tolerance` 1e-8 (`models/scene_rollers.xml:4`; measured off the compiled model) | `dWorldStep` — the direct big-matrix solver, because `fullSpeed = false` (`sslworld.h:43`) selects the `sync == false` branch (`pworld.cpp:308-311`) |
| friction cone | `cone="elliptic"`, `impratio = 1` (`models/scene_rollers.xml:4`) | ODE's friction pyramid with `dContactApprox1` (`pworld.cpp:26`) |
| gravity | `0 0 -9.81` (`models/scene_rollers.xml:4`) | `dWorldSetGravity(world, 0, 0, -9.81)` (`sslworld.cpp:112`, `pworld.cpp:52`) |
| soft-contact parameterisation | per-geom/per-pair `solref` (time constant, damping ratio) and `solimp` (`models/scene_rollers.xml:21-23`); model-wide defaults `solref = [0.02, 1]`, `solimp = [0.9, 0.95, 0.001, 0.5, 2]` (measured) | global ERP/CFM, never set → ODE defaults ERP `0.2`, CFM `1e-5` (single) / `1e-10` (double) per the ODE manual; plus one per-contact `soft_cfm = 0.002` on wheel/ground (`sslworld.cpp:52`) |
| joint damping / armature | present on every joint: chassis slides `damping="0.1"`, yaw `0.005` (`models/robot_kinematic.xml:152-154`), wheel hubs `damping="1e-5" armature="1e-5"` (`models/robot_rollers.xml:298`), rollers `1e-6`/`1e-7` (`models/robot_rollers.xml:308`), dribbler bar `1e-6`/`1e-7` (`models/robot_rollers.xml:262`), kicker/chipper `damping="2.0"` (`models/robot_rollers.xml:271,282`) | none — no `dBodySetLinearDamping`/`dBodySetAngularDamping` call exists anywhere in `src/robosim` |
| air drag | none: `density = 0`, `viscosity = 0`, `wind = 0 0 0` (measured off the compiled model) | none |
| ball rolling resistance | contact rolling friction on the `ball`/`floor` pair, `friction="0.35 0.35 0.005 0.0009 0.0009"` with `condim="6"` (`models/scene_rollers.xml:23`) | an explicit Coulomb drag force re-applied every substep: `fk = 0.05 · 0.043 · 9.81` opposite the velocity plus a matching torque, and a hard kill below 0.01 m/s (`sslworld.cpp:285-298`, coefficient at `sslconfig.h:124`) |

**`P1` (High).** These are two different constraint formulations, and no mapping between them
exists. MuJoCo's contacts are soft in a *specified* way — `solref` is `(time constant, damping
ratio)` and `solimp` five impedance parameters (MuJoCo XML reference, `geom/solref`,
`geom/solimp`), so `solref="0.006 1"` on the ball/floor pair is a 6 ms critically damped
contact and `solref="0.004 0.5"` on the ball/dribbler pair is a 4 ms *under*damped one.
ODE's softness is one global CFM plus a global ERP of 0.2, meaning "correct 20% of the
constraint error per step" (ODE manual, `dWorldSetERP`). The observable consequence is that
mjsoccer's contacts have a designed compliance and settling time per contact type, whereas
rSim's have one stiffness for everything except the wheels. Anything that depends on how a
contact rings — ball retention in a dribbler, a robot leaning on a wall, robot-robot
scrubbing — is qualitatively different, not merely differently tuned. It also means CFM's
value in rSim is unresolved: it is `1e-5` or `1e-10` depending on whether the linked ODE was
built `dSINGLE` or `dDOUBLE`, which the CMake does not constrain. **Unresolved**; printing
`dWorldGetCFM` from a built `robosim`, or checking `ODE_LIBRARIES` for the precision suffix,
would settle it.

**`P2` (Medium).** Physics `dt` is 2 ms in mjsoccer (`models/scene_rollers.xml:4`) and 5 ms in
rSim (`static_defenders.py:50` sets `time_step = 0.025`; `sslworld.cpp:300` steps
`timeStep * 0.2`). A 2.5× coarser step with ODE's ERP-based error correction makes rSim's
impacts softer and its high-speed contacts more penetrating; mjsoccer resolves a 6 m/s ball
in 12 mm per step, rSim a 5 m/s ball in 25 mm — larger than the ball radius, so rSim can
tunnel a fast ball past a thin kicker plate.

**`P3` (Medium).** mjsoccer damps every degree of freedom; rSim damps none. A coasting rSim
robot decelerates only through wheel/ground friction, and a spinning rSim wheel never slows.
An mjsoccer roller has `damping="1e-6"` and `armature="1e-7"`, so 80 rollers per robot bleed a
small, always-on amount of energy that rSim has no analogue for.

**`P4` (Medium).** Ball deceleration is modelled by two incompatible mechanisms. rSim applies
a constant 0.49 m/s² Coulomb deceleration independent of speed and then snaps the ball dead
below 0.01 m/s (`sslworld.cpp:295-298`). mjsoccer bounds the rolling-friction torque at
`μ_roll · f_n = 0.0009 · 0.451 N = 4.06e-4 N·m`, which for a rolling 46 g / 21.5 mm sphere is
about 0.29 m/s² (derived from `models/ball.xml:7` and `models/scene_rollers.xml:23`). A 5 m/s
pass therefore runs roughly 43 m in mjsoccer against 26 m in rSim, and mjsoccer's ball creeps
instead of parking. On a 12 × 9 m field both cross it.

**Reframed (ticket 02).** This is not mjsoccer's ball being too lively. rSim's 0.49 m/s²
traces to ER-Force's Bullet workaround (`hackFactor = 1.4`), and the measured rolling
deceleration on carpet is 0.26-0.31 m/s², so **mjsoccer's 0.293 m/s² is right and rSim's is
the outlier.**

**`P5` (Low).** `sslconfig.h:119` (`ballSlip = 1`), `:127` (`ballLinearDamp = 0.004`) and
`:128` (`ballAngularDamp = 0.004`) are dead configuration: `grep` over `src/robosim` finds no
reader for any of the three, and `sslworld.cpp:157` explicitly comments out
`slip1 = cfg->ballslip()`. Anyone reading rSim's config to reproduce its ball behaviour will
model damping that is not there.

**`P6` (Low).** `pworld.cpp:307` calls `dWorldSetQuickStepNumIterations(world, 20)` on every
step, but the very next branch takes `dWorldStep` because `sync` is false
(`pworld.cpp:308-311`, `sslworld.h:43`). The iteration count is inert; rSim runs the O(m³)
direct solver, not QuickStep (ODE manual, `dWorldStep` vs `dWorldQuickStep`). This matters
for anyone estimating rSim's cost or its accuracy from that line.

---

## 2. Units and coordinate frames

| aspect | mjsoccer | rSoccer/rSim |
|---|---|---|
| length | metres throughout | metres throughout |
| angle, internal | radians (`<compiler angle="radian">`, `models/scene_rollers.xml:3`) | radians inside ODE, but every API boundary is **degrees** |
| angle, at the API | radians; yaw from `atan2(R[1,0], R[0,0])` ∈ (−π, π] (`scene.py:104`) | degrees; `getDir` returns `absAng` or `360 − absAng`, i.e. **[0, 360)** (`sslrobot.cpp:313-323`); `Frame` docstring confirms "degrees, degrees/s" (`Frame.py:8`) |
| where conversion happens | nowhere — radians all the way to the observation, which emits `sin θ`, `cos θ` (`base.py:231-232`) | `setDir` multiplies by π/180 (`sslrobot.cpp:345`); the envs call `np.deg2rad` on the way in (`static_defenders.py:118`) and out (`static_defenders.py:101-102`) |
| field frame | origin at field centre, +x toward the attacking goal, +y left, +z up | origin at field centre, same handedness |
| goal axis | ±x; the scoring goal is +x (`scene.py:286-292`, `tasks.py:99-105`) | ±x; scoring goal is +x (`static_defenders.py:190-194`) |
| field | half-length 6.0 m, half-width 4.5 m (`scene.py:57-58`) | `fieldLength/2 = 3.0`, `fieldWidth/2 = 2.0` for the type-2 field all four SSL envs use (`sslconfig.h:103-104`) |
| vertical datum | the floor plane sits at **z = +0.02** (`models/field.xml:42`); the robot body spawns at `z = 0.021` (`models/robot_rollers.xml:150`), the ball at `0.0415` (`models/ball.xml:5`) | the ground plane is at **z = 0** (`pground.cpp:35`); the chassis centre spawns at `height/2 + bottomHeight = 0.077` (`sslrobot.h:103`), the ball at `z = ballRadius` (`sslworld.cpp:113`) |
| implicit scaling | none | none |

**`U1` (High).** All four SSL environments in rSoccer pass `field_type = 2`
(`static_defenders.py:45`, `dribbling.py:47`, `contested_possession.py:43`,
`pass_endurance.py:48`), which `sslconfig.h:99-109` defines as the **hardware-challenge**
field: 6.00 × 4.00 m, penalty area 2.00 × 0.80, goal 0.70 m. mjsoccer is Division A: 12 × 9 m,
goal 1.8 m (`scene.py:57-61`, asserted against the CAD in `scene.py:269-279`). The playing
area is four times larger here. Task-for-task, `StaticDefenders-v0`'s six defenders sit on a
12 m pitch (`tasks.py:192-199`) and rSoccer's on a 6 m one; the ratio of robot diameter to
field length differs by 2×, so obstacle density, shot angles and travel times are all
different. Nothing about spatial policy behaviour survives this.

**`U2` (High).** The goal test is 3-D here and 2-D upstream. `scene.in_goal`
(`scene.py:286-292`) requires `|y| < 0.9` **and** `z < GOAL_HEIGHT = 0.155`;
`static_defenders.py:190-194` and `contested_possession.py:183-190` only require
`ball.x > half_len` and `|ball.y| < half_goal_wid`. A chipped ball flying over the crossbar
scores in rSoccer and does not score here. Combined with the 1.8 m vs 0.70 m mouth, a shot of
identical geometry has a very different scoring probability.

**`U3` (High).** Angles cross the API in different units and different wrap ranges.
mjsoccer's yaw is radians in (−π, π]; rSoccer's is degrees in [0, 360). Both envs feed
`sin`/`cos` to the policy so a trained network is insulated, but every scripted controller,
every log, every reward that touches an angle, and rSim's own finite-difference angular
velocity (`smallestAngleDiff`, `utils.cpp:3-18`, which works in degrees against ±180
thresholds) are unit-coupled. Porting a heading controller across without conversion produces
a 57× gain error.

**`U4` (Medium).** The vertical datum differs by 20 mm. mjsoccer's floor is a plane at
`z = 0.02` (`models/field.xml:42`), so a "ball z" of 0.0415 means 21.5 mm above the playing
surface, and the `GOAL_HEIGHT = 0.155` normaliser applied to the raw `bp[2]`
(`base.py:220`) is therefore measuring height above the *world* origin, not above the carpet
— a systematic +0.02/0.155 = +0.129 offset in that observation channel. rSim's ground is at
`z = 0` so no such offset exists.

**`U5` (Medium).** Which x/y bound counts as "out" is defined differently. mjsoccer uses the
rulebook constants (`scene.in_field`, `scene.py:282-283`) at ±6.0/±4.5, while the physical
walls are at ±6.32/±4.82 and only 0.20 m tall (**measured** AABB of `col_f_walls_*`,
`models/field.xml:44-55`). So the episode always ends before the ball reaches a wall
(`base.py:126-128`) and the wall geometry is effectively decorative for the ball. rSoccer's
walls are 0.4 m tall boxes at `±(L/2 + 0.3 + 0.025)` (`sslworld.cpp:216-243`) and its
out-of-bounds rules are per-env and asymmetric (`ball.x < 0` counts as out in
`static_defenders.py:187`).

**`U6` (Low).** rSim's goal boxes are built with `goalHeight = 0.16` (`sslconfig.h:32`) against
mjsoccer's `GOAL_HEIGHT = 0.155` (`scene.py:61`), and the goalpost collision meshes here
measure x ∈ [6.0, 6.2], |y| ≤ 0.94, z ≤ 0.17 (**measured** AABB, `models/field.xml:58-65`). A
5 mm crossbar difference is cosmetic, but it means the two "under the bar" tests are not
numerically identical.

---

## 3. Bodies and collision shapes

| body | mjsoccer | rSoccer/rSim |
|---|---|---|
| chassis | CAD convex-hull decomposition: 47 chassis hulls plus motor housings, bumpers, dribbler structure and gear housings — **209 collision geoms per robot** (measured; 193 since ticket 07 pruned 16 buried hulls, 299 geoms in total with the visuals); `type="mesh"`, `condim="3"`, `friction="0.5 0.005 0.0001"` (`models/robot_rollers.xml`, e.g. the `r_col_chassis_*` block) | one `dCreateCylinder`, radius 0.090, height 0.146 (`sslrobot.cpp:242`, `pcylinder.cpp:43`, `sslconfig.h:150-151`) |
| chassis footprint | **measured** collision half-extent 87.7 × 88.0 mm about the body axis | 90.0 mm radius |
| chassis height | **measured** top at 156 mm above the carpet | `bottomHeight + height = 0.004 + 0.146 = 150` mm (`sslconfig.h:151-152`) |
| wheels | 4 hub bodies at 60/135/225/300°, mount radius **measured** 79.60 mm, each carrying **20 free-spinning roller bodies** on their own hinges — 84 bodies and 84 joints per robot beyond the chassis (`models/robot_rollers.xml:297,428,559,690`; roller pattern at `:305-311` onward) | 4 solid `PCylinder` wheels, radius 0.02475, thickness 0.005, mount radius `0.090 + 0.005/2 = 0.0925` (`sslrobot.cpp:27-36`) |
| effective wheel radius | **measured** 30.089 mm (axle world z 0.0510892 minus lowest roller vertex 0.0209999) | 24.75 mm (`sslconfig.h:153`) |
| dribbler | a real mechanism: `dribbler_arm` on a sprung, limited pivot (`range="0 0.14"`, `stiffness="1"`) carrying a `dribbler` bar on a free hinge, with two dedicated collision cylinders `r_col_axisDribbler` (r 5 mm, half-length 47 mm) and `r_col_barDribbler` (r 9 mm, half-length 36 mm) (`models/robot_rollers.xml:259-267`) | **no dribbler geometry at all.** `dribblerOn` is a boolean that creates a joint (`sslrobot.cpp:152-155`, `:202-222`) |
| kicker | two independent sprung plungers — `kicker` (mass 0.1413 kg) and `chipper` (0.1372 kg), each a `slide` joint with `range="0 0.03"`, `stiffness="2000"`, `damping="2.0"`, each with its own mesh plate (`models/robot_rollers.xml:270-296`) | one `PBox` 0.005 × 0.080 × 0.040, mass 0.020, hinged to the chassis with `LoStop = HiStop = 0` so it never moves (`sslrobot.cpp:82-100`) |
| floor | `type="plane"` at z = 0.02, `contype="4" conaffinity="4"`, `condim="3"`, `friction="0.8 0.005 0.0001"` (`models/field.xml:42`) | `dCreatePlane(space, 0,0,1, 0)` (`pground.cpp:35`) |
| walls | 12 convex hulls from CAD, **measured** ±6.32/±4.82, 0.20 m tall, `condim="3"`, `friction="0.6 …"`, `solref="0.004 1"` (`models/field.xml:44-55`) | 4 `PFixedBox` walls 0.4 m tall (`sslworld.cpp:233-243`) |
| goal walls | 8 goalpost hulls per goal, x ∈ [6.0, 6.2] (`models/field.xml:58-74`) | 3 `PFixedBox` per goal — back plate plus two sides (`sslworld.cpp:247-263`) |
| ball | sphere r = 0.0215, mass 0.046, `condim="4"`, `contype="7" conaffinity="7"` (`models/ball.xml:7`) | `dCreateSphere` r = 0.0215, `dMassSetSphereTotal` 0.043 (`sslconfig.h:121,123`, `pball.cpp:31-44`) |

**`B1` (High).** The chassis is a 209-hull (now 193) CAD decomposition here and a single cylinder there.
Every robot-robot and robot-wall interaction is therefore a different event: mjsoccer's robots
catch on each other's bumpers, dribbler structures and motor housings, generate multiple
simultaneous contact points at different heights, and produce torques a smooth cylinder
cannot. rSim's cylinders slide off each other cleanly. This is the single biggest source of
unexpected yaw and tipping impulses in a crowd.

**`B2` (High).** Tier 1 models omni-wheels as 80 physical rollers with their own hinge joints
and contacts; rSim models them as four solid cylinders with anisotropic Coulomb friction (see
`C1`). mjsoccer's version is the higher-fidelity one, and it costs: 100 DOF versus tier 0's 13
(measured `nv`), a contact set that switches discretely as rollers enter and leave the floor,
and per-roller normal-force spikes. **Measured** counter-evidence for one plausible artefact:
the roller crown radius varies only between 29.96 and 30.09 mm across the 18° roller pitch, so
the wheel does not measurably bob as it rolls — the ripple is 0.4%.

**`B3` (High).** rSoccer has no dribbler body. Possession is a joint, not a mechanism (`L3`).
mjsoccer's bar is a real spinning cylinder on a sprung arm with tuned contact against the ball
(`models/scene_rollers.xml:21-22`). Any behaviour involving *taking* or *losing* a ball is
therefore not comparable.

**`B4` (High).** rSoccer has no kicker mechanism either: the plate is welded in place
(`LoStop = HiStop = 0`, `sslrobot.cpp:99-100`) and the kick is a velocity assignment. mjsoccer
fires a sprung plunger, so it has stroke, recoil on the chassis, a dependence on where the
ball is sitting, and an exit speed that emerges rather than being commanded.

**`B5` (High).** Neither chassis collides with the ground, but for different reasons and with
the same consequence. mjsoccer's chassis geoms are `contype="1" conaffinity="1"` while the
floor is `contype="4" conaffinity="4"` (`models/robot_rollers.xml` chassis block versus
`models/field.xml:42`), and MuJoCo requires `(contype1 & conaffinity2) | (contype2 &
conaffinity1) ≠ 0` (MuJoCo XML reference, `geom/contype`) — which is zero here. rSim simply
never creates a chassis/ground `PSurface` (`sslworld.cpp:171-197` enumerates every surface
it makes; chassis appears only against walls, the ball and other chassis). **Only the wheels
touch the floor on both sides.** A tipped robot therefore has nothing to land on: it either
hangs on its wheel rim or its chassis passes through the carpet. This is why turnover needs an
explicit handler in both simulators, and it constrains the fixes in `TO2`.

**`B6` (Medium).** The ball collides with the wheels and rollers in mjsoccer — the ball is
`contype="7" conaffinity="7"` (`models/ball.xml:7`) and the roller geoms are `contype="4"
conaffinity="4"` (`models/robot_rollers.xml:311`), so bit 4 matches. rSim never collides the
ball with a wheel: `pworld.cpp:287-305` enumerates the collide calls and `spaceWheel` appears
only against the ground. A ball rolling under an mjsoccer robot gets deflected by the wheel; in
rSoccer it passes through the wheels untouched. *(Measured by ticket 07: it is the wheel
**hub** that deflects the ball; ball↔roller contact changes the exit velocity by ≤ 0.04 m/s.)*

**`B7` (Medium).** Wall geometry differs in height and in extent: 0.20 m of 12-hull CAD at
±6.32/±4.82 here versus 0.4 m boxes at ±3.325/±2.325 (type-2 field) there. Because mjsoccer
terminates on ball-out at ±6.0/±4.5 first (`base.py:126-128`), its walls only ever matter for
robots; in rSoccer a ball can legitimately bounce off a wall inside the episode.

**`B8` (Medium).** Goal structure: 8 convex hulls per goal here against 3 boxes there. A ball
that hits an mjsoccer post gets a mesh-accurate deflection; rSim's goal is a rectangular
pocket.

**`B9` (Low).** Ball mass differs — 0.046 kg (`models/ball.xml:7`) versus 0.043 kg
(`sslconfig.h:123`). Radius is identical at 0.0215 m.

**Reclassified (ticket 02).** This is not a 7 % modelling difference but a **transcription
error** upstream: 0.043 is the rulebook's §2.2 ball *diameter* in metres pasted into the mass
field, and rSim's `ballRadius` is derived from it by halving. mjsoccer's 0.046 kg is the
rulebook mass.

---

## 4. Mass, inertia and centre of mass

| quantity | mjsoccer | rSoccer/rSim |
|---|---|---|
| chassis body mass | 2.27657 kg (`models/robot_rollers.xml:153`); tier 0 uses 2.39076 kg (`models/robot_kinematic.xml:155`) | 2.200 kg (`sslconfig.h:155`) |
| wheel mass | 0.019739 kg hub + 20 × 0.000440446 kg rollers = 0.0285475 kg per wheel (measured) | 0.050 kg (`sslconfig.h:156`) |
| kicker mass | 0.141298 + 0.137206 kg (kicker + chipper, `models/robot_rollers.xml:272,283`) | 0.020 kg (`sslconfig.h:167`) |
| dribbler mass | 0.008 kg arm + 0.0148091 kg bar (`models/robot_rollers.xml:260,263`) | — |
| **total robot mass** | **2.69207 kg** (measured `body_subtreemass`) | **2.42 kg** (2.2 + 4×0.05 + 0.02) |
| how inertia is specified | CAD `fullinertia` on every body, e.g. chassis `0.00877972 0.00764781 0.00783416 2.01248e-05 −4.18293e-05 −2.02144e-05` (`models/robot_rollers.xml:153`) | analytic primitives: `dMassSetCylinderTotal` for chassis and wheels, `dMassSetBoxTotal` for the kicker, `dMassSetSphereTotal` for the ball (`pcylinder.cpp:34`, `pbox.cpp:46`, `pball.cpp:44`) |
| inertial-frame offset | chassis `ipos = (−0.00796, 0.00021, 0.05801)` (`models/robot_rollers.xml:153`) | zero — every primitive's mass frame is centred on its geom |
| **CoM height above the driving surface** | **53.99 mm** (measured `subtree_com` z 0.07399 minus floor 0.02) | **72.25 mm** (derived: (2.2·0.077 + 4·0.05·0.02475 + 0.02·0.025)/2.42, from `sslrobot.h:103`, `sslrobot.cpp:35,80`) |
| **CoM lateral offset from the spin axis** | **3.459 mm** (measured) | **0** by construction |
| wheel-contact footprint radius | 79.60 mm (measured) | 92.5 mm (`sslconfig.h:150,154`) |
| static tip angle, forward | derived `atan(43.25/53.99) = 38.7°` about the front wheel line at x = +39.8 mm (the binding edge; rear is 44.4°, lateral 49.2°) | derived `atan(46.25/72.25) = 32.6°` about the front wheel line at x = +46.25 mm |

**`M1` (High).** The CAD chassis has its centre of mass **3.459 mm off the wheel/spin axis**
(measured: `subtree_com − xpos` in the body's own frame, with all joints at `qpos0`). rSim's
chassis is a centred cylinder, so its offset is exactly zero. Under yaw at rate ω this offset
produces a centrifugal moment `m ω² d h` about the tipping edge, and the restoring moment is
`m g d_edge`. Setting them equal gives

```
ω_tip = sqrt( g · d_edge / (h · d_com) )
      = sqrt( 9.81 · 0.04325 / (0.05399 · 0.003459) )  =  47.7 rad/s
```

**Measured confirmation** (uniform wheel command, `place`d at the origin, 3 s of physics, no
other body in contact): at 120 rad/s of wheel command the peak yaw is 42.1 rad/s and the
chassis tilts 6.0°; at 160 rad/s the peak yaw is 55.6 rad/s and the tilt reaches 43.2°; at 200
rad/s peak yaw 59.4 rad/s and tilt 46.6°. The onset brackets the predicted 47.7 rad/s (`d_edge`
is the 43.25 mm from the CoM to the front wheel line, the binding edge of the three). This
is the mechanism behind every turnover in this repository, and rSim has no counterpart to it.

**`M2` (Medium).** mjsoccer's CoM is *lower* than rSim's — 54.0 mm against 72.3 mm — so
"mjsoccer tips because its CoM is high" is false. What is worse here is the footprint: 79.6 mm
against 92.5 mm. The two roughly cancel in the static tip angle (38.7° here, 32.6° there —
mjsoccer is actually the *more* stable one statically). Turnover here is dynamic, driven by
`M1` plus the acceleration authority in `L1`/`L2`, not by geometry.

**`M3` (Medium).** Total mass differs by 11% (2.692 vs 2.42 kg) and the mass *distribution*
differs much more: the kicker and chipper plungers alone are 0.2785 kg here against a 0.020 kg
plate there, mounted forward of centre and free to slide. Their motion under a kick shifts the
CoM within the step.

**`M4` (Medium).** Inertia is CAD-derived here and analytic there. mjsoccer's chassis has
non-zero off-diagonal terms (`models/robot_rollers.xml:153`), so its principal axes are not
the body axes; rSim's are exactly aligned. Rotational response to an off-axis impulse
therefore differs in direction, not just magnitude.

**`M5` (Low).** rSim's cylinder inertias are mis-axed. `dMassSetCylinderTotal(&m, mass, 1,
radius, length)` (`pcylinder.cpp:34`) passes direction `1` = the x axis (ODE manual,
`dMassSetCylinderTotal`), while `dCreateCylinder` builds a **z**-aligned cylinder
(`pcylinder.cpp:43`). So the chassis's 0.146 m dimension is treated as lying along x for
inertia purposes: derived `Ixx = 0.00891`, `Iyy = Izz = 0.008363` where the correct assignment
is `Izz = 0.00891`, `Ixx = Iyy = 0.008363` — a 6% error, small because `r ≈ L/1.6`. The same
bug halves each wheel's spin inertia (derived 7.76e-6 instead of 1.53e-5 kg·m²), which makes
rSim's wheels spin up about twice as fast as their geometry implies. Both effects are small
against the chassis mass; flagged because anyone reproducing rSim's numbers analytically will
not match.

**`M6` (Low).** mjsoccer's two tiers do not have the same chassis mass — 2.39076 kg on tier 0
(`models/robot_kinematic.xml:155`) against 2.27657 kg on tier 1
(`models/robot_rollers.xml:153`), because tier 0 folds the wheels into the chassis. Total
subtree mass is identical at 2.69207 kg (measured on both), so this is bookkeeping, but a
reader comparing the two XMLs will see two different numbers for "the robot".

---

## 5. Contacts

| contact pair | mjsoccer | rSoccer/rSim |
|---|---|---|
| wheel ↔ ground | roller meshes, `condim="4"`, `friction="1.0 0.005 0.0001"`, `solref="0.004 1"` — isotropic; the omni behaviour comes from the rollers actually rolling (`models/robot_rollers.xml:311`) | **anisotropic**: `mode = dContactFDir1 \| dContactMu2 \| dContactApprox1 \| dContactSoftCFM`, `mu = 0.05` along `fdir1`, `mu2 = 0.8` across it, `soft_cfm = 0.002`, with `fdir1` set to the wheel's own axle direction each step (`sslworld.cpp:27-63`, coefficients `sslconfig.h:157-158`) |
| ball ↔ ground | explicit pair, `condim="6"`, `friction="0.35 0.35 0.005 0.0009 0.0009"`, `solref="0.006 1"` → critically damped, **no bounce** (`models/scene_rollers.xml:23`) | `mode = dContactBounce \| dContactApprox1`, `mu = 1`, `bounce = 0.5`, `bounce_vel = 0.1` (`sslworld.cpp:152-161`, `sslconfig.h:125-126`) |
| ball ↔ wall | geom defaults, `solref="0.004 1"`, no restitution (`models/field.xml:44-55`) | the same bouncy surface as ball/ground, `bounce = 0.5` (`sslworld.cpp:168-169`) |
| ball ↔ dribbler | two explicit pairs, `condim="4"`, `friction="2.5 2.5 0.02 0.0001 0.0001"`, `solref="0.004 0.5"`, `solimp="0.95 0.99 0.001"` (`models/scene_rollers.xml:21-22`), replicated per robot at build time (`scene.py:222-227`) | none — no dribbler geom exists |
| ball ↔ kicker plate | geom-mixed from `friction="0.4 0.005 0.0001"` on the plate meshes and the ball's own `0.35 0.004 0.0002` (`models/robot_rollers.xml:274,292`, `models/ball.xml:7`) | `mode = dContactApprox1`, `mu = 0.8`, `slip1 = 5` (`sslworld.cpp:163-166,182`, `sslconfig.h:169`) |
| ball ↔ chassis | geom-mixed; MuJoCo takes the elementwise max friction and averages `solref`, giving derived `(0.5, 0.005, 0.0002)` at `condim = 4` | `PSurface` default `mode = dContactApprox1`, `mu = 0.5` (`pworld.cpp:26-27`), **plus a callback that suppresses the contact entirely inside the ±25.84° kicker sector** (`sslworld.cpp:66-99`, angle from `sslconfig.h:170`) |
| robot ↔ robot | 209 (now 193) mesh hulls against 209, `condim="3"`, `friction="0.5 0.005 0.0001"` | cylinder against cylinder, `PSurface` default `mu = 0.5` (`sslworld.cpp:190-196`) |
| robot ↔ wall | mesh hulls against the 12 wall hulls, `friction` 0.5 vs 0.6 | cylinder against box, `PSurface` default `mu = 0.5` (`sslworld.cpp:173-176`) |
| chassis ↔ ground | **not collided** (`contype`/`conaffinity` mismatch) | **not collided** (no surface created) |
| wheel ↔ wall, wheel ↔ wheel, wheel ↔ chassis, kicker ↔ anything but the ball | collided (wheel hubs are `contype="1" conaffinity="1"`; kicker plates are `contype="2" conaffinity="0"` so only the ball reaches them) | **not collided at all** — `pworld.cpp:287-305` lists every pair rSim tests |
| restitution model | via `solref`'s damping ratio; every pair here is at damping ratio 1 (critical) except ball/dribbler at 0.5 | explicit `dContactBounce` with a `bounce_vel` threshold |

**`C1` (High).** The two simulators get omni-directional motion by opposite means. rSim fakes
it: a solid wheel with `mu = 0.05` along the axle and `mu2 = 0.8` across it
(`sslworld.cpp:50-51`), with `fdir1` recomputed from the wheel's rotation matrix on every
contact (`sslworld.cpp:54-62`). That is a correct and cheap model, and it means an rSim robot
resists a sideways push with only `0.05 · 2.42 · 9.81 = 1.19 N` — it slides sideways almost
freely. mjsoccer lets 80 rollers actually roll, with isotropic `μ = 1.0` at each roller
contact (`models/robot_rollers.xml:311`). The sideways resistance is then the rollers' own
rolling friction (0.0001) and bearing damping (1e-6), so it is *lower* still, but the energy
goes into roller spin rather than into a slip that Coulomb-dissipates. Consequences: an
mjsoccer robot pushed sideways keeps momentum in its rollers and coasts; an rSim robot
dissipates it. Contact-rich play — a scrum, a wall graze — evolves differently.

**`C2` (High).** The ball bounces in rSoccer and does not here. `bounce = 0.5`,
`bounce_vel = 0.1` on ball/ground *and* ball/wall (`sslworld.cpp:153-156,168-169`) against
critically damped `solref="0.006 1"` on ball/floor (`models/scene_rollers.xml:23`) and
`solref="0.004 1"` on the walls (`models/field.xml:44`). A chip that lands rebounds to half
its impact speed upstream and dies here; a hard shot that misses rebounds off the back wall
upstream and stops here. This is one of the most immediately visible differences in a replay.

**Reframed (ticket 02).** The measured ball COR on carpet is 0.47-0.56, so rSim's
`bounce = 0.5` is its one defensible contact constant and **mjsoccer's zero-COR ball is the
fidelity gap**, not a neutral difference. The target is COR 0.50 on floor and wall; the
`solref` damping ratio that reaches it is ticket 17.

**`C3` (High).** Ball/ground sliding friction is 0.35 here (`models/scene_rollers.xml:23`) and
1.0 there (`sslworld.cpp:154`), with rSim additionally overlaying its explicit Coulomb drag
(`P4`). A backspun or sliding ball grips almost three times harder upstream, so it transitions
from sliding to rolling in about a third of the distance.

**`C4` (High).** Ball retention has no analogue upstream. mjsoccer's ball/dribbler pairs carry
`friction="2.5 …"` — a sliding coefficient of 2.5, deliberately far above unity — and
`solref="0.004 0.5"`, a 4 ms *underdamped* contact, tuned so a spinning bar can hold the ball
(`models/scene_rollers.xml:21-22`). `scene.build` goes out of its way to re-emit these pairs
per robot after the template body is deleted, because falling back to geom mixing "silently
discards the ball/bar contact tuning" (`scene.py:14-17`, code at `scene.py:197-227`). rSoccer
has nothing here; see `L3`.

**`C5` (Medium).** rSim needs a callback to let the ball reach its kicker at all: the chassis
is a solid cylinder, so `sslBallCallBack` returns `false` (suppress the contact) whenever the
ball is inside the ±`acos(0.081/0.090) = ±25.84°` sector in front of the robot
(`sslworld.cpp:66-99`). mjsoccer's CAD has a real open mouth, so no such override exists. The
upstream hack has a side effect: within that sector the ball can pass *through* the chassis
wall entirely, so a ball approaching the front quarter of an rSim robot from outside behaves
as if the robot were not there.

**`C6` (Medium).** rSim's ball/kicker surface sets `slip1 = 5` (`sslworld.cpp:166`), a large
force-dependent slip along the first friction direction, which lets the ball skid across the
plate. mjsoccer's plate is a normal frictional mesh at `μ = 0.4`
(`models/robot_rollers.xml:274`).

**`C7` (Medium).** rSim's collision set is minimal and hand-enumerated
(`pworld.cpp:287-305`): wheel↔ground, ball↔ground, ball↔kicker, chassis↔ball, ball↔wall,
chassis↔wall, chassis↔chassis. Nothing else is ever tested. mjsoccer's bitmask admits
robot↔robot on every hull, ball↔roller, and hub↔wall. rSim's robots therefore cannot catch
wheels on each other or on a wall; mjsoccer's can, and that is a trip-inducing impulse with no
upstream counterpart.

**`C8` (Low).** rSim applies `soft_cfm = 0.002` only to wheel/ground (`sslworld.cpp:52`).
mjsoccer's compliance is per-pair and explicit throughout. Comparable in spirit, not in value.

**`C9` (Low).** Most mjsoccer ball/robot contacts are *not* explicit pairs — only the two
dribbler geoms are (`models/scene_rollers.xml:21-22`). The rest come from MuJoCo's default
geom mixing (elementwise max friction, `condim` max, `solref` averaged by `solmix`). That is a
deliberate choice, but it means the ball/kicker-plate compliance here is an emergent average
rather than a tuned value.

---

## 6. Wheel actuation

| aspect | mjsoccer tier 1 (default) | mjsoccer tier 0 | rSoccer/rSim |
|---|---|---|---|
| wheels | 4, at 60/135/225/300°, mount radius 79.60 mm (measured) | none — 3 chassis joints | 4, at 60/135/225/300° (`sslconfig.h:146-149`), mount radius 92.5 mm |
| what the action is | 4 raw wheel angular velocities (`base.py:13`, `base.py:143-156`) | body-frame `(vx, vy, ω)`, rotated into world axes before writing (`base.py:148-155`) | body-frame `(vx, vy, vθ)` for all four SSL envs; a wheel-speed path exists but no env uses it (`rsim.py:137-153`, `sslworld.cpp:310-316`) |
| kinematic matrix | **none in the Python** | none | `setDesiredSpeedLocal`, `sslrobot.cpp:376-391` |
| how force reaches the ground | MuJoCo `velocity` actuator on each hub joint, `kv = 0.103696`, `forcerange="-3 3"` (`models/robot_rollers.xml:827-830`); the contact does the rest | `velocity` actuators directly on the world-axis slide/hinge joints, `forcerange="-19.1261 19.1261"` (x,y) and `"-3.427 3.427"` (yaw) (`models/robot_kinematic.xml:389-391`) — **no ground contact is involved at all** | ODE `dJointCreateAMotor` per wheel, `dParamVel` = the target, `dParamFMax = 0.070` N·m (`sslrobot.cpp:50-56,63-68`) |
| slip | genuine: rollers roll and the tyre contact can break away at `μ = 1.0` | none — the servos push the chassis directly, with no friction limit | Coulomb, anisotropic (`C1`) |
| motor model | force-clipped proportional velocity servo (`force = kv·(ref − v)`, clipped to `forcerange`; MuJoCo XML reference, `velocity/kv`) | same | a hard velocity constraint with a torque bound (ODE manual, `dParamVel`/`dParamFMax`) |

**Measured wheel-to-chassis map (tier 1).** With the model compiled and `mj_forward` run at
`qpos0`, each hub's spin axis in world coordinates is `R_wheel · jnt_axis`, and its drive
direction on the floor is `axis × ẑ`:

| wheel | mount (m) | mount angle | spin axis (world) | ground drive direction |
|---|---|---|---|---|
| `wheel1` | (+0.0398, +0.0689) | +60.00° | (0.500, 0.866, 0) | (+0.866, −0.500) = −30.00° |
| `wheel2` | (−0.0563, +0.0563) | +135.00° | (−0.707, 0.707, 0) | (+0.707, +0.707) = +45.00° |
| `wheel3` | (−0.0563, −0.0563) | −135.00° | (−0.707, −0.707, 0) | (−0.707, +0.707) = +135.00° |
| `wheel4` | (+0.0398, −0.0689) | −60.00° | (0.500, −0.866, 0) | (−0.866, −0.500) = −150.00° |

which gives, with `r_eff = 0.030089` m and `R = 0.0796` m,

```
mjsoccer tier 1:  w_i = (1 / 0.030089) · (  vx·sin α_i − vy·cos α_i − 0.0796·ω )
rSim            :  w_i = (1 / 0.02475 ) · ( −vx·sin α_i + vy·cos α_i + 0.090 ·vw )   (sslrobot.cpp:383-386)
```

so numerically

| command | mjsoccer wheel targets (rad/s) | rSim wheel targets (rad/s) |
|---|---|---|
| `vx = 2.5` m/s | `[ +71.96, +58.75, −58.75, −71.96 ]` | `[ −87.48, −71.42, +71.42, +87.48 ]` |
| `ω = 10` rad/s | `[ −26.45, −26.45, −26.45, −26.45 ]` | `[ +36.36, +36.36, +36.36, +36.36 ]` |
| a saturated uniform command | `[200,200,200,200]` requests `(vx, vy, ω) = (0, 0, −75.60)`; measured 59.4 achieved | `[163.05]×4` (the motor clamp, `sslrobot.cpp:65`) requests `(0, 0, +44.8)`, but the envs cap the yaw command at 10 rad/s first |

**`W1` (High).** The two action spaces are different objects. mjsoccer tier 1 exposes four raw
wheel speeds and does no kinematic conversion anywhere (`base.py:145-147` just scales by
`ctrlrange`); every rSoccer SSL env commands body velocity and lets `setDesiredSpeedLocal` do
the conversion. A policy trained on one cannot be evaluated on the other without an explicit
adapter. The repo's own scripted controller demonstrates the mismatch: `goto_ball_policy`
returns tier-0 body velocities and `tasks.py:311-312`/`tasks.py:331-333` note that on tier 1
"those land on three of the four wheels, which under-drives it".

**`W2` (High).** **Every sign is inverted**, and both radii differ. The relation is exactly
`A = −B` with `r_w: 0.02475 → 0.030089` and `R: 0.090 → 0.0796`. A wheel vector copied from
rSoccer into mjsoccer drives backwards and yaws the wrong way. mjsoccer's docs never state
this matrix at all (`docs/index.md:69-77` lists only the `ctrlrange`), so a porter has nothing
to check against — flagged as a documentation gap in §12. *(Closed by ticket 15:
`docs/index.md` now states the matrix, its radii and this sign inversion.)*

**`W3` (High).** Torque authority differs by 43× (`L1`). rSim's `wheelMotorMaxTorque = 0.070`
carries the comment "Maxon EC45 flat 50w with 18:60 gear ratio, **reduced to 3m/s2 accel**"
(`sslconfig.h:159`) — the limit is deliberate and is the upstream acceleration model. mjsoccer
sets `forcerange="-3 3"` (`models/robot_rollers.xml:827`), which at `r_eff = 0.030089` is a
399 N ground force per wheel — far above the `μ · m · g = 1.0 · 2.692 · 9.81 = 26.4 N`
friction budget, so mjsoccer is friction-limited at about 1 g and rSim is torque-limited at
about 0.38 g. Derived: mjsoccer's 9.81 m/s² friction ceiling is above its own forward static tip
threshold (`g · 0.04325 / 0.05399 = 7.86` m/s²), while rSim's 3.68 m/s² torque ceiling is
below its own (`g · 0.04625 / 0.07225 = 6.28` m/s²) — and below its own 7.85 m/s² friction
ceiling, so rSim is torque-limited and cannot tip itself by accelerating.

**Corrected (ticket 01).** The first version of this finding gave rSim's ceiling as
4.67 m/s², which assumes all four wheels project their force at unit weight. The
60/135/225/300° geometry limits the projection to `Σ|sin α| = 3.14626` forward and
`Σ|cos α| = 2.41421` lateral (the force-maximising sign pattern has `Σ f_i = 0`, so it costs
no yaw), giving **3.68 m/s² forward and 2.82 m/s² lateral** — which matches rSim's own
"reduced to 3m/s2 accel" comment on the weak axis. The conclusion (rSim torque-limited,
mjsoccer friction-limited) is unchanged. The same factors apply to every acceleration
quoted for either robot.

**`W4` (High).** Wheel speed and radius: 200 rad/s × 30.089 mm = **6.018 m/s** of surface speed
here (`models/robot_rollers.xml:827`) against 163.05 rad/s × 24.75 mm = **4.035 m/s** upstream
(`sslconfig.h:160`, converted at `ssl_gym_base.py:56-57`) — and the envs then cap that to
2.5 m/s (`static_defenders.py:76`). So the reachable speed differs by 2.4×.

**`W5` (High).** Tier 0 is not a cheaper version of rSim's body-velocity mode; it is a
different physical object. Its actuators act on **world-axis slide and hinge joints**
(`models/robot_kinematic.xml:152-154`), so `base._drive_robot` has to rotate the body-frame
command into world axes before writing it (`base.py:148-155`), and — crucially — the chassis
never touches the floor, so there is no friction limit, no slip, no traction budget and no
tipping. rSim's body-velocity mode still runs the command through the four wheel motors and
their Coulomb contacts (`sslrobot.cpp:376-391` then `Wheel::step`), so it retains slip
saturation and wheel-torque limiting. rSoccer's "kinematic-looking" interface is dynamically
grounded; mjsoccer tier 0's is not.

**`W6` (High).** Consequently the two tiers differ from rSim in opposite directions: tier 1
over-delivers force (no torque limit, real slip) and tier 0 over-delivers ideality (no slip,
no tipping, force cap of 19.13 N = 7.1 m/s²). Neither is a drop-in match. The repo's own
documentation makes the same point about the tiers relative to each other
(`docs/index.md:145-150`, `docs/index.md:199-212`).

**`W7` (Medium).** The motor models are not the same kind of controller. MuJoCo's `velocity`
actuator is a proportional servo whose output force is clipped: with `kv = 0.103696` and a
200 rad/s error it demands 20.7 N·m and delivers 3, so it saturates on any large error and the
hub reaches command in about 1 ms (derived from the hub's `Iyy = 5.93e-6` plus
`armature="1e-5"`). ODE's `AMotor` with `dParamVel`/`dParamFMax` is a velocity *constraint*
with a force bound, solved inside the LCP — so it tracks exactly whenever 0.070 N·m suffices
and clips hard when it does not, with no proportional region. Under saturation the two behave
similarly; near the target rSim is stiffer and mjsoccer has a finite tracking error.

---

## 7. Limits, kicker, dribbler and possession

| limit | mjsoccer | rSoccer/rSim |
|---|---|---|
| max linear speed | ~6.0 m/s achievable on tier 1 (derived from `W4`); 4.0 m/s hard `ctrlrange` on tier 0 (`models/robot_kinematic.xml:389-390`) | `max_v = 2.5` m/s enforced in the env (`static_defenders.py:76`, `dribbling.py:67`, `contested_possession.py:64`, `pass_endurance.py:71`), on top of a 4.035 m/s motor ceiling |
| max yaw rate | ~59 rad/s measured on tier 1; 30 rad/s `ctrlrange` on tier 0 | `max_w = 10` rad/s (same four files) |
| max wheel speed | 200 rad/s (`models/robot_rollers.xml:827-830`) | 163.05 rad/s (`sslconfig.h:160`, clamped at `sslrobot.cpp:65`) |
| acceleration limit | none — friction-limited at ≈9.81 m/s² (derived), above the 7.86 m/s² forward tip threshold | 0.070 N·m per wheel ⇒ ≈3.68 m/s² forward, 2.82 lateral (`sslconfig.h:159`; see the correction under `W3`), below its own 6.28 m/s² tip threshold |
| planar speed clipping | none; each wheel is clipped independently by `ctrlrange` | `‖(vx, vy)‖` scaled down to `max_v` (`static_defenders.py:143-146`) |
| kick | 200 N on a 0.1413 kg plunger against `stiffness="2000"`, for exactly one control step; per-plunger 250 ms lockout (`base.py:160-163`, `scene.py:52`, `models/robot_rollers.xml:271`) | `dBodySetLinearVel(ball, vx, vy, vz)` with `vx` from the env, gated on `isTouchingBall()`, plus a 10-substep `kickerCounter` (`sslrobot.cpp:167-200`) |
| kick speed | emergent; docs give 6-7 m/s for a flat kick (`base.py:35`) | exactly `kick_speed_x = 5.0` m/s (`static_defenders.py:78`, `contested_possession.py:66`); `pass_endurance.py:116` scales it by the action |
| chip | a second, independent sprung plunger with its own lockout (`base.py:160`, `models/robot_rollers.xml:282`) | `kick_v_z` exists in the API (`Robot.py:16`, `sslrobot.cpp:184,193`) but **no SSL env ever sets it** |
| kick cooldown | 250 ms per plunger = 16 control steps (`base.py:89`, `docs/index.md:91-96`) | none; `kickerCounter = 10` substeps only suppresses re-classification of the state, not re-kicking |
| dribbler | velocity servo on the bar joint at 314 rad/s, force-limited to ±0.05 N·m (`base.py:166`, `scene.py:43`, `models/robot_rollers.xml:824`) | a boolean (`sslrobot.cpp:152-155`) |
| possession / capture | ball↔dribbler **contact** over a 50 ms grace window (`base.py:36`, `base.py:188-211`) | an AABB test in the kicker frame — `\|along\| < 0.0315`, `\|lateral\| < 0.040`, `\|Δz\| < 0.020` — reported as `infrared` (`sslrobot.cpp:127-145`), and while `dribblerOn` a **hinge joint** welds the ball to the plate (`sslrobot.cpp:202-222`) |
| action clipping | `np.clip(action, −1, 1)` then scale by `max(hi, −lo)` (`base.py:108`, `base.py:145-147`); MuJoCo re-clips to `ctrlrange` | denormalise, rotate to local, then scale `(vx, vy)` down to `max_v` (`static_defenders.py:132-148`) |
| observation normalisation | divide by constants; the `Box` is `(−inf, inf)` and nothing clips (`base.py:79`, `base.py:213-239`) | divide, then `np.clip(·, −1.2, 1.2)` (`ssl_gym_base.py:27`, `:213-220`) |

**`L1` (High).** See `W3`. The 43× torque gap is the largest single number in this audit and it
is the upstream *acceleration model* being absent here.

**`L2` (High).** The yaw ceiling differs by ~6× and is the proximate cause of turnover.
`pinv(A) @ [200,200,200,200] = (0, 0, −75.60)` (measured) — a saturated uniform wheel command
*asks* for 75.6 rad/s, and 59.4 rad/s is achieved. rSoccer caps yaw at 10 rad/s
(`static_defenders.py:77`), comfortably below its own tipping regime. mjsoccer's action space
admits commands that tip the robot by construction, and `env.action_space.sample()` — which is
exactly what `examples/random.py:47` does — produces them regularly.

**`L3` (High).** Possession is not the same phenomenon. Upstream, `holdBall` zeroes the ball's
velocity and creates `dJointCreateHinge` between the kicker box and the ball
(`sslrobot.cpp:218-221`). It is released only when `isTouchingBall()` goes false or the
dribbler is switched off (`sslrobot.cpp:112`) — but the joint itself prevents the ball from
leaving the AABB, so in practice **a held ball cannot be taken by contact at all**. Two
consequences. (a) `SSLContestedPossession-v0` is not a possession-stealing task: the ball is
welded to the opponent and the env terminates if the opponent moves faster than 0.1 m/s
(`contested_possession.py:166-169`), so the only route to reward is shooting past it.
mjsoccer's `ContestedPossession-v0` terminates when *blue* holds the ball for 0.5 s
(`tasks.py:206`, `tasks.py:226-235`) — a genuinely different objective. (b) The joint is
created with **no anchor and no axis set**, so ODE defaults to a hinge about world-x through
the world origin; the relative pose is frozen at grab time, which makes it a weld with one
arbitrarily-oriented rotational DOF rather than a model of a dribbler. Physically arbitrary,
but behaviourally decisive.

**`L4` (High).** The kick is a mechanism here and an assignment there. mjsoccer's 200 N pulse
lasts one control step = 8 physics substeps = 16 ms and the plunger is stroke-limited to
30 mm, so exit speed depends on where the ball sits, how the chassis is moving and how the
plunger and spring interact — and the chassis takes the recoil. rSim writes the ball's velocity
directly and adds a correction from the ball's incoming velocity
(`vn = −(v·d) · kickerDampFactor` with `kickerDampFactor = 0.2`, and the tangential component
fully cancelled, `sslrobot.cpp:186-190`), so a kick both sets the speed exactly and *erases*
the ball's prior lateral motion. No recoil is ever applied to the robot.

**`L5` (High).** The dribbler is a 314 rad/s velocity servo here and a boolean there. On
mjsoccer the bar's holding force is a genuine friction budget: `forcerange="-0.05 0.05"` N·m at
9 mm radius is 5.6 N of tangential force (`models/robot_rollers.xml:824`), and `μ = 2.5` on the
contact (`models/scene_rollers.xml:22`). This is why the repo's own measurements show tier 1
losing raw ball contact on 45% of steps (`docs/index.md:190-197`) — the bar sticks, slips and
re-grips. rSoccer's ball is either welded or free.

**`L6` (High).** Possession *reporting* differs too. mjsoccer scans `data.contact` each step and
holds the flag for `POSSESSION_GRACE = 0.05` s (`base.py:36`, `base.py:188-211`), deliberately
*not* using the arm sensor because on this build a seated ball deflects the arm the wrong way
(measured −0.003 to −0.015 rad, `scene.py:45-50`). rSoccer's `infrared` is a geometric AABB
(`sslrobot.cpp:127-145`) with no hysteresis and no contact involvement. So `has_ball` in the
observation means "the bar touched the ball within 50 ms" here and "the ball's centre is inside
a 63 × 80 × 40 mm box in front of the plate" there.

**`L7` (High).** rSoccer has no kick cooldown, and mjsoccer's 250 ms per-plunger lockout is
load-bearing: without it "a held kick action pins the plunger against its 30 mm stop and the
ball is never released" (`base.py:84-87`). A policy that learns to hold `kick = 1` continuously
works upstream (it re-kicks whenever the ball is in the box) and is a no-op here after the
first pulse.

**`L8` (Medium).** mjsoccer fires the kicker whether or not a ball is present
(`base.py:161-163` checks only the lockout), so a spurious kick costs 16 steps of lockout.
rSim's `kick` no-ops unless `isTouchingBall()` (`sslrobot.cpp:177`). mjsoccer therefore has a
real cost for mistimed kicks and rSoccer does not.

**`L9` (Medium).** Observation ranges are handled oppositely. rSoccer clips every channel to
±1.2 (`ssl_gym_base.py:27,213-220`), so a policy never sees an out-of-range input. mjsoccer's
`Box` is `(−inf, inf)` and nothing clips (`base.py:79`). Combined with `L2` this bites:
`MAX_W = 30.0` is documented as "the tier-0 chassis servo `ctrlrange`" (`base.py:34`,
`docs/index.md:133`) but tier 1 reaches 59 rad/s, so the `omega` channel routinely reads about
2.0 — and `docs/index.md:135-137` claims "In normal play the values sit in roughly [−1, 1]",
attributing excursions only to "a tier-1 chassis pushed by another robot". That is a
**documentation/code disagreement**: tier 1's *own* drive exceeds `MAX_V` and `MAX_W` under a
saturated command, with nothing pushing it.

**`L10` (Low).** rSoccer's speed clipping uses `c = v_norm < self.max_v or self.max_v /
v_norm` (`static_defenders.py:145`). This works — a `True` multiplies as 1 — but it is a
Python idiom, not arithmetic, and it silently yields `c = True` rather than a float when under
the cap. Behaviourally identical; noted because it looks like a bug and is not.

---

## 8. Update ordering within a step

| stage | mjsoccer (`base.py:107-135`) | rSoccer (`ssl_gym_base.py:73-90` → `rsim.py:128-163` → `sslworld.cpp:267-303`) |
|---|---|---|
| 1 | `action = np.clip(action, −1, 1)` (`:108`) | `self.steps += 1` (`:74`) |
| 2 | `self.data.ctrl[:] = 0.0` — **every actuator is zeroed each step** (`:109`) | `_get_commands(action)` builds a `List[Robot]` (`:76`) |
| 3 | `_apply(...)` writes the controlled robots' drive, kick, chip, dribble (`:110`) | `send_commands` fills an `(n, 8)` array; robots absent from the list keep an **all-zero row** (`rsim.py:129-153`) |
| 4 | `_script()` writes the scripted robots' actuators (`:111`) | `setActions` — a zero row takes the `else` branch and calls `setDesiredSpeedLocal(0,0,0)`, i.e. scripted robots are **actively servo-braked** (`sslworld.cpp:310-316`) |
| 5 | `decimation` × `mj_step` (`:112-113`), default 8 × 2 ms | `robots[k]->step()` sets each AMotor's `dParamVel`/`dParamFMax` and runs the kicker state machine (`sslworld.cpp:272-275`) |
| 6 | `self.steps += 1`; decrement the plunger locks (`:114-115`) | 5 × { re-apply the ball drag force, `physics->step(dt·0.2)` } (`sslworld.cpp:279-301`) |
| 7 | `_obs()` — which first runs `_update_possession()` over the last substep's contact list (`:117`, `:214`) | `get_frame()` → `getState()`: reads positions, **fires the turnover reset**, finite-differences velocities (`sslworld.cpp:349-420`) |
| 8 | `_reward_done(action)` (`:118`) | `_frame_to_observations()` (`:86`) |
| 9 | truncation; finiteness check; ball-out; **tip check** (`:119-132`) | `_calculate_reward_and_done()` (`:87`) |
| 10 | render if human (`:133-134`) | render if human (`:88-89`) |

**`O1` (High).** rSim reports **finite-differenced** velocities, not body velocities. Ball
`v_x`, `v_y` are `(x − x_last) / timeStep` (`sslworld.cpp:368-369`), and robot `v_x`, `v_y`,
`v_θ` likewise (`sslworld.cpp:400-402`). Three consequences. (a) The first `getState()` after a
reset reports **zero** velocity for everything (`sslworld.cpp:371-376,404-409`), so the first
observation of every rSoccer episode is systematically wrong. (b) The divisor is `timeStep`
(25 ms), which is correct only because `getState` is called exactly once per env step — any
extra call corrupts the estimate, because `getState` mutates `this->state` as its side effect
(`sslworld.cpp:351-353`). (c) Velocity is one step stale and smoothed. mjsoccer reads `qvel`
directly (`scene.py:106-115`, `scene.py:142-143`), so its velocities are instantaneous and
exact. Ball `v_z` is never reported upstream at all — `Ball.v_z` stays at its `0.0` default
(`Ball.py:9`, and `FrameSSL.parse` never assigns it, `Frame.py:55-59`) — while mjsoccer's
observation includes it (`base.py:223`).

**`O2` (High).** The turnover reset runs **inside the observation getter**
(`sslworld.cpp:388-392`), between reading the robot's pose and pushing it into the state. The
pushed `x`, `y`, `theta` are the *pre*-reset values, while the simulator's actual state is the
post-reset one. So the observation on the reset step describes a tipped robot that no longer
exists, and the finite difference on the *following* step compares a tipped pose against an
uprighted one — producing a spurious velocity and angular-velocity spike. See `TO1`.

**`O3` (Medium).** mjsoccer zeroes all controls every step (`base.py:109`). This makes the kick
a true one-step pulse and forces `_script` to rewrite scripted commands each step (it does:
`tasks.py:223-224`, `tasks.py:263-264`), but it also means an unscripted robot's wheel servos
are commanded to **zero velocity** with the full ±3 N·m available — so mjsoccer's six static
defenders (`tasks.py:186-199`) are nearly immovable. rSoccer's scripted robots are also braked
to zero, but with 0.070 N·m, which is why `dribbling.py:143-145` and
`contested_possession.py:166-169` can meaningfully terminate the episode when a yellow robot
exceeds 0.05 / 0.1 m/s: upstream they are shovable and here they are not.

**`O4` (Medium).** Actions are written once and held across all substeps on both sides
(mjsoccer `base.py:110-113`; rSim `sslworld.cpp:270-275` then the loop at `:279`), so both are
zero-order-hold. But rSim re-applies the ball's drag force inside every substep
(`sslworld.cpp:281-294`) while mjsoccer's ball forces come only from contacts, which the
solver handles per substep automatically. Equivalent in kind.

**`O5` (Medium).** Reward timing differs. mjsoccer computes the observation *before* the reward
(`base.py:117-118`) and the tip check *after* the reward (`base.py:129-132`), so a tipping step
receives its task reward and then the −100 on top. rSoccer computes observation then reward
(`ssl_gym_base.py:86-87`) with the turnover reset having already fired inside step 7, so a
tipping step's shaping is computed from the pre-reset pose but the next step's from the
post-reset one — an uncompensated discontinuity in every potential-based term.

**`O6` (Low).** Neither simulator injects sensor latency, vision delay, actuation delay or
observation noise. Both are perfect-information, zero-latency. That is a shared gap relative
to hardware, not a difference between them.

---

## 9. Timestep and control rate

| quantity | mjsoccer | rSoccer/rSim |
|---|---|---|
| physics `dt` | 0.002 s (`models/scene_rollers.xml:4`) | 0.005 s (`0.025 × 0.2`, `sslworld.cpp:300`) |
| control `dt` | 0.016 s (`decimation = round(1/(60·0.002)) = 8`, `base.py:68-69`) | 0.025 s (`static_defenders.py:50` and the three siblings) |
| control rate | **62.5 Hz**, not the requested 60 (`docs/index.md:302-305`) | **40 Hz** |
| substeps per control step | 8 | 5 |
| frame skip | the `decimation` loop (`base.py:112-113`) | the 5-substep loop (`sslworld.cpp:279-301`) |
| horizon, seconds | 20 / 30 / 40 (`__init__.py:20-27`) | 25 / 30 / 120 (from `max_episode_steps` × 0.025, `rsoccer_gym/__init__.py:7-30`) |
| horizon, steps | 1250 / 1875 / 2500 (`max_steps = round(horizon_s / dt)`, `base.py:71`) | 1000 / 1200 / 4800 |
| who truncates | the env itself; `max_episode_steps=None` at registration (`__init__.py:33`) | Gymnasium's `TimeLimit`, via `max_episode_steps` (`rsoccer_gym/__init__.py:11,17,23,29`); the env always returns `truncated=False` (`ssl_gym_base.py:90`) |

**`T1` (High).** The control rate differs by 1.56× and the horizon in *steps* by up to 2×. Task
for task: `Dribbling-v0` is 40 s / 2500 steps here and `SSLDribbling-v0` is 120 s / 4800 steps
upstream; `ContestedPossession-v0` is 20 s / 1250 versus 30 s / 1200. Discount factors,
`n_step` returns, GAE horizons and replay-buffer sizes tuned on one are mistuned on the other,
and a per-step shaping coefficient means a different per-second rate.

**`T2` (Medium).** Substep structure differs: 8 × 2 ms against 5 × 5 ms. mjsoccer resolves
contact events 2.5× finer within a control interval, which matters most for the kick (a 16 ms
pulse resolved in 8 pieces) and for fast ball impacts (`P2`).

**`T3` (Medium).** Truncation is reported differently. mjsoccer returns a real `truncated`
flag from its own step counter (`base.py:119`, `base.py:135`); rSoccer hard-codes
`truncated=False` (`ssl_gym_base.py:90`) and relies on the wrapper. Any bootstrapping code
that distinguishes termination from truncation gets the wrong signal from rSoccer unless the
`TimeLimit` wrapper is present.

**`T4` (Low).** `control_hz` is a request, not a guarantee, and the repo says so
(`docs/index.md:302-305`): `round(1/(60 · 0.002)) = 8` gives 62.5 Hz. Anyone quoting "60 Hz"
for this repo is 4% off, and `max_steps` is computed from the real `dt`
(`base.py:69-71`), so a "20 s" episode is 1250 steps and not 1200.

---

## 10. Randomness, seeding and determinism

| aspect | mjsoccer | rSoccer/rSim |
|---|---|---|
| RNG | `self.np_random`, seeded via `super().reset(seed=seed)` (`base.py:96-97`) | the **global `random` module** — `random.uniform` (`static_defenders.py:222-228`, `contested_possession.py:217-218`, `pass_endurance.py:162-166`) |
| seeding path | `gym.Env.reset(seed=)` → `np_random` → every `uniform` call | `super().reset(seed=seed)` is called (`ssl_gym_base.py:93`) and seeds `self.np_random`, which the SSL layouts **never use** |
| randomised at reset: positions | agent and ball uniformly over the field with rejection sampling (`base.py:285-298`, `tasks.py:66-71` etc.) | `StaticDefenders`: ball + 6 defenders random, blue robot **fixed at (0,0)**; `ContestedPossession`: opponent + ball random, blue **fixed at (0,0)**; `PassEndurance`: ball and receiver random; `Dribbling`: **fully deterministic** (`dribbling.py:187-201`) |
| randomised at reset: yaw | uniform on (−π, π] for every placed robot (`tasks.py:69,95,137,212,214,246`) | defenders `random.uniform(0, 360)` (`static_defenders.py:227-228`); the blue robot always `theta = 0`; `Dribbling` all `theta = 180` |
| randomised at reset: velocity | none — zeroed (`scene.py:94,97,136-137`) | none — zeroed (`sslworld.cpp:427-428`, `sslrobot.cpp:292-298`) |
| observation noise | none | none |
| actuation noise | none | none |
| model randomisation | none; the `MjModel` is `lru_cache`d and shared, which `scene.py:187-189` notes blocks domain randomisation until the cache is dropped | none |

**`R1` (High).** rSoccer's SSL layouts are drawn from the **process-global** `random` module, so
`env.reset(seed=42)` does not reproduce a layout, two envs in the same process interfere, and
`random.seed()` anywhere in the training script silently changes the environment.
`ssl_gym_base.py:93` does call `super().reset(seed=seed)`, so `self.np_random` *is* seeded —
it is simply never consulted by the SSL envs. mjsoccer routes everything through
`self.np_random` (`base.py:289-294`, and every `_place`), so its resets are reproducible from
the seed. Note the corollary in the repo's own example: `examples/random.py:44` offsets the
seed per episode precisely because seeding works here.

**`R2` (High).** How much is randomised differs enormously. mjsoccer scatters the agent, the
ball and (for `ContestedPossession`/`PassEndurance`) the other robots uniformly over a 12 × 9 m
field at uniform yaw. Upstream, the *controlled* robot starts at the origin facing +x in
`StaticDefenders` and `ContestedPossession` (`static_defenders.py:232`,
`contested_possession.py:220`), and `SSLDribbling-v0` has **no randomness at all**
(`dribbling.py:187-201` writes five fixed poses). A policy trained upstream on `Dribbling` sees
one initial state for its whole training run; here it sees a fresh one every episode. This is
the difference between a memorisable task and a generalisation task.

**`R3` (High).** Initial yaw: uniform here, fixed at 0° (or 180°) for the controlled robot
upstream. Combined with `R2` this means an rSoccer SSL policy never has to solve "orient
yourself first", which is a substantial part of `GoToBall-v0` here (its termination requires
`cos > 0.9` toward the ball, `tasks.py:80`).

**`R4` (Medium).** Determinism guarantees differ in kind. mjsoccer is deterministic given the
seed, modulo floating-point reproducibility of MuJoCo itself; the shared cached `MjModel` is
safe because each env owns its `MjData` (`scene.py:187-189`). rSoccer is *not* deterministic
from the Gymnasium seed (`R1`), and additionally rebuilds the ODE world on every reset
(`X1`), which resets ODE's internal warm-start and contact caches — arguably *more* clean per
episode, but at the cost of never being reproducible from a seed.

**`R5` (Low).** Neither randomises reset velocities; both hard-zero them. Parity.

**`R6` (Low).** Neither injects observation or actuation noise, and neither randomises masses,
frictions or dimensions. Parity, and a shared gap for sim-to-real.

---

## 11. Reset behaviour, out-of-bounds and termination

| aspect | mjsoccer | rSoccer/rSim |
|---|---|---|
| what reset does to the world | `mujoco.mj_resetData` then write `qpos`/`qvel` (`base.py:98-103`, `scene.py:87-97`, `scene.py:130-137`) | **deletes and reconstructs the entire `SSLWorld`** — new ODE world, spaces, bodies, joints and surfaces (`robosim_py.cpp:56-60`) |
| settling inside reset | `Dribbling`/`ContestedPossession` run `seat_ball`: 0.35 s = 175 substeps of physics with the bar spinning (`base.py:252-283`) | the fresh `SSLWorld` constructor runs 30 substeps of `timeStep · 0.1` = 75 ms, then `replace()` snaps everything back (`sslworld.cpp:199-201`) |
| what is zeroed | robot linear+angular velocity, ball linear+angular velocity, ball quaternion to identity; all other joints left at `qpos0` by `mj_resetData` (`scene.py:94,97,135-137`) | chassis and wheel linear+angular velocity, wheel desired speeds, ball linear+angular velocity (`sslrobot.cpp:291-298`, `sslworld.cpp:427-428`) |
| ball z on reset | fixed by `qpos0` at 0.0415 (`models/ball.xml:5`); `place` writes only x, y (`scene.py:133-137`) | **preserved from the previous state** — `replace` reads `zz` and writes it back (`sslworld.cpp:425-426`) |
| overlap rejection | rejection sampling, 200 attempts, per-task clearance 0.5 / 1.0 / 1.5 m, `RuntimeError` on failure (`base.py:285-298`) | `KDTree` with `min_dist = 0.2` m in an unbounded `while`, **`StaticDefenders` only** (`static_defenders.py:241-252`); no rejection in the other three |
| out-of-bounds, ball | terminate when the ball centre leaves ±6.0 / ±4.5, unless it is a goal (`base.py:126-128`, `scene.py:282-283`) | per-env and asymmetric: `ball.x < 0 or \|ball.y\| > half_wid` (`static_defenders.py:187`, `contested_possession.py:180`) |
| out-of-bounds, robot | not checked | `robot.x < −0.2 or \|robot.y\| > half_wid` terminates (`static_defenders.py:179`), and entering the GK area terminates (`:183`) |
| goal detection | 3-D: past the line, inside the mouth, under the crossbar (`scene.py:286-292`) | 2-D: `ball.x > half_len` and `\|ball.y\| < half_goal_wid` (`static_defenders.py:190-194`) |
| divergence | `RuntimeError("simulation diverged")` on non-finite `qpos`/`qvel` (`base.py:120-125`) | `catch (...) { std::cout << "step error\n"; }` — every exception swallowed (`pworld.cpp:314-317`) |
| turnover | terminate, `−100.0` (`base.py:129-132`) | teleport upright and continue (`sslworld.cpp:388-392`) |
| spawn clearance | the robot spawns **1.0 mm above the floor** (measured: lowest roller vertex at z = 0.0209999 against the floor at 0.02), so every reset includes a 1 mm free fall | wheels spawn at `centerz = z − height/2 + wheelRadius − bottomHeight` = exactly `wheelRadius` above the plane (`sslrobot.cpp:35`) — flush |

**`X1` (High).** rSoccer rebuilds the world from scratch on every reset
(`robosim_py.cpp:56-60`: `delete m_world; m_world = new SSLWorld(...)`). That is 10 walls, up
to 22 robots × (1 chassis + 4 wheels + 1 kicker + 5 joints), a fresh `sur_matrix`, and 30
warm-up substeps, per episode. mjsoccer calls `mj_resetData` and writes `qpos` (`base.py:98`).
Behaviourally this means rSoccer's ODE solver state (contact caches, joint warm-starts) is
pristine each episode while mjsoccer's `MjData` retains only what `mj_resetData` does not clear
— and practically it makes rSoccer's reset far more expensive, which matters for
sample-throughput comparisons.

**`X2` (High).** mjsoccer runs real physics inside `reset` for two tasks. `seat_ball` spins the
bar up and rolls a ball in at 2 m/s from 200 mm over 0.35 s (`base.py:252-283`), with the
documented reasoning that placing the ball at the tangency "is a launcher, not a seat"
(`base.py:261-264`, `docs/index.md:214-220`). rSoccer places the ball 0.1 m in front of the
robot (`contested_possession.py:224`) and lets the hinge latch grab it on the first
dribbler-on step. So the initial possession state is an *earned* physical configuration here
and an assigned one there — and the mjsoccer version means the first observation after reset
already reflects 175 substeps of settled contact.

**`X3` (High).** Termination conditions barely overlap. mjsoccer has exactly four global ones —
task success, ball out of the rulebook field, non-finite state, controlled robot tipped
(`base.py:118-132`) — plus per-task success. rSoccer's SSL envs add rule terminations that have
no analogue here: robot out on the left (`robot.x < −0.2`), robot inside the goalkeeper area,
a yellow robot moving faster than 0.05 or 0.1 m/s, the ball "stopped" for 20 steps, the ball
outside the shooter-receiver bounding box, and holding for more than 15 steps
(`static_defenders.py:179-197`, `dribbling.py:143-155`, `contested_possession.py:166-182`,
`pass_endurance.py:141`, `:187-214`). Episode length distributions and therefore the value
function's scale differ completely.

**`X4` (High).** Goal detection ignores height upstream (`U2`) and requires it here. On a 0.70 m
mouth versus a 1.8 m mouth, plus 2-D versus 3-D, the same shooting policy scores at very
different rates.

**`X5` (Medium).** Overlap rejection exists here for every task and upstream for one. mjsoccer
rejection-samples with clearances of 0.5 m (`GoToBall`, `GoToBallShoot`, `Dribbling`), 1.0 m
(`ContestedPossession`) and 1.5 m (`PassEndurance`) and raises if it cannot place everything in
200 tries (`base.py:288-298`, callers in `tasks.py`). `SSLContestedPossession-v0` and
`SSLPassEndurance-v0` do no rejection at all, so upstream episodes can begin with robots
overlapping — and because rSim's reset rebuilds the world and runs 30 warm-up substeps
(`X1`), an overlap resolves as an explosive separation during the warm-up.

**`X6` (Medium).** The ball's z is preserved across rSoccer resets (`sslworld.cpp:425-426`
reads the current `zz` and writes it back), so an episode that ended with the ball in the air
begins the next one with the ball at that height. mjsoccer always restores `qpos0`
(`scene.py:133-137` writes x, y and the identity quaternion; z comes from `mj_resetData`).

**`X7` (Medium).** Failure handling is opposite. mjsoccer raises loudly on non-finite state,
explicitly "rather than training a policy on NaN observations" (`base.py:124`). rSim wraps its
entire step in `catch (...)` and prints to stdout (`pworld.cpp:314-317`), so a solver failure
becomes a silently skipped step.

**`X8` (Low).** mjsoccer's robot spawns 1.0 mm airborne (measured). Every reset therefore
includes a 0.14 m/s impact across four wheels — a ~0.09 N·s total impulse. Because the wheel
joints start at `qpos0 = 0`, the contact geometry at spawn is identical every time, so this is
a deterministic transient rather than a source of variance. Worth knowing when reading the
first few steps of a trajectory.

**`X9` (Low).** `mj_resetData` leaves the kicker, chipper and dribbler-arm joints at their rest
poses, which `RobotIds.place` explicitly relies on (`scene.py:88-89`). rSim's `resetRobot`
does not touch the kicker plate's hinge (it is locked at 0 anyway) and does not reset
`holdingBall` — so a robot that was holding the ball when `resetRobot` fired keeps its
`robot_to_ball` joint into the next episode unless `unholdBall` happens to be called. Latent;
in practice `Kicker::step` clears it on the next step when `isTouchingBall()` goes false
(`sslrobot.cpp:112`).

---

## 12. Robot turnover

**`TO1` (High).** The response to a tipped robot is opposite: mjsoccer terminates the episode
with a −100 penalty at 60° of tilt (`base.py:41-42`, `base.py:129-132`); rSim teleports the
robot upright at 26°, restores its spawn heading, zeroes every velocity and continues
(`sslconfig.h:120`, `sslworld.cpp:388-392`, `sslrobot.cpp:289-303`). Upstream episodes
therefore never end from tipping, never report it in the observation or `info`, and hand the
policy a free full-stop-and-reorient primitive; here tipping is a terminal event whose penalty
dominates the return.

**`TO2` (High).** The *reachability* of the tipped state differs, and for a reason neither
simulator documents: mjsoccer's CAD centre of mass is 3.459 mm off the spin axis (`M1`) and its
action space permits 59 rad/s of yaw (`L2`), which together put a saturated-command robot at
46.6° of tilt — marginally below its own 60° trigger. rSim's chassis is an axisymmetric
cylinder and its envs cap yaw at 10 rad/s, so the same failure mode does not exist upstream;
its 26° threshold fires on collision-induced tilt instead. The two turnover handlers are
therefore responding to different phenomena, not to the same one with different thresholds.

### 12.1 What each simulator actually does

**mjsoccer: detect, penalise, terminate.** Commit `3e19dc4` ("feat: add tipping penalty and
related logic to environments") added the whole mechanism — 24 lines in `base.py`, 20 lines of
test, and six identical doc paragraphs. The current code is:

```python
TIPPED_COS = 0.5                                              # base.py:41
TIPOVER_PENALTY = 100.0                                       # base.py:42

def tipped(self) -> bool:                                     # base.py:170-182
    return any(
        self.data.xmat[r.body].reshape(3, 3)[2, 2] < TIPPED_COS
        for r in self.robots[: self.n_controlled]
    )

if self.tipped():                                             # base.py:129-132
    reward -= TIPOVER_PENALTY
    terminated = True
    info["tipped"] = True
```

- **Trigger:** the robot's own +z, read as column 2 of its world rotation matrix, falls below
  `cos 60°` (`base.py:37-41`). Only the first `n_controlled` robots count, deliberately: "a
  scripted opponent knocked on its side is the agent's problem to exploit"
  (`base.py:171-176`).
- **Action taken:** a −100 reward and `terminated = True`. Nothing is teleported; the episode
  simply stops. Tier 0 can never trigger it — its chassis is a planar joint
  (`models/robot_kinematic.xml:152-154`) — so this is a tier-1-only condition.
- **Test coverage:** `tests/test_envs.py` (added in the same commit) rolls the chassis 90°
  by writing the quaternion directly and asserts `terminated`, `info["tipped"]` and
  `reward < −5.0`.
- **Documentation drift (`docs/` disagrees with the code).** All six environment pages state
  "The step pays `-10.0`" (`docs/environments/go_to_ball.md:136` and the same line in the other
  five). The constant was 10.0 when `3e19dc4` landed and is **100.0** today, changed in
  `cd495d7`. `docs/index.md` never mentions tipping at all — it is absent from the
  "terminates on" column of the environment index (`docs/index.md:37-47`) and from the
  Timing/Observation sections. Two concrete doc fixes: update the six `-10.0` occurrences, and
  add tipping to `docs/index.md:46`.

**rSim: teleport upright and carry on.**

```cpp
double  resetTurnOver = true;                                  // sslconfig.h:120
// inside SSLWorld::getState(), per robot:
robotDir = this->robots[i]->getDir(robotK);                    // sslworld.cpp:384
if (SSLConfig::World().getResetTurnOver() && robotK < 0.9)     // sslworld.cpp:388
{
    std::cout << "turnover " << robotK << ...;                 // sslworld.cpp:390
    this->robots[i]->resetRobot();                             // sslworld.cpp:391
}
```

`robotK` is `r[10]` of the ODE rotation matrix (`pobject.cpp:100-109`), which is exactly
`R[2][2]` — the same up-cosine mjsoccer uses. So the two thresholds are directly comparable:
**0.9 (25.84°) upstream against 0.5 (60°) here.** `resetRobot` (`sslrobot.cpp:289-303`) zeroes
the wheel targets, zeroes the chassis and wheel linear and angular velocities, re-writes the
chassis to `SSL_ROBOT_START_Z()` at the *current* x, y, and calls `setDir(this->_dir)` — where
`_dir` is the **episode's spawn heading**, assigned once in the constructor
(`sslrobot.cpp:239`) and never updated by `setDir` itself.

**Can rSim's robot tip at all?** Yes. Its chassis has a full 6-DOF `dBodyCreate`
(`pcylinder.cpp:40`) with no constraint on roll or pitch anywhere in `src/robosim` — no
`dJointCreatePlane2D`, no `dJointCreateAMotor` on the chassis, no `dBodySetFiniteRotationMode`
restriction. The only thing holding it level is the four wheel hinges to the ground. So the
turnover handler is not a belt-and-braces measure: it is upstream's *only* answer, and the
0.9 threshold means it fires at 26°, long before the robot is anywhere near over.

### 12.2 Why this happens here and not there

From `M1`, `W3` and `L2`, the causal chain in mjsoccer is:

1. The CAD chassis's CoM is **3.459 mm off the spin axis** (measured); rSim's is exactly 0.
2. Wheel torque is **±3 N·m** (`models/robot_rollers.xml:827`) against 0.070 N·m
   (`sslconfig.h:159`), so wheel commands are executed essentially instantly and the chassis is
   only friction-limited.
3. A saturated uniform wheel command requests **75.6 rad/s** of yaw (measured `pinv(A)`) and
   achieves **59.4 rad/s** (measured); upstream the env caps yaw at 10 rad/s
   (`static_defenders.py:77`).
4. The centrifugal moment on the offset CoM equals the restoring moment at
   `ω_tip = sqrt(g · d_edge / (h · d_com)) = 47.7 rad/s` (derived, with `d_edge = 43.25` mm to
   the front wheel line). **Measured** sweep: 6.0° of tilt at 42 rad/s, 43.2° at 55.6 rad/s,
   46.6° at 59.4 rad/s.
5. `TIPPED_COS = 0.5` is 60°, so a steady saturated spin sits **marginally below** the
   threshold — at 46.6° it is one contact impulse from crossing. `B1`, `C7` and `B6` supply
   exactly those impulses: a bumper catching another robot's motor housing, a wheel hub
   grazing a wall hull, a roller hitting the ball.

So turnover here is a *dynamic* failure produced by unbounded actuation acting on a small CAD
asymmetry — not a geometry problem. mjsoccer's static tip angle (38.7°, derived in §4) is
actually **better** than rSim's (32.6°); mjsoccer's CoM is 25% lower. Any fix aimed at
lowering the CoM is aimed at the wrong variable.

### 12.3 Why teleporting is the wrong fix

rSim's approach is available here and should be rejected:

- **Discontinuous state.** The robot's pose jumps by up to 26° of tilt in zero time. Any
  observation built from a finite difference — which upstream's velocities *are* (`O1`) — reads
  a physically impossible spike on the following step. Even with exact velocities, position
  jumps break the Markov transition the critic is fitting.
- **Broken momentum and energy.** `resetRobot` zeroes all six chassis velocity components and
  all four wheel bodies' (`sslrobot.cpp:292-298`). Angular momentum and kinetic energy vanish;
  the robot that was travelling at 3 m/s is at rest, and whatever it was pushing against is
  not.
- **Exploitable.** A free, instantaneous full stop with a restored heading is a *useful
  primitive*. A policy can learn to induce a turnover on purpose — dive at 26° of tilt — to
  teleport to a canonical orientation, cancel unwanted velocity, or escape a losing scrum. The
  yaw restoration makes it worse: `setDir(this->_dir)` snaps to the **spawn** heading
  (`sslrobot.cpp:302`), so tipping is a free "reorient to episode start" action.
- **Unrealistic and invisible.** Hardware does not right itself. And because the handler lives
  inside `getState()` and only prints to stdout (`sslworld.cpp:390`), neither the observation
  nor the `info` dict records that it happened — so turnover rate is not measurable from the
  env at all. It is a silent state corruption.
- **It hides the real bug.** In this repo, tipping is a symptom of `L1`/`L2`: an action space
  that permits 59 rad/s. Papering over it with a teleport would leave the 6× yaw-rate
  overshoot and the 43× torque overshoot in place, and those distort *every* manoeuvre, tipped
  or not.

### 12.4 Non-teleporting alternatives

**(a) Prevention by geometry, mass and inertia.**

*What:* move the CoM back onto the spin axis (the 3.459 mm offset is the whole problem, `M1`);
widen the wheel footprint from 79.6 mm toward rSim's 92.5 mm (`M2`); optionally lower the CoM
further from 54.0 mm.

- *Fidelity cost:* moving the CoM means editing `<inertial pos>` on
  `models/robot_rollers.xml:153`, i.e. **lying about the CAD**. If the real robot's CoM is
  3.5 mm off-axis, zeroing it makes the model less faithful, not more. Widening the wheel
  radius contradicts the CAD outright.
- *Implementation cost:* one attribute for the CoM (trivial); the footprint requires moving
  four wheel bodies and their 80 rollers, which is a model regeneration, not an edit.
- *Side effects on normal driving:* zeroing the CoM offset removes a real asymmetry that
  affects straight-line tracking; you would lose a genuine behaviour. Widening the footprint
  slows yaw response (larger `R` means more wheel speed per rad/s, `W2`).
- *Exploitable:* no.
- *Verdict:* the CoM edit is a **diagnostic**, not a fix — set it to zero temporarily to
  confirm the mechanism, then put it back. Do not ship it.

**(b) MuJoCo constraint-based restriction of roll and pitch.**

Four sub-options, in increasing fidelity cost:

| variant | how | fidelity cost | implementation cost |
|---|---|---|---|
| replace the free joint with `slide-x` + `slide-y` + `hinge-z` | what tier 0 already does (`models/robot_kinematic.xml:152-154`) | **total** — no roll, no pitch, no z, no wheel-ground normal force, no traction limit. This *is* tier 0, and `W5` explains why tier 0 is not rSim | zero: it already exists |
| add `slide-z` to the above | a 4-DOF planar+heave chassis | wheels still cannot generate a tipping moment; contact normal forces become meaningful again but roll/pitch response is gone entirely | moderate: new joint chain, and `RobotIds.place`/`velocity` (`scene.py:87-115`) branch on `free` and would need a third case |
| keep the free joint, add two `<equality>` constraints pinning roll and pitch to 0 | soft constraints with their own `solref`/`solimp`, so the robot *can* lean transiently and is pulled back | best of the constraint options: the transient lean under acceleration (documented as "a couple of degrees", `base.py:38-39`) survives, and the constraint has a tunable time constant | moderate: two `<equality>` elements plus tuning; MuJoCo solves them in the same LCP as the contacts, so no Python changes |
| `<weld>` or a rigid equality to the world | the robot cannot tilt at all | as bad as the planar joint, and it also fights the wheel contacts, which will show up as constraint-force noise | low |

The **soft roll/pitch equality** is the only variant in this family worth considering. Note
what it costs even so: the wheel-ground normal-force distribution becomes fictitious. Real
weight transfer under acceleration is what limits traction per wheel; a constraint that holds
the chassis level supplies the missing moment from the constraint solver instead of from the
contacts, so per-wheel normal forces — and therefore per-wheel friction budgets — stop being
physical. If any downstream work depends on per-wheel slip, this is disqualifying.

**(c) Restoring/righting torque through an actuator or `xfrc_applied`.**

*What:* compute the tilt each step and apply a saturated, first-order righting moment.

- *Fidelity cost:* high in kind, low in magnitude if tuned well. There is no physical
  mechanism on an SSL robot that produces a righting moment, so this is an invented force. But
  unlike a constraint, it is *bounded* and *visible*: you choose the saturation and the time
  constant, and you can log how much authority it used.
- *Implementation cost:* low — a few lines writing `data.xfrc_applied[body, 3:6]` in `step`
  before the substep loop, with a first-order filter on the commanded moment. No model change.
- *Side effects on normal driving:* directly proportional to the deadband. With a deadband
  above the normal driving lean (a few degrees) and a saturation just above the moment needed
  to arrest a 48 rad/s coning spin, normal driving is untouched. With a tight deadband it
  stiffens the chassis and suppresses the recoil that `docs/index.md:207-212` identifies as the
  tier-1 behaviour worth having.
- *Exploitable:* **yes, if unsaturated.** An unbounded righting torque is a free reaction
  moment against the world, and a policy will find that it can yaw or translate by inducing
  tilt and harvesting the correction. A hard saturation and a rate limit are mandatory, and
  the applied impulse must be logged and audited.
- *Verdict:* the best *compatibility* device, and strictly better than teleporting because it
  is continuous and bounded. But it treats the symptom.

**(d) Contact-parameter tuning.**

*What:* reduce the impulses that convert a marginal 46° coning spin into a fall — soften
`solref` on the roller/floor and chassis/wall contacts, raise `solimp`'s width, reduce roller
`μ` from 1.0, raise `condim` to 6 on the roller contacts to add rolling resistance, or raise
`impratio` so friction is harder and slip is less abrupt.

- *Fidelity cost:* low to moderate, and *in the right direction*: `μ = 1.0` on a rubber roller
  against carpet (`models/robot_rollers.xml:311`) is at the high end, and rSim's effective
  tangential figure is 0.8 (`sslconfig.h:157`). Softening the wall and robot-robot contacts
  from `solref="0.004 1"` toward the model default `0.02 1` is also defensible — 4 ms is a very
  stiff contact for a plastic bumper.
- *Implementation cost:* low; XML attribute edits, no code.
- *Side effects on normal driving:* real. Lowering roller `μ` lowers the acceleration ceiling
  (which is arguably desirable, see (e)) and changes every dribbling and braking behaviour.
  Softer contacts increase penetration and can make the 1 mm spawn clearance (`X8`) matter.
- *Exploitable:* no.
- *Verdict:* worth doing on its own merits regardless of turnover, but it will not fix a
  centrifugal instability — it only removes the trigger, not the marginality.

**(e) Accept tipping; fix the action space and shape the reward.**

*What:* keep the current terminate-and-penalise, but bring the action limits into a physical
range so the marginal regime is not reachable: cap the wheel `forcerange` toward a real motor
figure, and/or cap the *commanded* yaw rate so `|ω_cmd| ≤ 30` rad/s (matching `MAX_W`, which
the observation already assumes, `base.py:34`).

- *Fidelity cost:* **negative** — this makes the model more faithful, not less. rSim's
  0.070 N·m carries the explicit comment "reduced to 3m/s2 accel" (`sslconfig.h:159`); a real
  SSL robot has a motor torque limit and a current limit, and `±3 N·m` per wheel — 399 N of
  ground force against a 26 N friction budget — is not a physical actuator. Making the action
  space respect the motor is the correction, not a workaround.
- *Implementation cost:* one XML attribute per wheel actuator, or a clip in
  `base._drive_robot`. Trivial. The XML edit is better because `docs/index.md:79-82` promises
  that "the bounds come from the model rather than from a constant in the Python".
- *Side effects on normal driving:* yes, and they are the point. The robot becomes
  acceleration-limited, which is what hardware is. Existing trained policies become invalid.
- *Exploitable:* no. And the −100 termination stays as a safety net with a bounded, logged
  rate.
- *Verdict:* **this is the fix.**

### 12.5 Recommendation

**Do this, in order.**

1. **Bound the actuation to something physical (e).** Reduce the wheel `forcerange` in
   `models/robot_rollers.xml:827-830` from `±3` to a figure justified by the motor and gearbox,
   and re-derive the achievable acceleration and yaw rate from it. Reduce `ctrlrange` from
   `±200` rad/s if the resulting surface speed exceeds what the drivetrain does. Success
   criterion: the reachable yaw rate falls below `MAX_W = 30` rad/s, which is well under the
   derived `ω_tip = 47.7` rad/s, and the observation normalisers become honest again (`L9`).
2. **Audit and soften the contacts (d).** Bring the roller `μ` from 1.0 toward 0.8 and the
   wall/bumper `solref` from `0.004 1` toward the model default, and re-measure. Cheap and
   independently justified.
3. **Confirm the mechanism with the CoM diagnostic (a).** Temporarily zero
   `models/robot_rollers.xml:153`'s `ipos` x/y and re-run the tip sweep. If tipping disappears,
   `M1` is confirmed and the CAD asymmetry is not a bug to fix but a fact to design around.
   **Revert immediately after.**
4. **Keep terminate-and-penalise, and start logging it.** `info["tipped"]` already exists
   (`base.py:132`); add the tilt angle and the step index so the rate and the time-to-tip
   distribution are measurable. Consider whether −100 is the right magnitude once the rate
   drops: at the current value one tip erases 5-10 successful episodes of return, which makes
   the gradient signal for everything else very noisy.
5. **Only if turnover persists after 1-3**, add a **saturated righting moment (c)** with a
   deadband above the normal driving lean, a first-order time constant, and mandatory logging
   of the applied impulse. Treat any episode where the righting moment saturates as a failure
   to be investigated, not as a solved case.

**What I would not do.**

- **Do not teleport** (§12.3). It is discontinuous, non-Markov, exploitable and invisible.
- **Do not replace the free joint with a planar joint chain** to "fix" tier 1. That is tier 0,
  which already exists and which `W5` shows is *not* equivalent to rSim's body-velocity mode.
  It would delete the traction, slip and recoil behaviour that `docs/index.md:145-150` gives as
  the entire reason tier 1 is the default.
- **Do not weld roll and pitch to zero.** It fights the wheel contacts and makes per-wheel
  normal forces fictitious.
- **Do not falsify the CoM or the wheel geometry** as a permanent change. Use the CoM edit as a
  one-off diagnostic and revert it.
- **Do not raise `TIPPED_COS` toward rSim's 0.9.** Firing at 26° would terminate on the
  marginal coning spin that a saturated command produces today, turning a rare failure into a
  constant one, and 60° is well-argued in `base.py:37-41`.
- **Do not lower `TIPOVER_PENALTY` to make training smoother** before fixing the cause. A
  cheap penalty with a high rate teaches the policy that tipping is affordable.

---

## 13. Experiments and metrics

None of the following was run; this is the harness and the acceptance criteria.

### 13.1 Quantities to measure

| metric | definition | why |
|---|---|---|
| tip rate | tips per episode-minute of simulated time, from `info["tipped"]` (`base.py:132`) | the headline number; must be comparable across configurations of different episode length |
| time-to-tip | distribution of simulated seconds from reset to the first tip, censored at truncation | separates "tips immediately from a bad spawn" from "tips after a collision" |
| chassis roll/pitch | mean and max `arccos(xmat[body][2,2])` per episode, plus the 99th percentile | the marginality measure; the coning-spin regime shows up as a high mean with no tips |
| yaw rate | distribution and max of `qvel[qvel_adr+5]`, and the fraction of steps above 30 rad/s | tests whether fix (e) actually bound the reachable yaw |
| contact impulse | per-step `sum(abs(data.efc_force))` restricted to the robot's contacts, reported as p50/p95/p99/max | identifies which contacts supply the trip impulse; discriminates fix (d) |
| speed and acceleration before a tip | the 0.5 s of `velocity()` history preceding each tip | tests the `M1` mechanism directly: tips should cluster at high `\|ω\|`, not high `\|v\|` |
| righting authority used | if (c) is implemented, the p95 and max of the applied moment, and the fraction of steps at saturation | a saturating righting torque is a hidden failure |
| possession duration | consecutive `holds(0)` steps per possession, mean and p95 (`base.py:184-186`) | the side-effect check: any contact or actuation change must not degrade retention |
| episode return | mean and std per env, per config | the thing that actually matters |
| sample efficiency | steps to reach a fixed success rate on `GoToBall-v0` and `Dribbling-v0` | whether the fix helps or hurts learning |
| wall-clock cost | steps/s at fixed batch size, comparable to `docs/index.md:157-160` | (b) and (c) add constraints/forces; (e) adds nothing |

### 13.2 Scenarios

Each is a scripted, deterministic sequence written as a `pytest` case against the unwrapped
env, so nothing depends on a policy.

1. **Saturated uniform spin.** `place` at the origin, command `w = [k, k, k, k]` for
   `k ∈ {0.2, …, 1.0}`, 5 s. Reproduces the measured sweep in §12.2 and is the primary
   regression test for fix (e).
2. **Head-on robot-robot.** Two robots on `ContestedPossession-v0`, facing, closing at
   {1, 2, 3, 4} m/s. Isolates `B1` (mesh-on-mesh) as a trip source.
3. **Wall graze.** Drive at {2, 4, 6} m/s at {5°, 15°, 30°} to the wall at ±4.5. Isolates
   `C7`/`B7`.
4. **Hard turn while dribbling.** `Dribbling-v0`, `seat_ball`, then a step change to a
   commanded yaw of {10, 20, 30, 50} rad/s. The scenario that matters most for match play, and
   the one where a righting torque would most distort behaviour.
5. **Kick recoil.** `GoToBallShoot-v0`, seated ball, fire the kicker at {0, 2, 4} m/s of
   chassis speed. Measures the recoil pitch that `L4` says rSim has no analogue for.
6. **Random-action rollout.** `env.action_space.sample()` on all six ids, both tiers. The
   baseline tip rate, and the one that reflects early training.
7. **Baseline drive.** `goto_ball_policy` on tier 0 (`tasks.py:279-297`) and a wheel-space
   adaptation on tier 1, to confirm that a competent controller does not tip.

### 13.3 Sample sizes and seeds

Seeds `0..49` per (scenario × configuration), fixed and shared across configurations so
differences are paired; 50 episodes gives a ±3 percentage-point standard error on a 10% tip
rate, which is enough to detect the halvings that matter. For the learning-curve metrics, 5
seeds per configuration on `GoToBall-v0` and `Dribbling-v0` only — the rest is too expensive
to be worth it before the physics is settled. Report medians and interquartile ranges, not
means, for anything skewed (time-to-tip, contact impulse).

### 13.4 Harness shape

Given this repo's layout, everything is a read-only addition alongside `tests/test_envs.py`
and `examples/random.py`; nothing needs a new dependency.

```bash
# baseline tip rate, all envs, both tiers, 50 seeds
uv run python examples/random.py --env GoToBall-v0 --tier 1 --episodes 50 --seed 0

# the scripted scenarios as a marked pytest suite (new file, e.g. tests/test_turnover.py)
uv run python -m pytest tests -q -k turnover
uv run python -m pytest tests -q -k "spin or wall or dribble_turn"

# per-episode metrics to CSV for the distributions above (new script, e.g.
# scripts/tip_sweep.py, following scripts/make_gifs.py's shape: build the env unwrapped,
# drive it from a fixed schedule, write one row per step)
uv run --group dev python scripts/tip_sweep.py --env Dribbling-v0 --tier 1 \
    --seeds 0-49 --scenario dribble_turn --out /tmp/tips.csv

# learning curves, using the existing sb3 examples as the trainer
uv run --group rl python examples/sac_agent.py --env GoToBall-v0 --tier 1 --seed 0
```

The existing `tests/test_envs.py` tip test (added in `3e19dc4`) is the template: build with
`gym.make(env_id, tier=1).unwrapped`, write `data.qpos` directly for the initial condition,
step, and assert on `info`. Extend it with the metrics above rather than writing a parallel
harness.

### 13.5 Acceptance criteria

The choice is settled when, relative to the current `147d2af` baseline on the same 50 seeds:

1. **Tip rate** under random actions on tier 1 falls below 1 per episode-minute on every one of
   the six ids, and is **zero** under scenario 7 (competent drive).
2. **Reachable yaw rate** p99 is below 30 rad/s across scenarios 1-6, so `MAX_W`
   (`base.py:34`) becomes an honest normaliser and `docs/index.md:135-137`'s claim becomes true.
3. **Max chassis roll** p99 stays below 15° in scenarios 4-7 — i.e. the marginal coning regime
   is not merely below the 60° trigger but genuinely gone.
4. **Possession duration** p50 on `Dribbling-v0` degrades by no more than 5% against the
   figures in `docs/index.md:199-212` (111-112 of 120 idle steps on tier 1). A contact-tuning
   change that fixes tipping by ruining retention is not a fix.
5. **Achievable acceleration** is within a factor of 1.5 of a defensible motor-derived figure,
   and is *stated* in `docs/index.md` alongside the action table.
6. **Episode return and sample efficiency** on `GoToBall-v0` and `Dribbling-v0` are no worse
   than baseline at equal environment steps, across 5 seeds.
7. **Wall-clock** steps/s is within 10% of baseline (this rules (b) and (c) out if they are
   expensive, and is free for (e)).
8. If a righting moment (c) is in play, it **saturates on fewer than 0.1% of steps**; above
   that, the underlying cause is not fixed.

Criteria 1-3 discriminate the fixes; 4-7 are the guardrails that stop a fix from being a
regression somewhere else.

---

## Appendix: source inventory

Every file read to produce this document, and what it was read for.

### This repository (`/home/mateus/doc/ssl-rl`, `147d2af`)

| file | role |
|---|---|
| `src/mjsoccer/base.py` | the Gymnasium contract: step ordering, action scaling, observation, possession, tipping, timing constants |
| `src/mjsoccer/scene.py` | multi-robot assembly via `MjSpec`, contact-pair replication, field constants and their CAD assertion, `RobotIds`/`BallIds` accessors |
| `src/mjsoccer/tasks.py` | the six task definitions: layouts, scripts, rewards, terminations; the scripted `goto_ball_policy` |
| `src/mjsoccer/__init__.py` | env ids and per-task horizons; `max_episode_steps=None` |
| `models/scene.xml`, `models/scene_rollers.xml` | `<option>` (timestep, integrator, solver, cone, gravity) and the three explicit contact pairs |
| `models/field.xml` | floor plane, wall and goalpost collision meshes, contact bitmasks and frictions |
| `models/ball.xml` | ball radius, mass, `condim`, friction, `solref`, bitmask |
| `models/robot_kinematic.xml` | tier 0: the three chassis joints, chassis inertial, chassis/dribbler/kicker/chipper bodies and geoms, the six actuators |
| `models/robot_rollers.xml` | tier 1: the free joint, chassis inertial, four wheel hubs with 20 rollers each, kicker/chipper/dribbler bodies, the seven actuators, all collision geom parameters |
| `docs/index.md` | the repo's stated contract for actions, observations, tiers, possession, field constants and timing — the source of the doc/code disagreements in `L9` and §12.1 |
| `docs/environments/*.md` (all six) | per-task contracts and the `-10.0` tipping figure that the code no longer matches |
| `docs/analysis/ssl-simulator-rsoccer-comparison.md` | a pre-existing audit of the same comparison at the same revisions; read for cross-checking, not cited as evidence |
| `tests/test_envs.py` | the tipping regression test and the harness shape reused in §13.4 |
| `examples/random.py` | the random-rollout entry point used in §13.4 |
| `pyproject.toml` | dependency pins (`mujoco>=3.11.0`), the `dev` and `rl` groups used in §13.4 |
| `git show 3e19dc4` | the commit that introduced tipping detection; establishes that the penalty was 10.0 at introduction |
| `git log -S'TIPOVER_PENALTY = 100.0'` | establishes that `cd495d7` changed it to 100.0 without updating `docs/` |

### `robocin/rSim` (`/tmp/rsim-upstream`, `b413932`)

| file | role |
|---|---|
| `src/robosim/sslconfig.h` | **the authoritative parameter table**: field types, ball radius/mass/friction/bounce/damping, robot radius/height/wheel radius/masses/frictions/motor torque and RPM, kicker geometry and damping, `resetTurnOver` |
| `src/robosim/sslworld.cpp` | world construction, wall layout, the surface matrix, the wheel and ball collision callbacks, the 5-substep step loop with the ball drag force, `setActions`, `getState` (including the turnover reset and the finite-differenced velocities), `replace` |
| `src/robosim/sslworld.h` | `fullSpeed = false` — which selects `dWorldStep` over `dWorldQuickStep` |
| `src/robosim/sslrobot.cpp` | wheel construction and the AMotor model, the kicker plate and its locked hinge, `isTouchingBall`, `kick`, `holdBall`/`unholdBall`, `resetRobot`, `getDir`, `setXY`, `setDir`, `setDesiredSpeedLocal` (the omni matrix) |
| `src/robosim/sslrobot.h` | `SSL_ROBOT_START_Z()`, the `Wheel`/`Kicker` interfaces, `KickStatus` |
| `src/robosim/sslconfig.cpp` | empty (0 lines) — all configuration is header-inline |
| `src/robosim/utils.{h,cpp}` | `smallestAngleDiff` (degrees) and `fric` (the `-1 → dInfinity` convention) |
| `src/robosim/robosim_py.cpp` | the pybind11 surface, and the fact that `reset` destroys and reconstructs the world |
| `src/robosim/physics/pworld.{h,cpp}` | ODE world/space creation, the hand-enumerated collide calls, `PSurface` defaults, the `dWorldStep`/`QuickStep` branch, the swallowed exception |
| `src/robosim/physics/pobject.{h,cpp}` | body/geom pose accessors; `getBodyDirection(…, k)` where `k = r[10]` is the up-cosine used by the turnover test |
| `src/robosim/physics/pball.cpp` | sphere geom and `dMassSetSphereTotal` |
| `src/robosim/physics/pcylinder.cpp` | cylinder geom and `dMassSetCylinderTotal(…, 1, …)` — the mis-axed inertia in `M5` |
| `src/robosim/physics/pbox.cpp` | box geom and `dMassSetBoxTotal` |
| `src/robosim/physics/pfixedbox.cpp` | static wall boxes (mass 0, geom-only) |
| `src/robosim/physics/pground.cpp` | `dCreatePlane(space, 0,0,1, 0)` |
| `CMakeLists.txt`, `setup.py` | `find_package(ODE REQUIRED)` with no precision constraint — the basis for the unresolved CFM in `P1` |

Read and found not to bear on the SSL comparison: `vssworld.*`, `vssrobot.*`, `vssconfig.*`
(the IEEE VSSS league).

### `robocin/rSoccer` (`/tmp/rsoccer-upstream`, `3adf7c3`)

| file | role |
|---|---|
| `rsoccer_gym/Simulators/rsim.py` | the `robosim` wrapper: command packing, `get_frame`, and `reset` delegating to the world rebuild |
| `rsoccer_gym/ssl/ssl_gym_base.py` | the SSL base env: step/reset ordering, `max_v`/`max_w` derivation from the motor spec, the ±1.2 normalisers, `truncated=False` |
| `rsoccer_gym/Entities/Frame.py` | `FrameSSL.parse` — the 11-value-per-robot state layout, and the units docstring ("degrees, degrees/s") |
| `rsoccer_gym/Entities/Robot.py` | the command dataclass: `wheel_speed`, `v_wheel0..3`, `kick_v_x`, `kick_v_z`, `dribbler`, `infrared` |
| `rsoccer_gym/Entities/Ball.py` | the ball dataclass; `v_z` exists and is never assigned |
| `rsoccer_gym/Entities/Field.py` | the field-parameter dataclass populated from `getFieldParams` |
| `rsoccer_gym/ssl/ssl_hw_challenge/static_defenders.py` | `field_type=2`, `time_step=0.025`, `max_v=2.5`, `max_w=10`, `kick_speed_x=5.0`, the rule terminations, the KDTree placement, the global-`random` layout |
| `rsoccer_gym/ssl/ssl_hw_challenge/dribbling.py` | the fully deterministic layout, the checkpoint reward, the yellow-robot-moved termination |
| `rsoccer_gym/ssl/ssl_hw_challenge/contested_possession.py` | the welded-ball premise, the collision termination, the goal test |
| `rsoccer_gym/ssl/ssl_hw_challenge/pass_endurance.py` | the yaw-only action space, the `infrared`-based success test, the "wrong ball" termination |
| `rsoccer_gym/ssl/ssl_hw_challenge/__init__.py`, `rsoccer_gym/ssl/__init__.py` | exports; `ssl/__init__.py` is empty |
| `rsoccer_gym/__init__.py` | the four SSL registrations and their `max_episode_steps` |
| `pyproject.toml`, `setup.py` | `rc-robosim = "^1.2"` — establishing that the pinned rSim source is the right physics reference |

Read and found not to bear on the SSL comparison: `rsoccer_gym/vss/*`,
`rsoccer_gym/Render/*`, `rsoccer_gym/Utils/kdtree.py` (a plain 2-D KD-tree, no behavioural
content beyond the `min_dist` in `X5`).

### Documentation

| source | used for |
|---|---|
| <https://mujoco.readthedocs.io/en/stable/XMLreference.html> | `option/timestep`, `integrator` (`implicitfast` drops Coriolis and centrifugal terms), `solver`, `iterations`, `ls_iterations`, `cone` (pyramidal default vs elliptic), `impratio`, `geom/condim` (1/3/4/6), `solref` (time constant, damping ratio), `solimp` (five impedance parameters), `friction` (sliding, torsional, rolling), `margin`, the `contype`/`conaffinity` bitmask rule, `joint/armature`, `joint/damping`, `velocity/kv`, `actuator/forcerange` |
| <https://ode.org/wiki/index.php/Manual> | ERP default 0.2 and its "proportion of joint error fixed per step" semantics; CFM defaults 1e-5 (single) / 1e-10 (double); `dWorldStep` O(m³) direct vs `dWorldQuickStep` O(m·N) iterative; `dWorldSetQuickStepNumIterations` default 20; `dContactApprox1`/`dContactMu2`/`dContactFDir1`/`dContactSoftCFM`/`dContactBounce`+`bounce_vel`; `mu` as the maximum tangential-to-normal force ratio; `dParamVel`/`dParamFMax` on an AMotor; `dMassSetCylinderTotal`'s direction argument (1=x, 2=y, 3=z) |

### Measured quantities and how

All read out of the compiled `MjModel`/`MjData` with `scene.build(1, 0, tier)` followed by
`mujoco.mj_forward`, without modifying any tracked file:

- `nq`/`nv`/`nu`/`nbody`/`ngeom`/`njnt`; `opt.*` (integrator, solver, iterations, cone,
  impratio, density, viscosity, wind, tolerance) and the model-wide `o_solref`/`o_solimp`.
- `body_subtreemass[robot]` = 2.69207 kg; `subtree_com[robot]` = (−0.00345505, 0.00016361,
  0.07399425) → CoM 53.99 mm above the floor plane and 3.459 mm off the spin axis.
- Per-body mass, `ipos` and `inertia` for the chassis, dribbler arm and bar, kicker, chipper
  and the four wheel hubs; 80 rollers at 0.000440446 kg each.
- Effective wheel radius 30.089 mm, from the world z of `b0_wheel1` (0.0510892) minus the
  lowest world-space vertex of any `b0_*roller*` collision mesh (0.0209999); the same
  computation gives the 1.0 mm spawn clearance in `X8` and the 0.4% roller-crown ripple in
  `B2`.
- Wheel mount positions and angles: 79.60 mm at exactly 60/135/225/300°.
- Per-wheel spin axis (`xmat · jnt_axis`) and ground drive direction (`axis × ẑ`), from which
  the tier-1 inverse-kinematics matrix in §6 is assembled and compared against
  `sslrobot.cpp:383-386`.
- The uniform-wheel-command tip sweep in §12.2: `place` at the origin, constant
  `w = [k,k,k,k]`, 190 control steps, tracking `min xmat[2,2]` and `max |qvel[qvel_adr+5]|`.
- World-space AABBs of the wall, goalpost, field-line and robot collision meshes, giving the
  wall extents (±6.32/±4.82, 0.20 m tall), the goalpost extents (x ∈ [6.0, 6.2], |y| ≤ 0.94,
  z ≤ 0.17) and the robot's 87.7 × 88.0 × 156 mm collision envelope.
- Actuator `ctrlrange`, `forcerange`, `gainprm` and `biasprm` for all seven tier-1 and six
  tier-0 actuators; the three compiled contact pairs with their `condim`, 5-element friction,
  `solref` and `solimp`.
