# Map: physics fidelity for the mjsoccer SSL simulator

## Destination

A settled, ordered **fix specification**: every behaviourally significant MuJoCo parameter in
`models/*.xml` and every actuation/observation constant in `src/mjsoccer/` either justified by
hardware/first principles, or explicitly recorded as a deliberate approximation — with the
acceptance gates that prove it. The map is done when nothing is left to decide and someone can
execute the edits without re-opening a question.

Reaching it does **not** mean rSoccer parity. It means the numbers are defensible.

## Notes

**Domain.** RoboCup SSL Division A omni robot in MuJoCo (`mjsoccer`), 6 Gymnasium tasks, two
physics tiers. Reference audits, both cited to source, validated against each other:
- `docs/research/rsoccer-vs-mjsoccer.md` — 80 differences vs `robocin/rSoccer` + `robocin/rSim`, ids `P*`/`U*`/`B*`/`M*`/`C*`/`W*`/`L*`/`O*`/`T*`/`R*`/`X*`/`TO*`. Cite these ids.
- `docs/analysis/ssl-simulator-rsoccer-comparison.md` — independent audit. Sound except its CoM
  comparison, which omits the 20 mm floor riser (`models/field.xml:42`, floor plane at z=0.02):
  mjsoccer's CoM is 53.99 mm above the carpet against rSim's 72.25 mm, i.e. **18.3 mm lower**,
  not 2 mm higher. Its two findings the other audit missed are real and verified: the dribbler's
  affine action map, and rSim's latching dribbler.
- Upstream clones for citation: `/tmp/rsim-upstream` (`b413932`), `/tmp/rsoccer-upstream` (`3adf7c3`).

**Skills every session should consult.** `mattpocock-skills:grilling` and
`mattpocock-skills:domain-modeling` for the `grilling` tickets; `mattpocock-skills:research` for
the `research` tickets. CodeGraph (`codegraph_*`) before reading source by hand.

**Standing constraints, settled while charting (2026-09-08).**
1. **Destination is hardware fidelity, not rSoccer parity.** rSoccer/rSim is a differential
   reference only. Of rSim's numbers only two are physically sourced —
   `wheelMotorMaxTorque = 0.070` and `wheelMotorMaxRPM = 1557` (`sslconfig.h:159-160`, "Maxon
   EC45 flat 50w with 18:60 gear ratio, reduced to 3m/s2 accel"). Everything else in rSim is an
   ODE workaround and is not a target.
2. **Motor envelope: Maxon EC 45 flat 50 W, direct drive.** ~~18:60 gearing~~ — **superseded by
   `01`**, which found a real drivetrain spec after all. RobôCIn's 2024 ETDP §1.2 records the move
   to direct drive on a 62 mm wheel at ~10 % of stall torque, and this CAD's wheel meshes are
   named `Direct_Transmission_*` with `r_eff = 30.089` mm = a 60.18 mm wheel. rSim's 0.070 N·m
   and 1557 rpm describe the *previous, geared* generation. The charting answer picked 18:60 only
   because no spec was thought to exist; the spec exists, so this is the upgrade that answer
   asked for first. Reversible — say so and `01` re-derives on 18:60.
3. **Existing trained policies are invalidated.** No compatibility tier, no second tier-1
   physics. `models/` and `runs/` checkpoints get retrained.
4. **Tier 0 is kept** as the ideal-kinematics debug tier and gets the *same speed/acceleration
   envelope* as tier 1 (not the same mechanism), so one action head still means one thing.

**Mode.** Planning. Tickets produce decisions; the edits happen after the map closes. The
exceptions are the `task` tickets, which run measurements — those may touch `models/` only as a
reverted diagnostic, and must revert.

## Decisions so far

<!-- one line per closed ticket -->

- [Physical constants](issues/02-physical-constants.md): ball mass 0.046 kg and radius 0.0215 m
  are rulebook-fixed and already right; ball↔floor μ 0.35 and `μ_roll` 0.0009 are measured-correct
  and were merely unjustified; the ball needs **COR ≈ 0.50** (it has none) and **β = 0.5** (it has
  MuJoCo's 0.4 default); roller↔floor μ should be **0.25**, not 1.0; kicks are capped at
  **6.5 m/s** by §8.4.2. rSim's ball mass is a typo, its ball drag a copied Bullet hack, its
  bounce its one right number. Workings in
  [`research/02-physical-constants.md`](research/02-physical-constants.md).
- [Drivetrain envelope](issues/01-drivetrain-envelope.md): `forcerange="-0.078 0.078"` N·m,
  `ctrlrange="-95 95"` rad/s → `a_max` 2.98 m/s², `v_max` 3.30 m/s, `ω_max` 35.9 rad/s; clears the
  static tip threshold by 62 % and `ω_tip` by 25 %. **The robot is direct drive, not 18:60** — so
  rSim's numbers describe the previous robot generation and never transferred. Workings in
  [`research/01-drivetrain-envelope.md`](research/01-drivetrain-envelope.md).
- [Contact parameters](issues/05-contact-parameters.md): roller↔floor μ 1.0 → **0.25**
  (reopens `01`'s binding constraint: friction-limited at 2.45 m/s², not torque-limited at 2.98,
  still clears the tip threshold); ball COR target **0.50** on both floor and wall, `solref`
  dampratio deferred to new task `17`; ball inertia gets an explicit `diaginertia` for **β = 0.5**.
  Walls/bumpers `solref`, ball↔dribbler μ = 2.5, and robot↔robot/wall μ all stay unchanged,
  recorded as deliberate approximations. `impratio` stays at 1, revisit only if `14` finds slip
  problems.

## Not yet specified

- **Sim-to-real gaps neither simulator addresses** (`O6`, `R6`): no observation noise, no
  actuation or vision latency, no domain randomisation — and `scene.py:187-189` notes the
  `lru_cache`d `MjModel` blocks randomisation until the cache is dropped. In scope for
  "defensible parameters", but not phrasable as a decision until the deterministic envelope is
  settled.
- **Per-wheel slip fidelity.** Whether anything downstream needs physical per-wheel normal
  forces and friction budgets. Decides itself once `07` fixes the collision topology; only
  matters if a constraint-based turnover fix ever comes back on the table.
- **Robot↔robot and robot↔wall friction** (`friction="0.5"` on 209 hulls, 0.6 on the walls).
  No source for either figure. `05` recorded it as a deliberate approximation, unchanged, and
  established the source-class taxonomy (RULEBOOK > MEASURED > VENDOR > DELIBERATE
  APPROXIMATION) other rows now use. Revisit only if a source appears.
- **Transient overcurrent.** A constant `forcerange` cannot express an ESC that allows several
  times `I_nom` for tens of milliseconds; a real torque envelope would. Recorded by `01` as a
  deliberate approximation. Becomes specifiable if anyone needs launch acceleration to be right.
- **`kv` for the wheel servos.** `01` left `kv = 0.103696` untouched and flagged that it was never
  derived from the motor's 8.77 rpm/mN·m speed/torque gradient. Under the new `forcerange` the
  servo saturates above 0.75 rad/s of error instead of 28.9, so `kv` now governs the whole
  proportional region. `04` may absorb this or it may graduate into its own ticket.
- **Oblique and ball-off-robot restitution.** `02` found only unbacked config values, and the two
  published redirect figures differ by 3× (TIGERs 0.2, ER-Force ~0.60). Nothing to decide until
  someone measures or a source appears.
- **Dribbler ball-coverage compliance.** Rulebook §3.2.4 caps coverage at 20 %, which `02`
  converts to ≤10.9 mm of hull penetration. Whether this model complies is unmeasured, and it
  constrains the dribbler geometry rather than any parameter — so it is not yet a decision.
- **Cost of scale.** 100 DOF and 209 collision geoms per robot × up to 7 robots
  (`StaticDefenders-v0`). Whether the fidelity is affordable at match sizes; needs `07`'s and
  `13`'s numbers first.
- **The 1 mm spawn drop** (`X8`): deterministic 0.14 m/s impact every reset. Probably harmless,
  unquantified.

## Out of scope

The destination rules these beyond this effort. They return only if the destination is redrawn.

- **Adopting rSoccer's field, goal test, tasks, rewards or horizons.** 6×4 m field, 0.70 m
  goal, 2-D `ball.x > half_len` goal test, per-env rule terminations, 25/30/120 s horizons
  (`U1`, `U2`, `X3`, `X4`, `T1`). Division A stays.
- **Reproducing rSim's ball-handling model.** Hinge-weld possession (`L3`), boolean dribbler,
  velocity-assignment kick (`L4`), the chassis-suppression callback (`C5`). All are ODE
  workarounds; mjsoccer's mechanisms are the higher-fidelity side.
- **Teleport turnover** (`TO1`, §12.3 of the research doc): discontinuous, non-Markov,
  exploitable, invisible.
- **Matching rSim's wheel-matrix sign convention** (`W2`). Tier 1's matrix is self-consistent;
  the inversion only matters for porting, which is not a goal. It gets *documented*, not changed.
- **Return comparability with published rSoccer results.** Follows from destination (1a).
