# Drivetrain envelope: what can this robot's motors actually do?

Type: research
Status: resolved
Blocked by: —

## Question

What are the physically defensible bounds for a tier-1 wheel actuator, derived from a Maxon
EC45 flat 50 W through 18:60 gearing onto this model's measured wheel?

Today `models/robot_rollers.xml:827-830` declares `forcerange="-3 3"` N·m and
`ctrlrange="-200 200"` rad/s per wheel. At the **measured** `r_eff = 30.089 mm` that is 399 N of
ground force per wheel against a total friction budget of `μ·m·g = 1.0 · 2.69207 · 9.81 = 26.4 N`
(`W3`, `L1`). It is not an actuator; it is "infinite torque, friction-limited". rSim's
counterpart is `wheelMotorMaxTorque = 0.070` N·m with `wheelMotorMaxRPM = 1557`
(`/tmp/rsim-upstream/src/robosim/sslconfig.h:159-160`), commented "Maxon EC45 flat 50w with
18:60 gear ratio, reduced to 3m/s2 accel" — the one rSim number sourced from hardware.

Deliver:
1. The EC45 flat 50 W figures from Maxon's own datasheet: nominal and stall torque, nominal and
   no-load speed, and the current limit that "50 W" implies.
2. Through 18:60 (3.333:1 reduction), the torque and speed **at the wheel**, and whether rSim's
   0.070 N·m corresponds to nominal, continuous or a deliberately derated figure — the comment
   says "reduced to 3m/s2 accel", so establish what it was reduced *from*.
3. With `r_eff = 30.089 mm` and `R = 0.0796 m` (both measured, §6 of the research doc), the
   resulting `a_max`, `v_max` and `ω_max` for a 2.69207 kg robot, and the `forcerange` /
   `ctrlrange` pair that expresses them.
4. Whether the result clears the two thresholds that matter: the derived forward static tip
   threshold `g · 0.04325 / 0.05399 = 7.86` m/s², and `ω_tip = 47.7` rad/s (`M1`). A defensible
   envelope should sit comfortably under both, the way rSim's 4.67 m/s² sits under its own 6.28.
5. Whether SSL rules or common Division A practice cap wheel speed or acceleration independently.

Primary sources only: Maxon's published datasheet for the EC45 flat 50 W, the SSL rulebook, and
the two clones. Cite each number to the document that owns it. Note explicitly anything that
cannot be resolved without the real robot's spec — the team has no drivetrain document, so a
stated assumption is an acceptable answer, an invented number is not.

## Answer

Resolved 2026-09-08. Full workings: [`research/01-drivetrain-envelope.md`](../research/01-drivetrain-envelope.md).

**Recommended envelope** — `forcerange="-0.078 0.078"` N·m and `ctrlrange="-95 95"` rad/s per
wheel, `kv` unchanged. Yields `a_max` 2.98 m/s² forward / 2.30 lateral, `v_max` 3.30 / 4.04 m/s,
`ω_max` 35.9 rad/s. Clears the forward static tip threshold (7.86 m/s²) by 62 % and `ω_tip`
(47.7 rad/s) by 25 %, and demands 31 % of the friction budget instead of 4750 % of it.

The two halves have different standing: **`forcerange` is a hardware fact** (`I_nom · k_M` for the
motor, direct drive), **`ctrlrange` is a design choice** — the motor's no-load speed is ~702 rad/s
at the wheel, so 95 rad/s is chosen to bound yaw at a 25 % margin below `ω_tip`, not because the
drive cannot spin faster. `04` owns that choice.

**The finding that matters: this robot is direct drive, not 18:60.** Three independent routes.
(a) Every wheel hub's mesh is `Direct_Transmission_new_Simplify_2`
(`models/robot_rollers.xml:300-301`); the only 18-tooth gear in the CAD is
`r_vis_gearMotorDribbler` on the chassis, driving the dribbler
(`models/robot_rollers.xml:164-165,250-257`). (b) RobôCIn's 2024 ETDP §1.2 *Direct Drive* records
the move off the 45 mm geared (3.3:1) wheel to a direct-drive 62 mm wheel, and states the design
operating point as "around 10 percent of the stall torque, near maximum efficiency" — this model's
measured `r_eff = 30.089` mm is a 60.18 mm wheel. (c) Arithmetic: with 18:60 the wheel would make
27.2 N of ground force against a 26.4 N friction budget, leaving the robot friction-limited at
9.81 m/s² and the torque cap behaviourally inert. Three routes also converge on the torque:
`I_nom · k_M = 78.1` mN·m, nominal torque 83.4 mN·m, and 10 % of the 780 mN·m stall = 78.0 mN·m.

So rSim's `0.070` N·m and `1557` rpm describe the **previous, geared robot generation**. They were
never transferable as numbers, which is why the ticket re-derived rather than copied.

**Corrections this forces on the audits.** `W3` of `docs/research/rsoccer-vs-mjsoccer.md` gives
rSim's ceiling as 4.67 m/s²; that assumes all four wheels project at unit force. The geometry caps
the projection at `Σ|sin α| = 3.14626` forward and `Σ|cos α| = 2.41421` lateral (verified
independently: both reproduce, and the sign pattern that maximises force gives `Σf_i = 0`, so it
costs no yaw). rSim's real ceiling is **3.68 m/s² forward / 2.82 lateral**, which vindicates its
own "reduced to 3m/s2 accel" comment on the weak axis. `W3`'s conclusion (rSim torque-limited,
mjsoccer friction-limited) is unaffected.

**Unresolved, carried to `16`.** Which of the five EC 45 flat 50 W windings this robot carries is
not recorded anywhere in the repo. 251601 is assumed. Across the catalogue the design torque spans
40.2–91.8 mN·m, moving `a_max` between 1.54 and 3.51 m/s² — a factor of 2.3, and the single
largest unresolved input to `04`. Also unresolved and recorded as deliberate approximations: the
ESC's transient overcurrent allowance (a constant `forcerange` cannot express it), and whether
`kv` should be re-derived from the motor's 8.77 rpm/mN·m speed/torque gradient.

**Rulebook:** caps none of speed, acceleration, wheel speed or motor power. Chapter 3 *Robots* →
*Hardware And Software Constraints* constrains shape, colour, dribbler, vision pattern, radio and
autonomy only.

## Amendment (2026-09-09, from `02`)

`02` establishes the only measured omniwheel-roller-on-carpet coefficient as **0.25**, not the
1.0 this ticket assumed (the model declares roller `friction="1.0"`, floor 0.8, and MuJoCo takes
the elementwise max — so 1.0 is what is in force today, but it is not sourced). At μ = 0.25 the
budget is 6.602 N against the 8.16 N this ticket recommends demanding.

`forcerange="-0.078 0.078"` is unaffected — it is derived from the motor. What changes is the
*binding constraint*: the robot would be friction-limited at `μ·g = 2.45` m/s², not torque-limited
at 2.98. The conclusion "clears the tip threshold" survives with more margin, not less. `04` and
`05` must now be decided together; `05` has been added to `04`'s blockers.
