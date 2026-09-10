Type: study
Source: [`03-com-spin-diagnostic.md`](03-com-spin-diagnostic.md)

# Study guide: Diagnostic — confirm the CoM-offset tipping mechanism

The robot's chassis has its center of mass sitting 3.459 mm off the spin axis (a manufacturing/CAD
reality, not a bug). Spin the robot fast enough on the spot and that tiny offset creates enough
sideways force to tip it over — like an unbalanced washing-machine drum shaking itself across the
floor when it spins up. This ticket isn't asking you to fix anything yet: it's asking you to prove
that story is actually what's happening, and measure how bad it is today, before anyone touches
the model.

## Before you touch anything

- **Subtree center of mass in MuJoCo (`subtree_com`)** — this is the number the ticket compares
  against `xpos` to get the 3.459 mm offset; it's computed by mass-weighting every body's position
  up the kinematic tree. [MuJoCo Computation docs](https://mujoco.readthedocs.io/en/stable/computation.html)
- **The `inertial`/`ipos` element** — the attribute the ticket tells you to temporarily zero out
  (`models/robot_rollers.xml:153`) to falsify the mechanism. If a body has no explicit `inertial`
  child, MuJoCo infers mass/inertia from its geoms instead — good to know before you go editing it.
  [MuJoCo XML Reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html)
- **Gyroscopic/centrifugal tipping from an off-axis mass while spinning** — the general physics
  behind why an off-center spinning mass produces a tipping torque, explained through combat
  robots with spinning weapons (same phenomenon, more intuitive scale).
  [Designing Around the Gyroscopic Effect — T.i. Combat Robotics](https://runamok.tech/insanity/gyroscopiceffect.html)

**No primary source found:** I could not find a published derivation matching the ticket's
`ω_tip = sqrt(g·d_edge/(h·d_com))` exactly — it looks like an ad-hoc tipping analysis (gravity's
restoring torque about the contact edge vs. the centrifugal torque from the off-center mass),
plausible but not a named textbook formula. Worth re-deriving it yourself from the gravity torque
around the contact edge vs. the centripetal force `m·ω²·d_com·h` — good mechanics exercise before
you run the experiment.

## Real example

**[Designing Around the Gyroscopic Effect](https://runamok.tech/insanity/gyroscopiceffect.html)** —
combat-robot builders hit exactly this class of problem: a spinning mass whose axis isn't
perfectly aligned with the frame creates a torque that fights the chassis. Their fix path is the
same shape as this ticket's diagnostic: identify the torque direction via the cross product of the
angular momentum and the imposed rotation, confirm it experimentally, then design the mass
distribution (or control response) around it rather than guessing.

> "The gyroscopic effect is what causes robots with vertically spinning weapons to tip up when
> turning... the vector cross product of the torque from gravity acting through the center of mass
> with the force establishes a gyroscopic torque acting tangentially."

## Hands-on steps

1. Open `models/robot_rollers.xml` near line 153 — find the `inertial` element (or the body
   without one) and read the current `ipos`. Don't change anything yet, just confirm it matches
   the 3.459 mm cited in the ticket.
2. Write a small script (Python + `mujoco` + your `env`) that `place`s one tier-1 robot at the
   origin and commands `w = [k,k,k,k]` for each `k` in `[0.2, 0.4, 0.6, 0.8, 1.0]` of `ctrlrange`,
   5 s of physics each time, no other body nearby. Record peak yaw rate and peak tilt per `k` —
   check whether it's close to 42.1 rad/s → 6.0°, ..., 59.4 → 46.6°.
3. Temporarily zero the x/y components of `ipos` (local edit, don't commit it) and re-run the same
   sweep. If the tilt nearly disappears, the `M1` hypothesis is confirmed — revert the XML
   immediately (`git diff` must come back clean).
4. Run `env.action_space.sample()` on tier 1, seeds 0–49, and count tips per episode-minute, plus
   the p50/p95/p99 of chassis tilt — this is the baseline future tickets will be measured against.
5. Measure your environment's steps/s during this process, to compare against the number already
   documented in `docs/index.md:157-160`.
6. Fill in the `## Answer` section of the original ticket with your numbers — that part is your
   deliverable, not this guide's.

## If you get stuck

- If the sweep doesn't match the expected degrees: check whether your actuator's `ctrlrange`
  matches what the ticket assumes, and whether "tilt" is being measured the same way (angle of the
  chassis's z-axis from vertical, not yaw).
- To understand `subtree_com` vs `xipos` vs `xpos` better, the frames section of the MuJoCo
  Computation doc linked above covers the different frame conventions MuJoCo uses.
- For comparable implementations, search "MuJoCo center of mass offset tipping" in the
  `google-deepmind/mujoco` GitHub Discussions — that's where this kind of modeling question tends
  to show up.
