Type: study
Source: [`01-drivetrain-envelope.md`](01-drivetrain-envelope.md)

# Study guide: Drivetrain envelope — what can this robot's motors actually do?

Right now every wheel on this robot has `forcerange="-3 3"` N·m — enough torque to push 399 N
through the tyre contact, 15x more than friction will ever let through. It isn't modeling a motor,
it's modeling "infinite torque, friction-limited." The ticket derived the real number from a Maxon
EC45 flat 50W datasheet and the robot's own geometry; this guide is about *how* to get from a
motor datasheet to a MuJoCo actuator tag yourself, so you can redo the derivation and edit the XML
with your own hands.

## Before you touch anything

- **MuJoCo's `velocity` actuator and its `kv`/`forcerange`/`ctrlrange` attributes** — `kv` sets a
  proportional gain (`force = kv · (ctrl − joint_velocity)`), and `forcerange` clamps whatever that
  gain computes. They're independent knobs: `ctrlrange` says what speed you can *ask for*,
  `forcerange` says how hard the servo is *allowed to push* to get there. [XML Reference — MuJoCo Documentation](https://mujoco.readthedocs.io/en/stable/XMLreference.html)
- **How `gainprm`/`biasprm` actually produce that force**, if you want the underlying math instead
  of just the shorthand attributes. [Modeling — MuJoCo Documentation](https://mujoco.readthedocs.io/en/latest/modeling.html)
- **The real motor**: this robot's own model assumes a Maxon EC 45 flat 50 W, part 251601 — Hall
  sensors, 24 V, 6700 rpm no-load, 84.3 mNm nominal (continuous) torque, 822 mNm stall. These are
  the numbers the ticket's derivation starts from; check them against your own read of the ticket's
  arithmetic. [Maxon EC 45 flat 251601 datasheet (PDF)](https://mat.transtechnik.fr/LINMOT/documentation/Moteurs/EC%20Motors/Maxon/Maxon%20-%20EC_45_flat_251601.pdf)
- **Friction budget**: the maximum horizontal force a contact can sustain before it slips is
  `μ · N` (Coulomb friction) — this is the ceiling the ticket compares wheel torque against to
  decide whether the robot is *torque-limited* (motor gives up first) or *friction-limited* (tyre
  slips first). [Friction — Wikipedia](https://en.wikipedia.org/wiki/Friction)
- **Tip-over as a moment balance**: a robot tips when the overturning moment (here, the CoM's
  off-axis offset under yaw) exceeds the restoring moment from its own weight about the contact
  edge — the same static-stability idea used across wheeled-robot literature, just simplified to
  2D here because the CoM offset is small. [Tip-over stability analysis of a three-wheeled mobile robot (IISc, PDF)](https://mecheng.iisc.ac.in/~asitava/mimt2011_tharak.pdf)
- **`kv` is not free to raise** — pushing a velocity actuator's gain up to fight a small
  `forcerange` can destabilize the solver (NaN/Inf) rather than just track slower. Know this before
  you touch `kv` in `04`. [`kv` above 1 causes instability — mujoco#184](https://github.com/google-deepmind/mujoco/issues/184)

## Real example

**[mujoco_menagerie: kuka_iiwa_14 README](https://github.com/google-deepmind/mujoco_menagerie/blob/main/kuka_iiwa_14/README.md)** —
Google DeepMind's own curated MJCF model collection. Their URDF→MJCF conversion notes read simply:
*"Added forcelimits to match the torque limits in the [spec sheet]"* — i.e. they went to the
manufacturer's real datasheet and set `forcerange` to match, exactly the move this ticket makes.
They don't show their arithmetic either (their PDF spec sheet has the numbers directly), which is
normal: the "derive the number from the datasheet" step is usually a one-off calculation you do
once and keep, not something the model file itself has to re-derive every time it's read.

## Hands-on steps

1. Open `docs/research/01-drivetrain-envelope.md` and find where it turns the datasheet's nominal
   current and torque constant into a *nominal continuous torque at the wheel* — that's the
   `forcerange` derivation. Redo the arithmetic yourself using the datasheet numbers above (they
   differ slightly from the ticket's — 84.3 mNm nominal / 822 mNm stall here vs. 83.4 / 780 in the
   ticket — figure out whether that's a different datasheet edition or a rounding choice, and which
   one you'd trust).
2. Open `models/robot_rollers.xml` and find the four `wheel*_vel` actuators (currently
   `forcerange="-3 3"`, `ctrlrange="-200 200"`, all four identical). Change just one wheel's
   `forcerange` to your derived value and leave the other three at `-3 3` — load the model (e.g.
   via `mujoco.viewer` or this repo's own scene loader) and watch what an asymmetric drivetrain
   does to a straight-line command. This is a deliberately wrong intermediate step, purely to see
   the actuator clamp in action before you commit to the real symmetric edit.
3. Open `src/mjsoccer/base.py` around `_drive_robot` (roughly lines 143–156) and trace how an
   action in `[-1, 1]` becomes `data.ctrl` — note the comment about `ctrlrange` being the only
   place the bound is allowed to live, per `docs/index.md:79-82`. If you later touch `ctrlrange`
   in `04`, this is the code path whose contract you'd be changing.
4. With all four wheels set to your derived `forcerange`/`ctrlrange`, script a uniform wheel
   command sweep (the ticket's own measurement method: `place` the robot at the origin, run a few
   seconds of physics with no other body in contact, log peak yaw rate and chassis tilt at a few
   wheel-speed commands). Check whether your tilt onset brackets the ticket's `ω_tip = 47.7 rad/s`
   the way the ticket's own measurement did (42.1 rad/s at 120 rad/s wheel command, 55.6 rad/s at
   160). If your numbers disagree, that's the interesting result — go find out why before assuming
   the ticket is right.

## If you get stuck

Re-read `docs/research/01-drivetrain-envelope.md` in full — it shows every intermediate number,
not just the final `forcerange`/`ctrlrange`. If the MuJoCo actuator model itself is the confusing
part, the `google-deepmind/mujoco` GitHub Discussions tab (not Issues) is where most "why does my
actuator behave like X" questions get answered by the MuJoCo team directly — search it for
`velocity actuator` or `forcerange` before opening a new thread.
