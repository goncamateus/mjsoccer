Type: study
Source: [`05-contact-parameters.md`](05-contact-parameters.md)

# Study guide: Contact parameters that are defensible rather than tuned-to-taste

Two of this ticket's numbers are ready to type in today: the roller's grip on the floor was too
high by 4x (nobody had measured it, so it defaulted to "high enough to never slip"), and the
ball's moment of inertia has been silently wrong since the model was built (MuJoCo assumes a
solid, uniform sphere unless you tell it otherwise, and a real ball isn't one). A third number —
how bouncy the ball is — got a *target*, not a value, because MuJoCo has no literal "coefficient
of restitution" setting; that one becomes its own hands-on project in ticket `17`. This guide
covers the two you can actually go edit right now, plus enough of MuJoCo's contact model that the
"why" behind each isn't a black box.

## Before you touch anything

- **The `friction` 5-tuple and `condim`.** A MuJoCo geom's friction isn't one number: it's
  `(sliding, sliding, torsional, rolling, rolling)`, and how many of those actually apply is set
  by `condim` on the contact (1 = frictionless, 3 = sliding only, 4 = + torsional, 6 = + rolling).
  The roller↔floor contact here uses `condim="4"` — sliding and torsional, no rolling resistance
  modeled at that contact — so changing the *first* number is what actually changes the grip
  budget. [XML Reference — MuJoCo Documentation](https://mujoco.readthedocs.io/en/stable/XMLreference.html)
- **How two geoms' contact parameters combine when there's no explicit `<pair>`.** This model
  relies on default mixing for robot↔ball and robot↔robot contacts (only the two dribbler pairs
  and ball↔floor are explicit `<pair>` elements in `models/scene_rollers.xml`) — MuJoCo takes the
  *elementwise maximum* of the two geoms' `friction` values and a `solmix`-weighted average of
  `solref`, not an average of everything. [Modeling — MuJoCo Documentation](https://mujoco.readthedocs.io/en/latest/modeling.html)
- **`solref`'s two numbers are a time-constant and a damping ratio, not a stiffness and a
  restitution.** `dampratio = 1` is critically damped (what this model uses almost everywhere,
  including the roller contact you're about to edit) — it settles into contact with no rebound.
  Restitution/bounce is what happens when you go *below* 1 (underdamped), which is exactly the
  knob `17`'s calibration will need, but isn't what you're touching in this ticket.
  [XML Reference — MuJoCo Documentation](https://mujoco.readthedocs.io/en/stable/XMLreference.html)
- **The moment-of-inertia factor, β = I / (m·r²).** For a uniform solid sphere β = 0.4 exactly —
  that's what MuJoCo computes automatically from `mass` + `size` on a sphere geom with no
  `inertial` element, which is silently what this ball has today. A real ball (layered, dimpled,
  not uniform density) has a different mass distribution and therefore a different β; declaring
  `diaginertia` explicitly is how you override the automatic solid-sphere assumption.
  [Moment of inertia factor — Wikipedia](https://en.wikipedia.org/wiki/Moment_of_inertia_factor)
  (the same β used for planets and moons — the physics is identical, only the body changes)
- **Setting `diaginertia` by hand.** MuJoCo will compile a body's inertia from its geoms
  automatically unless you give it an explicit `<inertial>` element with `mass` and `diaginertia`
  (or `fullinertia`) — once you do, your numbers win outright, no averaging with the geom's
  implicit value. [Auto Inertia Computation discussion — mujoco#318](https://github.com/google-deepmind/mujoco/discussions/318)

## Real example

**[dm_control: `soccer_ball.py`](https://github.com/google-deepmind/dm_control/blob/main/dm_control/locomotion/soccer/soccer_ball.py)** —
Google DeepMind's own multi-agent MuJoCo soccer environment. Its ball module hard-codes
`_REGULATION_RADIUS = 0.117`, `_REGULATION_MASS = 0.45`, `_DEFAULT_FRICTION = (0.7, 0.05, 0.04)`
and `_DEFAULT_DAMP_RATIO = 0.4`, each commented as **"FIFA regulation parameters for a size 5
ball"** — i.e. sourced from the sport's own rulebook, exactly the move `02`/`05` make for this SSL
ball. The `damp_ratio` is passed straight into `solref=[0.02, damp_ratio]` with the comment "lower
implies less dampening upon contacts" — the same dampratio-controls-bounciness relationship this
ticket's COR discussion (and `17`'s calibration) leans on, stated in DeepMind's own words instead
of this project's.

## Hands-on steps

1. Open `models/robot_rollers.xml` and find the roller collision geoms (search for
   `friction="1.0 0.005 0.0001"` — there are 80 of them, one per roller, e.g.
   `col_wheel1_roller_1_c00`). Change the first number in each from `1.0` to `0.25`. Yes, all 80 —
   this is a good moment to reach for a project-wide find/replace rather than hand-editing each,
   but look at a couple first so you know exactly what you're replacing.
2. Load the model afterward (this repo's own scene loader, or a bare `mujoco.viewer.launch`) and
   drive the robot in a straight line at a saturating wheel command. Compare the achieved
   acceleration against the ticket's two numbers: before this change it should clip near
   `μ·g = 9.81` m/s² (with the old μ=1.0 effective from geom mixing); after, near `2.45` m/s².
   If you don't see the drop, check whether the wall/robot contacts are still handled by geom
   mixing rather than the pair you edited — mixing takes the *max* of both geoms' friction, so an
   unrelated geom with a higher value can mask your edit.
3. Open `models/ball.xml` and look at the single ball `<geom>` — notice it has `mass="0.046"` but
   no `<inertial>` sibling element at all, which is why MuJoCo is filling in the solid-sphere
   default today. Add an `<inertial>` element with `pos="0 0 0"`, `mass="0.046"`, and
   `diaginertia` computed from `β = 0.5` (work the arithmetic yourself: `I = β·m·r²` with
   `r = 0.0215` m, same value on all three diagonal entries since a sphere has no preferred axis).
4. Verify the inertia actually changed: `mujoco.MjModel.from_xml_path` the ball model, run
   `mj_forward`, and read `model.body_inertia` for the ball body — it should now show your value,
   not `0.4·m·r²`. Then think about what a higher rotational inertia (relative to mass) *does* to
   the ball physically — it means more torque is needed to spin it up, and it retains backspin/
   sidespin longer once given some. That's the behavior this change is actually buying.

## If you get stuck

`docs/research/02-physical-constants.md` has the full sourcing chain for β = 0.5 and the roller μ
= 0.25 measurement (Williams et al., tilt-table study) if the "why 0.25 and not something else"
isn't clear from the ticket alone. For the MuJoCo mechanics specifically, the
`google-deepmind/mujoco` GitHub *Discussions* tab (not Issues) has the maintainers answering
contact/inertia questions directly — search it for `diaginertia` or `condim` before opening a new
thread.
