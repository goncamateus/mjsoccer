Type: study
Source: [`17-ball-restitution-calibration.md`](17-ball-restitution-calibration.md)

# Study guide: find the `solref` that gives the ball its measured COR

This is the hands-on-est ticket of the three: there's no `## Answer` to explain yet, because the
answer is a number nobody has measured in *this* model. The physical target is settled — a real
SSL ball rebounds to about half its impact speed (COR ≈ 0.50) — but MuJoCo doesn't have a literal
"restitution" setting, so getting there means running the same experiment physicists use on a real
ball, inside the simulator, and adjusting `solref` until the simulated ball matches. That's a
genuinely open-ended calibration task, and this guide is about the checkpoint that proves you've
actually closed it: one number in, the right rebound out.

## Before you touch anything

- **The physical drop-test method you're reproducing in sim.** `e = sqrt(h_rebound / h_drop)` —
  because impact and rebound speed are each proportional to the square root of their respective
  heights, and COR is the ratio of outgoing to incoming speed. This is the exact method ER-Force
  used to get the 0.56 figure this ticket's target band comes from (see
  `docs/research/02-physical-constants.md` §2.2), and it's standard enough to have its own intro
  physics lab writeup. [Laboratory 4: Coefficient of Restitution (U. Alabama, PDF)](https://pleclair.ua.edu/ph105/Labs/old/coef_rest_lab.pdf)
- **Why there's no `solref` formula that just works.** `solref`'s two numbers are a reference time
  constant and a damping ratio for a constraint solver, not "restitution" — MuJoCo's own
  maintainers have confirmed there's no built-in coefficient-of-restitution parameter and no exact
  closed-form mapping to one. [`Coefficient of restitution` — mujoco#2081](https://github.com/google-deepmind/mujoco/discussions/2081)
- **A starting-point approximation, so you're not sweeping blind.** The same discussion above gives
  a damped-spring approximation: fix a stiffness `k` (their example uses `k = 2500`), then set the
  damping `b = -2·sqrt(k·m_eff)·ln(ce)/π` for a target restitution `ce`, where `m_eff` is the
  harmonic mean of the two colliding masses (for ball vs. an effectively-infinite floor/wall,
  `m_eff` → the ball's own mass). The author calls this "accurate enough" in their own tests, not
  exact — which is exactly why this ticket asks you to *verify* it with a drop test rather than
  trust the formula outright.
- **Reading a body's velocity in the Python API**, since your drop-test harness needs to measure
  impact and rebound speed, not just watch a viewer. `data.qvel` holds the full velocity vector;
  for a free-jointed body like the ball, the linear part is the first three entries of its slice.
  [Python bindings — MuJoCo Documentation](https://mujoco.readthedocs.io/en/stable/python.html)

## Real example

**[`Coefficient of restitution` — google-deepmind/mujoco Discussion #2081](https://github.com/google-deepmind/mujoco/discussions/2081)** —
someone hit exactly this problem (wanting a specific COR out of a MuJoCo contact) and got a
community answer with working numbers: `k = -solref[0]`, `b = -2·sqrt(k·m_eff)·ln(ce)/π`, with the
caveat *"you need to consider inertia matrices and jacobians for the colliding bodies if there are
joints on any of the bodies or if they are not point masses."* That caveat matters here: the ball
is a free body (no joints, effectively a point mass for this purpose), but the *floor* is a fixed
plane and the *wall* is a mesh collection with its own inertia considerations if it weren't static
— worth understanding why the formula's assumptions do or don't hold for each of your two target
surfaces before you trust the same `k`/`b` pair on both.

## Hands-on steps

1. Write a tiny standalone script (doesn't need to touch this repo's Gymnasium env machinery — a
   bare `mujoco.MjModel`/`mujoco.MjData` loop on `models/ball.xml` plus a floor plane is enough)
   that places the ball at a known height above `models/field.xml`'s floor (`z = 0.02` plus your
   drop height), zero velocity, and steps physics until it detects a local maximum in height after
   the first bounce.
2. Compute measured COR from your logged drop and rebound heights using the `sqrt(h/H)` formula
   above. Run it once against the *current* model (`solref="0.006 1"`, critically damped) first —
   you should measure COR ≈ 0, confirming your harness actually detects a bounce at all before you
   go looking for one that isn't there.
3. Open `models/ball.xml` and find the ball geom's `solref="0.006 1"`. Try the discussion's
   approximation formula for a target `ce = 0.50` to get a first candidate `(timeconst, dampratio)`
   pair — note MuJoCo's `solref` isn't literally `(k, b)`, it's `(timeconst, dampratio)`, so you'll
   need to convert, or use MuJoCo's documented negative-`solref` convention where the two numbers
   *are* direct stiffness/damping instead of timeconst/dampratio (check the XML reference for which
   sign convention your MuJoCo version expects).
4. Sweep around that candidate, re-running your harness each time, until measured COR lands in
   0.47–0.56 on the floor. Then point the same harness at a horizontal shot toward
   `models/field.xml`'s wall geoms (`friction="0.6 …"`, `solref="0.004 1"` today) and check whether
   your floor value also lands in-band there, or whether the wall needs its own pair — this is the
   ticket's open question about whether one value serves both surfaces.
5. Checkpoint: you're done with the measurement half of this ticket when you can report, for both
   surfaces, the `solref` pair used and the measured COR it produced, with your harness script
   still runnable (not one-off numbers you eyeballed and threw away) — that repeatability is what
   the acceptance gate in the ticket is asking for.

## If you get stuck

`docs/research/02-physical-constants.md` §2 has the full physical picture (why 0.47–0.56, and the
speed-dependence caveat — COR drops as impact speed rises, so note what drop height/speed you
calibrated at). If the MuJoCo side is the sticking point, search the `google-deepmind/mujoco`
GitHub *Discussions* tab for `solref` or `restitution` — several people have hit this same "MuJoCo
has no restitution knob" wall and the maintainers' answers there are more current than anything
written from memory.
