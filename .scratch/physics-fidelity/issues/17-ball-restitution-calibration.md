# Task: find the `solref` that gives the ball its measured COR

Type: task
Status: open
Blocked by: —

## Question

`05` set the target: ball↔floor and ball↔wall coefficient of restitution ≈ 0.50 (measured band
0.47–0.56, ER-Force/TIGERs). MuJoCo has no literal restitution coefficient — bounce is emergent
from `solref`'s damping ratio — so no closed-form formula hands you the value; it has to be found
by measurement, the same shape as `03`'s diagnostic.

Today both surfaces are critically damped (`solref="0.006 1"` on `models/scene_rollers.xml:23`,
`solref="0.004 1"` on `models/field.xml:44`), i.e. COR ≈ 0.

Deliver:
1. A drop-test harness: release the ball from a known height above the floor (and separately,
   toward the wall at a known normal speed), measure the rebound speed, compute
   `COR = v_rebound / v_impact`.
2. Sweep `solref`'s damping ratio (holding the reference time constant, or sweeping both, whichever
   converges faster) until the measured COR lands in 0.47–0.56, target 0.50.
3. Report the `solref` pair for each surface (they need not end up identical, since floor and wall
   are separate geoms), and whether one value serves both to the required tolerance or they must
   diverge.
4. Note any speed-dependence observed (the source measurements show COR falling with impact speed
   — `docs/research/02-physical-constants.md` §2.1) and whether that matters at the impact speeds
   this simulator actually produces (kick exit speed ≤ 6.5 m/s, per `02`).

This is a `task` ticket: touches `models/` only as a reverted diagnostic. Revert after measuring;
the actual edit lands when the map closes, per `map.md`'s Mode note.

## Answer

<!-- fill in when resolved -->
