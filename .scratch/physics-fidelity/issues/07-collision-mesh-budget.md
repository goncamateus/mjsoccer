# Decision: the collision-mesh budget

Type: grilling
Status: open
Blocked by: 06

## Question

Tier 1 carries **209 collision geoms** and 84 extra bodies per robot (measured): a CAD convex-hull
decomposition of the chassis, four wheel hubs and 80 free-spinning rollers. rSim carries one
cylinder, four wheel cylinders and a welded box, and tests exactly seven collision pairs
(`pworld.cpp:287-305`).

Under hardware fidelity mjsoccer's version is the *more* correct one, and it is not obviously
affordable. Three questions:

1. **Keep the 209-hull chassis?** It is why robots catch on each other's bumpers and motor
   housings and generate multi-height contact torques a cylinder cannot (`B1`). It is also the
   main supplier of the impulse that converts a marginal coning spin into a fall. Fidelity says
   keep; the trip impulses are real, not artefacts. Decide whether any hull is *CAD noise* rather
   than a real exterior surface — that is the only defensible pruning criterion.
2. **Keep ball↔roller contacts?** The ball is `contype="7" conaffinity="7"` and rollers are
   `contype="4" conaffinity="4"`, so bit 4 matches and a ball rolling under a robot gets kicked
   out by a roller (`B6`). rSim never collides ball with wheel. Physically the ball *does* hit
   the wheels. Keep — unless the measured effect is dominated by mesh artefacts.
3. **Keep hub↔wall?** Wheel hubs are `contype="1" conaffinity="1"` and reach the wall hulls
   (`C7`). Same reasoning.

Also settle the counting discrepancy between the two audits for the record: 209 collision geoms
(research doc) vs "315 descendant geoms" (analysis doc) — collision-only against
collision+visual. State which number the docs should quote.

Settled by a wall-clock measurement against `03`'s baseline plus the contact-impulse
attribution from `14`'s harness: which contacts actually supply trip impulses in normal play.
