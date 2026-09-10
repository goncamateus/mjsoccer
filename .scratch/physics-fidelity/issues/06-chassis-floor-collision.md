# Decision: should the chassis collide with the floor at all?

Type: grilling
Status: open
Blocked by: —

## Question

Today it cannot. The chassis geoms are `contype="1" conaffinity="1"` and the floor is
`contype="4" conaffinity="4"` (`models/field.xml:42`), and MuJoCo requires
`(contype1 & conaffinity2) | (contype2 & conaffinity1) ≠ 0` — which is zero. rSim has the same
gap for a different reason: `sslworld.cpp:171-197` never creates a chassis/ground surface (`B5`).

**Only the wheels touch the floor, on both sides.** So a tipped robot has nothing to land on: it
hangs on a wheel rim or its chassis passes through the carpet.

Under a hardware-fidelity destination this looks like a defect, not a choice — a real chassis
resting on its side is in contact with the carpet. But:

1. Is a tipped robot's post-fall behaviour something this simulator needs to represent at all?
   The episode terminates at 60° today (`TO1`), so the state is only ever a fraction of a second
   long. If termination stays, the fall is never simulated and the gap is free.
2. What does enabling it cost? 209 chassis hulls × a plane test every step, per robot, up to 7
   robots. `03`'s wall-clock baseline bounds the answer, and `07` may prune the hulls anyway.
3. Is there a cheap middle: one convex hull or a single primitive as a *floor-only* collision
   proxy for the chassis, on its own contype bit, leaving the 209 hulls for robot↔robot?

This ticket is unblocked because it is a scoping question, not a tuning one — and its answer
shapes `07`'s geom-mask design.
