# Decision: what "the same envelope" means for tier 0

Type: grilling
Status: open
Blocked by: 04

## Question

Constraint 4 in the map: tier 0 keeps its ideal-kinematics *mechanism* but gets tier 1's
speed/acceleration *envelope*, so one action head means one thing on both.

Tier 0 today: three world-axis joints (`slide-x`, `slide-y`, `hinge-z`) with velocity servos at
`ctrlrange` ±4 m/s and ±30 rad/s, `forcerange` ±19.1261 N (x, y) and ±3.427 N·m (yaw)
(`models/robot_kinematic.xml:152-154,389-391`). The chassis never touches the floor, so there is
no traction limit, no slip and no tipping (`W5`). 19.1261 N on 2.69207 kg is 7.1 m/s².

Decide:
1. **Which quantities are matched?** `v_max`, `ω_max` and `a_max` are the obvious three. Slip,
   traction and weight transfer cannot be matched — that is the point of the tier. Say so
   explicitly, because "the same envelope" needs to be a defined term, not a vibe.
2. **`forcerange` on the slides.** Set it to `m · a_max` from `04`? That makes tier 0's
   acceleration ceiling equal to tier 1's *friction-and-torque-limited* one, which is the closest
   available correspondence, but tier 0 will reach it in every direction at once where tier 1
   cannot.
3. **The yaw servo.** ±3.427 N·m against the chassis's `Izz` gives tier 0 an angular
   acceleration with no wheel-torque analogue. Match to `04`'s four-wheel yaw authority.
4. **Do the round numbers stay?** ±4 m/s and ±30 rad/s look chosen for tidiness. If `04` derives
   something else, tier 0 should follow the derivation, not the tidiness.
5. **Where is the mismatch documented?** `README.md` claims one action head transfers across
   tiers; `docs/index.md:145-150,199-212` explains the tiers' differences. After this ticket both
   need the envelope statement, which is `15`'s job.
