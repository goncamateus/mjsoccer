# Decision: make the observation contract honest

Type: grilling
Status: open
Blocked by: 04

## Question

Three defects, all of which need `04`'s final envelope before they can be fixed:

1. **`MAX_W = 30.0` is not the tier-1 maximum.** It is documented as "the tier-0 chassis servo
   `ctrlrange`" (`base.py:34`, `docs/index.md:133`), but tier 1 reaches 59.4 rad/s measured, so
   the `omega` channel routinely reads ≈2.0 under the robot's own drive. `docs/index.md:135-137`
   claims the values "sit in roughly [−1, 1]" and blames excursions on "a tier-1 chassis pushed
   by another robot" — false today (`L9`). After `04` this may fix itself; confirm, and decide
   whether `MAX_V`/`MAX_W` should be *derived from the model* rather than declared as constants.
2. **The ball-height channel has a datum error.** `base.py:220` divides the raw world
   `bp[2]` by `GOAL_HEIGHT = 0.155`, but the floor plane is at z = 0.02
   (`models/field.xml:42`), so the channel carries a systematic `+0.02/0.155 = +0.129` offset and
   measures height above the world origin, not above the carpet (`U4`). Fix the observation, or
   move the floor to z = 0? Moving the floor touches every spawn z in the model; fixing the
   observation is one subtraction. Decide which, and whether anything else reads a raw z.
3. **The `Box` is `(−inf, inf)` and nothing clips** (`base.py:79`). rSoccer clips every channel
   to ±1.2 (`ssl_gym_base.py:27,213-220`). Under fidelity, is an unbounded observation space
   correct-but-inconvenient, or should the declared `Box` at least be *true*?

Settled together because all three are the same question: does the observation contract describe
the simulator, or aspire to it.
