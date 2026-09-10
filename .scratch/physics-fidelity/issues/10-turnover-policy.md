# Decision: turnover handling once the actuation is bounded

Type: grilling
Status: open
Blocked by: 04

## Question

`04` removes the *cause* of turnover (a 59 rad/s reachable yaw acting on a 3.459 mm CoM offset).
What should the *handler* then be?

Today: `TIPPED_COS = 0.5` (60°), `TIPOVER_PENALTY = 100.0`, terminate the episode, controlled
robots only, `info["tipped"] = True` (`base.py:41-42,129-132,170-182`). Tier 0 is structurally
immune.

Decide:
1. **Keep terminate-and-penalise?** The research doc's recommendation is yes — as a *safety net*
   with a bounded, logged rate, not as the primary mechanism. Confirm, or argue for
   log-and-continue now that a tip means something has genuinely gone wrong.
2. **Is 100.0 still right?** At the current rate one tip erases 5-10 successful episodes of
   return, which makes the gradient for everything else noisy. Once `04` drops the rate, a
   smaller penalty may be both sufficient and better-conditioned. Do **not** lower it to make
   training smoother before `04` lands — that teaches the policy tipping is affordable.
3. **Keep 60°?** Not rSim's 0.9/25.84° (that would fire on a marginal coning spin), but is 60°
   the recoverable/unrecoverable boundary, or is it arbitrary? `base.py:37-41` argues it; check
   the argument against `03`'s measured tilt distribution.
4. **What gets logged?** `info["tipped"]` exists; add the tilt angle and the step index so the
   rate and the time-to-tip distribution are measurable from the env rather than from a bespoke
   harness (§12.5 step 4).
5. **Does a righting moment ever get built?** Only if turnover persists after `04`, `05` and
   `07`. If it does: hard saturation, a deadband above the normal driving lean, a first-order
   time constant, and the applied impulse logged — an unsaturated righting torque is a free
   reaction moment against the world and a policy will harvest it. Saturating on more than 0.1%
   of steps means the real cause is still unfixed.

Explicitly out of scope: teleporting, welding roll/pitch, and swapping the free joint for a
planar chain (that is tier 0, and `W5` shows tier 0 is not equivalent).
