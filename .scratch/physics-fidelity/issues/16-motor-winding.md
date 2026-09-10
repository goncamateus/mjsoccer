# Which EC 45 flat 50 W winding does this robot carry?

Type: task
Status: open
Blocked by: —

## Question

Nothing to decide until a fact arrives, and the fact is not in this repo — it is in the team's
hardware documentation or on the motor itself. HITL: someone has to look.

maxon catalogues five parts as "EC 45 flat … 50 W" across three windings and three frame
diameters. `01` assumed **251601** (Ø42.8 mm, 24 V, `k_M = 33.5` mN·m/A, stall 780 mN·m). The
spread across the five is not academic:

| part | `k_M` (mN·m/A) | stall (mN·m) | design torque at 10 % stall | resulting `a_max` forward |
|---|---|---|---|---|
| 339286 / 400107 (24 V, low speed) | 47.5 | 402 | 40.2 | **1.54 m/s²** |
| 251601 (24 V) — *assumed* | 33.5 | 780 | 78.0 | **2.98 m/s²** |
| 339287 (36 V) | 101 | 484 | 48.4 | ~1.85 m/s² |
| 651607 (Ø43.5, 24 V) | 36 | 918 | 91.8 | **3.51 m/s²** |

A factor of 2.3 between the extremes, and it propagates straight into `forcerange`, `a_max`, the
tier-0 slide `forcerange` (`12`), and every acceptance gate that mentions acceleration (`14`).

What settles it, cheapest first:
1. The part number printed on the motor, or the team's BOM / purchase records.
2. The supply voltage the robot runs at — 36 V immediately selects 339287 and rules out three.
3. A RobôCIn ETDP or internal hardware document naming the part. The 2024 ETDP names the motor
   family and the direct-drive design point but not the winding.
4. Failing all of that: measure. Stall current at a known voltage gives `k_M` directly.

If it cannot be resolved, `04` proceeds on 251601 and records it as a stated assumption with the
1.54–3.51 m/s² band written into the docs — but one line from the team removes the largest
uncertainty in this whole map, so ask before assuming.
