# Decision: what goes into the wheel actuators

Type: grilling
Status: open
Blocked by: 01, 03, 05, 16

## Question

Given `01`'s derived envelope and `03`'s measured baseline, what exactly is written to
`models/robot_rollers.xml:823-830`, and is anything clipped in Python as well?

Sub-decisions:
1. **`forcerange`.** The derived motor figure, or that figure with a stated safety margin? A
   real drive also has a current limit and a controller that is not a pure velocity servo.
2. **`ctrlrange`.** Today ±200 rad/s = 6.018 m/s of surface speed. Keep, or cut to the
   drivetrain's no-load speed? These interact: a `ctrlrange` the `forcerange` can never reach is
   a lie in the observation normalisers (`L9`).
3. **`kv`.** `kv = 0.103696` with a 200 rad/s error demands 20.7 N·m and delivers 3, so the
   servo is permanently saturated on large errors (`W7`). Once `forcerange` drops by ~40×, `kv`
   determines the whole tracking response. Re-tune, or leave it and accept the new time constant?
4. **Does the yaw *command* get clipped too?** Even a bounded per-wheel envelope admits a
   uniform command whose only effect is yaw. If `01`'s envelope still reaches beyond
   `MAX_W = 30` rad/s, decide between a lower `ctrlrange` and a clip in `_drive_robot`
   (`base.py:143-156`). `docs/index.md:79-82` promises the bounds come from the model, not from a
   Python constant — a clip breaks that promise and needs justifying.

Settled by: `01`'s numbers plus a re-run of `03`'s sweep against the candidate, showing the
reachable yaw rate falls below `MAX_W` and the peak tilt below 15°.
