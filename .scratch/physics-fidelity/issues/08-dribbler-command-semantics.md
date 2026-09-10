# Decision: what a dribbler action of 0 should mean

Type: grilling
Status: open
Blocked by: —

## Question

`base.py:166` maps the dribbler action affinely:

```python
self.data.ctrl[r.dribble] = scene.DRIBBLER_OMEGA * (a[k + 2] + 1.0) * 0.5
```

So `a = 0` spins the bar at **157 rad/s**, and the bar is off only at exactly `a = −1`. The
comment justifies the non-negativity ("a bar driven backwards throws the ball out of the mouth"),
which is sound, but not the neutral point. Consequences: every `action_space.sample()` rollout
dribbles at a mean of 157 rad/s continuously, and a policy must actively push to −1 to stop
dribbling.

Three options:
1. **Keep the affine map.** Continuous control of bar speed, neutral = half speed. Defensible if
   real firmware takes a speed, not a switch.
2. **Half-range map**: `a ≤ 0` → off, `a ∈ (0, 1]` → `[0, 314]`. Neutral means off; still
   continuous. Costs half the action resolution.
3. **Boolean**, `a > 0`. What every rSoccer env does — but rSim's is also a latching bug
   (`sslworld.cpp:320` only ever calls `setDribbler(true)`, so once on it stays on until the
   world is rebuilt), so this is not an endorsement.

What settles it: what the real robot's dribbler interface accepts — a duty cycle/speed or an
on/off — and whether 314 rad/s is itself a sourced figure (`scene.py:43`) or a guess. Note this
changes the action contract, so it must land before `15` documents it, and it interacts with `05`
(bar speed and contact μ jointly set the retention budget).
