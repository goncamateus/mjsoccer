# Decision: kick and chip calibration targets

Type: grilling
Status: open
Blocked by: 02, 05

## Question

What exit-speed distribution should the kicker produce, and do the mechanism's parameters need
re-tuning to hit it?

Today the kick is a genuine mechanism: `KICK_FORCE = 200` N applied for exactly one control step
(16 ms = 8 substeps) to a 0.1413 kg plunger on a `range="0 0.03"` slide against
`stiffness="2000"` / `damping="2.0"`, with a 250 ms per-plunger software lockout
(`base.py:84-90,160-163`, `models/robot_rollers.xml:270-296`). Exit speed is *emergent* —
`base.py:35` documents 6-7 m/s for a flat kick. The chipper is the same mechanism again.

Open points:
1. **Is 6-7 m/s legal and realistic?** `02` supplies the rulebook cap. If the rulebook says
   6.5 m/s, a mechanism that emergently exceeds it is a defect.
2. **Does the exit speed survive `05`?** Ball mass, ball↔plate friction and the plate's contact
   compliance all feed the impulse. Re-measure after `05`, do not calibrate before.
3. **Is a fixed 200 N the right interface?** The action is a boolean trigger today
   (`a > 0`, then a fixed force). Real SSL firmware commands a kick *speed*. Decide whether the
   action should scale the force, and if so how a policy's request maps to a capacitor charge.
4. **Is the 250 ms lockout physical?** It is load-bearing (`base.py:84-87`: without it a held
   kick action pins the plunger against its stop and the ball is never released), but the number
   should come from a real capacitor recharge time, not from what made the plunger behave.
5. **Recoil.** The chassis takes the reaction (`L4`), which rSim has no analogue for. Confirm it
   is the right magnitude rather than an artefact of the plunger's 0.1413 kg — the plungers are
   0.2785 kg of the robot's 2.692 kg and slide forward of centre (`M3`).

Measured against scenario 5 of the research doc's §13.2 (kick recoil at 0/2/4 m/s of chassis
speed) and its exit-velocity metric.
