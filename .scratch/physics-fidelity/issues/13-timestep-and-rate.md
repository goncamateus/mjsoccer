# Decision: timestep, control rate and the 60 Hz that is not 60 Hz

Type: grilling
Status: open
Blocked by: 07

## Question

Physics `dt = 0.002` s, `decimation = round(1/(60 · 0.002)) = 8`, so the control rate is
**62.5 Hz**, not the requested 60 (`base.py:68-69`, and `docs/index.md:302-305` says so). A "20 s"
episode is 1250 steps, not 1200 (`base.py:71`).

Decide:
1. **Is 2 ms the right physics step?** For a 100-DOF model with switching roller contacts and a
   16 ms kick pulse, plausibly yes — but nothing has established it. The test is a resolution
   sweep: results should move less than 5% between `dt` = 1 ms and 4 ms. `07`'s geom decision
   changes the cost of that answer, which is why this waits.
2. **Should `control_hz` be exact?** Either accept 62.5 and document it (already done), or make
   `control_hz` a request that snaps to an achievable value and *reports* it, or pick a `dt` that
   divides the target exactly (`dt = 1/480` gives 60 Hz at `decimation = 8`).
3. **Does the horizon stay in seconds?** Horizons are declared in seconds and converted
   (`__init__.py:20-27`), so a `dt` change silently changes every episode's step count and every
   per-step reward's per-second rate. Decide whether that conversion is the contract or a hazard.
4. **What does real SSL control at?** Vision is 60 Hz-ish; firmware control loops are much
   faster. If the env's rate is meant to represent the *policy's* rate, 60-62.5 Hz is right and
   the substeps represent firmware. Say which, because it determines whether `decimation` is a
   fidelity knob or a performance knob.
