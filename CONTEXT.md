# mjsoccer

Gymnasium environments for a RoboCup SSL Division A omni robot in MuJoCo, built to train
reinforcement-learning policies on six soccer tasks.

## Language

### Physics

**Fidelity**:
Agreement between the robot's physics and the real robot, every number justified by the hardware or
first principles.
_Avoid_: realism, accuracy

**Acceptance criteria**:
The hardware-derived physics checks the robot must pass (envelope, acceleration, tilt, kick speed).
_Avoid_: validation tests, benchmarks

### Control

**Target pose**:
The field position and heading the policy commands each decision; the robot is driven toward it.
_Avoid_: setpoint, waypoint, goal (the goal is the soccer goal)

**Navigation controller**:
The fixed go-to-point law, ported from Planning-the-path-with-rl, that turns a target pose into a
body-frame velocity every control step.
_Avoid_: planner, path planner (the policy plans; the controller only follows)

**Control step**:
One 25 ms tick of the navigation controller, at a fixed 40 Hz.
_Avoid_: step (alone), frame, tick

**Decision**:
One policy action, a target pose plus kick, chip and dribble, held for `repeat_action` control steps.
_Avoid_: step (alone), action repeat

### Simulation

**Backend**:
The engine that steps the physics of a task: the **CPU backend** (MuJoCo's C library, the reference
for what the physics should do), **MJX-JAX** or **MJX-Warp**. The two MJX backends run one shared
task implementation.
_Avoid_: simulator (when the engine is meant), wrapper, impl

**Model variant**:
A robot-and-field model generated from the base MJCF by a declared transform, so that one backend
can run it (MJX-JAX's variant swaps unsupported shapes); the CPU backend can run it too.
_Avoid_: second model, fork, simplified model

**GPU port**:
Reimplementing the six tasks once on MJX, so they run on both MJX backends alongside the CPU backend,
not replacing it.
_Avoid_: transfer, migration

**Parity**:
Agreement between an MJX backend and the CPU backend, judged statistically by the acceptance
criteria, not trajectory by trajectory.
_Avoid_: equivalence, matching, fidelity, policy transfer
