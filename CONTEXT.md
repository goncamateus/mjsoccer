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
The hardware-derived physics checks the robot must pass (envelope, acceleration, kick speed, ball bounce, possession, arrival at a target pose).
_Avoid_: validation tests, benchmarks. "Validation" names only the test marker
(`-m validation`), suite and sweep script that check the acceptance criteria.

**Possession**:
A robot holds the ball while its dribbler touches it, kept through a short grace of no contact
(the bar sticks, slips and re-grips).
_Avoid_: control, has_ball (the observation entry that reports it)

**Seated ball**:
A ball rolled into a robot's dribbler and caught by it at reset, rather than placed touching the
bar.
_Avoid_: placed ball, attached ball

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

**Firing step**:
A control step on which a kicker or chipper fires; still one 25 ms control step, only simulated more finely.
_Avoid_: kick step, substep (a substep is a physics step inside it)

**Decision**:
One policy action, a target pose plus kick, chip and dribble, held for `repeat_action` control steps.
_Avoid_: step (alone), action repeat

**Horizon**:
How long a task lasts, in seconds of simulated time, before it is truncated.
_Avoid_: episode length, max steps (the horizon counted in control steps)

**Observation noise**:
Gaussian noise added to every observation the policy sees; the physics never sees it. Belongs to
the task, off by default.
_Avoid_: sensor noise, DR

**Action latency**:
Each decision's action applied a fixed number of control steps late. Belongs to the task, off by
default.
_Avoid_: delay (alone), DR

### Simulation

**Backend**:
The engine that steps the physics of a task: the **CPU backend** (MuJoCo's C library, the reference
for what the physics should do), **MJX-Warp** (MJX's GPU engine, driven from JAX) or the **mjlab
backend** (ADR 0003). MJX-JAX, MJX's pure-JAX engine, is not a backend here (ADR 0002, upheld by ADR 0003).
_Avoid_: simulator (when the engine is meant), wrapper, impl

**Oracle**:
The CPU backend, as the reference the GPU backends are held to. When they disagree, the CPU is right.
_Avoid_: ground truth, baseline

**World**:
One simulated instance in a GPU batch, in its own slot; every world steps together.
_Avoid_: env (when one instance of a batch is meant), instance, replica

**Per-world failure**:
A world whose state stops being finite (diverged) or that overflows its contact or constraint
buffer (overflow) ends alone, and the rest of the batch runs on; a reset that finds no free spot is
counted with them (reset_overlap).
_Avoid_: crash, batch abort

**mjlab backend**:
The six tasks on mjlab's `ManagerBasedRlEnv`, in torch, over MuJoCo Warp on the exact base model,
trained with RSL-RL (`mjsoccer.mjlab`). The third backend, next to MJX-Warp and not replacing it;
same ids, spaces and observation layout as the other two.
_Avoid_: mjlab (alone, when the backend rather than the library is meant), torch backend

**GPU port**:
Reimplementing the six tasks for a GPU backend (MJX-Warp, then the mjlab backend), so they run
alongside the CPU backend, not replacing it.
_Avoid_: transfer, migration

**Parity**:
Agreement between a GPU backend and the CPU backend, judged statistically by the acceptance
criteria, not trajectory by trajectory. It is about the physics; the task logic has its own
**task-logic parity**.
_Avoid_: equivalence, matching, fidelity, policy transfer

**Task-logic parity**:
Agreement between a GPU backend's implementation of a task's logic (JAX or torch) and the CPU's
(observation, reward, termination, possession, navigation controller): given the same physical state and action, both give the same
outputs, up to float32 rounding. Judged exactly, state by state, unlike **parity**.
_Avoid_: parity (alone), consistency

**Domain randomization (DR)**:
Giving each parallel world on a GPU backend its own physical model, a **variant** of the nominal
one, sampled once per world for a whole training run. The CPU backend always runs the nominal model.
_Avoid_: noise (observation noise and action latency belong to the task, not the model), variation

**Variant / Nominal model**:
A variant is one world slot's DR-sampled model; the nominal model is the unrandomized one every
backend runs without DR.
_Avoid_: model variant in ADR 0002's sense (there, a convex-hull model for MJX-JAX, since dropped)

**First state**:
The cached start state of a world slot, already placed (ball seated where the task needs it), that the world restarts from when it
ends: Brax's per-slot cache on MJX-Warp, the **pool of first states** on the mjlab backend. Re-drawn
during training, not at every reset.
_Avoid_: reset cache, initial-state buffer

**Pool of first states**:
The mjlab backend's first states: one per slot, built in one batched physics loop and rebuilt each
epoch. It stands in for seating the ball at every reset, which mjlab cannot do because its step
advances every world at once.
_Avoid_: reset cache, initial-state buffer
