# GoToBallShoot-v0

![GoToBallShoot-v0](../assets/GoToBallShoot.gif)

| | |
|---|---|
| Action Space | `Box(-1.0, 1.0, (7,), float32)` on the default tier 1, `Box(-1.0, 1.0, (6,), float32)` on tier 0 |
| Observation Space | `Box(-inf, inf, (14,), float32)` |
| import | `gym.make("GoToBallShoot-v0")` |

## Description

One blue robot starts on its own half with a loose ball somewhere on the field, and has to
put the ball in the goal at `x = +6.0`. There is nothing in the way: no keeper, no
defenders, no opponent.

The task exists to close the loop between locomotion and the ball-handling mechanisms.
`GoToBall-v0` ends the moment the robot is in position; here that is only the first half of
the episode, and the reward keeps paying only for what happens to the *ball* afterwards.
The robot can dribble the ball down the field, shove it with the chassis, kick it flat or
chip it — every one of those is in the action space and none of them is privileged by the
reward, which cares only that the ball ends up past the goal line, inside the mouth and
under the crossbar.

`StaticDefenders-v0` is this task with six parked yellow robots between the ball and the
goal; the reward and termination there are identical.

## Action Space

The action is a `Box(-1, 1, (7,), float32)` on the default tier 1. Every element is
normalised and scaled internally by the actuator's own `ctrlrange`, so the physical bounds
below come from the model, not from a constant in the environment.

| Num | Action | Min | Max | Unit |
|---|---|---|---|---|
| 0 | wheel 1 speed (`wheel1_vel`) | -200 | 200 | rad/s |
| 1 | wheel 2 speed (`wheel2_vel`) | -200 | 200 | rad/s |
| 2 | wheel 3 speed (`wheel3_vel`) | -200 | 200 | rad/s |
| 3 | wheel 4 speed (`wheel4_vel`) | -200 | 200 | rad/s |
| 4 | kicker plunger force (`kicker_force`) | 0 | 200 | N, fired as a pulse when the action > 0 |
| 5 | chipper plunger force (`chipper_force`) | 0 | 200 | N, fired as a pulse when the action > 0 |
| 6 | dribbler bar speed (`dribbler_vel`) | 0 | 314 | rad/s |

On tier 0 the action is a `Box(-1, 1, (6,), float32)`: the four wheel servos are replaced by
the three chassis rows, and kicker, chipper and dribbler shift down to indices 3, 4 and 5.

| Num | Action | Min | Max | Unit |
|---|---|---|---|---|
| 0 | body-frame forward velocity (`vel_x`) | -4 | 4 | m/s |
| 1 | body-frame lateral velocity (`vel_y`) | -4 | 4 | m/s |
| 2 | yaw rate (`vel_yaw`) | -30 | 30 | rad/s |
| 3 | kicker plunger force (`kicker_force`) | 0 | 200 | N, fired as a pulse when the action > 0 |
| 4 | chipper plunger force (`chipper_force`) | 0 | 200 | N, fired as a pulse when the action > 0 |
| 5 | dribbler bar speed (`dribbler_vel`) | 0 | 314 | rad/s |

Rows 0-3 on tier 1 are wheel speeds written straight to the servos, and the roller contacts
decide what the chassis does with them. Rows 0-2 on tier 0 are instead a *body-frame*
velocity that is rotated into the world-axis chassis
slides before being written. The kicker and chipper are triggers, not proportional: any
action above zero writes the full 200 N for one control step and then locks that plunger out
for 250 ms — the two locks are independent, so a kick cannot block a chip. A flat kick
leaves the mouth at 6-7 m/s. The dribbler maps `[-1, 1]` onto `[0, 314] rad/s` and is never
driven backwards. See [the action space contract](../index.md#action-space) for the
reasoning behind each.

## Observation Space

`Box(-inf, inf, (14,), float32)`: 6 ball elements plus 8 for the single robot. Note that
the goal is not in the observation — it is at a fixed place on a fixed field, so its
position is a constant a policy can absorb into its weights.

| Num | Observation | Min | Max | Name in XML / source | Unit |
|---|---|---|---|---|---|
| 0 | ball x position | -Inf | Inf | `ball` | m, / 6.0 (`FIELD_HALF_X`) |
| 1 | ball y position | -Inf | Inf | `ball` | m, / 4.5 (`FIELD_HALF_Y`) |
| 2 | ball height | -Inf | Inf | `ball` | m, / 0.155 (`GOAL_HEIGHT`) |
| 3 | ball x velocity | -Inf | Inf | `ball` | m/s, / 8.0 (`MAX_BALL_V`) |
| 4 | ball y velocity | -Inf | Inf | `ball` | m/s, / 8.0 (`MAX_BALL_V`) |
| 5 | ball z velocity | -Inf | Inf | `ball` | m/s, / 8.0 (`MAX_BALL_V`) |
| 6 | x position of the controlled blue robot | -Inf | Inf | `b0_robot` | m, / 6.0 (`FIELD_HALF_X`) |
| 7 | y position of the controlled blue robot | -Inf | Inf | `b0_robot` | m, / 4.5 (`FIELD_HALF_Y`) |
| 8 | sin of the yaw of the controlled blue robot | -1 | 1 | `b0_robot` | unitless |
| 9 | cos of the yaw of the controlled blue robot | -1 | 1 | `b0_robot` | unitless |
| 10 | world-frame x velocity of the controlled blue robot | -Inf | Inf | `b0_robot` | m/s, / 4.0 (`MAX_V`) |
| 11 | world-frame y velocity of the controlled blue robot | -Inf | Inf | `b0_robot` | m/s, / 4.0 (`MAX_V`) |
| 12 | yaw rate of the controlled blue robot | -Inf | Inf | `b0_robot` | rad/s, / 30.0 (`MAX_W`) |
| 13 | whether the controlled blue robot has the ball | 0 | 1 | `BaseEnv.holds` | 0 or 1 |

Positions and velocities are divided by the constants shown in the Unit column; yaw is
given as its sine and cosine so the wrap at +/-pi is not a discontinuity in the input. Ball
height is normalised by `GOAL_HEIGHT = 0.155 m`, which makes a chipped ball read greater
than 1 exactly when it is too high to score.

## Rewards

Per step:

| term | value |
|---|---|
| approach shaping | `0.2 * (prev_dist - dist)`, the decrease in robot-to-ball distance |
| goal shaping | `prev_goal_dist - goal_dist`, the decrease in ball-to-goal distance |
| energy | `-0.001 * ||a||^2` |
| goal bonus | `+20.0` on the terminating step |

`goal_dist` is the 2-D distance from the ball to the nearest point of the goal mouth,
`(6.0, clip(ball_y, -0.9, 0.9))`, so a ball driven straight down a wing is credited for the
part of its travel that actually shortens the shot.

The ball term is weighted five times the approach term deliberately: moving the ball
goalward is the task, closing on the ball is only how you get to do it. Both are
potential-based — each is the *decrease* in a distance over one step — so total shaping over
an episode is bounded by the starting distances and no cycle pays. A policy that pushes the
ball away and re-collects it nets zero for the round trip.

`ENERGY = 0.001` multiplies the squared norm of the whole flat action, summing over all 7
dimensions on the default tier 1 (6 on tier 0) rather than just the drive channels, so a
saturated action costs `-0.007` per step on tier 1, `-0.006` on tier 0. It discourages
buzzing, including firing the
kicker at nothing, but it is small enough that a genuine shot is always worth taking.

## Starting State

`_place` puts the robot on the defending half so that shooting means covering ground: `x`
uniform in `[-5.7, 0.0]`, `y` uniform in `[-4.2, 4.2]`, yaw uniform in `[-pi, pi)`. The ball
is then rejection-sampled with `sample_pose` anywhere in the field inside a 0.3 m margin,
resampled until it is at least 0.5 m from the robot, up to 200 attempts. The ball starts at
rest on the floor.

`_reset_task` records both baselines: the initial robot-to-ball distance and the initial
ball-to-goal distance.

## Episode End

### Termination

- **Goal.** `scene.in_goal(ball)`: the ball centre is past `x = 6.0`, inside the mouth
  (`|y| < 0.9`) and under the crossbar (`z < 0.155`). `info["goal"]` is `True`.
- **Ball out of the field.** `scene.in_field` fails for the ball centre (`|x| > 6.0 m` or
  `|y| > 4.5 m`) *and* `info["goal"]` is false, so a scored ball is never also reported as
  out. `info["ball_out"]` is set. A shot that crosses the goal line wide, or over the
  crossbar, ends the episode as a miss.
- **Controlled robot turned over.** The robot's own +z tilts more than 60 deg off world
  +z (`cos < 0.5`). The step pays `-10.0` on top of whatever the task scored and
  `info["tipped"]` is set. Only the robots the policy controls count — a scripted robot on
  its side is the agent's to exploit. Tier 0 never trips this: its chassis is a planar
  joint and cannot leave the ground.
- **Diverged simulation.** If `qpos` or `qvel` stops being finite, `step` raises
  `RuntimeError` rather than returning a NaN observation to train on. This is an error, not
  a normal termination.

### Truncation

`steps >= max_steps`, where `max_steps = round(horizon_s / dt)`. The default horizon is
**30 s**, and at the default `control_hz=60` the actual `dt` is 0.016 s, so **1875 steps**.

## Arguments

| parameter | type | default | description |
|---|---|---|---|
| `tier` | `int` | `1` | `1` = roller wheels with real slip, the default; `0` = the faster kinematic chassis |
| `control_hz` | `int` | `60` | requested control rate; rounded to a whole number of 2 ms physics steps, so 60 becomes 62.5 Hz |
| `horizon_s` | `float` | `30.0` | truncation horizon, in seconds |
| `render_mode` | `str \| None` | `None` | `"human"` or `"rgb_array"` |

```python
import gymnasium as gym
import mjsoccer

env = gym.make("GoToBallShoot-v0", tier=0, control_hz=60, horizon_s=30.0, render_mode="rgb_array")
```

## Info dictionary

| key | type | description |
|---|---|---|
| `is_success` | `bool` | the ball ended up in the goal |
| `goal` | `bool` | the same flag; also read by `base.step` to keep a scored ball from being reported as out of the field |
| `ball_out` | `bool` | present only on the step the ball leaves the field without scoring |
| `tipped` | `bool` | present only on the step a controlled robot turned over |

## Version History

- v0: initial release.
