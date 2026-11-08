# GoToBallShoot-v0

![GoToBallShoot-v0](../assets/GoToBallShoot.gif)

| | |
|---|---|
| Action Space | `Box(-1.0, 1.0, (7,), float32)` on the default tier 1, `Box(-1.0, 1.0, (6,), float32)` on tier 0 |
| Observation Space | `Box((14,), float32)`, bounded per row below |
| import | `gym.make("GoToBallShoot-v0")` |

## Description

One blue robot and a loose ball start in the attacking half; the robot has to put the ball
in the goal at `x = +6.0`. Nothing is in the way. This is rSoccer's `SSLGoToBallShoot-v0`
with `random_init=True, enter_goal_area=False`: the reward pays the robot closing on the ball
and the ball closing on the goal, +5 for the goal, and the episode ends the moment the robot
or the ball leaves the attacking half, or the robot steps into the penalty area.

The robot can dribble, shove, kick or chip the ball; none of them is privileged by the reward.
`StaticDefenders-v0` and `ContestedPossession-v0` share this reward.

## Action Space

The action is a `Box(-1, 1, (7,), float32)` on the default tier 1. Every element is
normalised and scaled internally by the actuator's own `ctrlrange`, so the physical bounds
below come from the model, not from a constant in the environment.

| Num | Action | Min | Max | Unit |
|---|---|---|---|---|
| 0 | wheel 1 speed (`wheel1_vel`) | -95 | 95 | rad/s |
| 1 | wheel 2 speed (`wheel2_vel`) | -95 | 95 | rad/s |
| 2 | wheel 3 speed (`wheel3_vel`) | -95 | 95 | rad/s |
| 3 | wheel 4 speed (`wheel4_vel`) | -95 | 95 | rad/s |
| 4 | kicker plunger force (`kicker_force`) | 0 | 500 | N, a one-step pulse for `6.5 * a` m/s when a > 0 |
| 5 | chipper plunger force (`chipper_force`) | 0 | 500 | N, a one-step pulse for `6.5 * a` m/s when a > 0 |
| 6 | dribbler bar speed (`dribbler_vel`) | 0 | 314 | rad/s |

On tier 0 the action is a `Box(-1, 1, (6,), float32)`: the four wheel servos are replaced by
the three chassis rows, and kicker, chipper and dribbler shift down to indices 3, 4 and 5.

| Num | Action | Min | Max | Unit |
|---|---|---|---|---|
| 0 | body-frame forward velocity (`vel_x`) | -4 | 4 | m/s |
| 1 | body-frame lateral velocity (`vel_y`) | -4 | 4 | m/s |
| 2 | yaw rate (`vel_yaw`) | -30 | 30 | rad/s |
| 3 | kicker plunger force (`kicker_force`) | 0 | 500 | N, a one-step pulse for `6.5 * a` m/s when a > 0 |
| 4 | chipper plunger force (`chipper_force`) | 0 | 500 | N, a one-step pulse for `6.5 * a` m/s when a > 0 |
| 5 | dribbler bar speed (`dribbler_vel`) | 0 | 314 | rad/s |

Rows 0-3 on tier 1 are wheel speeds written straight to the servos, and the roller contacts
decide what the chassis does with them. Rows 0-2 on tier 0 are instead a *body-frame*
velocity that is rotated into the world-axis chassis
slides before being written. The kicker and chipper ask for a speed: `a <= 0` does not
fire, `(0, 1]` sends the ball out at `6.5 * a` m/s, and each shot locks that plunger out for
250 ms — the two locks are independent, so a kick cannot block a chip. The dribbler is off for `a <= 0`, maps `(0, 1]` onto `(0, 314] rad/s`,
and is never driven backwards. See [the action space contract](../index.md#action-space) for the
reasoning behind each.

## Observation Space

`Box((14,), float32)`, bounded per row below: 6 ball elements plus 8 for the single robot. Note that
the goal is not in the observation — it is at a fixed place on a fixed field, so its
position is a constant a policy can absorb into its weights.

| Num | Observation | Min | Max | Name in XML / source | Unit |
|---|---|---|---|---|---|
| 0 | ball x position | -Inf | Inf | `ball` | m, / 6.0 (`FIELD_HALF_X`) |
| 1 | ball y position | -Inf | Inf | `ball` | m, / 4.5 (`FIELD_HALF_Y`) |
| 2 | ball height above the carpet | 0 | Inf | `ball` | m, / 0.155 (`GOAL_HEIGHT`) |
| 3 | ball x velocity | -Inf | Inf | `ball` | m/s, / 8.0 (`MAX_BALL_V`) |
| 4 | ball y velocity | -Inf | Inf | `ball` | m/s, / 8.0 (`MAX_BALL_V`) |
| 5 | ball z velocity | -Inf | Inf | `ball` | m/s, / 8.0 (`MAX_BALL_V`) |
| 6 | x position of the controlled blue robot | -1.05 | 1.05 | `b0_robot` | m, / 6.0 (`FIELD_HALF_X`) |
| 7 | y position of the controlled blue robot | -1.067 | 1.067 | `b0_robot` | m, / 4.5 (`FIELD_HALF_Y`) |
| 8 | sin of the yaw of the controlled blue robot | -1 | 1 | `b0_robot` | unitless |
| 9 | cos of the yaw of the controlled blue robot | -1 | 1 | `b0_robot` | unitless |
| 10 | world-frame x velocity of the controlled blue robot | -Inf | Inf | `b0_robot` | m/s, / 3.3 (`MAX_V`) |
| 11 | world-frame y velocity of the controlled blue robot | -Inf | Inf | `b0_robot` | m/s, / 3.3 (`MAX_V`) |
| 12 | yaw rate of the controlled blue robot | -Inf | Inf | `b0_robot` | rad/s, / 37.1 (`MAX_W`) |
| 13 | whether the controlled blue robot has the ball | 0 | 1 | `BaseEnv.holds` | 0 or 1 |

Positions and velocities are divided by the constants shown in the Unit column; yaw is
given as its sine and cosine so the wrap at +/-pi is not a discontinuity in the input. Ball
height is measured from the carpet and normalised by `GOAL_HEIGHT = 0.155 m`, which makes a chipped ball read greater
than 1 exactly when it is too high to score.

## Rewards

Ported from rSoccer's `_calculate_reward_and_done` (`ssl_go_to_ball_shoot`, and the
identical one in `ssl_hw_challenge`). Per step, unless the step ends the episode:

| term | value |
|---|---|
| ball distance | `clip(prev_dist - dist, -1, 1) / BALL_DIST_SCALE`, robot-to-ball closing; `BALL_DIST_SCALE = hypot(9.0, 6.0) = 10.8` |
| ball gradient | `clip(prev - now, -1, 1) / BALL_GRAD_SCALE`, the ball closing on the goal centre `(6.0, 0)`; `BALL_GRAD_SCALE = hypot(4.5, 6.0) / 4 = 1.875` |
| energy | `-mean(|a_drive|) / 1200` |

On an ending step the only reward is the goal: `+5.0` for a ball past `x = 6.0` inside the
mouth, `0` for anything else. There are no penalties. The scales are rSoccer's formulas on our
field. rSoccer's energy is `sum |wheel speed| / (4 * 160 rad/s * 1200)`; here the commanded drive,
as a fraction of its range, stands in for the wheel speed (tier 0 commands `(vx, vy, w)`, not
wheels), so a saturated drive costs `1 / 1200` a step as there.

## Starting State

rSoccer's `random_init`. The ball is uniform over `x` in `[0.3, 3.9]` (0.3 m inside the half
line and off the penalty area, `6.0 - 1.8 - 0.3`) and `y` in `[-4.4, 4.4]`. The robot is drawn
from the same box with its `y` negated, and redrawn while it is within
`(ball radius + robot radius) * 1.1 = 0.122 m` of the ball on both axes; yaw uniform over
`[0, 2 pi)`. Everything starts at rest.

## Episode End

### Termination

Checked in rSoccer's order, the first that holds ends the episode:

- **Robot out.** The robot leaves the attacking half (`x < -0.2`) or the field's width
  (`|y| > 4.5`). `info["robot_out"]`.
- **Goalkeeper area.** The robot enters the +x penalty area (`x > 4.2`, `|y| < 1.8`, Div A's
  1.8 x 3.6 m). `info["gk_area"]`.
- **Ball out.** The ball leaves the attacking half (`x < 0`) or the width. `info["ball_out"]`.
- **Goal line.** The ball crosses `x = 6.0`: `+5.0` and `info["goal"]` inside the mouth (and
  under the crossbar), `0` wide.
- **Controlled robot turned over** (tier 1 only). The robot's own +z tilts more than 60 deg
  off world +z (`cos < 0.5`); the step pays `-100.0` and `info["tipped"]` is set. rSoccer's
  robots cannot tip, so this is the one ending that is ours, kept so a robot on its back never
  collects shaping.
- **Diverged simulation.** If `qpos` or `qvel` stops being finite, `step` raises
  `RuntimeError` rather than returning a NaN observation to train on.

### Truncation

`steps >= max_steps`, where `max_steps = round(horizon_s / dt)`. The default horizon is
**60 s**, rSoccer's: **2400 steps** at the 40 Hz (0.025 s) the agents act at, rSoccer's step,
and 3600 at the default `control_hz=60`.

## Arguments

| parameter | type | default | description |
|---|---|---|---|
| `tier` | `int` | `1` | `1` = roller wheels with real slip, the default; `0` = the faster kinematic chassis |
| `control_hz` | `int` | `60` | control rate, exact: it must divide the 960 Hz physics rate (30, 60, 120, ...), or the env raises `ValueError` |
| `horizon_s` | `float` | `60.0` | truncation horizon, in seconds |
| `render_mode` | `str \| None` | `None` | `"human"` or `"rgb_array"` |

```python
import gymnasium as gym
import mjsoccer

env = gym.make("GoToBallShoot-v0", tier=0, control_hz=60, horizon_s=60.0, render_mode="rgb_array")
```

## Info dictionary

| key | type | description |
|---|---|---|
| `is_success` | `bool` | the ball ended up in the goal |
| `goal` | `bool` | present on a step the ball crossed the goal line: inside the mouth or not |
| `robot_out` | `bool` | present when the robot left the attacking half or the width |
| `gk_area` | `bool` | present when the robot entered the penalty area |
| `ball_out` | `bool` | present when the ball left the attacking half or the width |
| `tipped` | `bool` | present only on the step a controlled robot turned over |
| `tilt_deg` | `float` | with `tipped`: the worst controlled robot's tilt off vertical, degrees |
| `tip_step` | `int` | with `tipped`: control steps since reset, so time-to-tip is `tip_step * dt` |

## Version History

- v0: initial release.
