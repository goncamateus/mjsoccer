# GoToBallShoot-v0

![GoToBallShoot-v0](../assets/GoToBallShoot.gif)

| | |
|---|---|
| Action Space | `Box(-1.0, 1.0, (7,), float32)` |
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

The action is a `Box(-1, 1, (7,), float32)` and drives the one blue robot.

| Num | Action | Min | Max | Unit |
|---|---|---|---|---|
| 0 | target x | -6.0 | 6.0 | m, `a * 6.0` (`FIELD_HALF_X`) |
| 1 | target y | -4.5 | 4.5 | m, `a * 4.5` (`FIELD_HALF_Y`) |
| 2 | sin of the target heading | -1 | 1 | heading `atan2(a_2, a_3)` |
| 3 | cos of the target heading | -1 | 1 | |
| 4 | kick speed (`kicker_force`) | 0 | 6.5 | m/s, `6.5 * a` for `a > 0`; `a <= 0` does not fire |
| 5 | chip speed (`chipper_force`) | 0 | 6.5 | m/s, `6.5 * a` for `a > 0`; `a <= 0` does not fire |
| 6 | dribbler bar speed (`dribbler_vel`) | 0 | 314 | rad/s, `314 * max(a, 0)` (`DRIBBLER_OMEGA`) |

Rows 0-3 are a target pose (target velocity 0) that the navigation controller turns into a
body-frame velocity every 25 ms control step. Rows 4-5 ask for a ball speed, not a force:
`KICK_TABLE_MS` maps `6.5 * a` m/s to a one-step plunger force (`a = 1` lands at ~295-315 N;
the actuators' `ctrlrange` is 0..500 N). Each shot locks that plunger out for 250 ms; the
kicker and chipper locks are independent, so a kick cannot block a chip. Row 6 is off for
`a <= 0` and never drives the bar backwards (its `ctrlrange` is +-1500 rad/s; the action uses
only 0..314). See [the action space contract](../index.md#action-space).

## Observation Space

`Box((14,), float32)`: 6 ball elements plus 8 for the single robot. The goal is not in the
observation: it is at a fixed place on a fixed field, so its position is a constant a policy
can absorb into its weights.

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

Positions and velocities are divided by the constants in the Unit column; yaw is given as
its sine and cosine so the wrap at +-pi is not a discontinuity. Ball height is measured from
the carpet and normalised by `GOAL_HEIGHT = 0.155 m`, so a chipped ball reads greater than 1
exactly when it is too high to score. Only robot position, yaw and `has_ball` are bounded:
nothing clips, so a bound exists only where the field walls guarantee it.

## Rewards

Ported from rSoccer's `_calculate_reward_and_done` (`ssl_go_to_ball_shoot`, and the
identical one in `ssl_hw_challenge`). The three shaping terms are paid on every step that
does not end on robot out, goalkeeper area, ball out or the goal line; such a step pays only
the goal term (`+5.0` or `0`). There are no terminal penalties. The scales are rSoccer's
formulas on our field. `drive_cmd` is the navigation controller's body-frame `(vx, vy, w)`
command as a fraction of the chassis servos' `ctrlrange`, clipped to `[-1, 1]`; it stands in
for rSoccer's `sum |wheel speed| / (4 * 160 rad/s * 1200)` (the chassis is commanded in
`(vx, vy, w)`, not wheels), so a saturated drive costs `1 / 1200` a step as there.

| term | value |
|---|---|
| ball distance | `clip(prev_dist - dist, -1, 1) / BALL_DIST_SCALE`, robot-to-ball closing; `BALL_DIST_SCALE = hypot(9.0, 6.0) = 10.8` |
| ball gradient | `clip(prev - now, -1, 1) / BALL_GRAD_SCALE`, the ball closing on the goal centre `(6.0, 0)`; `BALL_GRAD_SCALE = hypot(4.5, 6.0) / 4 = 1.875` |
| energy | `-mean(\|drive_cmd\|) / 1200` |
| goal | `+5.0` on the step the ball crosses `x = 6.0` inside the mouth, `0` wide; ends the episode |

## Starting State

rSoccer's `random_init`. The ball is uniform over `x` in `[0.3, 3.9]` (0.3 m inside the half
line and off the penalty area, `6.0 - 1.8 - 0.3`) and `y` in `[-4.4, 4.4]`. The robot's first
draw is from the same box with its `y` negated; while it is within
`(ball radius + robot radius) * 1.1 = 0.122 m` of the ball on both axes it is redrawn from the
box as is (no negation). Yaw uniform over `[0, 2 pi)`. Everything starts at rest.
(GPU backends: the first clear draw out of a fixed candidate set (8 for the robot); if none is clear, the roomiest one, flagged `reset_overlap`. See [the backends](../backends/index.md).)

## Episode End

### Termination

Checked in rSoccer's order; the first that holds ends the episode:

- **Robot out.** The robot leaves the attacking half (`x < -0.2`) or the field's width
  (`|y| > 4.5`). `info["robot_out"]`.
- **Goalkeeper area.** The robot enters the +x penalty area (`x > 4.2`, `|y| < 1.8`, Div A's
  1.8 x 3.6 m). `info["gk_area"]`.
- **Ball out.** The ball leaves the attacking half (`x < 0`) or the width. `info["ball_out"]`.
- **Goal line.** The ball crosses `x = 6.0`: `+5.0` and `info["goal"]` inside the mouth (and
  under the crossbar), `0` wide.
- **Diverged simulation.** If `qpos` or `qvel` stops being finite, `step` raises
  `RuntimeError` rather than returning a NaN observation to train on. (GPU backends end that
  world with obs and reward 0, count it in `diverged` and auto-reset; see [the backends](../backends/index.md).)

### Truncation

`steps >= max_steps`, where `max_steps = round(horizon_s / dt)`. The default horizon is
**60 s**, rSoccer's: **2400 control steps** at the fixed 40 Hz (0.025 s), rSoccer's step.

## Arguments

| parameter | type | default | description |
|---|---|---|---|
| `repeat_action` | `int` | `1` | control steps (25 ms each, fixed 40 Hz) one decision is held for |
| `horizon_s` | `float` | `60.0` | truncation horizon, in seconds |
| `render_mode` | `str \| None` | `None` | `"human"` or `"rgb_array"` |
| `viewer_gfx` | `str` | `"opengl_headless"` | how `"human"` mode draws: `"opengl_headless"` or `"web"` |

```python
import gymnasium as gym
import mjsoccer

env = gym.make("GoToBallShoot-v0", horizon_s=60.0, render_mode="rgb_array")
```

`n_yellow` (0) and `energy_steps` (1200, the energy term's divisor) are class attributes, not `gym.make` keywords; the robot counts determine the compiled model, and therefore the observation size. The GPU backends read them off the CPU class.

## Info dictionary

| key | type | description |
|---|---|---|
| `is_success` | `bool` | the ball ended up in the goal |
| `goal` | `bool` | present on a step the ball crossed the goal line: inside the mouth or not |
| `robot_out` | `bool` | present when the robot left the attacking half or the width |
| `gk_area` | `bool` | present when the robot entered the penalty area |
| `ball_out` | `bool` | present when the ball left the attacking half or the width |

These keys are returned by the CPU (Gymnasium) env only; the GPU backends do not return them. See [the backends](../backends/index.md).

## Version History

- v0: initial release.
