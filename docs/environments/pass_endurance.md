# PassEndurance-v0

![PassEndurance-v0](../assets/PassEndurance.gif)

| | |
|---|---|
| Action Space | `Box(-1.0, 1.0, (7,), float32)` |
| Observation Space | `Box((22,), float32)`, bounded per row below |
| import | `gym.make("PassEndurance-v0")` |

## Description

rSoccer's `SSLPassEndurance-v0`: the passer starts with the ball in its mouth and can only
turn, kick, chip and dribble — it never translates. A stationary team-mate, dribbler running,
faces it from the other side of the line `y = 0`. The pass has to arrive: the episode pays +1
when the receiver holds the ball, and -1 for a wrong ball.

The action space is still the shared one; the env reduces the passer's drive to a spin
(it zeroes `vx, vy` and keeps the yaw rate).

## Action Space

The action is a `Box(-1, 1, (7,), float32)` and drives the passer only. The team-mate is
scripted and never appears in it, which is why the action is 7 wide even though the model
contains two robots.

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

The passer cannot translate: the navigation controller runs whole and then its `vx, vy` go
to 0, so rows 0 and 1 are ignored and only the heading (rows 2-3) turns it. The kicker
(row 4) is the natural control: pass strength is the action. The chipper is the alternative
over an obstacle, though there are no obstacles here.

## Observation Space

`Box((22,), float32)`: 6 ball elements plus 8 for each of the 2 robots. Both robots are
blue: the controlled passer first, the scripted receiver second.

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
| 14 | x position of the blue team-mate (scripted) | -1.05 | 1.05 | `b1_robot` | m, / 6.0 (`FIELD_HALF_X`) |
| 15 | y position of the blue team-mate (scripted) | -1.067 | 1.067 | `b1_robot` | m, / 4.5 (`FIELD_HALF_Y`) |
| 16 | sin of the yaw of the blue team-mate (scripted) | -1 | 1 | `b1_robot` | unitless |
| 17 | cos of the yaw of the blue team-mate (scripted) | -1 | 1 | `b1_robot` | unitless |
| 18 | world-frame x velocity of the blue team-mate (scripted) | -Inf | Inf | `b1_robot` | m/s, / 3.3 (`MAX_V`) |
| 19 | world-frame y velocity of the blue team-mate (scripted) | -Inf | Inf | `b1_robot` | m/s, / 3.3 (`MAX_V`) |
| 20 | yaw rate of the blue team-mate (scripted) | -Inf | Inf | `b1_robot` | rad/s, / 37.1 (`MAX_W`) |
| 21 | whether the blue team-mate (scripted) has the ball | 0 | 1 | `BaseEnv.holds` | 0 or 1 |

Positions and velocities are divided by the constants in the Unit column; yaw is given as
its sine and cosine so the wrap at +-pi is not a discontinuity. Ball height is measured from
the carpet and normalised by `GOAL_HEIGHT = 0.155 m`, so a chipped ball reads greater than 1
exactly when it is too high to score. Only robot position, yaw and `has_ball` are bounded:
nothing clips, so a bound exists only where the field walls guarantee it.

Elements 14 and 15 are the target of the pass, the only thing that varies between episodes
other than the passer's and the ball's start, since the receiver does not move. Elements 18
to 20 are zero unless the receiver has been hit. Element 21 is the success condition itself:
the step it goes to 1 is the last step of the episode.

## Rewards

rSoccer's, per step:

| term | value |
|---|---|
| received | `+1.0` when the receiver holds the ball (`holds(1)`, rSoccer's infrared); ends the episode |
| ball gradient | otherwise `clip(prev - now, -1, 1) / 1.875`, the ball closing on the receiver |
| wrong ball | `-1.0` added, and the episode ends, when the ball is outside the box the two robots span (positions truncated to whole centimetres, as rSoccer compares them), or its distance to the receiver has changed by under 0.01 m on more than 20 consecutive steps |

The stopped rule means the passer has about half a second (21 steps at 40 Hz) to kick.
rSoccer also ends on `holding_steps > 15`, but never increments it, so it is left out.

## Starting State

rSoccer's layout. The ball is uniform over `[-1.5, 1.5]^2`. The passer stands 0.115 m from it
on the side away from `y = 0`, facing it (yaw -90 deg for a ball at `y > 0`, +90 deg otherwise),
and `seat_ball` rolls the ball into its mouth. The receiver is at `(x, -ball_y)`, `x` uniform
over `[-1.5, 1.5]` redrawn until it is at least 1 m from the ball's `x`, facing the passer.
(GPU backends: the first clear draw out of a fixed candidate set (64 for the receiver); if none is clear, the roomiest one, flagged `reset_overlap`. See [the backends](../backends/index.md).)

## Episode End

### Termination

Received is checked first, then wrong ball, which is always evaluated. Both on the same step
net reward `0` (`+1 - 1`), end the episode and still set `info["is_success"]`.

- **Received.** The receiver holds the ball: `+1.0`, `info["is_success"]`.
- **Wrong ball.** Out of the passer-receiver box, or stopped for more than 20 steps: `-1.0`.
- **Diverged simulation.** If `qpos` or `qvel` stops being finite, `step` raises
  `RuntimeError` rather than returning a NaN observation to train on. (GPU backends end that
  world with obs and reward 0, count it in `diverged` and auto-reset; see [the backends](../backends/index.md).)

### Truncation

`steps >= max_steps`, where `max_steps = round(horizon_s / dt)`. The default horizon is
**30 s**, rSoccer's: **1200 control steps** at the fixed 40 Hz (0.025 s), rSoccer's step.

## Arguments

| parameter | type | default | description |
|---|---|---|---|
| `repeat_action` | `int` | `1` | control steps (25 ms each, fixed 40 Hz) one decision is held for |
| `horizon_s` | `float` | `30.0` | truncation horizon, in seconds |
| `render_mode` | `str \| None` | `None` | `"human"` or `"rgb_array"` |
| `viewer_gfx` | `str` | `"opengl_headless"` | how `"human"` mode draws: `"opengl_headless"` or `"web"` |

```python
import gymnasium as gym
import mjsoccer

env = gym.make("PassEndurance-v0", horizon_s=30.0, render_mode="rgb_array")
```

`n_blue` (2) and `stopped_limit` (20) are class attributes, not `gym.make` keywords; the robot counts determine the compiled model, and therefore the observation size. The GPU backends read them off the CPU class. `n_controlled`
stays at 1, which is why the team-mate widens the observation by 8 but leaves the action
alone.

## Info dictionary

| key | type | description |
|---|---|---|
| `is_success` | `bool` | the team-mate received the ball |

These keys are returned by the CPU (Gymnasium) env only; the GPU backends do not return them. See [the backends](../backends/index.md).

## Version History

- v0: initial release.
