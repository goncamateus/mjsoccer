# GoToBall-v0

![GoToBall-v0](../assets/GoToBall.gif)

| | |
|---|---|
| Action Space | `Box(-1.0, 1.0, (7,), float32)` on the default tier 1, `Box(-1.0, 1.0, (6,), float32)` on tier 0 |
| Observation Space | `Box((62,), float32)`, bounded per row below |
| import | `gym.make("GoToBall-v0")` |

## Description

One blue robot, a loose ball and six parked yellow robots are dropped at random on the
SSL Division A field. The robot has to get its centre within 0.2 m of the ball; facing does
not matter. This is rSoccer's `SSLGoToBall-v0` (`kwargs={'field_type': 2, 'n_robots_yellow': 6}`):
the yellow robots get no commands and are only in the way.

It is the one task with a closed-form solution — `mjsoccer.tasks.goto_ball_policy` is a
P-controller that solves it — which makes it the smoke test of the suite. The kicker, chipper
and dribbler are in the action space but the reward ignores them.

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

Rows 0-3 on tier 1 are wheel speeds written straight to the servos; what the chassis does
with them is left to the roller contacts, which can slip. Rows 0-2 on tier 0 are instead a
*body-frame* velocity that is rotated into the world-axis chassis slides before being
written, which is what makes the two tiers mean the same thing. The
kicker and chipper ask for a speed: `a <= 0` does not fire, `(0, 1]` sends the ball out at
`6.5 * a` m/s, and each shot locks that plunger out for 250 ms. The dribbler is
off for `a <= 0`, maps `(0, 1]` onto `(0, 314] rad/s` and is never driven backwards. See
[the action space contract](../index.md#action-space) for why each of those is the way it
is.

## Observation Space

`Box((62,), float32)`: 6 ball elements plus 8 for each of the 7 robots, the controlled blue
robot first and then the six yellow ones, laid out exactly as in
[`StaticDefenders-v0`](static_defenders.md#observation-space).

## Rewards

rSoccer's, sparse: `+1.0` on the step the robot-to-ball distance (centre to centre) drops
under 0.2 m, which also ends the episode, and `0` on every other step. No shaping, no facing,
no energy, no penalties.

## Starting State

rSoccer's layout. The ball is uniform over the field less a 0.1 m margin (`x` in
`[-5.9, 5.9]`, `y` in `[-4.4, 4.4]`). The blue robot, then the six yellow ones, are drawn from
the same box, each resampled until it is at least 0.2 m (centre to centre) from everything
already placed, with yaw uniform over `[0, 2 pi)`. Everything starts at rest.

## Episode End

### Termination

- **Success.** Robot-to-ball distance under 0.2 m: `+1.0`.
- **Controlled robot turned over** (tier 1 only). The robot's own +z tilts more than 60 deg
  off world +z (`cos < 0.5`); the step pays `-100.0` and `info["tipped"]` is set. rSoccer's
  robots cannot tip, so this is the one ending that is ours, kept so a robot on its back never
  collects shaping.
- **Diverged simulation.** If `qpos` or `qvel` stops being finite, `step` raises
  `RuntimeError` rather than returning a NaN observation to train on.

### Truncation

`steps >= max_steps`, where `max_steps = round(horizon_s / dt)`. The default horizon is
**30 s**, rSoccer's: **1200 steps** at the 40 Hz (0.025 s) the agents act at, rSoccer's step,
and 1800 at the default `control_hz=60`.

## Arguments

| parameter | type | default | description |
|---|---|---|---|
| `tier` | `int` | `1` | `1` = roller wheels with real slip, the default; `0` = the faster kinematic chassis |
| `control_hz` | `int` | `60` | control rate, exact: it must divide the 960 Hz physics rate (30, 60, 120, ...), or the env raises `ValueError` |
| `horizon_s` | `float` | `30.0` | truncation horizon, in seconds |
| `render_mode` | `str \| None` | `None` | `"human"` or `"rgb_array"` |

```python
import gymnasium as gym
import mjsoccer

env = gym.make("GoToBall-v0", tier=0, control_hz=60, horizon_s=30.0, render_mode="rgb_array")
```

## Info dictionary

| key | type | description |
|---|---|---|
| `is_success` | `bool` | the robot got within 0.2 m of the ball |
| `dist` | `float` | robot-to-ball distance in metres after this step |
| `tipped` | `bool` | present only on the step a controlled robot turned over |
| `tilt_deg` | `float` | with `tipped`: the worst controlled robot's tilt off vertical, degrees |
| `tip_step` | `int` | with `tipped`: control steps since reset, so time-to-tip is `tip_step * dt` |

## Version History

- v0: initial release.
