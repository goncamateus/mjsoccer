"""The navigation controller: turns a target pose into a body-frame velocity, every control step.

`go_to_point_new` from Planning-the-path-with-rl (`envs/navigation.py`, commit 2287531),
ported as a pure function. Everything it does is kept, quirks included
(.scratch/gpu-port/issues/07-navigation-controller-port.md):

- MAX_DV = 0.1 m/s per call, added to the *measured* velocity: 4 m/s^2 asked at 40 Hz,
  and the chassis servo is what actually limits acceleration (1.30 m/s^2).
- Braking is planned at MAX_ACCEL = 1 m/s^2.
- No position deadband: at the target it chatters +-0.1 m/s, which the servo damps.
- A 2.5 deg deadzone on heading, then omega = 2 * error with no limit of its own (the
  yaw servo's ctrlrange cuts it).
- max_velocity is per axis, in the frame rotated onto the robot->target line.

Two changes only: `entry` is no longer mutated (the original built a fresh one per step,
so nothing observable changes) and max_velocity defaults to MAX_V = 3.3 m/s, the chassis
envelope, rather than 2.2.
"""

from __future__ import annotations

import math

import numpy as np

MAX_VELOCITY = 3.3  # m/s, base.MAX_V; the original's 2.2 was its own robot's
ANGLE_KP = 2.0
MAX_ACCEL = 1.0
MAX_DV = 0.1
EPS = 10e-4
MIN_ANGLE_TO_ROTATE = np.deg2rad(2.5)


def _rotate(v, angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.dot(np.array([[c, -s], [s, c]]), np.array([v[0], v[1]]))


def smallest_angle_diff(a: float, b: float) -> float:
    """b - a wrapped to [-pi, pi), the original's way (fmod, then shift)."""
    angle = b - a
    mod = 2 * math.pi
    if not -mod <= angle <= mod:
        angle = math.fmod(angle, mod)
    if angle < 0:
        angle += mod
    if angle >= math.pi:
        angle -= mod
    elif angle < -math.pi:
        angle += mod
    return angle


def _axis(to_target: float, v_current: float, v_desired: float, target: float, pos: float,
          max_velocity: float) -> float:
    """One axis of the rotated frame: accelerate toward the target, or brake for it."""
    passed = to_target * v_desired < 0
    moving_away = to_target * v_current < 0
    if passed or moving_away:  # aim at the point where v_desired would stop instead
        zero_point = target - np.sign(v_desired) * v_desired**2 / (2 * MAX_ACCEL)
        to_target = zero_point - pos
        v_desired = 0
    if abs(to_target) < abs(v_current**2 - v_desired**2) / (2 * MAX_ACCEL):
        v = v_current + np.sign(v_desired - v_current) * MAX_DV
    else:
        v = v_current + np.sign(to_target) * MAX_DV
    if abs(v) > max_velocity:
        v = np.sign(v) * max_velocity
    return v


def go_to_point(
    pos,
    vel,
    angle: float,
    target,
    target_angle: float,
    target_velocity=(0.0, 0.0),
    max_velocity: float = MAX_VELOCITY,
    k_p: float = ANGLE_KP,
) -> tuple[float, float, float]:
    """(vx, vy) in the robot's body frame and omega, from the robot's world-frame position,
    velocity and yaw and a target pose. Units are metres, m/s and radians."""
    if np.linalg.norm(target_velocity) <= EPS:
        target_velocity_angle = np.arctan2(target[1] - pos[1], target[0] - pos[0])
    else:
        target_velocity_angle = np.arctan2(target_velocity[1], target_velocity[0])

    tv = _rotate(target_velocity, -target_velocity_angle)
    v = _rotate(vel, -target_velocity_angle)
    t = _rotate(target, -target_velocity_angle)
    p = _rotate(pos, -target_velocity_angle)
    to_target = (t[0] - p[0], t[1] - p[1])

    vx = _axis(to_target[0], v[0], tv[0], t[0], p[0], max_velocity)
    vy = _axis(to_target[1], v[1], tv[1], t[1], p[1], max_velocity)

    world = _rotate((vx, vy), target_velocity_angle)
    c, s = np.cos(angle), np.sin(angle)
    local_x = c * world[0] + s * world[1]
    local_y = -s * world[0] + c * world[1]

    turn = smallest_angle_diff(angle, target_angle)
    w = 0 if abs(turn) < MIN_ANGLE_TO_ROTATE else k_p * turn
    return float(local_x), float(local_y), float(w)
