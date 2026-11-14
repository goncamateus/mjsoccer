"""The navigation controller: Planning-the-path-with-rl's `go_to_point_new` (2287531) as a
pure function, every quirk kept, with max_velocity = MAX_V = 3.3 m/s."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

from mjsoccer.navigation import go_to_point

sys.path.insert(0, str(Path(__file__).parent))
import _ptp_navigation as ptp  # noqa: E402


def _original(pos, vel, angle, target, target_angle, max_velocity=3.3):
    entry = ptp.GoToPointEntryNew()
    entry.target = ptp.Point2D(*target)
    entry.target_angle = target_angle
    entry.target_velocity = ptp.Point2D(0.0, 0.0)
    entry.max_velocity = max_velocity
    move = ptp.go_to_point_new(ptp.Point2D(*pos), ptp.Point2D(*vel), angle, entry)
    return move.velocity.x, move.velocity.y, move.angular_velocity


def test_matches_the_original_on_random_states():
    """Bit-for-bit against the original on states across the field, including the
    no-deadband chatter at the target and the braking branch at speed."""
    rng = np.random.default_rng(0)
    for _ in range(5000):
        pos = rng.uniform((-6, -4.5), (6, 4.5))
        vel = rng.uniform(-3.5, 3.5, 2)
        angle = rng.uniform(-10, 10)  # the yaw hinge is unwrapped
        target = rng.uniform((-6, -4.5), (6, 4.5))
        if rng.random() < 0.2:
            target = pos + rng.normal(0, 0.05, 2)  # near arrival
        target_angle = rng.uniform(-np.pi, np.pi)
        got = go_to_point(pos, vel, angle, target, target_angle)
        want = _original(pos, vel, angle, target, target_angle)
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)


def test_is_pure():
    pos, vel, target = np.array([1.0, 2.0]), np.array([0.5, -0.3]), np.array([-1.0, 0.0])
    before = copy.deepcopy((pos, vel, target))
    a = go_to_point(pos, vel, 0.3, target, 1.0)
    b = go_to_point(pos, vel, 0.3, target, 1.0)
    assert a == b
    for x, y in zip(before, (pos, vel, target)):
        np.testing.assert_array_equal(x, y)


def test_accelerates_by_max_dv_per_control_step():
    """MAX_DV = 0.1 m/s per call, from the measured velocity, toward the target."""
    vx, vy, _ = go_to_point((0.0, 0.0), (0.0, 0.0), 0.0, (3.0, 0.0), 0.0)
    assert (vx, vy) == pytest.approx((0.1, 0.0))
    vx, _, _ = go_to_point((0.0, 0.0), (1.0, 0.0), 0.0, (3.0, 0.0), 0.0)
    assert vx == pytest.approx(1.1)


def test_brakes_on_max_accel_one():
    """At 2 m/s the braking distance at MAX_ACCEL = 1 is 2 m: inside it, slow by MAX_DV."""
    vx, _, _ = go_to_point((0.0, 0.0), (2.0, 0.0), 0.0, (1.9, 0.0), 0.0)
    assert vx == pytest.approx(1.9)
    vx, _, _ = go_to_point((0.0, 0.0), (2.0, 0.0), 0.0, (2.1, 0.0), 0.0)
    assert vx == pytest.approx(2.1)


def test_caps_at_max_v():
    vx, _, _ = go_to_point((-5.0, 0.0), (3.3, 0.0), 0.0, (5.0, 0.0), 0.0)
    assert vx == pytest.approx(3.3)


def test_output_is_body_frame():
    """Facing +y, a +x world move is a -y body move."""
    vx, vy, _ = go_to_point((0.0, 0.0), (0.0, 0.0), np.pi / 2, (3.0, 0.0), np.pi / 2)
    assert (vx, vy) == pytest.approx((0.0, -0.1), abs=1e-12)


@pytest.mark.parametrize(
    "dtheta,w", [(np.deg2rad(2.4), 0.0), (np.deg2rad(2.6), 2 * np.deg2rad(2.6)), (3.0, 6.0)]
)
def test_turns_at_twice_the_heading_error_past_a_2_5_degree_deadzone(dtheta, w):
    *_, got = go_to_point((0.0, 0.0), (0.0, 0.0), 0.5, (0.0, 0.0), 0.5 + dtheta)
    assert got == pytest.approx(w)


def test_heading_error_takes_the_short_way_round():
    *_, w = go_to_point((0.0, 0.0), (0.0, 0.0), 3.0, (0.0, 0.0), -3.0)
    assert w == pytest.approx(2 * (2 * np.pi - 6.0))
