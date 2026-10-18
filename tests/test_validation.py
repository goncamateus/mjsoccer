"""Ticket 14's acceptance criteria (§13.5 of docs/research/rsoccer-vs-mjsoccer.md) on the
§13.2 scenarios. Slow, so deselected by default:

    uv run python -m pytest tests -m validation

The scenarios themselves live in scripts/validation_sweep.py, which runs the same ones over
seeds 0-49 and writes the distributions; this asserts on one seed apiece. Pinned on tier 1
unless a scenario says otherwise: tier 0's chassis is planar and cannot tip.
"""

from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

import mjsoccer
from mjsoccer.base import MAX_W

# ponytail: scripts/ is not a package; the path hack keeps one definition of each scenario.
sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from validation_sweep import SCENARIOS, episode  # noqa: E402

pytestmark = pytest.mark.validation

SCRIPTED = [(n, e, t, p)
            for n in ("spin", "head_on", "wall", "dribble_turn", "kick_recoil")
            for e, t, p in SCENARIOS[n][1]]


@pytest.mark.parametrize("scenario,env_id,tier,param", SCRIPTED)
def test_scripted_scenario(scenario, env_id, tier, param):
    """Criteria 2 and 3: yaw p99 under MAX_W in every scenario, roll/pitch p99 under 15 deg
    in the dribbling turn and the kick recoil. And nothing scripted tips.

    Measured (ticket 20): worst tilt 6.3 deg (head-on at 4 m/s), worst yaw 36.2 rad/s in
    the full spins (MAX_W = 37.1, ticket 12), 33.5 rad/s (6 m/s grazing the wall at 30 deg)
    outside them.
    """
    row = episode(scenario, env_id, tier, param, 0)["row"]
    assert not row["tipped"], row
    assert row["yaw_p99"] < MAX_W, row
    if scenario in ("dribble_turn", "kick_recoil"):
        assert row["tilt_p99"] < 15.0, row


@pytest.mark.parametrize("env_id", list(mjsoccer.ENV_IDS))
@pytest.mark.parametrize("tier", (0, 1))
def test_random_actions(env_id, tier):
    """Criteria 1-3 on one seed of scenario 6. The rate (< 1 tip per episode-minute) and
    the pooled tilt bound (p95 < 4, p99 < 5 deg) need the 50-seed sweep; one episode
    without a tip is their regression check. Worst single-episode p99 measured: 5.05 deg."""
    row = episode("random", env_id, tier, None, 0)["row"]
    assert not row["tipped"], row
    assert row["yaw_p99"] < MAX_W, row
    assert row["tilt_p99"] < 15.0, row


@pytest.mark.parametrize("tier", (0, 1))
@pytest.mark.parametrize("seed", range(5))
def test_competent_controller(tier, seed):
    """Criteria 1-3 on scenario 7: a controller that drives at the ball never tips."""
    row = episode("competent", "GoToBall-v0", tier, None, seed)["row"]
    assert not row["tipped"], row
    assert row["yaw_p99"] < MAX_W, row
    assert row["tilt_p99"] < 15.0, row


def _drive(axis: int, seconds: float, ramp: float = np.inf) -> tuple[np.ndarray, float]:
    """Tier 1 from rest along body axis 0 (x) or 1 (y), wheel_command saturating, on the
    field diagonal; the commanded speed rises at `ramp` m/s^2. Body-frame speed per step, dt."""
    from mjsoccer.tasks import wheel_command

    env = gym.make("GoToBall-v0", tier=1).unwrapped
    env.reset(seed=0)
    r = env.robots[0]
    r.place(env.data, -5.5, -3.8, 0.6435 - axis * np.pi / 2)  # the axis on the diagonal
    env.ball.place(env.data, 5.5, -4.0)  # off the path
    a = np.zeros(env.action_space.shape, dtype=np.float32)
    v = []
    for i in range(round(seconds / env.dt)):
        cmd = np.zeros(3)
        cmd[axis] = min(10.0, ramp * (i + 1) * env.dt)
        a[:4] = wheel_command(env, 0, *cmd)
        env.step(a)
        R = env.data.xmat[r.body].reshape(3, 3)
        v.append(float((R.T @ (*r.velocity(env.data)[:2], 0.0))[axis]))
    env.close()
    return np.array(v), env.dt


@pytest.mark.parametrize("axis,a_max", ((0, 1.95), (1, 1.30)), ids=("forward", "lateral"))
def test_achievable_acceleration(axis, a_max):
    """Criterion 5: tier 1 accelerates within 1.5x of its derived, friction-bound figure on
    each axis, 1.95 forward and 1.30 lateral (tickets 19, 12; tier 0 gets the 1.30).
    Measured 2.24 / 1.63 (ticket 20), best 0.1 s window from rest along the commanded axis:
    the launch transient, above the sustained figure the band is centred on."""
    v, dt = _drive(axis, 0.5)
    k = round(0.1 / dt)
    acc = (v[k:] - v[:-k]).max() / (k * dt)
    assert a_max / 1.5 < acc < a_max * 1.5, f"{acc:.2f} m/s^2"


@pytest.mark.parametrize("axis,ramp,v_kin", ((0, np.inf, 3.30), (1, 1.0, 4.04)),
                         ids=("forward", "lateral"))
def test_top_speed(axis, ramp, v_kin):
    """Tier 1 reaches the kinematic top speed of tickets 01/04 (3.30 frontal, 4.04 lateral)
    because its rollers spin free (ticket 20). Lateral ramps at 1 m/s^2: a full lateral step
    turns into yaw, the friction limit with the front wheels 22 % more loaded (ticket 19),
    which nothing in the model is meant to stop. A braked roller drags along the wheel axle
    and plateaus forward near 2.4 m/s."""
    v, _ = _drive(axis, 4.5, ramp)
    assert v[-1] > 0.97 * v_kin, f"{v[-1]:.2f} m/s"
