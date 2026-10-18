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

import pytest

import mjsoccer
from mjsoccer.base import MAX_W

# ponytail: scripts/ is not a package; the path hack keeps one definition of each scenario.
sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from validation_sweep import SCENARIOS, episode  # noqa: E402

pytestmark = pytest.mark.validation

SCRIPTED = [(n, e, t, p) for n in ("spin", "head_on", "wall", "dribble_turn", "kick_recoil")
            for e, t, p in SCENARIOS[n][1]]


@pytest.mark.parametrize("scenario,env_id,tier,param", SCRIPTED)
def test_scripted_scenario(scenario, env_id, tier, param):
    """Criteria 2 and 3: yaw p99 under MAX_W in every scenario, roll/pitch p99 under 15 deg
    in the dribbling turn and the kick recoil. And nothing scripted tips.

    Measured (ticket 14): worst tilt 5.3 deg (head-on at 3 m/s), worst yaw 33.3 rad/s
    (6 m/s grazing the wall at 30 deg) -- 7 % under the 35.9 kinematic ceiling.
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


@pytest.mark.parametrize("axis", (0, 1), ids=("forward", "lateral"))
def test_achievable_acceleration(axis):
    """Criterion 5: tier 1 accelerates within 1.5x of the derived 2.30 m/s^2 (tickets 01/05,
    the figure tier 0 is given by ticket 12). Measured 2.47 forward, 1.87 lateral, best
    0.1 s window from rest. Top speed is not asserted: 2.14 forward and 0.43 lateral
    against 3.30 is the open wheel-contact loss of ticket 18."""
    import gymnasium as gym
    import numpy as np

    from mjsoccer.tasks import wheel_command

    env = gym.make("GoToBall-v0", tier=1).unwrapped
    env.reset(seed=0)
    r = env.robots[0]
    r.place(env.data, -5.0, -3.5, 0.6435)  # 7 m of diagonal before a wall
    env.ball.place(env.data, 5.5, 4.0)
    cmd = np.zeros(3)
    cmd[axis] = 10.0  # saturates: the wheels' whole envelope on that axis
    a = np.zeros(env.action_space.shape, dtype=np.float32)
    v = []
    for _ in range(30):
        a[:4] = wheel_command(env, 0, *cmd)
        env.step(a)
        R = env.data.xmat[r.body].reshape(3, 3)
        vx, vy, _ = r.velocity(env.data)
        v.append(float(np.linalg.norm((R.T @ (vx, vy, 0.0))[:2])))
    env.close()
    k = round(0.1 / env.dt)
    acc = max(v[i + k] - v[i] for i in range(len(v) - k)) / (k * env.dt)
    assert 2.30 / 1.5 < acc < 2.30 * 1.5, f"{acc:.2f} m/s^2"
