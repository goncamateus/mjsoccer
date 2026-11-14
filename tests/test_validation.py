"""Ticket 14's acceptance criteria (§13.5 of
.scratch/physics-fidelity/research/rsoccer-vs-mjsoccer.md) on the §13.2 scenarios, plus
gpu-port ticket 05's for the navigation controller (`navigate`). Slow, so deselected by default:

    uv run python -m pytest tests -m validation

The scenarios themselves live in scripts/validation_sweep.py, which runs the same ones over
seeds 0-4 and writes the distributions; this asserts on one seed apiece.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

import mjsoccer
from mjsoccer.base import MAX_W

# ponytail: scripts/ is not a package; the path hack keeps one definition of each scenario.
sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from validation_sweep import (  # noqa: E402
    NAV_CHATTER_RMS, NAV_POS_ERR, NAV_TIME_RATIO, NAV_YAW_ERR_DEG, SCENARIOS, episode,
    trapezoid_time,
)

pytestmark = pytest.mark.validation

SCRIPTED = [(n, e, p) for n in ("spin", "kick_recoil") for e, p in SCENARIOS[n][1]]


@pytest.mark.parametrize("scenario,env_id,param", SCRIPTED)
def test_scripted_scenario(scenario, env_id, param):
    """Criterion 2: yaw p99 at most MAX_W in every scripted scenario. At most, not under:
    MAX_W is the chassis yaw servo's own ctrlrange, so a full spin command sits on it."""
    row = episode(scenario, env_id, param, 0)["row"]
    assert row["yaw_p99"] <= MAX_W, row


@pytest.mark.parametrize("env_id", list(mjsoccer.ENV_IDS))
def test_random_actions(env_id):
    """Criterion 2 on one seed of scenario 6."""
    row = episode("random", env_id, None, 0)["row"]
    assert row["yaw_p99"] <= MAX_W, row


@pytest.mark.parametrize("seed", range(5))
def test_competent_controller(seed):
    """Criterion 2 on scenario 7: a controller that drives at the ball."""
    row = episode("competent", "GoToBall-v0", None, seed)["row"]
    assert row["yaw_p99"] <= MAX_W, row


NAV = [p for _, p in SCENARIOS["navigate"][1]]


@pytest.mark.parametrize("param", NAV, ids=[f"{d}m-{t}deg" for d, t in NAV])
def test_navigation(param):
    """Ticket 05: the navigation controller, from rest, to a target pose d metres ahead with
    a turn of t degrees. Arrival, time against the analytic trapezoid, chatter at the target
    and the commanded yaw ceiling. Thresholds are the CPU measurement plus margin."""
    row = episode("navigate", "GoToBall-v0", param, 0)["row"]
    assert row["nav_pos_err"] < NAV_POS_ERR, row
    assert row["nav_yaw_err_deg"] < NAV_YAW_ERR_DEG, row
    assert row["nav_time_ratio"] <= NAV_TIME_RATIO, row
    assert row["nav_chatter_rms"] < NAV_CHATTER_RMS, row
    assert row["nav_cmd_w_max"] <= MAX_W, row


def test_trapezoid_time():
    """The analytic profile the time criterion compares against: 3.3 m/s, up at 1.30 m/s^2,
    down at 1.0. Worked by hand: 6 m never reaches 3.3 m/s, peaks at sqrt(12 / (1/1.3 + 1))."""
    v = (12 / (1 / 1.3 + 1)) ** 0.5
    assert trapezoid_time(6.0) == pytest.approx(v / 1.3 + v / 1.0)
    # 20 m cruises: 3.3/1.3 + 3.3/1.0 s of ramps covering 4.188 + 5.445 m, the rest at 3.3.
    assert trapezoid_time(20.0) == pytest.approx(3.3 / 1.3 + 3.3 + (20 - 4.18846 - 5.445) / 3.3, rel=1e-4)
