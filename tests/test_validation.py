"""Ticket 14's acceptance criteria (§13.5 of
.scratch/physics-fidelity/research/rsoccer-vs-mjsoccer.md) on the
§13.2 scenarios. Slow, so deselected by default:

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
from validation_sweep import SCENARIOS, episode  # noqa: E402

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
