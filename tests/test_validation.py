"""Ticket 14's acceptance criteria (§13.5 of
.scratch/physics-fidelity/research/rsoccer-vs-mjsoccer.md) on the §13.2 scenarios, gpu-port
ticket 05's for the navigation controller (`navigate`), and gpu-port ticket 12's MJX-Warp x
CPU parity. Slow, so deselected by default:

    uv run python -m pytest tests -m validation                       # CPU; MJX skipped
    uv sync --group gpu && systemd-run --user --scope -p MemoryMax=10G \\
        uv run python -m pytest tests -m validation                   # CPU + MJX + parity

Every criterion runs on both backends with the same threshold (`backend` parameter); the
MJX cases skip without the `gpu` group or a CUDA device. The scenarios live in
scripts/validation_sweep.py. Each backend's whole sweep (`--seeds 0-4` on the CPU, `0-49` on
MJX) runs once per session, in a subprocess (the MJX one must not inherit the JAX-CPU of
tests/test_mjx_task_logic.py, and gives the GPU back on exit), and the tests read its
episodes.csv.

Parity (spec, Testing Decisions; .scratch/gpu-port/issues/09-acceptance-and-parity.md):
  deterministic -- same initial qpos/qvel on both: |MJX - CPU| / CPU <= 2 % on the envelope,
                   the kick and the COR, <= 5 % on the arrival time; arrival error within
                   1 cm / 1 deg absolute;
  seeded        -- `random` and `competent` on every env, seeds 0-4 on the CPU and 0-49
                   on MJX (one episode / one world per seed), the CPU median of the return
                   and of the possession rate inside the MJX range (min-max; the planned
                   [p25, p75] rejects identical distributions half the time, see
                   `test_parity_seeded`). No seed-by-seed pairing, no distribution test.
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import mjsoccer
from mjsoccer.base import MAX_W

# ponytail: scripts/ is not a package; the path hack keeps one definition of each scenario.
SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from validation_sweep import (  # noqa: E402
    BOUNCE_GRID, COR_RANGE, ENVELOPE, ENVELOPE_REL, HOLD_TURN_COPIES, HOLD_TURN_MEDIAN_MIN,
    HOLD_TURN_MIN, KICK_ABS, KICK_GRID,
    NAV_CHATTER_RMS, NAV_POS_ERR, NAV_TIME_RATIO, NAV_YAW_ERR_DEG, NO_MJX, SCENARIOS,
    trapezoid_time,
)

pytestmark = pytest.mark.validation

BACKENDS = ["cpu", "mjx"]
SEEDS = range(5)  # the CPU's, and the per-seed criteria's on both backends
# MJX's for the seeded parity only (extra worlds of the batch are cheap): the smallest count
# that keeps the simulated false-failure rate under target, see
# test_mjx_seed_count_holds_the_false_failure_rate.
MJX_SEEDS = range(50)


def _value(v: str):
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return {"True": True, "False": False}.get(v, v)


@pytest.fixture(scope="session")
def sweep(tmp_path_factory):
    """backend -> {(scenario, env, str(param), seed): row}, each backend's whole sweep run
    once in a subprocess. The MJX one skips (exit NO_MJX) without the gpu group or CUDA."""
    cache = {}

    def get(backend):
        if backend not in cache:
            out = tmp_path_factory.mktemp(f"sweep-{backend}")
            env = {k: v for k, v in os.environ.items() if k != "JAX_PLATFORMS"}
            seeds = MJX_SEEDS if backend == "mjx" else SEEDS
            p = subprocess.run(
                [sys.executable, str(SCRIPTS / "validation_sweep.py"), "--backend", backend,
                 "--seeds", f"{seeds[0]}-{seeds[-1]}", "--out", str(out)],
                env=env, capture_output=True, text=True,
            )
            if p.returncode == NO_MJX:
                cache[backend] = p.stdout.strip() or "MJX unavailable"
            else:
                assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-4000:]
                with open(out / "episodes.csv") as f:
                    rows = [{k: _value(v) for k, v in r.items()} for r in csv.DictReader(f)]
                cache[backend] = {(r["scenario"], r["env"], str(r["param"]), r["seed"]): r
                                  for r in rows}
        if isinstance(cache[backend], str):
            pytest.skip(cache[backend])
        return cache[backend]

    return get


def _row(sweep, backend, scenario, env_id, param, seed=0):
    return sweep(backend)[(scenario, env_id, "" if param is None else str(param), seed)]


SCRIPTED = [(n, e, p) for n in ("spin", "kick_recoil") for e, p in SCENARIOS[n][1]]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("scenario,env_id,param", SCRIPTED)
def test_scripted_scenario(sweep, backend, scenario, env_id, param):
    """Criterion 2: yaw p99 at most MAX_W in every scripted scenario. At most, not under:
    MAX_W is the chassis yaw servo's own ctrlrange, so a full spin command sits on it."""
    row = _row(sweep, backend, scenario, env_id, param)
    assert row["yaw_p99"] <= MAX_W, row


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("env_id", list(mjsoccer.ENV_IDS))
def test_random_actions(sweep, backend, env_id):
    """Criterion 2 on scenario 6, seeds 0-4."""
    for seed in SEEDS:
        row = _row(sweep, backend, "random", env_id, None, seed)
        assert row["yaw_p99"] <= MAX_W, row


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("seed", SEEDS)
def test_competent_controller(sweep, backend, seed):
    """Criterion 2 on scenario 7: a controller that drives at the ball."""
    row = _row(sweep, backend, "competent", "GoToBall-v0", None, seed)
    assert row["yaw_p99"] <= MAX_W, row


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("axis", list(ENVELOPE))
def test_envelope(sweep, backend, axis):
    """The chassis envelope through the raw velocity path, from rest: 1.30 m/s^2 up to
    3.30 m/s on either slide, 55 rad/s^2 up to 37.1 rad/s on the hinge (ADR 0001)."""
    _, a_max, v_max = ENVELOPE[axis]
    row = _row(sweep, backend, "envelope", "GoToBall-v0", axis)
    assert row["env_accel"] == pytest.approx(a_max, rel=ENVELOPE_REL), row
    assert row["env_v_end"] == pytest.approx(v_max, rel=ENVELOPE_REL), row


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("param", KICK_GRID, ids=[f"{p}-{a}" for p, a in KICK_GRID])
def test_kick_exit_speed(sweep, backend, param):
    """Kick and chip leave at KICK_SPEED_MAX * a by KICK_TABLE_MS, to 0.1 m/s."""
    row = _row(sweep, backend, "kick", "Dribbling-v0", param)
    assert abs(row["kick_speed"] - row["kick_target"]) <= KICK_ABS, row


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("param", BOUNCE_GRID, ids=[f"{s}-{v}" for s, v in BOUNCE_GRID])
def test_ball_bounce(sweep, backend, param):
    """COR 0.47-0.56 off the carpet and the wall at 1 and 6.5 m/s, and the ball's inertia
    the measured golf ball's (I = 0.5 m r^2) in the backend's own model."""
    row = _row(sweep, backend, "bounce", "GoToBall-v0", param)
    assert COR_RANGE[0] <= row["cor"] <= COR_RANGE[1], row
    assert row["ball_beta"] == pytest.approx(0.5, rel=1e-5), row


def test_seated_ball_survives_a_2_rad_s_turn(sweep):
    """Possession: a seated ball through a 2 rad/s turn, bar at full speed -- the CPU's
    exact trajectory, a bit-for-bit regression check (78/120, bar 5 % under).

    CPU only (ticket 16, user's decision): the bar's stick-slip makes the hold count
    chaotic, and the 78 is one trajectory that repeats only because the CPU is
    deterministic. 1e-5 of noise on qvel, or rounding the state to float32 each substep
    (69), spreads it. MJX-Warp is float32 and not deterministic run to run, so it is judged
    on the spread instead: test_seated_ball_turn_over_perturbed_copies."""
    row = _row(sweep, "cpu", "hold_turn", "Dribbling-v0", 2.0)
    assert row["held_frac"] >= HOLD_TURN_MIN, row


@pytest.mark.parametrize("backend", BACKENDS)
def test_seated_ball_turn_over_perturbed_copies(sweep, backend):
    """The same turn from HOLD_TURN_COPIES copies of its state, each with 1e-5 m/s / rad/s
    of Gaussian noise on qvel: the median hold at least HOLD_TURN_MEDIAN_MIN on both
    backends. The bar comes from the perturbed CPU (ticket 16): medians of 64 copies 71-73
    over five noise draws, MJX-Warp 73.5; 69/120 is 5 % under the CPU's pooled 72, as 74
    is under the exact 78."""
    held = [_row(sweep, backend, "hold_turn_noisy", "Dribbling-v0", k)["held_frac"]
            for k in range(HOLD_TURN_COPIES)]
    assert np.median(held) >= HOLD_TURN_MEDIAN_MIN, sorted(round(h * 120) for h in held)


NAV = [p for _, p in SCENARIOS["navigate"][1]]
NAV_IDS = [f"{d}m-{t}deg" for d, t in NAV]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("param", NAV, ids=NAV_IDS)
def test_navigation(sweep, backend, param):
    """Ticket 05: the navigation controller, from rest, to a target pose d metres ahead with
    a turn of t degrees. Arrival, time against the analytic trapezoid, chatter at the target
    and the commanded yaw ceiling. Thresholds are the CPU measurement plus margin."""
    row = _row(sweep, backend, "navigate", "GoToBall-v0", param)
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


# ------------------------------------------------------------------------ parity

REL, TIME_REL, POS_ABS, YAW_ABS_DEG = 0.02, 0.05, 0.01, 1.0
# (scenario, env, param, column, kind): "rel" = |MJX - CPU| / |CPU| <= tol, "abs" = |MJX - CPU|
DETERMINISTIC = (
    [("envelope", e, p, c, "rel", REL) for e, p in SCENARIOS["envelope"][1]
     for c in ("env_accel", "env_v_end")]
    + [("spin", e, p, "yaw_p99", "rel", REL) for e, p in SCENARIOS["spin"][1]]
    + [("kick", e, p, "kick_speed", "rel", REL) for e, p in SCENARIOS["kick"][1]]
    + [("bounce", e, p, "cor", "rel", REL) for e, p in SCENARIOS["bounce"][1]]
    + [("navigate", e, p, c, kind, tol) for e, p in SCENARIOS["navigate"][1]
       for c, kind, tol in (("nav_settle_s", "rel", TIME_REL), ("nav_pos_err", "abs", POS_ABS),
                            ("nav_yaw_err_deg", "abs", YAW_ABS_DEG))]
)


@pytest.mark.parametrize("scenario,env_id,param,column,kind,tol", DETERMINISTIC,
                         ids=[f"{s}-{p}-{c}" for s, _, p, c, _, _ in DETERMINISTIC])
def test_parity_deterministic(sweep, scenario, env_id, param, column, kind, tol):
    """The same scripted scenario from the same initial qpos/qvel on both backends: each
    scalar within the cross-backend tolerance."""
    mjx = _row(sweep, "mjx", scenario, env_id, param)[column]
    cpu = _row(sweep, "cpu", scenario, env_id, param)[column]
    err = abs(mjx - cpu) / abs(cpu) if kind == "rel" else abs(mjx - cpu)
    assert err <= tol, f"{column}: MJX {mjx} vs CPU {cpu} ({kind} {err:.4g} > {tol})"


SEEDED = [(s, e, c) for s in ("random", "competent") for e in mjsoccer.ENV_IDS
          for c in ("ret", "held_frac")]
SEEDED_FAMILY_FALSE_FAIL = 0.02  # target: P(any of the SEEDED cells fails | equal distributions)


def _seeded_parity_ok(cpu, mjx):
    """The seeded parity rule, over the last axis: the CPU median inside MJX's [min, max]."""
    med = np.median(cpu, axis=-1)
    return (np.min(mjx, axis=-1) <= med) & (med <= np.max(mjx, axis=-1))


def test_mjx_seed_count_holds_the_false_failure_rate():
    """MJX_SEEDS is the smallest count in the grid whose simulated family-wise false-failure
    rate (all SEEDED cells, independent, both backends drawing from one continuous
    distribution) stays under SEEDED_FAMILY_FALSE_FAIL, and still catches a gross divergence:
    a 3-sigma shift fails a cell most of the time. Ties only help (the range is inclusive),
    so the continuous case is the worst one."""
    rng = np.random.default_rng(0)
    trials = 200_000

    def family_fail(m, shift=0.0):
        cpu = rng.standard_normal((trials, len(SEEDS)))
        mjx = rng.standard_normal((trials, m)) + shift
        cell = 1 - np.mean(_seeded_parity_ok(cpu, mjx))
        return 1 - (1 - cell) ** len(SEEDED), cell

    grid = (10, 20, 30, 40, 50, 60)
    passing = [m for m in grid if family_fail(m)[0] <= SEEDED_FAMILY_FALSE_FAIL]
    assert passing and len(MJX_SEEDS) == passing[0], (len(MJX_SEEDS), passing)
    assert family_fail(len(MJX_SEEDS), shift=3.0)[1] > 0.8


@pytest.mark.parametrize("scenario,env_id,column", SEEDED)
def test_parity_seeded(sweep, scenario, env_id, column):
    """Seeds 0-4 on the CPU, MJX_SEEDS on MJX (one episode / one world apiece): the CPU
    median inside the MJX range (min to max), see `_seeded_parity_ok`.

    Revised twice (ticket 12), as the spec allows. The planned [p25, p75] with 5 seeds a
    side rejects identical distributions half the time (failed 5 of 24 cells, all
    `competent`). The MJX median inside the CPU's 5-seed range then still failed a fixed
    fraction of cells under the null -- 0.17 per cell at 5 MJX seeds and never under 0.0625
    however many (the true median itself falls outside 5 draws' range 2/32 of the time), and
    it did fail (`competent` StaticDefenders' return, MJX-Warp being non-deterministic run to
    run). So the roles swap: the 5 CPU seeds give the median, the many cheap MJX worlds the
    range, and the false-failure rate falls with MJX_SEEDS. A check for gross divergence (a
    3-sigma shift fails a cell > 0.8 of the time), not a test of equal distributions."""
    mjx = [_row(sweep, "mjx", scenario, env_id, None, s)[column] for s in MJX_SEEDS]
    cpu = [_row(sweep, "cpu", scenario, env_id, None, s)[column] for s in SEEDS]
    assert _seeded_parity_ok(cpu, mjx), (
        f"{column}: CPU median {np.median(cpu):.4g} {cpu} outside MJX "
        f"[{min(mjx):.4g}, {max(mjx):.4g}] (median {np.median(mjx):.4g}, {len(mjx)} seeds)")
