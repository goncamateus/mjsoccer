# Validation

The physics acceptance criteria, the GPU × CPU parity, and the test that pins each one. All
of it lives in `tests/test_validation.py`, marked `validation`; the default `pytest` run
excludes it (`addopts` in `pyproject.toml`). The scenarios and most thresholds are defined
in `scripts/validation_sweep.py` and imported by the tests.

## Backends

Every criterion runs on all three backends with the same threshold (`BACKENDS = ["cpu",
"mjx", "mjlab"]`). Each GPU backend (`GPU = ["mjx", "mjlab"]`) is also held to the CPU by
the [parity](#gpu--cpu-parity) tests. A GPU backend's cases skip without its uv group or a
CUDA device.

Each backend's whole sweep runs once per session, in a subprocess, and the tests read its
`episodes.csv`. Seeds: 0-4 on the CPU (`SEEDS`), 0-49 on a GPU backend (`MJX_SEEDS`). The
per-seed criteria use seeds 0-4 on every backend; the extra GPU worlds feed only the seeded
parity.

## Acceptance criteria

`MAX_W` = 37.1 rad/s (`mjsoccer.base`). Constants in capitals are in
`scripts/validation_sweep.py`.

| criterion | threshold | scenario | test |
|---|---|---|---|
| yaw rate, scripted | yaw p99 ≤ `MAX_W` | `spin` (raw yaw command 0.2..1.0, 7.5 s), `kick_recoil` (full kick from 0/2/4 m/s) | `test_scripted_scenario` |
| yaw rate, random actions | yaw p99 ≤ `MAX_W`, every seed | `random`, all six envs, seeds 0-4 | `test_random_actions` |
| yaw rate, competent controller | yaw p99 ≤ `MAX_W`, every seed | `competent` (`goto_ball_policy`), GoToBall, seeds 0-4 | `test_competent_controller` |
| chassis envelope | accel and final speed within 3 % (`ENVELOPE_REL`) of 1.30 m/s² / 3.30 m/s on `vx` and `vy`, 55 rad/s² / 37.1 rad/s on `omega` (`ENVELOPE`) | `envelope`: full raw command from rest, 3 s | `test_envelope` |
| kick and chip exit speed | within 0.1 m/s (`KICK_ABS`) of `6.5 * a`, `a` in 0.5, 1 (`KICK_GRID`) | `kick`: off the seated ball, standing | `test_kick_exit_speed` |
| ball bounce | COR in 0.47-0.56 (`COR_RANGE`) off floor and wall at 1 and 6.5 m/s (`BOUNCE_GRID`); ball β = 0.5 to 1e-5 | `bounce` | `test_ball_bounce` |
| possession through a turn (CPU) | ≥ 74/120 `holds()` (`HOLD_TURN_MIN`); measured 78 | `hold_turn`: seated ball, raw 2 rad/s turn, bar at full speed, 3 s | `test_seated_ball_survives_a_2_rad_s_turn` |
| possession through a turn (all backends) | median over 64 copies (`HOLD_TURN_COPIES`) ≥ 69/120 (`HOLD_TURN_MEDIAN_MIN`) | `hold_turn_noisy`: the same, 1e-5 Gaussian noise on `qvel` per copy | `test_seated_ball_turn_over_perturbed_copies` |
| navigation: final position | < 1 cm (`NAV_POS_ERR`) | `navigate`: from rest to a pose 0.5 / 2 / 6 m ahead, turned 0 / 90 / 180 deg, 10 s | `test_navigation` |
| navigation: final heading | < 2.5 deg (`NAV_YAW_ERR_DEG`) | `navigate` | `test_navigation` |
| navigation: time | settle time ≤ 1.05 × the analytic trapezoid (`NAV_TIME_RATIO`; 3.3 m/s, up at 1.30 m/s², down at 1.0) | `navigate` | `test_navigation` |
| navigation: chatter | chassis speed RMS over the last second < 0.05 m/s (`NAV_CHATTER_RMS`) | `navigate` | `test_navigation` |
| navigation: yaw command | commanded omega ≤ `MAX_W` | `navigate` | `test_navigation` |

`yaw p99 ≤ MAX_W` is "at most", not "under": `MAX_W` is the yaw servo's own `ctrlrange`, so
a full spin command sits on it.

### Possession through a turn

The 2 rad/s turn is chaotic. The CPU's 78/120 is one trajectory that repeats only because
the CPU is deterministic; 1e-5 of noise on `qvel`, or rounding the state to float32, spreads
it to ~66-80. So the exact-trajectory check (`≥ 74`, 5 % under 78) runs on the CPU only.
Every backend instead runs 64 perturbed copies of the state and is judged on the median. The
69/120 bar is 5 % under the perturbed CPU's pooled median of 72.

### Helpers

`test_trapezoid_time` checks the analytic trapezoid by hand. `test_transplant_keeps_the_scenario_state_in_another_scene`
checks that a scripted state moved between scenes by joint and body name (`transplant`, used
to carry CPU-built scenarios into the mjlab model) keeps every pose, velocity and applied
force.

## GPU × CPU parity

Run for each GPU backend against the CPU (parametrized on `GPU`).

**Deterministic** (`test_parity_deterministic`). The same scripted scenario from the same
initial `qpos`/`qvel` on both backends:

| quantity | scenario | tolerance |
|---|---|---|
| envelope accel and final speed | `envelope` | within 2 % relative (`REL`) |
| spin yaw p99 | `spin` | within 2 % relative |
| kick and chip exit speed | `kick` | within 2 % relative |
| COR | `bounce` | within 2 % relative |
| arrival (settle) time | `navigate` | within 5 % relative (`TIME_REL`) |
| final position error | `navigate` | within 1 cm absolute (`POS_ABS`) |
| final heading error | `navigate` | within 1 deg absolute (`YAW_ABS_DEG`) |

**Seeded** (`test_parity_seeded`). Scenarios `random` and `competent` on every env, return
and possession rate (24 cells per GPU backend). The CPU median over seeds 0-4 must fall
inside the GPU min-max over seeds 0-49. No seed-by-seed pairing (JAX's RNG is not NumPy's).
This is a gross-divergence check, not a distribution test: a 3-sigma shift fails a cell
more than 80 % of the time, and under equal distributions the chance that any of the 24
cells fails is ~2 % (`SEEDED_FAMILY_FALSE_FAIL`). `test_mjx_seed_count_holds_the_false_failure_rate`
simulates that rate and pins 50 as the smallest seed count in the grid that keeps it.

GPU backends run in float32 and Warp's solver is not bitwise reproducible, so parity is
to a tolerance, never to the bit.

## Running it

CPU, about a minute (GPU cases skip):

```bash
uv run python -m pytest tests -m validation -q
```

The distributions behind the criteria, as CSV:

```bash
uv run python scripts/validation_sweep.py --out runs/validation --seeds 0-4
uv run python scripts/validation_sweep.py --scenario spin --out runs/v   # one scenario
```

`validation_sweep.py --backend {cpu,mjx,mjlab}` runs the same scenarios on any backend and
writes `episodes.csv`, `summary.csv`, `navigation.csv`, `impulse.csv` (CPU) and, with the
`random` scenario on the CPU, `throughput.csv` (steps/s per env). Every row carries its
`backend`. Its docstring lists the scenarios. A GPU backend exits 77 when it cannot run.
Throughput is a report, never a criterion.

All three backends in one run need both GPU groups and a CUDA device. Run under a RAM cap:

```bash
uv sync --group gpu --group mjlab
systemd-run --user --scope -p MemoryMax=12G \
    uv run python -m pytest tests -m validation -q
```

Per-backend commands and timings: [MJX-Warp](backends/mjx.md#tests),
[mjlab](backends/mjlab.md#tests).

## Other physics tests

Not marked `validation`, so they run in the default suite (`tests/test_envs.py`):

| test | checks |
|---|---|
| `test_chassis_has_the_hardware_envelope` | step response within 3 % of the envelope |
| `test_velocity_normalisers_are_the_chassis_servo_range` | `MAX_V`, `MAX_W` equal the servo `ctrlrange` |
| `test_seated_ball_is_possessed` | Dribbling holds a seated ball ≥ 111/120 idle steps, seeds 0-4 |
| `test_seated_ball_survives_a_2_rad_s_turn` | ≥ 74/120 through a 2 rad/s turn, seeds 0-2 |
| `test_possession_reads_no_contact_list` | possession comes from the contact sensors, not `data.contact` |
| `test_horizon_is_seconds_at_exactly_40_hz` | `dt = 1/40` s and each env's step count |
