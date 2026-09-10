# Validation harness and acceptance gates

Type: task
Status: open
Blocked by: 04, 05

## Question

Nothing to decide beyond the shape. Build the thing that proves every other ticket, then state the
gates.

Metrics (§13.1 of the research doc): tip rate per episode-minute; time-to-tip; chassis roll/pitch
mean/max/p99; yaw-rate distribution and the fraction of steps above `MAX_W`; per-step contact
impulse p50/p95/p99/max attributed to contact pair; the 0.5 s of velocity history before each tip;
possession duration mean/p95; episode return; steps-to-success on `GoToBall-v0` and `Dribbling-v0`;
steps/s.

Scenarios (§13.2), each a deterministic scripted sequence against the unwrapped env, no policy:
saturated uniform spin; head-on robot-robot at 1/2/3/4 m/s; wall graze at 2/4/6 m/s and
5°/15°/30°; hard turn while dribbling at 10/20/30/50 rad/s commanded yaw; kick recoil at 0/2/4 m/s
chassis speed; random-action rollout on all six ids and both tiers; and `goto_ball_policy` as the
competent-controller control.

Seeds 0-49 per (scenario × configuration), shared across configurations so comparisons are paired.
5 seeds for the learning curves, on `GoToBall-v0` and `Dribbling-v0` only. Medians and IQRs, not
means, for anything skewed.

Shape: extend `tests/test_envs.py`'s existing tip test as the template (`gym.make(id, tier=1)
.unwrapped`, write `data.qpos`, step, assert on `info`) as a marked suite; one CSV-emitting script
alongside `scripts/make_gifs.py` for the distributions; the existing sb3 examples as the trainer.
No new dependencies.

The gates, from §13.5, to be confirmed once `04` and `05` supply real numbers: tip rate under
random actions below 1 per episode-minute on all six ids and zero under a competent controller;
yaw-rate p99 below `MAX_W`; chassis-roll p99 below 15°; possession p50 within 5% of
`docs/index.md:199-212`; achievable acceleration within 1.5× of `01`'s motor figure; return and
sample efficiency no worse than `03`'s baseline at equal steps; steps/s within 10% of baseline;
and if a righting moment exists, saturation on fewer than 0.1% of steps.

Deliver the harness plan and the final gate values in `## Answer`. Writing the files is execution,
and happens after the map closes.
