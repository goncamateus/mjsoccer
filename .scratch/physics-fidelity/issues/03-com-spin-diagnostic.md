# Diagnostic: confirm the CoM-offset tipping mechanism and measure the baseline

Type: task
Status: open
Blocked by: —

## Question

Nothing to decide. Run the measurement that later tickets depend on, then revert.

The claim under test (`M1`, `TO2`): tier 1 tips because the CAD chassis's centre of mass sits
**3.459 mm off the spin axis** (verified: `|subtree_com − xpos|` on the compiled model, with
`subtree_com = [·, ·, 0.0739943]` and the floor plane at z = 0.02), and the action space permits
59.4 rad/s of yaw. The predicted threshold is
`ω_tip = sqrt(g · d_edge / (h · d_com)) = sqrt(9.81 · 0.04325 / (0.05399 · 0.003459)) = 47.7 rad/s`.

Do:
1. **Reproduce the spin sweep.** `place` one tier-1 robot at the origin, command
   `w = [k,k,k,k]` for `k` across `[0.2 … 1.0]` of `ctrlrange`, 5 s of physics, no other body in
   contact. Record peak yaw rate and peak tilt per `k`. Expected from the research doc: 42.1 rad/s
   → 6.0°; 55.6 → 43.2°; 59.4 → 46.6°.
2. **Falsify it.** Temporarily zero the x/y components of `ipos` on
   `models/robot_rollers.xml:153` and re-run. If the tilt collapses, `M1` is confirmed.
   **Revert the edit in the same session** — `git diff` must be empty when this ticket closes.
   This is a diagnostic, never a shipped change (§12.4(a)).
3. **Baseline tip rate.** `env.action_space.sample()` on all six ids, tier 1, seeds 0-49: tips
   per episode-minute, time-to-tip distribution, and the p50/p95/p99 of chassis tilt. This is the
   number every later fix is measured against, so record the exact command and the commit.
4. **Baseline wall-clock** steps/s per env, for comparison with `docs/index.md:157-160`.

Write the numbers into this ticket's `## Answer`, and the raw rows to
`.scratch/physics-fidelity/baseline/`. Do not modify anything under `src/`, `models/`, `examples/`,
`scripts/` or `tests/` except the reverted `ipos` diagnostic.
