# 07: Sweep `--backend mjlab`: critérios e paridade

Spec: [backend mjlab](../spec.md) — Testing Decisions (seam 2)

**What to build:** o pesquisador de física roda `scripts/validation_sweep.py --backend mjlab` e os mesmos testes de `-m validation` do MJX valem para o mjlab, com os mesmos limiares. Inclui resolver o frescor da obs: o mjlab faz `forward` antes da obs e o MJX/CPU não; decidir se precisa compensar para a obs ver o mesmo estado do CPU.

**Blocked by:** 04

**Status:** done

- [x] `episode(..., backend="mjlab")`, coluna `backend` = `mjlab` nos CSVs
- [x] Critérios de aceitação (envelope, kick/chip × `KICK_TABLE_MS`, COR e inércia, posse com giro de 2 rad/s, `spin`, `kick_recoil`, controlador) passam no mjlab
- [x] `test_parity_deterministic` (≤ 2 % envelope/kick/COR, ≤ 5 % chegada, ≤ 1 cm / 1°) e `hold_turn` (64 cópias) no mjlab
- [x] `test_parity_seeded` da ContestedPossession (mediana mjlab no [p25, p75] da CPU, seeds 0-4)
- [x] Frescor da obs medido e decidido; casos mjlab pulam sem CUDA

## Comments

Done (ticket 07): `scripts/validation_sweep.py --backend mjlab` (`run_mjlab`, `transplant`),
`SoccerEnv.raw`/`set_state` in `src/mjsoccer/mjlab/env.py`, the `mjlab` cases in
`tests/test_validation.py` (80 pass, 30 skip on the five unported envs' seeded cells),
freshness in `.scratch/mjlab-port/bench/obs_freshness.py` (+ `.txt`).

Decisions and deviations, for the spec owner:

- **Scripted scenarios borrow ContestedPossession's scene** while GoToBall/Dribbling are not
  ported: set up on the CPU env of their own id, then carried into the mjlab scene by joint
  and body name, slides re-based (`transplant`): robot 0, the ball and the shared yellow in
  the same world state. The host task's endings are off in those worlds (CP would end the
  envelope at x = -5 on `robot_out`) and their `ret` is NaN; `sim_env` in the CSV names the
  scene that ran. Once 08/09 port an env, its rows switch to its own scene (`mjlab_env`).
  Consequence: `kick` 1.0 runs all 31 steps on mjlab, where Dribbling's CPU ends it at step 3
  (a yellow knocked); the peak speed is the same (6.5525 both).
- **Seeded parity rule**: the ticket's "mjlab median in the CPU's [p25, p75]" is the rule
  gpu-port ticket 12 retired; mjlab uses the current one (CPU median of seeds 0-4 inside the
  mjlab range of 50 worlds), the same threshold as MJX. Only ContestedPossession runs (one
  `make(seed=0)` generator draws every world's first state; no seed pairing).
- **Deterministic parity** came out near bit-level: envelope, kick, COR within 1e-4 relative;
  arrival time within 1 %, error within 2.6 mm / 0.1 deg.
- **Freshness, measured and not compensated.** obs/reward/endings are taken before mjlab's
  `forward()` (as MJX). The only reader of the refreshed kinematics is the next decision's
  controller (`xmat` for the drive rotation). Idle worlds: `forward()` changes nothing (their
  timestep-0 iterations refreshed it, as on MJX). After a firing step: yaw at most 0.0056 rad
  fresher (mean 0.0013), 7 253 decisions measured (one run, random actions): at most 0.6 % of
  a 3.3 m/s drive command rotated, on one control step after each shot. Too small for the parity
  tolerances to see (so they are no evidence either way) and smaller than MJX's own idle-world
  gap to the CPU, which passes; not compensated.
  (Against the CPU, both GPU backends read idle-world kinematics one substep fresher; the same
  trait MJX already passes with.)
- `episode(..., backend="mjlab")` raises NotImplementedError for a seeded scenario on an
  unported env; `run(..., "mjlab")` leaves those units out.
