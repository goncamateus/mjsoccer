# 08: GoToBall no mjlab (7 robôs)

Spec: [backend mjlab](../spec.md) — Ordem do port

**What to build:** a primeira cena de 7 robôs no mjlab, onde mora o risco do #1263 (ITERATIONS/LS_ITERATIONS). Lógica torch da GoToBall com task-logic parity, `make`, smoke, buffers (`nconmax`/`njmax`) dimensionados pelos candidatos do broadphase como no MJX, `nworld` máximo estável, paridade com seed e decisões/s contra o MJX-Warp.

**Blocked by:** 04, 07

**Status:** done

- [x] Task-logic parity da GoToBall (seeds 0-4) passa
- [x] Smoke da GoToBall com 0 overflow e 0 NaN; taxa de bits ITERATIONS medida no lote default
- [x] Buffers por mundo e `nworld` máximo medidos e registrados
- [x] `test_parity_seeded` da GoToBall no mjlab
- [x] Decisões/s ponta a ponta contra o MJX-Warp (~2,2 k na 3060), relatório

## Comments

Done (ticket 08): `GoToBallEnv` in `src/mjsoccer/mjlab/tasks.py` (+ `scatter`,
`TaskLogic.place_ball`); task-logic parity in `tests/test_mjlab_task_logic.py`; the smoke
parametrized over ContestedPossession and GoToBall in `tests/test_mjlab_smoke.py`; report
in `research/05-go-to-ball.md`, raw numbers in `bench/results_mjlab_08.txt`.

- **#1263 does not show**: 0 overflow bits of any type in ~4 M world-decisions (256 to
  32 768 worlds, random and CHASE); explicit 0 ITERATIONS/LS_ITERATIONS rate in the default
  batch (2 048, 113 k world-decisions). The solver's last substep peaks at 4 of 100 iterations.
- **Buffers**: the MJX sizing as is (`nconmax` 168, `njmax` 672 per world); broadphase
  candidates peak at 120.8 per world (MJX: 120.5), `nefc` at 67. No code change.
- **`nworld`**: 32 768 runs (9.9 GB device, one process per size), 40 960 OOMs at graph
  creation on the 12 GB 3060. "Stable" is 23 decisions per world, not a training run
  (ticket 11). `njmax` has 10x headroom (shared rule with MJX): the lever for more worlds.
- **Decisions/s**: within ~1-2 % of MJX-Warp (2 194 vs 2 211 at 1 024; 2 108 vs 2 131 at
  4 096); saturates from 512 worlds. CHASE is 5 % behind, not investigated.
- No env change was needed: `make`/`SoccerEnv` were already generic in the robot count and
  skip the seat loop when `seat_robot` is None (`_seat_ball` now reuses `place_ball`).
