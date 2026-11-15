# 09: As quatro tasks restantes no mjlab

Spec: [backend mjlab](../spec.md) — Ordem do port

**What to build:** GoToBallShoot, Dribbling, StaticDefenders e PassEndurance no mjlab, fechando o catálogo: lógica torch com task-logic parity, `make`, smoke e paridade com seed. Inclui os robôs scripted da PassEndurance (escrevendo `ctrl`).

**Blocked by:** 08

**Status:** done

- [x] Task-logic parity das quatro (seeds 0-4) passa; as 6 tasks cobertas
- [x] Smoke das 6 tasks no mjlab com kick, 0 overflow, 0 NaN
- [x] `test_parity_seeded` das quatro no mjlab
- [x] Nenhum id do `ENV_IDS` dá `NotImplementedError`

## Comments

Done (ticket 09): `GoToBallShootEnv`, `StaticDefendersEnv`, `DribblingEnv`,
`PassEnduranceEnv` (+ `first_where`, `PASS_CANDIDATES`) in `src/mjsoccer/mjlab/tasks.py`,
line by line from `mjsoccer.mjx.tasks`. PassEndurance's scripted robots write `ctrl` through
`script` (receiver's dribbler always on) and `shape_drive` (passer turns only); Dribbling
and PassEndurance seat the ball on robot 0 (`seat_robot = 0`). No env change.

- **Task-logic parity**: 6 tasks x seeds 0-4 (placement with each redraw loop's rejection
  path, first state, control steps, raw control) plus the Dribbling zigzag and the
  PassEndurance endings walks, ported from the MJX test; 108 pass. PassEndurance is exempt
  from the plunger re-arm coverage check: its first kick ends the episode within the cooldown.
- **Smoke**: all six ids, repeat 1 and 2, kick on the first decision, 0 overflow bits, 0 NaN.
  Dribbling and PassEndurance end often under random actions, so the per-decision step count
  and kick lock are checked on the worlds that never ended.
- **Validation**: `-m validation -k mjlab` 110 pass, 0 skip: every seeded cell of the four
  (`test_parity_seeded`) and their scripted scenarios in their own scenes.
- With every id ported, `make` lost its `NotImplementedError` branch and the validation
  sweep its borrowed-scene path (`MJLAB_HOST`, `sim_env` column, `mjlab_env`).
