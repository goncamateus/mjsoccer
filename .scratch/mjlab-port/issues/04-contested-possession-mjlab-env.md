# 04: `mjsoccer.mjlab.make("ContestedPossession-v0")` + smoke

Spec: [backend mjlab](../spec.md) — Fase 2, Testing Decisions (seam 1)

**What to build:** o usuário cria a ContestedPossession no mjlab e passo em lote na GPU. Cena inteira via `SceneCfg.spec_fn` (modelo base exato, nomes do CPU, âncora vazia para o action term, `MujocoCfg` igual ao `<option>`); passo de disparo com `decimation = 48 × repeat_action` e `opt.timestep` por mundo (agenda 32 × dt/4 + 16 × dt ao disparar, 24 × dt + 24 × 0 senão); horizonte, recompensa e time-out nossos; pool de primeiros estados no `startup`; termination por overflow e por NaN com obs/recompensa zeradas; métricas `diverged`/`overflow`/`reset_overlap`. Usa a lógica do 03; os managers são cola fina.

**Blocked by:** 03

**Status:** done

- [x] `make` lê `ENV_IDS`; mesmos espaços e layout de obs/ação do CPU; `repeat_action` com a mesma semântica
- [x] Erro claro sem o grupo `mjlab` ou sem CUDA; `NotImplementedError` para id não portado
- [x] Smoke (runner de subprocesso de GPU, pula sem CUDA): `make` + `reset` + ~10 passos, kick disparado, `diverged == overflow == 0`, obs finita
- [x] Mundo ocioso bit a bit (`qpos`, `qvel`, `act`, `time`, `ctrl`) nas iterações a `timestep = 0`
- [x] Lote misto com buffers subdimensionados em parte dos mundos: só esses terminam; o reset limpa só os bits deles
- [x] Decisões/s ponta a ponta registradas contra o MJX-Warp (relatório)

## Comments

Hand-offs from 03 (`mjsoccer.mjlab.base`/`tasks`), left to the env:

- Per-world RNG: the logic takes `place`'s uniforms as an argument; the env owns the generator.
- `reset_overlap` on the first decision (`MjxEnv._first_decision`): `init_state` returns it in the reset metrics only.
- `seat_ball` (the pool of first states) and the firing schedule (`MjxEnv.timesteps`) are not in the logic.
- The task-logic parity runs in float64; the env will feed float32 from Warp. The controller's lateral sign at rest is rounding noise there, as on MJX.

Done (ticket 04): `mjsoccer.mjlab.make` → `mjsoccer.mjlab.env.SoccerEnv`; tests in
`tests/test_mjlab_smoke.py`; report in `research/03-env-dps.md` (16 598 dec/s against 16 537
on MJX-Warp at 4 096 worlds; 17 534 at 16 384).

Decisions and deviations, for the spec owner:

- **BROADPHASE marks every world.** Warp's contact buffer is shared by the batch; when it
  overflows, the bit lands in every world, so the whole batch ends. Only NEFC (`njmax`, per
  world) is per world: the mixed-batch test undersizes `njmax` with `nconmax` 512. Story 8
  holds per bit; amend it or decide.
- **NaN without mjlab's `nan_detection`.** `TaskLogic.settle`'s `isfinite(qpos, qvel)` ends
  the world and zeroes obs/reward (as MJX); a second term would only add a kernel per step.
- **The spec's "o controlador roda uma vez por decisão" is wrong for `repeat_action > 1`**:
  the controller runs once per control step, as on CPU and MJX (the ticket's "mesma
  semântica"). A world that ends mid-decision runs the rest at timestep 0 (bit-frozen).
- **Controller freshness** (for 07): obs/reward/endings are taken before mjlab's `forward()`
  (as `MjxEnv.settle`), but a decision's first control step reads kinematics that `forward()`
  refreshed; CPU/MJX read them one substep stale (idle worlds: same as MJX). Seeded parity
  judges whether to compensate.
- **Extra body**: the anchor entity (`nbody` 13 vs 12); ids are re-resolved by name
  (`scene.resolve`, `TaskLogic.bind`). ADR 0003 says "exact base model" without the anchor.
- `make(randomize=...)` raises NotImplementedError until ticket 10. `reset(seed=)` reseeds
  mjlab's global RNG, not the pool's generator (`make(seed=)`).
- Streams: torch runs on Warp's stream inside `step`/`reset`/`refill_pool`, ordered against
  the caller's stream at entry/exit (ticket 05 measures).
- `NACON_*`/`NJMAX_PER_CONTACT` and the firing schedule (`scene.firing_schedule`) now live in
  `scene`, shared by both GPU backends.
