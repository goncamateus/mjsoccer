# 10: DR, ruído de observação e latência de ação no mjlab

Spec: [backend mjlab](../spec.md) — Fase 2 (DR, ruído e latência)

**What to build:** o pesquisador usa no mjlab a mesma DR, ruído e latência do MJX. `randomize=` aceita os mesmos parâmetros e ranges, sorteados por mundo (eventos `set_const` do mjlab; evento próprio para `pair_solref`/`pair_solimp` da bola). `--obs-noise` com a semântica do `ObsNoiseWrapper` (mesma seed, mesmo ruído; primeira obs pós-reset ruidosa). `--action-latency` em passos de controle sobre a decisão, antes do controlador, no action term. `--eval-randomize` no script.

**Blocked by:** 06

**Status:** done

- [x] Mesmos parâmetros/ranges/granularidade de DR do MJX; testes no formato dos de DR do MJX
- [x] Ruído de obs e latência com testes no formato dos de ruído/latência do MJX
- [x] `--obs-noise`, `--action-latency`, `--eval-randomize` funcionam no script de treino

## Notas de implementação

- **DR**: `randomize=` é o mesmo `sampler(mj_model, rng)` do MJX. `variants` saiu de
  `mjsoccer.mjx.randomize` para `mjsoccer.randomize` (numpy + mujoco, sem JAX) e os dois
  backends o chamam. No mjlab, um evento `startup` antes do pool roda o sampler no modelo CPU
  cacheado por mundo (rng de `SeedSequence(seed).spawn(n)`: o mundo i depende só de seed e i),
  com `mj_setConst` na CPU como no MJX, e escreve no modelo Warp o que mudou (campos expandidos
  por `sim.expand_model_fields`; `opt.*` trocado por array por mundo, grafo recapturado). Não
  usa os eventos `set_const` do mjlab, como o spec sugeria: com eles o sampler não seria o mesmo
  nos dois backends. Os `pair_*` da bola saem do mesmo caminho, sem evento próprio. As linhas
  `body_*` vão para o corpo de mesmo nome (a âncora do mjlab desloca os ids).
- Campos que podem variar: os `array("*", ...)` do `mujoco_warp` — superconjunto dos do MJX
  (o `mjx.Model` JAX não tem `site_quat`, `actuator_armature`, `opt.tolerance` e afins). Todo
  sampler aceito no MJX é aceito no mjlab.
- **Ruído**: `NoiseModelCfg` do mjlab no termo de obs, com gerador próprio semeado por `seed`.
- **Latência**: `world["action_queue"]` no `TaskLogic` (port do `_delay`/`_hold_action`), rodada
  pelo action term antes do controlador; o pool a guarda, então o reset a restaura.
