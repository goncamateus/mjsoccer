# 10: DR, ruído de observação e latência de ação no mjlab

Spec: [backend mjlab](../spec.md) — Fase 2 (DR, ruído e latência)

**What to build:** o pesquisador usa no mjlab a mesma DR, ruído e latência do MJX. `randomize=` aceita os mesmos parâmetros e ranges, sorteados por mundo (eventos `set_const` do mjlab; evento próprio para `pair_solref`/`pair_solimp` da bola). `--obs-noise` com a semântica do `ObsNoiseWrapper` (mesma seed, mesmo ruído; primeira obs pós-reset ruidosa). `--action-latency` em passos de controle sobre a decisão, antes do controlador, no action term. `--eval-randomize` no script.

**Blocked by:** 06

**Status:** ready-for-agent

- [ ] Mesmos parâmetros/ranges/granularidade de DR do MJX; testes no formato dos de DR do MJX
- [ ] Ruído de obs e latência com testes no formato dos de ruído/latência do MJX
- [ ] `--obs-noise`, `--action-latency`, `--eval-randomize` funcionam no script de treino
