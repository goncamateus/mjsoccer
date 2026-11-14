# 09: As quatro tasks restantes no mjlab

Spec: [backend mjlab](../spec.md) — Ordem do port

**What to build:** GoToBallShoot, Dribbling, StaticDefenders e PassEndurance no mjlab, fechando o catálogo: lógica torch com task-logic parity, `make`, smoke e paridade com seed. Inclui os robôs scripted da PassEndurance (escrevendo `ctrl`).

**Blocked by:** 08

**Status:** ready-for-agent

- [ ] Task-logic parity das quatro (seeds 0-4) passa; as 6 tasks cobertas
- [ ] Smoke das 6 tasks no mjlab com kick, 0 overflow, 0 NaN
- [ ] `test_parity_seeded` das quatro no mjlab
- [ ] Nenhum id do `ENV_IDS` dá `NotImplementedError`
