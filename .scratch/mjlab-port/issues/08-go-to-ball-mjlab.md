# 08: GoToBall no mjlab (7 robôs)

Spec: [backend mjlab](../spec.md) — Ordem do port

**What to build:** a primeira cena de 7 robôs no mjlab, onde mora o risco do #1263 (ITERATIONS/LS_ITERATIONS). Lógica torch da GoToBall com task-logic parity, `make`, smoke, buffers (`nconmax`/`njmax`) dimensionados pelos candidatos do broadphase como no MJX, `nworld` máximo estável, paridade com seed e decisões/s contra o MJX-Warp.

**Blocked by:** 04, 07

**Status:** ready-for-agent

- [ ] Task-logic parity da GoToBall (seeds 0-4) passa
- [ ] Smoke da GoToBall com 0 overflow e 0 NaN; taxa de bits ITERATIONS medida no lote default
- [ ] Buffers por mundo e `nworld` máximo medidos e registrados
- [ ] `test_parity_seeded` da GoToBall no mjlab
- [ ] Decisões/s ponta a ponta contra o MJX-Warp (~2,2 k na 3060), relatório
