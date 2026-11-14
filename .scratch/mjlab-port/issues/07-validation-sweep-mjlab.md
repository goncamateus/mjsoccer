# 07: Sweep `--backend mjlab`: critérios e paridade

Spec: [backend mjlab](../spec.md) — Testing Decisions (seam 2)

**What to build:** o pesquisador de física roda `scripts/validation_sweep.py --backend mjlab` e os mesmos testes de `-m validation` do MJX valem para o mjlab, com os mesmos limiares. Inclui resolver o frescor da obs: o mjlab faz `forward` antes da obs e o MJX/CPU não; decidir se precisa compensar para a obs ver o mesmo estado do CPU.

**Blocked by:** 04

**Status:** ready-for-agent

- [ ] `episode(..., backend="mjlab")`, coluna `backend` = `mjlab` nos CSVs
- [ ] Critérios de aceitação (envelope, kick/chip × `KICK_TABLE_MS`, COR e inércia, posse com giro de 2 rad/s, `spin`, `kick_recoil`, controlador) passam no mjlab
- [ ] `test_parity_deterministic` (≤ 2 % envelope/kick/COR, ≤ 5 % chegada, ≤ 1 cm / 1°) e `hold_turn` (64 cópias) no mjlab
- [ ] `test_parity_seeded` da ContestedPossession (mediana mjlab no [p25, p75] da CPU, seeds 0-4)
- [ ] Frescor da obs medido e decidido; casos mjlab pulam sem CUDA
