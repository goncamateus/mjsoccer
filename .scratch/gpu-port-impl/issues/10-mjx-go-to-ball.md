# 10: GoToBall no MJX (cena de 7 robôs) e relatório de decisões/s

Spec: [port GPU](../../gpu-port/spec.md)

**What to build:** GoToBall roda no MJX-Warp com a cena de 7 robôs, com `naconmax` dimensionado pelos candidatos do broadphase. Medir decisões/s ponta a ponta (env completo: task, controlador, reset) de ContestedPossession e GoToBall contra a base CPU 4.0 do ticket 02. É relatório, não portão: serve para pegar regressão grosseira da lógica JAX.

**Blocked by:** 08 (Kick no lote do MJX-Warp), 09 (Falhas por mundo no MJX-Warp)

**Status:** ready-for-agent

- [ ] GoToBall no task-logic parity e no Warp smoke
- [ ] `overflow == 0` com o `naconmax` escolhido
- [ ] Decisões/s das duas tasks (MJX × CPU 4.0) anotadas neste ticket
