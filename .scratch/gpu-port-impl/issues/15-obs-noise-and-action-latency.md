# 15: Ruído de observação e latência de ação no MJX

Spec: [port GPU](../../gpu-port/spec.md)

**What to build:** Opções do wrapper de treino do lado MJX: ruído de observação (via `info["rng"]`) e latência de ação como buffer de N passos de controle no `State`. Desligadas por default, fora do task-logic parity, sem `actuator_delay` nativo.

**Blocked by:** 13 (Script de treino `examples/ppo_mjx.py`)

**Status:** ready-for-agent

- [ ] Com as opções desligadas, comportamento idêntico ao de antes
- [ ] Ruído ligado altera a obs de forma reprodutível por seed
- [ ] Latência de N passos aplica a ação N passos de controle depois
