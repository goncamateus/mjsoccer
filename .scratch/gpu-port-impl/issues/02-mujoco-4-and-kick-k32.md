# 02: MuJoCo 4.0 no backend CPU, com kick K = 32

Spec: [port GPU](../../gpu-port/spec.md)

**What to build:** A base sobe de `mujoco 3.11` para `mujoco 4.0.0` (avaliar depender só de `mujoco-core`). O passo de disparo vira K = 32: 32 passos de física a `dt/4` + 16 a `dt`, 48 iterações, o passo de controle segue com 25 ms (`FIRING_SUBSTEPS = 4`, novo `FIRING_FINE_STEPS = 32`). A `KICK_TABLE_MS`, chave só por plunger, é recalibrada no CPU 4.0 float64 com K = 32 a 40 Hz pelo `calibrate.py` do candidate-09 adaptado. Os testes de fidelidade do `physics-fidelity` rodam de novo no 4.0 (o 4.0 muda derivadas de atuador saturado em `implicitfast` e `nativeccd`/`multiccd`).

**Blocked by:** 01 (Remover o tier 1)

**Status:** ready-for-agent

- [ ] `pyproject.toml` pina `mujoco 4.0.0`; `uv sync` limpo
- [ ] Passo de disparo com 48 iterações (32 a `dt/4`, 16 a `dt`) e duração de 25 ms
- [ ] `KICK_TABLE_MS` indexada só por plunger, recalibrada no 4.0; velocidade de saída do kick e do chip dentro dos critérios
- [ ] Testes de fidelidade (envelope, bola, posse, kick/chip) verdes no 4.0, `-m validation` incluso
- [ ] Base CPU de decisões/s remedida no 4.0 (ContestedPossession 24 envs, GoToBall 12 envs) e anotada neste ticket
