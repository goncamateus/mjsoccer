# 01: Remover o tier 1

Spec: [port GPU](../../gpu-port/spec.md)

**What to build:** Um robô só, o chassi cinemático, nas 6 tasks (ADR 0001). O roller robot, a cena dele, o kwarg `tier`, o `wheel_command`, os critérios de tilt/tombamento (`test_tipover_ends_episode`, `pretip.csv`) e os cenários `head_on`, `wall`, `dribble_turn` saem. A palavra "tier" sai do código e da doc. Remoção mecânica, comportamento do chassi inalterado.

**Blocked by:** None (can start immediately)

**Status:** ready-for-agent

- [ ] `gymnasium.make(id)` nas 6 tasks roda só o chassi cinemático; não existe kwarg `tier`
- [ ] Nenhuma ocorrência de "tier" ou do roller robot em código, modelos, scripts e `docs/`
- [ ] Testes de tilt/tombamento e cenários `head_on`/`wall`/`dribble_turn` removidos do sweep e do `test_validation`
- [ ] `uv run pytest` e `uv run pytest -m validation` verdes
