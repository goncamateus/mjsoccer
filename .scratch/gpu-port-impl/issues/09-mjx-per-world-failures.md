# 09: Falhas por mundo no MJX-Warp

Spec: [port GPU](../../gpu-port/spec.md)

**What to build:** Um mundo com `qpos`/`qvel` não finito, ou com qualquer bit do `overflow` do Warp, termina (terminal, não truncation), com recompensa e obs zeradas, e o auto-reset do Brax/Playground o restaura; sem abort dentro do jit. `State.metrics` ganha `diverged`, `overflow` e `reset_overlap` (0/1 por passo). Os testes MJX checam os contadores no host e abortam com os bits de overflow por tipo na mensagem. O CPU mantém o `RuntimeError`.

**Blocked by:** 07 (Primeiro passo do MJX: grupo `gpu`, `mjsoccer.mjx.make` e ContestedPossession sem kick)

**Status:** ready-for-agent

- [ ] Mundo forçado a NaN termina, sai com recompensa/obs 0 e volta pelo auto-reset sem derrubar os outros
- [ ] Overflow forçado (buffer subdimensionado) trata o mundo igual ao NaN
- [ ] Os três contadores aparecem em `State.metrics`
- [ ] Task-logic parity e smoke falham com mensagem clara se `diverged` ou `overflow` > 0
