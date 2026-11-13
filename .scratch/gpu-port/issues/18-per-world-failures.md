# Falhas por mundo: NaN e overflow do Warp

Type: grilling
Status: resolved
Blocked by: 17

## Pergunta

No CPU uma divergência levanta `RuntimeError` (`src/mjsoccer/base.py:153-159`). Num lote de
milhares de mundos, um mundo que diverge não pode derrubar o treino, e o Warp tem overflow de
`naconmax`/`njmax` que é máscara em `Data.overflow` e não para a simulação
([benchmark](05-mjx-benchmark.md)). Decidir, sobre o reset do
[ticket de reset em lote](17-batch-reset-and-world-state.md):
- NaN/inf num mundo: resetar o mundo (com que recompensa/`done`), marcar numa métrica, ou abortar;
- overflow do Warp: abortar o run (buffers mal dimensionados), resetar o mundo, ou só contar;
- o que o `State.metrics` expõe para o log do treino (contadores por época), junto do `reset_overlap` que o reset em lote já conta;
- se o eval e os testes de paridade tratam diferente (abortar onde o treino reseta).

## Answer

Resolvido em grilling (2026-11-13), todas as recomendações aceitas.

- **NaN/inf no treino**: mundo com `qpos`/`qvel` não finito termina o episódio (`done = 1`,
  terminal, não truncation: bootstrap de obs lixo não vale); recompensa e obs sanitizadas para 0
  (`jnp.where(isfinite…)`) para o NaN não chegar ao gradiente; o auto-reset do
  [ticket de reset em lote](17-batch-reset-and-world-state.md) restaura o estado em cache. Sem abort.
- **Overflow do Warp no treino**: qualquer bit de `_impl.overflow` (bitmask por mundo, `(nworld,)`)
  trata o mundo igual ao NaN — termina, sanitiza, reseta — mesmo com física finita, porque o
  resultado é indefinido (e o pool do `naconmax` é global). Sem abort: buffer mal dimensionado
  aparece na taxa do contador e o dev decide.
- **`State.metrics`**: três contadores 0/1 por passo — `diverged`, `overflow` (qualquer bit, sem
  separar por tipo) e o `reset_overlap` do ticket 17; o `EpisodeWrapper` do Brax soma por episódio e
  o log mostra a taxa.
- **Eval e testes**: o eval dentro do `ppo.train` herda o tratamento do treino (não aborta de dentro
  do jit). O eval standalone (`examples/ppo_mjx.py`) e os testes de paridade/task-logic **abortam**
  se `diverged` ou `overflow` > 0, checado no host depois do rollout, com os bits de overflow por
  tipo na mensagem.
- **CPU intacto**: o backend CPU mantém o `RuntimeError` (`src/mjsoccer/base.py:153-159`). A
  assimetria é deliberada: o oráculo falha alto, o lote reseta. O task-logic parity não cobre esse
  caminho (snapshot finito).
