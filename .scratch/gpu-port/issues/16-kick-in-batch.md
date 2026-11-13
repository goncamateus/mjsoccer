# Como o lote dispara o kick, e com que tabela

Type: grilling
Status: resolved
Blocked by: 15

## Pergunta

Com os números do [ticket de medição do kick no MJX-Warp](15-kick-on-warp.md), decidir:
- o mecanismo de disparo no MJX, entre os medidos (4 096 mundos, 2 robôs): o de hoje com `timestep`
  por mundo e ociosos a `timestep = 0` (exato, 3.8× mais lento sempre), K = 32 (2×, tabela própria),
  ou o passo curto de 6.25 ms (~grátis, tabela própria, passo de disparo mais curto para a política);
- se o MJX-Warp usa a `KICK_TABLE_MS` do CPU ou uma tabela própria em float32 (gatilho: saída fora
  de 2 % da CPU), e onde ela mora (`scene.py`, lida pelos dois backends — [ticket 10](10-shared-task-logic.md));
- se o CPU muda junto (paridade de mecanismo) ou fica como está;
- como o `_plunger_lock`/cooldown vive no estado do mundo.

## Answer

Resolvido em 2026-11-13 (grilling).

- **Mecanismo: K = 32 nos dois backends.** No passo de disparo, 32 passos de física a `dt/4` e
  16 a `dt` (48 iterações, o passo segue com 25 ms). É o menor esquema monotônico do
  [ticket 15](15-kick-on-warp.md) que preserva a duração do passo de controle; custa 0.51× no lote
  (4 096 mundos, 2 robôs) contra 0.26× do K = 96 de hoje. O passo curto sai porque encurta o passo
  de disparo para 6.25 ms, quebrando o controlador a 40 Hz fixo (o `MAX_DV` por passo vira 4× a
  aceleração), e porque tem zona morta abaixo de ~75 N. `FIRING_SUBSTEPS = 4` fica, e entra
  `FIRING_FINE_STEPS = 32` em `scene.py`.
- **No MJX-Warp**: `opt.timestep` por mundo via `in_axes`. Os mundos que disparam rodam a `dt/4`
  nas 32 primeiras iterações e a `dt` nas 16 seguintes; os outros rodam 24 a `dt` e ficam
  congelados com `timestep = 0` nas 24 restantes. Sem `Data.where` por passo.
- **O CPU muda junto** (paridade de mecanismo), no mesmo passo do upgrade para o 4.0 e antes do
  port. O passo de disparo cai de 96 para 48 iterações. Os testes de kick/chip do
  `physics-fidelity` rodam de novo nessa rodada.
- **Tabela única**: a `KICK_TABLE_MS` em `scene.py`, com chave só por plunger (o tier 1 saiu,
  ADR 0001), recalibrada no CPU 4.0 float64 com K = 32 a 40 Hz pelo `calibrate.py` do candidate-09
  adaptado. O MJX a lê de `scene.py`, sem tabela float32 própria; o desvio é coberto pelos ≤ 2 % do
  teste de paridade.
- **`_plunger_lock` no estado do mundo**: contador `int32` de shape `(n_robots, 2)`, folha do pytree
  da task (`State.info`), decrementado a cada passo de controle. Cooldown `round(0.25 / dt)` = 10,
  lido do CPU. O `fired` por robô sai do mesmo cálculo, como `bool` no estado, e escolhe quais
  mundos ganham o `timestep` fino. O zero no reset segue o
  [ticket do reset em lote](17-batch-reset-and-world-state.md). É espelho exato do CPU.
