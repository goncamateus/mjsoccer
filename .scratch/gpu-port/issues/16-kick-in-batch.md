# Como o lote dispara o kick, e com que tabela

Type: grilling
Status: open
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
